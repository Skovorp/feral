from model import HFModel
from dataset import ClsDataset, collate_fn_val, collate_fn_inference
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
import json
import datetime
import numpy as np 
import random
from metrics import generate_empty_logits, ensemble_predictions
from metrics import calculate_multiclass_metrics, calc_frame_level_map, generate_raster_plot, save_inference_results, calculate_f1_metrics
from utils import prep_for_answers, get_weights
from timm.utils import ModelEma
from torchvision.transforms.v2 import MixUp
import sys
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="No positive class found in y_true, recall is set to one for all thresholds.",
    category=UserWarning,
    module="sklearn.metrics._ranking"
)

torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def main(cfg):
    with open(cfg['data']['label_json'], 'r') as f:
        labels_json = json.load(f)
    class_names = {int(x): y for x, y in labels_json['class_names'].items()}
    num_classes = len(class_names)

    os.makedirs("answers", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = True

    wandb.init(
        entity=cfg['wandb']['entity'],
        project=cfg['wandb']['project'],
        name=cfg['run_name'],
        config=cfg,
        mode='disabled' if cfg['run_name'] == 'debug' else 'online'
    )

    if 'train' in labels_json['splits'] and len(labels_json['splits']['train']) > 0:
        train_dataset = ClsDataset(partition='train', model_name=cfg['model_name'],
                                num_classes=num_classes, predict_per_item=cfg['predict_per_item'], **cfg['data'])
        train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, drop_last=True, persistent_workers=cfg['training']['num_workers'] > 0,
                            batch_size=cfg['training']['train_bs'], num_workers=cfg['training']['num_workers'])
        assert labels_json['is_multilabel'] == train_dataset.is_multilabel, f"Config is_multilabel doesn't match the data! config: {labels_json['is_multilabel']} dataset: {train_dataset.is_multilabel}"
    else:
        print("No train dataset")
        train_loader = None
    
    if 'val' in labels_json['splits'] and len(labels_json['splits']['val']) > 0:
        val_dataset = ClsDataset(partition='val', model_name=cfg['model_name'], 
                                num_classes=num_classes, predict_per_item=cfg['predict_per_item'], **cfg['data'])
        val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, drop_last=False, persistent_workers=cfg['training']['num_workers'] > 0,
                                batch_size=cfg['training']['val_bs'], num_workers=cfg['training']['num_workers'], collate_fn=collate_fn_val)
        assert labels_json['is_multilabel'] == val_dataset.is_multilabel, f"Config is_multilabel doesn't match the data! config: {labels_json['is_multilabel']} dataset: {val_dataset.is_multilabel}"
    else:
        print("No val dataset")
        val_loader = None
    
    if 'test' in labels_json['splits'] and len(labels_json['splits']['test']) > 0:
        test_dataset = ClsDataset(partition='test', model_name=cfg['model_name'], 
                            num_classes=num_classes, predict_per_item=cfg['predict_per_item'], **cfg['data'])
        test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, drop_last=False, persistent_workers=cfg['training']['num_workers'] > 0,
                            batch_size=cfg['training']['val_bs'], num_workers=cfg['training']['num_workers'], collate_fn=collate_fn_val)
        assert labels_json['is_multilabel'] == test_dataset.is_multilabel, f"Config is_multilabel doesn't match the data! config: {labels_json['is_multilabel']} dataset: {test_dataset.is_multilabel}"
    else:
        print("No test dataset")
        test_loader = None
        
    if 'inference' in labels_json['splits'] and len(labels_json['splits']['inference']) > 0:
        inference_dataset = ClsDataset(partition='inference', model_name=cfg['model_name'], 
                            num_classes=num_classes, predict_per_item=cfg['predict_per_item'], **cfg['data'])
        inference_loader = DataLoader(inference_dataset, shuffle=False, pin_memory=True, drop_last=False, persistent_workers=cfg['training']['num_workers'] > 0,
                            batch_size=cfg['training']['val_bs'], num_workers=cfg['training']['num_workers'], collate_fn=collate_fn_inference)
    else:
        inference_loader = None

    device = torch.device(cfg.get('device', 'cuda'))

    if train_loader is not None:
        model = HFModel(model_name=cfg['model_name'], num_classes=num_classes, predict_per_item=cfg['predict_per_item'], **cfg['model'])
        model.to(device)

        if cfg['ema_decay'] is not None:
            model_ema = ModelEma(
                model,
                decay=cfg['ema_decay'],
                device=cfg.get('device', 'cuda')
            )
        else:
            model_ema = None

        if cfg['training']['compile']:
            model = torch.compile(model, mode="max-autotune", dynamic=True)

        tot = 0
        for el in model.state_dict().values():
            tot += el.numel()
        print(f"parameters: {tot:_d}")
        print(f"Dataset is multilabel: {labels_json['is_multilabel']}")

        class_weights = get_weights(train_dataset.json_data, cfg['model']['class_weights'], device)
        if labels_json['is_multilabel']:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg['training']['label_smoothing'], weight=class_weights)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])

        total_steps = len(train_loader) * cfg['training']['epochs']
        warmup_steps = round(total_steps * cfg['training']['part_warmup'])
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        mixup = None if cfg['mixup_alpha'] is None else MixUp(alpha=cfg['mixup_alpha'], num_classes=cfg['training']['train_bs'])
        
        best_checkpoint_path = os.path.join("checkpoints", f"{cfg['run_name']}_best_checkpoint.pt")
        best_map = -1
        epochs_without_updates = 0

        for epoch in range(cfg['training']['epochs']):
            model.train()
            answers = []
            losses = []
            for data, target in tqdm(train_loader, total=len(train_loader)):
                data = data.to(device)
                target = target.to(device)
                
                # Create eye matrix based on actual batch size (needed for multi-GPU)
                batch_size = data.shape[0]
                eye = torch.eye(batch_size, device=device)
                
                optimizer.zero_grad()

                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    if mixup is not None:
                        N, T, C, A, B = data.shape
                        data = data.reshape(N, T, C, A * B)
                        data, mix = mixup(data, eye)
                        data = data.reshape(N, T, C, A, B)
                        if cfg['predict_per_item'] != 1:
                            target = target.permute(1, 0, 2)
                            target = mix.unsqueeze(0) @ target
                            target = target.permute(1, 0, 2)
                        else:
                            target = mix @ target 
                    target = target.reshape(-1, num_classes)
                    output = model(data)
                    output_prob = torch.sigmoid(output) if labels_json['is_multilabel'] else torch.nn.functional.softmax(output, 1)
                    loss = criterion(output, target)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                if model_ema is not None:
                    model_ema.update(model)
                wandb.log({
                    'batch_loss': loss.item(), 
                    'lr': lr_scheduler.get_last_lr()[0]
                })

                answers.extend(prep_for_answers(output_prob, target))
                losses.append(loss.item())
                if cfg['run_name'] == 'debug':
                    break
            logs = {
                **calculate_multiclass_metrics(answers, class_names, 'train'),
                'train_loss': sum(losses) / len(losses)
            }
            print(logs)
            wandb.log(logs)
            
            if val_loader is None:
                if hasattr(model, '_orig_mod'):
                    model = model._orig_mod
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"Epoch {epoch}: Saved model")
                continue

            model.eval()
            if model_ema is not None:
                model_ema.ema.eval()
            answers = []
            losses = []
            answers_ema = []
            losses_ema = []
            with torch.no_grad():
                for data, target, names in tqdm(val_loader, total=len(val_loader)):
                    data = data.to(device)
                    target = target.to(device)
                    target = target.view(-1, num_classes)
                    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                        output = model(data)
                        loss = criterion(output, target)
                        output_prob = torch.sigmoid(output) if labels_json['is_multilabel'] else torch.nn.functional.softmax(output, 1)
                        answers.extend(prep_for_answers(output_prob, target, names))
                        losses.append(loss.item())

                        if model_ema is not None:
                            output_ema = model_ema.ema(data)
                            loss_ema = criterion(output_ema, target)
                            output_ema_prob = torch.sigmoid(output_ema) if labels_json['is_multilabel'] else torch.nn.functional.softmax(output_ema, 1)
                            answers_ema.extend(prep_for_answers(output_ema_prob, target, names))
                            losses_ema.append(loss_ema.item())
                        if cfg['run_name'] == 'debug':
                            break
            with open(os.path.join("answers", f"{cfg['run_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"), 'w') as f:
                json.dump(answers, f)
            logs = {
                **calculate_multiclass_metrics(answers, class_names, 'val'),
                **calculate_f1_metrics(answers, labels_json, 'val', labels_json['is_multilabel'], 'val'),
                'val_frame_level_map': calc_frame_level_map(answers, labels_json, 'val'),
                'val_loss': sum(losses) / len(losses),
                'val_raster_plot': wandb.Image(generate_raster_plot(answers, labels_json, 'val'))
            }
            if model_ema is not None:
                ema_logs = {
                    **calculate_multiclass_metrics(answers_ema, class_names, 'ema_val'),
                    **calculate_f1_metrics(answers_ema, labels_json, 'val', labels_json['is_multilabel'], 'ema_val'),
                    'ema_val_frame_level_map': calc_frame_level_map(answers_ema, labels_json, 'val'),
                    'ema_val_loss': sum(losses_ema) / len(losses_ema),
                    'ema_val_raster_plot': wandb.Image(generate_raster_plot(answers_ema, labels_json, 'val'))
                }
                logs.update(ema_logs)
            print(logs)
            wandb.log(logs)

            # save best model
            val_map = logs['val_frame_level_map']
            ema_map = logs.get('ema_val_frame_level_map', -2)
            if val_map > ema_map and val_map > best_map:
                m = model
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                torch.save(m.state_dict(), best_checkpoint_path)
                best_map = val_map
                epochs_without_updates = 0  
                print(f"Epoch {epoch}: Saved base model checkpoint with val_frame_level_map={val_map:.4f}")
            elif model_ema is not None and ema_map > val_map and ema_map > best_map:
                m = model_ema.ema
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                torch.save(m.state_dict(), best_checkpoint_path)
                best_map = ema_map
                epochs_without_updates = 0
                print(f"Epoch {epoch}: Saved EMA model checkpoint with ema_val_frame_level_map={ema_map:.4f}")
            else:
                epochs_without_updates += 1
                print(f"Epoch {epoch}: Didnt improve for {epochs_without_updates} epochs")

                if cfg['training']['patience'] is not None and epochs_without_updates >= cfg['training']['patience']:
                    print(f"Epoch {epoch}: Early stopping: no improvement for {cfg['training']['patience']} epochs, stopping training.")
                    break
        
        print("Finished training")
        print(f"Best checkpoint: {best_checkpoint_path}. best_map: {best_map:4f}")
        del model 
        del model_ema
        del optimizer
        del lr_scheduler
        torch.cuda.empty_cache()
    if inference_loader is None and test_loader is None:
        return
    
    print("Loading model for test/inference")
    if train_loader is not None:
        test_checkpoint_path = best_checkpoint_path
    else:
        test_checkpoint_path = cfg['starting_checkpoint']
    
    best_model = HFModel(model_name=cfg['model_name'], num_classes=num_classes, predict_per_item=cfg['predict_per_item'], **cfg['model'])
    best_model_state = torch.load(test_checkpoint_path, map_location="cpu")
    best_model.load_state_dict(best_model_state)
    best_model.to(device)
    if cfg['training']['compile']:
        best_model = torch.compile(best_model, mode="max-autotune", dynamic=True)
    best_model.eval()

    if test_loader is not None:
        print("Running test...")
        answers = []
        with torch.no_grad():
            for data, target, names in tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                target = target.view(-1, num_classes)
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    output = best_model(data)
                    output_prob = torch.sigmoid(output) if labels_json['is_multilabel'] else torch.nn.functional.softmax(output, 1)
                    answers.extend(prep_for_answers(output_prob, target, names))
        with open(os.path.join("answers", f"{cfg['run_name']}_raw_test.json"), 'w') as f:
            json.dump(answers, f)
        predictions = generate_empty_logits(labels_json, 'test')
        predictions = ensemble_predictions(answers, predictions)
        with open(os.path.join("answers", f"{cfg['run_name']}_ensembled_test.json"), 'w') as f:
            if labels_json['is_multilabel']:
                json.dump({
                    'pred': {x: ((y > 0.85) * 1).tolist() for x, y in predictions.items()},
                    'gt': {x: labels_json['labels'][x] for x, y in predictions.items()},
                    'class_names': labels_json['class_names']
                }, f)
            else:
                json.dump({
                    'pred': {x: y.argmax(1).tolist() for x, y in predictions.items()},
                    'gt': {x: labels_json['labels'][x] for x, y in predictions.items()},
                    'class_names': labels_json['class_names']
                }, f)
        logs = {
            **calculate_multiclass_metrics(answers, class_names, 'test'),
            **calculate_f1_metrics(answers, labels_json, 'test', labels_json['is_multilabel'], 'test'),
            'test_frame_level_map': calc_frame_level_map(answers, labels_json, 'test'),
            'test_raster_plot': wandb.Image(generate_raster_plot(answers, labels_json, 'test'))
        }
        print(logs)
        wandb.log(logs)

    if inference_loader is not None:
        print("Running inference...")
        answers = []
        with torch.no_grad():
            for data, names in tqdm(inference_loader, total=len(inference_loader)):
                data = data.to(device)
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    output = best_model(data)
                    output_prob = torch.sigmoid(output) if labels_json['is_multilabel'] else torch.nn.functional.softmax(output, 1)
                    answers.extend(prep_for_answers(output_prob, None, names))
        out_pth = os.path.join("answers", f"_inference_{cfg['run_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        save_inference_results(answers, [], cfg['data']['prefix'], labels_json, out_pth)
       

if __name__ == '__main__':
    assert len(sys.argv) > 1 and len(sys.argv[1]) > 0, "Usage: python train.py <path_to_config.yaml>"

    with open(sys.argv[1], 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
