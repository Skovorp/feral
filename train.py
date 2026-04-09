from data import build_datasets_and_loaders
from loops import train_one_epoch, evaluate, run_inference
from modeling import build_model, build_training_objects, load_model_from_checkpoint
from utils import save_model, pick_and_save_best
import yaml
import torch
import wandb
import json
import datetime
import numpy as np
import random
from metrics import generate_empty_logits, ensemble_predictions
from metrics import calculate_multiclass_metrics, calc_frame_level_map, generate_raster_plot, save_inference_results, calculate_f1_metrics
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

def _str_now():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

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
        entity=cfg.get('wandb', {}).get('entity'),
        project=cfg.get('wandb', {}).get('project'),
        name=cfg['run_name'],
        config=cfg,
        mode='disabled' if cfg.get('wandb') is None else 'online'
    )

    datasets, loaders = build_datasets_and_loaders(cfg, labels_json, num_classes)
    train_dataset = datasets.get('train')
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test')
    inference_loader = loaders.get('inference')

    device = torch.device(cfg.get('device', 'cuda'))

    if train_loader is not None:
        model, model_ema = build_model(cfg, num_classes, device)
        criterion, optimizer, lr_scheduler, mixup = build_training_objects(
            cfg, model, train_dataset, train_loader, labels_json, device,
        )

        best_checkpoint_path = os.path.join("checkpoints", f"{cfg['run_name']}_best_checkpoint.pt")
        best_map = -1
        epochs_without_updates = 0

        for epoch in range(cfg['training']['epochs']):
            answers, train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, lr_scheduler,
                mixup=mixup, model_ema=model_ema,
                num_classes=num_classes, is_multilabel=labels_json['is_multilabel'],
                predict_per_item=cfg['predict_per_item'],
                device=device, log_fn=wandb.log, max_batches=cfg.get('max_batches'),
            )
            logs = {
                **calculate_multiclass_metrics(answers, class_names, 'train'),
                'train_loss': train_loss,
            }
            print(logs)
            wandb.log(logs)

            if val_loader is None:
                save_model(model, best_checkpoint_path)
                print(f"Epoch {epoch}: Saved model")
                continue

            answers, val_loss = evaluate(
                model, val_loader, criterion,
                num_classes=num_classes, is_multilabel=labels_json['is_multilabel'],
                device=device, max_batches=cfg.get('max_batches'),
            )
            if model_ema is not None:
                answers_ema, ema_val_loss = evaluate(
                    model_ema.ema, val_loader, criterion,
                    num_classes=num_classes, is_multilabel=labels_json['is_multilabel'],
                    device=device, max_batches=cfg.get('max_batches'),
                )
            with open(os.path.join("answers", f"{cfg['run_name']}_{_str_now()}.json"), 'w') as f:
                json.dump(answers, f)
            logs = {
                **calculate_multiclass_metrics(answers, class_names, 'val'),
                **calculate_f1_metrics(answers, labels_json, 'val', labels_json['is_multilabel'], 'val'),
                'val_frame_level_map': calc_frame_level_map(answers, labels_json, 'val'),
                'val_loss': val_loss,
                'val_raster_plot': wandb.Image(generate_raster_plot(answers, labels_json, 'val'))
            }
            if model_ema is not None:
                ema_logs = {
                    **calculate_multiclass_metrics(answers_ema, class_names, 'ema_val'),
                    **calculate_f1_metrics(answers_ema, labels_json, 'val', labels_json['is_multilabel'], 'ema_val'),
                    'ema_val_frame_level_map': calc_frame_level_map(answers_ema, labels_json, 'val'),
                    'ema_val_loss': ema_val_loss,
                    'ema_val_raster_plot': wandb.Image(generate_raster_plot(answers_ema, labels_json, 'val'))
                }
                logs.update(ema_logs)
            print(logs)
            wandb.log(logs)

            # save best model
            val_map = logs['val_frame_level_map']
            ema_map = logs.get('ema_val_frame_level_map', -2)
            best_map, saved = pick_and_save_best(
                model, model_ema, val_map, ema_map, best_map, best_checkpoint_path,
            )
            if saved == 'base':
                epochs_without_updates = 0
                print(f"Epoch {epoch}: Saved base model checkpoint with val_frame_level_map={val_map:.4f}")
            elif saved == 'ema':
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
    
    best_model = load_model_from_checkpoint(cfg, num_classes, device, test_checkpoint_path)

    if test_loader is not None:
        print("Running test...")
        answers, _ = evaluate(
            best_model, test_loader,
            num_classes=num_classes, is_multilabel=labels_json['is_multilabel'],
            device=device, max_batches=cfg.get('max_batches')
        )
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
        answers = run_inference(
            best_model, inference_loader,
            is_multilabel=labels_json['is_multilabel'], device=device, max_batches=cfg.get('max_batches')
        )
        out_pth = os.path.join("answers", f"_inference_{cfg['run_name']}_{_str_now()}.json")
        save_inference_results(answers, [], cfg['data']['prefix'], labels_json, out_pth)
       

if __name__ == '__main__':
    assert len(sys.argv) > 1 and len(sys.argv[1]) > 0, "Usage: python train.py <path_to_config.yaml>"

    with open(sys.argv[1], 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
