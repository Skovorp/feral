from feral.data import build_datasets_and_loaders
from feral.loops import train_one_epoch, evaluate, run_inference
from feral.modeling import build_model, build_training_objects, load_model_from_checkpoint
from feral.utils import save_model, pick_and_save_best, validate_labels_json, check_environment, is_classification
import yaml
import torch
import wandb
import json
import datetime
import numpy as np
import random
from feral.metrics import generate_empty_logits, ensemble_predictions
from feral.metrics import calculate_multiclass_metrics, calc_frame_level_map, generate_raster_plot, save_inference_results, calculate_f1_metrics
from feral.metrics import calculate_regression_metrics, calculate_video_level_regression_metrics, ensemble_regression_predictions
import sys
import os
import logging
import warnings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    check_environment(compile_enabled=cfg['training']['compile'])

    with open(cfg['data']['label_json'], 'r') as f:
        labels_json = json.load(f)

    validate_labels_json(labels_json, cfg['data'].get('prefix'))

    is_cls = is_classification(labels_json)
    task = 'classification' if is_cls else 'regression'
    if is_cls:
        class_names = {int(x): y for x, y in labels_json['class_names'].items()}
        num_classes = len(class_names)
        target_names = None
        num_targets = None
        model_save_metadata = {
            'task': 'classification',
            'class_names': class_names,
            'is_multilabel': labels_json['is_multilabel'],
            'cfg': cfg,
        }
    else:
        target_names = {int(x): y for x, y in labels_json['target_names'].items()}
        num_targets = len(target_names)
        class_names = None
        num_classes = 0
        model_save_metadata = {
            'task': 'regression',
            'target_names': target_names,
            'cfg': cfg,
        }

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
        model, model_ema = build_model(cfg, num_classes, device, task=task, num_targets=num_targets)
        criterion, optimizer, lr_scheduler, mixup = build_training_objects(
            cfg, model, train_dataset, train_loader, labels_json, device,
        )

        best_checkpoint_path = os.path.join("checkpoints", f"{cfg['run_name']}_best_checkpoint.pt")
        # Classification tracks "best map" (higher is better). Regression tracks
        # "best score" = -val_mse (so 'higher is better' still works in pick_and_save_best).
        best_map = -1 if is_cls else -float('inf')
        epochs_without_updates = 0

        for epoch in range(cfg['training']['epochs']):
            answers, train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, lr_scheduler,
                mixup=mixup, model_ema=model_ema,
                num_classes=num_classes, is_multilabel=labels_json.get('is_multilabel', False),
                predict_per_item=cfg['predict_per_item'],
                device=device, log_fn=wandb.log, max_batches=cfg.get('max_batches'),
                task=task,
            )
            if is_cls:
                logs = {
                    **calculate_multiclass_metrics(answers, class_names, 'train'),
                    'train_loss': train_loss,
                }
            else:
                logs = {
                    **calculate_regression_metrics(answers, target_names, 'train'),
                    'train_loss': train_loss,
                }
            logger.info("%s", logs)
            wandb.log(logs)

            if val_loader is None:
                save_model(model, best_checkpoint_path, model_save_metadata)
                logger.info("Epoch %d: Saved model", epoch)
                continue

            answers, val_loss = evaluate(
                model, val_loader, criterion,
                num_classes=num_classes, is_multilabel=labels_json.get('is_multilabel', False),
                device=device, max_batches=cfg.get('max_batches'), task=task,
            )
            if model_ema is not None:
                answers_ema, ema_val_loss = evaluate(
                    model_ema.ema, val_loader, criterion,
                    num_classes=num_classes, is_multilabel=labels_json.get('is_multilabel', False),
                    device=device, max_batches=cfg.get('max_batches'), task=task,
                )
            with open(os.path.join("answers", f"{cfg['run_name']}_{_str_now()}.json"), 'w') as f:
                json.dump(answers, f)
            if is_cls:
                logs = {
                    **calculate_multiclass_metrics(answers, class_names, 'val'),
                    **calculate_f1_metrics(answers, labels_json, 'val', labels_json['is_multilabel'], 'val', cfg['multilabel_threshold']),
                    'val_frame_level_map': calc_frame_level_map(answers, labels_json, 'val'),
                    'val_loss': val_loss,
                    'val_raster_plot': wandb.Image(generate_raster_plot(answers, labels_json, 'val'))
                }
                if model_ema is not None:
                    ema_logs = {
                        **calculate_multiclass_metrics(answers_ema, class_names, 'ema_val'),
                        **calculate_f1_metrics(answers_ema, labels_json, 'val', labels_json['is_multilabel'], 'ema_val', cfg['multilabel_threshold']),
                        'ema_val_frame_level_map': calc_frame_level_map(answers_ema, labels_json, 'val'),
                        'ema_val_loss': ema_val_loss,
                        'ema_val_raster_plot': wandb.Image(generate_raster_plot(answers_ema, labels_json, 'val'))
                    }
                    logs.update(ema_logs)
            else:
                logs = {
                    **calculate_regression_metrics(answers, target_names, 'val'),
                    **calculate_video_level_regression_metrics(answers, labels_json, 'val', target_names, 'val'),
                    'val_loss': val_loss,
                }
                if model_ema is not None:
                    ema_logs = {
                        **calculate_regression_metrics(answers_ema, target_names, 'ema_val'),
                        **calculate_video_level_regression_metrics(answers_ema, labels_json, 'val', target_names, 'ema_val'),
                        'ema_val_loss': ema_val_loss,
                    }
                    logs.update(ema_logs)
            logger.info("%s", logs)
            wandb.log(logs)

            # save best model — classification: maximise frame-level mAP; regression: maximise -val_mse.
            if is_cls:
                val_score = logs['val_frame_level_map']
                ema_score = logs.get('ema_val_frame_level_map', -2)
                score_name = 'val_frame_level_map'
            else:
                val_score = -logs['val_mse']
                ema_score = -logs.get('ema_val_mse', float('inf'))
                score_name = 'val_mse (lower is better)'
            best_map, saved = pick_and_save_best(
                model, model_ema, val_score, ema_score, best_map, best_checkpoint_path,
                model_save_metadata,
            )
            if saved == 'base':
                epochs_without_updates = 0
                logger.info("Epoch %d: Saved base model checkpoint with %s=%.4f", epoch, score_name, val_score)
            elif saved == 'ema':
                epochs_without_updates = 0
                logger.info("Epoch %d: Saved EMA model checkpoint with ema %s=%.4f", epoch, score_name, ema_score)
            else:
                epochs_without_updates += 1
                logger.info("Epoch %d: Didnt improve for %d epochs", epoch, epochs_without_updates)
                if cfg['training']['patience'] is not None and epochs_without_updates >= cfg['training']['patience']:
                    logger.info("Epoch %d: Early stopping: no improvement for %d epochs, stopping training.", epoch, cfg['training']['patience'])
                    break

        logger.info("Finished training")
        logger.info("Best checkpoint: %s. best score: %.4f", best_checkpoint_path, best_map)
        del model
        del model_ema
        del optimizer
        del lr_scheduler
        torch.cuda.empty_cache()
    if inference_loader is None and test_loader is None:
        return
    
    logger.info("Loading model for test/inference")
    if train_loader is not None:
        test_checkpoint_path = best_checkpoint_path
    else:
        test_checkpoint_path = cfg['starting_checkpoint']
    
    best_model, _checkpoint_meta = load_model_from_checkpoint(cfg, device, test_checkpoint_path, num_classes=num_classes)

    if test_loader is not None:
        logger.info("Running test...")
        answers, _ = evaluate(
            best_model, test_loader,
            num_classes=num_classes, is_multilabel=labels_json.get('is_multilabel', False),
            device=device, max_batches=cfg.get('max_batches'), task=task,
        )
        with open(os.path.join("answers", f"{cfg['run_name']}_raw_test.json"), 'w') as f:
            json.dump(answers, f)
        if is_cls:
            predictions = generate_empty_logits(labels_json, 'test')
            predictions = ensemble_predictions(answers, predictions)
            with open(os.path.join("answers", f"{cfg['run_name']}_ensembled_test.json"), 'w') as f:
                if labels_json['is_multilabel']:
                    json.dump({
                        'pred': {x: ((y > cfg['multilabel_threshold']) * 1).tolist() for x, y in predictions.items()},
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
                **calculate_f1_metrics(answers, labels_json, 'test', labels_json['is_multilabel'], 'test', cfg['multilabel_threshold']),
                'test_frame_level_map': calc_frame_level_map(answers, labels_json, 'test'),
                'test_raster_plot': wandb.Image(generate_raster_plot(answers, labels_json, 'test'))
            }
        else:
            preds_per_video = ensemble_regression_predictions(answers)
            with open(os.path.join("answers", f"{cfg['run_name']}_ensembled_test.json"), 'w') as f:
                json.dump({
                    'pred': {fn: v.tolist() for fn, v in preds_per_video.items()},
                    'gt': {fn: labels_json['labels'][fn] for fn in preds_per_video.keys()},
                    'target_names': labels_json['target_names'],
                }, f)
            logs = {
                **calculate_regression_metrics(answers, target_names, 'test'),
                **calculate_video_level_regression_metrics(answers, labels_json, 'test', target_names, 'test'),
            }
        logger.info("%s", logs)
        wandb.log(logs)

    if inference_loader is not None:
        logger.info("Running inference...")
        answers = run_inference(
            best_model, inference_loader,
            is_multilabel=labels_json.get('is_multilabel', False),
            device=device, max_batches=cfg.get('max_batches'), task=task,
        )
        out_pth = os.path.join("answers", f"_inference_{cfg['run_name']}_{_str_now()}.json")
        if is_cls:
            save_inference_results(answers, [], cfg['data']['prefix'], labels_json, out_pth)
        else:
            preds_per_video = ensemble_regression_predictions(answers)
            with open(out_pth, 'w') as f:
                json.dump({
                    'preds': {fn: v.tolist() for fn, v in preds_per_video.items()},
                    'target_names': labels_json['target_names'],
                }, f)
       

if __name__ == '__main__':
    from feral.cli import main as cli_main
    cli_main()
