from feral.data import build_datasets_and_loaders
from feral.loops import train_one_epoch, evaluate, run_inference
from feral.modeling import build_model, build_training_objects, load_model_from_checkpoint
from feral.utils import save_model, pick_and_save_best, validate_labels_json, check_environment
import yaml
import torch
import wandb
import json
import datetime
import numpy as np
import random
from feral.metrics import generate_empty_logits, ensemble_predictions
from feral.metrics import calculate_multiclass_metrics, calc_frame_level_map, generate_raster_plot, save_inference_results, calculate_f1_metrics, calculate_optimal_f1_metrics, generate_multilabel_raster_plot, compute_optimal_per_class_thresholds
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
    """Return the current local time as a filename-safe 'YYYY-MM-DD_HH-MM-SS' string."""
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def _add_raster_logs(logs, answers, labels_json, partition, prefix, optimal_prefix, multilabel_threshold):
    """
    Adds raster image entries to `logs` under `{prefix}/raster_plot` (regular
    threshold) and, for multilabel, `{optimal_prefix}/raster_plot` (per-class
    optimal-F1 thresholds). Every step is guarded; failures are logged but never
    raised, so a broken plot can't kill training.
    """
    is_ml = labels_json['is_multilabel']
    try:
        if is_ml:
            img = generate_multilabel_raster_plot(answers, labels_json, partition, multilabel_threshold)
        else:
            img = generate_raster_plot(answers, labels_json, partition)
        logs[f'{prefix}/raster_plot'] = wandb.Image(img)
    except Exception:
        logger.exception("regular raster plot failed for %s", prefix)

    if not is_ml or optimal_prefix is None:
        return
    try:
        opt_thr = compute_optimal_per_class_thresholds(answers, labels_json, partition)
        img = generate_multilabel_raster_plot(answers, labels_json, partition, opt_thr)
        logs[f'{optimal_prefix}/raster_plot'] = wandb.Image(img)
    except Exception:
        logger.exception("optimal raster plot failed for %s", optimal_prefix)

def main(cfg):
    """Run the full training/eval/inference pipeline for one config: build data, model,
    and training objects, train with per-epoch validation and best-checkpoint selection
    (with optional EMA and early stopping), then load the best checkpoint to run test
    and/or inference. Logs metrics and raster plots to wandb and writes answers/checkpoints
    to disk. Returns None."""
    check_environment(compile_enabled=cfg['training']['compile'])

    with open(cfg['data']['label_json'], 'r') as f:
        labels_json = json.load(f)

    validate_labels_json(labels_json, cfg['data'].get('prefix'))

    class_names = {int(x): y for x, y in labels_json['class_names'].items()}
    num_classes = len(class_names)
    model_save_metadata = {
        'class_names': class_names,
        'is_multilabel': labels_json['is_multilabel'],
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
        model, model_ema = build_model(cfg, num_classes, device)
        if wandb.run is not None:
            wandb.run.summary['n_params'] = sum(p.numel() for p in model.parameters())
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
                grad_clip_norm=cfg['training'].get('grad_clip_norm'),
                log_grad_norm=cfg['training'].get('log_grad_norm', True),
                heavy_log_every=cfg['training'].get('heavy_log_every'),
            )
            logs = {
                **calculate_multiclass_metrics(answers, class_names, 'train'),
                'train/loss': train_loss,
            }
            if torch.cuda.is_available():
                logs['perf/gpu_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
            logger.info("%s", logs)
            wandb.log(logs)

            if val_loader is None:
                save_model(model, best_checkpoint_path, model_save_metadata)
                logger.info("Epoch %d: Saved model", epoch)
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
                **calculate_f1_metrics(answers, labels_json, 'val', labels_json['is_multilabel'], 'val', cfg['multilabel_threshold']),
                **calculate_optimal_f1_metrics(answers, labels_json, 'val', labels_json['is_multilabel'], 'val_optimal'),
                'val/frame_level_map': calc_frame_level_map(answers, labels_json, 'val'),
                'val/loss': val_loss,
            }
            _add_raster_logs(logs, answers, labels_json, 'val', 'val', 'val_optimal', cfg['multilabel_threshold'])
            if model_ema is not None:
                ema_logs = {
                    **calculate_multiclass_metrics(answers_ema, class_names, 'val_ema'),
                    **calculate_f1_metrics(answers_ema, labels_json, 'val', labels_json['is_multilabel'], 'val_ema', cfg['multilabel_threshold']),
                    **calculate_optimal_f1_metrics(answers_ema, labels_json, 'val', labels_json['is_multilabel'], 'val_ema_optimal'),
                    'val_ema/frame_level_map': calc_frame_level_map(answers_ema, labels_json, 'val'),
                    'val_ema/loss': ema_val_loss,
                }
                _add_raster_logs(ema_logs, answers_ema, labels_json, 'val', 'val_ema', 'val_ema_optimal', cfg['multilabel_threshold'])
                logs.update(ema_logs)
            logger.info("%s", logs)
            wandb.log(logs)

            # save best model
            val_map = logs['val/frame_level_map']
            ema_map = logs.get('val_ema/frame_level_map', -2)
            best_map, saved = pick_and_save_best(
                model, model_ema, val_map, ema_map, best_map, best_checkpoint_path,
                model_save_metadata,
            )
            if saved == 'base':
                epochs_without_updates = 0
                logger.info("Epoch %d: Saved base model checkpoint with val/frame_level_map=%.4f", epoch, val_map)
            elif saved == 'ema':
                epochs_without_updates = 0
                logger.info("Epoch %d: Saved EMA model checkpoint with val_ema/frame_level_map=%.4f", epoch, ema_map)
            else:
                epochs_without_updates += 1
                logger.info("Epoch %d: Didnt improve for %d epochs", epoch, epochs_without_updates)
                patience = cfg['training'].get('patience')
                if patience is not None and epochs_without_updates >= patience:
                    logger.info("Epoch %d: Early stopping: no improvement for %d epochs, stopping training.", epoch, patience)
                    break

        logger.info("Finished training")
        logger.info("Best checkpoint: %s. best_map: %.4f", best_checkpoint_path, best_map)
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
            **calculate_optimal_f1_metrics(answers, labels_json, 'test', labels_json['is_multilabel'], 'test_optimal'),
            'test/frame_level_map': calc_frame_level_map(answers, labels_json, 'test'),
        }
        _add_raster_logs(logs, answers, labels_json, 'test', 'test', 'test_optimal', cfg['multilabel_threshold'])
        logger.info("%s", logs)
        wandb.log(logs)

    if inference_loader is not None:
        logger.info("Running inference...")
        answers = run_inference(
            best_model, inference_loader,
            is_multilabel=labels_json['is_multilabel'], device=device, max_batches=cfg.get('max_batches')
        )
        out_pth = os.path.join("answers", f"_inference_{cfg['run_name']}_{_str_now()}.json")
        save_inference_results(answers, [], cfg['data']['prefix'], labels_json, out_pth)
       

if __name__ == '__main__':
    from feral.cli import main as cli_main
    cli_main()
