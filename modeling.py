import logging

import torch
from torch.optim import AdamW
from timm.utils import ModelEma
from torchvision.transforms.v2 import MixUp
from transformers import get_cosine_schedule_with_warmup

from model import HFModel
from utils import get_weights

logger = logging.getLogger(__name__)


def build_model(cfg, num_classes, device, *, with_ema=True):
    """Construct HFModel, move to device, optionally compile and wrap in EMA.

    Returns (model, model_ema). model_ema is None if cfg['ema_decay'] is None
    or with_ema is False.
    """
    model = HFModel(
        model_name=cfg['model_name'],
        num_classes=num_classes,
        predict_per_item=cfg['predict_per_item'],
        **cfg['model'],
    )
    model.to(device)

    if cfg['training']['compile']:
        model = torch.compile(model, mode="max-autotune", dynamic=True)

    model_ema = None
    if with_ema and cfg['ema_decay'] is not None:
        model_ema = ModelEma(model, decay=cfg['ema_decay'], device=device)

    n_params = sum(el.numel() for el in model.state_dict().values())
    logger.info("parameters: %s", f"{n_params:_d}")

    return model, model_ema


def load_model_from_checkpoint(cfg, num_classes, device, checkpoint_path):
    """Build a fresh model and load weights from checkpoint_path. Returns model."""
    model = HFModel(
        model_name=cfg['model_name'],
        num_classes=num_classes,
        predict_per_item=cfg['predict_per_item'],
        **cfg['model'],
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    if cfg['training']['compile']:
        model = torch.compile(model, mode="max-autotune", dynamic=True)
    model.eval()
    return model


def build_training_objects(cfg, model, train_dataset, train_loader, labels_json, device):
    """Build criterion, optimizer, lr_scheduler, mixup. Returns dict-of-objects."""
    class_weights = get_weights(train_dataset.json_data, cfg['model']['class_weights'], device)
    if labels_json['is_multilabel']:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=cfg['training']['label_smoothing'],
            weight=class_weights,
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    total_steps = len(train_loader) * cfg['training']['epochs']
    warmup_steps = round(total_steps * cfg['training']['part_warmup'])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    mixup = (None if cfg['mixup_alpha'] is None
             else MixUp(alpha=cfg['mixup_alpha'], num_classes=cfg['training']['train_bs']))

    return criterion, optimizer, lr_scheduler, mixup
