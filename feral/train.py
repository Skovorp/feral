from model import HFModel
from new_dataset import ClsDataset, collate_fn_val
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

from metrics import calculate_multiclass_metrics, calc_frame_level_map
from utils import prep_for_answers
from timm.utils import ModelEma
from torchvision.transforms.v2 import MixUp
import sys
import os

torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs("answers", exist_ok=True)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = True

    wandb.init(
        project="Veles",
        name=cfg['run_name'],
        config=cfg,
        mode='disabled' if cfg['run_name'] == 'debug' else 'online'
    )

    train_dataset = ClsDataset(partition='train', model_name=cfg['model_name'],
                            num_classes=cfg['num_classes'], predict_per_item=cfg['predict_per_item'], **cfg['data'])
    val_dataset = ClsDataset(partition='val', model_name=cfg['model_name'], 
                            num_classes=cfg['num_classes'], predict_per_item=cfg['predict_per_item'], **cfg['data'])

    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, drop_last=True, in_order=False, persistent_workers=False,
                            batch_size=cfg['training']['train_bs'], num_workers=cfg['training']['num_workers'])
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, drop_last=False, persistent_workers=False,
                            batch_size=cfg['training']['val_bs'], num_workers=cfg['training']['num_workers'], collate_fn=collate_fn_val)

    device = torch.device('cuda:2')

    model = HFModel(model_name=cfg['model_name'], num_classes=cfg['num_classes'], predict_per_item=cfg['predict_per_item'])
    model.to(device)
    model.train()

    model_ema = ModelEma(
        model,
        decay=cfg['ema_decay'],
        device='cuda:2'
    )

    tot = 0
    for el in model.state_dict().values():
        tot += el.numel()
    print(f"parameters: {tot:_d}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])

    total_steps = len(train_loader) * cfg['training']['epochs']
    warmup_steps = len(train_loader) * cfg['training']['warmup_epochs']
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


    mixup = None if cfg['mixup_alpha'] is None else MixUp(alpha=cfg['mixup_alpha'], num_classes=cfg['training']['train_bs'])

    for epoch in range(cfg['training']['epochs']):
        model.train()
        answers = []
        losses = []
        eye = torch.eye(cfg['training']['train_bs'], device=device)
        for data, target in tqdm(train_loader, total=len(train_loader)):
            data = data.to(device)
            target = target.to(device)
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
                target = target.reshape(-1, cfg['num_classes'])
                output = model(data)
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            model_ema.update(model)
            wandb.log({
                'batch_loss': loss.item(), 
                'lr': lr_scheduler.get_last_lr()[0]
            })

            answers.extend(prep_for_answers(output, target))
            losses.append(loss.item())
            if cfg['run_name'] == 'debug':
                break
        logs = {
            **calculate_multiclass_metrics(answers, cfg['class_names'], 'train'),
            'train_loss': sum(losses) / len(losses)
        }
        print(logs)
        wandb.log(logs)

        model.eval()
        answers = []
        losses = []
        answers_ema = []
        losses_ema = []
        with torch.no_grad():
            for data, target, names in tqdm(val_loader, total=len(val_loader)):
                data = data.to(device)
                target = target.to(device)
                target = target.view(-1, cfg['num_classes'])
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    output = model(data)
                    loss = criterion(output, target)
                    answers.extend(prep_for_answers(output, target, names))
                    losses.append(loss.item())

                    output_ema = model_ema.ema(data)
                    loss_ema = criterion(output_ema, target)
                    answers_ema.extend(prep_for_answers(output_ema, target, names))
                    losses_ema.append(loss.item())
                    if cfg['run_name'] == 'debug':
                        break
        with open(os.path.join("answers", f"{cfg['run_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"), 'w') as f:
            json.dump(answers, f)
        logs = {
            **calculate_multiclass_metrics(answers, cfg['class_names'], 'val'),
            **calculate_multiclass_metrics(answers_ema, cfg['class_names'], 'ema_val'),
            # 'val_frame_level_map': calc_frame_level_map(answers, cfg['predict_per_item'] > 1, cfg['class_names']),
            'val_loss': sum(losses) / len(losses),
            # 'ema_val_frame_level_map': calc_frame_level_map(answers_ema, cfg['predict_per_item'] > 1, cfg['class_names']),
            'ema_val_loss': sum(losses_ema) / len(losses_ema),
        }
        print(logs)
        wandb.log(logs)

if __name__ == '__main__':
    assert len(sys.argv) > 1 and len(sys.argv[1]) > 0, "Usage: python train.py <path_to_config.yaml>"
    main(sys.argv[1])