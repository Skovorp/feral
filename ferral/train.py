from model import HFModel
from dataset import ClsDataset
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
from timm.utils import ModelEma



with open('cfg.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

torch.manual_seed(cfg['seed'])
np.random.seed(cfg['seed'])
random.seed(cfg['seed'])
torch.backends.cudnn.benchmark = True

wandb.init(
    project="Veles",
    name=cfg['run_name'],
    config=cfg
)

train_dataset = ClsDataset(partition='train', model_name=cfg['model_name'], **cfg['data'])
val_dataset = ClsDataset(partition='val', model_name=cfg['model_name'], **cfg['data'])

train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=cfg['training']['train_bs'], num_workers=cfg['training']['num_workers'])
val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, batch_size=cfg['training']['val_bs'], num_workers=cfg['training']['num_workers'])

device = torch.device('cuda')

model = HFModel(model_name=cfg['model_name'], **cfg['model'])
model.to(device)
model.train()

# # make batchnorm stats ok
# with torch.no_grad():
#     for it, (data, _), in enumerate(train_loader):
#         if it >= 10:
#             break
#         data = data.to(device)
#         model(data)

model_ema = ModelEma(
    model,
    decay=cfg['ema_decay'],
    device='cuda'
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


for epoch in range(cfg['training']['epochs']):
    model.train()
    answers = []
    losses = []
    for data, target in tqdm(train_loader, total=len(train_loader)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
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
        answers.extend(list(zip(output.cpu().detach().tolist(), target.cpu().detach().tolist())))
        losses.append(loss.item())
    wandb.log({
        **calculate_multiclass_metrics(answers, cfg['class_names'], 'train'),
        'train_loss': sum(losses) / len(losses)
    })

    model.eval()
    answers = []
    losses = []
    with torch.no_grad():
        for data, target, names in tqdm(val_loader, total=len(val_loader)):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                output = model(data)
                loss = criterion(output, target)
                answers.extend(list(zip(names, output.cpu().detach().tolist(), target.cpu().detach().tolist())))
                losses.append(loss.item())
    with open(f"answers/{cfg['run_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json", 'w') as f:
        json.dump(answers, f)
    wandb.log({
        **calculate_multiclass_metrics(answers, cfg['class_names'], 'val'),
        'val_frame_level_map': calc_frame_level_map(answers, False, cfg['class_names']),
        'val_loss': sum(losses) / len(losses)      
    })

