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
import time

from metrics import calculate_multiclass_metrics, calc_frame_level_map
from timm.utils import ModelEma
from torchvision.transforms.v2 import MixUp



with open('configs/profile.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

torch.manual_seed(cfg['seed'])
np.random.seed(cfg['seed'])
random.seed(cfg['seed'])
torch.backends.cudnn.benchmark = True


train_dataset = ClsDataset(partition='train', model_name=cfg['model_name'], 
                           num_classes=cfg['num_classes'], predict_per_item=cfg['predict_per_item'], **cfg['data'])
train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, drop_last=True, in_order=False, persistent_workers=True,
                          batch_size=cfg['training']['train_bs'], num_workers=cfg['training']['num_workers'])

device = torch.device('cuda')

model = HFModel(model_name=cfg['model_name'], num_classes=cfg['num_classes'])
model.to(device)
model.train()

model_ema = ModelEma(
    model,
    decay=cfg['ema_decay'],
    device='cuda'
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])

total_steps = len(train_loader) * cfg['training']['epochs']
warmup_steps = len(train_loader) * cfg['training']['warmup_epochs']
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


mixup = None if cfg['mixup_alpha'] is None else MixUp(alpha=cfg['mixup_alpha'], num_classes=cfg['training']['train_bs'])
iteration_times = []

# with torch.profiler.profile(
#             schedule=torch.profiler.schedule(wait=3, warmup=20, active=10, repeat=1),
#             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_256_mixup_bigger_again'),
#             record_shapes=True,
#             profile_memory=True,
#             with_stack=True
#     ) as prof:
times = []
t = time.time()
for batch_idx, (data, target) in enumerate(train_loader):
    # prof.step()
    if batch_idx >= 100: # ((3 + 20 + 3) * 1):
        break
    start_time = time.time()

    data = data.to(device)
    target = target.to(device)

    if mixup is not None:
        N, T, C, A, B = data.shape
        data = data.reshape(N, T, C, A * B)
        data, mix = mixup(data, torch.eye(data.shape[0], device=device))
        data = data.reshape(N, T, C, A, B)
        if cfg['predict_per_item'] != 1:
            target = target.permute(1, 0, 2)
            target = mix.unsqueeze(0) @ target
            target = target.permute(1, 0, 2)
        else:
            target = mix @ target 
    target = target.view(-1, cfg['num_classes'])
    optimizer.zero_grad()
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        output = model(data)
        loss = criterion(output, target)

    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    model_ema.update(model)

    iteration_times.append(time.time() - start_time)
    times.append(time.time() - t)
    t = time.time()
    print(f"{times[-1] * 1000:.2f} iteration_times {iteration_times[-1] * 1000:.2f}")
# print(f"Single iteration takes {sum(iteration_times[-3:]) * 1000 / 3:.2f}ms")
print(f"Max: {max(iteration_times[-3:]) * 1000:.2f}ms")
times.pop(0)
print(f"my times avg {sum(times) / len(times)* 1000:.2f} max: {max(times)* 1000:.2f}")
print(f"Maximum GPU memory recorded per step: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
    