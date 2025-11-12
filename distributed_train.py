import datetime
import json
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import MixUp
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import wandb

from dataset import ClsDataset, collate_fn_val
from metrics import (
    calculate_multiclass_metrics,
    calculate_f1_metrics,
    calc_frame_level_map,
    generate_raster_plot,
)
from model import HFModel
from utils import get_weights, prep_for_answers

warnings.filterwarnings(
    "ignore",
    message="No positive class found in y_true, recall is set to one for all thresholds.",
    category=UserWarning,
    module="sklearn.metrics._ranking",
)


def setup_distributed():
    if not dist.is_available():
        raise RuntimeError("torch.distributed must be available for distributed_train.py")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            local_rank = rank % torch.cuda.device_count()
        else:
            local_rank = 0
    else:
        raise RuntimeError("Distributed environment variables are not set")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return device, local_rank


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def gather_lists_from_all_processes(data_list):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, data_list)
    merged = []
    for part in gathered:
        merged.extend(part)
    return merged


def reduce_loss_sums(loss_sum, count, device):
    tensor = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor[0] / tensor[1]).item() if tensor[1] > 0 else 0.0


def main(cfg):
    device, local_rank = setup_distributed()
    rank = get_rank()

    with open(cfg["data"]["label_json"], "r") as f:
        labels_json = json.load(f)
    class_names = {int(x): y for x, y in labels_json["class_names"].items()}
    num_classes = len(class_names)

    os.makedirs("answers", exist_ok=True)

    seed = cfg["seed"] + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    if is_main_process():
        wandb.init(
            entity=cfg["wandb"]["entity"],
            project=cfg["wandb"]["project"],
            name=cfg["run_name"],
            config=cfg,
            mode="disabled" if cfg["run_name"] == "debug" else "online",
        )

    train_dataset = ClsDataset(
        partition="train",
        model_name=cfg["model_name"],
        num_classes=num_classes,
        predict_per_item=cfg["predict_per_item"],
        **cfg["data"],
    )
    val_dataset = ClsDataset(
        partition="val",
        model_name=cfg["model_name"],
        num_classes=num_classes,
        predict_per_item=cfg["predict_per_item"],
        **cfg["data"],
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["training"]["num_workers"] > 0,
        batch_size=cfg["training"]["train_bs"],
        num_workers=cfg["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg["training"]["num_workers"] > 0,
        batch_size=cfg["training"]["val_bs"],
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn_val,
    )

    model = HFModel(
        model_name=cfg["model_name"],
        num_classes=num_classes,
        predict_per_item=cfg["predict_per_item"],
        **cfg["model"],
    )
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        find_unused_parameters=False,
    )

    if is_main_process():
        tot = sum(p.numel() for p in model.module.state_dict().values())
        print(f"parameters: {tot:_d}")
        print(f"Dataset is multilabel: {train_dataset.is_multilabel}")

    if train_dataset.is_multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=cfg["training"]["label_smoothing"],
            weight=get_weights(
                train_dataset.json_data,
                cfg["model"]["class_weights"],
                device,
            ),
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = round(total_steps * cfg["training"]["part_warmup"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    mixup = (
        None
        if cfg["mixup_alpha"] is None
        else MixUp(alpha=cfg["mixup_alpha"], num_classes=cfg["training"]["train_bs"])
    )

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        answers = []
        losses = []
        train_sampler.set_epoch(epoch)
        progress = tqdm(train_loader, total=len(train_loader), disable=not is_main_process())
        for data, target in progress:
            data = data.to(device)
            target = target.to(device)

            batch_size = data.shape[0]
            eye = torch.eye(batch_size, device=device)

            optimizer.zero_grad()

            with torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
                if mixup is not None:
                    n, t, c, a, b = data.shape
                    data = data.reshape(n, t, c, a * b)
                    data, mix = mixup(data, eye)
                    data = data.reshape(n, t, c, a, b)
                    if cfg["predict_per_item"] != 1:
                        target = target.permute(1, 0, 2)
                        target = mix.unsqueeze(0) @ target
                        target = target.permute(1, 0, 2)
                    else:
                        target = mix @ target
                target = target.reshape(-1, num_classes)
                output = model(data)
                output_prob = (
                    output
                    if train_dataset.is_multilabel
                    else torch.nn.functional.softmax(output, 1)
                )
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            answers.extend(prep_for_answers(output_prob, target))
            losses.append(loss.item())
            if is_main_process():
                wandb.log({"batch_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
            if cfg["run_name"] == "debug":
                break

        train_loss = reduce_loss_sums(sum(losses), len(losses), device)
        gathered_answers = gather_lists_from_all_processes(answers)
        if is_main_process():
            logs = {
                **calculate_multiclass_metrics(gathered_answers, class_names, "train"),
                "train_loss": train_loss,
            }
            print(logs)
            wandb.log(logs)

        model.eval()
        answers = []
        losses = []
        with torch.no_grad():
            val_progress = tqdm(val_loader, total=len(val_loader), disable=not is_main_process())
            for data, target, names in val_progress:
                data = data.to(device)
                target = target.to(device)
                target = target.view(-1, num_classes)
                with torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
                    output = model(data)
                    loss = criterion(output, target)
                    output_prob = (
                        output
                        if train_dataset.is_multilabel
                        else torch.nn.functional.softmax(output, 1)
                    )
                    answers.extend(prep_for_answers(output_prob, target, names))
                    losses.append(loss.item())
                    if cfg["run_name"] == "debug":
                        break
        val_loss = reduce_loss_sums(sum(losses), len(losses), device)
        gathered_answers = gather_lists_from_all_processes(answers)
        if is_main_process():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(os.path.join("answers", f"{cfg['run_name']}_{timestamp}.json"), "w") as f:
                json.dump(gathered_answers, f)
            logs = {
                **calculate_multiclass_metrics(gathered_answers, class_names, "val"),
                **calculate_f1_metrics(
                    gathered_answers,
                    labels_json,
                    "val",
                    train_dataset.is_multilabel,
                    "val",
                ),
                "val_frame_level_map": calc_frame_level_map(
                    gathered_answers,
                    cfg["predict_per_item"],
                    labels_json,
                    "val",
                ),
                "val_loss": val_loss,
            }
            logs["val_raster_plot"] = wandb.Image(
                generate_raster_plot(gathered_answers, labels_json, "val")
            )
            print(logs)
            wandb.log(logs)

    if is_main_process():
        print("Finished training")
        if wandb.run is not None:
            wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    assert len(sys.argv) > 1 and len(sys.argv[1]) > 0, "Usage: python distributed_train.py <path_to_config.yaml>"

    with open(sys.argv[1], "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)