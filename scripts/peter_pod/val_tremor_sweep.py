"""Re-evaluate the best tremor checkpoint at two new chunkings:
    (chunk_length=32, chunk_step=2)  -- 64-frame span, sparse
    (chunk_length=16, chunk_step=4)  -- 64-frame span, sparser

Loads exp_tremor_cam34_vitb_best_checkpoint.pt and runs feral's evaluate()
+ regression metrics on the cam34 val split. Reports per-target val_mse and
val_corr at chunk and video level.
"""
import os, sys, json, time, copy
sys.path.insert(0, "/root/feral")
import torch
import yaml

from feral.model import FeralModel
from feral.dataset import ClsDataset
from feral.loops import evaluate
from feral.metrics import calculate_regression_metrics, calculate_video_level_regression_metrics
from torch.utils.data import DataLoader

CKPT  = "/root/feral/checkpoints/exp_tremor_cam34_vitb_best_checkpoint.pt"
CFG   = "/root/configs_gait/exp_tremor_cam34_vitb.yaml"
LABEL = "/root/tulip_labels/tremor_labels_resplit_cam34.json"
DEVICE = "cuda"

with open(CFG) as fp:
    cfg = yaml.safe_load(fp)
labels_doc = json.load(open(LABEL))
num_targets = len(labels_doc["target_names"])
target_names = labels_doc["target_names"]

# Build model from the trained config and load checkpoint
model = FeralModel(
    backbone=cfg["backbone"], num_classes=1,
    predict_per_item=cfg["predict_per_item"],
    fc_drop_rate=cfg["model"]["fc_drop_rate"],
    freeze_encoder_layers=cfg["model"]["freeze_encoder_layers"],
    pretrained=False, task="regression", num_targets=num_targets,
).to(DEVICE).eval()
sd = torch.load(CKPT, map_location=DEVICE, weights_only=False)
if isinstance(sd, dict):
    for k in ("state_dict", "model"):
        if k in sd:
            sd = sd[k]; break
model.load_state_dict(sd, strict=True)
print(f"loaded {CKPT}")

CONFIGS = [
    ("32x2", 32, 2),
    ("16x4", 16, 4),
]

for label, chunk_length, chunk_step in CONFIGS:
    print(f"\n=== eval at chunk_length={chunk_length}, chunk_step={chunk_step} "
          f"(spans {(chunk_length-1)*chunk_step+1} frames per chunk) ===")
    ds = ClsDataset(
        partition="val", label_json_dict=labels_doc, do_aa=False,
        predict_per_item=chunk_length, num_classes=1, num_targets=num_targets,
        prefix="/root/tulip/tremor_videos", resize_to=256,
        chunk_shift=chunk_length, chunk_length=chunk_length, chunk_step=chunk_step,
        resize_style="square", part_sample=1.0, subsample_keep_rare_threshold=None,
        task="regression",
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
    t0 = time.perf_counter()
    answers, _ = evaluate(model, loader, criterion=None, num_classes=1,
                          is_multilabel=False, device=DEVICE, task="regression")
    t1 = time.perf_counter()

    chunk_metrics = calculate_regression_metrics(answers, target_names=target_names, prefix="val_")
    vid_metrics   = calculate_video_level_regression_metrics(
        answers, labels_json=labels_doc, partition="val",
        target_names=target_names, prefix="val_vid_",
    )

    print(f"chunks={len(ds.samples)}  videos={len({s[0] for s in ds.samples})}  time={t1-t0:.1f}s")
    keep = lambda k: k.endswith("_mse") or k.endswith("_corr") or k.endswith("_mae")
    print("  per-target:")
    for t in target_names.values():
        for kind in ("mse", "corr", "mae"):
            k = f"val_{kind}_{t}"
            if k in chunk_metrics:
                vk = f"val_vid_{kind}_{t}"
                print(f"    {t:<10} {kind:<4} chunk={chunk_metrics[k]:>7.3f}  vid={vid_metrics.get(vk, float('nan')):>7.3f}")
    print("  overall:")
    for kind in ("mse", "mae", "r2", "corr"):
        ck = f"val_{kind}"; vk = f"val_vid_{kind}"
        if ck in chunk_metrics:
            print(f"    {kind:<4} chunk={chunk_metrics[ck]:>7.3f}  vid={vid_metrics.get(vk, float('nan')):>7.3f}")
