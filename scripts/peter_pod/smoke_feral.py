"""Verify feral can load the new gait dataset + count train/val chunks."""
import sys; sys.path.insert(0, "/root/feral")
import yaml, json
from feral.dataset import ClsDataset

with open("/root/exp_gait_mixup.yaml") as fp:
    cfg = yaml.safe_load(fp)
with open(cfg["data"]["label_json"]) as fp:
    labels_doc = json.load(fp)

shared = dict(
    label_json_dict=labels_doc,
    task="regression",
    num_classes=1,
    num_targets=1,
    predict_per_item=cfg["predict_per_item"],
    prefix=cfg["data"]["prefix"],
    resize_to=cfg["data"]["resize_to"],
    chunk_shift=cfg["data"]["chunk_shift"],
    chunk_length=cfg["data"]["chunk_length"],
    chunk_step=cfg["data"]["chunk_step"],
    resize_style=cfg["data"]["resize_style"],
    part_sample=cfg["data"]["part_sample"],
    do_aa=cfg["data"]["do_aa"],
    subsample_keep_rare_threshold=cfg["data"]["subsample_keep_rare_threshold"],
)
for split in ("train", "val"):
    ds = ClsDataset(partition=split, **shared)
    n_vids = len({s[0] for s in ds.samples})
    print(f"{split:<6} chunks={len(ds.samples):>6} videos={n_vids}")

ds = ClsDataset(partition="train", **shared)
video, target = ds[len(ds.samples) // 2]
print(f"sample[mid]: video shape={tuple(video.shape)}, target={target}")
