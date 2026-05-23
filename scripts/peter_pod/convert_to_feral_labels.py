"""Convert unified_gait_labels.json into feral's expected regression-labels schema.

feral's dataset reads:
  task: "regression"
  splits: {train: [fn,...], val: [fn,...], test: [...], inference: [...]}
  labels: {fn: [float, ...]}      <-- list of regression targets per video
  target_names + normalization optional

Each fn is the video path relative to the dataset prefix. We use the same keys
as in unified_gait_labels.json (e.g. "auto-gait/0.mp4", "tulip/gait/subject01_camera1.mp4").
The data prefix in the training config is /root/data_precrop, so feral will load
e.g. /root/data_precrop/tulip/gait/subject01_camera1.mp4.

Target = gait_severity, z-scored using train-set stats. Stats saved in the
output `normalization` block so val/test predictions can be inverted.
"""
import json
import numpy as np

SRC = "/root/labels/unified_gait_labels_precrop.json"  # use the precrop variant so splits stay aligned
BBOX = "/root/bboxes/unified_gait_bboxes_precrop.json"
DST  = "/root/labels/feral_gait_labels.json"

with open(SRC) as fp:
    src = json.load(fp)
with open(BBOX) as fp:
    bboxes = json.load(fp)
labels_by_key = src["labels"]
splits = src["splits"]
frame_mask = {k: bboxes[k]["patient_visible"] for k in labels_by_key if k in bboxes}

train_keys = splits.get("train", [])
val_keys   = splits.get("val", [])

# Compute z-score from train only
train_targets = np.array([float(labels_by_key[k]["gait_severity"]) for k in train_keys])
mean = float(train_targets.mean())
std  = float(train_targets.std())
if std < 1e-6:
    std = 1.0  # constant -> avoid div by zero

def norm(v: float) -> float:
    return (v - mean) / std

feral_labels = {k: [norm(float(labels_by_key[k]["gait_severity"]))] for k in labels_by_key}

out = {
    "task": "regression",
    "is_multilabel": False,
    "target_names": {"0": "gait_severity"},
    "normalization": {
        "mean": [mean],
        "std":  [std],
        "raw_label_range": [0.0, 4.0],
        "note": "z-scored from train split. raw = norm * std + mean.",
    },
    "splits": {
        "train":     train_keys,
        "val":       val_keys,
        "test":      [],
        "inference": [],
    },
    "labels": feral_labels,
    "frame_mask": frame_mask,
}

with open(DST, "w") as fp:
    json.dump(out, fp, indent=2)
print(f"wrote {DST}")
print(f"train: {len(train_keys)}  val: {len(val_keys)}")
print(f"train target stats raw: mean={mean:.3f}  std={std:.3f}")
arr = np.array([feral_labels[k][0] for k in train_keys])
print(f"train target stats z-scored: mean={arr.mean():.3f}  std={arr.std():.3f}  "
      f"min={arr.min():.3f}  max={arr.max():.3f}")
