import sys
sys.path.insert(0, "/root")
from gait_aug import GaitUnifiedDataset

for split in ["train", "val"]:
    ds = GaitUnifiedDataset(
        bboxes_json="/root/bboxes/unified_gait_bboxes.json",
        labels_json="/root/labels/unified_gait_labels.json",
        split=split, chunk_len=64, out_hw=(384, 384), stride=1,
    )
    print(f"split={split:<6} videos={len(ds.videos):<5} chunks={len(ds):<7}")

# Load one sample
ds = GaitUnifiedDataset(
    bboxes_json="/root/bboxes/unified_gait_bboxes.json",
    labels_json="/root/labels/unified_gait_labels.json",
    split="train", chunk_len=16, out_hw=(224, 224), stride=2,
)
print(f"\ntrain @ chunk_len=16 stride=2 224x224 -> {len(ds)} chunks")
sample = ds[len(ds) // 2]
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

# Per-dataset chunk breakdown
from collections import Counter
breakdown = Counter(k.rsplit("/", 1)[0] for k, _ in ds.chunks)
print("\nper-dataset chunk counts (train):", dict(breakdown))
