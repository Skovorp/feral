import json, sys, numpy as np
sys.path.insert(0, "/root")
from gait_aug import GaitVideoDataset

CHUNK_LEN = 64
STRIDE = 1
CONFIGS = [
    ("auto-gait",     "/root/bboxes/bboxes_auto-gait.json",     "/root/data/auto-gait/videos"),
    ("koa-pd-nm-gait","/root/bboxes/bboxes_koa-pd-nm-gait.json","/root/data/koa-pd-nm-gait/videos"),
    ("tulip/gait",    "/root/bboxes/bboxes_tulip-gait.json",    "/root/data/tulip/gait_videos"),
]
print("=== visibility (fraction of frames with patient_visible=True) ===")
for name, bjson, vroot in CONFIGS:
    raw = json.load(open(bjson))
    vt = ft = 0
    for v, m in raw.items():
        vis = np.array(m["patient_visible"], dtype=bool)
        vt += int(vis.sum())
        ft += len(vis)
    print("  {:<18} {:>6.1%}  ({:,}/{:,} frames)".format(name, vt / max(ft, 1), vt, ft))

print()
print("=== valid chunks at chunk_len={}, stride={}, 50% overlap ===".format(CHUNK_LEN, STRIDE))
print("{:<18} {:>7} {:>9} {:>8}".format("dataset", "videos", "chunks", "avg/vid"))
for name, bjson, vroot in CONFIGS:
    raw = json.load(open(bjson))
    ds = GaitVideoDataset(video_root=vroot, bboxes_json=bjson,
                          chunk_len=CHUNK_LEN, stride=STRIDE, out_hw=(384, 384))
    nv = len(ds.videos)
    print("  {:<18} {:>7,} {:>9,} {:>8.1f}".format(name, nv, len(ds), len(ds) / max(nv, 1)))
