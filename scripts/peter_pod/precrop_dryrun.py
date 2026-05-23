"""Dry run: compute union_large per video and report any problematic cases."""
import json, os
import numpy as np

bboxes = json.load(open("/root/bboxes/unified_gait_bboxes.json"))
labels = json.load(open("/root/labels/unified_gait_labels.json"))
video_roots = labels["video_roots"]

bad = []
sizes = []
short_sides = []
n_missing_video = 0
for key, m in bboxes.items():
    head, _, video = key.rpartition("/")
    src = os.path.join(video_roots[head], video)
    if not os.path.exists(src):
        n_missing_video += 1
        bad.append((key, "missing source"))
        continue
    W, H = int(m["img_w"]), int(m["img_h"])
    large = np.asarray(m["large_bboxes"], dtype=np.float32)
    vis = np.asarray(m["patient_visible"], dtype=bool)
    if not vis.any():
        bad.append((key, "no visible frames"))
        continue
    b = large[vis]
    x1 = max(0, int(np.floor(b[:, 0].min())))
    y1 = max(0, int(np.floor(b[:, 1].min())))
    x2 = min(W, int(np.ceil(b[:, 2].max())))
    y2 = min(H, int(np.ceil(b[:, 3].max())))
    cw, ch = x2 - x1, y2 - y1
    sizes.append((cw, ch, W, H, key))
    short_sides.append(min(cw, ch))
    if cw <= 0 or ch <= 0:
        bad.append((key, f"degenerate crop {cw}x{ch} from {W}x{H}"))
    elif cw < 64 or ch < 64:
        bad.append((key, f"tiny crop {cw}x{ch} from {W}x{H}"))

print(f"total videos: {len(bboxes)}")
print(f"missing source files: {n_missing_video}")
print(f"bad (degenerate/tiny): {len(bad)}")
for k, reason in bad[:20]:
    print(f"  {k:<55}  {reason}")
short_arr = np.array(short_sides)
print()
print(f"crop short-side distribution (min, p10, p50, p90, max):")
print(f"  {short_arr.min()}, {np.percentile(short_arr,10):.0f}, "
      f"{np.percentile(short_arr,50):.0f}, {np.percentile(short_arr,90):.0f}, {short_arr.max()}")
print()
# Per-dataset
from collections import defaultdict
ds_sizes = defaultdict(list)
for cw, ch, W, H, key in sizes:
    head = key.rpartition("/")[0]
    ds_sizes[head].append((cw, ch, W, H))
for head, ss in ds_sizes.items():
    arr_cw = np.array([s[0] for s in ss])
    arr_ch = np.array([s[1] for s in ss])
    arr_orig_w = np.array([s[2] for s in ss])
    src_h = np.array([s[3] for s in ss])
    print(f"{head:<20} n={len(ss):<3} "
          f"crop avg {arr_cw.mean():.0f}x{arr_ch.mean():.0f}  "
          f"(orig avg {arr_orig_w.mean():.0f}x{src_h.mean():.0f})  "
          f"crop/orig area ratio avg {(arr_cw*arr_ch).mean()/(arr_orig_w*src_h).mean():.1%}")
