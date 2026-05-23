"""Across all visible-frame large_bboxes, what % have at least one side > 512?

Also: the user-proposed "square = large_bbox + 50% on each side" --> the side length
is max(bw, bh) * 2.0 (since +50% on EACH side means the square has to encompass
that). Report how often THAT goes above 512 too.
"""
import json, numpy as np
b = json.load(open("/root/bboxes/unified_gait_bboxes.json"))

# Per-frame: large bbox dims
total_large = 0
large_with_side_gt_512 = 0
# Per-frame: required side of the +50%-padded square
square_sides = []

for key, m in b.items():
    large = np.asarray(m["large_bboxes"], dtype=np.float32)
    vis = np.asarray(m["patient_visible"], dtype=bool)
    if not vis.any(): continue
    bx = large[vis]
    bw = bx[:, 2] - bx[:, 0]
    bh = bx[:, 3] - bx[:, 1]
    total_large += len(bx)
    large_with_side_gt_512 += int(((bw > 512) | (bh > 512)).sum())

    # Square = encompassing bbox + 50% margin per side => side_of_square = max(bw, bh) * 2
    side = np.maximum(bw, bh) * 2.0
    square_sides.append(side)

square_sides = np.concatenate(square_sides)
print(f"frames with visible patient:                       {total_large:,}")
print(f"  large bbox has at least one side > 512:          {large_with_side_gt_512:,}  ({large_with_side_gt_512/max(total_large,1):.2%})")
print(f"  +50%-margin square side > 512:                   {int((square_sides>512).sum()):,}  ({(square_sides>512).mean():.2%})")
print()
print("large bbox side (width OR height, per frame) percentile:")
all_sides = []
for key, m in b.items():
    large = np.asarray(m["large_bboxes"], dtype=np.float32)
    vis = np.asarray(m["patient_visible"], dtype=bool)
    if not vis.any(): continue
    bx = large[vis]
    bw = bx[:, 2] - bx[:, 0]
    bh = bx[:, 3] - bx[:, 1]
    all_sides.append(np.maximum(bw, bh))
all_sides = np.concatenate(all_sides)
for p in [10, 25, 50, 75, 90, 95, 99, 99.9, 100]:
    print(f"  p{p:>5}:  max(bw,bh) = {np.percentile(all_sides, p):.0f}")
print()
print("required square-side (with +50% margin) percentile:")
for p in [10, 25, 50, 75, 90, 95, 99, 99.9, 100]:
    print(f"  p{p:>5}:  side = {np.percentile(square_sides, p):.0f}")
print()
# Per-dataset
print("by dataset (square-side > 512 fraction):")
for prefix in ["auto-gait/", "koa-pd-nm-gait/", "tulip/gait/"]:
    sides = []
    for key, m in b.items():
        if not key.startswith(prefix): continue
        large = np.asarray(m["large_bboxes"], dtype=np.float32)
        vis = np.asarray(m["patient_visible"], dtype=bool)
        if not vis.any(): continue
        bx = large[vis]
        side = np.maximum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]) * 2.0
        sides.append(side)
    sides = np.concatenate(sides)
    print(f"  {prefix:<18} n={len(sides):>8,}  median_side={np.median(sides):.0f}  "
          f"p95_side={np.percentile(sides,95):.0f}  fraction>512: {(sides>512).mean():.2%}")
