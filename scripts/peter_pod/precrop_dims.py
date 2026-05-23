"""Compare union vs per-frame max for koa-pd-nm-gait to confirm the bug."""
import json, numpy as np
b = json.load(open("/root/bboxes/unified_gait_bboxes.json"))
for ds_name, ds_filter in [("auto-gait", "auto-gait/"),
                            ("koa-pd-nm-gait", "koa-pd-nm-gait/"),
                            ("tulip/gait", "tulip/gait/")]:
    union_ratios = []
    perframe_max_ratios = []
    perframe_mean_ratios = []
    for key, m in b.items():
        if not key.startswith(ds_filter): continue
        W, H = m["img_w"], m["img_h"]
        large = np.asarray(m["large_bboxes"], dtype=np.float32)
        vis = np.asarray(m["patient_visible"], dtype=bool)
        if not vis.any(): continue
        bx = large[vis]
        # Union
        union_w = max(0, bx[:, 2].max() - bx[:, 0].min())
        union_h = max(0, bx[:, 3].max() - bx[:, 1].min())
        union_ratios.append((union_w * union_h) / (W * H))
        # Per-frame max dim (the size needed to fit any single frame's bbox)
        pf_w = (bx[:, 2] - bx[:, 0]).max()
        pf_h = (bx[:, 3] - bx[:, 1]).max()
        perframe_max_ratios.append((pf_w * pf_h) / (W * H))
        # Per-frame avg area (just for reference)
        pf_areas = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
        perframe_mean_ratios.append(pf_areas.mean() / (W * H))
    ur = np.array(union_ratios)
    pm = np.array(perframe_max_ratios)
    pa = np.array(perframe_mean_ratios)
    print(f"{ds_name:<18} n={len(ur):<3}  "
          f"union {ur.mean():.1%}   "
          f"per-frame-MAX-bbox {pm.mean():.1%}   "
          f"per-frame-mean-bbox {pa.mean():.1%}")
