"""Recompute per-video corr / r² on the ORIGINAL task videos only.

Splits val items into:
  - orig: the original task videos (subject*_camera*, CONTROL*, EXTPK*, etc.)
  - negs: appended FoG-neg pool = PDFE*, vlog_*, kinetics700__*, anything else
We detect "orig" by membership in the prior canonical val set, NOT by name pattern.
"""
import json, os
import numpy as np
from collections import defaultdict

ANS = "/root/feral/answers"

# Canonical (pre-negs) val splits — same videos the prior baselines used.
ORIG_VAL = {
    "chair":     json.load(open("/root/tulip_labels/chair_labels_resplit.json"))["splits"]["val"],
    "walking":   json.load(open("/root/labels/feral_gait_labels.json"))["splits"]["val"],
    "fingertap": json.load(open("/root/data/hubu-fis/hubu-fis_regression_labels.json"))["splits"]["val"],
}
ORIG_VAL = {k: set(v) for k, v in ORIG_VAL.items()}
for k, v in ORIG_VAL.items():
    print(f"orig val {k}: {len(v)} vids")
print()

BEST = {
    "chair_basic+negs (ep4 saved)":       ("chair",     "exp_chair_vitb_with_negs_2026-05-22_11-29-45.json"),
    "chair_basic+negs (ep7 best per-vid)":("chair",     "exp_chair_vitb_with_negs_2026-05-22_12-53-33.json"),
    "chair_strong+negs (ep4 saved)":      ("chair",     "exp_chair_strong_with_negs_2026-05-23_03-01-12.json"),
    "walking_basic+negs (ep2 saved)":     ("walking",   "exp_gait_vitb_with_negs_2026-05-22_16-11-54.json"),
    "walking_basic+negs (ep5 last)":      ("walking",   "exp_gait_vitb_with_negs_2026-05-22_18-29-42.json"),
    "walking_strong+negs (ep2 saved)":    ("walking",   "exp_gait_strong_with_negs_2026-05-23_12-02-03.json"),
    "walking_strong+negs (ep5 last/peak)":("walking",   "exp_gait_strong_with_negs_2026-05-23_15-58-31.json"),
    "fingertap_basic+negs (ep3 saved)":   ("fingertap", "exp_fingertap_vitb_with_negs_2026-05-22_20-33-05.json"),
    "fingertap_basic+negs (ep5)":         ("fingertap", "exp_fingertap_vitb_with_negs_2026-05-22_21-34-24.json"),
    "fingertap_strong+negs (ep2 saved)":  ("fingertap", "exp_fingertap_strong_with_negs_2026-05-23_18-54-37.json"),
}

def per_video(items):
    p_by, t_by = defaultdict(list), {}
    for vid_meta, pred, tgt in items:
        v = vid_meta[0]
        p_by[v].append(pred[0])
        t_by[v] = tgt[0]
    return {v: (float(np.mean(p_by[v])), float(t_by[v])) for v in p_by}

def corr_r2(items):
    if len(items) < 2:
        return float("nan"), float("nan"), len(items)
    p = np.array([x[0] for x in items], dtype=np.float64)
    t = np.array([x[1] for x in items], dtype=np.float64)
    if p.std() < 1e-9 or t.std() < 1e-9:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(p, t)[0, 1])
    mse = float(((p - t) ** 2).mean())
    var = float(t.var())
    r2 = float("nan") if var < 1e-9 else 1.0 - mse / var
    return corr, r2, len(items)

print(f"{'experiment':<42} | orig-only        | full (orig+negs)")
print(f"{'':<42} | n  corr   r²     | n   corr   r²")
print("-" * 110)
for name, (task, fn) in BEST.items():
    p = os.path.join(ANS, fn)
    if not os.path.exists(p):
        print(f"{name:<42} | MISSING {fn}")
        continue
    items = json.load(open(p))
    pv = per_video(items)
    orig_set = ORIG_VAL[task]
    orig_items = [pv[v] for v in pv if v in orig_set]
    full_items = list(pv.values())
    cO, rO, nO = corr_r2(orig_items)
    cF, rF, nF = corr_r2(full_items)
    print(f"{name:<42} | {nO:>2d} {cO:+.3f} {rO:+.3f} | {nF:>3d} {cF:+.3f} {rF:+.3f}")
