"""Score every deployable checkpoint on the canonical (orig) val split.

A 'deployable' checkpoint is one whose `_best_checkpoint.pt` is on disk.
Each best_checkpoint.pt's mtime matches the answers JSON of the last epoch
that was saved as best. So per checkpoint, we use the answers file
with the matching timestamp.
"""
import json, os, re, glob
import numpy as np
from collections import defaultdict
from datetime import datetime

ANS = "/root/feral/answers"
CKPT = "/root/feral/checkpoints"

ORIG_VAL = {
    "chair":     set(json.load(open("/root/tulip_labels/chair_labels_resplit.json"))["splits"]["val"]),
    "walking":   set(json.load(open("/root/labels/feral_gait_labels.json"))["splits"]["val"]),
    "fingertap": set(json.load(open("/root/data/hubu-fis/hubu-fis_regression_labels.json"))["splits"]["val"]),
}

# experiment-name → task
TASK = {
    "exp_chair_vitb_with_negs": "chair",
    "exp_chair_strong_with_negs": "chair",
    "exp_gait_vitb_with_negs": "walking",
    "exp_gait_strong_with_negs": "walking",
    "exp_gait_unified_mixup": "walking",
    "exp_gait_mixup_tnoise": "walking",
    "exp_gait_tnoise": "walking",
    "exp_fingertap_vitb_with_negs": "fingertap",
    "exp_fingertap_strong_with_negs": "fingertap",
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
    corr = float("nan") if p.std() < 1e-9 or t.std() < 1e-9 else float(np.corrcoef(p, t)[0, 1])
    mse = float(((p - t) ** 2).mean())
    var = float(t.var())
    r2 = float("nan") if var < 1e-9 else 1.0 - mse / var
    return corr, r2, len(items)

# For each best_checkpoint.pt, find the matching-mtime answers JSON
def match_answers(exp_name, ckpt_mtime):
    """Find the answers JSON whose timestamp is closest to ckpt mtime."""
    pattern = os.path.join(ANS, exp_name + "_*.json")
    files = glob.glob(pattern)
    best, best_diff = None, 1e18
    for f in files:
        m = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json$", f)
        if not m: continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S").timestamp()
        diff = abs(ts - ckpt_mtime)
        if diff < best_diff:
            best_diff, best = diff, f
    return best, best_diff

print(f"{'experiment':<40} | task     | ckpt_epoch_ts       | n  corr   r²    | full_corr full_r²")
print("-" * 130)
results = defaultdict(list)
for ckpt in sorted(glob.glob(os.path.join(CKPT, "*_best_checkpoint.pt"))):
    name = os.path.basename(ckpt).replace("_best_checkpoint.pt", "")
    if name not in TASK: continue
    task = TASK[name]
    mtime = os.path.getmtime(ckpt)
    ans, diff = match_answers(name, mtime)
    if ans is None:
        print(f"{name:<40} | {task:<8} | NO ANSWERS FOUND")
        continue
    ts_str = datetime.fromtimestamp(os.path.getmtime(ans)).strftime("%m-%d %H:%M")
    items = json.load(open(ans))
    pv = per_video(items)
    orig = [pv[v] for v in pv if v in ORIG_VAL[task]]
    full = list(pv.values())
    cO, rO, nO = corr_r2(orig)
    cF, rF, nF = corr_r2(full)
    results[task].append((name, cO, rO, nO, cF, rF))
    print(f"{name:<40} | {task:<8} | {ts_str}        | {nO:>2d} {cO:+.3f} {rO:+.3f} | {cF:+.3f}     {rF:+.3f}")

print()
print("=" * 130)
print("BEST DEPLOYABLE CHECKPOINT PER TASK (by orig-val corr):")
for task in ["chair", "walking", "fingertap"]:
    if not results[task]: continue
    best = max(results[task], key=lambda x: x[1] if not np.isnan(x[1]) else -1)
    print(f"  {task:<10}: {best[0]:<40} corr={best[1]:+.3f}  r²={best[2]:+.3f}  (orig n={best[3]})")
