"""Parquet-driven patient crop (uses the real Sapiens pose detections, no SAM
guessing). Auto-seeds the patient as the track that WALKS the most (clinicians
are static), tracks it with the same greedy-NN logic as propagate_patient, then
smooth-pan crops to the patient's (padded) bbox. CPU + parallel; frees the GPU.

Out:
  /workspace/data_gaitcrop/<fn>            384x384 mp4 (libx264 crf23 yuv420p)
  /workspace/cropmeta/<fn>.json            per-frame crop params for the mask step
Fallback (no parquet): centred full-frame square crop, logged.
"""
import os, sys, json, glob, subprocess, traceback
import numpy as np, cv2, pandas as pd
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/workspace/feral/scripts/peter_pod")
from gait_aug import compute_bboxes_for_video  # reuse the exact bbox builder

LABELS = "/workspace/labels/feral_gait_labels_phoneaug_cleanval.json"
RAW = "/workspace/data_gaitraw"; CROP = "/workspace/data_gaitcrop"
META = "/workspace/cropmeta"; POSE = "/workspace/poses"
OUT = 384; PAD_SIDE = 1.15; SIGMA = 8.0
# dataset-root -> pose subdir under /workspace/poses
POSEDIR = {"auto-gait": "auto-gait", "koa-pd-nm-gait": "koa-pd-nm-gait", "tulip": "tulip"}
BASE_COLS = ["video", "frame", "det_score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
LB = list(range(17)); KPT_COLS = sum([[f"k{k}_x", f"k{k}_y"] for k in (13, 14, 15, 16)], [])

os.makedirs(CROP, exist_ok=True); os.makedirs(META, exist_ok=True)


def parquet_path(fn):
    root = fn.split("/")[0]; base = os.path.basename(fn)
    pd_ = os.path.join(POSE, POSEDIR.get(root, root))
    for cand in (base + ".parquet", os.path.splitext(base)[0] + ".parquet"):
        p = os.path.join(pd_, cand)
        if os.path.exists(p):
            return p
    return None


def track_patient(df, seed_idx):
    df = df.copy()
    df["xc"] = (df["bbox_x1"] + df["bbox_x2"]) / 2
    df["yc"] = (df["bbox_y1"] + df["bbox_y2"]) / 2
    seed = df.loc[seed_idx]; sf = int(seed["frame"])
    by = {f: g for f, g in df.groupby("frame")}
    frames = sorted(by)
    chosen = {sf: seed_idx}
    lx, ly = float(seed["xc"]), float(seed["yc"])
    for f in [x for x in frames if x > sf]:
        g = by[f]; d = (g["xc"] - lx) ** 2 + (g["yc"] - ly) ** 2
        pick = g.loc[d.idxmin()]; chosen[f] = pick.name; lx, ly = float(pick["xc"]), float(pick["yc"])
    lx, ly = float(seed["xc"]), float(seed["yc"])
    for f in [x for x in reversed(frames) if x < sf]:
        g = by[f]; d = (g["xc"] - lx) ** 2 + (g["yc"] - ly) ** 2
        pick = g.loc[d.idxmin()]; chosen[f] = pick.name; lx, ly = float(pick["xc"]), float(pick["yc"])
    return df.loc[[chosen[f] for f in sorted(chosen)]]


def auto_seed(df):
    real = df[df["det_score"].notna()]
    if real.empty: return None
    mid = int(real["frame"].median())
    cf = int(real.iloc[(real["frame"] - mid).abs().argsort().iloc[0]]["frame"])
    cands = real[real["frame"] == cf]
    best, bd = None, -1
    for idx in cands.index:
        tr = track_patient(df, idx); rt = tr[tr["det_score"].notna()]
        if len(rt) < 2: disp = 0.0
        else:
            xc = (rt["bbox_x1"] + rt["bbox_x2"]).values / 2; yc = (rt["bbox_y1"] + rt["bbox_y2"]).values / 2
            disp = float(np.sqrt(np.diff(xc) ** 2 + np.diff(yc) ** 2).sum())
        if disp > bd: bd, best = disp, tr
    return best


def smooth_params(large, vis, W, H):
    n = len(large); large = np.asarray(large, float)
    cx = (large[:, 0] + large[:, 2]) / 2; cy = (large[:, 1] + large[:, 3]) / 2
    side = np.maximum(large[:, 2] - large[:, 0], large[:, 3] - large[:, 1]) * PAD_SIDE
    vis = np.asarray(vis, bool); idx = np.arange(n)
    if vis.any():
        first = int(np.argmax(vis)); last = n - 1 - int(np.argmax(vis[::-1]))
        for a in (cx, cy, side):
            a[:first] = a[first]; a[last + 1:] = a[last]
            good = vis | (idx < first) | (idx > last)
            if (~good).any(): a[~good] = np.interp(idx[~good], idx[good], a[good])
    cx = gaussian_filter1d(cx, SIGMA); cy = gaussian_filter1d(cy, SIGMA); side = gaussian_filter1d(side, SIGMA)
    side = np.clip(side, 24, min(W, H))
    cx = np.clip(cx, side / 2, W - side / 2); cy = np.clip(cy, side / 2, H - side / 2)
    return cx, cy, side


def process(fn):
    out_vid = os.path.join(CROP, fn); out_meta = os.path.join(META, fn + ".json")
    if os.path.exists(out_vid) and os.path.exists(out_meta):
        return fn, "skip", 0
    raw = os.path.join(RAW, fn)
    if not os.path.exists(raw):
        return fn, "missing_raw", 0
    os.makedirs(os.path.dirname(out_vid), exist_ok=True); os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    cap = cv2.VideoCapture(raw); W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pq = parquet_path(fn); mode = "parquet"
    if pq is not None:
        try:
            df = pd.read_parquet(pq, columns=[c for c in BASE_COLS + KPT_COLS])
            n_frames = max(nf, int(df["frame"].max()) + 1)
            tr = auto_seed(df)
            pay = compute_bboxes_for_video(tr, n_frames, W or 1280, H or 720, float(fps))
            large = np.array(pay["large_bboxes"]); vis = np.array(pay["patient_visible"])
            if len(large) < nf:  # pad
                large = np.vstack([large, np.repeat(large[-1:], nf - len(large), 0)]); vis = np.concatenate([vis, np.repeat(vis[-1:], nf - len(vis))])
            cx, cy, side = smooth_params(large[:nf], vis[:nf], W, H)
        except Exception:
            mode = "fallback_err"; pq = None
    if pq is None:
        s = min(W, H); cx = np.full(nf, W / 2); cy = np.full(nf, H / 2); side = np.full(nf, s)
        mode = "fallback_center" if mode != "fallback_err" else "fallback_err"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{OUT}x{OUT}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264",
                           "-crf", "23", "-pix_fmt", "yuv420p", out_vid], stdin=subprocess.PIPE)
    params = []
    for i in range(nf):
        ok, fr = cap.read()
        if not ok: break
        x0 = int(round(cx[i] - side[i] / 2)); y0 = int(round(cy[i] - side[i] / 2)); s = int(round(side[i]))
        x0 = max(0, min(x0, W - s)); y0 = max(0, min(y0, H - s))
        crop = fr[y0:y0 + s, x0:x0 + s]
        crop = cv2.cvtColor(cv2.resize(crop, (OUT, OUT)), cv2.COLOR_BGR2RGB)
        ff.stdin.write(crop.astype(np.uint8).tobytes())
        params.append([int(x0), int(y0), int(s)])
    ff.stdin.close(); ff.wait(); cap.release()
    json.dump({"W": W, "H": H, "fps": fps, "out": OUT, "params": params, "mode": mode}, open(out_meta, "w"))
    return fn, mode, len(params)


def main():
    lab = json.load(open(LABELS))
    vids = sorted(set(v for v in lab["splits"]["train"] + lab["splits"]["val"] if not v.startswith("fog_negs/")))
    print(f"crop {len(vids)} gait videos (parquet-driven patient track)", flush=True)
    modes = {}
    with ProcessPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(process, fn): fn for fn in vids}
        for j, fut in enumerate(as_completed(futs)):
            try:
                fn, mode, n = fut.result()
            except Exception:
                fn, mode, n = futs[fut], "FAIL", 0
                print("FAIL", fn, traceback.format_exc()[-200:], flush=True)
            modes[mode] = modes.get(mode, 0) + 1
            if j % 25 == 0 or mode.startswith("fallback") or mode == "FAIL":
                print(f"[{j+1}/{len(vids)}] {fn} {mode} nf={n} :: {modes}", flush=True)
    print("DONE", modes, flush=True)


if __name__ == "__main__":
    main()
