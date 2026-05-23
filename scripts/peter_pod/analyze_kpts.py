"""Per-dataset fraction of frames with hips+knees+ankles visible."""
import glob, os
import pandas as pd
import numpy as np

LB = [11, 12, 13, 14, 15, 16]
NEED_COLS = ["video", "frame", "det_score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
for k in LB:
    NEED_COLS += [f"k{k}_x", f"k{k}_y"]

def kpt_in_bbox(df, idxs):
    mask = np.ones(len(df), dtype=bool)
    bx1 = df["bbox_x1"].to_numpy(); bx2 = df["bbox_x2"].to_numpy()
    by1 = df["bbox_y1"].to_numpy(); by2 = df["bbox_y2"].to_numpy()
    for k in idxs:
        x = df[f"k{k}_x"].to_numpy(); y = df[f"k{k}_y"].to_numpy()
        mask &= (x > bx1) & (x < bx2) & (y > by1) & (y < by2)
    return mask

def analyze(glob_pat, label):
    files = sorted(glob.glob(glob_pat))
    if not files:
        print(f"!! no parquets at {glob_pat}")
        return None
    tot_frames = 0
    det_frames = 0
    lb_frames = 0
    n_videos = 0
    for f in files:
        try:
            df = pd.read_parquet(f, columns=NEED_COLS)
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        if df.empty: continue
        n_videos += 1
        frames_in_video = int(df["frame"].max()) + 1
        tot_frames += frames_in_video
        det = df[df["det_score"].notna()]
        det_frames += det["frame"].nunique()
        if len(det) == 0: continue
        ok = kpt_in_bbox(det, LB)
        lb_frames += det.loc[ok, "frame"].nunique()
    return label, n_videos, tot_frames, det_frames, lb_frames

DATASETS = [
    ("/root/data/auto-gait/poses_sapiens03b_goliath/*.parquet", "auto-gait"),
    ("/root/data/gavd/poses_sapiens03b_goliath/*.parquet", "gavd"),
    ("/root/data/koa-pd-nm-gait/poses_sapiens03b_goliath/*.parquet", "koa-pd-nm-gait"),
    ("/root/data/tulip/gait/poses_sapiens03b_goliath/*.parquet", "tulip/gait"),
    ("/root/data/tulip/stability/poses_sapiens03b_goliath/*.parquet", "tulip/stability"),
    ("/root/data/tulip/spontaneity/poses_sapiens03b_goliath/*.parquet", "tulip/spontaneity"),
]

hdr = ("dataset", "videos", "frames", "det_frames", "det%", "lb_frames", "lb%")
print("{:<22} {:>6} {:>10} {:>12} {:>6} {:>12} {:>6}".format(*hdr))
print("-" * 80)
totals = [0, 0, 0]
for pat, label in DATASETS:
    r = analyze(pat, label)
    if r is None: continue
    label, n_vid, tot_f, det_f, lb_f = r
    detp = f"{det_f/max(tot_f,1):.1%}"
    lbp = f"{lb_f/max(tot_f,1):.1%}"
    print("{:<22} {:>6,} {:>10,} {:>12,} {:>6} {:>12,} {:>6}".format(label, n_vid, tot_f, det_f, detp, lb_f, lbp))
    totals[0] += tot_f; totals[1] += det_f; totals[2] += lb_f
print("-" * 80)
detp = f"{totals[1]/max(totals[0],1):.1%}"
lbp = f"{totals[2]/max(totals[0],1):.1%}"
print("{:<22} {:>6} {:>10,} {:>12,} {:>6} {:>12,} {:>6}".format("TOTAL", "", totals[0], totals[1], detp, totals[2], lbp))
