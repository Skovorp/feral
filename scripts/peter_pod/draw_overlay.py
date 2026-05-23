"""Draw lower-body keypoint overlay on a few sample frames per dataset.

For each dataset: pick 4 videos, pick 4 frames per video (evenly spaced over the
detected-frame range), draw bbox + COCO lower-body kpts (hips/knees/ankles) +
the 6-kpt anatomy check (red bbox = lb-kpts NOT all inside bbox or anat ordering bad).
"""
import os, glob, sys
import pandas as pd
import numpy as np
import cv2

LB = [11, 12, 13, 14, 15, 16]
NAMES = {11:"Lh",12:"Rh",13:"Lk",14:"Rk",15:"La",16:"Ra"}
COLOR = {11:(255,128,0),12:(255,128,0),13:(0,255,255),14:(0,255,255),15:(0,255,0),16:(0,255,0)}
NEED = ["video","frame","det_score","bbox_x1","bbox_y1","bbox_x2","bbox_y2"]
for k in LB: NEED += [f"k{k}_x", f"k{k}_y"]

def in_bbox(row, k):
    return (row["bbox_x1"] < row[f"k{k}_x"] < row["bbox_x2"] and
            row["bbox_y1"] < row[f"k{k}_y"] < row["bbox_y2"])

def anat_ok(row):
    return (row["k11_y"] < row["k13_y"] < row["k15_y"] and
            row["k12_y"] < row["k14_y"] < row["k16_y"])

def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else None

def render_one(video_path, df_video, frames, save_path, label):
    """Render a single composite image stacking up to 4 frames horizontally."""
    panels = []
    for fr in frames:
        img = read_frame(video_path, int(fr))
        if img is None:
            continue
        H, W = img.shape[:2]
        rows = df_video[df_video["frame"] == fr]
        for _, r in rows.iterrows():
            all_in = all(in_bbox(r, k) for k in LB)
            anat = anat_ok(r)
            color = (0,255,0) if (all_in and anat) else (0,0,255)
            x1,y1,x2,y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
            cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
            for k in LB:
                kx, ky = int(r[f"k{k}_x"]), int(r[f"k{k}_y"])
                cv2.circle(img, (kx,ky), 6, COLOR[k], -1)
                cv2.circle(img, (kx,ky), 7, (0,0,0), 1)
                cv2.putText(img, NAMES[k], (kx+6,ky-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[k], 1, cv2.LINE_AA)
            # Image bottom line
            cv2.line(img, (0,H-1),(W,H-1),(255,255,255),1)
        cv2.putText(img, f"{label} {os.path.basename(video_path)} f{fr}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{label} {os.path.basename(video_path)} f{fr}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
        # resize to a uniform panel height
        target_h = 600
        scale = target_h / H
        panels.append(cv2.resize(img, (int(W*scale), target_h)))
    if not panels:
        return False
    # pad to same width
    max_w = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 0, 0, max_w - p.shape[1], cv2.BORDER_CONSTANT, value=0) for p in panels]
    composite = np.vstack(panels)
    cv2.imwrite(save_path, composite)
    return True

DATASETS = [
    ("/root/data/auto-gait/poses_sapiens03b_goliath", "/root/data/auto-gait/videos", "auto-gait"),
    ("/root/data/gavd/poses_sapiens03b_goliath", "/root/data/gavd/videos", "gavd"),
    ("/root/data/koa-pd-nm-gait/poses_sapiens03b_goliath", "/root/data/koa-pd-nm-gait/videos", "koa-pd-nm-gait"),
    ("/root/data/tulip/gait/poses_sapiens03b_goliath", "/root/data/tulip/gait_videos", "tulip/gait"),
    ("/root/data/tulip/stability/poses_sapiens03b_goliath", "/root/data/tulip/stability_videos", "tulip/stability"),
    ("/root/data/tulip/spontaneity/poses_sapiens03b_goliath", "/root/data/tulip/spontaneity_videos", "tulip/spontaneity"),
]

OUT = "/root/overlays"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

for parquet_dir, videos_dir, label in DATASETS:
    files = sorted(glob.glob(parquet_dir + "/*.parquet"))
    if not files:
        print(f"!! no parquets at {parquet_dir}"); continue
    # pick 4 videos: 2 random + first + last for variety
    idxs = sorted({0, len(files)//3, 2*len(files)//3, len(files)-1})
    chosen_files = [files[i] for i in idxs][:4]
    for pf in chosen_files:
        df = pd.read_parquet(pf, columns=NEED)
        if df.empty: continue
        vid_name = df["video"].iloc[0]
        vpath = os.path.join(videos_dir, vid_name)
        if not os.path.exists(vpath):
            print(f"!! missing video: {vpath}"); continue
        det = df[df["det_score"].notna()]
        if det.empty: continue
        fmin, fmax = int(det["frame"].min()), int(det["frame"].max())
        frames = np.linspace(fmin, fmax, 4, dtype=int)
        safe_label = label.replace("/", "-")
        out_name = f"{safe_label}__{os.path.splitext(vid_name)[0]}.jpg"
        save_path = os.path.join(OUT, out_name)
        ok = render_one(vpath, df, frames, save_path, label)
        print(f"  {'OK' if ok else 'FAIL'}  {save_path}")
