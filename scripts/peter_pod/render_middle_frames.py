"""Render the middle frame of EACH detection bout for every video in the 3 gait
datasets. A bout = a run of detected frames with consecutive gap <= 60 raw frames
(~2 sec @ 30fps). Draw numbered bbox overlays for every detected person on that
middle frame and write a manifest.json.
"""
import os, glob, json
import pandas as pd
import numpy as np
import cv2

GAP_THRESH = 60  # raw frames; ~2 sec at 30 fps. Split into separate bouts beyond this.

DATASETS = [
    ("/root/data/auto-gait/poses_sapiens03b_goliath", "/root/data/auto-gait/videos", "auto-gait"),
    ("/root/data/koa-pd-nm-gait/poses_sapiens03b_goliath", "/root/data/koa-pd-nm-gait/videos", "koa-pd-nm-gait"),
    ("/root/data/tulip/gait/poses_sapiens03b_goliath", "/root/data/tulip/gait_videos", "tulip/gait"),
]

OUT_DIR = "/root/patient_labeler"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

BOX_COLORS = [(0, 220, 220), (0, 200, 255), (200, 100, 255), (255, 200, 100),
              (180, 255, 100), (255, 150, 200), (255, 100, 100), (100, 200, 255)]

def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, f = cap.read()
    cap.release()
    return f if ok else None

def find_bouts(det_frames, gap):
    """Group sorted frame indices into runs where consecutive gap <= gap (raw frames)."""
    if not len(det_frames): return []
    arr = sorted(det_frames)
    runs = [[arr[0], arr[0]]]
    bout_frames = [[arr[0]]]
    for f in arr[1:]:
        if f - runs[-1][1] <= gap:
            runs[-1][1] = f
            bout_frames[-1].append(f)
        else:
            runs.append([f, f])
            bout_frames.append([f])
    return list(zip(runs, bout_frames))

manifest = []

for parquet_dir, videos_dir, label in DATASETS:
    safe = label.replace("/", "-")
    files = sorted(glob.glob(parquet_dir + "/*.parquet"))
    for f in files:
        df = pd.read_parquet(f, columns=["video","frame","det_score","bbox_x1","bbox_y1","bbox_x2","bbox_y2"])
        if df.empty: continue
        vid = df["video"].iloc[0]
        det = df[df["det_score"].notna()]
        if det.empty: continue
        det_frames = sorted(det["frame"].unique().tolist())
        bouts = find_bouts(det_frames, GAP_THRESH)
        vpath = os.path.join(videos_dir, vid)
        if not os.path.exists(vpath):
            print(f"  missing video: {vpath}"); continue
        for bout_idx, ((start_f, end_f), frames_in_bout) in enumerate(bouts):
            middle_frame = frames_in_bout[len(frames_in_bout) // 2]
            img = read_frame(vpath, middle_frame)
            if img is None:
                print(f"  cv2 failed: {vpath} @ {middle_frame}"); continue
            H, W = img.shape[:2]
            on_frame = det[det["frame"] == middle_frame].reset_index(drop=False)
            bboxes_meta = []
            for i, r in on_frame.iterrows():
                color = BOX_COLORS[i % len(BOX_COLORS)]
                x1,y1,x2,y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
                cv2.rectangle(img, (x1,y1),(x2,y2), color, 3)
                tag = str(i + 1)
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                badge_pt1 = (x1, max(0, y1 - th - 12))
                badge_pt2 = (x1 + tw + 14, y1)
                cv2.rectangle(img, badge_pt1, badge_pt2, color, -1)
                cv2.putText(img, tag, (x1 + 7, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
                bboxes_meta.append({
                    "id": i + 1,
                    "row_idx_in_parquet": int(r["index"]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
            n_bouts = len(bouts)
            bout_tag = f"bout {bout_idx+1}/{n_bouts}" if n_bouts > 1 else "bout 1/1"
            title = f"{label}   {vid}   {bout_tag}   frame {middle_frame}   bout=[{start_f}-{end_f}]   ({len(on_frame)} ppl)"
            cv2.putText(img, title, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(img, title, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1, cv2.LINE_AA)
            img_name = f"{safe}__{os.path.splitext(vid)[0]}__b{bout_idx+1}.jpg"
            img_path = os.path.join(IMG_DIR, img_name)
            max_dim = 1280
            scale = min(1.0, max_dim / max(H, W))
            if scale < 1.0:
                new_w, new_h = int(W*scale), int(H*scale)
                img_out = cv2.resize(img, (new_w, new_h))
                for b in bboxes_meta:
                    b["x1"] = int(b["x1"] * scale); b["y1"] = int(b["y1"] * scale)
                    b["x2"] = int(b["x2"] * scale); b["y2"] = int(b["y2"] * scale)
                disp_w, disp_h = new_w, new_h
            else:
                img_out = img
                disp_w, disp_h = W, H
            cv2.imwrite(img_path, img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            manifest.append({
                "dataset": label,
                "video": vid,
                "bout_idx": bout_idx,
                "bout_total": n_bouts,
                "bout_start_frame": int(start_f),
                "bout_end_frame": int(end_f),
                "middle_frame": int(middle_frame),
                "image": "images/" + img_name,
                "img_w": disp_w, "img_h": disp_h,
                "orig_w": W, "orig_h": H,
                "parquet": f,
                "video_path": vpath,
                "bboxes": bboxes_meta,
            })

manifest_path = os.path.join(OUT_DIR, "manifest.json")
with open(manifest_path, "w") as fp:
    json.dump(manifest, fp, indent=2)
print(f"wrote {len(manifest)} entries to {manifest_path}")
print(f"images at {IMG_DIR}")
