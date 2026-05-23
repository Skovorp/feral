"""Draw patient-only lower-body overlay (gait datasets).

For each video: track all persons by greedy NN, pick the highest-travel track as the
patient, render 4 evenly spaced frames stacked vertically with:
  - patient bbox in green if lower-body visible (in_bbox + anat), else red
  - other tracks in dim gray
  - lower-body kpts of patient as colored dots
"""
import os, glob
from collections import defaultdict
import cv2, json, subprocess
import pandas as pd
import numpy as np

LB = [11, 12, 13, 14, 15, 16]
NAMES = {11:"Lh",12:"Rh",13:"Lk",14:"Rk",15:"La",16:"Ra"}
COLOR = {11:(255,128,0),12:(255,128,0),13:(0,255,255),14:(0,255,255),15:(0,255,0),16:(0,255,0)}
NEED = ["video","frame","det_score","bbox_x1","bbox_y1","bbox_x2","bbox_y2"]
for k in LB: NEED += [f"k{k}_x", f"k{k}_y"]
MAX_FRAME_GAP = 30

def img_dims(video_path):
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=width,height","-of","json", video_path
        ], timeout=10)
        s = json.loads(out)["streams"][0]
        return int(s["width"]), int(s["height"])
    except Exception:
        return None, None

def track_persons(df, max_match_pix):
    df = df.sort_values(["frame"]).reset_index(drop=False)
    df["xc"] = (df["bbox_x1"] + df["bbox_x2"]) / 2
    df["yc"] = (df["bbox_y1"] + df["bbox_y2"]) / 2
    tracks = []
    for fid, fdf in df.groupby("frame", sort=True):
        local = fdf.reset_index(drop=False)
        for li in range(len(local)):
            row = local.iloc[li]
            best_t = None; best_d = float("inf")
            for t in tracks:
                if fid - t["last_frame"] > MAX_FRAME_GAP: continue
                d = ((row["xc"] - t["last_xc"])**2 + (row["yc"] - t["last_yc"])**2) ** 0.5
                if d < best_d and d <= max_match_pix:
                    best_d = d; best_t = t
            if best_t is None:
                tracks.append({"last_xc":row["xc"], "last_yc":row["yc"], "last_frame":fid,
                               "row_idxs":[int(row["index"])], "xcs":[row["xc"]]})
            else:
                best_t["last_xc"]=row["xc"]; best_t["last_yc"]=row["yc"]
                best_t["last_frame"]=fid
                best_t["row_idxs"].append(int(row["index"]))
                best_t["xcs"].append(row["xc"])
    return tracks

def kpt_in_bbox_row(r):
    for k in LB:
        x, y = r[f"k{k}_x"], r[f"k{k}_y"]
        if not (r["bbox_x1"] < x < r["bbox_x2"] and r["bbox_y1"] < y < r["bbox_y2"]):
            return False
    return True

def anat_ok_row(r):
    return (r["k11_y"] < r["k13_y"] < r["k15_y"]) and (r["k12_y"] < r["k14_y"] < r["k16_y"])

def read_frame(video_path, idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, f = cap.read()
    cap.release()
    return f if ok else None

def render_video(vpath, df_all, patient_idx_set, frames, label, out_path):
    panels = []
    H_total = 600
    for fr in frames:
        img = read_frame(vpath, fr)
        if img is None: continue
        Hi, Wi = img.shape[:2]
        for _, r in df_all[df_all["frame"] == fr].iterrows():
            is_patient = int(r.name) in patient_idx_set
            x1,y1,x2,y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
            if not is_patient:
                cv2.rectangle(img, (x1,y1),(x2,y2), (140,140,140), 1)
                cv2.putText(img, "doctor?", (x1+4, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140,140,140), 1, cv2.LINE_AA)
                continue
            in_bb = kpt_in_bbox_row(r)
            anat = anat_ok_row(r)
            color = (0,255,0) if (in_bb and anat) else (0,0,255)
            cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
            for k in LB:
                kx, ky = int(r[f"k{k}_x"]), int(r[f"k{k}_y"])
                cv2.circle(img, (kx,ky), 6, COLOR[k], -1)
                cv2.circle(img, (kx,ky), 7, (0,0,0), 1)
                cv2.putText(img, NAMES[k], (kx+6,ky-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[k], 1, cv2.LINE_AA)
        cv2.line(img, (0,Hi-1),(Wi,Hi-1),(255,255,255),1)
        title = f"{label}  {os.path.basename(vpath)}  f{fr}"
        cv2.putText(img, title, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, title, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        scale = H_total / Hi
        panels.append(cv2.resize(img, (int(Wi*scale), H_total)))
    if not panels: return False
    max_w = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 0, 0, max_w - p.shape[1], cv2.BORDER_CONSTANT, value=0) for p in panels]
    composite = np.vstack(panels)
    cv2.imwrite(out_path, composite)
    return True

DATASETS = [
    ("/root/data/auto-gait/poses_sapiens03b_goliath", "/root/data/auto-gait/videos", "auto-gait"),
    ("/root/data/koa-pd-nm-gait/poses_sapiens03b_goliath", "/root/data/koa-pd-nm-gait/videos", "koa-pd-nm-gait"),
    ("/root/data/tulip/gait/poses_sapiens03b_goliath", "/root/data/tulip/gait_videos", "tulip/gait"),
]

OUT = "/root/overlays_patient"
os.makedirs(OUT, exist_ok=True)

for parquet_dir, videos_dir, label in DATASETS:
    files = sorted(glob.glob(parquet_dir + "/*.parquet"))
    # pick 6 videos spread across the dataset
    if len(files) <= 6:
        chosen = files
    else:
        step = len(files) // 6
        chosen = [files[i*step] for i in range(6)]
    for pf in chosen:
        try:
            df = pd.read_parquet(pf, columns=NEED)
        except Exception as e:
            print(f"  skip {pf}: {e}"); continue
        if df.empty: continue
        vid_name = df["video"].iloc[0]
        vpath = os.path.join(videos_dir, vid_name)
        if not os.path.exists(vpath):
            print(f"  missing video: {vpath}"); continue
        W, H = img_dims(vpath)
        max_match = (W * 0.25) if W else 200
        tracks = track_persons(df, max_match)
        if not tracks: continue
        scored = sorted(tracks, key=lambda t: -float(np.abs(np.diff(t["xcs"])).sum() if len(t["xcs"])>1 else 0))
        patient_track = scored[0]
        patient_idx_set = set(patient_track["row_idxs"])
        det = df[df["det_score"].notna()]
        if det.empty: continue
        fmin, fmax = int(det["frame"].min()), int(det["frame"].max())
        frames = np.linspace(fmin, fmax, 4, dtype=int)
        safe_label = label.replace("/", "-")
        out_name = f"{safe_label}__{os.path.splitext(vid_name)[0]}.jpg"
        save_path = os.path.join(OUT, out_name)
        ok = render_video(vpath, df, patient_idx_set, frames, label, save_path)
        print(f"  {'OK' if ok else 'FAIL'}  {save_path}  (tracks={len(tracks)})")
