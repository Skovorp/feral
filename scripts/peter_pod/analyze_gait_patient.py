"""Gait-only analysis with patient/doctor disambiguation.

For each video:
  1. Greedy-track persons frame-to-frame by bbox-centre nearest-neighbor.
  2. Score each track by total horizontal travel of its bbox-centre.
  3. Pick the highest-travel track == "the moving patient".
  4. Compute lower-body visibility metrics on that track only.

Datasets: auto-gait, koa-pd-nm-gait, tulip/gait.
"""
import glob, json, os, subprocess
from collections import defaultdict
import pandas as pd
import numpy as np

LB = [11, 12, 13, 14, 15, 16]
NEED = ["video","frame","det_score","bbox_x1","bbox_y1","bbox_x2","bbox_y2"]
for k in LB: NEED += [f"k{k}_x", f"k{k}_y"]

MAX_FRAME_GAP = 30     # track allowed to skip this many sampled frames before being closed
MAX_MATCH_PIX = None   # set per-video as 25% of frame width

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
    """Greedy NN tracking. Returns: list of tracks, each is a list of df row indices."""
    df = df.sort_values(["frame"]).reset_index(drop=False)  # keep original index in 'index'
    df["xc"] = (df["bbox_x1"] + df["bbox_x2"]) / 2
    df["yc"] = (df["bbox_y1"] + df["bbox_y2"]) / 2

    tracks = []  # {last_xc, last_yc, last_frame, row_idxs:[orig idx], frames:[frame], xcs:[xc]}
    for frame_id, fdf in df.groupby("frame", sort=True):
        # match each person in this frame to the closest active track
        unmatched = list(range(len(fdf)))
        # iterate sorted by row order; greedy
        local = fdf.reset_index(drop=False)  # cols: index(of df), original 'index', xc, yc, ...
        for local_i in list(unmatched):
            row = local.iloc[local_i]
            best_t = None; best_d = float("inf")
            for t in tracks:
                if frame_id - t["last_frame"] > MAX_FRAME_GAP:
                    continue
                d = ((row["xc"] - t["last_xc"])**2 + (row["yc"] - t["last_yc"])**2) ** 0.5
                if d < best_d and d <= max_match_pix:
                    best_d = d; best_t = t
            if best_t is None:
                tracks.append({
                    "last_xc": row["xc"], "last_yc": row["yc"], "last_frame": frame_id,
                    "row_idxs": [int(row["index"])], "frames": [frame_id], "xcs": [row["xc"]],
                })
            else:
                best_t["last_xc"] = row["xc"]
                best_t["last_yc"] = row["yc"]
                best_t["last_frame"] = frame_id
                best_t["row_idxs"].append(int(row["index"]))
                best_t["frames"].append(frame_id)
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

def analyze_dataset(parquet_dir, videos_dir, label):
    files = sorted(glob.glob(parquet_dir + "/*.parquet"))
    n_videos = len(files)
    tot_frames = 0
    patient_det_frames = 0
    patient_lb_in_bbox = 0
    patient_lb_anat = 0
    patient_lb_anat_unclipped = 0
    n_clipped_videos = 0
    n_two_person_videos = 0
    travel_ratios = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=NEED)
        except Exception as e:
            continue
        if df.empty: continue
        vid_name = df["video"].iloc[0]
        vpath = os.path.join(videos_dir, vid_name)
        W, H = img_dims(vpath)
        max_match = (W * 0.25) if W else 200
        frames_total = int(df["frame"].max()) + 1
        tot_frames += frames_total

        tracks = track_persons(df, max_match)
        if not tracks: continue

        # score tracks by total horizontal travel
        scored = []
        for t in tracks:
            if len(t["xcs"]) < 2:
                travel = 0
            else:
                travel = float(np.abs(np.diff(t["xcs"])).sum())
            scored.append((travel, t))
        scored.sort(key=lambda x: -x[0])
        patient_track = scored[0][1]
        if len(scored) > 1 and scored[1][0] > 0:
            n_two_person_videos += 1
            ratio = scored[1][0] / max(scored[0][0], 1)
            travel_ratios.append(ratio)

        patient_rows = df.loc[patient_track["row_idxs"]]
        det = patient_rows[patient_rows["det_score"].notna()]
        if det.empty: continue
        patient_det_frames += det["frame"].nunique()

        # per-row checks
        in_bbox_mask = det.apply(kpt_in_bbox_row, axis=1)
        anat_mask = det.apply(anat_ok_row, axis=1)
        clip_ok = (det["bbox_y2"] < (H - 4)) if H is not None else pd.Series(True, index=det.index)

        patient_lb_in_bbox += det.loc[in_bbox_mask, "frame"].nunique()
        patient_lb_anat += det.loc[in_bbox_mask & anat_mask, "frame"].nunique()
        patient_lb_anat_unclipped += det.loc[in_bbox_mask & anat_mask & clip_ok, "frame"].nunique()
        if H is not None and (~clip_ok).any():
            n_clipped_videos += 1
    return dict(
        label=label, n_videos=n_videos, tot_frames=tot_frames,
        patient_det_frames=patient_det_frames,
        patient_lb_in_bbox=patient_lb_in_bbox,
        patient_lb_anat=patient_lb_anat,
        patient_lb_anat_unclipped=patient_lb_anat_unclipped,
        n_clipped_videos=n_clipped_videos,
        n_two_person_videos=n_two_person_videos,
        median_travel_ratio=(float(np.median(travel_ratios)) if travel_ratios else None),
    )

DATASETS = [
    ("/root/data/auto-gait/poses_sapiens03b_goliath", "/root/data/auto-gait/videos", "auto-gait"),
    ("/root/data/koa-pd-nm-gait/poses_sapiens03b_goliath", "/root/data/koa-pd-nm-gait/videos", "koa-pd-nm-gait"),
    ("/root/data/tulip/gait/poses_sapiens03b_goliath", "/root/data/tulip/gait_videos", "tulip/gait"),
]

def pct(a,b): return f"{a/max(b,1)*100:5.1f}%"

print("Patient-only lower-body visibility (gait datasets).")
print("  Tracking: greedy NN by bbox-centre; max-match = 25% of frame width.")
print("  Patient track: largest total horizontal travel per video.")
print("  All % are of TOTAL frames in dataset.")
print()
hdr = ("dataset","videos","total_f","patient_det","det%","+bbox","+anat","+unclipped",
       "2-person","2nd_track_travel/1st")
print("{:<18} {:>6} {:>9} {:>11} {:>6} {:>7} {:>7} {:>11} {:>10} {:>22}".format(*hdr))
print("-"*120)
for parquet_dir, videos_dir, lab in DATASETS:
    r = analyze_dataset(parquet_dir, videos_dir, lab)
    if r is None: continue
    mtr = "n/a" if r["median_travel_ratio"] is None else f"{r['median_travel_ratio']:.3f}"
    print("{:<18} {:>6,} {:>9,} {:>11,} {:>6} {:>7} {:>7} {:>11} {:>5}/{:<4} {:>22}".format(
        r["label"], r["n_videos"], r["tot_frames"], r["patient_det_frames"],
        pct(r["patient_det_frames"], r["tot_frames"]),
        pct(r["patient_lb_in_bbox"], r["tot_frames"]),
        pct(r["patient_lb_anat"], r["tot_frames"]),
        pct(r["patient_lb_anat_unclipped"], r["tot_frames"]),
        r["n_two_person_videos"], r["n_videos"], mtr,
    ))
