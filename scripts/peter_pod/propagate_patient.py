"""Propagate the user-labeled patient bbox through every frame of each gait
video, then write a per-dataset bboxes.json in the format gait_aug.py expects.

Input:
  /root/selections_labeled.json   { "dataset|video|bout_idx": {kind, row_idx_in_parquet, ...} }

For each entry with kind=="patient":
  1. Open the corresponding per-video parquet.
  2. Read the seed row (the user-picked bbox on the middle frame).
  3. Greedy NN bbox-centre tracking forward and backward through every frame.
     The parquet is dense (one row per frame per detection, including stride-2
     interpolated rows with det_score=NaN).
  4. Patient frame = the row whose bbox-centre is closest to the previous
     patient bbox-centre (Euclidean distance). No gating - if no detection on a
     frame, patient_visible[frame] is False and the bbox is later linearly
     interpolated from neighbors.

Entries with kind!="patient" are skipped (the user said to drop them).

Output: /root/bboxes_<dataset_safe>.json with one entry per video,
    matching the format documented at the top of gait_aug.py.
"""
from __future__ import annotations
import json, os, sys, glob, subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "/root")
from gait_aug import compute_bboxes_for_video  # noqa: E402

SELECTIONS_PATH = "/root/selections_labeled.json"

DATASETS = {
    # name: (parquet_dir, videos_dir, entry_window_ms)
    "auto-gait":      ("/root/data/auto-gait/poses_sapiens03b_goliath",      "/root/data/auto-gait/videos",      1000.0),
    "koa-pd-nm-gait": ("/root/data/koa-pd-nm-gait/poses_sapiens03b_goliath", "/root/data/koa-pd-nm-gait/videos",  300.0),
    "tulip/gait":     ("/root/data/tulip/gait/poses_sapiens03b_goliath",     "/root/data/tulip/gait_videos",     1000.0),
}

LB = [11, 12, 13, 14, 15, 16]
KPT_COLS = sum([[f"k{k}_x", f"k{k}_y"] for k in LB], [])
BASE_COLS = ["video","frame","det_score","bbox_x1","bbox_y1","bbox_x2","bbox_y2"]

def video_info(path):
    """Return (width, height, fps, n_frames). Header-only (no frame count walk)."""
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=width,height,r_frame_rate,nb_frames",
            "-of","json", path
        ], timeout=10)
        s = json.loads(out)["streams"][0]
        n_num, n_den = s["r_frame_rate"].split("/")
        fps = float(n_num) / float(n_den) if float(n_den) != 0 else 30.0
        nb = s.get("nb_frames")
        nf = int(nb) if (nb and nb != "N/A") else 0  # 0 -> caller falls back to parquet
        return int(s["width"]), int(s["height"]), fps, nf
    except Exception:
        return None, None, None, 0

def track_patient(df: pd.DataFrame, seed_row_idx: int) -> pd.DataFrame:
    """Greedy NN tracking forward+backward from a seed row.

    df: per-video parquet (dense, all frames have at least one row).
    seed_row_idx: pandas index of the seed row (i.e. df.loc[seed_row_idx]).

    Returns a DataFrame of the patient track, one row per frame the patient
    was tracked through. Order: ascending by frame.
    """
    df = df.copy()
    df["xc"] = (df["bbox_x1"] + df["bbox_x2"]) / 2
    df["yc"] = (df["bbox_y1"] + df["bbox_y2"]) / 2
    seed = df.loc[seed_row_idx]
    seed_frame = int(seed["frame"])
    last_xc, last_yc = float(seed["xc"]), float(seed["yc"])

    # group by frame for O(1) lookup
    by_frame = {int(f): grp for f, grp in df.groupby("frame", sort=True)}
    all_frames = sorted(by_frame.keys())
    if not all_frames: return df.iloc[0:0]

    chosen_idx = {seed_frame: seed_row_idx}

    # forward
    last_xc_f, last_yc_f = last_xc, last_yc
    for f in [x for x in all_frames if x > seed_frame]:
        grp = by_frame[f]
        d = (grp["xc"] - last_xc_f)**2 + (grp["yc"] - last_yc_f)**2
        pick = grp.loc[d.idxmin()]
        chosen_idx[f] = pick.name
        last_xc_f, last_yc_f = float(pick["xc"]), float(pick["yc"])

    # backward
    last_xc_b, last_yc_b = last_xc, last_yc
    for f in [x for x in reversed(all_frames) if x < seed_frame]:
        grp = by_frame[f]
        d = (grp["xc"] - last_xc_b)**2 + (grp["yc"] - last_yc_b)**2
        pick = grp.loc[d.idxmin()]
        chosen_idx[f] = pick.name
        last_xc_b, last_yc_b = float(pick["xc"]), float(pick["yc"])

    rows = df.loc[[chosen_idx[f] for f in sorted(chosen_idx)]]
    return rows

def main():
    with open(SELECTIONS_PATH) as fp:
        sel = json.load(fp)

    out_per_dataset = defaultdict(dict)
    skipped, ok = 0, 0
    summary = []

    for key, info in sel.items():
        dataset, video, bout_idx = key.split("|")
        bout_idx = int(bout_idx)
        if info["kind"] != "patient":
            skipped += 1
            summary.append((dataset, video, bout_idx, "skipped:" + info["kind"], None, None, None))
            continue
        if dataset not in DATASETS:
            print(f"  unknown dataset: {dataset} for key {key}")
            continue
        parquet_dir, videos_dir, entry_window_ms = DATASETS[dataset]

        # the parquet file might be named e.g. "<video>.parquet" or "<basename(.mp4)>.parquet"
        # render_middle_frames used os.path.splitext(vid)[0] + ".jpg"; the actual parquet
        # files we saw were e.g. "0.mp4.parquet". So path = videos_dir/<video>, parquet = parquet_dir/<video>.parquet
        parquet_path = os.path.join(parquet_dir, video + ".parquet")
        if not os.path.exists(parquet_path):
            # try without the .mp4 piece
            alt = os.path.join(parquet_dir, os.path.splitext(video)[0] + ".parquet")
            if os.path.exists(alt):
                parquet_path = alt
            else:
                print(f"  MISSING parquet: tried {parquet_path}")
                continue

        video_path = os.path.join(videos_dir, video)
        if not os.path.exists(video_path):
            print(f"  MISSING video: {video_path}")
            continue

        W, H, fps, nf_probe = video_info(video_path)
        df = pd.read_parquet(parquet_path, columns=BASE_COLS + KPT_COLS)
        if df.empty:
            print(f"  empty parquet: {parquet_path}")
            continue

        # Use the larger of (probed frame count, parquet max frame + 1) as the truth.
        n_frames = max((nf_probe or 0), int(df["frame"].max()) + 1)

        seed_row_idx = info["row_idx_in_parquet"]
        if seed_row_idx not in df.index:
            print(f"  seed row {seed_row_idx} missing in {parquet_path}")
            continue

        track = track_patient(df, seed_row_idx)
        # build bbox payload (gait_aug.compute_bboxes_for_video)
        payload = compute_bboxes_for_video(
            patient_rows=track,
            n_frames=n_frames, img_w=W or 1280, img_h=H or 720,
            fps=float(fps or 30.0),
            entry_window_ms=entry_window_ms,
        )

        # If the user had a multi-bout video we currently overwrite per video — that's
        # fine for our gait datasets since only 3 koa videos had multi-bouts and the
        # patient ID stays consistent across bouts. We just keep the bbox arrays once
        # per video (whichever bout came last).
        out_per_dataset[dataset][video] = payload
        ok += 1
        n_det = int(track["det_score"].notna().sum())
        n_in_frame = int(track["frame"].between(0, n_frames - 1).sum())
        summary.append((dataset, video, bout_idx, "ok",
                        len(track), n_det, n_frames))

    # Write per-dataset files
    out_dir = "/root/bboxes"
    os.makedirs(out_dir, exist_ok=True)
    for dataset, vids in out_per_dataset.items():
        safe = dataset.replace("/", "-")
        out_path = os.path.join(out_dir, f"bboxes_{safe}.json")
        with open(out_path, "w") as fp:
            json.dump(vids, fp)
        print(f"wrote {out_path}  ({len(vids)} videos)")

    # Manifest of per-video stats
    stats_path = os.path.join(out_dir, "track_stats.json")
    with open(stats_path, "w") as fp:
        json.dump([
            {"dataset": d, "video": v, "bout_idx": b, "status": st,
             "track_len": tl, "n_det_rows": nd, "n_frames": nf}
            for (d,v,b,st,tl,nd,nf) in summary
        ], fp, indent=2)
    print(f"wrote {stats_path}")
    print(f"OK: {ok}  skipped: {skipped}  total: {len(sel)}")

if __name__ == "__main__":
    main()
