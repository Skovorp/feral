"""Stricter lower-body visibility.

Adds two checks beyond kpts-inside-bbox:
  (A) anatomical ordering: for both legs, hip_y < knee_y < ankle_y (img y grows downward).
  (B) bbox bottom not at image bottom (subject's feet below frame -> bbox clipped).
      Image dims read from ffprobe per video.
"""
import glob, json, os, subprocess
import pandas as pd
import numpy as np

LB = [11, 12, 13, 14, 15, 16]
NEED = ["video","frame","det_score","bbox_x1","bbox_y1","bbox_x2","bbox_y2"]
for k in LB:
    NEED += [f"k{k}_x", f"k{k}_y"]

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

def kpt_in_bbox(df, ks):
    bx1=df["bbox_x1"].to_numpy(); bx2=df["bbox_x2"].to_numpy()
    by1=df["bbox_y1"].to_numpy(); by2=df["bbox_y2"].to_numpy()
    m = np.ones(len(df), dtype=bool)
    for k in ks:
        x=df[f"k{k}_x"].to_numpy(); y=df[f"k{k}_y"].to_numpy()
        m &= (x>bx1)&(x<bx2)&(y>by1)&(y<by2)
    return m

def anat_ok(df):
    h11 = df["k11_y"].to_numpy(); h12 = df["k12_y"].to_numpy()
    k13 = df["k13_y"].to_numpy(); k14 = df["k14_y"].to_numpy()
    a15 = df["k15_y"].to_numpy(); a16 = df["k16_y"].to_numpy()
    return (h11 < k13) & (k13 < a15) & (h12 < k14) & (k14 < a16)

def bbox_not_clipped(df, H, margin=4):
    return df["bbox_y2"].to_numpy() < (H - margin)

def stats(parquet_dir, videos_dir, label):
    files = sorted(glob.glob(parquet_dir + "/*.parquet"))
    if not files: return None
    tot=det=lb=lb_anat=lb_clip=0
    dims_cache = {}
    n_clipped_videos = 0; n_total_videos = 0
    for f in files:
        try:
            df = pd.read_parquet(f, columns=NEED)
        except Exception: continue
        if df.empty: continue
        n_total_videos += 1
        vid_name = df["video"].iloc[0]
        vpath = os.path.join(videos_dir, vid_name)
        if vpath not in dims_cache:
            dims_cache[vpath] = img_dims(vpath)
        W, H = dims_cache[vpath]
        tot += int(df["frame"].max())+1
        d = df[df["det_score"].notna()]
        if d.empty: continue
        det += d["frame"].nunique()
        in_bbox = kpt_in_bbox(d, LB)
        anat = anat_ok(d)
        lb += d.loc[in_bbox, "frame"].nunique()
        lb_anat += d.loc[in_bbox & anat, "frame"].nunique()
        if H is not None:
            clip = bbox_not_clipped(d, H)
            lb_clip += d.loc[in_bbox & anat & clip, "frame"].nunique()
            if (~clip).sum() > 0:
                n_clipped_videos += 1
    return label, tot, det, lb, lb_anat, lb_clip, n_clipped_videos, n_total_videos

DATASETS = [
    ("/root/data/auto-gait/poses_sapiens03b_goliath", "/root/data/auto-gait/videos", "auto-gait"),
    ("/root/data/gavd/poses_sapiens03b_goliath", "/root/data/gavd/videos", "gavd"),
    ("/root/data/koa-pd-nm-gait/poses_sapiens03b_goliath", "/root/data/koa-pd-nm-gait/videos", "koa-pd-nm-gait"),
    ("/root/data/tulip/gait/poses_sapiens03b_goliath", "/root/data/tulip/gait_videos", "tulip/gait"),
    ("/root/data/tulip/stability/poses_sapiens03b_goliath", "/root/data/tulip/stability_videos", "tulip/stability"),
    ("/root/data/tulip/spontaneity/poses_sapiens03b_goliath", "/root/data/tulip/spontaneity_videos", "tulip/spontaneity"),
]

def pct(a,b): return f"{a/max(b,1)*100:5.1f}%"

print("Stricter lower-body visibility (% of TOTAL frames in dataset).")
print("  lb_in_bbox       = original check (kpts project inside bbox)")
print("  +anat_order      = AND hip_y < knee_y < ankle_y for both legs (rejects junk lower-body kpts)")
print("  +bbox_unclipped  = AND bbox bottom is >4px above image bottom (subject feet really in frame)")
print()
hdr = ("dataset","total_f","det_f","lb_in_bbox","+anat","+unclipped","clipped_vids")
print("{:<20} {:>9} {:>9} {:>11} {:>9} {:>11} {:>14}".format(*hdr))
print("-"*92)
for pd_dir, vid_dir, lab in DATASETS:
    r = stats(pd_dir, vid_dir, lab)
    if r is None: continue
    label, tot, det, lb, lba, lbc, nclip, nv = r
    print("{:<20} {:>9,} {:>9,} {:>11} {:>9} {:>11} {:>10}/{}".format(
        label, tot, det, pct(lb,tot), pct(lba,tot), pct(lbc,tot), nclip, nv))
