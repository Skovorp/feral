"""Pre-crop each video to the union of its per-frame large_bboxes and re-encode
in the FERAL convention (libx264 CRF 25 yuv420p, no audio, smallest side <= 570).

After this:
  - Pre-cropped videos at /root/data_precrop/<dataset>/<video>.
  - unified_gait_bboxes_precrop.json keeps only small_bboxes (in pre-crop +
    post-scale coords) plus img_w, img_h, fps, n_frames, patient_visible,
    crop_offset and orig dims for traceability. large_bboxes are dropped
    (the pre-crop IS the large region; the model lerps small <-> full frame).
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

UNIFIED_BBOXES = "/root/bboxes/unified_gait_bboxes.json"
UNIFIED_LABELS = "/root/labels/unified_gait_labels.json"
OUT_BBOXES     = "/root/bboxes/unified_gait_bboxes_precrop.json"
OUT_LABELS     = "/root/labels/unified_gait_labels_precrop.json"
OUT_DIR        = "/root/data_precrop"
NUM_WORKERS    = 6
SMALLEST_SIDE  = 570
CRF            = 25

def union_large(large_bboxes: np.ndarray, visible: np.ndarray, W: int, H: int):
    if visible.any():
        b = large_bboxes[visible]
    else:
        b = large_bboxes
    x1 = max(0, int(np.floor(b[:, 0].min())))
    y1 = max(0, int(np.floor(b[:, 1].min())))
    x2 = min(W, int(np.ceil (b[:, 2].max())))
    y2 = min(H, int(np.ceil (b[:, 3].max())))
    # ensure even W and H for libx264
    if (x2 - x1) % 2: x2 = x2 - 1 if x2 > x1 + 1 else min(W, x2 + 1)
    if (y2 - y1) % 2: y2 = y2 - 1 if y2 > y1 + 1 else min(H, y2 + 1)
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return 0, 0, (W // 2) * 2, (H // 2) * 2
    return x1, y1, x2, y2

def downscale_dims(cw: int, ch: int, smallest_side: int):
    """Return (out_w, out_h, scale) — out dims with smallest side <= smallest_side."""
    short = min(cw, ch)
    if short <= smallest_side:
        return cw, ch, 1.0
    s = smallest_side / short
    ow = int(round(cw * s)); oh = int(round(ch * s))
    ow -= ow % 2; oh -= oh % 2
    if ow == 0 or oh == 0:
        return cw, ch, 1.0
    return ow, oh, ow / cw   # treat scale uniformly using width ratio

def ffmpeg_crop_scale(src: str, dst: str, x: int, y: int, cw: int, ch: int,
                      sw: int, sh: int) -> bool:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    if (sw, sh) == (cw, ch):
        vf = f"crop={cw}:{ch}:{x}:{y}"
    else:
        vf = f"crop={cw}:{ch}:{x}:{y},scale={sw}:{sh}:flags=bicubic"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", src, "-an",
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(CRF), "-preset", "fast",
        "-pix_fmt", "yuv420p",
        dst,
    ]
    try:
        subprocess.check_call(cmd, timeout=1800)
        return True
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"  ffmpeg failed: {src}: {e}\n")
        return False
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"  ffmpeg timeout: {src}\n")
        return False

def process_one(key: str, payload: dict, src_path: str, dst_path: str):
    W, H = int(payload["img_w"]), int(payload["img_h"])
    large = np.asarray(payload["large_bboxes"], dtype=np.float32)
    small = np.asarray(payload["small_bboxes"], dtype=np.float32)
    vis   = np.asarray(payload["patient_visible"], dtype=bool)

    x1, y1, x2, y2 = union_large(large, vis, W, H)
    cw, ch = x2 - x1, y2 - y1
    sw, sh, scale = downscale_dims(cw, ch, SMALLEST_SIDE)
    if not ffmpeg_crop_scale(src_path, dst_path, x1, y1, cw, ch, sw, sh):
        return key, None

    # Translate then scale small bboxes; clip to final frame.
    small[:, [0, 2]] = (small[:, [0, 2]] - x1) * scale
    small[:, [1, 3]] = (small[:, [1, 3]] - y1) * scale
    small[:, [0, 2]] = np.clip(small[:, [0, 2]], 0, sw)
    small[:, [1, 3]] = np.clip(small[:, [1, 3]], 0, sh)

    new_payload = {
        "img_w": int(sw),
        "img_h": int(sh),
        "fps": payload.get("fps", 30.0),
        "n_frames": payload["n_frames"],
        "small_bboxes": small.tolist(),
        "patient_visible": payload["patient_visible"],
        "crop_offset": [int(x1), int(y1)],
        "crop_size": [int(cw), int(ch)],
        "scale": float(scale),
        "orig_img_w": W,
        "orig_img_h": H,
        "source_video": src_path,
    }
    return key, new_payload

def main():
    with open(UNIFIED_BBOXES) as fp: bboxes = json.load(fp)
    with open(UNIFIED_LABELS) as fp: labels = json.load(fp)
    video_roots = labels["video_roots"]

    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = []
    for key, payload in bboxes.items():
        head, _, video = key.rpartition("/")
        if head not in video_roots:
            print(f"  no video_root for {head!r}, skip {key}"); continue
        src = os.path.join(video_roots[head], video)
        if not os.path.exists(src):
            print(f"  missing source: {src}"); continue
        safe_head = head.replace("/", "-")
        dst = os.path.join(OUT_DIR, safe_head, video)
        tasks.append((key, payload, src, dst))

    print(f"pre-cropping {len(tasks)} videos with {NUM_WORKERS} workers, "
          f"CRF={CRF}, smallest_side={SMALLEST_SIDE}")
    new_bboxes = {}
    failed = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(process_one, *t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            try:
                k, payload = fut.result()
            except Exception as e:
                print(f"  EXC {futures[fut]}: {e}")
                failed.append(futures[fut]); continue
            if payload is None:
                failed.append(k); continue
            new_bboxes[k] = payload
            if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                print(f"  [{i+1}/{len(tasks)}] ok={len(new_bboxes)} failed={len(failed)}")

    with open(OUT_BBOXES, "w") as fp:
        json.dump(new_bboxes, fp)
    print(f"\nwrote {OUT_BBOXES}  ({len(new_bboxes)} videos)")

    # Write a parallel labels file pointing video_roots at the pre-crop dir.
    new_labels = dict(labels)
    new_labels["video_roots"] = {
        head: os.path.join(OUT_DIR, head.replace("/", "-"))
        for head in labels["video_roots"]
    }
    # Drop keys that didn't get a pre-cropped video.
    have = set(new_bboxes.keys())
    new_labels["labels"] = {k: v for k, v in labels["labels"].items() if k in have}
    new_labels["splits"] = {
        s: [k for k in keys if k in have] for s, keys in labels["splits"].items()
    }
    with open(OUT_LABELS, "w") as fp:
        json.dump(new_labels, fp, indent=2)
    print(f"wrote {OUT_LABELS}")

    if failed:
        print(f"failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")

if __name__ == "__main__":
    main()
