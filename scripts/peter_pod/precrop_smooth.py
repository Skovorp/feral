"""Per-frame smooth-pan square pre-crop, 1024x1024, FERAL codec.

For each video:
  per-frame raw  cx, cy, side  from the +50%-padded large_bbox (max(bw, bh)).
  side capped to min(W, H) so the square always fits inside the source image
    (no black padding).
  cx, cy clamped to keep the square inside the source image.
  Entry/exit phases (patient_visible=False) hold the crop at the first/last
    visible value so the camera is fixed while the patient walks in/out.
  (cx, cy, side) gaussian-smoothed (sigma=8 frames ~ 250ms @ 30fps) so the
    output looks like a smooth pan, not a jittery rectangle.

Output:
  /root/data_precrop/<safe_dataset>/<video>  -- 1024x1024 mp4, libx264 CRF 25,
    yuv420p, no audio.
  /root/bboxes/unified_gait_bboxes_precrop.json  -- only small_bboxes in the
    new 1024x1024 canvas coords (clipped to [0,1024]); large is implicitly the
    full frame. Includes per-frame crop params for traceability.
  /root/labels/unified_gait_labels_precrop.json -- mirror of unified labels
    with video_roots updated to point to /root/data_precrop.
"""
from __future__ import annotations
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

UNIFIED_BBOXES = "/root/bboxes/unified_gait_bboxes.json"
UNIFIED_LABELS = "/root/labels/unified_gait_labels.json"
OUT_BBOXES     = "/root/bboxes/unified_gait_bboxes_precrop.json"
OUT_LABELS     = "/root/labels/unified_gait_labels_precrop.json"
OUT_DIR        = "/root/data_precrop"
NUM_WORKERS    = 6
OUTPUT_SIZE    = 1024     # square canvas
SIGMA_FRAMES   = 8.0      # smoothing sigma (~250ms at 30fps)
CRF            = 25

def smoothed_crop_params(large: np.ndarray, vis: np.ndarray, W: int, H: int):
    """Return (cx, cy, side) arrays of length n_frames, all-inside-image."""
    n = len(large)
    bw = large[:, 2] - large[:, 0]
    bh = large[:, 3] - large[:, 1]
    cx = (large[:, 0] + large[:, 2]) * 0.5
    cy = (large[:, 1] + large[:, 3]) * 0.5
    side = np.maximum(bw, bh).astype(np.float64)

    if vis.any():
        first = int(np.argmax(vis))
        last  = n - 1 - int(np.argmax(vis[::-1]))
        # Hold crop constant during entry/exit so camera is static and the
        # patient walks into / out of frame.
        cx[:first]    = cx[first]
        cy[:first]    = cy[first]
        side[:first]  = side[first]
        cx[last + 1:] = cx[last]
        cy[last + 1:] = cy[last]
        side[last + 1:] = side[last]
    else:
        cx[:] = W * 0.5
        cy[:] = H * 0.5
        side[:] = min(W, H)

    cx_s   = gaussian_filter1d(cx,   sigma=SIGMA_FRAMES)
    cy_s   = gaussian_filter1d(cy,   sigma=SIGMA_FRAMES)
    side_s = gaussian_filter1d(side, sigma=SIGMA_FRAMES)

    # Cap side so the square fits inside the source image, then clamp centre.
    side_s = np.minimum(side_s, float(min(W, H)))
    half   = side_s * 0.5
    cx_s   = np.clip(cx_s, half, W - half)
    cy_s   = np.clip(cy_s, half, H - half)
    return cx_s, cy_s, side_s

def open_ffmpeg_writer(dst: str, fps: float):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{OUTPUT_SIZE}x{OUTPUT_SIZE}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-crf", str(CRF), "-preset", "fast",
        "-pix_fmt", "yuv420p",
        dst,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10 * 1024 * 1024)

def process_one(key: str, payload: dict, src_path: str, dst_path: str):
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        # Already done; still need to compute new bbox payload for the json.
        pass
    W, H = int(payload["img_w"]), int(payload["img_h"])
    n_frames = int(payload["n_frames"])
    large = np.asarray(payload["large_bboxes"], dtype=np.float32)
    small = np.asarray(payload["small_bboxes"], dtype=np.float32)
    vis   = np.asarray(payload["patient_visible"], dtype=bool)
    fps = float(payload.get("fps", 30.0))

    cx_s, cy_s, side_s = smoothed_crop_params(large, vis, W, H)
    crop_origin_x = (cx_s - side_s * 0.5).astype(np.float32)
    crop_origin_y = (cy_s - side_s * 0.5).astype(np.float32)

    # Build new small_bbox payload in the 1024x1024 canvas coords.
    scale = OUTPUT_SIZE / side_s
    new_small = np.empty_like(small)
    new_small[:, 0] = (small[:, 0] - crop_origin_x) * scale
    new_small[:, 1] = (small[:, 1] - crop_origin_y) * scale
    new_small[:, 2] = (small[:, 2] - crop_origin_x) * scale
    new_small[:, 3] = (small[:, 3] - crop_origin_y) * scale
    np.clip(new_small, 0, OUTPUT_SIZE, out=new_small)

    new_payload = {
        "img_w": OUTPUT_SIZE,
        "img_h": OUTPUT_SIZE,
        "fps": fps,
        "n_frames": n_frames,
        "small_bboxes": new_small.tolist(),
        "patient_visible": payload["patient_visible"],
        "crop_origin_x": crop_origin_x.tolist(),
        "crop_origin_y": crop_origin_y.tolist(),
        "crop_side":     side_s.tolist(),
        "orig_img_w": W, "orig_img_h": H,
        "source_video": src_path,
    }

    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return key, new_payload

    # Read source video, do per-frame crop, write to ffmpeg.
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return key, None
    writer = open_ffmpeg_writer(dst_path, fps)
    try:
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                # ran out of frames; pad with a black frame so writer stays in sync
                frame = np.zeros((H, W, 3), dtype=np.uint8)
            x0 = int(round(crop_origin_x[i]))
            y0 = int(round(crop_origin_y[i]))
            s  = int(round(side_s[i]))
            x0 = max(0, min(W - s, x0))
            y0 = max(0, min(H - s, y0))
            crop = frame[y0:y0 + s, x0:x0 + s]
            if crop.shape[0] != s or crop.shape[1] != s:
                continue  # source frame mismatch; skip
            out = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
            writer.stdin.write(out.tobytes())
    finally:
        cap.release()
        writer.stdin.close()
        writer.wait()
    return key, new_payload

def main():
    with open(UNIFIED_BBOXES) as fp: bboxes = json.load(fp)
    with open(UNIFIED_LABELS) as fp: labels = json.load(fp)
    video_roots = labels["video_roots"]

    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = []
    for key, payload in bboxes.items():
        head, _, video = key.rpartition("/")
        src = os.path.join(video_roots[head], video)
        if not os.path.exists(src):
            print(f"  missing source: {src}"); continue
        safe_head = head.replace("/", "-")
        dst = os.path.join(OUT_DIR, safe_head, video)
        tasks.append((key, payload, src, dst))

    print(f"pre-cropping {len(tasks)} videos @ {OUTPUT_SIZE}x{OUTPUT_SIZE}, "
          f"sigma={SIGMA_FRAMES}, CRF={CRF}, {NUM_WORKERS} workers")
    new_bboxes = {}; failed = []
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
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                print(f"  [{i+1}/{len(tasks)}] ok={len(new_bboxes)} failed={len(failed)}")

    with open(OUT_BBOXES, "w") as fp:
        json.dump(new_bboxes, fp)
    print(f"\nwrote {OUT_BBOXES}  ({len(new_bboxes)} videos)")

    new_labels = dict(labels)
    new_labels["video_roots"] = {
        head: os.path.join(OUT_DIR, head.replace("/", "-"))
        for head in labels["video_roots"]
    }
    have = set(new_bboxes.keys())
    new_labels["labels"] = {k: v for k, v in labels["labels"].items() if k in have}
    new_labels["splits"] = {s: [k for k in ks if k in have] for s, ks in labels["splits"].items()}
    with open(OUT_LABELS, "w") as fp:
        json.dump(new_labels, fp, indent=2)
    print(f"wrote {OUT_LABELS}")
    if failed:
        print(f"failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")

if __name__ == "__main__":
    main()
