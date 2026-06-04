"""SAM3 'person' driven precrop + bg-replace masks for gait videos.

Replaces the (dead-pod) Sapiens smooth-pan precrop with a self-contained
SAM3-person pipeline so train/val framing is consistent AND we get the
per-frame person mask the committed BGReplacer needs.

For each NON-neg gait video in the labels file (train + val):
  1. read frames downscaled to MAXSIDE
  2. SAM3 video model, text prompt 'person'; per frame pick the LARGEST person
     mask (the patient dominates clinic gait clips; clinicians are smaller)
  3. per-frame person bbox -> hold over invisible frames -> square side
     = max(bw,bh)*PAD, clamped inside frame -> gaussian-smoothed (sigma=8)
     so the output is a smooth pan, not a jittery box
  4. crop each frame to that square, resize to OUT, pipe to ffmpeg
     (libx264 CRF23 yuv420p) -> /workspace/data_gaitcrop/<fn>
  5. crop the SAM mask the same way, resize to 256, pack bits ->
     /workspace/masks/<fn>.npz  (keys: packed, shape) -- exact format the
     committed feral.phone_aug.BGReplacer expects (mask in cropped-frame coords,
     aligns with the decoded 256 training frame)

Resumable: skips a video if BOTH its crop video and mask npz already exist.
Run in tmux; prints are flushed. fog_negs are NOT processed here (symlinked raw
by bootstrap -- they are full-frame 'other' negatives, no person-crop).
"""
import os, sys, json, glob, subprocess, traceback
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from transformers import Sam3VideoModel, Sam3VideoProcessor

LABELS   = os.environ.get("GAIT_LABELS", "/workspace/labels/feral_gait_labels_phoneaug_cleanval.json")
RAW_DIR  = os.environ.get("GAIT_RAW",   "/workspace/data_gaitraw")   # holds <fn> with same rel paths as labels
CROP_DIR = os.environ.get("GAIT_CROP",  "/workspace/data_gaitcrop")
MASK_DIR = os.environ.get("GAIT_MASKS", "/workspace/masks")
QA_DIR   = os.environ.get("GAIT_QA",    "/workspace/gait_qa")
MAXSIDE  = int(os.environ.get("SAM_MAXSIDE", "384"))
OUT      = int(os.environ.get("CROP_OUT", "384"))
MASK_RES = 256
PAD      = 1.30
SIGMA    = 8.0
dev = "cuda"

os.makedirs(CROP_DIR, exist_ok=True); os.makedirs(MASK_DIR, exist_ok=True); os.makedirs(QA_DIR, exist_ok=True)

print("loading SAM3 video ...", flush=True)
proc  = Sam3VideoProcessor.from_pretrained("facebook/sam3")
model = Sam3VideoModel.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(dev).eval()

lab = json.load(open(LABELS))
vids = [v for v in (lab["splits"]["train"] + lab["splits"]["val"]) if not v.startswith("fog_negs/")]
vids = sorted(set(vids))
print(f"gait videos to process (train+val, non-neg): {len(vids)}", flush=True)


def read_frames(path, maxside):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = maxside / max(h, w)
        if s < 1.0:
            rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    return frames, fps


def person_candidates(a):
    # a: (n_obj,H,W) float or (H,W). return list of (mask_bool, area, (cy,cx)).
    if a is None:
        return []
    if a.ndim == 2:
        a = a[None]
    out = []
    for k in range(a.shape[0]):
        m = a[k] > 0.5
        ar = int(m.sum())
        if ar < 50:
            continue
        ys, xs = np.where(m)
        out.append((m, ar, (ys.mean(), xs.mean())))
    return out


def pick_patient(cands, prev_cxy, H, W):
    # Greedy single-patient tracker: follow prev centroid; else prefer the
    # largest + most-central person (the patient dominates and walks centrally;
    # peripheral clinicians are smaller/off to the side).
    if not cands:
        return None, prev_cxy
    if prev_cxy is not None:
        amax = max(c[1] for c in cands)
        near = [c for c in cands if c[1] > 0.25 * amax]  # ignore tiny distractors
        c = min(near, key=lambda c: (c[2][0] - prev_cxy[0]) ** 2 + (c[2][1] - prev_cxy[1]) ** 2)
    else:
        cy0, cx0 = H / 2, W / 2
        amax = max(c[1] for c in cands)
        c = max(cands, key=lambda c: (c[1] / amax) - 0.6 * (((c[2][0] - cy0) / H) ** 2 + ((c[2][1] - cx0) / W) ** 2) ** 0.5)
    return c[0], (c[2][0], c[2][1])


def smooth_crop_params(bboxes, vis, W, H):
    n = len(bboxes)
    bboxes = np.asarray(bboxes, np.float64)
    bw = bboxes[:, 2] - bboxes[:, 0]; bh = bboxes[:, 3] - bboxes[:, 1]
    cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5; cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    side = np.maximum(bw, bh) * PAD
    vis = np.asarray(vis, bool)
    if vis.any():
        first = int(np.argmax(vis)); last = n - 1 - int(np.argmax(vis[::-1]))
        for arr in (cx, cy, side):
            arr[:first] = arr[first]; arr[last + 1:] = arr[last]
        # linear-interp interior invisible frames
        idx = np.arange(n)
        for arr in (cx, cy, side):
            good = vis | (idx < first) | (idx > last)
            if (~good).any():
                arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    else:
        cx[:] = W / 2; cy[:] = H / 2; side[:] = min(W, H)
    cx = gaussian_filter1d(cx, SIGMA); cy = gaussian_filter1d(cy, SIGMA); side = gaussian_filter1d(side, SIGMA)
    side = np.clip(side, 16, min(W, H))
    cx = np.clip(cx, side / 2, W - side / 2); cy = np.clip(cy, side / 2, H - side / 2)
    return cx, cy, side


def process(fn):
    raw = os.path.join(RAW_DIR, fn)
    out_vid = os.path.join(CROP_DIR, fn)
    out_mask = os.path.join(MASK_DIR, fn[:-4] + ".npz")
    if os.path.exists(out_vid) and os.path.exists(out_mask):
        return "skip", 0.0
    if not os.path.exists(raw):
        return "missing_raw", 0.0
    os.makedirs(os.path.dirname(out_vid), exist_ok=True)
    os.makedirs(os.path.dirname(out_mask), exist_ok=True)
    frames, fps = read_frames(raw, MAXSIDE)
    if not frames:
        return "empty", 0.0
    H, W = frames[0].shape[:2]
    session = proc.init_video_session(video=[Image.fromarray(f) for f in frames],
                                      inference_device=dev, dtype=torch.bfloat16)
    proc.add_text_prompt(session, "person")
    full_masks = np.zeros((len(frames), H, W), bool)
    bboxes = np.zeros((len(frames), 4), np.float64); vis = np.zeros(len(frames), bool)
    prev_cxy = None
    for i in range(len(frames)):
        with torch.inference_mode():
            o = model(inference_session=session, frame_idx=i)
        res = proc.postprocess_outputs(session, o, original_sizes=[[H, W]])
        obj = res[0] if isinstance(res, (list, tuple)) else res
        mk = obj.get("masks") if isinstance(obj, dict) else None
        cands = person_candidates(mk.float().cpu().numpy()) if mk is not None and len(mk) > 0 else []
        m, prev_cxy = pick_patient(cands, prev_cxy, H, W)
        if m is not None and m.any():
            full_masks[i] = m
            ys, xs = np.where(m)
            bboxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]; vis[i] = True
    cx, cy, side = smooth_crop_params(bboxes, vis, W, H)

    # encode cropped video via ffmpeg rawvideo pipe + crop masks to 256
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{OUT}x{OUT}", "-r", f"{fps:.4f}", "-i", "-",
         "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out_vid],
        stdin=subprocess.PIPE)
    cmasks = np.zeros((len(frames), MASK_RES, MASK_RES), bool)
    for i in range(len(frames)):
        x0 = int(round(cx[i] - side[i] / 2)); y0 = int(round(cy[i] - side[i] / 2)); s = int(round(side[i]))
        x0 = max(0, min(x0, W - s)); y0 = max(0, min(y0, H - s))
        crop = frames[i][y0:y0 + s, x0:x0 + s]
        ff.stdin.write(cv2.resize(crop, (OUT, OUT), interpolation=cv2.INTER_LINEAR).astype(np.uint8).tobytes())
        mc = full_masks[i][y0:y0 + s, x0:x0 + s].astype(np.uint8)
        cmasks[i] = cv2.resize(mc, (MASK_RES, MASK_RES), interpolation=cv2.INTER_NEAREST) > 0
    ff.stdin.close(); ff.wait()
    np.savez_compressed(out_mask, packed=np.packbits(cmasks), shape=np.array(cmasks.shape, np.int32))
    return ("ok", 100.0 * cmasks.mean())


done = skip = fail = 0
covs = []
for vi, fn in enumerate(vids):
    try:
        status, cov = process(fn)
    except Exception:
        status, cov = "FAIL", 0.0
        print("FAIL", fn, "\n", traceback.format_exc()[-500:], flush=True)
    if status == "ok":
        done += 1; covs.append(cov)
    elif status == "skip":
        skip += 1
    else:
        fail += 1
        if status not in ("FAIL",):
            print(f"  {status}: {fn}", flush=True)
    if status in ("ok",) or vi % 20 == 0:
        mc = np.mean(covs) if covs else 0
        print(f"[{vi+1}/{len(vids)}] {fn} {status} cov={cov:.1f}% "
              f"(done={done} skip={skip} fail={fail} meancov={mc:.1f}%)", flush=True)
print(f"DONE done={done} skip={skip} fail={fail} masks={len(glob.glob(MASK_DIR+'/**/*.npz', recursive=True))} "
      f"meancov={np.mean(covs) if covs else 0:.1f}%", flush=True)
