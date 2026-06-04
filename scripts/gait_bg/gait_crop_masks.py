"""SAM3 'person' masks on the already patient-cropped gait videos. Because the
crop is tightly centred on the patient, 'person' is now unambiguous -- take the
largest mask per frame. Output mask npz in cropped-frame coords at 256, the exact
format feral.phone_aug.BGReplacer expects. Resumable; only TRAIN gait videos need
masks (bg-replace), but we mask everything non-neg for simplicity.
"""
import os, sys, json, glob, traceback
import numpy as np, cv2, torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor

LABELS = "/workspace/labels/feral_gait_labels_phoneaug_cleanval.json"
CROP = "/workspace/data_gaitcrop"; MASK = "/workspace/masks"; MRES = 256
dev = "cuda"
os.makedirs(MASK, exist_ok=True)
proc = Sam3VideoProcessor.from_pretrained("facebook/sam3")
model = Sam3VideoModel.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(dev).eval()

lab = json.load(open(LABELS))
vids = sorted(set(v for v in lab["splits"]["train"] if not v.startswith("fog_negs/")))  # train only: bg-replace is train-only
print(f"mask {len(vids)} cropped TRAIN gait videos", flush=True)
MAXF = 1500  # skip very long clips (tulip ~4800f) to avoid SAM-video OOM; bg-replace degrades gracefully


def read(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release(); return fr


def largest(a):
    if a is None or (a.ndim == 3 and a.shape[0] == 0): return None
    if a.ndim == 2: return a > 0.5
    ar = (a > 0.5).reshape(a.shape[0], -1).sum(1)
    return a[int(ar.argmax())] > 0.5


done = skip = fail = 0; covs = []
for vi, fn in enumerate(vids):
    outp = os.path.join(MASK, fn[:-4] + ".npz")
    vpath = os.path.join(CROP, fn)
    if os.path.exists(outp): skip += 1; continue
    if not os.path.exists(vpath): fail += 1; print("no crop", fn, flush=True); continue
    try:
        frames = read(vpath)
        if not frames: fail += 1; continue
        if len(frames) > MAXF:
            print(f"skip long ({len(frames)}f) {fn}", flush=True); skip += 1; continue
        H, W = frames[0].shape[:2]
        sess = proc.init_video_session(video=[Image.fromarray(f) for f in frames], inference_device=dev, dtype=torch.bfloat16)
        proc.add_text_prompt(sess, "person")
        m = np.zeros((len(frames), MRES, MRES), bool)
        for i in range(len(frames)):
            with torch.inference_mode():
                o = model(inference_session=sess, frame_idx=i)
            res = proc.postprocess_outputs(sess, o, original_sizes=[[H, W]])
            ob = res[0] if isinstance(res, (list, tuple)) else res
            mk = ob.get("masks") if isinstance(ob, dict) else None
            lm = largest(mk.float().cpu().numpy()) if mk is not None and len(mk) > 0 else None
            if lm is not None and lm.any():
                m[i] = cv2.resize(lm.astype(np.uint8), (MRES, MRES), interpolation=cv2.INTER_NEAREST) > 0
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        np.savez_compressed(outp, packed=np.packbits(m), shape=np.array(m.shape, np.int32))
        done += 1; cov = 100 * m.mean(); covs.append(cov)
        if vi % 20 == 0 or done % 25 == 0:
            print(f"[{vi+1}/{len(vids)}] {fn} cov={cov:.1f}% (done={done} skip={skip} fail={fail} mean={np.mean(covs):.1f}%)", flush=True)
    except Exception:
        fail += 1; print("FAIL", fn, traceback.format_exc()[-200:], flush=True)
print(f"DONE done={done} skip={skip} fail={fail} mean={np.mean(covs) if covs else 0:.1f}%", flush=True)
