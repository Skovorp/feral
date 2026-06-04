"""Phone-capture augmentation stack for FERAL regression clips.

Operates on a clip tensor (T, C, H, W), uint8 [0,255], and returns
(T, C, size, size) uint8. All geometric/photometric params are sampled ONCE
per clip (a recording has one lens, one lighting, one codec); only sensor noise
and the handheld-shake walk vary per frame. Temporal jitter is conservative
(frame drops / freezes only — NO global fps rescale, which would change apparent
tapping speed and corrupt the severity label).
"""
import os
import glob
import random
import numpy as np
import torch
import cv2


def _to_np(video):
    # (T,C,H,W) uint8 torch -> (T,H,W,C) uint8 numpy
    return video.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)


def _to_torch(arr):
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def _aggressive_crop(frames, size, rng):
    T, H, W, C = frames.shape
    scale = rng.uniform(0.30, 1.0)            # aggressive zoom range
    ch = cw = int(round(min(H, W) * np.sqrt(scale)))
    ch = min(ch, H); cw = min(cw, W)
    y0 = rng.randint(0, H - ch + 1); x0 = rng.randint(0, W - cw + 1)
    out = np.empty((T, size, size, C), np.uint8)
    for t in range(T):
        out[t] = cv2.resize(frames[t, y0:y0 + ch, x0:x0 + cw], (size, size), interpolation=cv2.INTER_LINEAR)
    return out


def _lighting(frames, rng):
    bright = rng.uniform(0.55, 1.35)
    contrast = rng.uniform(0.7, 1.3)
    gamma = rng.uniform(0.7, 1.5)
    warm = rng.uniform(0.85, 1.15)
    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((lut - 0.5) * contrast + 0.5, 0, 1)
    lut = np.clip(lut * bright, 0, 1) ** gamma
    lut = (lut * 255).astype(np.uint8)
    out = cv2.LUT(frames, lut)
    out = out.astype(np.float32)
    out[..., 0] = np.clip(out[..., 0] * warm, 0, 255)
    out[..., 2] = np.clip(out[..., 2] * (2.0 - warm), 0, 255)
    return out.astype(np.uint8)


def _lens_warp(frames, rng):
    if rng.random() < 0.4:
        return frames
    k1 = rng.uniform(-0.35, 0.35)
    T, H, W, C = frames.shape
    cx, cy = W / 2.0, H / 2.0
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    nx = (xs - cx) / cx; ny = (ys - cy) / cy
    r2 = nx * nx + ny * ny
    f = 1 + k1 * r2
    mapx = (nx * f * cx + cx).astype(np.float32)
    mapy = (ny * f * cy + cy).astype(np.float32)
    out = np.empty_like(frames)
    for t in range(T):
        out[t] = cv2.remap(frames[t], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def _camera_shake(frames, rng):
    if rng.random() < 0.35:
        return frames
    T, H, W, C = frames.shape
    amp_t = rng.uniform(0.005, 0.04)      # translation amplitude (fraction of side)
    amp_r = rng.uniform(0.0, 2.5)          # rotation deg amplitude
    # smooth random walk -> low-pass cumulative noise
    dx = np.cumsum(rng.randn(T)); dy = np.cumsum(rng.randn(T)); dr = np.cumsum(rng.randn(T))
    for a in (dx, dy, dr):
        a -= a.mean()
    dx = dx / (np.abs(dx).max() + 1e-6) * amp_t * W
    dy = dy / (np.abs(dy).max() + 1e-6) * amp_t * H
    dr = dr / (np.abs(dr).max() + 1e-6) * amp_r
    out = np.empty_like(frames)
    for t in range(T):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), float(dr[t]), 1.0)
        M[0, 2] += dx[t]; M[1, 2] += dy[t]
        out[t] = cv2.warpAffine(frames[t], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def _blur(frames, rng):
    r = rng.random()
    if r < 0.4:
        return frames
    out = np.empty_like(frames)
    if r < 0.7:  # motion blur, per-clip kernel
        L = rng.randint(5, 17); ang = rng.uniform(0, 180)
        k = np.zeros((L, L), np.float32); k[L // 2, :] = 1.0
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((L / 2, L / 2), ang, 1.0), (L, L)); k /= k.sum()
        for t in range(len(frames)):
            out[t] = cv2.filter2D(frames[t], -1, k)
    else:        # defocus, per-clip radius
        rad = rng.randint(2, 7)
        k = np.zeros((2 * rad + 1, 2 * rad + 1), np.float32); cv2.circle(k, (rad, rad), rad, 1, -1); k /= k.sum()
        for t in range(len(frames)):
            out[t] = cv2.filter2D(frames[t], -1, k)
    return out


def _sensor_noise(frames, rng):
    if rng.random() < 0.4:
        return frames
    sigma = rng.uniform(3, 14); shot = rng.uniform(0.0, 0.06)
    x = frames.astype(np.float32)
    x = x + rng.randn(*x.shape) * (shot * np.sqrt(np.clip(x, 0, 255) * 255))
    x = x + rng.randn(*x.shape) * sigma
    return np.clip(x, 0, 255).astype(np.uint8)


def _jpeg(frames, rng):
    if rng.random() < 0.4:
        return frames
    q = int(rng.randint(12, 45))
    out = np.empty_like(frames)
    for t in range(len(frames)):
        ok, enc = cv2.imencode(".jpg", frames[t], [cv2.IMWRITE_JPEG_QUALITY, q])
        out[t] = cv2.imdecode(enc, 1)
    return out


def _temporal_jitter(frames, rng):
    # conservative: frame drops/dups + occasional freeze. NO global fps rescale.
    if rng.random() < 0.5:
        return frames
    T = len(frames); idx = np.arange(T)
    if rng.random() < 0.6:  # dropped frames held
        for _ in range(rng.randint(2, 7)):
            j = rng.randint(1, T); idx[j:] = np.clip(idx[j:] - 1, 0, T - 1)
    else:                    # brief freeze
        s = rng.randint(0, max(1, T - 10)); L = rng.randint(4, 10)
        idx[s:s + L] = s; idx[s + L:] = np.clip(np.arange(T - s - L) + s + 1, 0, T - 1)
    return frames[idx]


class PhoneAug:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        f = _to_np(video)                 # (T,H,W,C)
        f = _temporal_jitter(f, rng)
        f = _aggressive_crop(f, self.size, rng)
        f = _lighting(f, rng)
        f = _lens_warp(f, rng)
        f = _camera_shake(f, rng)
        f = _blur(f, rng)
        f = _sensor_noise(f, rng)
        f = _jpeg(f, rng)
        return _to_torch(f)


def build_phone_aug(size):
    return PhoneAug(size)


class BGReplacer:
    """Background replacement using precomputed SAM3 hand masks.

    Called in the dataset on the decoded clip (T,C,H,W uint8) with `names`
    = [(filename, frame_idx_in_video, chunk_idx), ...]. Loads the per-video
    mask npz (256x256 packed bits), composites the hand onto a random clean
    (people-free) background with erode-then-feather. Robust: ANY problem
    (missing mask, shape mismatch, index OOB, decode error) returns the clip
    unchanged — bg-replace must never crash training.
    """
    ERODE_PX = 5
    FEATHER = 2.5

    def __init__(self, mask_dir, bg_dir, p=0.5):
        self.mask_dir = mask_dir
        self.p = p
        self.ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.ERODE_PX, self.ERODE_PX))
        self.bgs = []
        for f in sorted(glob.glob(os.path.join(bg_dir, "*.png")) + glob.glob(os.path.join(bg_dir, "*.jpg"))):
            try:
                self.bgs.append(np.array(__import__("PIL.Image", fromlist=["Image"]).open(f).convert("RGB")))
            except Exception:
                pass

    def __call__(self, video, names):
        try:
            if not self.bgs or random.random() > self.p:
                return video
            fn = names[0][0]
            mp = os.path.join(self.mask_dir, fn[:-4] + ".npz")
            if not os.path.exists(mp):
                return video
            d = np.load(mp)
            masks = np.unpackbits(d["packed"]).reshape(tuple(d["shape"])).astype(bool)  # (Tvid,256,256)
            idxs = [t[1] for t in names]
            if max(idxs) >= masks.shape[0]:
                return video
            T, C, H, W = video.shape
            f = video.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)  # (T,H,W,C)
            bg = self.bgs[random.randrange(len(self.bgs))]
            bg = cv2.resize(bg, (W, H))
            out = np.empty_like(f)
            for t in range(T):
                m = masks[idxs[t]]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                a = cv2.GaussianBlur(cv2.erode(m.astype(np.float32), self.ek, 1), (0, 0), self.FEATHER)
                a = np.clip(a, 0, 1)[..., None]
                out[t] = (a * f[t] + (1 - a) * bg).astype(np.uint8)
            return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()
        except Exception:
            return video


def build_bg_replacer(mask_dir, bg_dir, p=0.5):
    return BGReplacer(mask_dir, bg_dir, p)
