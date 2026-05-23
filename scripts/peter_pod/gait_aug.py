"""Gait dataset crop augmentation + Dataset.

Data format
-----------
For each gait dataset (auto-gait / koa-pd-nm-gait / tulip-gait) we produce a
single `bboxes.json` of the form:

    {
      "video_filename.mp4": {
        "img_w": int, "img_h": int, "fps": float, "n_frames": int,
        "small_bboxes":     [[x1,y1,x2,y2], ...],   # per frame, knees+ankles + 20%
        "large_bboxes":     [[x1,y1,x2,y2], ...],   # per frame, whole patient + 50%
        "patient_visible":  [bool, ...],            # per frame
        "patient_row_idx":  [int, ...],             # row idx in pose parquet (or -1)
      },
      ...
    }

All bboxes are in the ORIGINAL image coordinate system (so augmentation/cropping
math is done at load time against the raw video). Bbox arrays are length =
`n_frames`. For frames where the patient was not detected, both bboxes are
interpolated from the nearest visible frames; `patient_visible[i]` records
ground-truth-ish visibility.

Training-time augmentation
--------------------------
For each chunk we sample ONCE (so all frames in the chunk get the same zoom):
  s     ~ Uniform(0, 1)                       # 0 = small (tight on knees/feet)
                                              # 1 = large (full patient)
  ar_x  ~ exp(Uniform(-log(R), log(R)))       # aspect-ratio jitter, R~1.3
  ar_y  ~ exp(Uniform(-log(R), log(R)))

Per frame:
  bbox  = lerp(small_bboxes[i], large_bboxes[i], s)
  bbox  = scale-around-centre(bbox, ar_x, ar_y)
  crop  = read_frame(i)[bbox]    (reflection-padded if bbox goes off image)
  out_i = resize(crop, OUT_HW)

Output tensor shape: (T, C, H, W) in [0, 1] float32 (or uint8 if you prefer).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import cv2
import decord
import numpy as np
import torch
from torch.utils.data import Dataset

# COCO body keypoint indices in Sapiens Goliath (first 17 == COCO body).
HIP_KPS = (11, 12)
KNEE_KPS = (13, 14)
ANKLE_KPS = (15, 16)
LOWER_KPS = KNEE_KPS + ANKLE_KPS  # used for small bbox


# ---------------------------------------------------------------------------
# Bbox precomputation (run once, after patient tracking is settled)
# ---------------------------------------------------------------------------

def _expand_bbox(x1: float, y1: float, x2: float, y2: float,
                 frac: float) -> tuple[float, float, float, float]:
    """Expand bbox by `frac` of its size on each side (so total grows by 2*frac)."""
    w, h = x2 - x1, y2 - y1
    dx, dy = w * frac, h * frac
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def compute_bboxes_for_video(
    patient_rows: "pandas.DataFrame",
    n_frames: int,
    img_w: int,
    img_h: int,
    fps: float = 30.0,
    small_pad: float = 0.20,
    large_pad: float = 0.50,
    smoothing_k: int = 2,
    entry_window_ms: float = 1000.0,
) -> dict:
    """Build the small/large bbox arrays + visibility flag for one video.

    Only real detections (det_score not NaN) seed the bbox arrays; stride-2
    interpolated rows from the pose extractor are ignored so we never train
    on invented kpts.

    Visibility is smoothed by a +/- smoothing_k frame morphological dilate so
    stride-2 gaps don't flicker the flag, then the first int(fps) frames of
    each visible run are forced False (the half-body entry transient).

    Bboxes are linearly interpolated within visible runs from the bracketing
    real detections. Frames where patient_visible=False keep an interpolated
    bbox value but should not be sampled at training time (use the visibility
    flag to gate chunk selection).
    """
    import pandas as pd

    small = np.full((n_frames, 4), np.nan, dtype=np.float32)
    large = np.full((n_frames, 4), np.nan, dtype=np.float32)
    real_det = np.zeros(n_frames, dtype=bool)

    real = patient_rows[patient_rows["det_score"].notna()] if "det_score" in patient_rows.columns else patient_rows
    for _, r in real.iterrows():
        f = int(r["frame"])
        if f < 0 or f >= n_frames:
            continue
        lx1, ly1, lx2, ly2 = _expand_bbox(
            r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"], large_pad
        )
        kxs = [r[f"k{k}_x"] for k in LOWER_KPS]
        kys = [r[f"k{k}_y"] for k in LOWER_KPS]
        sx1, sy1, sx2, sy2 = min(kxs), min(kys), max(kxs), max(kys)
        sx1, sy1, sx2, sy2 = _expand_bbox(sx1, sy1, sx2, sy2, small_pad)
        small[f] = (sx1, sy1, sx2, sy2)
        large[f] = (lx1, ly1, lx2, ly2)
        real_det[f] = True

    # Dilate real_det by smoothing_k frames each side to bridge stride-2 gaps.
    visible = _dilate_bool(real_det, smoothing_k)

    # Mask the first entry_window_ms of every visible run (entry half-body window).
    entry_window = max(1, int(round(fps * entry_window_ms / 1000.0)))
    visible = _mask_run_starts(visible, entry_window)

    # Linear interp the bbox arrays so consumers always have a number, but
    # callers MUST gate on `patient_visible` before using them.
    small = _interp_nan_bboxes(small)
    large = _interp_nan_bboxes(large)

    return {
        "img_w": int(img_w), "img_h": int(img_h),
        "fps": float(fps),
        "n_frames": int(n_frames),
        "small_bboxes": small.tolist(),
        "large_bboxes": large.tolist(),
        "patient_visible": visible.tolist(),
    }


def _dilate_bool(arr: np.ndarray, k: int) -> np.ndarray:
    """Boolean morphological dilation by +/- k along the 1D array."""
    if k <= 0:
        return arr.copy()
    out = arr.copy()
    for shift in range(1, k + 1):
        out[shift:] |= arr[:-shift]
        out[:-shift] |= arr[shift:]
    return out


def _mask_run_starts(visible: np.ndarray, n_mask: int) -> np.ndarray:
    """Set the first n_mask frames of every contiguous True run to False."""
    out = visible.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            i += 1; continue
        j = i
        while j < n and out[j]:
            j += 1
        # run is [i, j); mask [i, min(j, i+n_mask))
        out[i : min(j, i + n_mask)] = False
        i = j
    return out


def _interp_nan_bboxes(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN rows. Edges are filled by nearest valid."""
    out = arr.copy()
    n = len(out)
    valid = ~np.isnan(out[:, 0])
    if not valid.any():
        out[:] = 0  # nothing to do; caller should treat patient as never visible
        return out
    valid_idx = np.where(valid)[0]
    for c in range(4):
        out[:, c] = np.interp(np.arange(n), valid_idx, arr[valid_idx, c])
    return out


# ---------------------------------------------------------------------------
# Augmentation core
# ---------------------------------------------------------------------------

@dataclass
class CropParams:
    """Crop-augmentation params sampled once per chunk."""
    s: float           # lerp factor: 0 = small bbox, 1 = large bbox
    ar_x: float        # multiplicative width jitter around bbox centre
    ar_y: float        # multiplicative height jitter
    flip_x: bool       # horizontal flip (subject-level)

    @staticmethod
    def sample(rng: np.random.Generator,
               ar_max: float = 1.3,
               flip_p: float = 0.5,
               s_min: float = 0.0,
               s_max: float = 1.0) -> "CropParams":
        log_r = math.log(ar_max)
        return CropParams(
            s=float(rng.uniform(s_min, s_max)),
            ar_x=float(math.exp(rng.uniform(-log_r, log_r))),
            ar_y=float(math.exp(rng.uniform(-log_r, log_r))),
            flip_x=bool(rng.uniform() < flip_p),
        )


def lerp_bbox(small: np.ndarray, large: np.ndarray, s: float) -> np.ndarray:
    """small, large: shape (..., 4). s: scalar in [0, 1]. Returns (..., 4)."""
    return (1.0 - s) * small + s * large


def jitter_aspect(bbox: np.ndarray, ar_x: float, ar_y: float) -> np.ndarray:
    """Scale bbox w/h by (ar_x, ar_y) around its centre. bbox shape (..., 4)."""
    x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1) * ar_x, (y2 - y1) * ar_y
    return np.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], axis=-1)


def crop_and_resize(
    frame: np.ndarray, bbox: np.ndarray, out_h: int, out_w: int,
    flip_x: bool = False,
) -> np.ndarray:
    """Crop a single frame to `bbox` (float HxW image coords) and resize.

    bbox may extend outside the image — out-of-image regions are padded BLACK
    (constant 0). Returns uint8 array (out_h, out_w, C).
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    # Integer crop coords, allowing negative / out-of-image extents
    ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    # Pad amount on each side
    pad_l = max(0, -ix1)
    pad_t = max(0, -iy1)
    pad_r = max(0, ix2 - W)
    pad_b = max(0, iy2 - H)
    if pad_l or pad_t or pad_r or pad_b:
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        ix1 += pad_l; ix2 += pad_l
        iy1 += pad_t; iy2 += pad_t
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        # degenerate (bbox of zero area); return a zero image
        return np.zeros((out_h, out_w, frame.shape[2]), dtype=np.uint8)
    out = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    if flip_x:
        out = out[:, ::-1].copy()
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class GaitChunk:
    """One training sample: a contiguous window of frames from one video."""
    video_path: str
    start_frame: int                # inclusive
    end_frame: int                  # exclusive
    stride: int                     # temporal stride within the chunk


class GaitVideoDataset(Dataset):
    """Loads chunks of cropped patient frames with per-chunk augmentation.

    Chunks are pre-filtered at construction: only (video, start) windows where
    patient_visible is True for ALL `chunk_len` sampled frames are kept. This
    makes len(dataset) the exact count of usable chunks — every __getitem__
    returns a chunk with the patient fully present.

    Args:
        video_root: dir holding source MP4s (videos referenced in bboxes_json).
        bboxes_json: path to the per-dataset bbox sidecar described above.
        chunk_len: number of frames per training chunk (T).
        out_hw: (H, W) of output crops.
        stride: temporal stride within a chunk (1 = adjacent frames).
        chunk_start_stride: gap (in raw frames) between adjacent chunk starts.
            Defaults to chunk_len * stride / 2 (50% overlap).
        labels_fn: optional callable(video_filename, start_frame, end_frame) ->
            torch.Tensor. If None, returns just the video tensor.
        ar_max: max aspect-ratio jitter factor (e.g. 1.3 = +/- 30%).
        flip_p: probability of horizontal flip per chunk.
        s_range: (s_min, s_max) for the small↔large lerp factor.
        seed: RNG seed for crop-param sampling.
    """

    def __init__(
        self,
        video_root: str,
        bboxes_json: str,
        chunk_len: int = 64,
        out_hw: tuple[int, int] = (384, 384),
        stride: int = 1,
        chunk_start_stride: Optional[int] = None,
        labels_fn: Optional[Callable[[str, int, int], torch.Tensor]] = None,
        ar_max: float = 1.3,
        flip_p: float = 0.5,
        s_range: tuple[float, float] = (0.0, 1.0),
        seed: int = 0,
    ):
        self.video_root = video_root
        with open(bboxes_json) as fp:
            raw = json.load(fp)
        self.chunk_len = chunk_len
        self.out_hw = out_hw
        self.stride = stride
        self.chunk_start_stride = chunk_start_stride or max(1, (chunk_len * stride) // 2)
        self.labels_fn = labels_fn
        self.ar_max = ar_max
        self.flip_p = flip_p
        self.s_range = s_range
        self.base_seed = seed

        # Pre-convert bbox arrays to numpy and enumerate valid (vid, start) chunks.
        self.meta: dict[str, dict] = {}
        self.chunks: list[tuple[str, int]] = []
        span = chunk_len * stride
        for vid, m in raw.items():
            small = np.asarray(m["small_bboxes"], dtype=np.float32)
            n = int(m["n_frames"])
            if "large_bboxes" in m:
                large = np.asarray(m["large_bboxes"], dtype=np.float32)
            else:
                W, H = int(m["img_w"]), int(m["img_h"])
                large = np.broadcast_to(
                    np.array([0.0, 0.0, W, H], dtype=np.float32), (n, 4)
                ).copy()
            vis = np.asarray(m["patient_visible"], dtype=bool)
            if n < span:
                continue
            self.meta[vid] = {
                "img_w": m["img_w"], "img_h": m["img_h"],
                "n_frames": n, "fps": m.get("fps", 30.0),
                "small": small, "large": large, "visible": vis,
            }
            for start in range(0, n - span + 1, self.chunk_start_stride):
                # Need visible at every sampled frame within the chunk.
                sub = vis[start : start + span : stride]
                if len(sub) == chunk_len and sub.all():
                    self.chunks.append((vid, start))

    def __len__(self) -> int:
        return len(self.chunks)

    def _rng(self, idx: int) -> np.random.Generator:
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi is not None else 0
        return np.random.default_rng(self.base_seed * 100003 + idx * 31 + worker_id)

    @property
    def videos(self) -> list[str]:
        return list(self.meta.keys())

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        vid, start = self.chunks[idx]
        m = self.meta[vid]
        end = start + self.chunk_len * self.stride
        params = CropParams.sample(rng, ar_max=self.ar_max, flip_p=self.flip_p,
                                   s_min=self.s_range[0], s_max=self.s_range[1])

        frame_idxs = np.arange(start, end, self.stride, dtype=np.int64)
        # Per-frame bbox after lerp + AR jitter (shared params across the chunk)
        small = m["small"][frame_idxs]
        large = m["large"][frame_idxs]
        bbox = lerp_bbox(small, large, params.s)
        bbox = jitter_aspect(bbox, params.ar_x, params.ar_y)

        video_path = os.path.join(self.video_root, vid)
        vr = decord.VideoReader(video_path, num_threads=1)
        # decord returns RGB
        frames = vr.get_batch(frame_idxs.tolist()).asnumpy()  # (T, H, W, 3)

        out = np.empty((len(frame_idxs), self.out_hw[0], self.out_hw[1], 3), dtype=np.uint8)
        for i in range(len(frame_idxs)):
            out[i] = crop_and_resize(
                frames[i], bbox[i], self.out_hw[0], self.out_hw[1],
                flip_x=params.flip_x,
            )

        # (T, H, W, 3) uint8  ->  (T, 3, H, W) float [0,1]
        video = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous().float().div_(255.0)

        sample = {
            "video": video,
            "vid_name": vid,
            "start": int(start),
            "end": int(end),
            "stride": int(self.stride),
            "crop_params": {"s": params.s, "ar_x": params.ar_x,
                            "ar_y": params.ar_y, "flip_x": params.flip_x},
        }
        if self.labels_fn is not None:
            sample["label"] = self.labels_fn(vid, int(start), int(end))
        return sample


class GaitUnifiedDataset(GaitVideoDataset):
    """Concatenation of auto-gait + koa + tulip with labels attached.

    Constructed from:
      bboxes_json:  unified_gait_bboxes.json  (keys like "<dataset>/<video>")
      labels_json:  unified_gait_labels.json  (with splits + per-video label dict)

    Selecting a split:
      split = "train" or "val"   (whatever is in labels_json["splits"]).

    Resolves video paths via labels_json["video_roots"][dataset]. Returns a
    sample dict identical to GaitVideoDataset plus "label" (a float32 tensor of
    shape (1,) with the gait_severity target) and "dataset" / "key".
    """

    def __init__(
        self,
        bboxes_json: str,
        labels_json: str,
        split: str = "train",
        chunk_len: int = 64,
        out_hw: tuple[int, int] = (384, 384),
        stride: int = 1,
        chunk_start_stride: Optional[int] = None,
        ar_max: float = 1.3,
        flip_p: float = 0.5,
        s_range: tuple[float, float] = (0.0, 1.0),
        seed: int = 0,
    ):
        with open(bboxes_json) as fp:
            self._raw_bboxes = json.load(fp)
        with open(labels_json) as fp:
            self._labels_doc = json.load(fp)
        self._labels = self._labels_doc["labels"]
        self._video_roots = self._labels_doc["video_roots"]
        split_keys = set(self._labels_doc["splits"].get(split, []))
        if not split_keys:
            raise ValueError(f"empty split {split!r}; available: {list(self._labels_doc['splits'])}")

        self.chunk_len = chunk_len
        self.out_hw = out_hw
        self.stride = stride
        self.chunk_start_stride = chunk_start_stride or max(1, (chunk_len * stride) // 2)
        self.ar_max = ar_max
        self.flip_p = flip_p
        self.s_range = s_range
        self.base_seed = seed
        self.labels_fn = None  # unused — we resolve labels by key directly

        # Build per-key meta + enumerate valid chunks.
        # Pre-cropped bboxes files don't store `large_bboxes` (the pre-crop is
        # the implicit large region); synthesize a full-frame large bbox.
        self.meta: dict[str, dict] = {}
        self.chunks: list[tuple[str, int]] = []
        span = chunk_len * stride
        for key in split_keys:
            if key not in self._raw_bboxes:
                continue
            m = self._raw_bboxes[key]
            n = int(m["n_frames"])
            if n < span:
                continue
            small = np.asarray(m["small_bboxes"], dtype=np.float32)
            if "large_bboxes" in m:
                large = np.asarray(m["large_bboxes"], dtype=np.float32)
            else:
                # implicit large = full frame, broadcast across all frames
                W, H = int(m["img_w"]), int(m["img_h"])
                large = np.broadcast_to(
                    np.array([0.0, 0.0, W, H], dtype=np.float32), (n, 4)
                ).copy()
            vis = np.asarray(m["patient_visible"], dtype=bool)
            self.meta[key] = {
                "img_w": m["img_w"], "img_h": m["img_h"],
                "n_frames": n, "fps": m.get("fps", 30.0),
                "small": small, "large": large, "visible": vis,
            }
            for start in range(0, n - span + 1, self.chunk_start_stride):
                sub = vis[start : start + span : stride]
                if len(sub) == chunk_len and sub.all():
                    self.chunks.append((key, start))

    @property
    def videos(self) -> list[str]:
        return list(self.meta.keys())

    def _resolve_video_path(self, key: str) -> str:
        # key = "<dataset>/<video_basename>"; some datasets have nested keys
        # like "tulip/gait/<file>"; split off the LAST '/' for the video name.
        head, _, video = key.rpartition("/")
        # head might be "tulip/gait" or "auto-gait" etc — the dict has both.
        if head not in self._video_roots:
            raise KeyError(f"no video_root for dataset prefix {head!r}")
        return os.path.join(self._video_roots[head], video)

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        key, start = self.chunks[idx]
        m = self.meta[key]
        end = start + self.chunk_len * self.stride
        params = CropParams.sample(rng, ar_max=self.ar_max, flip_p=self.flip_p,
                                   s_min=self.s_range[0], s_max=self.s_range[1])
        frame_idxs = np.arange(start, end, self.stride, dtype=np.int64)
        small = m["small"][frame_idxs]
        large = m["large"][frame_idxs]
        bbox = lerp_bbox(small, large, params.s)
        bbox = jitter_aspect(bbox, params.ar_x, params.ar_y)

        video_path = self._resolve_video_path(key)
        vr = decord.VideoReader(video_path, num_threads=1)
        frames = vr.get_batch(frame_idxs.tolist()).asnumpy()

        out = np.empty((len(frame_idxs), self.out_hw[0], self.out_hw[1], 3), dtype=np.uint8)
        for i in range(len(frame_idxs)):
            out[i] = crop_and_resize(frames[i], bbox[i], self.out_hw[0], self.out_hw[1],
                                     flip_x=params.flip_x)
        video = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous().float().div_(255.0)

        head, _, video_name = key.rpartition("/")
        label_info = self._labels.get(key, {})
        gait_sev = float(label_info.get("gait_severity", float("nan")))
        return {
            "video": video,
            "key": key,
            "dataset": head,
            "vid_name": video_name,
            "start": int(start),
            "end": int(end),
            "stride": int(self.stride),
            "crop_params": {"s": params.s, "ar_x": params.ar_x,
                            "ar_y": params.ar_y, "flip_x": params.flip_x},
            "label": torch.tensor([gait_sev], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Tiny smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Smoke test: build a dummy bboxes.json from one parquet+video and load 1 chunk.

    Usage:
        python gait_aug.py \\
            --parquet /root/data/auto-gait/poses_sapiens03b_goliath/0.mp4.parquet \\
            --video /root/data/auto-gait/videos/0.mp4 \\
            --out /tmp/smoke_bboxes.json
    """
    import argparse, sys, subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="/tmp/smoke_bboxes.json")
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.parquet)
    det = df[df["det_score"].notna()]
    # heuristic: highest-bbox-area track-less stand-in for "patient" — for smoke only
    det = det.copy()
    det["area"] = (det["bbox_x2"] - det["bbox_x1"]) * (det["bbox_y2"] - det["bbox_y1"])
    # take the row with largest area per frame
    patient_rows = det.sort_values("area", ascending=False).drop_duplicates("frame")

    # video dims via ffprobe
    out = subprocess.check_output([
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,nb_read_frames","-count_frames",
        "-of","json", args.video
    ])
    info = json.loads(out)["streams"][0]
    W, H = int(info["width"]), int(info["height"])
    N = int(info.get("nb_read_frames", df["frame"].max() + 1))

    meta = compute_bboxes_for_video(patient_rows, n_frames=N, img_w=W, img_h=H)
    bboxes_json = {os.path.basename(args.video): {**meta, "fps": 30.0}}
    with open(args.out, "w") as fp:
        json.dump(bboxes_json, fp)
    print(f"wrote {args.out}")

    ds = GaitVideoDataset(
        video_root=os.path.dirname(args.video),
        bboxes_json=args.out,
        chunk_len=16, out_hw=(224, 224), chunks_per_video=1, stride=2,
    )
    print(f"len(ds)={len(ds)}")
    sample = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in sample.items()})
