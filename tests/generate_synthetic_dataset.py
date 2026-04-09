#!/usr/bin/env python3
"""
Generate a synthetic test dataset for FERAL.

Renders 18 videos of two squares ("mice") in a 256x256 arena with a red
feeder rect in the bottom-right corner. Each video is composed of behavior
segments in random order. Two label files are produced from the same videos:
  - labels_singlelabel.json (5 classes, single label per frame, precedence rule)
  - labels_multilabel.json  (6 mouse-specific flags per frame)

Behavior segment types:
  - copulation:        both mice adjacent, not moving (rare; only in 2 videos)
  - playing:           both mice orbit a midpoint at offset phases
  - a_eats_b_sleeps:   mouse_a in feeder, mouse_b stationary elsewhere
  - a_sleeps_b_eats:   mirror image
  - a_sleeps_b_sleeps: both mice stationary in different corners
  - wandering:         both mice random-walking (single-label "other")

Each video has the 5 non-copulation segments in a random order with random
lengths (150-250 frames each). The 2 designated copulation videos additionally
have a copulation segment inserted at a random position.

Smooth transitions
------------------
Between segments the mice smoothly lerp from their previous positions to the
new segment's target start positions over TRANSITION_FRAMES frames. The first
TRANSITION_FRAMES of every segment (except the first segment of each video)
are these lerp frames; the remaining frames are the steady-state behavior.

Transition frames are labeled as "other" (single-label class 0, all-zero
multi-label row), not as the upcoming segment. They are visually ambiguous —
the mice are mid-move and not yet exhibiting the new behavior — so labeling
them as "other" is the honest call. This also means "other" gets meaningful
coverage from two sources (wandering segments + transitions), exercising the
`cls_name != 'other'` exclusion in metrics.py.

Determinism
-----------
The script is fully deterministic for a fixed SEED. Re-running should produce
identical labels and visually identical videos (h264 byte output may vary
slightly across ffmpeg builds, but the visible content is stable).

Usage:
    conda run -n feral python tests/generate_synthetic_dataset.py

Outputs:
    tests/fixtures/videos/synthetic_00.mp4 ... synthetic_17.mp4
    tests/fixtures/labels_singlelabel.json
    tests/fixtures/labels_multilabel.json
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ---- config ----
SEED = 0
NUM_VIDEOS = 18
RESOLUTION = 256
SQUARE_SIZE = 16
FRAME_RATE = 30
SEG_LEN_MIN = 150
SEG_LEN_MAX = 250
TRANSITION_FRAMES = 10
COPULATION_VIDEO_INDICES = (0, 1)

FEEDER_TL = (216, 216)
FEEDER_BR = (248, 248)

REQUIRED_SEGMENTS = (
    "playing",
    "a_eats_b_sleeps",
    "a_sleeps_b_eats",
    "a_sleeps_b_sleeps",
    "wandering",
)

SINGLE_LABEL_CLASS_NAMES = {
    "0": "other",
    "1": "copulation",
    "2": "eating",
    "3": "sleeping",
    "4": "playing",
}

MULTI_LABEL_CLASS_NAMES = {
    "0": "mouse_a_eating",
    "1": "mouse_a_sleeping",
    "2": "mouse_b_eating",
    "3": "mouse_b_sleeping",
    "4": "copulation",
    "5": "playing",
}

# "_transition" is a synthetic pseudo-segment used internally to mark frames
# that are mid-transition. It maps to "other" / no flags.
SEGMENT_TO_SINGLE = {
    "copulation": 1,
    "playing": 4,
    "a_eats_b_sleeps": 2,
    "a_sleeps_b_eats": 2,
    "a_sleeps_b_sleeps": 3,
    "wandering": 0,
    "_transition": 0,
}

SEGMENT_TO_MULTI = {
    "copulation": (4,),
    "playing": (5,),
    "a_eats_b_sleeps": (0, 3),
    "a_sleeps_b_eats": (1, 2),
    "a_sleeps_b_sleeps": (1, 3),
    "wandering": (),
    "_transition": (),
}

# Splits over the 18 videos. Copulation videos (0, 1) live in train so the
# rare-class subsampling code path has something to find.
SPLITS = {
    "train": list(range(12)),
    "val": list(range(12, 15)),
    "test": list(range(15, 17)),
    "inference": [17],
}


# ---- arena rendering ----

def draw_frame(pos_a, pos_b):
    frame = np.full((RESOLUTION, RESOLUTION, 3), 255, dtype=np.uint8)
    frame[FEEDER_TL[0]:FEEDER_BR[0], FEEDER_TL[1]:FEEDER_BR[1]] = (255, 0, 0)
    ya, xa = pos_a
    frame[ya:ya + SQUARE_SIZE, xa:xa + SQUARE_SIZE] = (0, 0, 0)
    yb, xb = pos_b
    frame[yb:yb + SQUARE_SIZE, xb:xb + SQUARE_SIZE] = (128, 128, 128)
    return frame


def clamp_pos(y, x):
    return (
        int(np.clip(y, 0, RESOLUTION - SQUARE_SIZE)),
        int(np.clip(x, 0, RESOLUTION - SQUARE_SIZE)),
    )


def lerp_pos(start, end, frac):
    return clamp_pos(
        start[0] + (end[0] - start[0]) * frac,
        start[1] + (end[1] - start[1]) * frac,
    )


def is_away_from_feeder(y, x):
    return (y + SQUARE_SIZE <= FEEDER_TL[0] or y >= FEEDER_BR[0] or
            x + SQUARE_SIZE <= FEEDER_TL[1] or x >= FEEDER_BR[1])


def random_pos_away_from_feeder(rng):
    while True:
        y = rng.randint(0, RESOLUTION - SQUARE_SIZE + 1)
        x = rng.randint(0, RESOLUTION - SQUARE_SIZE + 1)
        if is_away_from_feeder(y, x):
            return y, x


def feeder_pos(rng):
    y = rng.randint(FEEDER_TL[0], FEEDER_BR[0] - SQUARE_SIZE + 1)
    x = rng.randint(FEEDER_TL[1], FEEDER_BR[1] - SQUARE_SIZE + 1)
    return y, x


def far_apart(pa, pb, min_dist=40):
    return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) >= min_dist


def _split_transition_steady(n_frames, has_prev):
    """Return (transition_n, steady_n) for a segment of n_frames frames."""
    if has_prev:
        transition_n = min(TRANSITION_FRAMES, n_frames)
    else:
        transition_n = 0
    return transition_n, n_frames - transition_n


def _render_lerp(prev_pa, prev_pb, target_a, target_b, n):
    """Render n lerp frames from prev positions to target positions."""
    frames = []
    pa, pb = prev_pa, prev_pb
    for t in range(n):
        frac = (t + 1) / n
        pa = lerp_pos(prev_pa, target_a, frac)
        pb = lerp_pos(prev_pb, target_b, frac)
        frames.append(draw_frame(pa, pb))
    return frames, pa, pb


# ---- segment renderers ----
# Each renderer takes (rng, n_frames, prev_pa, prev_pb) and returns
# (frames, last_pa, last_pb). The first TRANSITION_FRAMES of n_frames are
# spent lerping from (prev_pa, prev_pb) to the segment's target start
# positions, except when prev positions are None (first segment of a video).

def render_copulation(rng, n_frames, prev_pa=None, prev_pb=None):
    while True:
        base_a = random_pos_away_from_feeder(rng)
        bb_y, bb_x = base_a[0], base_a[1] + SQUARE_SIZE + 1
        if bb_x + SQUARE_SIZE <= RESOLUTION and is_away_from_feeder(bb_y, bb_x):
            base_b = (bb_y, bb_x)
            break

    transition_n, steady_n = _split_transition_steady(n_frames, prev_pa is not None)
    frames = []
    pa, pb = base_a, base_b

    if transition_n:
        t_frames, pa, pb = _render_lerp(prev_pa, prev_pb, base_a, base_b, transition_n)
        frames.extend(t_frames)

    for _ in range(steady_n):
        pa = clamp_pos(base_a[0] + rng.randint(-1, 2), base_a[1] + rng.randint(-1, 2))
        pb = clamp_pos(base_b[0] + rng.randint(-1, 2), base_b[1] + rng.randint(-1, 2))
        frames.append(draw_frame(pa, pb))

    return frames, pa, pb


def render_playing(rng, n_frames, prev_pa=None, prev_pb=None):
    cy = rng.randint(60, 171)
    cx = rng.randint(60, 171)
    radius = 30
    phase_a = rng.uniform(0, 2 * np.pi)

    transition_n, steady_n = _split_transition_steady(n_frames, prev_pa is not None)
    omega = (1.5 * 2 * np.pi) / max(steady_n, 1)

    target_a = clamp_pos(int(round(cy + radius * np.sin(phase_a))),
                         int(round(cx + radius * np.cos(phase_a))))
    target_b = clamp_pos(int(round(cy + radius * np.sin(phase_a + np.pi))),
                         int(round(cx + radius * np.cos(phase_a + np.pi))))

    frames = []
    pa, pb = target_a, target_b

    if transition_n:
        t_frames, pa, pb = _render_lerp(prev_pa, prev_pb, target_a, target_b, transition_n)
        frames.extend(t_frames)

    for t in range(steady_n):
        ang_a = phase_a + omega * t
        ang_b = ang_a + np.pi
        pa = clamp_pos(int(round(cy + radius * np.sin(ang_a))),
                       int(round(cx + radius * np.cos(ang_a))))
        pb = clamp_pos(int(round(cy + radius * np.sin(ang_b))),
                       int(round(cx + radius * np.cos(ang_b))))
        frames.append(draw_frame(pa, pb))

    return frames, pa, pb


def _render_eats_sleeps(rng, n_frames, who_eats, prev_pa=None, prev_pb=None):
    eater = feeder_pos(rng)
    while True:
        sleeper = random_pos_away_from_feeder(rng)
        if far_apart(eater, sleeper):
            break

    if who_eats == "a":
        target_a, target_b = eater, sleeper
    else:
        target_a, target_b = sleeper, eater

    transition_n, steady_n = _split_transition_steady(n_frames, prev_pa is not None)
    frames = []
    pa, pb = target_a, target_b

    if transition_n:
        t_frames, pa, pb = _render_lerp(prev_pa, prev_pb, target_a, target_b, transition_n)
        frames.extend(t_frames)

    for _ in range(steady_n):
        eater_jit = clamp_pos(eater[0] + rng.randint(-1, 2),
                              eater[1] + rng.randint(-1, 2))
        if who_eats == "a":
            pa, pb = eater_jit, sleeper
        else:
            pa, pb = sleeper, eater_jit
        frames.append(draw_frame(pa, pb))

    return frames, pa, pb


def render_a_eats_b_sleeps(rng, n_frames, prev_pa=None, prev_pb=None):
    return _render_eats_sleeps(rng, n_frames, "a", prev_pa, prev_pb)


def render_a_sleeps_b_eats(rng, n_frames, prev_pa=None, prev_pb=None):
    return _render_eats_sleeps(rng, n_frames, "b", prev_pa, prev_pb)


def render_a_sleeps_b_sleeps(rng, n_frames, prev_pa=None, prev_pb=None):
    target_a = random_pos_away_from_feeder(rng)
    while True:
        target_b = random_pos_away_from_feeder(rng)
        if far_apart(target_a, target_b):
            break

    transition_n, steady_n = _split_transition_steady(n_frames, prev_pa is not None)
    frames = []
    pa, pb = target_a, target_b

    if transition_n:
        t_frames, pa, pb = _render_lerp(prev_pa, prev_pb, target_a, target_b, transition_n)
        frames.extend(t_frames)

    for _ in range(steady_n):
        frames.append(draw_frame(target_a, target_b))
        pa, pb = target_a, target_b

    return frames, pa, pb


def render_wandering(rng, n_frames, prev_pa=None, prev_pb=None):
    target_a = random_pos_away_from_feeder(rng)
    while True:
        target_b = random_pos_away_from_feeder(rng)
        if far_apart(target_a, target_b):
            break

    transition_n, steady_n = _split_transition_steady(n_frames, prev_pa is not None)
    frames = []
    pa, pb = target_a, target_b

    if transition_n:
        t_frames, pa, pb = _render_lerp(prev_pa, prev_pb, target_a, target_b, transition_n)
        frames.extend(t_frames)

    pa, pb = target_a, target_b
    for _ in range(steady_n):
        pa = clamp_pos(pa[0] + rng.randint(-3, 4), pa[1] + rng.randint(-3, 4))
        pb = clamp_pos(pb[0] + rng.randint(-3, 4), pb[1] + rng.randint(-3, 4))
        frames.append(draw_frame(pa, pb))

    return frames, pa, pb


SEGMENT_RENDERERS = {
    "copulation": render_copulation,
    "playing": render_playing,
    "a_eats_b_sleeps": render_a_eats_b_sleeps,
    "a_sleeps_b_eats": render_a_sleeps_b_eats,
    "a_sleeps_b_sleeps": render_a_sleeps_b_sleeps,
    "wandering": render_wandering,
}


# ---- video composition ----

def build_video_plan(rng, video_index):
    segments = list(REQUIRED_SEGMENTS)
    rng.shuffle(segments)
    if video_index in COPULATION_VIDEO_INDICES:
        insert_at = rng.randint(0, len(segments) + 1)
        segments.insert(insert_at, "copulation")
    lengths = [int(rng.randint(SEG_LEN_MIN, SEG_LEN_MAX + 1)) for _ in segments]
    return list(zip(segments, lengths))


def render_video(rng, plan):
    """Render a video and emit per-frame segment labels.

    Each segment after the first contributes TRANSITION_FRAMES of '_transition'
    labels (which map to 'other') followed by (length - TRANSITION_FRAMES)
    labels of the segment name. The first segment of the video has no
    transition.
    """
    frames = []
    seg_labels = []
    prev_pa, prev_pb = None, None

    for i, (seg_name, length) in enumerate(plan):
        renderer = SEGMENT_RENDERERS[seg_name]
        seg_frames, prev_pa, prev_pb = renderer(rng, length, prev_pa, prev_pb)
        frames.extend(seg_frames)

        if i == 0:
            seg_labels.extend([seg_name] * length)
        else:
            n_transition = min(TRANSITION_FRAMES, length)
            seg_labels.extend(["_transition"] * n_transition)
            seg_labels.extend([seg_name] * (length - n_transition))

    assert len(frames) == len(seg_labels), (len(frames), len(seg_labels))
    return frames, seg_labels


def write_intermediate_video(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FRAME_RATE, (RESOLUTION, RESOLUTION))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"cv2.VideoWriter produced empty file at {path}")


# ---- label JSONs ----

def labels_for(seg_labels, mode):
    if mode == "single":
        return [SEGMENT_TO_SINGLE[s] for s in seg_labels]
    n = len(MULTI_LABEL_CLASS_NAMES)
    out = []
    for s in seg_labels:
        row = [0] * n
        for idx in SEGMENT_TO_MULTI[s]:
            row[idx] = 1
        out.append(row)
    return out


def build_labels_json(seg_labels_per_video, video_filenames, mode):
    labels = {fn: labels_for(seg_labels_per_video[i], mode)
              for i, fn in enumerate(video_filenames)}
    splits = {k: [video_filenames[i] for i in v] for k, v in SPLITS.items()}
    class_names = SINGLE_LABEL_CLASS_NAMES if mode == "single" else MULTI_LABEL_CLASS_NAMES
    return {
        "class_names": class_names,
        "is_multilabel": mode == "multi",
        "labels": labels,
        "splits": splits,
    }


# ---- main ----

def main():
    script_dir = Path(__file__).resolve().parent
    fixtures_dir = script_dir / "fixtures"
    out_videos_dir = fixtures_dir / "videos"
    repo_root = script_dir.parent

    fixtures_dir.mkdir(parents=True, exist_ok=True)
    if out_videos_dir.exists():
        shutil.rmtree(out_videos_dir)

    rng = np.random.RandomState(SEED)
    seg_labels_per_video = []
    video_filenames = []

    with tempfile.TemporaryDirectory(prefix="feral_synth_") as tmp:
        tmp_dir = Path(tmp)
        print(f"Rendering {NUM_VIDEOS} intermediate videos...")
        for i in range(NUM_VIDEOS):
            plan = build_video_plan(rng, i)
            frames, seg_labels = render_video(rng, plan)
            fn = f"synthetic_{i:02d}.mp4"
            video_filenames.append(fn)
            seg_labels_per_video.append(seg_labels)
            write_intermediate_video(frames, tmp_dir / fn)
            seg_summary = ", ".join(s for s, _ in plan)
            print(f"  {fn}: {len(frames):5d} frames  [{seg_summary}]")

        print(f"\nRe-encoding via reencode_videos.py -> {out_videos_dir}")
        result = subprocess.run(
            [sys.executable, str(repo_root / "reencode_videos.py"),
             str(tmp_dir), str(out_videos_dir), "-p", "1"],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("reencode_videos.py failed")

    single_json = build_labels_json(seg_labels_per_video, video_filenames, "single")
    multi_json = build_labels_json(seg_labels_per_video, video_filenames, "multi")
    with open(fixtures_dir / "labels_singlelabel.json", "w") as f:
        json.dump(single_json, f)
    with open(fixtures_dir / "labels_multilabel.json", "w") as f:
        json.dump(multi_json, f)

    total_frames = sum(len(s) for s in seg_labels_per_video)
    print(f"\nDone. {NUM_VIDEOS} videos, {total_frames} total frames.")
    print(f"  videos:        {out_videos_dir}/")
    print(f"  single-label:  {fixtures_dir / 'labels_singlelabel.json'}")
    print(f"  multi-label:   {fixtures_dir / 'labels_multilabel.json'}")


if __name__ == "__main__":
    main()
