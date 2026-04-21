import json
import os
import shutil
import subprocess
import tempfile

import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from feral.train import main as train_main

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def _build_smoke_cfg(label_json_name):
    with open(os.path.join(REPO_ROOT, 'feral', 'default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name'] = 'debug'
    cfg['max_batches'] = 1
    cfg.pop('wandb', None)  # disable wandb
    cfg['mixup_alpha'] = None
    cfg['ema_decay'] = None
    cfg['data']['prefix'] = os.path.join(FIXTURES_DIR, 'videos')
    cfg['data']['label_json'] = os.path.join(FIXTURES_DIR, label_json_name)
    cfg['training']['epochs'] = 1
    cfg['training']['train_bs'] = 1
    cfg['training']['val_bs'] = 1
    cfg['training']['num_workers'] = 0
    cfg['training']['compile'] = False
    cfg['training']['part_warmup'] = 0.0
    return cfg


videos_dir = os.path.join(FIXTURES_DIR, 'videos')
_skip_no_fixtures = pytest.mark.skipif(
    not os.path.isdir(videos_dir) or not os.listdir(videos_dir),
    reason=f"Synthetic fixture videos not found at {videos_dir}. Run: python tests/generate_synthetic_dataset.py",
)

_needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)


@_skip_no_fixtures
def test_singlelabel_smoke():
    train_main(_build_smoke_cfg('labels_singlelabel.json'))


@_skip_no_fixtures
def test_multilabel_smoke():
    train_main(_build_smoke_cfg('labels_multilabel.json'))


@_skip_no_fixtures
@pytest.mark.parametrize("resize_to,resize_style", [
    (192, "square"),
    (256, "rectangle"),
    (192, "rectangle"),
])
def test_smoke_resize_variants(resize_to, resize_style):
    """Run a single train+val+test+inference iteration under non-default
    resize configs. Exercises the rectangle code path and alternate
    resolutions end-to-end on the existing (square) fixtures."""
    cfg = _build_smoke_cfg('labels_singlelabel.json')
    cfg['data']['resize_to'] = resize_to
    cfg['data']['resize_style'] = resize_style
    cfg['run_name'] = f'smoke_{resize_style}_{resize_to}'
    train_main(cfg)


def _make_nonsquare_fixture(tmp_dir, width, height, n_frames):
    """Create a non-square test video + matching labels JSON + splits.

    Returns (prefix_dir, labels_json_path). All labels are class 0 ("other");
    we're only checking the pipeline runs — not learning anything."""
    import cv2
    prefix = os.path.join(tmp_dir, 'videos')
    os.makedirs(prefix, exist_ok=True)
    video_fn = 'nonsquare.mp4'
    video_path = os.path.join(prefix, video_fn)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"testsrc=duration={n_frames / 30:.3f}:size={width}x{height}:rate=30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
         video_path],
        check=True, capture_output=True,
    )
    cap = cv2.VideoCapture(video_path)
    try:
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    labels_json = {
        "class_names": {"0": "other", "1": "a", "2": "b"},
        "is_multilabel": False,
        "labels": {video_fn: [0] * actual_frames},
        "splits": {
            "train": [video_fn],
            "val": [video_fn],
            "test": [video_fn],
            "inference": [video_fn],
        },
    }
    labels_path = os.path.join(tmp_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(labels_json, f)
    return prefix, labels_path


@_needs_ffmpeg
def test_smoke_rectangle_nonsquare_video(tmp_path):
    """Full train+val+test+inference iteration with an actual non-square
    video under resize_style=rectangle. This is the real check that rectangle
    tensors survive the whole pipeline (dataset → loader → model → head)."""
    prefix, labels_path = _make_nonsquare_fixture(
        str(tmp_path), width=320, height=240, n_frames=80,
    )
    with open(os.path.join(REPO_ROOT, 'feral', 'default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name'] = 'smoke_rect_nonsquare'
    cfg['max_batches'] = 1
    cfg.pop('wandb', None)
    cfg['mixup_alpha'] = None
    cfg['ema_decay'] = None
    cfg['data']['prefix'] = prefix
    cfg['data']['label_json'] = labels_path
    cfg['data']['resize_to'] = 192
    cfg['data']['resize_style'] = 'rectangle'
    cfg['model']['class_weights'] = None  # dummy all-zero labels → can't use inv-freq
    # input is 320x240 -> rectangle/192 keeps aspect: (H,W) = (192, 256)
    cfg['training']['epochs'] = 1
    cfg['training']['train_bs'] = 1
    cfg['training']['val_bs'] = 1
    cfg['training']['num_workers'] = 0
    cfg['training']['compile'] = False
    cfg['training']['part_warmup'] = 0.0
    train_main(cfg)
