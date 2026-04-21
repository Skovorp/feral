"""Forward-pass shape test for every entry in BACKBONES.

Builds one real train batch per unique backbone `img_size` (from the synthetic
video fixtures), then runs `FeralModel(backbone=..., pretrained=False)(batch)`
for each registered key and asserts the output shape.

We use `pretrained=False` so this test is purely a wiring check — confirms the
registry, adapter branching (HF vs torch.hub), hidden_dim plumbing, input
transpose, and freeze logic all line up across every size. Downloading real
weights for all seven variants would be ~15GB; the shape test doesn't need them.
"""
import gc
import json
import os

import pytest
import torch
import yaml

from feral.backbones import BACKBONES
from feral.data import build_datasets_and_loaders
from feral.model import FeralModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
VIDEOS_DIR = os.path.join(FIXTURES_DIR, 'videos')
LABELS_JSON = os.path.join(FIXTURES_DIR, 'labels_singlelabel.json')

_skip_no_fixtures = pytest.mark.skipif(
    not os.path.isdir(VIDEOS_DIR) or not os.listdir(VIDEOS_DIR),
    reason=f"Synthetic fixture videos not found at {VIDEOS_DIR}. "
           f"Run: python tests/generate_synthetic_dataset.py",
)


def _load_test_cfg(img_size):
    with open(os.path.join(REPO_ROOT, 'feral', 'default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['data']['prefix'] = VIDEOS_DIR
    cfg['data']['label_json'] = LABELS_JSON
    cfg['data']['resize_to'] = img_size
    cfg['training']['train_bs'] = 1
    cfg['training']['val_bs'] = 1
    cfg['training']['num_workers'] = 0
    cfg['training']['compile'] = False
    cfg['model']['freeze_encoder_layers'] = 0
    return cfg


@pytest.fixture(scope="module")
def train_batches():
    """One (x, y, num_classes, predict_per_item) tuple per unique img_size."""
    with open(LABELS_JSON) as f:
        labels_json = json.load(f)
    num_classes = len(labels_json['class_names'])

    unique_sizes = sorted({v['img_size'] for v in BACKBONES.values()})
    cache = {}
    for img_size in unique_sizes:
        cfg = _load_test_cfg(img_size)
        _datasets, loaders = build_datasets_and_loaders(cfg, labels_json, num_classes)
        x, y = next(iter(loaders['train']))
        cache[img_size] = (x, y, num_classes, cfg['predict_per_item'])
    return cache


@pytest.fixture
def cuda_cleanup():
    """Free each model's GPU memory between parametrized cases — pytest
    doesn't GC across parametrize iterations, so without this the allocator
    accumulates until OOM."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@_skip_no_fixtures
@pytest.mark.parametrize("backbone_key", sorted(BACKBONES))
def test_backbone_forward(backbone_key, train_batches, cuda_cleanup):
    """Build FeralModel with `backbone_key`, run a forward pass on a real batch,
    check the output shape is (B * predict_per_item, num_classes)."""
    entry = BACKBONES[backbone_key]
    x_cpu, _y, num_classes, predict_per_item = train_batches[entry['img_size']]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FeralModel(
        backbone=backbone_key,
        num_classes=num_classes,
        predict_per_item=predict_per_item,
        fc_drop_rate=0.0,
        freeze_encoder_layers=0,
        pretrained=False,
    ).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x_cpu.to(device))

    B = x_cpu.shape[0]
    assert out.shape == (B * predict_per_item, num_classes), (
        f"{backbone_key}: expected {(B * predict_per_item, num_classes)}, "
        f"got {tuple(out.shape)}"
    )
