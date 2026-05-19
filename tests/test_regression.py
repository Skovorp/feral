"""Unit tests for the regression code path: shape sanity, labels JSON, metrics."""
import json
import os

import numpy as np
import pytest
import torch

from feral.dataset import ClsDataset
from feral.metrics import (
    calculate_regression_metrics,
    calculate_video_level_regression_metrics,
    ensemble_regression_predictions,
)
from feral.utils import is_classification, validate_labels_json


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def _load_regression_json():
    with open(os.path.join(FIXTURES_DIR, 'labels_regression.json')) as f:
        return json.load(f)


def test_is_classification_detects_regression():
    reg = _load_regression_json()
    assert is_classification(reg) is False


def test_is_classification_detects_classification():
    cls = {'class_names': {'0': 'a', '1': 'b'}, 'is_multilabel': False,
           'labels': {}, 'splits': {}}
    assert is_classification(cls) is True


def test_is_classification_explicit_task():
    assert is_classification({'task': 'classification'}) is True
    assert is_classification({'task': 'regression'}) is False


def test_validate_regression_labels_json_accepts_good():
    reg = _load_regression_json()
    # videos directory exists in fixtures so validation can run end-to-end
    validate_labels_json(reg, os.path.join(FIXTURES_DIR, 'videos'))


def test_validate_regression_labels_json_rejects_wrong_length():
    reg = _load_regression_json()
    # Mutate one label to have the wrong number of targets.
    first_fn = next(iter(reg['labels']))
    reg['labels'][first_fn] = [0.1]  # was length 2
    with pytest.raises(ValueError, match='regression labels'):
        validate_labels_json(reg, None)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(FIXTURES_DIR, 'videos'))
    or not os.listdir(os.path.join(FIXTURES_DIR, 'videos')),
    reason='Synthetic fixture videos not found.',
)
def test_regression_dataset_replicates_video_target_to_chunks():
    reg = _load_regression_json()
    ds = ClsDataset(
        partition='train',
        label_json_dict=reg,
        do_aa=False,
        predict_per_item=4,
        num_classes=0,
        prefix=os.path.join(FIXTURES_DIR, 'videos'),
        resize_to=64,
        resize_style='square',
        chunk_shift=32,
        chunk_length=4,
        chunk_step=1,
        task='regression',
        num_targets=2,
    )
    # Every chunk's target should equal that video's per-video target.
    for i, (fn, _frames) in enumerate(ds.samples):
        assert ds.labels[i] == reg['labels'][fn]


def test_regression_model_output_shape():
    """FeralModel(task='regression') should return (B, num_targets).

    Skipped if HF/torch.hub network access is not available — we use
    pretrained=False to avoid downloading weights, but the architecture
    config still needs to be fetched for HF backbones. Fall back gracefully.
    """
    from feral.model import FeralModel
    try:
        model = FeralModel(
            backbone='vjepa2_vitl_diving48',
            num_classes=0,
            predict_per_item=4,
            fc_drop_rate=0.0,
            freeze_encoder_layers=0,
            pretrained=False,
            task='regression',
            num_targets=3,
        ).eval()
    except Exception as e:
        pytest.skip(f'cannot construct backbone offline: {e}')
    B, T, C, H, W = 2, 8, 3, 256, 256
    x = torch.zeros(B, T, C, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, 3)


def test_calculate_regression_metrics_basic():
    target_names = {0: 'a', 1: 'b'}
    # 3 chunks, 2 targets, perfect predictions
    ans = [
        (('v1.mp4', 0), [1.0, 2.0], [1.0, 2.0]),
        (('v1.mp4', 1), [3.0, 4.0], [3.0, 4.0]),
        (('v2.mp4', 0), [5.0, 6.0], [5.0, 6.0]),
    ]
    res = calculate_regression_metrics(ans, target_names, prefix='val')
    assert res['val_mse'] == pytest.approx(0.0)
    assert res['val_mae'] == pytest.approx(0.0)
    assert res['val_corr'] == pytest.approx(1.0)


def test_calculate_regression_metrics_imperfect():
    target_names = {0: 'a'}
    ans = [
        (('v1.mp4', 0), [1.0], [2.0]),
        (('v1.mp4', 1), [2.0], [3.0]),
        (('v2.mp4', 0), [5.0], [5.0]),
    ]
    res = calculate_regression_metrics(ans, target_names, prefix='val')
    assert res['val_mse_a'] == pytest.approx((1 + 1 + 0) / 3)
    assert res['val_mae_a'] == pytest.approx((1 + 1 + 0) / 3)


def test_ensemble_regression_predictions_groups_by_video():
    ans = [
        (('v1.mp4', 0), [1.0, 2.0], [10.0, 20.0]),
        (('v1.mp4', 1), [3.0, 4.0], [10.0, 20.0]),
        (('v2.mp4', 0), [5.0, 5.0], [50.0, 50.0]),
    ]
    out = ensemble_regression_predictions(ans)
    np.testing.assert_allclose(out['v1.mp4'], [2.0, 3.0])
    np.testing.assert_allclose(out['v2.mp4'], [5.0, 5.0])


def test_video_level_regression_metrics():
    target_names = {0: 'a'}
    labels_json = {
        'splits': {'val': ['v1.mp4', 'v2.mp4']},
        'labels': {'v1.mp4': [2.0], 'v2.mp4': [5.0]},
    }
    ans = [
        (('v1.mp4', 0), [1.0], [2.0]),
        (('v1.mp4', 1), [3.0], [2.0]),  # avg = 2.0, perfect
        (('v2.mp4', 0), [5.0], [5.0]),  # perfect
    ]
    res = calculate_video_level_regression_metrics(ans, labels_json, 'val', target_names, 'val')
    assert res['val_vid_mse'] == pytest.approx(0.0)
    assert res['val_vid_mae'] == pytest.approx(0.0)
