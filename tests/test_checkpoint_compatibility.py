"""Published checkpoint layouts must retain weights AND predictions.

History: fcf0a48/6938ab8 (0.1/0.2: HFModel + metadata), d0ba878
(embedded cfg, still model_name), da03845 (0.3 onward: backbone registry).
Use a real, tiny V-JEPA 2 encoder and the historical head on CPU; no downloads,
training, or regenerated video fixtures are needed.
"""
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path, PureWindowsPath, WindowsPath
import pickle
import shutil

import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoModel, VJEPA2Config, VJEPA2Model

from feral.backbones import BACKBONES
from feral.checkpoints import (
    checkpoint_metadata, checkpoint_state_dict, legacy_inference_config,
    normalize_checkpoint_config, read_checkpoint,
)
from feral.modeling import load_model_from_checkpoint


class HistoricalHFModel(nn.Module):
    """HFModel from 0.1/0.2, with dimensions reduced for a CPU regression test."""

    def __init__(self, encoder):
        super().__init__()
        self.model = encoder
        self.model.predictor = None
        self.clip_projector = nn.Module()
        pool = self.clip_projector
        pool.x_q = nn.Parameter(torch.randn(4, 32))
        pool.ln_q = nn.LayerNorm(32)
        pool.ln_x = nn.LayerNorm(32)
        pool.attn = nn.MultiheadAttention(32, 16, batch_first=True)
        self.fc_norm = nn.BatchNorm1d(32)
        self.fc_dropout = nn.Dropout(0.5)
        self.head = nn.Linear(32, 3)

    def forward(self, x):
        x = self.model(x, skip_predictor=True).last_hidden_state
        pool = self.clip_projector
        q = pool.ln_q(pool.x_q.unsqueeze(0).expand(x.size(0), -1, -1))
        kv = pool.ln_x(x)
        x, _ = pool.attn(q, kv, kv, need_weights=False)
        x = self.fc_norm(x.reshape(-1, 32))
        return self.head(self.fc_dropout(x))


@pytest.fixture
def historical_model(monkeypatch):
    config = VJEPA2Config(hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=4, mlp_ratio=2, crop_size=32,
                         frames_per_clip=4, pred_hidden_size=32,
                         pred_num_hidden_layers=1, pred_num_attention_heads=4)
    config._attn_implementation = 'eager'
    for entry in BACKBONES.values():
        if entry['source'] == 'hf':
            monkeypatch.setitem(entry, 'hidden_dim', 32)
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **k: deepcopy(config))
    monkeypatch.setattr(AutoModel, 'from_pretrained',
                        lambda *a, **k: pytest.fail('checkpoint load downloaded pretrained weights'))
    torch.manual_seed(12)
    model = HistoricalHFModel(VJEPA2Model(config)).eval()
    # Non-default BN statistics catch dropped buffers as well as parameters.
    model.fc_norm.running_mean.fill_(0.2)
    model.fc_norm.running_var.fill_(0.7)
    return model


def old_config():
    return {
        'model_name': 'facebook/vjepa2-vitl-fpc32-256-diving48',
        'predict_per_item': 4,
        'model': {'fc_drop_rate': 0.5, 'freeze_encoder_layers': 0},
        'data': {'resize_to': 32, 'chunk_length': 4, 'chunk_shift': 2, 'chunk_step': 1},
        'training': {'compile': False},
    }


# Each release is represented explicitly so new releases extend this contract.
RELEASES = ['bare', '0.1.0', '0.2.0', 'embedded-model-name', '0.3.0', '0.4.0',
            '1.0.0', '1.0.1', '1.0.2', '1.0.3', '1.0.4', '1.0.5']


def historical_checkpoint(model, version, prefix=''):
    cfg = old_config()
    modern = version not in ('bare', '0.1.0', '0.2.0', 'embedded-model-name')
    if modern:
        cfg.pop('model_name')
        cfg['backbone'] = 'vjepa2_vitl_diving48'
    state = OrderedDict()
    for key, value in model.state_dict().items():
        if modern and key.startswith('model.'):
            key = 'backbone.' + key
        state[prefix + key] = value
    if version == 'bare':
        return state, cfg
    raw = {'state_dict': state, 'class_names': {0: 'rest', 1: 'walk', 2: 'groom'},
           'is_multilabel': False}
    if version not in ('0.1.0', '0.2.0'):
        raw['cfg'] = cfg
    return raw, cfg


@pytest.mark.parametrize('version', RELEASES)
@pytest.mark.parametrize('prefix', ['', '_orig_mod.', 'module.', 'module._orig_mod.'])
def test_historical_weights_and_predictions(version, prefix, historical_model, tmp_path):
    raw, cfg = historical_checkpoint(historical_model, version, prefix)
    original_cfg = deepcopy(cfg)
    path = tmp_path / 'old.pt'
    torch.save(raw, path)
    loaded, metadata = load_model_from_checkpoint(cfg, 'cpu', path, num_classes=3)
    assert cfg == original_cfg
    assert not loaded.training
    assert (metadata is None) == (version == 'bare')
    x = torch.randn(2, 4, 3, 32, 32)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        torch.testing.assert_close(loaded(x), historical_model(x), rtol=0, atol=0)
    for key, value in historical_model.state_dict().items():
        target = 'backbone.' + key if key.startswith('model.') else key
        torch.testing.assert_close(loaded.state_dict()[target], value, rtol=0, atol=0)


@pytest.mark.parametrize('damage', ['missing', 'unexpected', 'shape', 'collision'])
def test_incompatible_weights_are_not_silently_ignored(damage, historical_model, tmp_path):
    raw, cfg = historical_checkpoint(historical_model, '0.1.0')
    state = raw['state_dict']
    if damage == 'missing':
        del state['head.weight']
    elif damage == 'unexpected':
        state['unknown.weight'] = torch.ones(1)
    elif damage == 'shape':
        state['head.weight'] = torch.ones(2, 32)
    else:
        key = next(key for key in state if key.startswith('model.'))
        state['backbone.' + key] = state[key]
    path = tmp_path / 'bad.pt'
    torch.save(raw, path)
    with pytest.raises((RuntimeError, ValueError), match='Missing key|Unexpected key|size mismatch|conflicting weights'):
        load_model_from_checkpoint(cfg, 'cpu', path)


@pytest.mark.parametrize('backbone', [k for k, v in BACKBONES.items() if v['source'] == 'hf'])
def test_legacy_model_name_aliases(backbone):
    cfg = old_config()
    cfg['model_name'] = BACKBONES[backbone]['slug']
    assert normalize_checkpoint_config(cfg)['backbone'] == backbone
    assert 'backbone' not in cfg


def test_unknown_or_conflicting_backbone_is_not_guessed():
    cfg = old_config()
    cfg['model_name'] = 'unknown/model'
    with pytest.raises(ValueError, match='Unsupported legacy model_name'):
        normalize_checkpoint_config(cfg)
    cfg = old_config()
    cfg['backbone'] = 'vjepa2_1_vitb_384'
    with pytest.raises(ValueError, match='conflicting'):
        normalize_checkpoint_config(cfg)


def test_modern_backbone_keys_and_module_versions_are_preserved():
    state = OrderedDict((key, torch.ones(1)) for key in [
        'backbone.model.blocks.0.attn.qkv.weight',  # V-JEPA 2.1
        'backbone.model.videoprism.spatial_encoder.x_layers.0.weight',
        'backbone._vp_mean', 'backbone._vp_std', 'fc_norm.num_batches_tracked',
    ])
    state._metadata = {'': {'version': 1}, 'fc_norm': {'version': 2}}
    loaded = checkpoint_state_dict(state)
    assert list(loaded) == list(state)
    assert loaded._metadata == state._metadata
    assert all(loaded[key] is value for key, value in state.items())


class SavedWindowsPath:
    def __reduce__(self):
        return WindowsPath, ('C:/videos/train.mp4',)


@pytest.mark.parametrize('value, expected', [
    (Path('/old/videos'), '/old/videos'),
    (PureWindowsPath('C:/videos'), 'C:\\videos'),
    (SavedWindowsPath(), 'C:\\videos\\train.mp4'),
])
def test_old_path_configs_load_safely_across_platforms(tmp_path, value, expected):
    path = tmp_path / 'paths.pt'
    cfg = old_config()
    cfg['data']['prefix'] = value
    torch.save({'cfg': cfg}, path)
    loaded = normalize_checkpoint_config(read_checkpoint(path)['cfg'])
    assert loaded['data']['prefix'] == expected


class UnapprovedObject:
    pass


def test_does_not_retry_with_unsafe_pickle(tmp_path):
    path = tmp_path / 'unsafe.pt'
    torch.save({'cfg': UnapprovedObject()}, path)
    with pytest.raises(pickle.UnpicklingError):
        read_checkpoint(path)


@pytest.mark.parametrize('is_multilabel', [False, True])
def test_bare_checkpoint_metadata_from_labels(is_multilabel):
    labels = {'class_names': {'0': 'rest'}, 'is_multilabel': is_multilabel}
    assert checkpoint_metadata({'head.weight': torch.ones(1, 32)}, labels) == {**labels, 'cfg': None}


def test_metadata_must_be_complete_and_consistent():
    raw = {'state_dict': {}, 'class_names': {0: 'rest'}}
    with pytest.raises(ValueError, match='is_multilabel'):
        checkpoint_metadata(raw)
    labels = {'class_names': {'0': 'rest'}, 'is_multilabel': False}
    assert checkpoint_metadata(raw, labels)['is_multilabel'] is False
    with pytest.raises(ValueError, match='disagrees'):
        checkpoint_metadata(raw, {**labels, 'class_names': {'0': 'walk'}})


def test_no_metadata_fails_before_model_construction(tmp_path, monkeypatch):
    import feral.inference_folder as inf
    path = tmp_path / 'bare.pt'
    torch.save({'head.weight': torch.ones(3, 32)}, path)
    monkeypatch.setattr(inf, 'load_model_from_checkpoint', lambda *a, **k: pytest.fail('built model'))
    with pytest.raises(ValueError, match='--labels-json'):
        inf.run_inference_folder(path, tmp_path)


@pytest.mark.parametrize('version', ['bare', '0.1.0', 'embedded-model-name', '1.0.4'])
def test_folder_inference_plumbs_legacy_metadata_and_config(version, historical_model, tmp_path, monkeypatch):
    import feral.inference_folder as inf
    raw, cfg = historical_checkpoint(historical_model, version)
    cfg['data']['resize_style'] = 'rectangle'
    cfg['data']['eval_chunk_shift'] = 1
    cfg['data']['eval_smoothing_window'] = 3
    original_cfg = deepcopy(cfg)
    path = tmp_path / 'model.pt'
    torch.save(raw, path)
    label_path = tmp_path / 'labels.json'
    label_path.write_text(json.dumps({'class_names': {'0': 'rest', '1': 'walk', '2': 'groom'}, 'is_multilabel': True}))
    monkeypatch.setattr(inf, 'find_videos', lambda *a: ['video.mp4'])
    captured = {}

    def load(cfg, device, checkpoint_path, **kwargs):
        captured['cfg'] = deepcopy(cfg)
        return load_model_from_checkpoint(cfg, 'cpu', checkpoint_path, **kwargs)

    def dataset(**kwargs):
        captured['dataset'] = kwargs
        return []

    class Writer:
        def __init__(self, *a, **kwargs):
            captured['writer'] = kwargs

        def close(self):
            captured['closed'] = True

    monkeypatch.setattr(inf, 'load_model_from_checkpoint', load)
    monkeypatch.setattr(inf, 'ClsDataset', dataset)
    monkeypatch.setattr(inf, 'InferenceWriter', Writer)
    monkeypatch.setattr(inf, 'run_inference', lambda *a, **kw: captured.update(inference=kw))
    # Any second deserialization would double host-memory use for the backbone.
    real_load = torch.load
    calls = []
    def read(*args, **kwargs):
        calls.append(args[0])
        return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', read)
    inf.run_inference_folder(path, tmp_path, output=str(tmp_path / 'result.json'), num_workers=0,
                             labels_json=label_path if version == 'bare' else None,
                             training_cfg=cfg if version in ('bare', '0.1.0') else {'bad': 'ignored'})
    assert len(calls) == 1
    assert captured['cfg']['backbone'] == 'vjepa2_vitl_diving48'
    assert captured['dataset']['resize_style'] == 'rectangle'
    assert captured['dataset']['chunk_shift'] == 1
    assert captured['writer']['smoothing_window'] == 3
    assert captured['inference']['is_multilabel'] == (version == 'bare')
    assert captured['closed']
    assert cfg == original_cfg


def test_legacy_defaults_are_frozen():
    cfg = legacy_inference_config()
    assert cfg['backbone'] == 'vjepa2_vitl_diving48'
    assert cfg['data'] == {'resize_to': 256, 'resize_style': 'square',
                           'chunk_length': 64, 'chunk_shift': 32, 'chunk_step': 1}


def test_legacy_checkpoint_video_predictions_match_original(historical_model, tmp_path, monkeypatch):
    """Exercise real decoding, normalization, forward passes, overlap and JSON output."""
    import feral.inference_folder as inf
    from feral.dataset import get_frame_count
    fixture = Path(__file__).parent / 'fixtures/videos/synthetic_00.mp4'
    if not fixture.is_file():
        pytest.skip('Existing video fixture is not installed')
    videos = tmp_path / 'videos'
    videos.mkdir()
    shutil.copyfile(fixture, videos / fixture.name)
    raw, cfg = historical_checkpoint(historical_model, 'embedded-model-name')
    checkpoint = tmp_path / 'old.pt'
    torch.save(raw, checkpoint)

    # Folder inference is CUDA-only in production. Redirect device selection,
    # while exercising the real model loader and inference loop on CPU here.
    real_inference = inf.run_inference
    def inference_cpu(*args, **kwargs):
        kwargs['device'] = 'cpu'
        with sdpa_kernel(SDPBackend.MATH):
            return real_inference(*args, **kwargs)
    monkeypatch.setattr(inf, 'run_inference', inference_cpu)
    monkeypatch.setattr(inf, 'load_model_from_checkpoint',
                        lambda cfg, device, path, **kw: load_model_from_checkpoint(cfg, 'cpu', path, **kw))
    migrated_output = tmp_path / 'migrated.json'
    inf.run_inference_folder(checkpoint, videos, output=str(migrated_output), batch_size=8, num_workers=0)

    monkeypatch.setattr(inf, 'load_model_from_checkpoint',
                        lambda *a, **kw: (historical_model, checkpoint_metadata(raw)))
    reference_output = tmp_path / 'reference.json'
    inf.run_inference_folder(checkpoint, videos, output=str(reference_output), batch_size=8, num_workers=0)
    actual = json.loads(migrated_output.read_text())
    assert actual == json.loads(reference_output.read_text())
    predictions = torch.tensor(actual['preds'][fixture.name])
    assert predictions.shape == (get_frame_count(str(fixture)), 3)
    assert torch.isfinite(predictions).all()
    torch.testing.assert_close(predictions.sum(dim=1), torch.ones(len(predictions)))
