"""Diagnostics must remain machine-readable on invalid inputs and CPU hosts."""
import json
from types import SimpleNamespace

from feral import diagnostics
from feral.cli import main
import pytest


def test_validate_invalid_json(tmp_path, capsys, monkeypatch):
    labels = tmp_path / 'labels.json'
    labels.write_text('{')
    monkeypatch.setattr('sys.argv', ['feral', 'validate', str(tmp_path), str(labels), '--json'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert not report['ok']
    assert 'JSONDecodeError' in report['errors'][0]


def test_validation_summary(tmp_path, monkeypatch):
    import feral.utils
    monkeypatch.setattr(feral.utils, 'validate_labels_json', lambda *args: None)
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps({'class_names': {'0': 'rest', '1': 'run'},
                               'is_multilabel': False, 'labels': {'a.mp4': [0, 1, 1]},
                               'splits': {'train': ['a.mp4'], 'inference': ['unlabeled.mp4']}}))
    report = diagnostics.validate_dataset(str(tmp_path), str(path))
    assert report['ok']
    assert report['splits']['train'] == {'videos': 1, 'labeled_frames': 3,
                                         'class_positive_frames': {'0': 1, '1': 2}}
    assert report['splits']['inference']['labeled_frames'] == 0
    assert len(report['labels_sha256']) == 64
    assert 'labels' not in report


def test_doctor_cpu_and_import_failure(monkeypatch):
    def module(name):
        if name == 'decord':
            raise ImportError('missing decoder')
        if name == 'torch':
            return SimpleNamespace(__version__='test', version=SimpleNamespace(cuda=None),
                                   cuda=SimpleNamespace(device_count=lambda: 0, is_available=lambda: False))
        return SimpleNamespace(__version__='test')
    monkeypatch.setattr(diagnostics.importlib, 'import_module', module)
    report = diagnostics.doctor()
    assert not report['ok']
    assert len(report['errors']) == 2
    assert report['gpus'] == []


@pytest.mark.parametrize('major,bf16,expected_ok', [(7, True, False), (8, False, False), (8, True, True)])
def test_doctor_checks_current_gpu_capability(monkeypatch, major, bf16, expected_ok):
    cuda = SimpleNamespace(device_count=lambda: 1, is_available=lambda: True,
                           current_device=lambda: 0, get_device_capability=lambda index: (major, 5),
                           is_bf16_supported=lambda: bf16,
                           get_device_properties=lambda index: SimpleNamespace(name='GPU', total_memory=16_000_000_000, major=major, minor=5))
    monkeypatch.setattr(diagnostics.importlib, 'import_module', lambda name: SimpleNamespace(
        __version__='test', cuda=cuda, version=SimpleNamespace(cuda='12.4')))
    report = diagnostics.doctor()
    assert report['ok'] is expected_ok
    if not expected_ok:
        assert 'T4 and V100 are unsupported' in report['errors'][0]
