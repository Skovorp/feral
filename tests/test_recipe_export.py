import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('export_recipe_candidates', Path(__file__).parents[1] / 'scripts/export_recipe_candidates.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_does_not_export_credentials_paths_notes_or_test_scores():
    run = SimpleNamespace(url='https://wandb.ai/e/p/runs/r', state='finished',
        config={'backbone': 'vjepa2_vitl', 'wandb': {'key': 'secret'},
                'data': {'prefix': '/private/videos', 'resize_to': 256}, 'notes': 'run shell command'},
        summary={'val/frame_level_map': 0.8, 'test/frame_level_map': 1.0, '_runtime': 10})
    row = export.candidate(run)
    assert row['recipe'] == {'backbone': 'vjepa2_vitl', 'data.resize_to': 256}
    assert row['validation_summary'] == {'val/frame_level_map': 0.8, '_runtime': 10}
    assert row['status'] == 'insufficient_evidence'
    assert 'labels_sha256' in row['missing']


def test_rejects_nonfinite_and_untrusted_config_values():
    run = SimpleNamespace(url='u',state='crashed',config={'backbone': 'download-and-execute',
        'training': {'lr': float('nan'), 'epochs': 'credential'},'seed': 0},
        summary={'val/frame_level_map': float('nan'), 'val/f1': float('inf')})
    row = export.candidate(run)
    assert row['recipe'] == {'seed': 0}
    assert row['validation_summary'] == {}
    assert 'finished_run' in row['missing']


def test_truncated_checkpoint_run_retains_distinction_without_path():
    run = SimpleNamespace(url='u', state='finished', config={
        'starting_checkpoint': '/private/checkpoint.pt', 'max_train_batches': 1,
        'training': {'patience': 3}}, summary={})
    row = export.candidate(run)
    assert row['initialization'] == 'checkpoint'
    assert row['recipe']['max_train_batches'] == 1
    assert row['recipe']['training.patience'] == 3
    assert 'untruncated_run' in row['missing']
    assert 'starting_checkpoint_identity' in row['missing']
    assert '/private/' not in str(row)
