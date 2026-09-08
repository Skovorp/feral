"""Reject malformed annotation schemas before loading videos or training."""
import pytest

from feral import load_default_config, validate_labels_json


def valid():
    return {"class_names": {"0": "rest", "1": "walk"}, "is_multilabel": False,
            "labels": {"v.mp4": [0, 1]}, "splits": {"train": ["v.mp4"]}}


@pytest.mark.parametrize("value", [None, [], "labels.json"])
def test_top_level_requires_object(value):
    with pytest.raises(ValueError, match="must be an object"):
        validate_labels_json(value, None)


@pytest.mark.parametrize("key,value", [("splits", []), ("labels", []),
                                      ("class_names", {"00": "rest", "1": "walk"})])
def test_malformed_containers_raise_value_error(key, value):
    data = valid()
    data[key] = value
    with pytest.raises(ValueError):
        validate_labels_json(data, None)


@pytest.mark.parametrize("value", [[0, [], "bad"], [True, 1], [0, None]])
def test_invalid_single_labels_report_error(value):
    data = valid()
    data["labels"]["v.mp4"] = value
    with pytest.raises(ValueError, match="single-label IDs"):
        validate_labels_json(data, None)


@pytest.mark.parametrize("value", [float("nan"), 2, "1", None])
def test_multilabel_must_be_binary(value):
    data = valid()
    data["is_multilabel"] = True
    data["labels"]["v.mp4"] = [[0, value]]
    with pytest.raises(ValueError, match="binary"):
        validate_labels_json(data, None)


def test_reject_train_validation_overlap():
    data = valid()
    data["splits"]["val"] = ["v.mp4"]
    with pytest.raises(ValueError, match="overlaps splits"):
        validate_labels_json(data, None)


@pytest.mark.parametrize("videos", [["v.mp4", "v.mp4"], [[]], "v.mp4"])
def test_bad_split_rejected_before_filesystem_checks(videos, tmp_path):
    data = valid()
    data["splits"]["train"] = videos
    with pytest.raises(ValueError):
        validate_labels_json(data, tmp_path)


def test_inference_can_include_training_video():
    data = valid()
    data["splits"]["inference"] = ["v.mp4"]
    validate_labels_json(data, None)


def test_packaged_defaults_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_default_config()
    cfg['data']['prefix'] = 'modified'
    assert load_default_config()['data'].get('prefix') != 'modified'


def test_unlabeled_inference_video_must_be_readable(tmp_path):
    (tmp_path / 'bad.mp4').write_text('not a video')
    data = valid()
    data['labels'] = {}
    data['splits'] = {'inference': ['bad.mp4']}
    with pytest.raises(ValueError, match='cannot be decoded'):
        validate_labels_json(data, tmp_path)
