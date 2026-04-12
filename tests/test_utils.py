import numpy as np
import pytest
import torch

from feral.utils import (
    get_class_frequencies,
    get_weights,
    last_nonzero_index,
    next_nonzero_index,
    prep_for_answers,
    validate_labels_json,
)


# ===================================================================
# last_nonzero_index / next_nonzero_index
# ===================================================================

class TestLastNonzeroIndex:
    def test_all_nonzero(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = last_nonzero_index(arr)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_all_zero(self):
        arr = np.array([0.0, 0.0, 0.0])
        result = last_nonzero_index(arr)
        np.testing.assert_array_equal(result, [-1, -1, -1])

    def test_gaps(self):
        arr = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        result = last_nonzero_index(arr)
        np.testing.assert_array_equal(result, [0, 0, 0, 3, 3, 3, 6, 6])

    def test_leading_zeros(self):
        arr = np.array([0.0, 0.0, 5.0])
        result = last_nonzero_index(arr)
        np.testing.assert_array_equal(result, [-1, -1, 2])

    def test_single_element_nonzero(self):
        arr = np.array([3.0])
        result = last_nonzero_index(arr)
        np.testing.assert_array_equal(result, [0])

    def test_single_element_zero(self):
        arr = np.array([0.0])
        result = last_nonzero_index(arr)
        np.testing.assert_array_equal(result, [-1])


class TestNextNonzeroIndex:
    def test_all_nonzero(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = next_nonzero_index(arr)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_all_zero(self):
        arr = np.array([0.0, 0.0, 0.0])
        result = next_nonzero_index(arr)
        np.testing.assert_array_equal(result, [-1, -1, -1])

    def test_gaps(self):
        arr = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        result = next_nonzero_index(arr)
        np.testing.assert_array_equal(result, [0, 3, 3, 3, 6, 6, 6, -1])

    def test_trailing_zeros(self):
        arr = np.array([5.0, 0.0, 0.0])
        result = next_nonzero_index(arr)
        np.testing.assert_array_equal(result, [0, -1, -1])

    def test_symmetry_with_last(self):
        """For a fully populated array, both functions should return identity."""
        arr = np.array([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(last_nonzero_index(arr), [0, 1, 2, 3])
        np.testing.assert_array_equal(next_nonzero_index(arr), [0, 1, 2, 3])


# ===================================================================
# get_class_frequencies
# ===================================================================

class TestGetClassFrequencies:
    def test_singlelabel_uniform(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        freqs = get_class_frequencies(labels, num_classes=3)
        np.testing.assert_allclose(freqs, [1 / 3, 1 / 3, 1 / 3])

    def test_singlelabel_skewed(self):
        labels = np.array([0, 0, 0, 1])
        freqs = get_class_frequencies(labels, num_classes=2)
        np.testing.assert_allclose(freqs, [0.75, 0.25])

    def test_singlelabel_missing_class(self):
        labels = np.array([0, 0, 0])
        freqs = get_class_frequencies(labels, num_classes=3)
        np.testing.assert_allclose(freqs, [1.0, 0.0, 0.0])

    def test_singlelabel_auto_num_classes(self):
        labels = np.array([0, 1, 2])
        freqs = get_class_frequencies(labels)
        assert len(freqs) == 3

    def test_multilabel(self):
        labels = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ])
        freqs = get_class_frequencies(labels)
        np.testing.assert_allclose(freqs, [0.5, 0.5, 0.5])

    def test_multilabel_skewed(self):
        labels = np.array([
            [1, 0],
            [1, 0],
            [1, 1],
            [1, 1],
        ])
        freqs = get_class_frequencies(labels)
        np.testing.assert_allclose(freqs, [1.0, 0.5])


# ===================================================================
# get_weights
# ===================================================================

class TestGetWeights:
    def _make_json(self, labels, is_multilabel=False):
        return {
            "labels": {"vid.mp4": labels},
            "splits": {"train": ["vid.mp4"]},
            "is_multilabel": is_multilabel,
        }

    def test_none_returns_none(self):
        json_data = self._make_json([0, 1, 0, 1])
        assert get_weights(json_data, None, "cpu") is None

    def test_invalid_weight_type(self):
        json_data = self._make_json([0, 1])
        with pytest.raises(AssertionError):
            get_weights(json_data, "invalid", "cpu")

    def test_inv_freq_singlelabel(self):
        # 3 of class 0, 1 of class 1 → freqs [0.75, 0.25] → inv [1.333, 4.0]
        json_data = self._make_json([0, 0, 0, 1])
        w = get_weights(json_data, "inv_freq", "cpu")
        assert isinstance(w, torch.Tensor)
        np.testing.assert_allclose(w.numpy(), [1 / 0.75, 1 / 0.25], rtol=1e-5)

    def test_inv_freq_sqrt_singlelabel(self):
        json_data = self._make_json([0, 0, 0, 1])
        w = get_weights(json_data, "inv_freq_sqrt", "cpu")
        expected = np.sqrt([1 / 0.75, 1 / 0.25])
        np.testing.assert_allclose(w.numpy(), expected, rtol=1e-5)

    def test_inv_freq_multilabel(self):
        # freqs [1.0, 0.5] → ratio [(1-1)/1, (1-0.5)/0.5] = [0, 1]
        labels = [[1, 0], [1, 0], [1, 1], [1, 1]]
        json_data = self._make_json(labels, is_multilabel=True)
        w = get_weights(json_data, "inv_freq", "cpu")
        np.testing.assert_allclose(w.numpy(), [0.0, 1.0], rtol=1e-5)

    def test_zero_freq_clamped(self):
        # Class 1 never appears → freq 0 → would be inf, should be clamped
        json_data = self._make_json([0, 0, 0, 0])
        json_data["labels"]["vid.mp4"] = [0, 0, 0, 0]
        w = get_weights(json_data, "inv_freq", "cpu")
        assert w.max().item() <= 1000000.0


# ===================================================================
# prep_for_answers
# ===================================================================

class TestPrepForAnswers:
    def test_with_names_and_targets(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        names = ["a", "b"]
        result = prep_for_answers(outputs, targets, names)
        assert len(result) == 2
        assert result[0][0] == "a"
        assert result[1][0] == "b"

    def test_with_names_no_targets(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        names = ["a", "b"]
        result = prep_for_answers(outputs, None, names)
        assert len(result) == 2
        assert len(result[0]) == 2  # (name, output)

    def test_no_names(self):
        outputs = torch.tensor([[0.1, 0.9]])
        targets = torch.tensor([1])
        result = prep_for_answers(outputs, targets, None)
        assert len(result) == 1
        assert len(result[0]) == 2  # (output, target)

    def test_nested_names_flattened(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        names = [["a", "b"]]  # nested list
        result = prep_for_answers(outputs, targets, names)
        assert result[0][0] == "a"
        assert result[1][0] == "b"

    def test_length_mismatch_raises(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1])
        with pytest.raises(AssertionError):
            prep_for_answers(outputs, targets, None)


# ===================================================================
# validate_labels_json
# ===================================================================

def _valid_singlelabel_json():
    return {
        "class_names": {"0": "other", "1": "walk", "2": "run"},
        "is_multilabel": False,
        "labels": {
            "vid1.mp4": [0, 1, 2, 0, 1],
            "vid2.mp4": [1, 1, 2],
        },
        "splits": {
            "train": ["vid1.mp4"],
            "val": ["vid2.mp4"],
        },
    }


def _valid_multilabel_json():
    return {
        "class_names": {"0": "groom", "1": "eat", "2": "sleep"},
        "is_multilabel": True,
        "labels": {
            "vid1.mp4": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "vid2.mp4": [[1, 1, 0], [0, 0, 1]],
        },
        "splits": {
            "train": ["vid1.mp4"],
            "val": ["vid2.mp4"],
        },
    }


class TestValidateLabelsJson:
    # --- valid cases pass without error ---

    def test_valid_singlelabel(self):
        validate_labels_json(_valid_singlelabel_json(), video_folder=None)

    def test_valid_multilabel(self):
        validate_labels_json(_valid_multilabel_json(), video_folder=None)

    # --- missing top-level keys ---

    def test_missing_top_level_key(self):
        data = _valid_singlelabel_json()
        del data["is_multilabel"]
        with pytest.raises(ValueError, match="missing required top-level keys"):
            validate_labels_json(data, video_folder=None)

    def test_missing_multiple_keys(self):
        data = {"class_names": {"0": "a"}}
        with pytest.raises(ValueError, match="missing required top-level keys"):
            validate_labels_json(data, video_folder=None)

    # --- is_multilabel ---

    def test_is_multilabel_not_bool(self):
        data = _valid_singlelabel_json()
        data["is_multilabel"] = "false"
        with pytest.raises(ValueError, match="must be a boolean"):
            validate_labels_json(data, video_folder=None)

    # --- class_names ---

    def test_class_names_empty(self):
        data = _valid_singlelabel_json()
        data["class_names"] = {}
        with pytest.raises(ValueError, match="non-empty dict"):
            validate_labels_json(data, video_folder=None)

    def test_class_names_non_integer_keys(self):
        data = _valid_singlelabel_json()
        data["class_names"] = {"a": "walk", "b": "run"}
        with pytest.raises(ValueError, match="integer strings"):
            validate_labels_json(data, video_folder=None)

    def test_class_names_non_sequential(self):
        data = _valid_singlelabel_json()
        data["class_names"] = {"0": "a", "2": "b"}
        with pytest.raises(ValueError, match="sequential starting from 0"):
            validate_labels_json(data, video_folder=None)

    # --- labels validation ---

    def test_singlelabel_bad_class_id(self):
        data = _valid_singlelabel_json()
        data["labels"]["vid1.mp4"] = [0, 1, 99]  # 99 out of range
        with pytest.raises(ValueError, match="single-label IDs must be ints"):
            validate_labels_json(data, video_folder=None)

    def test_singlelabel_negative_id(self):
        data = _valid_singlelabel_json()
        data["labels"]["vid1.mp4"] = [0, -1, 1]
        with pytest.raises(ValueError, match="single-label IDs must be ints"):
            validate_labels_json(data, video_folder=None)

    def test_multilabel_wrong_width(self):
        data = _valid_multilabel_json()
        data["labels"]["vid1.mp4"] = [[1, 0], [0, 1], [1, 0]]  # width 2 not 3
        with pytest.raises(ValueError, match="multilabel frames must each have 3 values"):
            validate_labels_json(data, video_folder=None)

    def test_empty_labels_list(self):
        data = _valid_singlelabel_json()
        data["labels"]["vid1.mp4"] = []
        with pytest.raises(ValueError, match="non-empty list"):
            validate_labels_json(data, video_folder=None)

    # --- splits validation ---

    def test_unknown_split_name(self):
        data = _valid_singlelabel_json()
        data["splits"]["foo"] = ["vid1.mp4"]
        with pytest.raises(ValueError, match="Unknown split names"):
            validate_labels_json(data, video_folder=None)

    def test_split_references_missing_video(self):
        data = _valid_singlelabel_json()
        data["splits"]["train"] = ["nonexistent.mp4"]
        with pytest.raises(ValueError, match="no entry in 'labels'"):
            validate_labels_json(data, video_folder=None)

    def test_empty_split(self):
        data = _valid_singlelabel_json()
        data["splits"]["train"] = []
        with pytest.raises(ValueError, match="empty"):
            validate_labels_json(data, video_folder=None)

    # --- video_folder frame count validation ---

    def test_missing_video_file(self, tmp_path):
        data = _valid_singlelabel_json()
        with pytest.raises(ValueError, match="file not found"):
            validate_labels_json(data, video_folder=str(tmp_path))

    # --- inference split doesn't need labels ---

    def test_inference_split_no_labels_ok(self):
        data = _valid_singlelabel_json()
        data["splits"]["inference"] = ["unlabeled.mp4"]
        # Should not raise — inference videos don't need label entries
        validate_labels_json(data, video_folder=None)
