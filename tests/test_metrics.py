import numpy as np
import pytest

from feral.metrics import (
    calc_frame_level_map,
    calculate_f1_metrics,
    ensemble_predictions,
    generate_empty_logits,
    postprocess_predictions,
    smooth_per_frame_probs,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal labels_json fixtures
# ---------------------------------------------------------------------------

def _singlelabel_labels_json(n_classes=3, n_frames=10):
    """Single-label: labels[fn] is a flat list of ints."""
    class_names = {str(i): f"cls{i}" for i in range(n_classes)}
    labels = {"vid.mp4": list(np.random.randint(0, n_classes, size=n_frames))}
    return {
        "class_names": class_names,
        "labels": labels,
        "splits": {"val": ["vid.mp4"]},
    }


def _multilabel_labels_json(n_classes=3, n_frames=10):
    """Multi-label: labels[fn] is a (n_frames, n_classes) binary matrix."""
    class_names = {str(i): f"cls{i}" for i in range(n_classes)}
    lab = np.zeros((n_frames, n_classes))
    for i in range(n_frames):
        lab[i, i % n_classes] = 1
    labels = {"vid.mp4": lab.tolist()}
    return {
        "class_names": class_names,
        "labels": labels,
        "splits": {"val": ["vid.mp4"]},
    }


# ===================================================================
# ensemble_predictions
# ===================================================================

class TestEnsemblePredictions:
    def test_single_chunk_identity(self):
        """With one prediction per frame the ensemble should return the original logits."""
        logits = {"a.mp4": np.zeros((5, 3))}
        raw = np.array([[0.1, 0.2, 0.7],
                        [0.3, 0.4, 0.3],
                        [0.5, 0.3, 0.2],
                        [0.0, 1.0, 0.0],
                        [0.2, 0.2, 0.6]])
        ans = []
        for i in range(5):
            ans.append((("a.mp4", i, 0), raw[i].tolist()))

        result = ensemble_predictions(ans, logits)
        np.testing.assert_allclose(result["a.mp4"], raw)

    def test_two_chunks_average(self):
        """Two chunks predicting the same frame should be averaged."""
        logits = {"a.mp4": np.zeros((3, 2))}
        ans = [
            (("a.mp4", 1, 0), [1.0, 0.0]),
            (("a.mp4", 1, 1), [0.0, 1.0]),
        ]
        result = ensemble_predictions(ans, logits)
        np.testing.assert_allclose(result["a.mp4"][1], [0.5, 0.5])

    def test_gap_interpolation(self):
        """Frames with no predictions between two predicted frames should be interpolated."""
        logits = {"a.mp4": np.zeros((5, 2))}
        # Only predict frames 0 and 4
        ans = [
            (("a.mp4", 0, 0), [1.0, 0.0]),
            (("a.mp4", 4, 0), [0.0, 1.0]),
        ]
        result = ensemble_predictions(ans, logits)
        # Frame 0 and 4 are exact
        np.testing.assert_allclose(result["a.mp4"][0], [1.0, 0.0])
        np.testing.assert_allclose(result["a.mp4"][4], [0.0, 1.0])
        # Frame 2 is midpoint — equal inverse-distance weighting
        np.testing.assert_allclose(result["a.mp4"][2], [0.5, 0.5])

    def test_gap_left_only(self):
        """Frames after the last prediction copy from the left neighbor."""
        logits = {"a.mp4": np.zeros((3, 2))}
        ans = [
            (("a.mp4", 0, 0), [0.8, 0.2]),
        ]
        result = ensemble_predictions(ans, logits)
        np.testing.assert_allclose(result["a.mp4"][0], [0.8, 0.2])
        # Frame 1 and 2 have no right neighbor, should copy left
        np.testing.assert_allclose(result["a.mp4"][1], [0.8, 0.2])
        np.testing.assert_allclose(result["a.mp4"][2], [0.8, 0.2])

    def test_gap_right_only(self):
        """Frames before the first prediction copy from the right neighbor."""
        logits = {"a.mp4": np.zeros((3, 2))}
        ans = [
            (("a.mp4", 2, 0), [0.3, 0.7]),
        ]
        result = ensemble_predictions(ans, logits)
        np.testing.assert_allclose(result["a.mp4"][0], [0.3, 0.7])
        np.testing.assert_allclose(result["a.mp4"][1], [0.3, 0.7])
        np.testing.assert_allclose(result["a.mp4"][2], [0.3, 0.7])

    def test_multiple_files(self):
        """Ensemble should work across multiple files independently."""
        logits = {
            "a.mp4": np.zeros((2, 2)),
            "b.mp4": np.zeros((2, 2)),
        }
        ans = [
            (("a.mp4", 0, 0), [1.0, 0.0]),
            (("a.mp4", 1, 0), [0.0, 1.0]),
            (("b.mp4", 0, 0), [0.5, 0.5]),
            (("b.mp4", 1, 0), [0.3, 0.7]),
        ]
        result = ensemble_predictions(ans, logits)
        np.testing.assert_allclose(result["a.mp4"][0], [1.0, 0.0])
        np.testing.assert_allclose(result["b.mp4"][1], [0.3, 0.7])


# ===================================================================
# smooth_per_frame_probs / eval_smoothing_window
# ===================================================================

class TestSmoothPerFrameProbs:
    def test_none_window_is_identity(self):
        arr = np.array([[0.0], [1.0], [0.0]])
        np.testing.assert_array_equal(smooth_per_frame_probs(arr, None), arr)

    def test_window_one_is_identity(self):
        arr = np.array([[0.0], [1.0], [0.0]])
        np.testing.assert_array_equal(smooth_per_frame_probs(arr, 1), arr)

    def test_window_geq_length_is_noop(self):
        # A window >= T would collapse every frame to the video-wide mean; we
        # treat it as a no-op instead.
        arr = np.array([[0.0], [1.0], [0.0]])
        np.testing.assert_array_equal(smooth_per_frame_probs(arr, 5), arr)

    def test_centered_average_spike(self):
        # A single spike spreads into its 3-frame neighborhood; edges divide by
        # the number of real frames in the (truncated) window.
        arr = np.array([[0.0], [0.0], [3.0], [0.0], [0.0]])
        out = smooth_per_frame_probs(arr, 3)
        np.testing.assert_allclose(out[:, 0], [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_constant_signal_preserved_at_boundaries(self):
        # A constant signal must stay constant everywhere — boundary truncation
        # normalizes by the count of contributing frames, not the window size.
        arr = np.full((4, 1), 2.0)
        out = smooth_per_frame_probs(arr, 3)
        np.testing.assert_allclose(out[:, 0], [2.0, 2.0, 2.0, 2.0])

    def test_channels_smoothed_independently(self):
        arr = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 3.0], [1.0, 0.0], [1.0, 0.0]])
        out = smooth_per_frame_probs(arr, 3)
        np.testing.assert_allclose(out[:, 0], [1.0, 1.0, 1.0, 1.0, 1.0])  # constant channel untouched
        np.testing.assert_allclose(out[:, 1], [0.0, 1.0, 1.0, 1.0, 0.0])  # spike channel smoothed


class TestPostprocessPredictions:
    def test_smooths_ensembled_matrix(self):
        preds = {"a.mp4": np.array([[0.0], [0.0], [3.0], [0.0], [0.0]])}
        result = postprocess_predictions(preds, smoothing_window=3)
        np.testing.assert_allclose(result["a.mp4"][:, 0], [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_chains_with_ensemble(self):
        logits = {"a.mp4": np.zeros((5, 1))}
        raw = [0.0, 0.0, 3.0, 0.0, 0.0]
        ans = [(("a.mp4", i, 0), [raw[i]]) for i in range(5)]
        result = postprocess_predictions(ensemble_predictions(ans, logits), smoothing_window=3)
        np.testing.assert_allclose(result["a.mp4"][:, 0], [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_does_not_cross_video_boundary(self):
        preds = {
            "a.mp4": np.full((3, 1), 1.0),
            "b.mp4": np.full((3, 1), 5.0),
        }
        result = postprocess_predictions(preds, smoothing_window=3)
        # Each video keeps its own constant level — no bleed across the boundary.
        np.testing.assert_allclose(result["a.mp4"][:, 0], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(result["b.mp4"][:, 0], [5.0, 5.0, 5.0])

    def test_none_window_leaves_matrix_untouched(self):
        preds = {"a.mp4": np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])}
        original = preds["a.mp4"].copy()
        result = postprocess_predictions(preds, smoothing_window=None)
        np.testing.assert_array_equal(result["a.mp4"], original)


# ===================================================================
# calc_frame_level_map
# ===================================================================

class TestCalcFrameLevelMap:
    def test_perfect_singlelabel(self):
        """Perfect predictions should give mAP = 1.0."""
        n_classes, n_frames = 3, 30
        labels_json = _singlelabel_labels_json(n_classes, n_frames)
        targets = np.array(labels_json["labels"]["vid.mp4"])

        # Build perfect logits: high score on the correct class
        logits = {"vid.mp4": np.zeros((n_frames, n_classes))}
        for i in range(n_frames):
            logits["vid.mp4"][i, targets[i]] = 10.0

        result = calc_frame_level_map(logits, labels_json, "val")
        assert result == pytest.approx(1.0)

    def test_perfect_multilabel(self):
        """Perfect predictions for multi-label should give mAP = 1.0."""
        n_classes, n_frames = 3, 30
        labels_json = _multilabel_labels_json(n_classes, n_frames)
        targets = np.array(labels_json["labels"]["vid.mp4"])

        logits = {"vid.mp4": np.full((n_frames, n_classes), -10.0)}
        for i in range(n_frames):
            for c in range(n_classes):
                if targets[i, c] == 1:
                    logits["vid.mp4"][i, c] = 10.0

        result = calc_frame_level_map(logits, labels_json, "val")
        assert result == pytest.approx(1.0)

    def test_other_excluded_from_map(self):
        """The 'other' class AP should not be included in mAP average."""
        n_frames = 30
        class_names = {"0": "walk", "1": "other"}
        # All frames are class 0
        labels_json = {
            "class_names": class_names,
            "labels": {"vid.mp4": [0] * n_frames},
            "splits": {"val": ["vid.mp4"]},
        }
        # Perfect predictions for class 0, garbage for class 1
        logits = {"vid.mp4": np.zeros((n_frames, 2))}
        logits["vid.mp4"][:, 0] = 10.0

        result = calc_frame_level_map(logits, labels_json, "val")
        # mAP should be 1.0 since only class 0 counts
        assert result == pytest.approx(1.0)


# ===================================================================
# calculate_f1_metrics
# ===================================================================

class TestCalculateF1Metrics:
    def test_perfect_singlelabel(self):
        n_classes, n_frames = 3, 30
        labels_json = _singlelabel_labels_json(n_classes, n_frames)
        targets = np.array(labels_json["labels"]["vid.mp4"])

        logits = {"vid.mp4": np.zeros((n_frames, n_classes))}
        for i in range(n_frames):
            logits["vid.mp4"][i, targets[i]] = 10.0

        res = calculate_f1_metrics(logits, labels_json, "val", is_multilabel=False, prefix="test", multilabel_threshold=0.5)
        assert res["test/f1"] == pytest.approx(1.0)
        assert res["test/accuracy"] == pytest.approx(1.0)
        assert res["test/precision"] == pytest.approx(1.0)
        assert res["test/recall"] == pytest.approx(1.0)

    def test_perfect_multilabel(self):
        n_classes, n_frames = 3, 30
        labels_json = _multilabel_labels_json(n_classes, n_frames)
        targets = np.array(labels_json["labels"]["vid.mp4"])

        logits = {"vid.mp4": np.full((n_frames, n_classes), -10.0)}
        for i in range(n_frames):
            for c in range(n_classes):
                if targets[i, c] == 1:
                    logits["vid.mp4"][i, c] = 10.0

        res = calculate_f1_metrics(logits, labels_json, "val", is_multilabel=True, prefix="test", multilabel_threshold=0.0)
        assert res["test/f1"] == pytest.approx(1.0)
        assert res["test/accuracy"] == pytest.approx(1.0)

    def test_all_wrong_singlelabel(self):
        """Completely wrong predictions should give f1 = 0."""
        n_classes, n_frames = 2, 20
        labels_json = {
            "class_names": {"0": "a", "1": "b"},
            "labels": {"vid.mp4": [0] * n_frames},
            "splits": {"val": ["vid.mp4"]},
        }
        # Predict class 1 for everything
        logits = {"vid.mp4": np.zeros((n_frames, n_classes))}
        logits["vid.mp4"][:, 1] = 10.0

        res = calculate_f1_metrics(logits, labels_json, "val", is_multilabel=False, prefix="t", multilabel_threshold=0.5)
        assert res["t/f1"] == pytest.approx(0.0)
        assert res["t/accuracy"] == pytest.approx(0.0)

    def test_per_class_f1_keys(self):
        """Result dict should contain per-class f1 keys for non-other classes."""
        n_classes, n_frames = 3, 30
        labels_json = _singlelabel_labels_json(n_classes, n_frames)
        targets = np.array(labels_json["labels"]["vid.mp4"])
        logits = {"vid.mp4": np.zeros((n_frames, n_classes))}
        for i in range(n_frames):
            logits["vid.mp4"][i, targets[i]] = 10.0

        res = calculate_f1_metrics(logits, labels_json, "val", is_multilabel=False, prefix="v", multilabel_threshold=0.5)
        for i in range(n_classes):
            assert f"v/f1_cls{i}" in res

    def test_other_excluded_from_f1(self):
        """'other' class should be excluded from macro F1."""
        n_frames = 20
        labels_json = {
            "class_names": {"0": "walk", "1": "other"},
            "labels": {"vid.mp4": [0] * n_frames},
            "splits": {"val": ["vid.mp4"]},
        }
        logits = {"vid.mp4": np.zeros((n_frames, 2))}
        logits["vid.mp4"][:, 0] = 10.0

        res = calculate_f1_metrics(logits, labels_json, "val", is_multilabel=False, prefix="t", multilabel_threshold=0.5)
        # Only class 0 ("walk") counts, and it's perfect
        assert res["t/f1"] == pytest.approx(1.0)


# ===================================================================
# generate_empty_logits
# ===================================================================

class TestGenerateEmptyLogits:
    def test_shape(self):
        labels_json = _singlelabel_labels_json(n_classes=4, n_frames=7)
        logits = generate_empty_logits(labels_json, "val")
        assert logits["vid.mp4"].shape == (7, 4)
        np.testing.assert_array_equal(logits["vid.mp4"], 0.0)
