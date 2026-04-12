import pytest
import torch

from feral.dataset import collate_fn_val, collate_fn_inference
from feral.loops import _to_prob


# ===================================================================
# collate_fn_val
# ===================================================================

class TestCollateFnVal:
    def test_basic(self):
        batch = [
            (torch.ones(3, 4, 4), torch.tensor([1.0, 0.0]), [("a.mp4", 0, 0)]),
            (torch.ones(3, 4, 4) * 2, torch.tensor([0.0, 1.0]), [("b.mp4", 1, 0)]),
        ]
        tensors, targets, names = collate_fn_val(batch)
        assert tensors.shape == (2, 3, 4, 4)
        assert targets.shape == (2, 2)
        assert len(names) == 2

    def test_single_item(self):
        batch = [
            (torch.zeros(3, 2, 2), torch.tensor([1.0]), [("v.mp4", 0, 0)]),
        ]
        tensors, targets, names = collate_fn_val(batch)
        assert tensors.shape == (1, 3, 2, 2)


# ===================================================================
# collate_fn_inference
# ===================================================================

class TestCollateFnInference:
    def test_basic(self):
        batch = [
            (torch.ones(3, 4, 4), [("a.mp4", 0, 0)]),
            (torch.ones(3, 4, 4) * 2, [("b.mp4", 1, 0)]),
        ]
        tensors, names = collate_fn_inference(batch)
        assert tensors.shape == (2, 3, 4, 4)
        assert len(names) == 2

    def test_single_item(self):
        batch = [
            (torch.zeros(3, 2, 2), [("v.mp4", 0, 0)]),
        ]
        tensors, names = collate_fn_inference(batch)
        assert tensors.shape == (1, 3, 2, 2)


# ===================================================================
# _to_prob
# ===================================================================

class TestToProb:
    def test_multilabel_sigmoid(self):
        logits = torch.tensor([[0.0, 0.0]])
        probs = _to_prob(logits, is_multilabel=True)
        # sigmoid(0) = 0.5
        assert torch.allclose(probs, torch.tensor([[0.5, 0.5]]))

    def test_singlelabel_softmax(self):
        logits = torch.tensor([[100.0, 0.0]])
        probs = _to_prob(logits, is_multilabel=False)
        # softmax should put ~1.0 on first class
        assert probs[0, 0].item() > 0.99
        assert probs.sum().item() == pytest.approx(1.0)

    def test_softmax_sums_to_one(self):
        logits = torch.randn(4, 5)
        probs = _to_prob(logits, is_multilabel=False)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4))

    def test_sigmoid_in_range(self):
        logits = torch.randn(4, 5)
        probs = _to_prob(logits, is_multilabel=True)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
