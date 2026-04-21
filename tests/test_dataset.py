import pytest
import torch

from feral.dataset import build_resize_transform, collate_fn_val, collate_fn_inference
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


# ===================================================================
# build_resize_transform
# ===================================================================

class TestBuildResizeTransform:
    def test_square_on_landscape_squishes(self):
        t = build_resize_transform(resize_to=128, resize_style="square")
        video = torch.zeros(4, 3, 240, 480)  # (T, C, H, W)
        out = t(video)
        assert out.shape == (4, 3, 128, 128)

    def test_square_on_portrait_squishes(self):
        t = build_resize_transform(resize_to=64, resize_style="square")
        video = torch.zeros(4, 3, 480, 240)
        out = t(video)
        assert out.shape == (4, 3, 64, 64)

    def test_rectangle_preserves_aspect_landscape(self):
        t = build_resize_transform(resize_to=128, resize_style="rectangle")
        video = torch.zeros(4, 3, 240, 480)
        out = t(video)
        # Smallest side (H=240) becomes 128, width scales: 480 * 128/240 = 256
        assert out.shape == (4, 3, 128, 256)

    def test_rectangle_preserves_aspect_portrait(self):
        t = build_resize_transform(resize_to=128, resize_style="rectangle")
        video = torch.zeros(4, 3, 480, 240)
        out = t(video)
        # Smallest side (W=240) becomes 128, height scales: 480 * 128/240 = 256
        assert out.shape == (4, 3, 256, 128)

    def test_rectangle_on_square_stays_square(self):
        t = build_resize_transform(resize_to=96, resize_style="rectangle")
        video = torch.zeros(4, 3, 256, 256)
        out = t(video)
        assert out.shape == (4, 3, 96, 96)

    def test_rectangle_does_upscale(self):
        """Dataset-side resize is expected to always match the target size
        (even upscaling). Downscale-only is handled at reencode time."""
        t = build_resize_transform(resize_to=512, resize_style="rectangle")
        video = torch.zeros(4, 3, 240, 320)
        out = t(video)
        # Smallest side (H=240) becomes 512, width scales: 320 * 512/240 ≈ 682
        assert out.shape[:3] == (4, 3, 512)
        assert out.shape[3] in (682, 683)

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="resize_style"):
            build_resize_transform(resize_to=128, resize_style="circle")
