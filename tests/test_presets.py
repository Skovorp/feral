import copy

import pytest

from feral.backbones import BACKBONES
from feral.presets import (
    MODE_HELP,
    PRESETS,
    apply_mode,
    infer_chunk_shift,
)


def _base_cfg():
    # Minimal stand-in for default_config.yaml with the fields presets touch.
    return {
        "backbone": "vjepa2_vitl_diving48",
        "predict_per_item": 64,
        "model": {
            "fc_drop_rate": 0.5,
            "class_weights": "inv_freq_sqrt",
            "freeze_encoder_layers": 12,
            "max_class_weight": None,
        },
        "data": {
            "chunk_length": 64,
            "chunk_shift": 32,
            "chunk_step": 1,
            "resize_to": 256,
            "resize_style": "square",
            "do_aa": True,
        },
        "training": {
            "epochs": 10,
            "lr": 4.0e-5,
            "compile": True,
            "grad_clip_norm": None,
        },
        "mixup_alpha": 0.8,
        "ema_decay": 0.999,
        "multilabel_threshold": 0.85,
    }


class TestApplyMode:
    def test_none_is_passthrough(self):
        cfg = _base_cfg()
        assert apply_mode(cfg, None) == cfg

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            apply_mode(_base_cfg(), "turbo")

    def test_does_not_mutate_input(self):
        cfg = _base_cfg()
        snapshot = copy.deepcopy(cfg)
        apply_mode(cfg, "fast")
        assert cfg == snapshot  # overlay returns a new dict, input untouched

    @pytest.mark.parametrize("mode", sorted(PRESETS))
    def test_backbone_is_a_real_registry_key(self, mode):
        out = apply_mode(_base_cfg(), mode)
        assert out["backbone"] in BACKBONES

    @pytest.mark.parametrize("mode", sorted(PRESETS))
    def test_resize_inherits_default(self, mode):
        # Presets no longer pin resize_to: the 384-native backbones are fed at
        # the default 256 (fewer tokens / faster) via interpolated pos-embeds.
        out = apply_mode(_base_cfg(), mode)
        assert out["data"]["resize_to"] == 256

    @pytest.mark.parametrize("mode", sorted(PRESETS))
    def test_deep_merge_keeps_untouched_keys(self, mode):
        out = apply_mode(_base_cfg(), mode)
        # 'seed'-like base keys not in any preset survive; and nested model dict
        # keeps base keys the overlay doesn't mention.
        assert "fc_drop_rate" in out["model"]
        assert out["data"]["chunk_step"] == 1

    def test_every_mode_has_help_text(self):
        assert set(MODE_HELP) == set(PRESETS)

    @pytest.mark.parametrize("mode", sorted(PRESETS))
    def test_predict_per_item_matches_chunk_length(self, mode):
        # The head emits predict_per_item tokens/item; targets are built per
        # chunk frame. A preset that changes chunk_length must keep them equal,
        # or loss shapes mismatch (regression: max once changed chunk_length
        # without pinning predict_per_item).
        out = apply_mode(_base_cfg(), mode)
        assert out["predict_per_item"] == out["data"]["chunk_length"]


class TestFast:
    def test_smallest_vjepa21_backbone(self):
        out = apply_mode(_base_cfg(), "fast")
        assert out["backbone"] == "vjepa2_1_vitb_384"

    def test_full_finetune_recipe(self):
        out = apply_mode(_base_cfg(), "fast")
        assert out["model"]["freeze_encoder_layers"] == 0
        assert out["training"]["lr"] == pytest.approx(4.0e-5)
        # fast is a plain full fine-tune: stabilization OFF.
        assert out["training"].get("grad_clip_norm") in (None,)
        assert out["model"].get("max_class_weight") in (None,)

    def test_ema_off(self):
        out = apply_mode(_base_cfg(), "fast")
        assert out["ema_decay"] is None

    def test_default_overlap_is_50pct(self):
        out = apply_mode(_base_cfg(), "fast")
        cl, cs = out["data"]["chunk_length"], out["data"]["chunk_shift"]
        assert cs == cl // 2


class TestMax:
    def test_vitl_backbone(self):
        out = apply_mode(_base_cfg(), "max")
        assert out["backbone"] == "vjepa2_1_vitl_384"

    def test_freezes_half_the_layers(self):
        out = apply_mode(_base_cfg(), "max")
        assert out["model"]["freeze_encoder_layers"] == 12

    def test_75pct_overlap(self):
        out = apply_mode(_base_cfg(), "max")
        cl, cs = out["data"]["chunk_length"], out["data"]["chunk_shift"]
        assert cs == cl // 4
        assert (1 - cs / cl) == pytest.approx(0.75)

    def test_ema_off(self):
        out = apply_mode(_base_cfg(), "max")
        assert out["ema_decay"] is None


class TestRare:
    def test_ema_and_mixup_off(self):
        out = apply_mode(_base_cfg(), "rare")
        assert out["ema_decay"] is None
        assert out["mixup_alpha"] is None

    def test_stabilization_on(self):
        out = apply_mode(_base_cfg(), "rare")
        assert out["training"]["grad_clip_norm"] == 1.0
        assert out["model"]["max_class_weight"] == 20

    def test_label_smoothing_off(self):
        out = apply_mode(_base_cfg(), "rare")
        assert out["training"]["label_smoothing"] == 0.0


class TestPublicAPI:
    def test_version_is_a_string(self):
        import feral
        assert isinstance(feral.__version__, str)

    def test_lazy_apply_mode_is_the_real_function(self):
        import feral
        from feral.presets import apply_mode as real
        assert feral.apply_mode is real

    def test_lazy_exports_resolve(self):
        import feral
        # Touch each public name; lazy __getattr__ should import + return it.
        for name in feral.__all__:
            assert getattr(feral, name) is not None

    def test_unknown_attr_raises(self):
        import feral
        with pytest.raises(AttributeError):
            feral.does_not_exist


class TestInferChunkShift:
    def test_fast_is_50pct(self):
        assert infer_chunk_shift("fast", 64) == 32

    def test_max_is_75pct(self):
        assert infer_chunk_shift("max", 64) == 16

    def test_rare_is_noop(self):
        assert infer_chunk_shift("rare", 64) is None

    def test_never_returns_zero(self):
        assert infer_chunk_shift("max", 2) == 1
