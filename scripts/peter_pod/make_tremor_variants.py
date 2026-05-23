"""Generate v5-v8 from v4 base, varying one knob each."""
import copy, yaml
from pathlib import Path

BASE = Path("/root/configs_gait/exp_tremor_cam34_vitb_v4.yaml")
OUTDIR = Path("/root/configs_gait")
base = yaml.safe_load(BASE.read_text())

def make(name, patches):
    cfg = copy.deepcopy(base)
    for path, val in patches.items():
        keys = path.split(".")
        d = cfg
        for k in keys[:-1]:
            if d.get(k) is None: d[k] = {}
            d = d[k]
        d[keys[-1]] = val
    cfg["run_name"] = name
    (OUTDIR / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {name}.yaml")

# v5: even tighter step cap. v4 used 30, try 10.
make("exp_tremor_cam34_vitb_v5", {"max_batches": 10})

# v6: head-only (linprobe) with bigger head LR.
make("exp_tremor_cam34_vitb_v6", {
    "model.freeze_encoder_layers": 12,
    "training.lr": 5.0e-4,
    "max_batches": 30,
})

# v7: shorter chunks at step 2 (32×2 = 64-frame span, same temporal extent,
# fewer transformer tokens -> smaller "model" effective)
make("exp_tremor_cam34_vitb_v7", {
    "data.chunk_length": 32,
    "data.chunk_shift": 16,
    "data.chunk_step": 2,
    "predict_per_item": 32,
    "max_batches": 30,
})

# v8: full FT but cap very tight (5 batches per "epoch") so we see val every
# 5 SGD steps with the full encoder updating. If anything is going to bend
# the overfit curve this is the most-aggressive-trainable + tightest-leash combo.
make("exp_tremor_cam34_vitb_v8", {
    "model.freeze_encoder_layers": 0,
    "training.lr": 1.0e-5,
    "max_batches": 5,
})
