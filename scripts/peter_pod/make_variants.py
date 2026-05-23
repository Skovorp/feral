"""Generate 5 variant configs as deltas off exp_gait_mixup.yaml."""
import copy, yaml, sys
from pathlib import Path

BASE = Path("/root/exp_gait_mixup.yaml")
OUTDIR = Path("/root/configs_gait")
OUTDIR.mkdir(exist_ok=True)

base = yaml.safe_load(BASE.read_text())

def make(name, patches):
    cfg = copy.deepcopy(base)
    # Strip the run-base header comment by overwriting; keep cfg in pure dict form.
    for path, val in patches.items():
        keys = path.split(".")
        d = cfg
        for k in keys[:-1]:
            if d.get(k) is None: d[k] = {}
            d = d[k]
        d[keys[-1]] = val
    cfg["run_name"] = name
    out = OUTDIR / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out}")

# Mirrors EXP3 — target noise replacing mixup
make("exp_gait_tnoise", {
    "mixup_alpha": None,
    "target_noise_std": 0.15,
})
# Mirrors EXP4 — mixup + tnoise
make("exp_gait_mixup_tnoise", {
    "mixup_alpha": 0.8,
    "target_noise_std": 0.15,
})
# Mirrors EXP5 — heavier dropout + WD only
make("exp_gait_dropout", {
    "mixup_alpha": None,
    "target_noise_std": 0.0,
    "model.fc_drop_rate": 0.7,
    "training.weight_decay": 0.2,
})
# Mirrors EXP10 — mixup + heavier dropout/WD stacked
make("exp_gait_mixup_dropout", {
    "mixup_alpha": 0.8,
    "target_noise_std": 0.0,
    "model.fc_drop_rate": 0.7,
    "training.weight_decay": 0.2,
})
# Mirrors EXP12 — ViT-L backbone (24 blocks; freeze 22 leaves last 2 + head)
make("exp_gait_vitl_mixup", {
    "backbone": "vjepa2_1_vitl_384",
    "model.freeze_encoder_layers": 22,
    "training.train_bs": 2,
    "training.val_bs": 2,
})
