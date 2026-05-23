# regression_with_negs — 2026-05-23

Six training experiments adding 241 FoG-dataset negatives to the chair,
walking, and finger-tapping regression heads, with two augmentation profiles
(trivial vs strong = RandomResizedCrop(0.5, 1.0) + ColorJitter +
TrivialAugmentWide + RandomErasing).

Best deployable checkpoints — scored on the **canonical (orig) val split**,
no FoG-negs in val:

| Symptom | Config | corr | r² | vs prior best |
|---|---|---|---|---|
| Walking | `exp_gait_strong_with_negs.yaml` | 0.683 | +0.225 | vs prod `exp_gait_unified_mixup` 0.489 / +0.188 |
| Chair-stand | `exp_chair_strong_with_negs.yaml` | 0.619 | +0.342 | vs 2026-05-19 `tulip_chair_long20_resplit` 0.57 / +0.26 |
| Finger-tap | `exp_fingertap_vitb_with_negs.yaml` | 0.456 | +0.128 | vs 2026-05-19 `exp12_hubu_fis_vitl_mixup` (ViT-L) 0.43 / +0.17 |

Strong aug helped chair + walking but **hurt finger-tap on the original val**
(0.37 vs basic's 0.46) — don't deploy `exp_fingertap_strong_with_negs`.

## How the dataset is built

The training set for each symptom uses two label files merged into one:
1. The original task labels (`chair_labels_resplit.json`, `feral_gait_labels.json`,
   `hubu-fis_regression_labels.json`)
2. 241 FoG-negative clips from `r2:feral/parkinson/videos/` (the same pool
   used by the iOS FoG demo) at z-scored target `(0 − mean) / std` per task.

The build scripts in `../../scripts/regression_with_negs/` symlink the videos
into a per-symptom directory and emit augmented labels JSON.

## What we did NOT do

- **Don't change val** alongside training augmentation — adding negs to val
  invalidates apples-to-apples comparison with prior baselines. See
  `shared_claude/skills/dont-mutate-val-without-permission/SKILL.md` in the
  feral_docs vault.
- Sapiens keypoint precrop on chair (TULIP task 20) — still pending. Other
  Tulip subtasks have parquets on R2; chair was never extracted. Future
  experiment: chair_precrop_strong.

## Artifacts on R2

`r2:feral/experiments/2026-05-23-negs-and-randaug/`
- `logs/` · `answers/` · `configs/` · `labels/` · `checkpoints/` (6 × 357 MB) · `scripts/`

## Headline-corrected metrics (apples-to-apples)

For why the originally reported full-val numbers were inflated by ~2× on
corr, see `raw/learnings/2026-05-23-dont-mutate-val-without-permission.md`
in the feral_docs vault.

## W&B

- chair_basic+negs:     https://wandb.ai/sposiboh/dikiy/runs/caay9yn7
- chair_strong+negs:    https://wandb.ai/sposiboh/dikiy/runs/h88aj4oj
- gait_basic+negs:      https://wandb.ai/sposiboh/dikiy/runs/ihti4mny
- gait_strong+negs:     https://wandb.ai/sposiboh/dikiy/runs/uapnt5ld
- fingertap_basic+negs: https://wandb.ai/sposiboh/dikiy/runs/u2malpar
- fingertap_strong+negs: https://wandb.ai/sposiboh/dikiy/runs/zezox88t
