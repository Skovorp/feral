#!/usr/bin/env bash
# Sequential gait training queue. Deployable bg-replace run first, then its
# baseline, then the 1B runs. Each run logs clean-18 val_vid_corr live; best
# checkpoint (lowest val_mse) -> checkpoints/<run>_best_checkpoint.pt, uploaded
# to R2 after each run so an AP-IN-1 host drop never loses a finished model.
set -u
export TORCH_HOME=/workspace/torch_home HF_HOME=/workspace/hf_home
export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID=9e353be19c8dded16a3863a6f63c1266
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=afad95e569808972da44939e33c3d29f61a7a9119d600613ffe9e579c3a2821b
export RCLONE_CONFIG_R2_ENDPOINT=https://47387f9c6f77f2530bac57c6398d4c7a.r2.cloudflarestorage.com
R2DEST=r2:feral/experiments/2026-06-04-gait-phoneaug-bg
BB=$TORCH_HOME/hub/facebookresearch_vjepa2_main/src/hub/backbones.py
CFG=/workspace/configs
mkdir -p /workspace/logs /workspace/feral/checkpoints
cd /workspace/feral

RUNS="exp_gait_vitb_phoneaug_bg exp_gait_vitb_phoneaug exp_gait_vitg_phoneaug_bg exp_gait_vitg_phoneaug"
for run in $RUNS; do
  log=/workspace/logs/$run.log
  ckpt=/workspace/feral/checkpoints/${run}_best_checkpoint.pt
  if [ -f "$ckpt" ] && grep -q "Best checkpoint" "$log" 2>/dev/null; then
    echo "=== $run already complete, skipping ==="; continue
  fi
  echo "=== $(date -u +%H:%M:%S) START $run ==="
  # re-apply vjepa localhost fix (hub can refresh)
  [ -f "$BB" ] && sed -i "s|http://localhost:8300|https://dl.fbaipublicfiles.com/vjepa2|" "$BB"
  python -m feral.cli train-config $CFG/$run.yaml 2>&1 | tee "$log"
  echo "=== $(date -u +%H:%M:%S) END $run ==="
  if [ -f "$ckpt" ]; then
    rclone copy "$ckpt" "$R2DEST/checkpoints/" --s3-no-check-bucket 2>/dev/null
  fi
  rclone copy "$log" "$R2DEST/logs/" --s3-no-check-bucket 2>/dev/null
  echo "--- $run best val_vid_corr trajectory ---"
  grep -oE "val_vid_corr[\"=: ]+[-0-9.]+" "$log" | tail -12
done
# upload configs + labels + scripts for provenance
rclone copy $CFG "$R2DEST/configs/" --s3-no-check-bucket 2>/dev/null
rclone copy /workspace/labels "$R2DEST/labels/" --s3-no-check-bucket 2>/dev/null
rclone copy /workspace/scripts "$R2DEST/scripts/" --s3-no-check-bucket 2>/dev/null
echo "QUEUE_DONE $(date -u +%H:%M:%S)"
