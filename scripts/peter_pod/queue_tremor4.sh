#!/usr/bin/env bash
set -u
INDEX=/root/logs/index.log
cd /root/feral
declare -A CFG=(
  [v17]=exp_tremor_cam4_vitb_v17
  [v18]=exp_tremor_cam3_vitb_v18
  [v19]=exp_tremor_allcam_vitb_v19
  [v20]=exp_tremor_cam34_vitb_v20
)
for v in v17 v18 v19 v20; do
    NAME=${CFG[$v]}
    YAML=/root/configs_gait/$NAME.yaml
    LOG=/root/logs/$NAME.log
    echo "[$(date -u +%FT%TZ)] START $NAME -> $LOG" >> "$INDEX"
    if PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python -u -m feral.cli train-config "$YAML" 2>&1 | tee "$LOG"; then
        echo "[$(date -u +%FT%TZ)] OK    $NAME" >> "$INDEX"
    else
        echo "[$(date -u +%FT%TZ)] FAIL  $NAME (continuing)" >> "$INDEX"
    fi
done
echo "[$(date -u +%FT%TZ)] TREMOR QUEUE 4 DONE" >> "$INDEX"
