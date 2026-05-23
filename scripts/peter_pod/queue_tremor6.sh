#!/usr/bin/env bash
set -u
INDEX=/root/logs/index.log
cd /root/feral
declare -A CFG=(
  [v25]=exp_tremor_cam23_vitb_v25
  [v26]=exp_tremor_cam2_vitb_v26
  [v27]=exp_tremor_cam235_vitb_v27
  [v28]=exp_tremor_cam234_vitb_v28
)
for v in v25 v26 v27 v28; do
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
echo "[$(date -u +%FT%TZ)] TREMOR QUEUE 6 DONE" >> "$INDEX"
