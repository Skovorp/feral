#!/usr/bin/env bash
set -u
INDEX=/root/logs/index.log
cd /root/feral
declare -A CFG=(
  [v21]=exp_tremor_cam3_vitb_v21
  [v22]=exp_tremor_cam35_vitb_v22
  [v23]=exp_tremor_cam23_vitb_v23
  [v24]=exp_tremor_cam3_vitb_v24
)
for v in v21 v22 v23 v24; do
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
echo "[$(date -u +%FT%TZ)] TREMOR QUEUE 5 DONE" >> "$INDEX"
