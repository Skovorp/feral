#!/usr/bin/env bash
set -u
INDEX=/root/logs/index.log
cd /root/feral
for NAME in exp_tremor_cam2_vitb_v29 exp_tremor_cam2_vitb_v30 exp_tremor_cam2_vitb_v31 exp_tremor_cam2_vitb_v32; do
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
echo "[$(date -u +%FT%TZ)] TREMOR QUEUE 7 DONE" >> "$INDEX"
