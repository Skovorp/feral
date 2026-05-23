#!/usr/bin/env bash
set -u
INDEX=/root/logs/index.log
cd /root/feral
for v in v13 v14 v15 v16; do
    NAME=exp_tremor_cam34_vitb_$v
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
echo "[$(date -u +%FT%TZ)] TREMOR QUEUE 3 DONE" >> "$INDEX"
