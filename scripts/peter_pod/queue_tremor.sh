#!/usr/bin/env bash
set -u
INDEX=/root/logs/index.log
echo "[$(date -u +%FT%TZ)] tremor queue waiter started, waiting for v4" >> "$INDEX"
while tmux has-session -t train 2>/dev/null; do sleep 30; done
echo "[$(date -u +%FT%TZ)] v4 finished, chaining v5-v8" >> "$INDEX"

cd /root/feral
for v in v5 v6 v7 v8; do
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
echo "[$(date -u +%FT%TZ)] TREMOR QUEUE DONE" >> "$INDEX"
