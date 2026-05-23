#!/usr/bin/env bash
# Wait for /root/logs/index.log to record "ALL DONE", then launch the tremor run.
set -u
mkdir -p /root/logs
LOG=/root/logs/exp_tremor_cam34_vitb.log
INDEX=/root/logs/index.log

echo "[$(date -u +%FT%TZ)] waiter started, polling $INDEX"

# Wait up to 24h
deadline=$(( $(date +%s) + 86400 ))
while ! grep -q "ALL DONE" "$INDEX" 2>/dev/null; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "[$(date -u +%FT%TZ)] waiter giving up after 24h" | tee -a "$INDEX"
        exit 1
    fi
    sleep 60
done

echo "[$(date -u +%FT%TZ)] START exp_tremor_cam34_vitb -> $LOG" | tee -a "$INDEX"
cd /root/feral
if python -m feral.cli train-config /root/configs_gait/exp_tremor_cam34_vitb.yaml 2>&1 | tee "$LOG"; then
    echo "[$(date -u +%FT%TZ)] OK    exp_tremor_cam34_vitb" | tee -a "$INDEX"
else
    echo "[$(date -u +%FT%TZ)] FAIL  exp_tremor_cam34_vitb" | tee -a "$INDEX"
fi
echo "[$(date -u +%FT%TZ)] TREMOR DONE" | tee -a "$INDEX"
