#!/usr/bin/env bash
# Sequentially run gait experiment variants, logging each to /root/logs/.
# Each run stops on early-stopping (patience=3) so total wall time bounded.
set -u
mkdir -p /root/logs
LOG_INDEX=/root/logs/index.log
cd /root/feral

for cfg in \
    /root/configs_gait/exp_gait_tnoise.yaml \
    /root/configs_gait/exp_gait_mixup_tnoise.yaml \
    /root/configs_gait/exp_gait_dropout.yaml \
    /root/configs_gait/exp_gait_mixup_dropout.yaml \
    /root/configs_gait/exp_gait_vitl_mixup.yaml \
    /root/configs_gait/exp_tremor_cam34_vitb.yaml \
; do
    name=$(basename "$cfg" .yaml)
    log=/root/logs/${name}.log
    echo "[$(date -u +%FT%TZ)] START $name -> $log" | tee -a "$LOG_INDEX"
    if python -m feral.cli train-config "$cfg" 2>&1 | tee "$log"; then
        echo "[$(date -u +%FT%TZ)] OK    $name" | tee -a "$LOG_INDEX"
    else
        echo "[$(date -u +%FT%TZ)] FAIL  $name (continuing)" | tee -a "$LOG_INDEX"
    fi
done

echo "[$(date -u +%FT%TZ)] ALL DONE" | tee -a "$LOG_INDEX"
 
