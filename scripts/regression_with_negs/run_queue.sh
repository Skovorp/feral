#!/usr/bin/env bash
# Sequential negs-experiments queue. Waits until GPU is free before each launch,
# so it can be safely started while another training run is already going.
set -u
cd /root/feral
mkdir -p /root/logs

wait_gpu_free() {
  local label=$1
  echo "[$(date -u +%FT%TZ)] $label: waiting for GPU to free (<5 GiB used)" | tee -a /root/logs/queue.log
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ -z "$used" ]; then used=0; fi
    if [ "$used" -lt 5000 ]; then
      echo "[$(date -u +%FT%TZ)] $label: GPU free (${used} MiB)" | tee -a /root/logs/queue.log
      sleep 5
      return 0
    fi
    sleep 30
  done
}

run() {
  local cfg=$1
  local name
  name=$(basename "$cfg" .yaml)
  local ckpt="checkpoints/${name}_best_checkpoint.pt"
  local log="/root/logs/${name}.log"
  if [ -f "$ckpt" ]; then
    echo "[$(date -u +%FT%TZ)] SKIP $name (checkpoint exists)" | tee -a /root/logs/queue.log
    return 0
  fi
  wait_gpu_free "$name"
  echo "[$(date -u +%FT%TZ)] START $name -> $log" | tee -a /root/logs/queue.log
  python -m feral.cli train-config "$cfg" 2>&1 | tee "$log"
  echo "[$(date -u +%FT%TZ)] END   $name" | tee -a /root/logs/queue.log
}

# Don't include chair_with_negs (basic) — it's running in tmux:gpu_train.
# Order: basic-aug walking, basic-aug fingertap, then strong-aug variants of all 3.
run /root/configs_gait/exp_gait_vitb_with_negs.yaml
run /root/configs_gait/exp_fingertap_vitb_with_negs.yaml
run /root/configs_gait/exp_chair_strong_with_negs.yaml
run /root/configs_gait/exp_gait_strong_with_negs.yaml
run /root/configs_gait/exp_fingertap_strong_with_negs.yaml
echo "[$(date -u +%FT%TZ)] QUEUE DONE" | tee -a /root/logs/queue.log
