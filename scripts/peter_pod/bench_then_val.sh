#!/usr/bin/env bash
set -u
cd /root
echo "[$(date -u +%FT%TZ)] bench start" >> /root/logs/index.log
python /root/bench_feral.py 2>&1 | tee /root/logs/bench_feral.log
echo "[$(date -u +%FT%TZ)] val sweep start" >> /root/logs/index.log
python /root/val_tremor_sweep.py 2>&1 | tee /root/logs/val_tremor_sweep.log
echo "[$(date -u +%FT%TZ)] both done" >> /root/logs/index.log
