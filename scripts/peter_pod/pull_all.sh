#!/usr/bin/env bash
set -e
export AWS_ACCESS_KEY_ID=9e353be19c8dded16a3863a6f63c1266
export AWS_SECRET_ACCESS_KEY=afad95e569808972da44939e33c3d29f61a7a9119d600613ffe9e579c3a2821b
export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT=https://47387f9c6f77f2530bac57c6398d4c7a.r2.cloudflarestorage.com

RC=/usr/local/bin/rclone
OPTS="--transfers=32 --checkers=32 --multi-thread-streams=8 --multi-thread-cutoff=100M --stats=10s"

mkdir -p /root/data/tulip
log() { echo "=== $(date -u +%H:%M:%S) $* ==="; }

log auto-gait
$RC copy r2:feral/auto-gait /root/data/auto-gait $OPTS

log gavd
$RC copy r2:feral/gavd /root/data/gavd $OPTS

log koa-pd-nm-gait
$RC copy r2:feral/koa-pd-nm-gait /root/data/koa-pd-nm-gait $OPTS

log tulip/labels_csv_files_202503
$RC copy r2:feral/tulip/labels_csv_files_202503 /root/data/tulip/labels_csv_files_202503 $OPTS

for sub in gait stability spontaneity; do
  log tulip/${sub}_videos
  $RC copy r2:feral/tulip/${sub}_videos /root/data/tulip/${sub}_videos $OPTS
  log tulip/${sub}
  $RC copy r2:feral/tulip/${sub} /root/data/tulip/${sub} $OPTS
done

log DONE
du -sh /root/data/*
