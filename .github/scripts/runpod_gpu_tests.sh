#!/usr/bin/env bash
# Run the full test suite on a rented RunPod GPU pod, print its log, and delete the pod.
# Needs RUNPOD_API_KEY, REPO (owner/name) and SHA (the commit to test).
set -euo pipefail

API=https://api.runpod.io/v2
IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel  # same base image as the Dockerfile
GPUS=("NVIDIA RTX A5000" "NVIDIA GeForce RTX 3090" "NVIDIA RTX A6000" "NVIDIA L4" "NVIDIA A40" "NVIDIA GeForce RTX 4090")
TIMEOUT_MINUTES=45

api() {
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" "$@"
}

# The pod's whole job: install the pinned dependencies, run the tests, record the exit
# code, then idle until this script deletes it.
POD_COMMAND="(
  set -ex
  apt-get update && apt-get install -y --no-install-recommends git ffmpeg
  git clone https://github.com/$REPO /feral && cd /feral && git checkout $SHA
  pip install -r requirements.txt && pip install . pytest
  python -m pytest tests -v
) > /gpu-tests.log 2>&1
echo \$? > /gpu-tests.exit
sleep infinity"

POD_ID=
delete_pod() {
  [ -z "$POD_ID" ] || api -X DELETE "$API/pods/$POD_ID" -o /dev/null -w "pod $POD_ID delete: HTTP %{http_code}\n" || true
}
trap delete_pod EXIT
trap 'exit 143' INT TERM

# GPU stock comes and goes, so take the first type that can be rented right now.
for cloud in SECURE COMMUNITY; do
  for gpu in "${GPUS[@]}"; do
    body=$(jq -n --arg name "feral-gpu-tests-${SHA:0:7}" --arg image "$IMAGE" --arg cloud "$cloud" \
      --arg gpu "$gpu" --arg command "$POD_COMMAND" \
      '{name: $name, image: $image, cloud: $cloud, disk: 40,
        gpu: {id: $gpu, count: 1, minCudaVersion: "12.8"},
        entrypoint: ["/bin/bash", "-c"], cmd: [$command]}')
    response=$(api -X POST "$API/pods" -d "$body" -w '\n%{http_code}')
    status=${response##*$'\n'}
    if [[ $status == 2* ]]; then
      POD_ID=$(jq -r .id <<<"${response%$'\n'*}")
      echo "created pod $POD_ID: $gpu on $cloud, \$$(jq -r .cost <<<"${response%$'\n'*}")/hr"
      [ -z "${GITHUB_ENV:-}" ] || echo "RUNPOD_POD_ID=$POD_ID" >> "$GITHUB_ENV"
      break 2
    fi
    echo "could not rent $gpu on $cloud (HTTP $status)"
  done
done
[ -n "$POD_ID" ] || { echo "no GPU available"; exit 1; }

# Ask the pod for its exit-code file until the tests finish.
deadline=$((SECONDS + TIMEOUT_MINUTES * 60))
probe='{"command": ["sh", "-c", "cat /gpu-tests.exit 2>/dev/null || tail -n 1 /gpu-tests.log"]}'
while :; do
  ((SECONDS < deadline)) || { echo "timed out after $TIMEOUT_MINUTES minutes"; exit 1; }
  sleep 30
  response=$(api -X POST "$API/pods/$POD_ID/exec" -d "$probe" -w '\n%{http_code}') || continue
  status=${response##*$'\n'}
  if [[ $status != 200 ]]; then
    echo "pod not ready yet (exec HTTP $status)"
    continue
  fi
  output=$(jq -r .stdout <<<"${response%$'\n'*}")
  if [[ $output =~ ^[0-9]+$ ]]; then
    tests_exit=$output
    break
  fi
  echo "running: ${output:0:200}"
done

api -X POST "$API/pods/$POD_ID/exec" -d '{"command": ["tail", "-c", "900000", "/gpu-tests.log"]}' | jq -r .stdout
echo "tests exit code: $tests_exit"
exit "$tests_exit"
