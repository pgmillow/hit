#!/usr/bin/env bash
# 4-GPU LoRA training via pi05 trainer (wraps accelerate launch).
# Fixes HF tokenizer hub/proxy failures: force offline + unset dead proxy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/train/config/lora.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

PYTHONPATH_ENTRIES=(
  "${PROJECT_DIR}/common/src"
  "${PROJECT_DIR}/train/src"
  "${PROJECT_DIR}/deploy/src"
)
for CANDIDATE in \
  "${PROJECT_DIR}/third_party/lerobot/src" \
  "${PROJECT_DIR}/../../third_party/lerobot/src" \
  "${PROJECT_DIR}/../third_party/lerobot/src"
do
  if [[ -d "${CANDIDATE}" ]]; then
    PYTHONPATH_ENTRIES+=("${CANDIDATE}")
    break
  fi
done
export PYTHONPATH="$(IFS=:; printf '%s' "${PYTHONPATH_ENTRIES[*]}")${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29508}"

echo "============================================"
echo " PI05 LoRA 4-GPU (offline HF, no proxy)"
echo " config: ${CONFIG_PATH}"
echo " GPUs:   ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

exec "${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  -m pi05.train.cli.train \
  --config "${CONFIG_PATH}" \
  "$@"


# CUDA_VISIBLE_DEVICES=4,5,6,7 bash train/scripts/train_lora_4gpu.sh
