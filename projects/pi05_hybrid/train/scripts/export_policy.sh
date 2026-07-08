#!/usr/bin/env bash

#cd /home/gexudi/hit/pi05_ws/projects/pi05_hybrid && \
#ADAPTER_DIR=/home/gexudi/hit/pi05_ws/projects/pi05_hybrid/outputs/checkpoints/pi05_hybrid_lora_v1/checkpoint_step_latest/adapter \
#OUTPUT_DIR=/home/gexudi/hit/pi05_ws/projects/pi05_hybrid/outputs/exports/pi05_hybrid_lora_v1 \
#OVERWRITE=true \
#bash train/scripts/export_policy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/train/config/lora.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OVERWRITE="${OVERWRITE:-false}"

PYTHONPATH_ENTRIES=(
  "${PROJECT_DIR}/common/src"
  "${PROJECT_DIR}/train/src"
  "${PROJECT_DIR}/deploy/src"
)

for CANDIDATE in   "${PROJECT_DIR}/third_party/lerobot/src"   "${PROJECT_DIR}/../third_party/lerobot/src"   "${PROJECT_DIR}/../../third_party/lerobot/src"
do
  if [[ -d "${CANDIDATE}" ]]; then
    PYTHONPATH_ENTRIES+=("${CANDIDATE}")
    break
  fi
done

export PYTHONPATH="$(IFS=:; printf '%s' "${PYTHONPATH_ENTRIES[*]}")${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  "${PYTHON_BIN}"
  -m
  pi05.train.cli.export_bundle
  --config
  "${CONFIG_PATH}"
)

if [[ -n "${RUN_DIR}" ]]; then
  CMD+=(--run-dir "${RUN_DIR}")
fi
if [[ -n "${ADAPTER_DIR}" ]]; then
  CMD+=(--adapter-dir "${ADAPTER_DIR}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  CMD+=(--overwrite)
fi
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Exporting PI05 deploy bundle with config %s\n' "${CONFIG_PATH}"
"${CMD[@]}"



