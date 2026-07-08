#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/deploy/config/deploy.yaml"

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

if [[ $# -gt 0 ]]; then
  CONFIG_PATH="$1"
fi

printf 'Launching Pi0.5 deployment with config %s\n' "${CONFIG_PATH}"
python -m pi05.deploy.cli.deploy_ros --config "${CONFIG_PATH}"
