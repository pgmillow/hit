#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/processed/lerobot_data}"
REPO_ID="${REPO_ID:-local/pi05}"
TASK_NAME="${TASK_NAME:-bimanual manipulation with dexterous hand}"
FPS="${FPS:-60}"
CONVERTER_SCRIPT="${CONVERTER_SCRIPT:-}"

if [[ -z "${CONVERTER_SCRIPT}" ]]; then
  for CANDIDATE in     "${PROJECT_DIR}/tools/mcap_to_lerobot_v3.py"     "${PROJECT_DIR}/../tools/mcap_to_lerobot_v3.py"     "${PROJECT_DIR}/../../tools/mcap_to_lerobot_v3.py"
  do
    if [[ -f "${CANDIDATE}" ]]; then
      CONVERTER_SCRIPT="${CANDIDATE}"
      break
    fi
  done
fi

usage() {
  cat <<'EOF'
Usage:
  bash train/scripts/prepare_dataset.sh mcap /path/to/demo.mcap
  bash train/scripts/prepare_dataset.sh mcap-dir /path/to/mcap_dir

Environment overrides:
  PYTHON_BIN        : python executable, default "python"
  OUT_DIR           : output LeRobot dataset dir, default data/processed/lerobot_data
  REPO_ID           : LeRobot repo id label
  TASK_NAME         : task description
  FPS               : dataset fps, default 60
  CONVERTER_SCRIPT  : path to mcap_to_lerobot_v3.py when it is outside this repo
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

if [[ -z "${CONVERTER_SCRIPT}" ]]; then
  printf 'Could not locate mcap_to_lerobot_v3.py. Set CONVERTER_SCRIPT=/abs/path/to/mcap_to_lerobot_v3.py\n' >&2
  exit 1
fi

MODE="$1"
INPUT_PATH="$2"

CMD=(
  "${PYTHON_BIN}"
  "${CONVERTER_SCRIPT}"
  --out "${OUT_DIR}"
  --repo-id "${REPO_ID}"
  --task "${TASK_NAME}"
  --fps "${FPS}"
)

case "${MODE}" in
  mcap)
    CMD+=(--mcap "${INPUT_PATH}")
    ;;
  mcap-dir)
    CMD+=(--mcap-dir "${INPUT_PATH}")
    ;;
  *)
    usage
    exit 1
    ;;
esac

mkdir -p "${OUT_DIR}"
printf 'Running dataset conversion into %s using %s\n' "${OUT_DIR}" "${CONVERTER_SCRIPT}"
"${CMD[@]}"
