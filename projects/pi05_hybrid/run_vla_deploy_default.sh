#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_PATH="${1:-${PROJECT_DIR}/deploy/config/deploy.yaml}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot312}"
VLA_ENABLE_DELAY_SEC="${VLA_ENABLE_DELAY_SEC:-25}"

ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
REALSENSE_SETUP="/home/hit/octopus/realsense_ws/install/setup.bash"
PICOTELE_SETUP="/home/hit/octopus/picotele/install/setup.bash"
LAUNCH_FILE="${PROJECT_DIR}/deploy/launch/pi05_picotele_mux.launch.py"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    printf 'Missing %s: %s\n' "${label}" "${path}" >&2
    exit 1
  fi
}

try_activate_conda() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]]; then
    return
  fi
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi
  if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV_NAME}"
  else
    printf 'conda is not available; continue with current Python environment.\n' >&2
  fi
}

require_file "${CONFIG_PATH}" "deploy config"
require_file "${ROS_SETUP}" "ROS setup"
require_file "${REALSENSE_SETUP}" "RealSense workspace setup"
require_file "${PICOTELE_SETUP}" "picotele workspace setup"
require_file "${LAUNCH_FILE}" "Pi0.5 integrated launch"

try_activate_conda

# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${REALSENSE_SETUP}"
# shellcheck disable=SC1090
source "${PICOTELE_SETUP}"

cd "${PROJECT_DIR}"

printf 'Starting Pi0.5 VLA deployment stack\n'
printf '  config: %s\n' "${CONFIG_PATH}"
printf '  ROS_DISTRO: %s\n' "${ROS_DISTRO}"
printf '  conda env: %s\n' "${CONDA_DEFAULT_ENV:-unknown}"
printf '  VLA enable delay: %ss\n' "${VLA_ENABLE_DELAY_SEC}"

(
  sleep "${VLA_ENABLE_DELAY_SEC}"
  printf 'Requesting VLA control via /mux/enable_vla\n'
  ros2 topic pub --times 5 --rate 1 /mux/enable_vla std_msgs/msg/Bool "{data: true}"
) &

ros2 launch "${LAUNCH_FILE}" \
  config:="${CONFIG_PATH}" \
  launch_realsense:=true \
  launch_pico:=true \
  launch_picotele:=true \
  launch_mux:=true \
  launch_bridge:=true \
  launch_vla:=true
