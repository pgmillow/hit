#!/usr/bin/env bash
# ==============================================================
# pi05 V5 final v2: eps=1e-6 + freeze vision encoder + schedule_offset=1
# 对应 config: dataV5_final_v3src_pad_v2
#
# 用法:
#   bash scripts/train_dataV5_final_pad_v2.sh
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_PI05_ENV="${TRAIN_PI05_ENV:-$SCRIPT_DIR/train_dataV5_final_pad_v2.env}"
exec bash "$SCRIPT_DIR/train_pi05.sh" "$@"
