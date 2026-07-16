#!/usr/bin/env bash
# ==============================================================
# pi05 双臂倒水 (double_V1) 训练启动器
# 默认读取 train_double_V1.env，逻辑复用 train_pi05.sh
#
# Checkpoint 输出:
#   /data/final_double_hands_openpi/pi05_double_V1/pi05_double_V1/
#   TensorBoard: .../tb
#
# 用法:
#   bash scripts/train_double_V1.sh
#   bash scripts/train_double_V1.sh EPOCHS=1 NUM_TRAIN_STEPS=500
#   bash scripts/train_double_V1.sh XLA_MEM_FRACTION=0.95
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_PI05_ENV="${TRAIN_PI05_ENV:-$SCRIPT_DIR/train_double_V1.env}"
exec bash "$SCRIPT_DIR/train_pi05.sh" "$@"
