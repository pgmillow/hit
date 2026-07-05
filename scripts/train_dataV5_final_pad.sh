#!/usr/bin/env bash
# ==============================================================
# pi05 V5 final 训练启动器 (letterbox padding 版本, dataV5_final_v3src_pad)
# 默认读取 train_dataV5_final_pad.env，逻辑复用 train_pi05.sh
#
# Checkpoint 输出:
#   /data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad/
#   TensorBoard: .../tb
#
# 用法:
#   bash scripts/train_dataV5_final_pad.sh
#   bash scripts/train_dataV5_final_pad.sh EPOCHS=1 NUM_TRAIN_STEPS=500
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_PI05_ENV="${TRAIN_PI05_ENV:-$SCRIPT_DIR/train_dataV5_final_pad.env}"
exec bash "$SCRIPT_DIR/train_pi05.sh" "$@"
