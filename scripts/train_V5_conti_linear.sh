#!/usr/bin/env bash
# ==============================================================
# pi05 V5 conti linear 训练启动器
# 从 step 12000 checkpoint 加载权重（不加载 optimizer state），
# 使用线性 LR 9e-6 → 5e-7 重新开始训练。
# 默认读取 train_V5_conti_linear.env，逻辑复用 train_pi05.sh
#
# Checkpoint 输出:
#   /data/gxdcheckpoint/dataV5_final_v3src_pad_linear/V5_conti_linear/
#   TensorBoard: .../tb
#
# 用法:
#   bash scripts/train_V5_conti_linear.sh
#   bash scripts/train_V5_conti_linear.sh NUM_TRAIN_STEPS=500 EXP_NAME=smoke
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_PI05_ENV="${TRAIN_PI05_ENV:-$SCRIPT_DIR/train_V5_conti_linear.env}"
exec bash "$SCRIPT_DIR/train_pi05.sh" "$@"
