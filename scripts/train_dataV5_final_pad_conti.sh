#!/usr/bin/env bash
# ==============================================================
# pi05 V5 final 续训启动器 (letterbox padding, dataV5_final_v3src_pad_conti)
# 从 step 12000 checkpoint 继续，LR schedule 偏置至 step 12001。
# 默认读取 train_dataV5_final_pad_conti.env，逻辑复用 train_pi05.sh
#
# 源 checkpoint (只读):
#   /data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad/12000/
# 新 checkpoint 输出:
#   /data/gxdcheckpoint/dataV5_final_v3src_pad_conti/dataV5_final_v3src_pad_conti/
#   TensorBoard: .../tb  (global step = local + 12000)
#
# 用法:
#   bash scripts/train_dataV5_final_pad_conti.sh
#   bash scripts/train_dataV5_final_pad_conti.sh NUM_TRAIN_STEPS=1000
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_PI05_ENV="${TRAIN_PI05_ENV:-$SCRIPT_DIR/train_dataV5_final_pad_conti.env}"
exec bash "$SCRIPT_DIR/train_pi05.sh" "$@"
