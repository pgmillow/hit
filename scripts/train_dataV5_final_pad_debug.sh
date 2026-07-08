#!/usr/bin/env bash
# ==============================================================
# DEBUG: 用 dataV5_final_v3src_pad 参数跑 1 步, 检查 encoder_norm.bias
#        save/load 是否损坏
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/train_dataV5_final_pad.env"

source "$ENV_FILE"

# ── 覆盖为 debug 参数 ──
NUM_TRAIN_STEPS=1
SAVE_INTERVAL=1
KEEP_PERIOD=1
LOG_INTERVAL=1
OVERWRITE=true
RESUME=false

echo "=== DEBUG: encoder_norm.bias save/load test ==="
echo "  Steps: ${NUM_TRAIN_STEPS}, Save every: ${SAVE_INTERVAL}"
echo "  Config: ${CONFIG_NAME}"

ARGS=(
  "${CONFIG_NAME}"
  --assets-base-dir "${ASSETS_BASE_DIR}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
  --fsdp-devices "${FSDP_DEVICES}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --log-interval "${LOG_INTERVAL}"
  --save-interval "${SAVE_INTERVAL}"
  --keep-period "${KEEP_PERIOD}"
  --exp-name "${EXP_NAME}"
  --ema-decay "${EMA_DECAY}"
  --debug-checkpoint-bias
  --overwrite
  --no-resume
  --no-wandb-enabled
  --no-tensorboard-enabled
  --train-image-augment
)

echo "  CMD: CUDA_VISIBLE_DEVICES=${GPUS} python train.py ${ARGS[*]}"
echo ""

cd "${SCRIPT_DIR}/.."
CUDA_VISIBLE_DEVICES="${GPUS}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}" \
  XLA_MEM_FRACTION="${XLA_MEM_FRACTION}" \
  JAX_CACHE_DIR="${JAX_CACHE_DIR}" \
  HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${ARGS[@]}"
