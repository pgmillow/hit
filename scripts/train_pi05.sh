#!/usr/bin/env bash
# ==============================================================
# pi05 双卡微调启动器 (launcher)
# 读取 train_pi05.env 配置并启动训练。代码与配置分离。
#
# 用法:
#   bash scripts/train_pi05.sh                       # 用默认配置文件跑
#   bash scripts/train_pi05.sh EXP_NAME=smoke NUM_TRAIN_STEPS=20   # 临时覆盖
#   TRAIN_PI05_ENV=/path/to/other.env bash scripts/train_pi05.sh   # 换配置文件
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TRAIN_PI05_ENV:-$SCRIPT_DIR/train_pi05.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[train_pi05] 配置文件不存在: $ENV_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

# 命令行临时覆盖: KEY=VALUE
for kv in "$@"; do
  if [[ "$kv" == *=* ]]; then
    export "${kv?}"
  else
    echo "[train_pi05] 忽略无法识别的参数: $kv (需为 KEY=VALUE 形式)" >&2
  fi
done

# 有效 batch = 全局 batch × 梯度累计步数 (每个优化器 step 实际消耗的样本数)
EFFECTIVE_BATCH=$(( BATCH_SIZE * GRAD_ACCUM_STEPS ))

# 若未手动指定 NUM_TRAIN_STEPS，则按 EPOCHS 自动换算 (换算里包含 batch_size)
if [[ -z "${NUM_TRAIN_STEPS:-}" ]]; then
  STEPS_PER_EPOCH=$(( (TOTAL_FRAMES + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))  # ceil 除法
  NUM_TRAIN_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
  STEP_SOURCE="自动(epoch换算)"
else
  STEPS_PER_EPOCH=$(( (TOTAL_FRAMES + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))
  STEP_SOURCE="手动指定"
fi

echo "[train_pi05] 配置文件 : $ENV_FILE"
echo "[train_pi05] 实验名   : ${EXP_NAME}"
echo "[train_pi05] GPU      : ${GPUS} (fsdp=${FSDP_DEVICES}, batch=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM_STEPS})"
echo "[train_pi05] 有效batch: ${EFFECTIVE_BATCH} (= batch ${BATCH_SIZE} × grad_accum ${GRAD_ACCUM_STEPS})"
echo "[train_pi05] 数据帧数 : ${TOTAL_FRAMES}  → 每epoch ${STEPS_PER_EPOCH} 步"
echo "[train_pi05] 训练时长 : ${EPOCHS} epoch  → 总步数 ${NUM_TRAIN_STEPS} (${STEP_SOURCE})"
if [[ -n "${LOG_STEP_OFFSET:-}" && "${LOG_STEP_OFFSET}" != "0" ]]; then
  GLOBAL_END=$(( LOG_STEP_OFFSET + NUM_TRAIN_STEPS ))
  echo "[train_pi05] 日志步数 : global = local + ${LOG_STEP_OFFSET} (local 0 → global ${LOG_STEP_OFFSET}, 结束 global ${GLOBAL_END})"
fi
echo "[train_pi05] NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"

# 组装训练参数
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
)

if [[ "${OVERWRITE}" == "true" ]]; then ARGS+=(--overwrite); else ARGS+=(--no-overwrite); fi
if [[ "${RESUME}" == "true" ]]; then ARGS+=(--resume); else ARGS+=(--no-resume); fi
if [[ "${WANDB}" == "true" ]]; then ARGS+=(--wandb-enabled); else ARGS+=(--no-wandb-enabled); fi
if [[ "${TENSORBOARD}" == "true" ]]; then ARGS+=(--tensorboard-enabled --tensorboard-subdir tb); else ARGS+=(--no-tensorboard-enabled); fi
if [[ "${train_image_augment:-true}" == "true" ]]; then ARGS+=(--train-image-augment); else ARGS+=(--no-train-image-augment); fi
if [[ -n "${LOG_STEP_OFFSET:-}" ]]; then ARGS+=(--log-step-offset "${LOG_STEP_OFFSET}"); fi

exec env \
  HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION}" \
  JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}" \
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${ARGS[@]}"
