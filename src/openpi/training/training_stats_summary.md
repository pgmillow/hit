# GXD pi05 训练统计总结

> 数据来源：`/data/gxdcheckpoint/*/tb/` TensorBoard 事件文件  
> 生成时间：2026-06-28  
> 说明：V4 与 V5 数据集不同，**loss 数值不可直接横向对比**，LR/step 规律仅供参考。

---

## 1. Global Step 定义

「真实 step」= 该 run 在整条训练链路上的累计步数。

| 策略 | global step 公式 | LR schedule 用的 step | 备注 |
|------|------------------|----------------------|------|
| **V5** `gxd_pi05_V5_mcap0625_rgb` | `global = local` | local | pi05_base 从头训 V5 数据 |
| **V4 stateconti** | `global = 4000 + local` | local | 从 4k ckpt 续训 |
| **V4 from6000_lrhalf** | `global = 6000 + local` | local | resume 6k，peak LR 减半 |
| **V4 from10000_staged** | `global = 10000 + local` | local | 从 10k ckpt 续训（TB 在 backup 目录） |
| **V4 from2000_staged** | `global = 12000 + local` | local + `schedule_offset=2000` | config 另有 `log_step_offset=2000`；**当前 TB 无数据** |

示例：from2000 接着 12k ckpt 训练，local step 500 → global step = **12500**。

---

## 2. 各策略 LR Schedule 概要

### V5 — `GxdV5WarmCosineLinearSchedule`

| 参数 | 值 |
|------|-----|
| warmup | 500 step → peak **1e-5** |
| 3k | **1e-6** |
| 6k | **7e-7** |
| 9k | **4e-7** |
| 10k | **1e-8** |
| 30k | **0** |

### V4 stateconti — `CosineDecaySchedule`

- warmup 2000 → peak **1e-5**，30k step cosine 衰减至 **5e-7**

### V4 from6000_lrhalf — `CosineDecaySchedule`

- warmup 50 → peak **2.5e-6**，30k step 衰减至 **1.25e-7**

### V4 from10000 / from2000 — `StagedCosineSchedule`

- 每 3k step 一段 cosine，每段结束 LR ÷ 5
- from10000：`init_lr=1.5e-6`，total 16668
- from2000：`init_lr=3e-6`，`schedule_offset=2000`

---

## 3. 每 1k Global Step：Loss & LR

### V5（openpi_V5，pi05_base）— TB 覆盖 global 0 ~ 5260

| global | local | loss | LR (TB) |
|--------|-------|------|---------|
| 1,000 | 1,000 | 0.0139 | 8.68e-6 |
| 2,000 | 2,000 | 0.0094 | 2.32e-6 |
| 3,000 | 3,000 | 0.0112 | 1.00e-6 |
| 4,000 | 4,000 | 0.0100 | 9.25e-7 |
| 5,000 | 5,000 | 0.0094 | 7.75e-7 |

### V4 stateconti（from 4k ckpt）— global 4000 ~ 10030

| global | local | loss | LR (TB) |
|--------|-------|------|---------|
| 4,000 | 0 | 0.0227 | ~0（warmup 起点） |
| 5,000 | 1,000 | 0.0069 | 5.00e-6 |
| 6,000 | 2,000 | 0.0068 | 1.00e-5 |
| 7,000 | 3,000 | 0.0067 | 9.97e-6 |
| 8,000 | 4,000 | 0.0060 | 9.88e-6 |
| 9,000 | 5,000 | 0.0055 | 9.73e-6 |
| 10,000 | 6,000 | 0.0050 | 9.53e-6 |

### V4 from6000_lrhalf — global 6000 ~ 16700

| global | local | loss | LR (TB) |
|--------|-------|------|---------|
| 6,000 | 0 | 0.0087 | 5.00e-6 |
| 7,000 | 1,000 | 0.0047 | 4.99e-6 |
| 8,000 | 2,000 | 0.0031 | 4.95e-6 |
| 9,000 | 3,000 | 0.0037 | 4.88e-6 |
| 10,000 | 4,000 | 0.0032 | 4.79e-6 |
| 11,000 | 5,000 | 0.0025 | 4.68e-6 |
| 12,000 | 6,000 | 0.0032 | 4.55e-6 |
| 13,000 | 7,000 | 0.0039 | 2.20e-6 |
| 14,000 | 8,000 | 0.0026 | 2.11e-6 |
| 15,000 | 9,000 | 0.0035 | 2.01e-6 |
| 16,000 | 10,000 | 0.0024 | 1.91e-6 |

> resume 时 optimizer 带历史状态，global 6000 处 TB 的 LR 与理论 schedule 不一致，**以 TB 实测为准**。

### V4 from10000_staged（backup TB）— global 10000 ~ 12540

| global | local | loss | LR (TB) |
|--------|-------|------|---------|
| 10,000 | 0 | 0.0023 | 1.50e-6 |
| 11,000 | 1,000 | 0.0020 | 1.41e-6 |
| 12,000 | 2,000 | 0.0017 | 1.16e-6 |

---

## 4. 首次 loss ≤ 0.01

| 策略 | global step | local step | loss | LR (TB) |
|------|-------------|------------|------|---------|
| **V5** | **~1,890** | 1,890 | 0.0098 | **2.91e-6** |
| V4 stateconti | ~4,200 | 200 | 0.0090 | 1.00e-6 |
| V4 from6000 | 6,000 | 0 | 0.0087 | 5.00e-6（启动时已低于 0.01） |
| V4 from10000 | 10,000 | 0 | 0.0023 | 1.50e-6（启动时已远低于 0.01） |

---

## 5. loss ≈ 0.01 时 LR 该设多少？

分两种场景：

### A. 第一次降到 0.01（仍在快速下降）

| 策略 | 典型 global step | 建议 LR 区间 |
|------|-----------------|-------------|
| V5 从头训 | ~1,500 – 2,000 | **2e-6 ~ 8e-6** |
| V4 stateconti 续 4k | ~4,000 – 5,000 | **1e-6 ~ 5e-6** |

### B. 已在 0.01 平台震荡（V5 当前 ~5k step）

- loss：0.008 – 0.012
- LR：**7e-7 ~ 1e-6**
- 属正常平台期，**不必强行加大 LR**

### 跨策略参考（V4 同数据）

- 续训且 loss 已在 0.01 以下：**1e-6 ~ 2.5e-6**
- 从头/早期刚碰到 0.01：**2e-6 ~ 1e-5**
- 想在 0.01 附近继续探索：**~1e-6** 稳，**~3e-6** 若 loss 仍有下降空间

---

## 6. V5 当前训练结论（截至 global ~5260）

1. **首次 loss ≤ 0.01**：global ~1890，LR ~**3e-6** — 与 schedule 一致，正常。
2. **2k 后进入平台**：loss ~0.009–0.011，LR 已降至 **1e-6 以下**。
3. **不宜因平台过早加大 LR**；更值得关注：
   - norm_stats 仍有 ~35% state clip
   - 单 prompt、train loss 低 ≠ 真机好
4. **建议**：用 3k / 5k checkpoint 做推理/真机验证，再决定是否调 schedule。

---

## 7. 数据路径 & 复现

```bash
# TensorBoard 查看 V5
/home/xudi_ge/openpi/.venv/bin/tensorboard \
  --logdir /data/gxdcheckpoint/gxd_pi05_V5_mcap0625_rgb/gxd_pi05_V5_mcap0625_rgb/tb \
  --host 0.0.0.0 --port 6011

# 合并多个 run
/home/xudi_ge/openpi/.venv/bin/tensorboard \
  --logdir_spec=\
V5:/data/gxdcheckpoint/gxd_pi05_V5_mcap0625_rgb/gxd_pi05_V5_mcap0625_rgb/tb,\
stateconti:/data/gxdcheckpoint/gxd_pi05_stateconti/gxd_pi05_standard_state_from4000_3epoch/tb,\
from6000:/data/gxdcheckpoint/gxd_pi05_stateconti_from6000_lrhalf/gxd_pi05_standard_state_from6000_lrhalf/tb \
  --host 0.0.0.0 --port 6011
```

Checkpoint 根目录：`/data/gxdcheckpoint/`

---

## 8. 相关配置

| 文件 | 内容 |
|------|------|
| `src/openpi/training/config.py` | 各 `gxd_pi05_*` TrainConfig |
| `src/openpi/training/optimizer.py` | `GxdV5WarmCosineLinearSchedule`、`StagedCosineSchedule` |
| `scripts/train_pi05_V5.sh` / `.env` | V5 启动器 |
| `assets/gxd_pi05/local/openpi_V5_mcap0625_rgb/norm_stats.json` | V5 归一化统计 |

---

## 9. `GxdMilestoneSchedule`（V5 当前策略）

### 曲线

| 阶段 | step 范围 | LR |
|------|-----------|-----|
| 平台 | 0 ~ 3k | **5e-6** 恒定 |
| 线性降 1 | 3k ~ 6k（3k step） | 5e-6 → **1e-6** |
| 线性降 2 | 6k ~ 12k（6k step） | 1e-6 → **1e-7** |
| 收尾 | 12k ~ 30k | 1e-7 → **0** |

### 关键节点

| step | LR |
|------|-----|
| 0 / 3k | 5e-6 |
| 6k | 1e-6 |
| 12k | 1e-7 |
| 30k | 0 |

### config 用法

```python
lr_schedule=_optimizer.GxdMilestoneSchedule(
    plateau_lr=5e-6,
    lr_at_6k=1e-6,
    lr_at_12k=1e-7,
    plateau_end=3_000,
    decay_steps_to_1e6=3_000,
    decay_steps_to_1e7=6_000,
    total_steps=30_000,
),
```

实现：`src/openpi/training/optimizer.py` → `GxdMilestoneSchedule`  
已应用于：`gxd_pi05_V5_mcap0625_rgb`

---

## 10. V5 从 5k 续训：`gxd_pi05_V5_from5k_milestone`

| 项 | 值 |
|----|-----|
| 初始权重 | `.../gxd_pi05_V5_mcap0625_rgb/5000/params` |
| LR | `GxdMilestoneSchedule`，`schedule_offset=5000` |
| local step | 0 → 25,000 |
| **global step** | **5,000 → ~30,000**（`log_step_offset=5000`） |
| 启动时 LR | **1e-6**（schedule 在 5k 节点） |
| checkpoint 目录 | `/data/gxdcheckpoint/gxd_pi05_V5_from5k_milestone/...` |

```bash
# 先停掉当前 V5 训练（如在跑），再：
cd ~/openpi
bash scripts/train_pi05_V5_milestone5k.sh
```

> checkpoint 文件夹名为 **local step**（1000/2000/…），对应 **global = local + 5000**。





  cd /home/xudi_ge/openpi

  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  HF_LEROBOT_HOME=/home/xudi_ge/data/lerobot_openpi \
  CUDA_VISIBLE_DEVICES=0,1 \
  NCCL_P2P_DISABLE=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  JAX_COMPILATION_CACHE_DIR=/home/xudi_ge/.cache/jax \
  /home/xudi_ge/openpi/.venv/bin/python /home/xudi_ge/openpi/scripts/train.py gxd_pi05_629_from4k_conti_lr \
    --fsdp-devices 2 \
    --no-resume \
    --no-overwrite \
    --tensorboard-enabled \
    --tensorboard-subdir tb