# pi05 本地数据微调操作文档（JAX + FSDP）

本文档整理了当前在 `openpi` 仓库内，使用本地数据进行 `pi05` 微调的可复用流程。

## 1. 当前训练目标

- 基础模型：`pi05_base`
- 微调数据：`local/dataV4_MCAP_pi05_rgb`
- 训练框架：JAX
- 并行方式：FSDP（2 卡）
- 梯度累计：已接入（`gradient_accumulation_steps`）
- 日志平台：TensorBoard（默认写入 checkpoint 下的 `tb` 子目录）

## 2. 关键配置位置

- 训练配置入口：`src/openpi/training/config.py`
  - 使用配置名：`gxd_pi05`
- 训练主逻辑：`scripts/train.py`
- TrainState 类型定义：`src/openpi/training/utils.py`

## 3. 环境准备

### 3.1 安装 TensorBoard（如果未安装）

```bash
HTTP_PROXY=http://127.0.0.1:27890 \
HTTPS_PROXY=http://127.0.0.1:27890 \
http_proxy=http://127.0.0.1:27890 \
https_proxy=http://127.0.0.1:27890 \
uv pip install --python /home/xudi_ge/openpi/.venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  tensorboard
```

验证：

```bash
/home/xudi_ge/openpi/.venv/bin/python -c "import tensorboard; print(tensorboard.__version__)"
```

## 4. 训练命令（3 epoch 版本）

> 当前按 `batch_size=2`、`gradient_accumulation_steps=2`、`fsdp_devices=2` 运行。  
> 等效 batch 约为 4（2 x 2）。

```bash
HF_LEROBOT_HOME=/home/xudi_ge/data/lerobot_openpi \
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
/home/xudi_ge/openpi/.venv/bin/python /home/xudi_ge/openpi/scripts/train.py gxd_pi05 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --fsdp-devices 2 \
  --num-train-steps 800058 \
  --log-interval 100 \
  --save-interval 5000 \
  --keep-period 50000 \
  --exp-name gxd_pi05_3epoch_gpu01_acc2 \
  --overwrite \
  --ema-decay None \
  --no-wandb-enabled \
  --tensorboard-enabled \
  --tensorboard-subdir tb
```

## 5. TensorBoard 启动命令

```bash
/home/xudi_ge/openpi/.venv/bin/tensorboard \
  --logdir /home/xudi_ge/openpi/checkpoints/gxd_pi05/gxd_pi05_3epoch_gpu01_acc2/tb \
  --host 0.0.0.0 \
  --port 6006
```

本机访问：

- `http://localhost:6006`

## 6. 运行中常见现象与排查

### 6.1 首步看起来“卡住”

若日志停在：

- `[stage] train: starting step 0`
- 或出现 `slow_operation_alarm` / `Constant folding ... taking > 1s`

通常是 JAX/XLA 首次编译开销大，不是报错。先观察是否后续出现 `finished step 0`。

### 6.2 bool 参数报错

如果出现：

- `Unrecognized arguments: False True`

说明 CLI 不接受 `--xxx False/True`。应改为 flag 形式：

- 关闭：`--no-wandb-enabled`
- 开启：`--tensorboard-enabled`

### 6.3 TensorBoard 模块缺失

如果出现：

- `ModuleNotFoundError: No module named 'tensorboard'`

执行第 3.1 节安装命令后重试。

## 7. 当前改动摘要（已完成）

本流程已在代码中加入以下能力：

- `scripts/train.py`
  - 支持 TensorBoard 指标写入
  - 支持梯度累计（通过 `optax.MultiSteps`）
- `src/openpi/training/config.py`
  - 新增 `tensorboard_enabled`
  - 新增 `tensorboard_subdir`
  - 新增 `gradient_accumulation_steps`
- `src/openpi/training/utils.py`
  - 放宽 `TrainState` 中 `tx/opt_state` 的类型注解，兼容 `MultiSteps`

---

如需扩展：可在此文档追加“恢复训练命令（resume）”“单卡保守参数”“2 卡稳定化参数模板”。
