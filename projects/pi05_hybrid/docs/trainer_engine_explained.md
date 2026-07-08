# `trainer.py` 详细解读

本文解读 `train/src/pi05/train/engine/trainer.py`，它是 Pi0.5 LoRA 微调流程的主训练引擎。这个文件不直接定义模型结构，也不直接读取原始数据格式，而是把配置、数据加载、模型构建、Accelerate 多卡训练、日志、checkpoint 和最终部署包导出串起来。

## 1. 文件定位

`trainer.py` 的核心职责可以概括为：

```text
ExperimentConfig
  -> 构建 DataLoader
  -> 构建 Accelerator
  -> 加载 PI05 base + 注入 LoRA
  -> 构建 preprocessor / optimizer / scheduler
  -> accelerator.prepare(...)
  -> 训练循环
  -> 保存 LoRA adapter
  -> 导出 deploy bundle
```

它依赖的主要模块如下：

- `pi05.common.config.schema.ExperimentConfig`：统一实验配置，来自 YAML 解析后的结构化对象。
- `pi05.common.model.builder.build_pi05_with_lora`：加载 PI05 预训练模型并包装 LoRA。
- `pi05.train.engine.builders`：构建 dataloader、optimizer、scheduler，并估算训练步数。
- `pi05.train.engine.batches.to_lerobot_pi05_batch`：把本项目 dataset 的 batch key 转成 LeRobot PI05 processor 需要的 key。
- `pi05.train.engine.checkpoints`：负责保存 epoch adapter、最终 adapter、恢复训练状态。
- `accelerate.Accelerator`：负责多卡、混合精度、梯度累积、日志同步和分布式封装。

## 2. 顶层入口

文件最后的入口是：

```python
def train_from_config(config: ExperimentConfig) -> None:
    Pi05LoraTrainer(config).run()
```

也就是说，CLI 侧只需要解析好配置，然后调用 `train_from_config(config)`。真正的训练逻辑都在 `Pi05LoraTrainer.run()` 里。

## 3. `_AverageMeter`：单指标平均值累加器

`_AverageMeter` 是一个很小的 dataclass：

```python
@dataclass
class _AverageMeter:
    total: float = 0.0
    count: int = 0
```

它只做两件事：

- `update(value)`：把新的指标值累加到 `total`，同时 `count += 1`。
- `avg`：返回 `total / count`，如果 `count == 0` 则返回 `0.0`。

这个类不是训练算法的一部分，只是为了把若干 step 的 loss、learning rate、grad norm 聚合成窗口平均值，避免每一步都打印日志。

## 4. `_LogWindow`：训练日志窗口

`_LogWindow` 内部维护：

```python
self._meters: dict[str, _AverageMeter] = {}
```

它可以同时记录多个指标，例如：

```text
train_loss
lr
grad_norm
```

关键方法：

- `update(**metrics)`：传入若干指标，值为 `None` 的指标会跳过。
- `as_dict()`：返回每个指标的平均值。
- `count`：返回当前窗口内累计了多少次。
- `reset()`：清空窗口，开始下一段统计。

训练循环里并不是每个 micro-batch 都更新日志，而是在 `accelerator.sync_gradients == True` 时才更新。原因是启用了梯度累积后，多个 micro-batch 才组成一个真正的 optimizer step。

## 5. `Pi05LoraTrainer` 主类

主类只保存一个配置：

```python
class Pi05LoraTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
```

后续所有组件都从这个 `config` 里拿参数，例如：

- `config.data`：数据集路径、图像尺寸、state/action 维度、worker 数量。
- `config.model`：预训练模型路径、dtype、chunk size、LoRA 目标模型配置。
- `config.training`：batch size、学习率、epoch、梯度累积、warmup、checkpoint 频率。
- `config.logging`：输出目录、TensorBoard、日志频率、run name。

## 6. `run()`：完整训练编排

`run()` 是最重要的函数，它按顺序完成所有训练准备和训练执行。

### 6.1 初始化日志和随机种子

```python
configure_logging()
set_training_seed(self.config.training.seed)
```

这里做两件事：

- 配置 Python logging。
- 固定随机种子，提高训练可复现性。

注意：多卡训练中完全 bit-level 可复现仍然受 CUDA kernel、DataLoader worker、分布式通信等影响，但固定 seed 仍是必要步骤。

### 6.2 构建 DataLoader 并估算训练步数

```python
dataloader = build_train_dataloader(self.config.data, self.config.training)
num_training_steps = count_training_steps(dataloader, self.config.training)
```

`build_train_dataloader` 会：

- 检查 dataset path 是否存在。
- 构建一次 dataset 以统计 state/action normalizer。
- 再构建正式 dataset，并传入 normalizer。
- 返回 PyTorch `DataLoader`。

`count_training_steps` 会考虑：

- dataloader 长度。
- `gradient_accumulation_steps`。
- `epochs`。
- 可选的 `max_steps_per_epoch`。

这个总步数会传给 cosine warmup scheduler，用于决定学习率曲线。

### 6.3 构建 Accelerator

```python
accelerator = self._build_accelerator()
```

`_build_accelerator()` 读取训练配置：

```python
return Accelerator(
    gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
    mixed_precision=mixed_precision,
    log_with="tensorboard" if log_cfg.use_tensorboard else None,
    project_dir=str(log_cfg.resolved_tensorboard_dir) if log_cfg.use_tensorboard else None,
)
```

它主要控制：

- 梯度累积步数。
- 混合精度模式，例如 `bf16`。
- TensorBoard 日志后端。
- 多进程训练中的主进程判断、barrier、日志同步、模型包装等。

如果配置里没有显式指定 `mixed_precision`，代码会自动判断：

```python
mixed_precision = "bf16" if torch.cuda.is_available() else "no"
```

### 6.4 初始化 TensorBoard

```python
self._init_trackers(accelerator)
```

如果 `logging.use_tensorboard == false`，这个函数直接返回。

如果开启 TensorBoard：

- `accelerator.init_trackers(...)` 初始化日志 tracker。
- 只有主进程会自动启动 TensorBoard。

这里用 `accelerator.is_main_process` 做保护，避免多卡训练时 4 个进程同时启动 4 个 TensorBoard。

### 6.5 主进程打印实验摘要

```python
if accelerator.is_main_process:
    log_run_summary(LOGGER, self.config.run_summary())
```

多卡训练时，每个 rank 都会执行同一份 Python 代码。如果不加主进程判断，日志会被重复打印多次。

## 7. `_load_model_staggered()`：多卡错峰加载模型

这是当前文件里非常关键的一段：

```python
for rank in range(accelerator.num_processes):
    if accelerator.local_process_index == rank:
        model = build_pi05_with_lora(...)
        gc.collect()
    accelerator.wait_for_everyone()
```

它的作用不是“只加载一次模型”，而是“让每个 rank 依次加载模型”。

假设你用：

```bash
accelerate launch --multi_gpu --num_processes 4 ...
```

那么会启动 4 个进程：

```text
rank0 -> GPU0
rank1 -> GPU1
rank2 -> GPU2
rank3 -> GPU3
```

DDP/Accelerate 普通多卡训练的特点是：每个进程都需要一份完整模型副本。因为每个 rank 都要独立做 forward、backward，然后通过分布式通信同步梯度。

所以最终一定会有：

```text
GPU0 上一份模型
GPU1 上一份模型
GPU2 上一份模型
GPU3 上一份模型
```

错峰加载解决的是 CPU 内存峰值问题。PI05 base 很大，如果 4 个进程同时读入权重并转换 dtype，主机内存可能瞬间变成 `4x` 峰值，导致 OOM Killer 杀掉某个 rank。错峰加载后流程变成：

```text
rank0 加载 -> barrier
rank1 加载 -> barrier
rank2 加载 -> barrier
rank3 加载 -> barrier
```

这样 CPU 瞬时峰值更接近单个 rank 的加载峰值。

### 7.1 `build_pi05_with_lora` 做了什么

这个函数在 `common/src/pi05/common/model/builder.py` 中，它负责：

- 从 `model.pretrained_path` 加载 PI05 base。
- 根据配置覆盖 chunk size、action steps、state/action dim。
- 调整 PI05 action projection，使 action 输出维度匹配当前项目。
- 创建 `LoraConfig`。
- 调用 `policy.wrap_with_peft(...)` 注入 LoRA。
- 启用 gradient checkpointing。
- 打印可训练参数比例。

日志里类似：

```text
trainable params: 20,648,832 / 4,164,016,766 (0.4959%)
```

说明总模型约 41.6 亿参数，但真正训练的是 LoRA 参数，大约 2064 万。

## 8. 构建 PI05 前处理器

模型加载后，`run()` 会取出 policy config：

```python
policy_config = get_pi05_policy_config(model)
policy_config.device = str(accelerator.device)
preprocessor, _ = make_pi05_pre_post_processors(policy_config, dataset_stats=None)
```

这里的 preprocessor 是 LeRobot PI05 官方处理流水线，负责把 batch 转成模型真正需要的格式。例如：

- 图像 resize/normalize 或 tensor 格式处理。
- state/action 归一化配置处理。
- task 文本 tokenization。
- 构造 PI05 forward 所需的字段。

这里 `dataset_stats=None` 是因为本项目 dataset builder 已经自己构建并应用 state/action normalizer，builder 中也把 PI05 config 的 `STATE` 和 `ACTION` normalizer 设置为 `IDENTITY`，避免重复归一化。

## 9. Optimizer 和 Scheduler

```python
optimizer = build_optimizer(model, self.config.training)
lr_scheduler = build_lr_scheduler(...)
```

`build_optimizer` 使用 `AdamW`，并且只把 `requires_grad=True` 的参数传进去：

```python
(param for param in model.parameters() if param.requires_grad)
```

这对 LoRA 很重要，因为 base model 参数被冻结，只有 LoRA adapter 参数参与训练。

`build_lr_scheduler` 使用 Transformers 的：

```python
get_cosine_schedule_with_warmup
```

学习率会先 warmup，再按 cosine 曲线衰减。

## 10. `accelerator.prepare(...)`

```python
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model,
    optimizer,
    dataloader,
    lr_scheduler,
)
```

这是进入分布式训练前的关键封装。

它会根据启动方式自动处理：

- 把模型放到对应设备。
- 包装成 DDP 或其他分布式形式。
- 处理 mixed precision。
- 处理 DataLoader 的 distributed sampling。
- 包装 optimizer 和 scheduler。

从这一行之后，`model` 不再一定是原始 PEFT model，可能是 Accelerate 包装后的模型。因此保存 adapter 时需要：

```python
unwrapped_model = accelerator.unwrap_model(model)
```

## 11. 断点恢复

```python
maybe_resume(accelerator, self.config.training.resume_from_checkpoint)
```

如果 `resume_from_checkpoint` 为 `None`，什么都不做。

如果配置了路径：

- 检查路径是否存在。
- 调用 `accelerator.load_state(...)` 恢复训练状态。

注意：当前 epoch adapter 保存函数只保存 LoRA adapter，不保存 optimizer/scheduler/rng 等完整训练状态。如果要完整断点恢复，通常需要配套使用 `accelerator.save_state(...)` 生成的状态目录。

## 12. `_train_loop()`：核心训练循环

训练循环结构如下：

```text
for epoch in range(epochs):
    for batch in dataloader:
        with accelerator.accumulate(model):
            processed_batch = preprocessor(to_lerobot_pi05_batch(batch))
            loss, loss_dict = model(processed_batch)
            accelerator.backward(loss)
            clip grad if needed
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(...)

        if accelerator.sync_gradients:
            total_steps += 1
            log_window.update(...)

    save epoch adapter checkpoint

flush final log window
export final adapter
export deploy bundle
```

### 12.1 `model.train()`

训练开始前调用：

```python
model.train()
```

这会让 dropout、LoRA dropout、部分 normalization 行为进入训练模式。

### 12.2 `total_steps` 与 `epoch_steps`

```python
total_steps = 0
epoch_steps = 0
```

这两个计数的是 optimizer step，而不是 dataloader iteration。

因为启用了梯度累积：

```yaml
gradient_accumulation_steps: 6
```

意味着 6 个 micro-batch 才对应 1 次真正参数更新。因此只有当：

```python
accelerator.sync_gradients
```

为 `True` 时，代码才执行：

```python
total_steps += 1
epoch_steps += 1
```

### 12.3 首个 batch 打印 state 维度

```python
if state_dim is None:
    state_dim = int(batch["state"].shape[-1])
    accelerator.print(f"[train] state_dim={state_dim}")
```

这用于快速确认数据集输出是否符合当前模型配置，例如你的配置中通常是：

```yaml
state_dim: 26
action_dim: 14
```

如果这里打印的 state_dim 不对，后续模型输入很可能会报 shape mismatch，或者训练出来的策略不可用。

### 12.4 batch key 转换

本项目 dataset 输出的 key 是：

```text
image_top
image_left_wrist
image_right_wrist
state
action_chunk
task
```

PI05 processor 需要的是 LeRobot 风格 key：

```text
observation.images.top
observation.images.left_wrist
observation.images.right_wrist
observation.state
action
task
```

所以训练循环里先做：

```python
processed_batch = preprocessor(to_lerobot_pi05_batch(batch))
```

`to_lerobot_pi05_batch` 只是 key 映射，真正复杂的图像、文本、state/action 处理由 `preprocessor` 完成。

### 12.5 前向计算

```python
loss, loss_dict = model(processed_batch)
```

PI05 训练返回：

- `loss`：用于反向传播的标量 tensor。
- `loss_dict`：用于日志记录的指标字典，里面通常包含 `"loss"`。

该模型训练目标通常是 Flow Matching 损失。`trainer.py` 不关心损失内部怎么计算，只把 batch 交给模型并拿回 loss。

### 12.6 反向传播

```python
accelerator.backward(loss)
```

不要直接调用：

```python
loss.backward()
```

原因是 Accelerate 需要统一处理：

- mixed precision。
- loss scaling。
- gradient accumulation。
- DDP 同步时机。

### 12.7 梯度裁剪

```python
if accelerator.sync_gradients and train_cfg.grad_clip_norm is not None:
    grad_norm = accelerator.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
```

这里只在 `sync_gradients == True` 时裁剪，是正确的。因为如果还在梯度累积中间，梯度还不是完整 step 的梯度。

`grad_norm_value` 会被转换成 float 后写入日志。

### 12.8 参数更新

```python
optimizer.step()
lr_scheduler.step()
optimizer.zero_grad(set_to_none=True)
```

这三步在 `accelerator.accumulate(model)` 上下文中执行。Accelerate 会处理梯度累积场景下的同步与跳步逻辑。

`set_to_none=True` 可以减少显存写入，并让下一次反向时重新分配梯度 tensor，通常比把梯度清零更高效。

### 12.9 日志窗口更新

```python
log_window.update(
    train_loss=self._resolve_loss_value(loss, loss_dict),
    lr=self._get_lr(optimizer, lr_scheduler),
    grad_norm=grad_norm_value,
)
```

日志只在真正 optimizer step 后更新，而不是每个 micro-batch 后更新。

`_resolve_loss_value` 优先从 `loss_dict["loss"]` 取值，如果没有，则用 `loss` tensor 本身。

`_get_lr` 优先从 scheduler 取 `get_last_lr()`，没有 scheduler 时回退到 optimizer param group。

### 12.10 定期打印日志

```python
if total_steps % log_cfg.log_freq == 0:
    self._flush_log_window(...)
```

`_flush_log_window` 会：

- 如果窗口为空则跳过。
- 调用 `accelerator.log(payload, step=total_steps)` 写入 TensorBoard。
- 仅主进程打印控制台日志。
- 清空窗口。

打印格式类似：

```text
epoch=000 step=000200 avg_over=200 train_loss=... grad_norm=... lr=...
```

### 12.11 `max_steps_per_epoch`

```python
if train_cfg.max_steps_per_epoch is not None and epoch_steps >= train_cfg.max_steps_per_epoch:
    break
```

这个参数常用于快速调试。比如你只想每个 epoch 跑 10 个 optimizer step，确认代码能走通，就可以设置它。

注意它限制的是 optimizer step，不是 dataloader batch 数。

## 13. Epoch checkpoint

每个 epoch 结束后：

```python
if (epoch + 1) % train_cfg.checkpoint_freq_epochs == 0:
    save_epoch_adapter_checkpoint(accelerator, model, run_output_dir, epoch + 1)
```

它会保存：

```text
outputs/checkpoints/<run_name>/checkpoint_epoch_<N>/adapter/
```

保存的是 LoRA adapter，不是完整 base model，也不是完整训练状态。

保存时只允许主进程写文件：

```python
if accelerator.is_main_process:
    _save_adapter(...)
accelerator.wait_for_everyone()
```

这样可以避免多个 rank 同时写同一个目录。

## 14. 训练结束后的导出

所有 epoch 跑完后，先刷新最后一段还没打印的日志：

```python
self._flush_log_window(...)
```

然后主进程导出最终 LoRA adapter：

```python
final_adapter_dir = export_final_adapter(accelerator, model, run_output_dir)
```

输出目录通常是：

```text
outputs/checkpoints/<run_name>/final_adapter/
```

接着尝试导出部署包：

```python
self._maybe_export_deploy_bundle(...)
```

部署包导出失败不会让训练失败，代码会捕获异常并打印 warning：

```python
except Exception as exc:
    accelerator.print(f"[train] warning: failed to export deploy bundle: {exc}")
```

这是一个比较实用的设计：训练产物 adapter 已经保存成功，即使部署 bundle 打包失败，也不应该丢掉训练结果。

## 15. `finally`：确保训练收尾

```python
finally:
    if log_cfg.use_tensorboard:
        accelerator.end_training()
```

无论训练成功、报错还是中断，只要启用了 TensorBoard tracker，都会调用 `end_training()` 收尾。

这可以避免 tracker 没有 flush 或资源没有释放。

## 16. 多卡训练中哪些代码每个进程都会执行

使用：

```bash
accelerate launch --multi_gpu --num_processes 4 ...
```

时，这个文件里的大部分 Python 代码都会执行 4 次，因为它们属于 4 个独立进程。

每个 rank 都会执行：

- 构建 dataloader。
- 构建 Accelerator。
- 加载一份模型。
- 构建 optimizer/scheduler。
- 进入训练循环。
- 对自己拿到的数据分片做 forward/backward。

只有被 `accelerator.is_main_process` 保护的逻辑才只执行一次，例如：

- 打印 run summary。
- 自动启动 TensorBoard。
- 导出最终 deploy bundle。
- 写部分 checkpoint 文件。

`accelerator.print(...)` 也通常只由主进程打印，可以减少重复日志。

## 17. 为什么 DDP 下不能只加载一次模型

当前训练方式是普通 DDP/Accelerate 多进程训练。它的基本模型是：

```text
一个 GPU 对应一个 Python 进程
一个 Python 进程持有一份完整模型
每个进程处理不同 batch 分片
反向传播后同步梯度
```

所以 4 卡训练最终需要 4 份模型副本。rank0 进程里的 Python 对象不能直接被 rank1 访问，因为它们是不同进程，内存空间隔离。

即使实现“rank0 从磁盘读一次，然后 broadcast 权重给其他 rank”，最终每张 GPU 还是要有一份完整参数。这样只能减少磁盘读取次数，不能减少显存中的模型副本数。

如果目标是让 4 张卡共同承载一份模型参数，需要使用：

- FSDP
- DeepSpeed ZeRO-3
- 其他参数分片训练方案

这些方案会把参数切成 shard 分布在多张卡上，但训练代码、checkpoint 保存、LoRA 合并和部署导出都会更复杂。

## 18. 当前训练循环的关键不变量

理解这个文件时，可以记住以下几个不变量：

- `total_steps` 统计的是 optimizer update step，不是 dataloader iteration。
- LoRA 训练只更新 `requires_grad=True` 的 adapter 参数。
- `accelerator.sync_gradients == True` 才代表当前 micro-batch 完成了一个真实优化步。
- checkpoint 保存的是 adapter，不是完整 merged model。
- deploy bundle 导出只在主进程执行。
- 错峰加载只降低 CPU 加载峰值，不改变 DDP 每卡一份模型的事实。

## 19. 常见日志含义

### 19.1 `loading pretrained model on local_rank=...`

表示当前 rank 正在加载 base model 并注入 LoRA。多卡下会按 rank 顺序出现多次。

### 19.2 `Loaded state dict from model.safetensors`

表示 PI05 base 权重已经从本地或 Hugging Face 缓存加载成功。

### 19.3 `Remapped 812 state dict keys`

表示加载过程中对 checkpoint key 做了名称映射，以匹配当前 PI05 实现里的模块名。

### 19.4 `All keys loaded successfully!`

表示 state dict 没有关键缺失或多余项，base 权重加载成功。

### 19.5 `trainable params: ...`

表示 LoRA 注入后可训练参数数量。这个数应该远小于总参数数量。如果可训练比例异常大，说明可能误开了全参训练。

### 19.6 `state_dim=...`

表示 dataloader 实际输出的 state 维度。它应该和配置里的 `data.state_dim`、`model.state_dim` 一致。

### 19.7 `Saved epoch LoRA adapter to: ...`

表示当前 epoch 的 LoRA adapter checkpoint 已保存。

### 19.8 `exported deploy bundle to: ...`

表示最终部署包导出成功。

## 20. 从源码角度看整体时序

完整训练时序如下：

```text
CLI 解析 YAML
  -> ExperimentConfig
  -> train_from_config(config)
  -> Pi05LoraTrainer(config).run()
      -> configure_logging()
      -> set_training_seed()
      -> build_train_dataloader()
      -> count_training_steps()
      -> _build_accelerator()
      -> _init_trackers()
      -> _load_model_staggered()
          -> build_pi05_with_lora()
              -> PI05Policy.from_pretrained()
              -> wrap_with_peft()
              -> enable gradient checkpointing
      -> make_pi05_pre_post_processors()
      -> build_optimizer()
      -> build_lr_scheduler()
      -> accelerator.prepare()
      -> maybe_resume()
      -> _train_loop()
          -> to_lerobot_pi05_batch()
          -> preprocessor(...)
          -> model(processed_batch)
          -> accelerator.backward(loss)
          -> clip_grad_norm_
          -> optimizer.step()
          -> scheduler.step()
          -> log window flush
          -> save_epoch_adapter_checkpoint()
      -> export_final_adapter()
      -> export_deploy_bundle()
      -> accelerator.end_training()
```

## 21. 阅读这个文件时最容易混淆的点

### 21.1 `batch_size` 和真实全局 batch

配置里的 `training.batch_size` 是每个进程 DataLoader 的 batch size。多卡和梯度累积后，有效全局 batch 约为：

```text
global_batch = batch_size * num_processes * gradient_accumulation_steps
```

例如：

```text
batch_size=1
num_processes=4
gradient_accumulation_steps=6
```

则一次 optimizer update 约等于 24 个样本的梯度。

### 21.2 `epoch_steps` 不是 batch 数

`epoch_steps` 只在 `sync_gradients` 时增加，因此它统计的是当前 epoch 已经完成多少次 optimizer update。

### 21.3 checkpoint adapter 不能直接当完整模型

`final_adapter/` 和 `checkpoint_epoch_*/adapter/` 只保存 LoRA adapter。部署时通常需要：

```text
PI05 base model + LoRA adapter + normalizer/config
```

如果想得到单独可加载的完整模型，需要额外执行 LoRA merge，把 adapter 合并进 base model。

### 21.4 `preprocessor` 不在 `accelerator.prepare` 里

`preprocessor` 不是普通 torch module 训练对象，不需要 optimizer，也不需要 DDP 包装。它在训练循环中直接处理 batch。

## 22. 总结

`trainer.py` 是一个训练编排层。它本身不实现 PI05 网络细节，也不实现 LoRA 算法细节，而是把项目里的各个组件按正确顺序组织起来：

- 数据部分由 `Pi05LeRobotDataset` 和 dataloader builder 负责。
- 模型部分由 `build_pi05_with_lora` 负责。
- 分布式训练由 `Accelerator` 负责。
- batch schema 适配由 `to_lerobot_pi05_batch` 和 PI05 preprocessor 负责。
- checkpoint 和最终导出由 `checkpoints.py` 与 `export_deploy_bundle` 负责。

因此，调试训练问题时可以按链路定位：

```text
数据 shape/key 问题 -> dataset / batches / preprocessor
模型加载或 LoRA 参数问题 -> common/model/builder.py
多卡、梯度累积、混合精度问题 -> trainer.py + Accelerate
保存/恢复/导出问题 -> checkpoints.py + runtime/bundle.py
```
