# PI0.5 触觉时序融合设计

## 1. 目标

把当前单帧触觉输入扩展为具有空间位置和历史信息的触觉表示，并将它注入 PI0.5 的 VLM 主干，使动作专家能够利用接触位置、接触强度变化和短期触觉动态。

目标数据流：

```text
历史触觉窗口 [B, W, 22, 1, 14, 14]
    -> 共享 Tiny CNN
    -> 22 x W 个触觉特征向量
    -> 加传感器位置、手侧、时间位置编码
    -> Tactile Transformer
    -> 取当前时刻对应的 22 个位置 token
    -> 投影到 VLM hidden size
    -> 注入 PI0.5 prefix hidden states
    -> 与视觉、语言、动作 token 联合注意力
```

其中：

- `B`：batch size。
- `W`：向前观察的触觉时间窗口长度。
- `22`：左右手各 11 个触觉区域。
- 每个触觉区域为 `[1, 14, 14]`。

## 2. 当前代码状态

当前数据集已经将左右触觉伪图拆成固定顺序的 22 个压力 patch：

```text
单样本: [22, 1, 14, 14]
合批后: [B, 22, 1, 14, 14]
```

顺序为：

```text
left:  12,13,22,23,32,33,42,43,52,54,61
right: 12,13,22,23,32,33,42,43,52,54,61
```

相关实现：

- `train/src/pi05/train/data/dataset.py`
  - `split_tactile_images`
  - `TACTILE_OUTPUT_KEY = "tactile"`

当前实现已经让 `to_lerobot_pi05_batch()` 显式转发 `observation.tactile` 和
`observation.tactile_mask`，触觉不会再被当作普通 RGB 图像送入 SigLIP。

## 3. 推荐总体架构

### 3.1 输入窗口

每个训练样本除了当前时刻 `t`，还返回过去 `W - 1` 帧触觉：

```text
tactile_window = [t-W+1, ..., t-1, t]
shape = [W, 22, 1, 14, 14]
```

窗口不能跨 episode。episode 开头缺少历史帧时，重复该 episode 第一帧，并输出有效性 mask：

```text
tactile_window_mask: [W]
```

推荐初始配置：

```yaml
tactile:
  enabled: true
  history_steps: 8
  history_stride: 2
```

数据集是 `60 FPS` 时，`W=8, stride=2` 覆盖约 `0.25s` 历史。相比直接取连续 8 帧，它能覆盖更明显的接触变化，同时控制 token 数量。

### 3.2 共享 Tiny CNN

对全部 22 个触觉区域、全部时间帧共享同一个 Tiny CNN。共享参数可以避免每个传感器单独训练编码器，也能让模型学习统一的局部压力模式。

建议结构：

```text
Input            [B*W*22, 1, 14, 14]
Conv 3x3, 1->32, stride 1 + GELU
Conv 3x3, 32->64, stride 2 + GELU
Conv 3x3, 64->128, stride 2 + GELU
AdaptiveAvgPool  [B*W*22, 128, 1, 1]
Linear           128 -> D_tactile
Output           [B, W, 22, D_tactile]
```

初始建议：

```text
D_tactile = 256
```

Tiny CNN 输入只使用压力值。若后续数据可以稳定提供压力变化率，可将输入通道扩展为：

```text
[pressure, delta_pressure]
```

不建议第一版直接为每个传感器使用独立 CNN。

### 3.3 时空位置编码

Tiny CNN 输出后，对每个 token 加入以下可学习编码：

```text
token[w, s] =
    cnn_feature[w, s]
    + sensor_position_embedding[s]
    + hand_embedding[hand(s)]
    + temporal_embedding[w]
```

编码含义：

- `sensor_position_embedding[22]`：区分每个触觉区域的具体位置。
- `hand_embedding[2]`：显式区分左手和右手。
- `temporal_embedding[W]`：区分历史时刻，`w=W-1` 表示当前帧。

第一版使用可学习时间位置编码即可。若未来需要支持可变 FPS 或不规则时间戳，再改为基于真实时间差 `delta_t` 的正弦编码或 MLP 编码。

### 3.4 Tactile Transformer

将 `[B, W, 22, D_tactile]` 展平为：

```text
[B, W*22, D_tactile]
```

送入独立的小型 Transformer Encoder：

```yaml
tactile:
  hidden_dim: 256
  transformer_layers: 4
  attention_heads: 8
  mlp_ratio: 4
  dropout: 0.1
```

当 `W=8` 时，序列长度为 `176`，计算量可控。

时间窗口只包含当前和过去信息，不包含未来信息。Transformer 可以在整个窗口内执行双向注意力；最终仅取当前时刻对应的 22 个位置输出，因此不会造成训练到部署之间的未来信息泄漏。

### 3.5 保留当前时刻的 22 个位置 Token

不把全部 `W*22` 个历史 token 注入 VLM。Tactile Transformer 完成时空融合后，只取序列最后一个时刻对应的 22 个输出：

```text
transformer output: [B, W*22, D_tactile]
current tokens:     [B, 22, D_tactile]
```

每个当前位置 token 已通过 Transformer 融合历史窗口和其他触觉位置的信息，同时保留明确的传感器位置语义。

## 4. VLM 对齐与注入

### 4.1 推荐方案：Prefix Hidden-State Injection

使用线性投影将 22 个触觉位置 token 对齐到 PaliGemma hidden size：

```text
tactile_tokens = LayerNorm(tactile_summary)
tactile_tokens = Linear(D_tactile, D_vlm)
tactile_tokens = tactile_tokens * learnable_gate

shape: [B, 22, D_vlm]
```

`D_vlm` 不硬编码，从当前 PaliGemma 配置中读取。
当前 `gemma_2b` VLM 的 `D_vlm=2048`，因此当前配置最终注入张量为
`[B, 22, 2048]`。

把触觉 token 作为新的 prefix hidden states，放在视觉 token 和语言 token 之间：

```text
prefix =
    [vision tokens]
    [tactile tokens]
    [language tokens]
```

推荐该方案的原因：

1. PI0.5 当前已经把 prefix token 与 action expert token 放入每一层联合注意力。
2. 触觉 token 对齐到 `D_vlm` 后，会自然参与 VLM hidden-state 更新。
3. 保留 22 个触觉位置的明确语义。
4. 不需要修改每一层 Gemma 的内部计算，能够兼容现有 gradient checkpointing。
5. 可以通过零初始化 gate，使新模型初始化时接近原始 PI0.5 行为。

建议 gate：

```text
tactile_gate = tanh(alpha)
alpha 初始化为 0
```

训练开始时触觉残差为 0，随后模型自行学习触觉贡献大小。

### 4.2 第二阶段可选方案：中间层 Gated Cross-Attention

如果 prefix token 方案效果不足，可以在部分 VLM 层增加触觉 cross-attention：

```text
h_l = h_l + tanh(alpha_l) * CrossAttention(
    query=h_l,
    key=tactile_memory,
    value=tactile_memory
)
```

只在少量层注入，例如总层数的 `1/3`、`2/3` 和最后一层附近。

该方案表达能力更强，但需要修改 `compute_layer_complete()`，同时处理：

- gradient checkpointing 参数传递；
- prefix 与 action expert 两条 hidden-state 分支；
- checkpoint 和 PEFT 权重保存；
- 训练与部署模型结构一致性。

因此建议先完成 prefix hidden-state injection，再通过实验决定是否增加中间层注入。

## 5. 模型接口设计

新增独立模块：

```python
class TactileTemporalEncoder(nn.Module):
    def forward(
        self,
        tactile_window: Tensor,       # [B, W, 22, 1, 14, 14]
        tactile_window_mask: Tensor,  # [B, W]
    ) -> Tensor:
        """Return tactile prefix tokens shaped [B, 22, D_vlm]."""
```

建议内部组件：

```text
SharedTinyCNN
SensorPositionEmbedding
HandEmbedding
TemporalEmbedding
TransformerEncoder
VLMProjector
ZeroInitGate
```

PI0.5 的 `embed_prefix()` 接口扩展为：

```python
embed_prefix(
    images,
    img_masks,
    tokens,
    masks,
    tactile_tokens=None,
    tactile_mask=None,
)
```

触觉 token 对应的 attention mask 与视觉、语言 prefix token 一致，使动作 token 能够关注触觉 token。

## 6. 数据与部署链路

### 6.1 训练数据

`Pi05LeRobotDataset.__getitem__()` 根据当前样本索引构造历史窗口：

```text
tactile_window:      [W, 22, 1, 14, 14]
tactile_window_mask: [W]
```

窗口索引必须限制在当前 episode 的 `[start, end)` 内。

建议对触觉压力单独计算归一化统计量。优先采用每个传感器独立的稳健归一化：

```text
x_norm = clip((x - median_s) / scale_s, -5, 5)
```

如果各传感器标定一致，也可以先使用全局均值和方差作为第一版。

### 6.2 Batch 转换

`to_lerobot_pi05_batch()` 必须显式转发：

```python
model_batch["observation.tactile"] = batch["tactile_window"]
model_batch["observation.tactile_mask"] = batch["tactile_window_mask"]
```

不要再把触觉伪装成 RGB 图像送入 SigLIP。

### 6.3 在线推理

部署端为每个机器人环境维护固定长度 rolling buffer：

```text
deque(maxlen=W)
```

每次收到新触觉帧：

1. 拆分为 22 个 `[1,14,14]` patch。
2. 写入 rolling buffer。
3. 不足 `W` 帧时重复第一帧，并生成 mask。
4. 与当前视觉、状态、任务文本一起送入模型。
5. 环境 reset 时清空 rolling buffer。

训练和部署必须使用相同的：

- patch 顺序；
- `W` 和 stride；
- 归一化统计量；
- 缺失帧填充规则。

当前实现已经支持把触觉编码器权重保存为 `tactile_encoder.safetensors`，并在部署
bundle 加载时恢复该权重。当前 ROS 部署节点尚未配置左右触觉数据源 topic 和 rolling
buffer，因此真实机器人推理接入仍需完成该部分。

## 7. 配置建议

建议在 `lora.yaml` 增加：

```yaml
tactile:
  enabled: true
  history_steps: 8
  history_stride: 2
  patch_count: 22
  patch_size: 14
  input_channels: 1
  hidden_dim: 256
  transformer_layers: 4
  attention_heads: 8
  dropout: 0.1
  injection_mode: prefix
  sensor_position_embedding: learned
  temporal_position_embedding: learned
  zero_init_gate: true
  tactile_dropout_prob: 0.1
```

`tactile_dropout_prob` 表示训练时随机屏蔽整组触觉输入，避免模型过度依赖触觉，并验证视觉路径在传感器异常时仍能工作。

## 8. 训练策略

推荐分三个阶段：

### 阶段 A：链路验证

- 冻结 PI0.5 主干和原 LoRA。
- 只训练 Tiny CNN、Tactile Transformer、VLM projector 和 gate。
- 使用较短训练确认 loss 能下降、gate 能离开 0。

### 阶段 B：触觉编码器与 LoRA 联合训练

- 训练触觉模块。
- 同时训练当前 VLM/action expert LoRA。
- 主损失仍使用 PI0.5 flow matching action loss。

### 阶段 C：消融与稳定性验证

至少比较：

```text
视觉基线
单帧触觉
时序触觉但无 sensor position embedding
完整时序触觉
完整时序触觉 + tactile dropout
```

第一版不建议增加复杂辅助损失。若动作损失无法驱动触觉模块学习，再考虑接触预测或未来触觉预测辅助任务。

## 9. 建议文件改动

实现时建议按以下边界拆分：

```text
common/src/pi05/common/model/tactile_encoder.py
    Tiny CNN、Tactile Transformer、VLM projector

common/src/pi05/common/model/builder.py
    创建并挂载 TactileTemporalEncoder

common/src/pi05/common/config/schema.py
    新增 TactileConfig

train/src/pi05/train/data/dataset.py
    构造历史触觉窗口和 mask

train/src/pi05/train/engine/batches.py
    转发 tactile_window 和 tactile_window_mask

lerobot/src/lerobot/policies/pi05/modeling_pi05.py
    embed_prefix 接收并拼接 tactile tokens

deploy/src/pi05/deploy/
    增加触觉 rolling buffer，并保持训练侧预处理一致
```

为了减少对第三方 LeRobot 文件的直接修改，最终实现时可以优先使用本项目内的 PI0.5 wrapper/subclass。如果 wrapper 无法稳定接入 `embed_prefix()`，再对本地 LeRobot PI0.5 实现做最小修改。

## 10. 验收标准

### 形状与链路

- 数据集输出 `[B, W, 22, 1, 14, 14]`。
- Tiny CNN 输出 `[B, W, 22, D_tactile]`。
- 当前位置输出 `[B, 22, D_tactile]`。
- 投影后输出 `[B, 22, D_vlm]`。
- PI0.5 prefix 长度准确增加 `22`。
- 触觉模块参数能够收到非零梯度。

### 因果性

- 训练样本不读取当前时刻之后的触觉。
- 历史窗口不跨 episode。
- 在线推理 buffer 与训练窗口完全一致。

### 稳定性

- `zero_init_gate=true` 时，初始化输出应接近原模型。
- 关闭触觉后仍可正常训练和部署。
- 缺失触觉帧或触觉全零时不会产生 NaN。

### 性能

- 记录加入触觉模块后的显存与单步耗时。
- 对比视觉基线和时序触觉模型的任务成功率。
- 单独评估需要接触反馈的阶段，例如抓稳、滑移、倒水接触和双手协同。

## 11. 推荐的第一版范围

第一版只实现以下功能：

1. 数据集历史窗口与 mask。
2. 共享 Tiny CNN。
3. sensor、hand、temporal embedding。
4. 4 层 Tactile Transformer。
5. 保留当前时刻对应的 22 个位置 token。
6. 投影到 VLM hidden size。
7. 作为 prefix hidden states 注入。
8. zero-init gate 和 tactile dropout。
9. 训练、部署两侧统一 rolling window。

暂不实现中间层 cross-attention 和辅助损失。完成第一版消融后，再根据结果决定是否增加更强的 hidden-state 注入方式。
