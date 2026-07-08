# dataV5_final_v3src_pad 数据处理管线指南

> 配置：`dataV5_final_v3src_pad` @ `src/openpi/training/config.py:1065`
> 模型：Pi05 (`discrete_state_input=True`, `action_dim=32`, `action_horizon=30`, `loss_action_dim=14`)

---

## 1. 数据流向总览

```
原始 LeRobot 数据
  → PromptFromLeRobotTask
    → RepackTransform
      → OctopusInputs
        → Normalize (quantile)
          → ClipState [-1, 1]
            → ResizeImages (224×224)
              → TokenizePrompt (state 离散化 + PaliGemma)
                → PadStatesAndActions (→ 32)
                  → Observation.from_dict (uint8 → float32 [-1,1])
                    → preprocess_observation (仅 train 时增强)
                      → 模型
```

**反向（推理输出处理）：**

```
模型输出 [30, 32]
  → Unnormalize (quantile)
    → OctopusOutputs (截断到 [:14])
      → 原始动作空间 [30, 14]
```

---

## 2. 各步骤详解

### Step 1: 原始数据加载

`LocalLeRobotParquetDataset` 从 `~/.cache/huggingface/lerobot/local/openpi_V5_mcap0625_v3src_pad` 读取。

每条样本产出：

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `observation.images.top` | `[H, W, 3]` | uint8 | 顶部相机 |
| `observation.images.left_wrist` | `[H, W, 3]` | uint8 | 左手腕相机 |
| `observation.images.right_wrist` | `[H, W, 3]` | uint8 | 右手腕相机 |
| `observation.state` | `[26]` | float32 | 机器人状态 |
| `action` | `[30, 14]` | float32 | 未来 30 步动作 |
| `task_index` | scalar | int64 | 任务编号 |

### Step 2: PromptFromLeRobotTask

`task_index` → 查 `tasks.jsonl` → 添加 `prompt` 字段（任务描述字符串）。

### Step 3: RepackTransform（键名重映射）

```
observation.images.top          → images.top
observation.images.left_wrist   → images.left_wrist
observation.images.right_wrist  → images.right_wrist
observation.state               → state
action                          → actions
```

### Step 4: OctopusInputs（格式标准化）

```python
# 图片重命名 + 格式处理
"images/top"           → "image/base_0_rgb"         # uint8 [H,W,3]
"images/left_wrist"    → "image/left_wrist_0_rgb"
"images/right_wrist"   → "image/right_wrist_0_rgb"

# 添加图片 mask（全 True）
"image_mask/base_0_rgb": True
"image_mask/left_wrist_0_rgb": True
"image_mask/right_wrist_0_rgb": True

# state, actions 保持 float32
"state"   → float32 [26]
"actions" → float32 [30, 14]
```

### Step 5: Normalize（分位数归一化）⚠️

使用 **quantile normalization**（`use_quantiles=True`），对 `state` 和 `actions` 分别归一化。

**公式：**

```python
span = q99 - q01
if span < 0.005:
    result = 0.0                         # 近常量维度 → 归零
else:
    result = (x - q01) / (span + 1e-6) * 2.0 - 1.0   # 映射到 [-1, 1]
```

**norm_stats 来源：** `assets/gxd_pi05_v3src_pad/local/openpi_V5_mcap0625_v3src_pad/norm_stats.json`

**维度详情：**

| 数据 | 维度数 | 归一化后 |
|------|--------|----------|
| state | 26 | 26 (值域 ~[-1, 1]) |
| actions | 14 | 14 (值域 ~[-1, 1]) |

**常量维度：** `state[12]` 和 `actions[12]` 的 `q01=q99=1000`，span=0 → 归一化后始终为 **0.0**。

### Step 6: ClipState

```python
state = np.clip(state, -1.0, 1.0)
```

原因：`clip_normalized_state=1.0`，防止归一化后越界。

### Step 7: ResizeImages

所有 3 张图片 resize + pad 到 **224×224**（保持宽高比，短边填黑边，uint8 填 0）。

### Step 8: TokenizePrompt ⚠️ 关键

`discrete_state_input=True`，state 被离散化嵌入文本 token。

```python
# 1. State 离散化到 256 bins
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
# 结果: 26 个整数，范围 [0, 255]

# 2. 构造 prompt 字符串
state_str = " ".join(map(str, discretized_state))
prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "

# 3. PaliGemma SentencePiece 编码 + BOS token
tokens = tokenizer.encode(prompt, add_bos=True)

# 4. Pad/truncate 到 max_token_len=200
```

### Step 9: PadStatesAndActions

```python
state:   [26]     → [32]      # 末尾补 6 个 0
actions: [30, 14] → [30, 32]  # 每步补 18 个 0
```

### Step 10: Observation.from_dict

```python
# 图片 uint8 → float32 [-1, 1]
image = image.astype(np.float32) / 255.0 * 2.0 - 1.0
```

### Step 11: preprocess_observation（模型内部）

- **train=True**: 应用 ColorJitter（`brightness=0.08, contrast=0.08, saturation=0.08, hue=0.03, p=0.1`），无 crop/rotate
- **train=False（推理）**: 不做任何增强，图片直接透传
- 填充缺失的 image_mask

---

## 3. 推理输出处理（反向管线）

### Unnormalize（分位数反归一化）

```python
span = q99 - q01
if span < 0.005:
    result = (q01 + q99) / 2.0        # 近常量维度 → 中点
else:
    result = (x + 1.0) / 2.0 * (span + 1e-6) + q01
```

**效果：** `actions[12]`（左手 cmd_pos0）span=0，反归一化后始终输出 **1000.0**。

### OctopusOutputs

```python
actions = actions[..., :14]   # 截断到 14 维
```

---

## 4. 推理时必须一致的清单

| # | 项目 | 训练时配置 | 推理要求 |
|---|------|-----------|----------|
| 1 | 图片 key 名 | `base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb` | 必须一致 |
| 2 | 图片格式 | uint8, HWC, resize+pad 224×224 | 必须一致 |
| 3 | 图片转 float32 | `(x / 255.0) * 2.0 - 1.0` | 必须一致 |
| 4 | 归一化方式 | **Quantile** normalization | 必须用同一个 `norm_stats.json` |
| 5 | State 维度 | 26 维，固定顺序 | 维度数、顺序必须一致 |
| 6 | State clip | `clip(state, -1.0, 1.0)` | 必须在归一化后、离散化前 clip |
| 7 | State 离散化 | `np.digitize(state, np.linspace(-1,1,257)[:-1]) - 1` | 256 bins 完全一致 |
| 8 | Prompt 格式 | `f"Task: {text}, State: {state_str};\nAction: "` | 格式、空格、标点完全一致 |
| 9 | Tokenizer | PaliGemma SentencePiece, `add_bos=True` | 相同 tokenizer |
| 10 | Token 长度 | `max_token_len=200` | pad/truncate 到 200 |
| 11 | State/actions pad | zero-pad 到 32 维 | 必须 pad |
| 12 | Actions 截断 | 取前 14 维 | 必须截断 |
| 13 | 常量维度 | dim 12 → 恒输出 1000 | 反归一化自动处理 |
| 14 | 图片增强 | **推理时不增强** | train=False |
| 15 | 采样步数 | 默认 10 步 denoising | 按需调整 |

---

## 5. norm_stats 关键数值速查

### state (26 dims)

| Dim | q01 | q99 | Span | 说明 |
|-----|-----|-----|------|------|
| 0-5 | ~0.52~2.09 | ~0.52~2.09 | <0.001~0.004 | 右臂关节（近常量） |
| 6 | 1.99 | 2.87 | 0.88 | 右臂某关节 |
| 7 | -1.30 | -0.28 | 1.01 | 左臂关节 |
| 8 | -2.27 | -1.03 | 1.24 | 左臂关节 |
| 9 | -2.08 | 0.15 | 2.23 | 左臂关节 |
| 10 | 0.52 | 1.57 | 1.05 | 关节 |
| 11 | -3.38 | -0.16 | 3.22 | 关节 |
| **12** | **1000.0** | **1000.0** | **0.0** | **左手（冻结常量）** |
| 13 | 301.0 | 999.8 | 698.8 | 手掌/手指 |
| 14-25 | 变化 | 变化 | 变化 | 其他状态量 |

### actions (14 dims)

| Dim | q01 | q99 | Span | 说明 |
|-----|-----|-----|------|------|
| 0-5 | ~0.52~2.09 | ~0.52~2.09 | <0.01 | 右臂关节 |
| 6 | 1.98 | 2.87 | 0.89 | |
| 7 | -1.30 | -0.24 | 1.07 | |
| 8 | -2.28 | -1.02 | 1.27 | |
| 9 | -2.09 | 0.15 | 2.24 | |
| 10 | 0.51 | 1.58 | 1.07 | |
| 11 | -3.38 | -0.12 | 3.26 | |
| **12** | **1000.0** | **1000.0** | **0.0** | **左手 cmd（冻结常量）** |
| 13 | 300.0 | 999.86 | 699.86 | 右手/手指 cmd |

---

## 6. 模型输出维度 mask

模型内部 32 维 action 的 loss mask：

```
[True, True, True, True, True, True, True, True, True, True, True, True, True, True,
 False, False, False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False]
```

即前 14 维参与训练，后 18 维为 padding，模型不学习。
