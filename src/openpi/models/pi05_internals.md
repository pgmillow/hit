# pi0.5 (JAX) 数据流 / Flow Matching / Loss 全流程拆解

> 本文针对 OpenPI 官方 JAX 实现的 `pi0.5`（即 `Pi0Config(pi05=True)`），从“一批样本离开 DataLoader”到“得到 loss / 采样动作”的每一步都定位到具体文件和函数。所有行号基于你当前 checkout 的代码。
>
> 仓库根：`/home/xudi_ge/openpi`
> 关键目录：
> - 模型：`src/openpi/models/`
> - 训练：`src/openpi/training/` + `scripts/train.py`
> - 数据变换：`src/openpi/transforms.py`
> - 推理封装：`src/openpi/policies/policy.py`

---

## 0. 总体架构（先看大局）

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 数据侧 (CPU / TPU-host)                                                    │
│                                                                            │
│  LeRobot / RLDS / Fake Dataset                                             │
│        │  原始字段: observation.images.*, observation.state, action, ...   │
│        ▼                                                                   │
│  repack_transforms  →  data_transforms  →  Normalize  →  model_transforms  │
│  (RepackTransform)    (AlohaInputs/        (z-score 或     (InjectDefault  │
│                       DeltaActions 等)     quantile)        Prompt,         │
│                                                            ResizeImages,    │
│                                                            TokenizePrompt,  │
│                                                            PadStatesAnd...) │
│        │                                                                   │
│        ▼  得到一个 nested dict, 字段: image/image_mask/state/              │
│           tokenized_prompt[_mask]/actions                                  │
│  TorchDataLoader  →  collate → sharded jax.Array                           │
│        │                                                                   │
│        ▼  DataLoaderImpl.__iter__ → Observation.from_dict(batch), actions  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (Observation, Actions)
┌────────────────────────────────────────────────────────────────────────────┐
│ 模型侧 (GPU/TPU)  — src/openpi/models/pi0.py::Pi0                          │
│                                                                            │
│  训练: Pi0.compute_loss_with_debug                                         │
│    1. preprocess_observation (图像增广 + resize + mask)                    │
│    2. 采样噪声 ε ~ N(0,I), 采样 t ~ Beta(1.5,1)                            │
│    3. 构造 flow matching 中间态:  x_t = t·ε + (1-t)·a                      │
│       目标速度场:           u_t = ε - a                                    │
│    4. embed_prefix(obs)        → 图像 tokens + 文本 tokens (PaliGemma 前缀) │
│    5. embed_suffix(obs, x_t,t) → state token + action tokens + adaRMS cond │
│    6. make_attn_mask + positions                                            │
│    7. PaliGemma.llm([prefix, suffix], adarms_cond=[None, t_emb]) 一次前向   │
│    8. v_t = action_out_proj(suffix_out[:, -action_horizon:])               │
│    9. loss = mean((v_t - u_t)^2)  (按 active 维度归一)                      │
│                                                                            │
│  推理: Pi0.sample_actions                                                  │
│    1. preprocess_observation (eval, 无增广)                                │
│    2. embed_prefix 一次 → 填充 KV cache                                     │
│    3. while t > 0:                                                          │
│         embed_suffix(obs, x_t, t) → 仅跑 suffix (用 KV cache)              │
│         v_t = action_out_proj(suffix_out)                                  │
│         x_t ← x_t + dt·v_t   (dt = -1/num_steps, Euler ODE)                │
│         t   ← t + dt                                                        │
│    4. 返回 x_0 (= 预测动作)                                                 │
│                                                                            │
│  训练驱动: scripts/train.py::train_step                                     │
│    loss_fn → nnx.value_and_grad → optax.apply_updates                      │
└────────────────────────────────────────────────────────────────────────────┘
```

核心思想：
- **Pi0.5 = PaliGemma (2B, ‘大模型/expert 0’) + Action Expert (300M, ‘expert 1’) 双专家 Transformer**，两个专家共享同一个 transformer 堆栈但各有自己的 width/MLP/QKV 投影，在 self-attention 里把两边的 token 拼起来一起算。
- **Flow Matching**：把 action 序列看作从噪声分布 (`t=1`) 到真实动作分布 (`t=0`) 的一条直线概率路径，模型学习的是这条路径上每一点的速度场 `v_t`。
- 训练时一次前向（prefix + suffix 一起），推理时为了省算力先跑一次 prefix 把 KV cache 填好，之后每一步去噪只跑 suffix。

---

## 1. 数据进入模型之前的全部变换

### 1.1 DataLoader 入口

`src/openpi/training/data_loader.py`

- `create_data_loader` (`data_loader.py:314`) — 训练入口，根据是否 RLDS 分发。
- `create_torch_data_loader` (`data_loader.py:362`) — 构造 `TorchDataLoader`。
- `transform_dataset` (`data_loader.py:263`) — 把变换按顺序套到 dataset 上：

```263:282:src/openpi/training/data_loader.py
def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )
```

顺序非常重要：**repack → data → normalize → model**。
- `repack_transforms`：把原始 LeRobot 字段名映射到 `{image, state, actions, prompt}` 的标准 schema（`RepackTransform` in `transforms.py:80`）。
- `data_transforms`：业务相关，例如 `AlohaInputs` / `DeltaActions` (`transforms.py:231`) 把绝对动作转成相对 state 的 delta。
- `Normalize` (`transforms.py:115`)：默认 pi0.5 用 **quantile 归一化**（见 `config.py:189` `use_quantile_norm=model_config.model_type != ModelType.PI0`），把每个维度映射到 `[-1, 1]`，对常量维 (span<0.005) 直接置 0：
  ```141:147:src/openpi/transforms.py
  def _normalize_quantile(self, x, stats: NormStats):
      assert stats.q01 is not None
      assert stats.q99 is not None
      q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
      span = q99 - q01
      # Near-constant dims (span < 0.005): map to 0 to avoid noise amplification.
      return np.where(span < 0.005, 0.0, (x - q01) / (span + 1e-6) * 2.0 - 1.0)
  ```
- `model_transforms`：由 `ModelTransformFactory` (`config.py:109`) 生成，pi0.5 走 `ModelType.PI05` 分支：
  ```128:140:src/openpi/training/config.py
  case _model.ModelType.PI05:
      assert isinstance(model_config, pi0_config.Pi0Config)
      return _transforms.Group(
          inputs=[
              _transforms.InjectDefaultPrompt(self.default_prompt),
              _transforms.ResizeImages(224, 224),
              _transforms.TokenizePrompt(
                  _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                  discrete_state_input=model_config.discrete_state_input,
              ),
              _transforms.PadStatesAndActions(model_config.action_dim),
          ],
      )
  ```
  - `TokenizePrompt` (`transforms.py:275`) 调用 `PaligemmaTokenizer.tokenize(prompt, state)`，对 pi0.5 会把 **state 离散化为文本 token** 拼进 prompt（`discrete_state_input=True` 默认即 pi05，见 `pi0_config.py:60`）。
  - `PadStatesAndActions` (`transforms.py:355`) 把 state 和 actions pad 到 `action_dim`（默认 32），让不同机器人共用同一个 action expert。

### 1.2 batch 出来后的封装

`DataLoaderImpl.__iter__` (`data_loader.py:644`) 把每个 batch 转成 `Observation` 对象 + actions：

```644:646:src/openpi/training/data_loader.py
    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
```

`Observation` 定义在 `src/openpi/models/model.py:82`：

```90:107:src/openpi/models/model.py
    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
```

`Observation.from_dict` (`model.py:109`) 还会顺手把 uint8 图像归一化到 `[-1, 1]`：

```116:120:src/openpi/models/model.py
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
```

**所以进入 `compute_loss` 时的张量形状**（`batch_size=B`，3 个相机，state 维 `s`，动作 `ah×ad`）：
- `obs.images[name]` : `float32[B, 224, 224, 3]` ∈ `[-1,1]`
- `obs.image_masks[name]` : `bool[B]`
- `obs.state` : `float32[B, s]`（已归一化）
- `obs.tokenized_prompt` : `int32[B, L]`（L=200 for pi05）
- `obs.tokenized_prompt_mask` : `bool[B, L]`
- `actions` : `float32[B, action_horizon, action_dim]`（已归一化、已 pad）

`IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")` (`model.py:39`)。

---

## 2. 模型对数据的处理：`Pi0` 类

文件：`src/openpi/models/pi0.py`，配置：`src/openpi/models/pi0_config.py`。

### 2.1 构造：双专家 Gemma + SigLIP 视觉编码器

`Pi0.__init__` (`pi0.py:67`)：

```81:112:src/openpi/models/pi0.py
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        if config.continuous_state_input or not config.pi05:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if not config.pi05:
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
```

要点：
- `_gemma.Module` 接收 **configs 列表**（两个专家的 Config），`pi05=True` 时整体启用 `adarms`，并用 `use_adarms=[False, True]` 表示 **只有 action expert (expert 1) 用 adaRMSNorm**。
- 视觉编码器是 **SigLIP So400m/14**：224×224 输入 → 14×14 patch = 256 个 image token，每个 token 投影到 `paligemma_config.width=2048`。
- 关键线性层：
  - `action_in_proj`：`action_dim → action_expert_width(1024)`，把每个 noisy action token 投到 action expert 的 embedding 空间。
  - `time_mlp_in/out`：仅 pi0.5 有，把 timestep 的 sin-cos embedding 转成 adaRMS 的 cond 向量。
  - `state_proj`：pi0.5 默认 `discrete_state_input=True`，state 已经进 prompt 了，所以 **不创建** `state_proj`（除非显式开 `continuous_state_input`）。
  - `action_out_proj`：`action_expert_width → action_dim`，把 transformer 输出投回动作空间，得到速度场 `v_t`。

### 2.2 前缀编码：`embed_prefix`

`pi0.py:117`，**prefix = 图像 + 语言**，全部由 PaliGemma (expert 0) 处理：

```117:149:src/openpi/models/pi0.py
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
```

- 每张图过 SigLIP 得到 `B×256×2048` 的 token；3 张图共 `768` 个 image token。
- 文本通过 `PaliGemma.llm(tokens, method="embed")` 走 `Embedder.encode` (`gemma.py:148`)，得到 `B×L×2048`。
- `ar_mask=False` 表示这些 token **互相可见**（prefix-LM 风格的双向 attention）；后续 `make_attn_mask` 会用 cumsum 把它转成 attention mask。
- `input_mask` 把 padding token 排掉。

### 2.3 后缀编码：`embed_suffix`

`pi0.py:151`，**suffix = (可选 state token) + noisy action tokens + timestep**，由 action expert (expert 1) 处理：

```151:198:src/openpi/models/pi0.py
    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if self.continuous_state_input or not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond
```

pi0.5 的关键差异点：
1. **state 不进 suffix**（因为已经在 prompt 里了），除非显式开 `continuous_state_input`。
2. **timestep 不和 action 拼接**，而是单独走 `time_mlp_in → swish → time_mlp_out → swish` 得到 `adarms_cond`（`B×1024`），后面用来调制 action expert 的 RMSNorm。
3. `ar_mask = [True] + [False]*(ah-1)`：action horizon 内部互相可见（一块儿预测），但 **prefix 不能 attend 到 action token**（`True` 表示从这一位开始 cumsum +1，把后面的 token 与前面隔开）。

timestep embedding 用的是 `posemb_sincos` (`pi0.py:48`)，`min_period=4e-3, max_period=4.0` —— 因为 flow matching 的时间 `t∈[0,1]`，需要对这个小区间敏感。

### 2.4 Attention mask 构造

`make_attn_mask` (`pi0.py:19`)：

```19:44:src/openpi/models/pi0.py
def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

对 pi0.5 整条序列的 `ar_mask` 大致是：

```
[图像×768 (F), 文本×L (F), (state×1 (T) 若开), action×ah (T, F, F, ..., F)]
                ↑ prefix                                  ↑ suffix
```

效果：
- 图像 ↔ 文本：双向
- suffix 不能被 prefix 看到（保持 causal 性质，避免 action token 影响 vision/language 表征）
- action horizon 内部：互相可见（chunk-level prediction）

### 2.5 双专家 Gemma 前向

`PaliGemma.llm([prefix_tokens, suffix_tokens], mask, positions, adarms_cond=[None, adarms_cond])`，进入 `gemma.py::Module.__call__` (`gemma.py:389`)：

```389:411:src/openpi/models/gemma.py
    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache
```

- `self.layers` 是 `nn.scan(Block, ...)` (`gemma.py:365`)，沿 depth 维 scan，权重 `params` 维度为 `[depth, ...]`。
- 每个 `Block` (`gemma.py:284`) 内部：对每个 expert 各跑一次 `RMSNorm(cond=adarms_cond[i])`，然后 **Attention 在所有 expert 之间共享**（拼接 QKV），FFN 又分专家：
  ```297:333:src/openpi/models/gemma.py
      attn = Attention(configs=self.configs, name="attn")

      pre_attn = []
      gates = []
      for i, x in enumerate(xs):
          if x is not None:
              x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
          pre_attn.append(x)
          gates.append(gate if x is not None else None)

      pre_attn = sharding.activation_sharding_constraint(pre_attn)
      post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
      ...
      out = []
      gates = []
      for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
          if x is not None:
              x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
              x = lora.FeedForward(  # noqa: PLW2901
                  features=config.width,
                  hidden_dim=config.mlp_dim,
                  name=_name("mlp", i),
                  lora_config=config.lora_configs.get("ffn"),
              )(x)
          out.append(x)
          gates.append(gate if x is not None else None)
  ```
- **Attention** (`gemma.py:158`)：每个 expert 用自己的 `q_einsum/kv_einsum` 把自己的 token 投到 Q/K/V（**共享 head_dim/num_heads/num_kv_heads**，这是混合专家 self-attention 能成立的硬约束），然后 `jnp.concatenate(q,k,v, axis=1)` 跨专家拼起来做 GQA：
  ```201:217:src/openpi/models/gemma.py
      q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

      q = _apply_rope(q, positions=positions)
      q *= self.configs[0].head_dim ** -0.5

      k = _apply_rope(k, positions=positions)
  ```
  这样 PaliGemma token 既能 attend 到自己，也能 attend 到 action expert token（如果 mask 允许），反过来 action token 也能 attend 到 PaliGemma token 的 K/V —— 但 mask 会阻止它发生反向（prefix 看不到 suffix）。
- **adaRMSNorm** (`gemma.py:113`)：当传 `cond` 时，RMSNorm 不仅缩放还 shift，并产生一个 gate 控制残差强度：
  ```126:131:src/openpi/models/gemma.py
          # adaptive RMSNorm
          modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
          scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
          normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
          return normed_inputs.astype(dtype), gate
  ```
  这就是 pi0.5 把 flow matching timestep 注入 action expert 的方式 —— DiT 风格的 modulation。
- 残差通过 `_gated_residual` (`gemma.py:453`)：`x + y * gate`（gate 为 None 时退化为普通残差）。

最后 `Pi0.compute_loss_with_debug` 拿 `suffix_out` 的最后 `action_horizon` 个 token 投回 action_dim 得到 `v_t`：

```256:259:src/openpi/models/pi0.py
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
```

---

## 3. Flow Matching 训练：`compute_loss_with_debug`

文件 `pi0.py:207`。这是训练时被 `train_step` 调用的核心。

```207:262:src/openpi/models/pi0.py
    def compute_loss_with_debug(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_aug_crop_enabled=self.image_aug_crop_enabled,
            ...
            image_aug_prob=self.image_aug_prob,
        )
        ...
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        if self.loss_action_dims is not None:
            action_dim_mask = jnp.isin(jnp.arange(self.action_dim), jnp.array(self.loss_action_dims))
        else:
            action_dim_mask = jnp.arange(self.action_dim) < self.loss_action_dim
        actions = jnp.where(action_dim_mask, actions, 0.0)
        noise = jnp.where(action_dim_mask, noise, 0.0)
        action_dim_count = jnp.sum(action_dim_mask)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        sq_err = jnp.square(v_t - u_t)
        active_sq_err = jnp.where(action_dim_mask, sq_err, 0.0)
        chunked_loss = jnp.sum(active_sq_err, axis=-1) / action_dim_count
        dim_loss = jnp.mean(active_sq_err, axis=tuple(range(active_sq_err.ndim - 1)))
```

### 3.1 Flow Matching 的数学

约定（注意代码注释里写的：与 pi0 论文方向相反，与扩散文献一致）：
- `t = 1` → 噪声分布 `ε ~ N(0, I)`
- `t = 0` → 真实动作分布 `a`
- 概率路径：`x_t = t·ε + (1-t)·a`  （线性插值）
- 速度场：`u_t = dx_t/dt = ε - a`

训练目标是最小化 `||v_θ(x_t, t, obs) - u_t||²`。

### 3.2 关键采样细节

- **时间采样**：`time = beta(1.5, 1) * 0.999 + 0.001`。Beta(1.5,1) 偏向 `t→1`（噪声端），论文里这么做是因为动作端 (`t→0`) 容易学（直接回归就行），噪声端更需要训练。`*0.999+0.001` 把 `[0,1]` 裁到 `[0.001, 1.0]`，避免数值边界问题。
- **噪声**：`noise ~ N(0, I)`，shape 与 `actions` 相同 `(B, ah, ad)`。
- **`action_dim_mask`**：处理 pad 维。如果 `loss_action_dim < action_dim`（例如 7-DoF 机器人 pad 到 32），只对真实维度算 loss，pad 维置 0。
- **`x_t` / `u_t`**：向量化构造，`time_expanded` 广播到 `(B, ah, ad)`。

### 3.3 损失归约

```260:263:src/openpi/models/pi0.py
        sq_err = jnp.square(v_t - u_t)
        active_sq_err = jnp.where(action_dim_mask, sq_err, 0.0)
        chunked_loss = jnp.sum(active_sq_err, axis=-1) / action_dim_count
```

- `chunked_loss` 形状 `(B, ah)`：**每个时间步、每个 batch 元素一个标量**，按 active 维数取均值。
- 在 `train.py:183` 上层再 `jnp.mean(chunked_loss)` 得到最终标量 loss。
- `dim_loss` 仅用于 debug/wandb 日志，按维度看哪些维度学得差。

### 3.4 上层训练驱动

`scripts/train.py:164` 的 `train_step`：

```177:208:scripts/train.py
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        train_image_augment = getattr(config, "train_image_augment", True)
        chunked_loss = model.compute_loss(rng, observation, actions, train=train_image_augment)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
    grad_norm = optax.global_norm(grads)
    grads_finite = jnp.isfinite(grad_norm)
    loss_finite = jnp.isfinite(loss)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    ...
    candidate_params = optax.apply_updates(params, updates)
    update_finite = ...
    updates = jax.tree.map(lambda update: jnp.where(update_finite, update, jnp.zeros_like(update)), updates)
    new_params = optax.apply_updates(params, updates)
```

- `config.trainable_filter`：由 `Pi0Config.get_freeze_filter` (`pi0_config.py:114`) 生成，LoRA 微调时只放行 lora 参数；全量微调时则是全部可训练。
- `nnx.value_and_grad(..., argnums=diff_state)`：只对 trainable 参数求梯度。
- **梯度安全**：若 `loss/grads/updates/candidate_params` 任意一个出现 NaN/Inf，就把这一步的 update 全置 0（`update_finite` mask），同时保留旧 opt_state，避免污染权重。
- EMA：`ema_decay` 不为 None 时维护参数的指数滑动平均，常用于推理时用 EMA 权重。

### 3.5 图像增广

`preprocess_observation` (`model.py:144`)，仅 `train=True` 时做 RandomCrop+Resize+Rotate（非 wrist 相机）+ ColorJitter，全部通过 `augmax` 库在 JAX 里实现。推理路径走 `train=False` 分支，只 resize。

---

## 4. Flow Matching 推理：`sample_actions`

`pi0.py:286`：

```286:348:src/openpi/models/pi0.py
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            ...
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
```

### 4.1 推理流程要点

1. **初始噪声** `x_1 ~ N(0, I)`，`time=1.0`。
2. **KV cache 预填**：先用 prefix 跑一遍 `PaliGemma.llm([prefix_tokens, None], ...)`，返回的 `kv_cache` 包含 prefix 在每层的 K/V。之后每一步去噪只需对 suffix token 重新跑 attention，suffix 的 Q 去 attend 已缓存的 prefix K/V + 当前 suffix K/V。
3. **Euler ODE 积分**：`dt = -1/num_steps`（默认 10 步），更新式
   ```
   x_{t+dt} = x_t + dt · v_θ(x_t, t, obs)        # dt<0, 沿速度场反向积分
   t        = t + dt
   ```
   从 `t=1` 积分到 `t=0`，最终 `x_0` 就是预测的动作 chunk。
4. **mask 构造**：suffix 的 query 既要 attend 到 prefix（用 `prefix_mask` 复制成 `(B, suffix_len, prefix_len)`）也要 attend 到 suffix 内部（`suffix_attn_mask`），拼成 `(B, suffix_len, prefix_len + suffix_len)`。
5. **positions**：suffix 的位置从 prefix 长度之后开始累加，保证 RoPE 编码连续。
6. `jax.lax.while_loop` 把循环编译进 XLA，避免 Python 循环带来的重新 trace。

### 4.2 训练 vs 推理对照

| 项 | 训练 (`compute_loss`) | 推理 (`sample_actions`) |
|---|---|---|
| 前向次数 | 1 次（prefix+suffix 一起） | 1 + num_steps 次（1 次 prefix 缓存 + num_steps 次 suffix） |
| `x_t` 来源 | `t·ε + (1-t)·a`（用真实 action 构造） | 上一步 `x_t + dt·v_t`（迭代） |
| 目标 `u_t` | `ε - a`，作为回归目标 | 不需要，直接用模型输出 `v_t` 积分 |
| `t` 采样 | `Beta(1.5,1)`，每个样本一个 | 从 1.0 等步长降到 0.0 |
| KV cache | 不用 | 用，prefix 只跑一次 |
| adaRMS cond | `time_mlp(time_emb)` | 同上，每步 `time` 不同 |

---

## 5. 推理封装到上层 policy

`src/openpi/policies/policy.py:24` 的 `Policy.infer`：

- 输入原始 obs dict → `_input_transform`（推理时也走 model_transforms 之前那些） → `Observation.from_dict` → `model.sample_actions(rng, obs, **sample_kwargs)` → `_output_transform`（包含 `Unnormalize` 把 `[-1,1]` 还原成原始动作空间、`AbsoluteActions` 把 delta 还原成绝对动作等）。

```67:101:src/openpi/policies/policy.py
    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        ...
        outputs = self._output_transform(outputs)
```

注意：推理时 `Unnormalize` 是 `Normalize` 的逆操作，对 quantile 模式：

```204:208:src/openpi/transforms.py
        return np.where(
            span < 0.005,
            (q01 + q99) / 2.0,
            (x + 1.0) / 2.0 * (span + 1e-6) + q01,
        )
```

---

## 6. 调试 / 验证自己理解的建议路径

1. **跑单测**：`src/openpi/models/model_test.py` 里直接 `model.compute_loss(key, obs, act)` / `model.sample_actions(key, obs, num_steps=10)`，可以用 `debug_pi05` 配置（`config.py:1055`，dummy 变体）在 CPU 上秒跑，打断点观察每一步形状。
2. **打印形状**：在 `embed_prefix` / `embed_suffix` 返回处打印 `tokens.shape, input_mask.shape, ar_mask`，能看到：
   - prefix: `(B, 768+L, 2048)`
   - suffix (pi05 默认): `(B, ah, 1024)`，`adarms_cond: (B, 1024)`
3. **对照论文**：Flow Matching 公式 `x_t = t·ε + (1-t)·a` 与 `u_t = ε - a` 对应 pi0 paper 中的 eq.（注意 t 方向反转的注释）。
4. **自己数据集的 norm stats**：跑 `scripts/compute_norm_stats.py --config-name=<your-pi05-config>`，结果存到 `assets/<asset_id>/`，`DataConfigFactory._load_norm_stats` (`config.py:192`) 加载，否则 `transform_dataset` 会直接抛错。

---

## 7. 关键文件 / 函数索引（速查）

| 关心的事 | 文件 | 函数 / 行 |
|---|---|---|
| 数据变换 pipeline | `src/openpi/training/data_loader.py` | `transform_dataset:263`, `create_torch_data_loader:362` |
| model_transforms 工厂 | `src/openpi/training/config.py` | `ModelTransformFactory:109` (PI05 分支 128) |
| Quantile 归一化 | `src/openpi/transforms.py` | `Normalize._normalize_quantile:141` |
| Observation 结构 | `src/openpi/models/model.py` | `Observation:82`, `from_dict:109`, `preprocess_observation:144` |
| Pi0.5 模型构造 | `src/openpi/models/pi0.py` | `Pi0.__init__:67` |
| Prefix 编码 (图+文) | `src/openpi/models/pi0.py` | `embed_prefix:117` |
| Suffix 编码 (action+time) | `src/openpi/models/pi0.py` | `embed_suffix:151` |
| timestep sin-cos | `src/openpi/models/pi0.py` | `posemb_sincos:48` |
| attention mask | `src/openpi/models/pi0.py` | `make_attn_mask:19` |
| 双专家 Transformer | `src/openpi/models/gemma.py` | `Module.__call__:389`, `Block:284`, `Attention:158` |
| adaRMSNorm | `src/openpi/models/gemma.py` | `RMSNorm:113` |
| RoPE | `src/openpi/models/gemma.py` | `_apply_rope:424` |
| Flow matching 训练 | `src/openpi/models/pi0.py` | `compute_loss_with_debug:207` |
| Flow matching 推理 | `src/openpi/models/pi0.py` | `sample_actions:286` |
| train_step / value_and_grad | `scripts/train.py` | `train_step:164`, `loss_fn:178` |
| 推理封装 | `src/openpi/policies/policy.py` | `Policy.infer:67` |
| 反归一化 | `src/openpi/transforms.py` | `Unnormalize:162` |

---

## 8. 一句话总结

> **pi0.5 = PaliGemma (vision-language 前缀) + Action Expert (动作后缀，受 timestep 通过 adaRMS 调制) 共享 self-attention 的双专家 Gemma。**
>
> 训练时把真实 action 和噪声做线性插值得到 `x_t`，模型一次前向预测速度 `v_t`，回归目标 `u_t = ε - a`，loss 是按 active 维度平均的 MSE。推理时从纯噪声出发，用 Euler 法按 `v_t` 反向积分 `num_steps` 步得到动作 chunk；prefix 只前向一次填 KV cache，之后每步只跑 suffix。
