"""Model builders for Pi0.5 LoRA fine-tuning.

This module intentionally stays thin and reuses the local LeRobot reference
implementation wherever possible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from peft import LoraConfig, PeftConfig
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_STATE
from pi05.common.config.schema import ExperimentConfig
from pi05.common.model.tactile_encoder import TactileTemporalEncoder


DEFAULT_PRETRAINED_PATH = "lerobot/pi05_base"
DEFAULT_CAMERAS = ("top", "left_wrist", "right_wrist")
TACTILE_CAMERAS = frozenset({"left_tactile", "right_tactile"})
DEFAULT_IMAGE_KEYS = tuple(f"observation.images.{camera}" for camera in DEFAULT_CAMERAS)
DEFAULT_STATE_DIM = 26
DEFAULT_ACTION_DIM = 14
_PRETRAINED_PATH_UNSET = object()
EXPERT_ONLY_LORA_TARGETS = (
    r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))"
)


def build_pi05_with_lora(
    config: ExperimentConfig | Mapping[str, Any] | None = None,
    pretrained_path: str | Path | None | object = _PRETRAINED_PATH_UNSET,
    *,
    config_dict: Mapping[str, Any] | None = None,
    init_adapter_from: Path | None = None,
) -> torch.nn.Module:
    """Load a Pi0.5 policy and wrap it with LoRA adapters.

    Args:
        config: Parsed experiment config. A raw mapping is accepted for backward
            compatibility with older local scripts.
        pretrained_path: Hugging Face repo id or local checkpoint directory.
        init_adapter_from: Optional path to a pre-trained LoRA adapter directory
            containing adapter_config.json and adapter_model.safetensors. When
            set, the saved adapter config defines the LoRA structure and the
            saved adapter weights are loaded as trainable parameters.
    """

    if config is None:
        config = config_dict
    if config is None:
        raise ValueError("build_pi05_with_lora requires `config` or legacy `config_dict`.")
    exp_config = _as_experiment_config(config)
    if pretrained_path is _PRETRAINED_PATH_UNSET:
        pretrained_path = exp_config.model.pretrained_path
    lora_cfg = exp_config.lora
    model_cfg = exp_config.model
    device = _resolve_device(model_cfg.device)
    dtype = model_cfg.dtype
    gradient_checkpointing = model_cfg.gradient_checkpointing
    train_expert_only = model_cfg.train_expert_only
    allow_random_init_peft = model_cfg.allow_random_init_peft
    chunk_size = model_cfg.chunk_size
    n_action_steps = model_cfg.n_action_steps
    state_dim = model_cfg.state_dim
    action_dim = model_cfg.action_dim
    max_action_dim = model_cfg.max_action_dim
    attention_implementation = model_cfg.attention_implementation
    if action_dim != DEFAULT_ACTION_DIM:
        raise ValueError(f"Pi0.5 action_dim is locked to {DEFAULT_ACTION_DIM}, got {action_dim}.")
    if max_action_dim != action_dim:
        raise ValueError(f"max_action_dim must equal action_dim ({action_dim}) for the new schema, got {max_action_dim}.")
    image_keys = _image_keys_from_cameras(exp_config.data.cameras)

    if pretrained_path in (None, "", "random_init", "dummy"):
        if not allow_random_init_peft:
            raise ValueError(
                "Random-init PEFT is disabled by default. Set "
                "`model.allow_random_init_peft=true` only for smoke tests."
            )
        policy_config = PI05Config(
            device=device,
            dtype=dtype,
            gradient_checkpointing=gradient_checkpointing,
            train_expert_only=train_expert_only,
            paligemma_variant=model_cfg.paligemma_variant,
            action_expert_variant=model_cfg.action_expert_variant,
            attention_implementation=attention_implementation,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_action_dim=action_dim,
            pretrained_path=Path("/tmp/pi05_random_init_smoke"),
        )
        _ensure_pi05_feature_specs(
            policy_config,
            state_dim=state_dim,
            action_dim=action_dim,
            image_keys=image_keys,
        )
        policy = PI05Policy(policy_config)
    else:
        policy_config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=pretrained_path,
            device=device,
            dtype=dtype,
            gradient_checkpointing=gradient_checkpointing,
        )
        if not isinstance(policy_config, PI05Config):
            raise TypeError(f"Expected PI05Config, got {type(policy_config).__name__}")
        # LeRobot's PreTrainedConfig.from_pretrained only applies explicit
        # overrides through cli_overrides; plain kwargs are ignored. Re-apply the
        # runtime training overrides here so each DDP rank builds the model with
        # the intended device/dtype/checkpointing settings instead of stale
        # checkpoint values such as device=mps or dtype=float32.
        policy_config.device = device
        policy_config.dtype = dtype
        policy_config.gradient_checkpointing = gradient_checkpointing
        # Override pretrained horizon settings so the model, masks, and preprocessor
        # stay aligned with our dataset-side chunking configuration.
        policy_config.chunk_size = chunk_size
        policy_config.n_action_steps = n_action_steps
        policy_config.pretrained_path = Path(str(pretrained_path))
        policy_config.train_expert_only = train_expert_only
        policy_config.attention_implementation = attention_implementation
        _ensure_pi05_feature_specs(
            policy_config,
            state_dim=state_dim,
            action_dim=action_dim,
            image_keys=image_keys,
        )
        policy = PI05Policy.from_pretrained(
            pretrained_path,
            config=policy_config,
            strict=False,
        )
        policy.config.pretrained_path = Path(str(pretrained_path))
        policy.config.train_expert_only = train_expert_only

    _resize_pi05_action_projections(policy, action_dim=action_dim)
    _force_pi05_attention_implementation(policy, attention_implementation=attention_implementation)
    _attach_tactile_encoder(policy, exp_config)

    # Our custom dataloader already normalizes state/action to [-1, 1].
    policy_config.normalization_mapping["STATE"] = NormalizationMode.IDENTITY
    policy_config.normalization_mapping["ACTION"] = NormalizationMode.IDENTITY

    if init_adapter_from is not None:
        adapter_path = _resolve_adapter_path(init_adapter_from)
        peft_config = PeftConfig.from_pretrained(str(adapter_path))
        peft_model = policy.wrap_with_peft(peft_config=peft_config)
        _load_adapter_weights_strict(peft_model, adapter_path)
        _enable_lora_parameters(peft_model)
        _enable_tactile_parameters(peft_model)
        _load_tactile_weights(peft_model, adapter_path)
        if hasattr(peft_model, "config"):
            peft_model.config.use_peft = True
        print(f"[builder] Loaded trainable LoRA adapter from: {adapter_path}")
        _print_loaded_lora_key_comparison(peft_model, adapter_path)
        _print_loaded_lora_module_comparison(peft_model, adapter_path)
    else:
        peft_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=_resolve_peft_target_modules(exp_config),
            bias="none",
        )
        peft_model = policy.wrap_with_peft(peft_config=peft_config)
        _enable_tactile_parameters(peft_model)
    peft_model.pi05_policy_config = policy.config
    _print_lora_target_modules(peft_model)

    _enable_gradient_checkpointing(peft_model)
    _print_trainable_ratio(peft_model)

    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()

    return peft_model


def _attach_tactile_encoder(policy: PI05Policy, config: ExperimentConfig) -> None:
    if not config.tactile.enabled:
        policy.model.tactile_encoder = None
        return
    vlm_hidden_dim = policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    encoder = TactileTemporalEncoder(config.tactile, vlm_hidden_dim)
    policy.model.tactile_encoder = encoder.to(next(policy.parameters()).device)


def _enable_tactile_parameters(model: torch.nn.Module) -> None:
    tactile_encoder = _find_tactile_encoder(model)
    if tactile_encoder is None:
        return
    for parameter in tactile_encoder.parameters():
        parameter.requires_grad = True


def _find_tactile_encoder(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        tactile_encoder = getattr(module, "tactile_encoder", None)
        if isinstance(tactile_encoder, torch.nn.Module):
            return tactile_encoder
    return None


def _load_tactile_weights(model: torch.nn.Module, adapter_path: Path) -> None:
    weights_path = adapter_path / "tactile_encoder.safetensors"
    tactile_encoder = _find_tactile_encoder(model)
    if tactile_encoder is None or not weights_path.is_file():
        return
    tactile_encoder.load_state_dict(load_safetensors(str(weights_path), device="cpu"), strict=True)
    print(f"[builder] Loaded tactile encoder from: {weights_path}")



def _resolve_peft_target_modules(config: ExperimentConfig) -> str | list[str]:
    if config.model.train_expert_only:
        print("[builder] train_expert_only=true; restricting LoRA targets to action expert/projection layers.")
        return EXPERT_ONLY_LORA_TARGETS

    raw_targets = config.lora.target_modules
    if isinstance(raw_targets, str):
        return raw_targets
    return list(raw_targets)


def _image_keys_from_cameras(cameras: tuple[str, ...]) -> tuple[str, ...]:
    image_keys = tuple(f"observation.images.{camera}" for camera in cameras if camera not in TACTILE_CAMERAS)
    if not image_keys:
        raise ValueError("At least one image camera/key must be configured.")
    return image_keys


def _resolve_adapter_path(configured_path: Path) -> Path:
    """Resolve an adapter directory, accepting a checkpoint directory containing adapter/."""
    adapter_path = configured_path.expanduser().resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"init_adapter_from does not exist: {adapter_path}")

    for candidate in (adapter_path, adapter_path / "adapter"):
        if (
            (candidate / "adapter_config.json").is_file()
            and (candidate / "adapter_model.safetensors").is_file()
        ):
            if candidate != adapter_path:
                print(f"[builder] Resolved adapter subdirectory: {candidate}")
            return candidate

    raise FileNotFoundError(
        "Could not find adapter_config.json and adapter_model.safetensors in "
        f"{adapter_path} or {adapter_path / 'adapter'}"
    )


def _load_adapter_weights_strict(model: torch.nn.Module, adapter_path: Path) -> None:
    """Load adapter weights without PEFT's version-sensitive tensor-parallel helper."""
    weights_path = adapter_path / "adapter_model.safetensors"
    saved_state = load_safetensors(str(weights_path), device="cpu")
    model_state = model.state_dict()
    mapped_state: dict[str, torch.Tensor] = {}
    missing_keys: list[str] = []
    mismatched_keys: list[str] = []

    for saved_key, tensor in saved_state.items():
        model_key = _resolve_saved_adapter_key(saved_key, model_state)
        if model_key is None:
            missing_keys.append(saved_key)
            continue
        if model_state[model_key].shape != tensor.shape:
            mismatched_keys.append(
                f"{saved_key}: checkpoint={tuple(tensor.shape)} model={tuple(model_state[model_key].shape)}"
            )
            continue
        mapped_state[model_key] = tensor

    if missing_keys or mismatched_keys:
        raise RuntimeError(
            f"Adapter is incompatible with the current base model. "
            f"Missing keys: {missing_keys}; shape mismatches: {mismatched_keys}"
        )

    load_result = model.load_state_dict(mapped_state, strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError(f"Adapter keys rejected by current model: {load_result.unexpected_keys}")
    print(f"[builder] Loaded {len(mapped_state)} adapter weight(s) from: {weights_path}")


def _resolve_saved_adapter_key(
    saved_key: str,
    model_state: Mapping[str, torch.Tensor],
) -> str | None:
    """Map PEFT's saved key format to the in-memory default adapter key."""
    if saved_key in model_state:
        return saved_key
    for marker in (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B."):
        if marker in saved_key:
            candidate = saved_key.replace(marker, f"{marker}default.", 1)
            if candidate in model_state:
                return candidate
    return None


def _enable_lora_parameters(model: torch.nn.Module) -> None:
    """Ensure loaded adapter weights remain trainable after using an inference-mode config."""
    enabled_params = 0
    enabled_tensors = 0
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        parameter.requires_grad = True
        enabled_params += parameter.numel()
        enabled_tensors += 1

    if enabled_tensors == 0:
        raise RuntimeError("No LoRA parameters were found to enable for training.")
    print(f"[builder] Enabled {enabled_params:,} LoRA parameter(s) across {enabled_tensors} tensor(s).")


def _print_lora_target_modules(model: torch.nn.Module) -> None:
    """Print only the number of base-model modules that received LoRA layers."""
    matched_modules = _lora_module_names_from_model(model)
    print(f"[builder] LoRA matched modules: {len(matched_modules)}")


def _print_loaded_lora_key_comparison(model: torch.nn.Module, adapter_path: Path) -> None:
    """Compare saved adapter LoRA keys with LoRA parameters injected into the model."""
    weights_path = adapter_path / "adapter_model.safetensors"
    if not weights_path.exists():
        print(f"[builder] LoRA key comparison skipped; weights not found: {weights_path}")
        return

    with safe_open(weights_path, framework="pt", device="cpu") as adapter_state:
        saved_lora_keys = {
            _normalize_lora_parameter_key(key)
            for key in adapter_state.keys()
            if "lora_" in key
        }
    current_lora_keys = {
        _normalize_lora_parameter_key(name)
        for name, _ in model.named_parameters()
        if "lora_" in name
    }
    matched_keys = saved_lora_keys & current_lora_keys
    missing_from_model = saved_lora_keys - current_lora_keys
    missing_from_adapter = current_lora_keys - saved_lora_keys

    print(
        f"[builder] LoRA key comparison: matched={len(matched_keys)}, "
        f"saved_not_matched={len(missing_from_model)}, "
        f"current_extra={len(missing_from_adapter)}"
    )


def _print_loaded_lora_module_comparison(model: torch.nn.Module, adapter_path: Path) -> None:
    """Compare adapter-saved LoRA module heads with the modules PEFT matched."""
    weights_path = adapter_path / "adapter_model.safetensors"
    if not weights_path.exists():
        print(f"[builder] LoRA module comparison skipped; weights not found: {weights_path}")
        return

    saved_modules = _lora_module_names_from_adapter(weights_path)
    matched_modules = set(_lora_module_names_from_model(model))
    common_modules = saved_modules & matched_modules
    missing_from_model = saved_modules - matched_modules
    missing_from_adapter = matched_modules - saved_modules

    print(
        f"[builder] LoRA module comparison: matched={len(common_modules)}, "
        f"saved_not_matched={len(missing_from_model)}, "
        f"current_extra={len(missing_from_adapter)}"
    )


def _lora_module_names_from_adapter(weights_path: Path) -> set[str]:
    with safe_open(weights_path, framework="pt", device="cpu") as adapter_state:
        return {
            key.split(".lora_", 1)[0]
            for key in adapter_state.keys()
            if ".lora_" in key
        }


def _lora_module_names_from_model(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]


def _normalize_lora_parameter_key(key: str) -> str:
    """Normalize PEFT save/runtime naming differences before comparing keys."""
    return key.removesuffix(".weight").replace(".default", "")


def get_pi05_policy_config(model: torch.nn.Module) -> Any:
    """Locate the PI0.5 policy config across common PEFT/wrapper layers."""
    if hasattr(model, "pi05_policy_config"):
        return model.pi05_policy_config
    for candidate in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ):
        if candidate is not None and hasattr(candidate, "config"):
            return candidate.config
    raise AttributeError("Could not locate PI0.5 policy config on wrapped model.")


def _as_experiment_config(config: ExperimentConfig | Mapping[str, Any]) -> ExperimentConfig:
    if isinstance(config, ExperimentConfig):
        return config
    return ExperimentConfig.from_mapping(config)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Enable gradient checkpointing on the inner PI05Pytorch model.

    The real ``gradient_checkpointing_enable`` lives on the inner ``PI05Pytorch``
    module (``PI05Policy.model``), which after PEFT wrapping is nested several
    levels deep (``peft_model.base_model.model.model``). We locate it by the
    ``gradient_checkpointing_enabled`` flag it defines, since shallow wrappers may
    expose unrelated no-op hooks.
    """
    target = None
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing_enabled") and callable(
            getattr(module, "gradient_checkpointing_enable", None)
        ):
            target = module
            break

    if target is not None:
        target.gradient_checkpointing_enable()
        print(f"[builder] Enabled gradient checkpointing on {type(target).__name__}")
        return

    # Fallback: shallow wrappers that expose a generic enable hook.
    for candidate in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "model", None),
    ):
        if candidate is None:
            continue
        hook = getattr(candidate, "gradient_checkpointing_enable", None)
        if callable(hook):
            hook()
            print(f"[builder] Enabled gradient checkpointing on {type(candidate).__name__}")
            return
    print("[builder] Gradient checkpointing hook not found; continuing without extra enable call.")


def _print_trainable_ratio(model: torch.nn.Module) -> None:
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    ratio = 0.0 if total_params == 0 else 100.0 * trainable_params / total_params
    print(
        f"[builder] trainable params: {trainable_params:,} / {total_params:,} "
        f"({ratio:.4f}%)"
    )


def _resize_pi05_action_projections(policy: PI05Policy, action_dim: int) -> None:
    pi05_model = getattr(policy, "model", None)
    if pi05_model is None:
        raise AttributeError("PI05Policy is missing inner model; cannot enforce action projection size.")

    action_in_proj = getattr(pi05_model, "action_in_proj", None)
    action_out_proj = getattr(pi05_model, "action_out_proj", None)
    if action_in_proj is None or action_out_proj is None:
        raise AttributeError("PI05 inner model is missing action projection layers.")
    if action_in_proj.in_features == action_dim and action_out_proj.out_features == action_dim:
        policy.config.max_action_dim = action_dim
        pi05_model.config.max_action_dim = action_dim
        return

    device = action_out_proj.weight.device
    dtype = action_out_proj.weight.dtype
    new_action_in = torch.nn.Linear(
        action_dim,
        action_in_proj.out_features,
        bias=action_in_proj.bias is not None,
        device=device,
        dtype=dtype,
    )
    new_action_out = torch.nn.Linear(
        action_out_proj.in_features,
        action_dim,
        bias=action_out_proj.bias is not None,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        copy_in_dim = min(action_dim, action_in_proj.in_features)
        new_action_in.weight[:, :copy_in_dim].copy_(action_in_proj.weight[:, :copy_in_dim])
        if action_in_proj.bias is not None:
            new_action_in.bias.copy_(action_in_proj.bias)

        copy_out_dim = min(action_dim, action_out_proj.out_features)
        new_action_out.weight[:copy_out_dim, :].copy_(action_out_proj.weight[:copy_out_dim, :])
        if action_out_proj.bias is not None:
            new_action_out.bias[:copy_out_dim].copy_(action_out_proj.bias[:copy_out_dim])

    pi05_model.action_in_proj = new_action_in
    pi05_model.action_out_proj = new_action_out
    # action projection 是真实最后输出层；这里强制 out_features=14，满足新 schema。
    policy.config.max_action_dim = action_dim
    pi05_model.config.max_action_dim = action_dim


def _force_pi05_attention_implementation(policy: PI05Policy, *, attention_implementation: str) -> None:
    """Force HF submodules used during deployment to use SDPA/FlashAttention instead of eager attention."""
    for config in _iter_attention_configs(policy):
        if config is None:
            continue
        _set_config_attr(config, "_attn_implementation", attention_implementation)
        _set_config_attr(config, "attn_implementation", attention_implementation)
    policy.config.attention_implementation = attention_implementation


def _iter_attention_configs(policy: PI05Policy):
    model = getattr(policy, "model", None)
    paligemma_with_expert = getattr(model, "paligemma_with_expert", None)
    paligemma = getattr(paligemma_with_expert, "paligemma", None)
    gemma_expert = getattr(paligemma_with_expert, "gemma_expert", None)
    candidates = [
        getattr(policy, "config", None),
        getattr(model, "config", None),
        getattr(paligemma, "config", None),
        getattr(getattr(paligemma, "config", None), "text_config", None),
        getattr(getattr(paligemma, "model", None), "config", None),
        getattr(getattr(getattr(paligemma, "model", None), "language_model", None), "config", None),
        getattr(gemma_expert, "config", None),
        getattr(getattr(gemma_expert, "model", None), "config", None),
    ]
    seen = set()
    for config in candidates:
        if config is None:
            continue
        marker = id(config)
        if marker in seen:
            continue
        seen.add(marker)
        yield config


def _set_config_attr(config: Any, name: str, value: str) -> None:
    try:
        setattr(config, name, value)
    except Exception:
        pass


def _ensure_pi05_feature_specs(
    policy_config: PI05Config,
    state_dim: int,
    action_dim: int,
    image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS,
) -> None:
    if policy_config.input_features is None:
        policy_config.input_features = {}
    if policy_config.output_features is None:
        policy_config.output_features = {}

    expected_image_keys = set(image_keys)
    for key, feature in list(policy_config.input_features.items()):
        if _is_visual_feature(feature) and key not in expected_image_keys:
            del policy_config.input_features[key]

    for image_key in image_keys:
        policy_config.input_features[image_key] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, *policy_config.image_resolution),
        )
    policy_config.input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE,
        # state 包含左右臂关节、灵巧手开合，以及左右末端 6D pose，总计 26D。
        shape=(state_dim,),
    )
    policy_config.output_features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION,
        # action head 仍死锁为 14D，不能随 observation.state 扩维。
        shape=(action_dim,),
    )


def _is_visual_feature(feature: Any) -> bool:
    feature_type = getattr(feature, "type", None)
    return feature_type is FeatureType.VISUAL or str(feature_type) in {"VISUAL", "FeatureType.VISUAL"}
