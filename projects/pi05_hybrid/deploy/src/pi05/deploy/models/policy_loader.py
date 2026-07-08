"""Bundle-only Pi0.5 policy loading for deployment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from peft import set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors

from pi05.common.config.schema import ExperimentConfig, load_experiment_config
from pi05.common.model.builder import build_pi05_with_lora, get_pi05_policy_config
from pi05.common.runtime.bundle import (
    EXPERIMENT_CONFIG_NAME,
    load_bundle_manifest,
    load_bundle_normalizers,
    resolve_bundle_adapter_dir,
)
from pi05.deploy.config.schema import DeployConfig
from pi05.deploy.runtime.shared_buffer import ObservationSnapshot


class Pi05PolicyRuntime:
    """Loaded Pi0.5 policy plus preprocessing and normalizer state."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        policy: Any,
        preprocessor: Any,
        state_normalizer: Any,
        action_normalizer: Any,
        device: torch.device,
        task: str,
        action_dim: int,
        output_chunk_size: int,
        clamp_normalized_action: bool,
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        self.model = model
        self.policy = policy
        self.preprocessor = preprocessor
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.device = device
        self.task = task
        self.action_dim = int(action_dim)
        self.output_chunk_size = int(output_chunk_size)
        self.clamp_normalized_action = bool(clamp_normalized_action)
        self._predict_fn = self.policy.predict_action_chunk
        self.compile_enabled = False
        if compile_model:
            self.compile_enabled = self._maybe_compile(compile_mode)

    def predict_action_chunk(self, observation: ObservationSnapshot) -> np.ndarray:
        """Run policy inference and return an unnormalized action chunk."""
        # Keep the complete LeRobot processor pipeline on CPU. In particular,
        # Pi05PrepareStateTokenizerProcessorStep discretizes state via NumPy;
        # moving state to CUDA before that step forces a GPU->CPU sync.
        batch = self.preprocessor(self._build_batch(observation))
        batch = _move_tensors_to_device(batch, self.device)
        with torch.inference_mode():
            norm_chunk = self._predict_fn(batch)
        norm_chunk = norm_chunk.detach().cpu().to(dtype=torch.float32)[0]
        if self.clamp_normalized_action:
            norm_chunk = norm_chunk.clamp(-1.0, 1.0)
        action_chunk = self.action_normalizer.unnormalize(norm_chunk).numpy().astype(np.float32, copy=False)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != self.action_dim:
            raise ValueError(f"Policy returned invalid action chunk shape: {action_chunk.shape}")
        return action_chunk[: self.output_chunk_size]

    def _build_batch(self, observation: ObservationSnapshot) -> dict[str, Any]:
        state = self.state_normalizer.normalize(
            torch.as_tensor(observation.encoded_state, dtype=torch.float32)
        )
        return {
            "observation.images.top": _require_cpu_float_tensor(observation.images["top"], "top image"),
            "observation.images.left_wrist": _require_cpu_float_tensor(
                observation.images["left_wrist"], "left_wrist image"
            ),
            "observation.images.right_wrist": _require_cpu_float_tensor(
                observation.images["right_wrist"], "right_wrist image"
            ),
            "observation.state": _require_cpu_float_tensor(state, "observation.state"),
            "task": self.task,
        }

    def _maybe_compile(self, compile_mode: str) -> bool:
        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            return False
        try:
            torch.set_float32_matmul_precision("high")
            self._predict_fn = compile_fn(self.policy.predict_action_chunk, mode=compile_mode, fullgraph=False)
            return True
        except Exception:
            self._predict_fn = self.policy.predict_action_chunk
            return False


def load_policy_runtime(config: DeployConfig) -> Pi05PolicyRuntime:
    """Load a Pi0.5 deployment policy from an exported bundle directory."""
    if config.runtime.multi_gpu_strategy != "none":
        raise NotImplementedError(
            "runtime.multi_gpu_strategy is reserved for future model sharding. "
            "Use runtime.device (for example cuda:0 or cuda:1) for current deployment."
        )
    bundle_dir = config.bundle.resolved_bundle_dir
    _validate_bundle(bundle_dir)
    manifest = load_bundle_manifest(bundle_dir)
    experiment_config = _load_bundle_experiment_config(bundle_dir, config)

    device = torch.device(config.runtime.device)
    model = build_pi05_with_lora(experiment_config, pretrained_path=experiment_config.model.pretrained_path)
    _load_adapter(model, resolve_bundle_adapter_dir(bundle_dir), device=device)
    model.to(device)
    model.eval()
    _configure_cuda_runtime(device)

    policy = _resolve_policy(model)
    policy_config = get_pi05_policy_config(model)
    policy_config.device = str(device)
    policy_config.dtype = config.runtime.dtype
    policy_config.chunk_size = int(config.runtime.chunk_size)
    policy_config.n_action_steps = int(config.runtime.chunk_size)
    preprocessor_config = copy.deepcopy(policy_config)
    preprocessor_config.device = "cpu"
    preprocessor, _ = make_pi05_pre_post_processors(preprocessor_config, dataset_stats=None)
    state_normalizer, action_normalizer = load_bundle_normalizers(bundle_dir)

    return Pi05PolicyRuntime(
        model=model,
        policy=policy,
        preprocessor=preprocessor,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        device=device,
        task=config.runtime.task,
        action_dim=int(manifest.get("model", {}).get("action_dim", config.runtime.action_dim)),
        output_chunk_size=config.runtime.chunk_size,
        clamp_normalized_action=config.safety.clamp_normalized_action,
        compile_model=config.runtime.compile_model,
        compile_mode=config.runtime.compile_mode,
    )


def _validate_bundle(bundle_dir: Path) -> None:
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Deployment bundle does not exist: {bundle_dir}")
    for name in ("manifest.json", "normalizers.json", EXPERIMENT_CONFIG_NAME):
        path = bundle_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Deployment bundle is missing {name}: {path}")


def _load_bundle_experiment_config(bundle_dir: Path, config: DeployConfig) -> ExperimentConfig:
    exp_config = load_experiment_config(bundle_dir / EXPERIMENT_CONFIG_NAME)
    object.__setattr__(exp_config.model, "device", config.runtime.device)
    object.__setattr__(exp_config.model, "dtype", config.runtime.dtype)
    object.__setattr__(exp_config.model, "gradient_checkpointing", False)
    object.__setattr__(exp_config.model, "chunk_size", config.runtime.chunk_size)
    object.__setattr__(exp_config.model, "n_action_steps", config.runtime.chunk_size)
    object.__setattr__(exp_config.model, "state_dim", config.runtime.state_dim)
    object.__setattr__(exp_config.model, "action_dim", config.runtime.action_dim)
    object.__setattr__(exp_config.model, "max_action_dim", config.runtime.action_dim)
    return exp_config


def _load_adapter(model: torch.nn.Module, adapter_dir: Path, *, device: torch.device) -> None:
    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"LoRA adapter weights not found: {weights}")
    state = load_safetensors(str(weights), device=str(device))
    set_peft_model_state_dict(model, state)
    tactile_weights = adapter_dir / "tactile_encoder.safetensors"
    tactile_encoder = _find_tactile_encoder(model)
    if tactile_encoder is not None:
        if not tactile_weights.exists():
            raise FileNotFoundError(f"Tactile encoder weights not found: {tactile_weights}")
        tactile_encoder.load_state_dict(
            load_safetensors(str(tactile_weights), device=str(device)),
            strict=True,
        )


def _find_tactile_encoder(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        tactile_encoder = getattr(module, "tactile_encoder", None)
        if isinstance(tactile_encoder, torch.nn.Module):
            return tactile_encoder
    return None


def _require_cpu_float_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise RuntimeError(f"{name} must stay on CPU until the PI05 processor has finished.")
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor.detach()


def _move_tensors_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.contiguous()
        non_blocking = device.type == "cuda"
        if non_blocking and tensor.device.type == "cpu":
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                pass
        return tensor.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    return value


def _configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        hook = getattr(cuda_backends, name, None)
        if callable(hook):
            hook(True)


def _resolve_policy(model: torch.nn.Module) -> Any:
    for candidate in (
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "base_model", None),
        getattr(model, "model", None),
        model,
    ):
        if candidate is not None and callable(getattr(candidate, "predict_action_chunk", None)):
            return candidate
    raise AttributeError("Could not find PI05Policy.predict_action_chunk on loaded model.")
