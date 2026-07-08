from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from pi05.common.model.builder import (
    _enable_lora_parameters,
    _load_adapter_weights_strict,
    _lora_module_names_from_adapter,
    _lora_module_names_from_model,
    _resolve_adapter_path,
)


def _tiny_lora_model() -> torch.nn.Module:
    base = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    return get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["0"],
            bias="none",
        ),
    )


def test_strict_loader_restores_trainable_adapter(tmp_path: Path) -> None:
    source = _tiny_lora_model()
    for name, parameter in source.named_parameters():
        if "lora_" in name:
            torch.nn.init.constant_(parameter, 0.25)

    adapter_dir = tmp_path / "adapter"
    source.save_pretrained(adapter_dir)

    target = _tiny_lora_model()
    for name, parameter in target.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = False
    _load_adapter_weights_strict(target, adapter_dir)
    _enable_lora_parameters(target)

    source_lora = {name: value for name, value in source.state_dict().items() if "lora_" in name}
    target_lora = {name: value for name, value in target.state_dict().items() if "lora_" in name}
    assert source_lora.keys() == target_lora.keys()
    for name in source_lora:
        torch.testing.assert_close(target_lora[name], source_lora[name])
    assert all(parameter.requires_grad for name, parameter in target.named_parameters() if "lora_" in name)


def test_resolve_adapter_path_accepts_checkpoint_parent(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint_epoch_5"
    adapter_dir = checkpoint_dir / "adapter"
    _tiny_lora_model().save_pretrained(adapter_dir)

    assert _resolve_adapter_path(checkpoint_dir) == adapter_dir.resolve()


def test_lora_module_names_match_saved_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    model = _tiny_lora_model()
    model.save_pretrained(adapter_dir)

    assert _lora_module_names_from_adapter(adapter_dir / "adapter_model.safetensors") == set(
        _lora_module_names_from_model(model)
    )
