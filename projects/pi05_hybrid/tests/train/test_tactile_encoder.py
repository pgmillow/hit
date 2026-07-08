from __future__ import annotations

import torch

from pi05.common.config.schema import TactileConfig
from pi05.common.model.tactile_encoder import TactileTemporalEncoder


def _config(*, zero_init_gate: bool = False) -> TactileConfig:
    return TactileConfig(
        enabled=True,
        history_steps=3,
        history_stride=1,
        hidden_dim=32,
        transformer_layers=2,
        attention_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        zero_init_gate=zero_init_gate,
        tactile_dropout_prob=0.0,
    )


def test_tactile_encoder_preserves_22_position_tokens() -> None:
    encoder = TactileTemporalEncoder(_config(), vlm_hidden_dim=64)
    tactile = torch.rand(2, 3, 22, 1, 14, 14)
    mask = torch.tensor([[False, True, True], [True, True, True]])

    output = encoder(tactile, mask)

    assert output.shape == (2, 22, 64)


def test_tactile_encoder_zero_gate_starts_as_no_op() -> None:
    encoder = TactileTemporalEncoder(_config(zero_init_gate=True), vlm_hidden_dim=64)
    output = encoder(torch.rand(1, 3, 22, 1, 14, 14), torch.ones(1, 3, dtype=torch.bool))

    torch.testing.assert_close(output, torch.zeros_like(output))


def test_tactile_encoder_receives_gradients() -> None:
    encoder = TactileTemporalEncoder(_config(), vlm_hidden_dim=64)
    output = encoder(torch.rand(1, 3, 22, 1, 14, 14), torch.ones(1, 3, dtype=torch.bool))
    output.square().mean().backward()

    assert encoder.patch_encoder[0].weight.grad is not None
    assert encoder.sensor_embedding.weight.grad is not None
    assert encoder.vlm_projector[1].weight.grad is not None
