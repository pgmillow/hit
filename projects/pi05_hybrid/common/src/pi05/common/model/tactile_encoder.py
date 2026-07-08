"""Temporal tactile encoder producing one VLM token per tactile sensor."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pi05.common.config.schema import TactileConfig


class TactileTemporalEncoder(nn.Module):
    """Encode a tactile history window into 22 position-preserving VLM tokens."""

    def __init__(self, config: TactileConfig, vlm_hidden_dim: int) -> None:
        super().__init__()
        self.config = config
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(config.input_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, config.hidden_dim),
        )
        self.sensor_embedding = nn.Embedding(config.patch_count, config.hidden_dim)
        self.hand_embedding = nn.Embedding(2, config.hidden_dim)
        self.temporal_embedding = nn.Embedding(config.history_steps, config.hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * config.mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.vlm_projector = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, vlm_hidden_dim),
        )
        gate_init = 0.0 if config.zero_init_gate else 1.0
        self.gate = nn.Parameter(torch.tensor(gate_init))

        hand_ids = torch.cat(
            [
                torch.zeros(config.patch_count // 2, dtype=torch.long),
                torch.ones(config.patch_count - config.patch_count // 2, dtype=torch.long),
            ]
        )
        self.register_buffer("hand_ids", hand_ids, persistent=False)

    def forward(self, tactile_window: Tensor, tactile_window_mask: Tensor | None = None) -> Tensor:
        """Return position-preserving tactile tokens shaped ``[B, 22, D_vlm]``."""
        if tactile_window.ndim != 6:
            raise ValueError(
                "Expected tactile_window shape [B,W,22,1,14,14], "
                f"got {tuple(tactile_window.shape)}."
            )
        batch_size, history_steps, patch_count, channels, height, width = tactile_window.shape
        expected = (
            self.config.history_steps,
            self.config.patch_count,
            self.config.input_channels,
            self.config.patch_size,
            self.config.patch_size,
        )
        if (history_steps, patch_count, channels, height, width) != expected:
            raise ValueError(
                "Tactile input shape does not match config: "
                f"got {(history_steps, patch_count, channels, height, width)}, expected {expected}."
            )

        x = tactile_window.reshape(batch_size * history_steps * patch_count, channels, height, width)
        x = self.patch_encoder(x)
        x = x.reshape(batch_size, history_steps, patch_count, self.config.hidden_dim)

        sensor_ids = torch.arange(patch_count, device=x.device)
        time_ids = torch.arange(history_steps, device=x.device)
        x = (
            x
            + self.sensor_embedding(sensor_ids)[None, None, :, :]
            + self.hand_embedding(self.hand_ids)[None, None, :, :]
            + self.temporal_embedding(time_ids)[None, :, None, :]
        )

        token_mask = None
        if tactile_window_mask is not None:
            if tactile_window_mask.shape != (batch_size, history_steps):
                raise ValueError(
                    f"Expected tactile_window_mask shape {(batch_size, history_steps)}, "
                    f"got {tuple(tactile_window_mask.shape)}."
                )
            token_mask = ~tactile_window_mask.bool()
            token_mask = token_mask[:, :, None].expand(batch_size, history_steps, patch_count)
            token_mask = token_mask.reshape(batch_size, history_steps * patch_count)

        x = x.reshape(batch_size, history_steps * patch_count, self.config.hidden_dim)
        x = self.temporal_transformer(x, src_key_padding_mask=token_mask)
        current_tokens = x[:, -patch_count:, :]

        if self.training and self.config.tactile_dropout_prob > 0:
            keep = torch.rand(batch_size, 1, 1, device=x.device) >= self.config.tactile_dropout_prob
            current_tokens = current_tokens * keep

        tokens = self.vlm_projector(current_tokens)
        return tokens * torch.tanh(self.gate).to(dtype=tokens.dtype)
