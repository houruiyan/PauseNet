"""PauseNet model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PauseNetConfig:
    input_length: int = 2114
    output_length: int = 1000
    channels: int = 128
    n_dilated_layers: int = 11
    dropout: float = 0.10
    profile_kernel_size: int = 75
    stem_kernel_size: int = 21
    count_head: str = "procapnet_mean"
    use_position_template: bool = False
    attention_heads: int = 4
    pooling_widths: tuple[int, ...] = (101, 251, 501, 1001, 2114)


class DilatedResidualBlock(nn.Module):
    """A residual Conv1D block with exponentially increasing dilation."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm(inputs))
        hidden = self.conv(hidden)
        hidden = self.pointwise(self.dropout(F.gelu(hidden)))
        return inputs + hidden


class SequenceBackbone(nn.Module):
    """Shared sequence backbone used by PauseNet."""

    def __init__(
        self,
        channels: int = 128,
        n_dilated_layers: int = 11,
        dropout: float = 0.10,
        stem_kernel_size: int = 21,
    ):
        super().__init__()
        if channels % 8:
            raise ValueError("channels must be divisible by 8 for GroupNorm")
        self.stem = nn.Conv1d(
            4,
            channels,
            kernel_size=stem_kernel_size,
            padding=stem_kernel_size // 2,
        )
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(channels, dilation=2**layer, dropout=dropout)
                for layer in range(n_dilated_layers)
            ]
        )
        self.final_norm = nn.GroupNorm(8, channels)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(sequence)
        for block in self.blocks:
            hidden = block(hidden)
        return F.gelu(self.final_norm(hidden))


def make_position_template(template: np.ndarray | None, output_length: int) -> torch.Tensor:
    """Return log position-template initialization."""

    if template is None:
        values = np.ones(output_length, dtype=np.float32)
    else:
        values = np.asarray(template, dtype=np.float32)
        if values.shape != (output_length,):
            raise ValueError(
                f"position template has shape {values.shape}, expected {(output_length,)}"
            )
        values = np.maximum(values, 1e-6)
    values = values / values.sum()
    return torch.as_tensor(np.log(values), dtype=torch.float32)


class ProfileHead(nn.Module):
    """Profile head for strand-oriented single-track profile prediction."""

    def __init__(
        self,
        channels: int,
        output_length: int,
        kernel_size: int = 75,
        position_template: np.ndarray | None = None,
        use_position_template: bool = True,
    ):
        super().__init__()
        self.output_length = output_length
        self.conv = nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.use_position_template = use_position_template
        if use_position_template:
            self.position_logits = nn.Parameter(
                make_position_template(position_template, output_length)
            )
            self.residual_scale = nn.Parameter(torch.tensor(0.25))
        else:
            self.position_logits = None
            self.residual_scale = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual_logits = self.conv(hidden).squeeze(1)
        if not self.use_position_template:
            return residual_logits
        return self.position_logits.unsqueeze(0) + self.residual_scale * residual_logits


class MultiScaleCountHead(nn.Module):
    """Optional experimental count head with multi-scale and attention pooling."""

    def __init__(
        self,
        channels: int,
        input_length: int,
        pooling_widths: Iterable[int],
        attention_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.input_length = input_length
        self.pooling_widths = tuple(pooling_widths)
        self.attention = nn.Conv1d(channels, attention_heads, kernel_size=1)
        pooled_channels = channels * (2 * len(self.pooling_widths) + attention_heads)
        sequence_composition_channels = 4 * len(self.pooling_widths)
        self.mlp = nn.Sequential(
            nn.Linear(pooled_channels + sequence_composition_channels, channels * 2),
            nn.LayerNorm(channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def _center_region(self, values: torch.Tensor, width: int) -> torch.Tensor:
        center = self.input_length // 2
        start = max(center - width // 2, 0)
        end = min(start + width, values.shape[-1])
        return values[..., start:end]

    def sequence_composition(self, sequence: torch.Tensor) -> torch.Tensor:
        features = []
        for width in self.pooling_widths:
            region = self._center_region(sequence, width)
            features.append(region.mean(dim=-1))
        return torch.cat(features, dim=1)

    def count_pooling(self, hidden: torch.Tensor) -> torch.Tensor:
        features = []
        for width in self.pooling_widths:
            region = self._center_region(hidden, width)
            features.extend([region.mean(dim=-1), region.amax(dim=-1)])
        attention = torch.softmax(self.attention(hidden), dim=-1)
        attention_pool = torch.einsum("bhl,bcl->bhc", attention, hidden)
        features.append(attention_pool.flatten(start_dim=1))
        return torch.cat(features, dim=1)

    def forward(self, sequence: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [self.count_pooling(hidden), self.sequence_composition(sequence)],
            dim=1,
        )
        return F.softplus(self.mlp(features).squeeze(1))


class MeanCountHead(nn.Module):
    """ProCapNet-style count head: mean pooling, linear projection, softplus."""

    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Linear(channels, 1)

    def forward(self, sequence: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        del sequence
        return F.softplus(self.linear(hidden.mean(dim=-1)).squeeze(1))


class PauseNet(nn.Module):
    """PauseNet sequence-to-profile/count model."""

    def __init__(
        self,
        config: PauseNetConfig | None = None,
        position_template: np.ndarray | None = None,
    ):
        super().__init__()
        self.config = config or PauseNetConfig()
        difference = self.config.input_length - self.config.output_length
        if difference < 0 or difference % 2:
            raise ValueError("input_length - output_length must be nonnegative and even")
        self.trim = difference // 2

        encoding = torch.zeros(5, 4, dtype=torch.float32)
        encoding[:4] = torch.eye(4)
        self.register_buffer("encoding", encoding, persistent=False)

        self.backbone = SequenceBackbone(
            channels=self.config.channels,
            n_dilated_layers=self.config.n_dilated_layers,
            dropout=self.config.dropout,
            stem_kernel_size=self.config.stem_kernel_size,
        )
        self.profile_head = ProfileHead(
            channels=self.config.channels,
            output_length=self.config.output_length,
            kernel_size=self.config.profile_kernel_size,
            position_template=position_template,
            use_position_template=self.config.use_position_template,
        )
        if self.config.count_head == "multi_scale":
            self.count_head = MultiScaleCountHead(
                channels=self.config.channels,
                input_length=self.config.input_length,
                pooling_widths=self.config.pooling_widths,
                attention_heads=self.config.attention_heads,
                dropout=self.config.dropout,
            )
        elif self.config.count_head in {"procapnet_mean", "simple_mean"}:
            self.count_head = MeanCountHead(self.config.channels)
        else:
            raise ValueError(f"Unknown count_head: {self.config.count_head}")

    def encode(self, sequence_codes: torch.Tensor) -> torch.Tensor:
        return self.encoding[sequence_codes.long()].transpose(1, 2)

    def forward(self, sequence_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.encode(sequence_codes)
        hidden = self.backbone(sequence)
        profile_region = hidden[:, :, self.trim : self.trim + self.config.output_length]
        profile_logits = self.profile_head(profile_region)
        predicted_log1p_count = self.count_head(sequence, hidden)
        return profile_logits, predicted_log1p_count

    def forward_count(self, sequence_codes: torch.Tensor) -> torch.Tensor:
        _, predicted_log1p_count = self(sequence_codes)
        return predicted_log1p_count
