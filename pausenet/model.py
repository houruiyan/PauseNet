"""PauseNet model definitions."""

from __future__ import annotations

from dataclasses import dataclass

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


class ProfileHead(nn.Module):
    """Profile head for strand-oriented single-track profile prediction."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 75,
    ):
        super().__init__()
        self.conv = nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden).squeeze(1)


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
            kernel_size=self.config.profile_kernel_size,
        )
        self.count_head = MeanCountHead(self.config.channels)

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
