"""Dataset utilities for PauseNet standard split directories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PauseNetDataset(Dataset):
    """Load one PauseNet split directory."""

    def __init__(self, split_dir: str | Path):
        self.split_dir = Path(split_dir)
        self.split_name = self.split_dir.name
        self.sequence_codes = np.load(self.split_dir / "sequence_codes.npy", mmap_mode="r")
        self.profiles = np.load(self.split_dir / "profiles.npy", mmap_mode="r")
        self.counts = np.load(self.split_dir / "counts.npy", mmap_mode="r")

        mask_path = self.split_dir / "profile_loss_mask.npy"
        self.profile_masks = (
            np.load(mask_path, mmap_mode="r")
            if mask_path.exists()
            else np.ones(len(self.counts), dtype=np.uint8)
        )
        sample_types_path = self.split_dir / "sample_types.npy"
        self.sample_types = (
            np.load(sample_types_path, mmap_mode="r")
            if sample_types_path.exists()
            else np.zeros(len(self.counts), dtype=np.uint8)
        )
        anchor_path = self.split_dir / "anchor_type_codes.npy"
        self.anchor_type_codes = (
            np.load(anchor_path, mmap_mode="r") if anchor_path.exists() else None
        )
        manifest_path = self.split_dir / "manifest.tsv"
        self.manifest = (
            pd.read_csv(manifest_path, sep="\t") if manifest_path.exists() else None
        )
        self._validate_lengths()

    def _validate_lengths(self) -> None:
        lengths = {
            len(self.sequence_codes),
            len(self.profiles),
            len(self.counts),
            len(self.profile_masks),
            len(self.sample_types),
        }
        if self.anchor_type_codes is not None:
            lengths.add(len(self.anchor_type_codes))
        if self.manifest is not None:
            lengths.add(len(self.manifest))
        if len(lengths) != 1:
            raise ValueError(f"Array length mismatch in {self.split_dir}: {lengths}")

    def __len__(self) -> int:
        return len(self.counts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sequence_codes": torch.from_numpy(np.array(self.sequence_codes[index], copy=True)),
            "profiles": torch.from_numpy(
                np.array(self.profiles[index], dtype=np.float32, copy=True)
            ),
            "counts": torch.tensor(float(self.counts[index]), dtype=torch.float32),
            "profile_mask": torch.tensor(int(self.profile_masks[index]), dtype=torch.bool),
            "sample_type": torch.tensor(int(self.sample_types[index]), dtype=torch.uint8),
            "index": torch.tensor(index, dtype=torch.int64),
        }
