#!/usr/bin/env python3
"""Compare a rebuilt PauseNet dataset with its source dataset on fixed samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--rebuilt-dir", required=True, type=Path)
    parser.add_argument("--samples-per-split", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    summary = {"seed": args.seed, "samples_per_split": args.samples_per_split, "splits": {}}
    for split in ("train", "validation", "test"):
        source_split = args.source_dir / split
        rebuilt_split = args.rebuilt_dir / split
        source_manifest = pd.read_csv(source_split / "manifest.tsv", sep="\t")
        rebuilt_manifest = pd.read_csv(rebuilt_split / "manifest.tsv", sep="\t")
        if len(source_manifest) != len(rebuilt_manifest):
            raise AssertionError(f"{split}: sample count differs")
        columns = ["sample_id", "chrom", "output_start", "output_end"]
        if not source_manifest[columns].equals(rebuilt_manifest[columns]):
            raise AssertionError(f"{split}: anchor ordering or coordinates differ")

        size = min(args.samples_per_split, len(source_manifest))
        indices = np.sort(rng.choice(len(source_manifest), size=size, replace=False))
        source_sequence = np.load(source_split / "sequence_codes.npy", mmap_mode="r")
        rebuilt_sequence = np.load(rebuilt_split / "sequence_codes.npy", mmap_mode="r")
        source_profile = np.load(source_split / "profiles.npy", mmap_mode="r")
        rebuilt_profile = np.load(rebuilt_split / "profiles.npy", mmap_mode="r")
        source_count = np.load(source_split / "counts.npy", mmap_mode="r")
        rebuilt_count = np.load(rebuilt_split / "counts.npy", mmap_mode="r")

        if not np.array_equal(source_sequence[indices], rebuilt_sequence[indices]):
            raise AssertionError(f"{split}: sequence codes differ")
        if not np.allclose(source_profile[indices], rebuilt_profile[indices], rtol=0.0, atol=0.0):
            raise AssertionError(f"{split}: profiles differ")
        if not np.allclose(source_count[indices], rebuilt_count[indices], rtol=0.0, atol=0.0):
            raise AssertionError(f"{split}: counts differ")
        summary["splits"][split] = {"n": len(source_manifest), "validated_indices": indices.tolist()}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
