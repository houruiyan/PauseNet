"""Offline CPU smoke test; all generated examples are synthetic, not genomic data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml


def make_dataset(root: Path) -> None:
    rng = np.random.default_rng(20260916)
    for split, n in (("train", 8), ("validation", 4), ("test", 4)):
        dest = root / split
        dest.mkdir(parents=True, exist_ok=False)
        sequences = rng.integers(0, 4, (n, 2114), dtype=np.uint8)
        sequences[0, :10] = 4  # Exercise ambiguous bases.
        profiles = np.zeros((n, 1000), dtype=np.float32)
        for i in range(1, n):
            centre = 200 + 70 * i
            profiles[i, centre - 2:centre + 3] = [1, 3, 8 + i, 3, 1]
        counts = profiles.sum(axis=1)
        for name, array in {
            "sequence_codes": sequences, "profiles": profiles,
            "counts": counts, "profile_loss_mask": (counts >= 1).astype(np.uint8),
        }.items():
            np.save(dest / (name + ".npy"), array)
        pd.DataFrame({
            "sample_id": [f"synthetic_{split}_{i}" for i in range(n)],
            "source": ["synthetic_not_genomic"] * n,
        }).to_csv(dest / "manifest.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory only; default: unique temporary directory (retained).")
    args = parser.parse_args()
    if args.output_dir is None:
        root = Path(tempfile.mkdtemp(prefix="pausenet-smoke-")).resolve()
    else:
        root = args.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
    print(f"Smoke-test files: {root}", flush=True)
    make_dataset(root / "data")
    config = {
        "data": {"data_dir": str(root / "data")},
        "model": {"input_length": 2114, "output_length": 1000,
                  "channels": 8, "n_dilated_layers": 2, "dropout": 0.0,
                  "profile_kernel_size": 75, "stem_kernel_size": 21},
        "training": {"output_dir": str(root / "training"), "device": "cpu",
                     "batch_size": 2, "num_workers": 0, "max_epochs": 2,
                     "patience": 2, "seed": 20260916, "shuffle_seed": 20260916,
                     "counts_weight": 100.0, "profile_loss": "mnll", "progress": False},
    }
    config_path = root / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def run(*arguments: str) -> None:
        subprocess.run([sys.executable, "-m", "pausenet.cli", *arguments], env=env, check=True)
    run("train", "--config", str(config_path))
    checkpoint = root / "training" / "best_model.pt"
    if not checkpoint.is_file():
        raise RuntimeError("Training did not produce best_model.pt")
    run("evaluate", "--data-dir", str(root / "data"), "--checkpoint", str(checkpoint),
        "--split", "test", "--output-dir", str(root / "evaluation"),
        "--device", "cpu", "--batch-size", "2", "--num-workers", "0")
    with np.load(root / "evaluation" / "test_profiles.npz", allow_pickle=False) as result:
        for key, shape in (("predicted_profiles", (4, 1000)), ("predicted_counts", (4,)),
                           ("observed_profiles", (4, 1000)), ("observed_counts", (4,))):
            array = result[key]
            if array.shape != shape or not np.isfinite(array).all() or (array < 0).any():
                raise RuntimeError(f"Invalid output: {key}")
        np.testing.assert_allclose(result["predicted_profiles"].sum(axis=1), 1, atol=1e-5)
        np.testing.assert_array_equal(result["observed_counts"], np.load(root / "data/test/counts.npy"))
        np.testing.assert_array_equal(result["profile_masks"], [False, True, True, True])
    metrics = json.loads((root / "evaluation/test_metrics.json").read_text())
    for key in ("loss", "profile_loss", "count_loss"):
        if not np.isfinite(metrics[key]):
            raise RuntimeError(f"Non-finite evaluation metric: {key}")
    print("PASS: dataset loading, CPU training, checkpoint reload and prediction checks.")
    print("Synthetic data and reduced model: this is NOT a biological accuracy benchmark.")


if __name__ == "__main__":
    main()
