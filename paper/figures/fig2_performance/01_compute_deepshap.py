#!/usr/bin/env python3
"""Calculate DeepSHAP-style expected-gradient attributions for PauseNet.

The script writes both full hypothetical contributions (N, L, 4) and
base-projected contributions (N, L). The full arrays are the TF-MoDISco input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pausenet.model import PauseNet, PauseNetConfig  # noqa: E402


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate DeepSHAP-style expected-gradient attributions for PauseNet."
    )
    parser.add_argument("--data-dir", required=True, help="PauseNet dataset root.")
    parser.add_argument("--checkpoint", required=True, help="PauseNet checkpoint path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument(
        "--background-split", default="train", choices=["train", "validation", "test"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-backgrounds", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--background-pool-size", type=int, default=2048)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--tasks", nargs="+", choices=["count", "profile"], default=["count", "profile"]
    )
    parser.add_argument("--seed", type=int, default=2026062401)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def config_from_checkpoint(checkpoint: dict, input_length: int, output_length: int) -> PauseNetConfig:
    """Build a PauseNet config from current or legacy checkpoint metadata."""
    config = checkpoint.get("config") or {}
    model_config = config.get("model", config) if isinstance(config, dict) else {}
    legacy_args = checkpoint.get("args") or {}
    return PauseNetConfig(
        input_length=input_length,
        output_length=output_length,
        channels=int(model_config.get("channels", legacy_args.get("channels", 128))),
        n_dilated_layers=int(
            model_config.get("n_dilated_layers", legacy_args.get("n_dilated_layers", 11))
        ),
        dropout=float(model_config.get("dropout", legacy_args.get("dropout", 0.10))),
        profile_kernel_size=int(model_config.get("profile_kernel_size", 75)),
        stem_kernel_size=int(model_config.get("stem_kernel_size", 21)),
    )


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the pre-release HEK293T head names to the final PauseNet names."""
    key_map = {
        "profile_head.weight": "profile_head.conv.weight",
        "profile_head.bias": "profile_head.conv.bias",
        "count_head.weight": "count_head.linear.weight",
        "count_head.bias": "count_head.linear.bias",
    }
    return {key_map.get(key, key): value for key, value in state_dict.items()}


def load_model(
    checkpoint_path: Path,
    input_length: int,
    output_length: int,
    device: torch.device,
) -> PauseNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PauseNet(config_from_checkpoint(checkpoint, input_length, output_length))
    state_dict = normalize_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


class OneHotPauseNet(nn.Module):
    """Expose differentiable count and profile targets from a one-hot sequence."""

    def __init__(self, model: PauseNet):
        super().__init__()
        self.model = model

    def forward_heads(self, onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model.backbone(onehot)
        profile_hidden = hidden[:, :, self.model.trim : self.model.trim + self.model.config.output_length]
        profile_logits = self.model.profile_head(profile_hidden)
        predicted_log1p_counts = self.model.count_head(onehot, hidden)
        return profile_logits, predicted_log1p_counts

    def count_scalar(self, onehot: torch.Tensor) -> torch.Tensor:
        return self.forward_heads(onehot)[1]

    def profile_shape_scalar(self, onehot: torch.Tensor) -> torch.Tensor:
        profile_logits, _ = self.forward_heads(onehot)
        centered = profile_logits - profile_logits.mean(dim=-1, keepdim=True)
        weights = torch.softmax(profile_logits.detach(), dim=-1)
        return torch.sum(centered * weights, dim=-1)


def codes_to_onehot(codes: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(codes.astype(np.int64), device=device)
    valid = tensor < 4
    onehot = torch.zeros((*tensor.shape, 4), device=device, dtype=torch.float32)
    onehot.scatter_(2, tensor.clamp(0, 3).unsqueeze(-1), 1.0)
    return (onehot * valid.unsqueeze(-1)).transpose(1, 2).contiguous()


def project_contributions(full_l4: np.ndarray, codes: np.ndarray) -> np.ndarray:
    projected = np.zeros(codes.shape, dtype=full_l4.dtype)
    valid = codes < 4
    rows, columns = np.where(valid)
    projected[rows, columns] = full_l4[rows, columns, codes[rows, columns]]
    return projected


def expected_integrated_gradients(
    model: OneHotPauseNet,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    task: str,
    alphas: torch.Tensor,
) -> torch.Tensor:
    scalar_fn = model.count_scalar if task == "count" else model.profile_shape_scalar
    total = torch.zeros_like(inputs)
    for reference_index in range(baselines.shape[1]):
        baseline = baselines[:, reference_index]
        difference = inputs - baseline
        for alpha in alphas:
            interpolated = (baseline + alpha * difference).detach().requires_grad_(True)
            scalar = scalar_fn(interpolated).sum()
            gradient = torch.autograd.grad(scalar, interpolated)[0]
            total += gradient * difference
    return total / float(baselines.shape[1] * len(alphas))


def main() -> None:
    args = parse_args()
    if args.num_backgrounds < 1 or args.num_steps < 1:
        raise ValueError("--num-backgrounds and --num-steps must both be positive")

    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    split_dir = data_dir / args.split
    background_dir = data_dir / args.background_split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_codes = np.load(split_dir / "sequence_codes.npy", mmap_mode="r")
    background_codes_all = np.load(background_dir / "sequence_codes.npy", mmap_mode="r")
    profiles = np.load(split_dir / "profiles.npy", mmap_mode="r")
    manifest = pd.read_csv(split_dir / "manifest.tsv", sep="\t")
    if len(sequence_codes) != len(manifest):
        raise ValueError("sequence_codes.npy and manifest.tsv have different numbers of rows")

    start = args.start_index
    stop = len(sequence_codes) if args.max_samples is None else min(len(sequence_codes), start + args.max_samples)
    n_samples = stop - start
    if n_samples <= 0:
        raise ValueError("No samples selected; check --start-index and --max-samples")

    suffix = (
        f"{args.split}_start{start}_n{n_samples}_bg{args.num_backgrounds}_steps{args.num_steps}"
    )
    dtype = np.float16 if args.dtype == "float16" else np.float32
    outputs: dict[str, dict[str, object]] = {}
    for task in args.tasks:
        full_path = output_dir / f"{task}_contribs_full_Lx4_{suffix}.npy"
        projected_path = output_dir / f"{task}_contribs_projected_{suffix}.npy"
        if (full_path.exists() or projected_path.exists()) and not args.overwrite:
            raise FileExistsError(f"Attribution output already exists for {task}; use --overwrite")
        outputs[task] = {
            "full": np.lib.format.open_memmap(
                full_path,
                mode="w+",
                dtype=dtype,
                shape=(n_samples, sequence_codes.shape[1], 4),
            ),
            "projected": np.lib.format.open_memmap(
                projected_path,
                mode="w+",
                dtype=dtype,
                shape=(n_samples, sequence_codes.shape[1]),
            ),
            "full_path": str(full_path),
            "projected_path": str(projected_path),
        }

    pool_size = min(args.background_pool_size, len(background_codes_all))
    pool_indices = rng.choice(len(background_codes_all), size=pool_size, replace=False)
    background_pool = np.asarray(background_codes_all[pool_indices], dtype=np.uint8)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = load_model(Path(args.checkpoint), sequence_codes.shape[1], profiles.shape[1], device)
    attribution_model = OneHotPauseNet(model).to(device).eval()
    alphas = torch.linspace(1.0 / args.num_steps, 1.0, args.num_steps, device=device)

    manifest.iloc[start:stop].to_csv(
        output_dir / f"manifest_{suffix}.tsv", sep="\t", index=False
    )
    started = time.time()
    for global_start in range(start, stop, args.batch_size):
        global_stop = min(stop, global_start + args.batch_size)
        local_start = global_start - start
        local_stop = local_start + (global_stop - global_start)
        batch_codes = np.asarray(sequence_codes[global_start:global_stop], dtype=np.uint8)
        inputs = codes_to_onehot(batch_codes, device)
        background_indices = rng.integers(
            0, len(background_pool), size=(len(batch_codes), args.num_backgrounds)
        )
        baseline_codes = background_pool[background_indices.reshape(-1)].reshape(
            len(batch_codes), args.num_backgrounds, sequence_codes.shape[1]
        )
        baselines = codes_to_onehot(
            baseline_codes.reshape(-1, sequence_codes.shape[1]), device
        ).reshape(len(batch_codes), args.num_backgrounds, 4, sequence_codes.shape[1])

        for task in args.tasks:
            contributions = expected_integrated_gradients(
                attribution_model, inputs, baselines, task, alphas
            )
            contributions_l4 = contributions.detach().cpu().numpy().transpose(0, 2, 1)
            contributions_l4 = contributions_l4.astype(dtype, copy=False)
            outputs[task]["full"][local_start:local_stop] = contributions_l4
            outputs[task]["projected"][local_start:local_stop] = project_contributions(
                contributions_l4, batch_codes
            )

        completed = global_stop - start
        if completed % max(args.batch_size * 10, 1) == 0 or global_stop == stop:
            elapsed = time.time() - started
            print(
                json.dumps(
                    {
                        "completed": int(completed),
                        "n_samples": int(n_samples),
                        "samples_per_second": completed / max(elapsed, EPS),
                    }
                ),
                flush=True,
            )

    for item in outputs.values():
        item["full"].flush()
        item["projected"].flush()

    metadata = {
        "method": "DeepSHAP-style expected integrated gradients with sampled sequence backgrounds",
        "model_class": "PauseNet",
        "count_target": "predicted log1p(counts)",
        "profile_target": "mean-centered profile logits weighted by detached predicted profile probabilities",
        "attribution_suffix": suffix,
        "data_dir": str(data_dir),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "background_split": args.background_split,
        "n_samples": int(n_samples),
        "input_length": int(sequence_codes.shape[1]),
        "output_length": int(profiles.shape[1]),
        "elapsed_seconds": time.time() - started,
        "arguments": vars(args),
    }
    (output_dir / f"metadata_{suffix}.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
