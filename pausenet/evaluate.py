"""Evaluate a trained PauseNet checkpoint.

Slim evaluation: saves only per-window counts and per-position profiles.

Outputs (written to --output-dir):
- ``{split}_profiles.npz``
    - ``observed_counts``    : (n_windows,) 每个窗口观测的 count 值
    - ``predicted_counts``   : (n_windows,) 每个窗口预测的 count 值
    - ``observed_profiles``  : (n_windows, 1000) 每个位置上的观测值
    - ``predicted_profiles`` : (n_windows, 1000) 每个位置上的预测值
    - ``profile_masks``      : (n_windows,) bool, profile loss mask
    - ``manifest_*``         : manifest 每列一份 (sample_id, chrom,
      output_start, output_end, strand 等), 用于把每行映射回基因组窗口
- ``{split}_metrics.json``: loss 汇总 (split, n, loss, profile_loss,
  count_loss, profile_valid_n)

不输出 JSD / Pearson correlation, 不生成 similarity / predictions TSV。
(train.py 无需任何修改: run_epoch 内部仍会顺带算 pearson/jsd 统计量,
但开销可忽略, 本文件在写出前已将其丢弃。)
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PauseNetDataset
from .model import PauseNet
from .train import model_config_from_dict, resolve_device, run_epoch


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[PauseNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint must contain a mapping.")

    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(
            "Checkpoint does not contain a valid config mapping; "
            "the model architecture cannot be reconstructed safely."
        )
    config = dict(config)
    if not isinstance(config.get("model"), Mapping):
        raise RuntimeError(
            "Checkpoint config does not contain a valid model section; "
            "refusing to evaluate with the default architecture."
        )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Checkpoint does not contain a valid model_state_dict.")

    model_config = model_config_from_dict(config)
    model = PauseNet(model_config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint weights do not exactly match the configured PauseNet model: {error}"
        ) from error
    return model.to(device).eval(), config


def evaluation_loss_config(checkpoint_config: Mapping) -> dict:
    """Recover the training loss definition stored in a checkpoint."""
    training_config = checkpoint_config.get("training")
    if not isinstance(training_config, Mapping):
        raise RuntimeError(
            "Checkpoint config does not contain a valid training section; "
            "evaluation loss cannot be reproduced."
        )
    required = ("counts_weight", "profile_loss")
    missing = [key for key in required if key not in training_config]
    if missing:
        raise RuntimeError(
            "Checkpoint training config is missing loss setting(s): "
            f"{', '.join(missing)}."
        )
    return {
        "training": {
            "counts_weight": float(training_config["counts_weight"]),
            "profile_loss": str(training_config["profile_loss"]),
            "progress": False,
        }
    }


def save_per_position_profiles(
    output_path: str | Path,
    outputs: Mapping,
    manifest: pd.DataFrame | None,
) -> None:
    """Save per-window counts and per-position observed/predicted profiles.

    Row order matches the dataset manifest exactly.
    """
    payload = {
        "observed_counts": np.asarray(outputs["observed_counts"]),
        "predicted_counts": np.asarray(outputs["predicted_counts"]),
        "observed_profiles": np.asarray(outputs["observed_profiles"], dtype=np.float32),
        "predicted_profiles": np.asarray(outputs["predicted_profiles"], dtype=np.float32),
        "profile_masks": np.asarray(outputs["profile_masks"], dtype=bool),
    }
    if manifest is not None:
        for column in manifest.columns:
            values = manifest[column].to_numpy()
            if values.dtype == object:
                values = values.astype(str)
            payload[f"manifest_{column}"] = values
    np.savez_compressed(output_path, **payload)


def evaluate_checkpoint(
    data_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "test",
    device: str = "auto",
    batch_size: int = 256,
    num_workers: int = 4,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = resolve_device(device)
    dataset = PauseNetDataset(Path(data_dir) / split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model, checkpoint_config = load_model(checkpoint, device_obj)
    config = evaluation_loss_config(checkpoint_config)
    # 使用原版 run_epoch (不改 train.py):
    # 它内部会顺带算 pearson/jsd 统计 (开销很小), 但这里不保存这些指标,
    # 真正耗时的是原 evaluate 里的 similarity 汇总 (随机打乱 + 多分辨率 JSD),
    # 已被整体移除。
    metrics, outputs = run_epoch(model, loader, device_obj, config, keep_profiles=True)
    # 只保留 loss 相关指标, 丢弃 pearson / jsd
    metrics = {
        key: value
        for key, value in metrics.items()
        if "pearson" not in key and "jsd" not in key
    }
    metrics["split"] = split
    metrics["n"] = int(len(dataset))
    with (output_dir / f"{split}_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    profiles_path = output_dir / f"{split}_profiles.npz"
    save_per_position_profiles(profiles_path, outputs, dataset.manifest)
    print(f"Saved per-position profiles to {profiles_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PauseNet checkpoint (counts + per-position profiles only)."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
