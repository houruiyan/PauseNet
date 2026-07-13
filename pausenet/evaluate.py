"""Evaluate a trained PauseNet checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PauseNetDataset
from .model import PauseNet, PauseNetConfig
from .train import model_config_from_dict, run_epoch


def load_model(checkpoint_path: str | Path, device: torch.device) -> PauseNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_config = model_config_from_dict(config) if config else PauseNetConfig()
    model = PauseNet(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model.to(device).eval()


def evaluate_checkpoint(
    data_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "test",
    device: str = "cuda:0",
    batch_size: int = 256,
    num_workers: int = 4,
    save_predictions: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PauseNetDataset(Path(data_dir) / split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    device_obj = torch.device(device)
    model = load_model(checkpoint, device_obj)
    config = {
        "training": {
            "counts_weight": 100.0,
            "profile_loss": "mnll",
            "progress": False,
        }
    }
    metrics, outputs = run_epoch(model, loader, device_obj, config, keep_profiles=True)
    metrics["split"] = split
    metrics["n"] = int(len(dataset))
    with (output_dir / f"{split}_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    if save_predictions:
        table = pd.DataFrame(
            {
                "index": range(len(outputs["observed_counts"])),
                "observed_counts": outputs["observed_counts"],
                "predicted_counts": outputs["predicted_counts"],
                "predicted_log1p_counts": outputs["predicted_log1p_counts"],
                "profile_loss_mask": outputs["profile_masks"].astype(int),
            }
        )
        if dataset.manifest is not None:
            table = pd.concat([dataset.manifest.reset_index(drop=True), table], axis=1)
        table.to_csv(output_dir / f"{split}_predictions.tsv", sep="\t", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PauseNet checkpoint.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-save-predictions", action="store_true")
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
        save_predictions=not args.no_save_predictions,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
