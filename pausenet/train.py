"""PauseNet training entry point."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PauseNetDataset, build_position_template
from .losses import pausenet_loss
from .metrics import js_distance_rows, safe_pearson
from .model import PauseNet, PauseNetConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def make_loader(dataset: PauseNetDataset, batch_size: int, num_workers: int, shuffle: bool):
    generator = torch.Generator()
    generator.manual_seed(12345)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


def model_config_from_dict(config: dict) -> PauseNetConfig:
    model_cfg = dict(config.get("model", {}))
    if "pooling_widths" in model_cfg:
        model_cfg["pooling_widths"] = tuple(model_cfg["pooling_widths"])
    return PauseNetConfig(**model_cfg)


def run_epoch(
    model: PauseNet,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    optimizer: torch.optim.Optimizer | None = None,
    keep_profiles: bool = False,
) -> tuple[dict, dict]:
    training = optimizer is not None
    model.train(training)
    loss_totals = np.zeros(4, dtype=np.float64)
    observed_counts_all = []
    predicted_log_counts_all = []
    masks_all = []
    observed_profiles_all = []
    predicted_profiles_all = []

    counts_weight = float(config["training"].get("counts_weight", 100.0))
    profile_loss = str(config["training"].get("profile_loss", "mnll"))
    grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))

    if training:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        iterator = tqdm(loader, leave=False, disable=not config["training"].get("progress", True))
        for batch_index, batch in enumerate(iterator):
            sequence_codes = batch["sequence_codes"].to(device, non_blocking=True)
            profiles = batch["profiles"].to(device, non_blocking=True)
            counts = batch["counts"].to(device, non_blocking=True)
            profile_mask = batch["profile_mask"].to(device, non_blocking=True)

            profile_logits, predicted_log_counts = model(sequence_codes)
            total_loss, profile_component, count_component = pausenet_loss(
                profile_logits,
                predicted_log_counts,
                profiles,
                counts,
                profile_mask,
                counts_weight=counts_weight,
                profile_loss=profile_loss,
            )

            if training:
                (total_loss / grad_accum_steps).backward()
                should_step = (
                    (batch_index + 1) % grad_accum_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = len(sequence_codes)
            loss_totals += np.array(
                [
                    float(total_loss.detach()),
                    float(profile_component.detach()),
                    float(count_component.detach()),
                    batch_size,
                ],
                dtype=np.float64,
            ) * np.array([batch_size, batch_size, batch_size, 1], dtype=np.float64)

            if not training:
                observed_counts_all.append(batch["counts"].numpy())
                predicted_log_counts_all.append(predicted_log_counts.detach().cpu().numpy())
                masks_all.append(batch["profile_mask"].numpy())
                if keep_profiles:
                    observed_profiles_all.append(batch["profiles"].numpy())
                    predicted_profiles_all.append(
                        torch.softmax(profile_logits.float(), dim=-1).detach().cpu().numpy()
                    )

    n = max(loss_totals[3], 1)
    metrics = {
        "loss": float(loss_totals[0] / n),
        "profile_loss": float(loss_totals[1] / n),
        "count_loss": float(loss_totals[2] / n),
    }
    outputs = {}
    if not training:
        observed_counts = np.concatenate(observed_counts_all)
        predicted_log_counts = np.concatenate(predicted_log_counts_all)
        predicted_counts = np.expm1(predicted_log_counts).clip(min=0)
        profile_masks = np.concatenate(masks_all).astype(bool)
        metrics["count_pearson_log1p"] = safe_pearson(
            np.log1p(observed_counts), predicted_log_counts
        )
        metrics["count_pearson_raw"] = safe_pearson(observed_counts, predicted_counts)
        outputs = {
            "observed_counts": observed_counts,
            "predicted_log1p_counts": predicted_log_counts,
            "predicted_counts": predicted_counts,
            "profile_masks": profile_masks,
        }
        if keep_profiles:
            observed_profiles = np.concatenate(observed_profiles_all)
            predicted_profiles = np.concatenate(predicted_profiles_all)
            outputs["observed_profiles"] = observed_profiles
            outputs["predicted_profiles"] = predicted_profiles
            valid = profile_masks & (observed_profiles.sum(axis=1) > 0)
            metrics["profile_jsd_n"] = int(valid.sum())
            for resolution in (1, 5, 10, 20):
                values = js_distance_rows(
                    observed_profiles[valid], predicted_profiles[valid], resolution
                )
                metrics[f"profile_jsd_{resolution}bp_mean"] = float(np.mean(values))
                metrics[f"profile_jsd_{resolution}bp_median"] = float(np.median(values))
    return metrics, outputs


def save_checkpoint(path: Path, model: PauseNet, config: dict, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation_metrics": metrics,
        },
        path,
    )


def train_from_config(config: dict) -> Path:
    set_seed(int(config["training"].get("seed", 20260713)))
    data_dir = Path(config["data"]["data_dir"])
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["training"].get("device", "cuda:0"))

    train_dataset = PauseNetDataset(data_dir / "train")
    validation_dataset = PauseNetDataset(data_dir / "validation")
    position_template = None
    if bool(
        config.get("model", {}).get(
            "use_position_template",
            PauseNetConfig().use_position_template,
        )
    ):
        position_template = build_position_template(train_dataset)
        np.save(output_dir / "position_template.npy", position_template)

    model = PauseNet(model_config_from_dict(config), position_template=position_template).to(device)
    train_loader = make_loader(
        train_dataset,
        int(config["training"].get("batch_size", 64)),
        int(config["training"].get("num_workers", 4)),
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_dataset,
        int(config["training"].get("batch_size", 64)),
        int(config["training"].get("num_workers", 4)),
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 2e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-5)),
    )

    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    best_metric = float("inf")
    best_path = output_dir / "best_model.pt"
    patience = int(config["training"].get("patience", 8))
    max_epochs = int(config["training"].get("max_epochs", 60))
    wait = 0
    history_path = output_dir / "training_history.tsv"
    with history_path.open("w") as history:
        history.write(
            "epoch\ttrain_loss\ttrain_profile_loss\ttrain_count_loss\t"
            "val_loss\tval_profile_loss\tval_count_loss\t"
            "val_count_pearson_log1p\tseconds\n"
        )
        for epoch in range(1, max_epochs + 1):
            start = time.time()
            train_metrics, _ = run_epoch(model, train_loader, device, config, optimizer=optimizer)
            val_metrics, _ = run_epoch(model, validation_loader, device, config)
            seconds = time.time() - start
            history.write(
                f"{epoch}\t{train_metrics['loss']:.6f}\t"
                f"{train_metrics['profile_loss']:.6f}\t{train_metrics['count_loss']:.6f}\t"
                f"{val_metrics['loss']:.6f}\t{val_metrics['profile_loss']:.6f}\t"
                f"{val_metrics['count_loss']:.6f}\t"
                f"{val_metrics['count_pearson_log1p']:.6f}\t{seconds:.1f}\n"
            )
            history.flush()
            selection_metric = float(val_metrics["loss"])
            if selection_metric < best_metric:
                best_metric = selection_metric
                wait = 0
                save_checkpoint(best_path, model, config, epoch, val_metrics)
            else:
                wait += 1
                if wait >= patience:
                    break
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PauseNet from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_path = train_from_config(load_config(args.config))
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
