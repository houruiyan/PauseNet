"""PauseNet training entry point."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields
import json
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PauseNetDataset
from .losses import pausenet_loss
from .metrics import js_distance_rows, safe_pearson
from .model import PauseNet, PauseNetConfig


TOP_LEVEL_CONFIG_KEYS = {"data", "model", "training"}
DATA_CONFIG_KEYS = {"data_dir"}
TRAINING_CONFIG_KEYS = {
    "output_dir",
    "device",
    "batch_size",
    "num_workers",
    "learning_rate",
    "weight_decay",
    "max_epochs",
    "patience",
    "seed",
    "shuffle_seed",
    "counts_weight",
    "profile_loss",
    "grad_accum_steps",
    "progress",
    "scheduler",
}
SCHEDULER_CONFIG_KEYS = {
    "type",
    "monitor",
    "factor",
    "patience",
    "min_lr",
    "threshold",
}
SCHEDULER_MONITORS = {
    "val_loss": "loss",
    "val_profile_loss": "profile_loss",
    "val_count_loss": "count_loss",
}
PROFILE_LOSSES = {"mnll", "multiresolution_jsd"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mapping_section(config: Mapping, name: str, required: bool = False) -> dict:
    if name not in config:
        if required:
            raise ValueError(f"Missing required configuration section: {name}")
        return {}
    section = config[name]
    if not isinstance(section, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")
    return dict(section)


def _reject_unknown_keys(section: Mapping, allowed: set[str], section_name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in configuration section '{section_name}': "
            f"{', '.join(unknown)}"
        )


def _require_nonempty_string(section: Mapping, key: str, section_name: str) -> None:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{section_name}.{key}' must be a non-empty string.")


def _validate_integer(
    section: Mapping,
    key: str,
    section_name: str,
    *,
    minimum: int,
) -> None:
    if key not in section:
        return
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"'{section_name}.{key}' must be an integer greater than or equal to {minimum}."
        )


def _validate_number(
    section: Mapping,
    key: str,
    section_name: str,
    *,
    minimum: float,
    inclusive: bool = True,
) -> None:
    if key not in section:
        return
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{section_name}.{key}' must be a finite number.")
    numeric = float(value)
    valid_bound = numeric >= minimum if inclusive else numeric > minimum
    if not math.isfinite(numeric) or not valid_bound:
        comparison = "greater than or equal to" if inclusive else "greater than"
        raise ValueError(
            f"'{section_name}.{key}' must be a finite number {comparison} {minimum}."
        )


def validate_config(config: dict) -> dict:
    """Validate a training configuration and reject misspelled or invalid fields."""

    if not isinstance(config, Mapping):
        raise ValueError("The training configuration must be a mapping.")
    config = dict(config)
    _reject_unknown_keys(config, TOP_LEVEL_CONFIG_KEYS, "root")

    data_cfg = _mapping_section(config, "data", required=True)
    model_cfg = _mapping_section(config, "model")
    training_cfg = _mapping_section(config, "training", required=True)

    _reject_unknown_keys(data_cfg, DATA_CONFIG_KEYS, "data")
    model_keys = {field.name for field in fields(PauseNetConfig)}
    _reject_unknown_keys(model_cfg, model_keys, "model")
    _reject_unknown_keys(training_cfg, TRAINING_CONFIG_KEYS, "training")

    _require_nonempty_string(data_cfg, "data_dir", "data")
    _require_nonempty_string(training_cfg, "output_dir", "training")
    if "device" in training_cfg:
        _require_nonempty_string(training_cfg, "device", "training")

    for key in (
        "input_length",
        "output_length",
        "channels",
        "n_dilated_layers",
        "profile_kernel_size",
        "stem_kernel_size",
    ):
        _validate_integer(model_cfg, key, "model", minimum=1)
    _validate_number(model_cfg, "dropout", "model", minimum=0.0)
    if float(model_cfg.get("dropout", 0.0)) >= 1.0:
        raise ValueError("'model.dropout' must be less than 1.0.")

    _validate_integer(training_cfg, "batch_size", "training", minimum=1)
    _validate_integer(training_cfg, "num_workers", "training", minimum=0)
    _validate_integer(training_cfg, "max_epochs", "training", minimum=1)
    _validate_integer(training_cfg, "patience", "training", minimum=0)
    _validate_integer(training_cfg, "seed", "training", minimum=0)
    _validate_integer(training_cfg, "shuffle_seed", "training", minimum=0)
    _validate_integer(training_cfg, "grad_accum_steps", "training", minimum=1)
    _validate_number(
        training_cfg,
        "learning_rate",
        "training",
        minimum=0.0,
        inclusive=False,
    )
    _validate_number(training_cfg, "weight_decay", "training", minimum=0.0)
    _validate_number(training_cfg, "counts_weight", "training", minimum=0.0)

    if "progress" in training_cfg and not isinstance(training_cfg["progress"], bool):
        raise ValueError("'training.progress' must be true or false.")
    profile_loss = str(training_cfg.get("profile_loss", "mnll"))
    if profile_loss not in PROFILE_LOSSES:
        raise ValueError(
            f"'training.profile_loss' must be one of: {', '.join(sorted(PROFILE_LOSSES))}."
        )

    scheduler_cfg = training_cfg.get("scheduler")
    if scheduler_cfg is not None:
        if not isinstance(scheduler_cfg, Mapping):
            raise ValueError("'training.scheduler' must be a mapping.")
        scheduler_cfg = dict(scheduler_cfg)
        _reject_unknown_keys(scheduler_cfg, SCHEDULER_CONFIG_KEYS, "training.scheduler")
        scheduler_type = str(scheduler_cfg.get("type", "reduce_on_plateau"))
        if scheduler_type not in {"none", "reduce_on_plateau"}:
            raise ValueError(
                "'training.scheduler.type' must be 'none' or 'reduce_on_plateau'."
            )
        monitor = str(scheduler_cfg.get("monitor", "val_loss"))
        if monitor not in SCHEDULER_MONITORS:
            raise ValueError(
                "'training.scheduler.monitor' must be one of: "
                f"{', '.join(sorted(SCHEDULER_MONITORS))}."
            )
        _validate_number(
            scheduler_cfg,
            "factor",
            "training.scheduler",
            minimum=0.0,
            inclusive=False,
        )
        factor = float(scheduler_cfg.get("factor", 0.5))
        if factor >= 1.0:
            raise ValueError("'training.scheduler.factor' must be less than 1.0.")
        _validate_integer(
            scheduler_cfg,
            "patience",
            "training.scheduler",
            minimum=0,
        )
        _validate_number(
            scheduler_cfg,
            "min_lr",
            "training.scheduler",
            minimum=0.0,
        )
        _validate_number(
            scheduler_cfg,
            "threshold",
            "training.scheduler",
            minimum=0.0,
        )
        scheduler_patience = int(scheduler_cfg.get("patience", 2))
        early_stopping_patience = int(training_cfg.get("patience", 8))
        if (
            scheduler_type == "reduce_on_plateau"
            and early_stopping_patience > 0
            and scheduler_patience >= early_stopping_patience
        ):
            warnings.warn(
                "training.scheduler.patience is greater than or equal to "
                "training.patience; early stopping may occur before the reduced "
                "learning rate has time to help.",
                UserWarning,
                stacklevel=2,
            )

    return config


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return validate_config(yaml.safe_load(handle))


def make_loader(
    dataset: PauseNetDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    *,
    seed: int = 12345,
    pin_memory: bool = False,
):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def model_config_from_dict(config: dict) -> PauseNetConfig:
    model_cfg = dict(config.get("model", {}))
    allowed = {field.name for field in fields(PauseNetConfig)}
    _reject_unknown_keys(model_cfg, allowed, "model")
    return PauseNetConfig(**model_cfg)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda devices and fail clearly for unavailable CUDA devices."""

    requested = str(requested).strip().lower()
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"Invalid training device '{requested}'. Use auto, cpu or cuda:<index>."
        ) from error

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Training device '{requested}' requires CUDA, but CUDA is not available. "
                "Use device: auto to permit CPU fallback."
            )
        index = 0 if device.index is None else device.index
        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            raise ValueError(
                f"CUDA device index {index} is unavailable; "
                f"torch.cuda.device_count() returned {device_count}."
            )
        return torch.device(f"cuda:{index}")
    if device.type != "cpu":
        raise ValueError(
            f"Unsupported training device type '{device.type}'. "
            "Use auto, cpu or cuda:<index>."
        )
    return device


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: Mapping,
) -> tuple[torch.optim.lr_scheduler.ReduceLROnPlateau | None, str | None]:
    """Build the configured scheduler and return its validation metric key."""

    scheduler_cfg = training_config.get("scheduler")
    if scheduler_cfg is None:
        return None, None
    scheduler_cfg = dict(scheduler_cfg)
    scheduler_type = str(scheduler_cfg.get("type", "reduce_on_plateau"))
    if scheduler_type == "none":
        return None, None
    monitor = str(scheduler_cfg.get("monitor", "val_loss"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 2)),
        threshold=float(scheduler_cfg.get("threshold", 1e-4)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
    )
    return scheduler, SCHEDULER_MONITORS[monitor]


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
    profile_loss_sum = 0.0
    profile_valid_count = 0
    count_loss_sum = 0.0
    sample_count = 0
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
            valid_profiles = profile_mask.bool() & (profiles.sum(dim=-1) > 0)
            valid_count = int(valid_profiles.sum().item())
            profile_loss_sum += float(profile_component.detach()) * valid_count
            profile_valid_count += valid_count
            count_loss_sum += float(count_component.detach()) * batch_size
            sample_count += batch_size

            if not training:
                observed_counts_all.append(batch["counts"].numpy())
                predicted_log_counts_all.append(predicted_log_counts.detach().cpu().numpy())
                masks_all.append(batch["profile_mask"].numpy())
                if keep_profiles:
                    observed_profiles_all.append(batch["profiles"].numpy())
                    predicted_profiles_all.append(
                        torch.softmax(profile_logits.float(), dim=-1).detach().cpu().numpy()
                    )

    profile_epoch_loss = (
        profile_loss_sum / profile_valid_count if profile_valid_count > 0 else 0.0
    )
    count_epoch_loss = count_loss_sum / max(sample_count, 1)
    metrics = {
        "loss": float(profile_epoch_loss + counts_weight * count_epoch_loss),
        "profile_loss": float(profile_epoch_loss),
        "count_loss": float(count_epoch_loss),
        "profile_valid_n": int(profile_valid_count),
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
    config = validate_config(config)
    training_config = config["training"]
    seed = int(training_config.get("seed", 20260713))
    shuffle_seed = int(training_config.get("shuffle_seed", 12345))
    set_seed(seed)
    data_dir = Path(config["data"]["data_dir"])
    output_dir = Path(training_config["output_dir"])
    device = resolve_device(training_config.get("device", "cuda:0"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PauseNetDataset(data_dir / "train")
    validation_dataset = PauseNetDataset(data_dir / "validation")
    model = PauseNet(model_config_from_dict(config)).to(device)
    train_loader = make_loader(
        train_dataset,
        int(training_config.get("batch_size", 64)),
        int(training_config.get("num_workers", 4)),
        shuffle=True,
        seed=shuffle_seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = make_loader(
        validation_dataset,
        int(training_config.get("batch_size", 64)),
        int(training_config.get("num_workers", 4)),
        shuffle=False,
        seed=shuffle_seed,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 2e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-5)),
    )
    scheduler, scheduler_metric_key = make_scheduler(optimizer, training_config)

    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    best_metric = float("inf")
    best_path = output_dir / "best_model.pt"
    patience = int(training_config.get("patience", 8))
    max_epochs = int(training_config.get("max_epochs", 60))
    wait = 0
    history_path = output_dir / "training_history.tsv"
    with history_path.open("w") as history:
        history.write(
            "epoch\ttrain_loss\ttrain_profile_loss\ttrain_count_loss\t"
            "val_loss\tval_profile_loss\tval_count_loss\t"
            "val_count_pearson_log1p\ttrain_profile_valid_n\t"
            "val_profile_valid_n\tlearning_rate\tseconds\n"
        )
        for epoch in range(1, max_epochs + 1):
            start = time.time()
            train_metrics, _ = run_epoch(model, train_loader, device, config, optimizer=optimizer)
            val_metrics, _ = run_epoch(model, validation_loader, device, config)
            seconds = time.time() - start
            learning_rate = float(optimizer.param_groups[0]["lr"])
            history.write(
                f"{epoch}\t{train_metrics['loss']:.6f}\t"
                f"{train_metrics['profile_loss']:.6f}\t{train_metrics['count_loss']:.6f}\t"
                f"{val_metrics['loss']:.6f}\t{val_metrics['profile_loss']:.6f}\t"
                f"{val_metrics['count_loss']:.6f}\t"
                f"{val_metrics['count_pearson_log1p']:.6f}\t"
                f"{train_metrics['profile_valid_n']}\t{val_metrics['profile_valid_n']}\t"
                f"{learning_rate:.10g}\t{seconds:.1f}\n"
            )
            history.flush()
            if scheduler is not None:
                assert scheduler_metric_key is not None
                scheduler.step(float(val_metrics[scheduler_metric_key]))
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
