from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from pausenet import train


def minimal_config(tmp_path: Path) -> dict:
    return {
        "data": {"data_dir": str(tmp_path / "data")},
        "model": {},
        "training": {
            "output_dir": str(tmp_path / "output"),
            "device": "cpu",
            "progress": False,
        },
    }


def loader_order(seed: int) -> list[int]:
    dataset = TensorDataset(torch.arange(24))
    loader = train.make_loader(
        dataset,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        seed=seed,
        pin_memory=False,
    )
    return torch.cat([batch[0] for batch in loader]).tolist()


def test_repository_yaml_configs_pass_strict_validation() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    paths = sorted(config_dir.rglob("*.yaml"))
    assert paths
    for path in paths:
        train.load_config(path)


@pytest.mark.parametrize(
    ("section", "bad_key"),
    [
        ("model", "channel"),
        ("training", "learnng_rate"),
    ],
)
def test_unknown_config_keys_raise(
    tmp_path: Path,
    section: str,
    bad_key: str,
) -> None:
    config = minimal_config(tmp_path)
    config[section][bad_key] = 123
    with pytest.raises(ValueError, match=bad_key):
        train.validate_config(config)


def test_unknown_scheduler_key_raises(tmp_path: Path) -> None:
    config = minimal_config(tmp_path)
    config["training"]["scheduler"] = {
        "type": "reduce_on_plateau",
        "patiance": 2,
    }
    with pytest.raises(ValueError, match="patiance"):
        train.validate_config(config)


def test_shuffle_seed_controls_training_order() -> None:
    first = loader_order(17)
    repeated = loader_order(17)
    changed = loader_order(18)
    assert first == repeated
    assert first != changed


def test_make_loader_uses_requested_pin_memory_setting() -> None:
    dataset = TensorDataset(torch.arange(4))
    cpu_loader = train.make_loader(
        dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    cuda_loader = train.make_loader(
        dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )
    assert cpu_loader.pin_memory is False
    assert cuda_loader.pin_memory is True


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert train.resolve_device("auto") == torch.device("cpu")


def test_resolve_device_explicit_cuda_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        train.resolve_device("cuda:0")


def test_resolve_device_rejects_out_of_range_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="index 1 is unavailable"):
        train.resolve_device("cuda:1")


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.zeros(()))

    def forward(self, sequence_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(sequence_codes)
        logits = self.placeholder + torch.zeros(batch_size, 1)
        predicted_counts = self.placeholder + torch.zeros(batch_size)
        return logits, predicted_counts


def make_batch(size: int, valid: int) -> dict[str, torch.Tensor]:
    profiles = torch.zeros(size, 1)
    profile_mask = torch.zeros(size, dtype=torch.uint8)
    profiles[:valid] = 1
    profile_mask[:valid] = 1
    return {
        "sequence_codes": torch.zeros(size, 1, dtype=torch.long),
        "profiles": profiles,
        "counts": torch.arange(1, size + 1, dtype=torch.float32),
        "profile_mask": profile_mask,
    }


def test_run_epoch_weights_profile_loss_by_valid_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [make_batch(size=4, valid=1), make_batch(size=2, valid=2)]

    def fake_loss(
        profile_logits: torch.Tensor,
        predicted_log1p_counts: torch.Tensor,
        observed_profiles: torch.Tensor,
        observed_counts: torch.Tensor,
        profile_masks: torch.Tensor,
        *,
        counts_weight: float,
        profile_loss: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del predicted_log1p_counts, observed_counts, profile_masks, profile_loss
        if len(observed_profiles) == 4:
            profile_value, count_value = 10.0, 2.0
        else:
            profile_value, count_value = 4.0, 6.0
        zero = profile_logits.sum() * 0.0
        profile_component = zero + profile_value
        count_component = zero + count_value
        return (
            profile_component + counts_weight * count_component,
            profile_component,
            count_component,
        )

    monkeypatch.setattr(train, "pausenet_loss", fake_loss)
    metrics, _ = train.run_epoch(
        DummyModel(),
        batches,
        torch.device("cpu"),
        {"training": {"counts_weight": 2.0, "progress": False}},
    )

    expected_profile = (10.0 * 1 + 4.0 * 2) / 3
    expected_count = (2.0 * 4 + 6.0 * 2) / 6
    assert metrics["profile_valid_n"] == 3
    assert metrics["profile_loss"] == pytest.approx(expected_profile)
    assert metrics["count_loss"] == pytest.approx(expected_count)
    assert metrics["loss"] == pytest.approx(expected_profile + 2.0 * expected_count)


def test_scheduler_steps_every_epoch_with_configured_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = minimal_config(tmp_path)
    config["training"].update(
        {
            "max_epochs": 3,
            "patience": 10,
            "scheduler": {
                "type": "reduce_on_plateau",
                "monitor": "val_profile_loss",
                "factor": 0.5,
                "patience": 2,
                "min_lr": 1e-6,
            },
        }
    )

    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(train, "PauseNetDataset", lambda path: object())
    monkeypatch.setattr(train, "PauseNet", lambda model_config: model)
    monkeypatch.setattr(train, "make_loader", lambda *args, **kwargs: object())

    validation_profile_losses = iter([3.0, 2.0, 1.0])

    def fake_run_epoch(
        model: torch.nn.Module,
        loader: object,
        device: torch.device,
        config: dict,
        optimizer: torch.optim.Optimizer | None = None,
        keep_profiles: bool = False,
    ) -> tuple[dict, dict]:
        del model, loader, device, config, keep_profiles
        profile_loss = 4.0 if optimizer is not None else next(validation_profile_losses)
        return (
            {
                "loss": profile_loss + 1.0,
                "profile_loss": profile_loss,
                "count_loss": 0.01,
                "count_pearson_log1p": 0.0,
                "profile_valid_n": 5,
            },
            {},
        )

    monkeypatch.setattr(train, "run_epoch", fake_run_epoch)

    scheduler_instances = []

    class FakeReduceLROnPlateau:
        def __init__(self, optimizer: torch.optim.Optimizer, **kwargs) -> None:
            self.optimizer = optimizer
            self.kwargs = kwargs
            self.metrics = []
            scheduler_instances.append(self)

        def step(self, metric: float) -> None:
            self.metrics.append(metric)

    monkeypatch.setattr(
        torch.optim.lr_scheduler,
        "ReduceLROnPlateau",
        FakeReduceLROnPlateau,
    )

    train.train_from_config(config)

    assert len(scheduler_instances) == 1
    assert scheduler_instances[0].metrics == [3.0, 2.0, 1.0]
    history = (tmp_path / "output" / "training_history.tsv").read_text()
    assert "learning_rate" in history.splitlines()[0]
