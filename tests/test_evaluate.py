from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from pausenet import cli, evaluate
from pausenet.model import PauseNet, PauseNetConfig


def checkpoint_config(
    *,
    counts_weight: float = 100.0,
    profile_loss: str = "mnll",
) -> dict:
    return {
        "model": asdict(PauseNetConfig()),
        "training": {
            "counts_weight": counts_weight,
            "profile_loss": profile_loss,
        },
    }


def save_checkpoint(
    path: Path,
    *,
    config: dict | None,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": state_dict
        if state_dict is not None
        else PauseNet().state_dict(),
    }
    if config is not None:
        checkpoint["config"] = config
    torch.save(checkpoint, path)


def test_load_model_requires_checkpoint_model_config(tmp_path: Path) -> None:
    path = tmp_path / "missing_config.pt"
    save_checkpoint(path, config=None)

    with pytest.raises(RuntimeError, match="config"):
        evaluate.load_model(path, torch.device("cpu"))


def test_load_model_rejects_incomplete_state_dict(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.pt"
    state_dict = PauseNet().state_dict()
    state_dict.pop(next(iter(state_dict)))
    save_checkpoint(path, config=checkpoint_config(), state_dict=state_dict)

    with pytest.raises(RuntimeError, match="do not exactly match"):
        evaluate.load_model(path, torch.device("cpu"))


def test_load_model_accepts_exact_checkpoint_and_returns_config(tmp_path: Path) -> None:
    path = tmp_path / "valid.pt"
    config = checkpoint_config(counts_weight=7.0, profile_loss="multiresolution_jsd")
    save_checkpoint(path, config=config)

    model, loaded_config = evaluate.load_model(path, torch.device("cpu"))

    assert isinstance(model, PauseNet)
    assert model.training is False
    assert loaded_config == config


def test_evaluate_inherits_checkpoint_loss_and_resolves_auto_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class FakeDataset:
        manifest = None

        def __len__(self) -> int:
            return 1

    def fake_resolve_device(requested: str) -> torch.device:
        captured["requested_device"] = requested
        return torch.device("cpu")

    def fake_loader(dataset: FakeDataset, **kwargs):
        captured["pin_memory"] = kwargs["pin_memory"]
        return object()

    def fake_load_model(checkpoint: str | Path, device: torch.device):
        captured["model_device"] = device
        return (
            object(),
            checkpoint_config(
                counts_weight=7.0,
                profile_loss="multiresolution_jsd",
            ),
        )

    def fake_run_epoch(
        model: object,
        loader: object,
        device: torch.device,
        config: dict,
        keep_profiles: bool,
    ):
        del model, loader, device, keep_profiles
        captured["loss_config"] = config
        return (
            {
                "loss": 1.0,
                "profile_loss": 0.5,
                "count_loss": 0.1,
            },
            {
                "observed_profiles": np.array([[1.0, 0.0]]),
                "predicted_profiles": np.array([[1.0, 0.0]]),
                "profile_masks": np.array([True]),
                "observed_counts": np.array([1.0]),
                "predicted_counts": np.array([1.0]),
                "predicted_log1p_counts": np.array([np.log(2.0)]),
            },
        )

    monkeypatch.setattr(evaluate, "resolve_device", fake_resolve_device)
    monkeypatch.setattr(evaluate, "PauseNetDataset", lambda path: FakeDataset())
    monkeypatch.setattr(evaluate, "DataLoader", fake_loader)
    monkeypatch.setattr(evaluate, "load_model", fake_load_model)
    monkeypatch.setattr(evaluate, "run_epoch", fake_run_epoch)
    monkeypatch.setattr(
        evaluate,
        "profile_similarity_by_count_threshold",
        lambda *args, **kwargs: [
            {
                "count_threshold": 0,
                "comparison": "PauseNet",
                "resolution_bp": 1,
                "similarity_1_minus_jsd": 1.0,
                "n": 1,
            }
        ],
    )

    evaluate.evaluate_checkpoint(
        data_dir=tmp_path,
        checkpoint=tmp_path / "unused.pt",
        output_dir=tmp_path / "evaluation",
        device="auto",
        num_workers=0,
        profile_count_thresholds=(0,),
    )

    assert captured["requested_device"] == "auto"
    assert captured["model_device"] == torch.device("cpu")
    assert captured["pin_memory"] is False
    assert captured["loss_config"] == {
        "training": {
            "counts_weight": 7.0,
            "profile_loss": "multiresolution_jsd",
            "progress": False,
        }
    }


def test_pausenet_cli_evaluate_defaults_to_auto_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_evaluate_checkpoint(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pausenet",
            "evaluate",
            "--data-dir",
            "data",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "evaluation",
        ],
    )

    cli.main()

    assert captured["device"] == "auto"
