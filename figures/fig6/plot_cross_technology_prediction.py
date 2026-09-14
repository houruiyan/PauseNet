#!/usr/bin/env python3
"""Evaluate arbitrary PauseNet models on arbitrary test datasets and plot a heatmap.

Each heatmap row is one trained model and each column is one test dataset.
For every model/test combination, the plotted value is the Pearson correlation
between log1p(observed_counts) and log1p(predicted_counts).

Example
-------
python 2.plot_cross_cell_model_test_heatmap.py \
  --model HEK293T=/path/to/hek293t/best_model.pt \
  --model K562=/path/to/k562/best_model.pt \
  --test HEK293T=/path/to/HEK293T/dataset \
  --test K562=/path/to/K562/dataset \
  --output-pdf cross_cell_count_prediction.pdf
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


DEFAULT_PAUSENET_REPO = "/mnt/HDD8TB/houruiyan/pausing_site/PauseNet"
ROOT = Path('/mnt/HDD8TB/houruiyan/pausing_site')
HERE = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = str(HERE / 'model_test_evaluations')
DEFAULT_OUTPUT = str(HERE / 'cross_technology_count_prediction_heatmap.pdf')
TECHNOLOGIES = [('NET-seq', 'hek293t_netseq', 'HEK293T_NETseq'),
                ('PRO-seq', 'hek293t_proseq', 'HEK293T_PROseq'),
                ('GRO-seq', 'hek293t_groseq', 'HEK293T_GROseq')]


def check_splits():
    """Abort cross-assay evaluation if test chromosomes occur in training."""
    import pandas as pd
    train_chroms = {}
    test_chroms = {}
    for label, _, dataset in TECHNOLOGIES:
        base = ROOT / 'data' / dataset / 'dataset'
        train_chroms[label] = set(pd.read_csv(base / 'train/manifest.tsv', sep='\t', usecols=['chrom']).chrom)
        test_chroms[label] = set(pd.read_csv(base / 'test/manifest.tsv', sep='\t', usecols=['chrom']).chrom)
    for source in train_chroms:
        for target in test_chroms:
            overlap = train_chroms[source] & test_chroms[target]
            if overlap:
                raise ValueError(f'Train/test chromosome leakage: {source} -> {target}: {overlap}')
    print('[checked] All 9 train/test chromosome combinations are disjoint.', flush=True)


def set_figure_style() -> None:
    """Apply final-figure-small typography and editable PDF fonts."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_labeled_path(value: str, option: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"{option} must use LABEL=/path syntax; received: {value!r}"
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(
            f"{option} must contain both a label and a path: {value!r}"
        )
    return label, Path(raw_path)


def unique_labeled_paths(
    values: Sequence[str], option: str
) -> List[Tuple[str, Path]]:
    parsed = [parse_labeled_path(value, option) for value in values]
    labels = [label for label, _ in parsed]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate labels in {option}: {', '.join(duplicates)}"
        )
    return parsed


def slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("._")
    return slug or "unnamed"


def validate_unique_slugs(items: Sequence[Tuple[str, Path]], option: str) -> None:
    slugs = [slugify(label).lower() for label, _ in items]
    if len(set(slugs)) != len(slugs):
        raise ValueError(
            f"Labels in {option} become non-unique directory names after sanitizing"
        )


def locate_profiles(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}_profiles.npz"


def evaluate_one_combination(
    checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    split: str,
    device: str,
    batch_size: int,
    num_workers: int,
    pausenet_repo: Path,
    force: bool,
) -> Path:
    profiles_path = locate_profiles(output_dir, split)
    if profiles_path.exists() and not force:
        print(f"[reuse] {profiles_path}")
        return profiles_path

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not (data_dir / split).is_dir():
        raise FileNotFoundError(
            f"Dataset split directory does not exist: {data_dir / split}"
        )
    if not pausenet_repo.is_dir():
        raise FileNotFoundError(f"PauseNet repository does not exist: {pausenet_repo}")

    repo_text = str(pausenet_repo.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    try:
        from pausenet.evaluate import evaluate_checkpoint
    except ImportError as error:
        raise ImportError(
            f"Could not import pausenet from {pausenet_repo}"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evaluate] model={checkpoint} test={data_dir / split}")
    evaluate_checkpoint(
        data_dir=data_dir,
        checkpoint=checkpoint,
        output_dir=output_dir,
        split=split,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if not profiles_path.is_file():
        raise RuntimeError(f"Evaluation did not create the expected file: {profiles_path}")

    # Release model/output memory before evaluating the next combination.
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return profiles_path


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denominator == 0:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def correlation_from_npz(
    profiles_path: Path,
    min_observed_count: float,
    use_profile_mask: bool,
) -> Tuple[float, int]:
    with np.load(profiles_path, allow_pickle=False) as data:
        required = {"observed_counts", "predicted_counts"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                f"{profiles_path} is missing NPZ keys: {', '.join(missing)}"
            )
        observed = np.asarray(data["observed_counts"], dtype=np.float64).reshape(-1)
        predicted = np.asarray(data["predicted_counts"], dtype=np.float64).reshape(-1)
        if observed.size != predicted.size:
            raise ValueError(f"Count arrays have different lengths in {profiles_path}")

        keep = (
            np.isfinite(observed)
            & np.isfinite(predicted)
            & (observed >= min_observed_count)
            & (predicted >= 0)
        )
        if use_profile_mask:
            if "profile_masks" not in data.files:
                raise KeyError(
                    f"--use-profile-mask was set but profile_masks is absent: "
                    f"{profiles_path}"
                )
            profile_mask = np.asarray(data["profile_masks"], dtype=bool).reshape(-1)
            if profile_mask.size != observed.size:
                raise ValueError(f"profile_masks has an invalid length in {profiles_path}")
            keep &= profile_mask

    n = int(keep.sum())
    if n < 2:
        raise ValueError(
            f"Only {n} windows remain after filtering in {profiles_path}"
        )
    r = pearson_r(np.log1p(observed[keep]), np.log1p(predicted[keep]))
    return r, n


def rounded_color_limits(
    values: np.ndarray,
    requested_vmin: float | None,
    requested_vmax: float | None,
) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite Pearson correlations were calculated")
    vmin = (
        float(requested_vmin)
        if requested_vmin is not None
        else max(-1.0, np.floor(finite.min() * 20.0) / 20.0)
    )
    vmax = (
        float(requested_vmax)
        if requested_vmax is not None
        else min(1.0, np.ceil(finite.max() * 20.0) / 20.0)
    )
    if vmax <= vmin:
        raise ValueError("Color scale requires --vmax to be greater than --vmin")
    return vmin, vmax


def plot_heatmap(
    values: np.ndarray,
    model_labels: Sequence[str],
    test_labels: Sequence[str],
    output_pdf: Path,
    title: str | None,
    vmin: float | None,
    vmax: float | None,
    figure_width: float | None,
    figure_height: float | None,
) -> None:
    n_models, n_tests = values.shape
    color_vmin, color_vmax = rounded_color_limits(values, vmin, vmax)
    norm = Normalize(vmin=color_vmin, vmax=color_vmax)
    cmap = plt.get_cmap("YlGnBu")

    # Use a wider canvas than the single-panel final-figure-small default.
    # The extra horizontal space protects row labels and the colorbar label,
    # while larger matrices continue to expand automatically.
    width = (
        figure_width
        if figure_width is not None
        else max(6.5, 1.00 * n_tests + 2.50)
    )
    height = (
        figure_height
        if figure_height is not None
        else max(4.4, 0.76 * n_models + 1.35)
    )
    fig, ax = plt.subplots(figsize=(width, height))
    # Reserve independent areas for y labels, rotated x labels, and colorbar.
    fig.subplots_adjust(left=0.19, right=0.77, bottom=0.28, top=0.85)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(n_tests))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(test_labels, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(model_labels)
    ax.set_xlabel("Test dataset", fontsize=14)
    ax.set_ylabel("Training model", fontsize=14)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=12)

    # White boundaries reproduce the compact matrix style in the reference.
    ax.set_xticks(np.arange(-0.5, n_tests, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_models, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(n_models):
        for col in range(n_tests):
            value = values[row, col]
            relative = norm(value)
            text_color = "white" if relative >= 0.55 else "black"
            ax.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=12,
                color=text_color,
            )

    colorbar_ax = fig.add_axes([0.81, 0.28, 0.025, 0.57])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Pearson R of log1p(counts)", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)

    output_pdf = Path(output_pdf)
    if output_pdf.suffix.lower() != ".pdf":
        raise ValueError("--output-pdf must end with .pdf")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Check all rendered text against the fixed page; never resize using tight bbox.
    fig.canvas.draw()
    from matplotlib.text import Text
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    for artist in fig.findobj(match=Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.width > 0 and box.height > 0 and (
                box.x0 < page.x0 or box.y0 < page.y0 or
                box.x1 > page.x1 or box.y1 > page.y1
            ):
                raise RuntimeError(f'Text outside figure boundary: {artist.get_text()!r}')
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every supplied PauseNet model on every supplied test "
            "dataset and plot a model-by-test Pearson correlation heatmap."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=False,
        metavar="LABEL=CHECKPOINT",
        help="Training-model label and best_model.pt path; repeat for each model.",
    )
    parser.add_argument(
        "--test",
        action="append",
        required=False,
        metavar="LABEL=DATA_DIR",
        help=(
            "Test-dataset label and dataset root containing test/; "
            "repeat for each dataset."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--pausenet-repo", default=DEFAULT_PAUSENET_REPO)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-pdf", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate and overwrite cached model/test results.",
    )
    parser.add_argument(
        "--min-observed-count",
        type=float,
        default=0.0,
        help="Minimum observed count retained within each model/test combination.",
    )
    parser.add_argument(
        "--use-profile-mask",
        action="store_true",
        help="Calculate each correlation only from windows with profile_masks=True.",
    )
    parser.add_argument(
        "--title",
        default="Cross-technology count prediction",
        help="Heatmap title; pass an empty string to omit it.",
    )
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--figure-width", type=float, default=6)
    parser.add_argument("--figure-height", type=float, default=4)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.model is not None or args.test is not None:
        parser.error('This preset uses the three HEK293T technology models and datasets; edit TECHNOLOGIES to change them.')
    args.model = [f'{label}={ROOT}/2_train_model/train/{model}/best_model.pt' for label, model, _ in TECHNOLOGIES]
    args.test = [f'{label}={ROOT}/data/{dataset}/dataset' for label, _, dataset in TECHNOLOGIES]
    return args


def main() -> None:
    args = parse_args()
    if args.split != 'test':
        raise ValueError('Only held-out test evaluation is allowed in this preset.')
    check_splits()
    for _, model, dataset in TECHNOLOGIES:
        if not (ROOT / '2_train_model/train' / model / 'best_model.pt').is_file():
            raise FileNotFoundError(model)
        if not (ROOT / 'data' / dataset / 'dataset/test/sequence_codes.npy').is_file():
            raise FileNotFoundError(dataset)
    if args.check_only:
        print('Input paths and held-out splits verified. No inference performed.')
        return
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.min_observed_count < 0:
        raise ValueError("--min-observed-count cannot be negative")
    if args.figure_width is not None and args.figure_width <= 0:
        raise ValueError("--figure-width must be positive")
    if args.figure_height is not None and args.figure_height <= 0:
        raise ValueError("--figure-height must be positive")

    models = unique_labeled_paths(args.model, "--model")
    tests = unique_labeled_paths(args.test, "--test")
    validate_unique_slugs(models, "--model")
    validate_unique_slugs(tests, "--test")

    work_dir = Path(args.work_dir)
    values = np.full((len(models), len(tests)), np.nan, dtype=np.float64)
    sample_sizes = np.zeros_like(values, dtype=np.int64)
    for row, (model_label, checkpoint) in enumerate(models):
        for col, (test_label, data_dir) in enumerate(tests):
            combination_dir = (
                work_dir
                / f"{slugify(model_label)}__on__{slugify(test_label)}"
            )
            profiles_path = evaluate_one_combination(
                checkpoint=checkpoint,
                data_dir=data_dir,
                output_dir=combination_dir,
                split=args.split,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pausenet_repo=Path(args.pausenet_repo),
                force=args.force,
            )
            r, n = correlation_from_npz(
                profiles_path=profiles_path,
                min_observed_count=args.min_observed_count,
                use_profile_mask=args.use_profile_mask,
            )
            values[row, col] = r
            sample_sizes[row, col] = n
            print(
                f"[R] {model_label} -> {test_label}: "
                f"R={r:.3f}, n={n:,}"
            )

    set_figure_style()
    plot_heatmap(
        values=values,
        model_labels=[label for label, _ in models],
        test_labels=[label for label, _ in tests],
        output_pdf=Path(args.output_pdf),
        title=args.title or None,
        vmin=args.vmin,
        vmax=args.vmax,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
    )

    print("\nPearson R matrix:")
    print("\t" + "\t".join(label for label, _ in tests))
    for (label, _), row in zip(models, values):
        print(label + "\t" + "\t".join(f"{value:.3f}" for value in row))
    print("\nWindow-count matrix:")
    print("\t" + "\t".join(label for label, _ in tests))
    for (label, _), row in zip(models, sample_sizes):
        print(label + "\t" + "\t".join(f"{int(value):,}" for value in row))
    print(f"\nSaved: {args.output_pdf}")


if __name__ == "__main__":
    main()
