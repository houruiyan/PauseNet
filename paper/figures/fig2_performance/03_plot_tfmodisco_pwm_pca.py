#!/usr/bin/env python3
"""Create a PWM-similarity motif atlas and PCA-style TF-MoDISco projections."""

import argparse
import json
from pathlib import Path

import h5py
import logomaker
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


BASES = ["A", "C", "G", "T"]
RC_COLUMNS = [3, 2, 1, 0]
TASK_COLORS = {"count": "#E45756", "profile": "#3FA7D6"}
SIGN_COLORS = {"pos": "#EF5285", "neg": "#60C5BA"}
ELLIPSE_COLORS = ["#8DD3C7", "#FDB462", "#B3DE69", "#BC80BD"]


def ppm_from_sequence(sequence):
    sequence = np.asarray(sequence, dtype=float)
    return sequence / np.maximum(sequence.sum(axis=1, keepdims=True), 1e-8)


def information_content(ppm):
    return 2.0 + np.sum(ppm * np.log2(np.maximum(ppm, 1e-8)), axis=1)


def trim_to_informative_core(ppm, flank=1, ic_absolute=0.35, ic_relative=0.35,
                             min_length=5, max_length=24):
    information = information_content(ppm)
    normalized = information / max(float(information.max()), 1e-8)
    informative = np.where((information >= ic_absolute) & (normalized >= ic_relative))[0]

    if len(informative) == 0:
        center = int(np.argmax(normalized))
        start = max(0, center - min_length // 2)
        end = min(len(ppm), start + min_length)
        start = max(0, end - min_length)
    else:
        splits = np.where(np.diff(informative) > 1)[0]
        starts = np.r_[informative[0], informative[splits + 1]]
        ends = np.r_[informative[splits], informative[-1]] + 1
        start, end = max(
            zip(starts, ends),
            key=lambda interval: float(normalized[interval[0]:interval[1]].sum()),
        )
        start = max(0, int(start) - flank)
        end = min(len(ppm), int(end) + flank)
        if end - start < min_length:
            center = (start + end) // 2
            start = max(0, center - min_length // 2)
            end = min(len(ppm), start + min_length)
            start = max(0, end - min_length)

    if end - start > max_length:
        center = int(np.round(np.average(
            np.arange(start, end), weights=normalized[start:end] + 1e-6
        )))
        start = max(0, center - max_length // 2)
        end = min(len(ppm), start + max_length)
        start = max(0, end - max_length)
    return ppm[start:end], int(start), int(end)


def reverse_complement_pwm(ppm):
    return ppm[::-1, :][:, RC_COLUMNS]


def logo_matrix(ppm):
    return ppm * information_content(ppm)[:, None]


def flat_pearson(left, right):
    left = left.ravel() - left.mean()
    right = right.ravel() - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)


def shifted_reverse_complement_similarity(left_ppm, right_ppm, min_overlap=5):
    left = logo_matrix(left_ppm)
    best = -1.0
    for reverse_complement in (False, True):
        right = logo_matrix(
            reverse_complement_pwm(right_ppm) if reverse_complement else right_ppm
        )
        left_length, right_length = len(left), len(right)
        for shift in range(-right_length + min_overlap, left_length - min_overlap + 1):
            left_start = max(0, shift)
            right_start = max(0, -shift)
            overlap = min(left_length - left_start, right_length - right_start)
            if overlap < min_overlap:
                continue
            score = flat_pearson(
                left[left_start:left_start + overlap],
                right[right_start:right_start + overlap],
            )
            score *= np.sqrt(overlap / min(left_length, right_length))
            best = max(best, score)
    return best


def compute_distance_matrix(pwms):
    n_patterns = len(pwms)
    similarity = np.eye(n_patterns)
    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            score = shifted_reverse_complement_similarity(pwms[i], pwms[j])
            similarity[i, j] = score
            similarity[j, i] = score
    scaled_similarity = (np.clip(similarity, -1.0, 1.0) + 1.0) / 2.0
    distance = 1.0 - scaled_similarity
    np.fill_diagonal(distance, 0.0)
    return similarity, distance


def pcoa(distance):
    n_patterns = len(distance)
    centering = np.eye(n_patterns) - np.ones((n_patterns, n_patterns)) / n_patterns
    gram = -0.5 * centering @ (distance ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    coordinates = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0.0))
    positive = eigenvalues[eigenvalues > 1e-10]
    variance = eigenvalues[:2] / positive.sum()
    return coordinates, variance, eigenvalues


def load_patterns(pattern_files):
    rows = []
    pwms = []
    for task, pattern_file in pattern_files.items():
        with h5py.File(pattern_file, "r") as h5:
            task_index = 0
            for group_name, sign in (("pos_patterns", "pos"), ("neg_patterns", "neg")):
                group = h5[group_name]
                pattern_names = sorted(group, key=lambda name: int(name.rsplit("_", 1)[1]))
                for pattern_name in pattern_names:
                    pattern = group[pattern_name]
                    ppm = ppm_from_sequence(pattern["sequence"][:])
                    core, start, end = trim_to_informative_core(ppm)
                    original_index = int(pattern_name.rsplit("_", 1)[1])
                    n_seqlets = int(pattern["seqlets"]["n_seqlets"][:][0])
                    label = f"{'C' if task == 'count' else 'P'}{task_index}"
                    rows.append({
                        "motif_uid": f"{task}_{sign}_pattern_{original_index}",
                        "label": label,
                        "task": task,
                        "sign": sign,
                        "pattern_index": original_index,
                        "task_index": task_index,
                        "n_seqlets": n_seqlets,
                        "core_start_0based": start,
                        "core_end_0based_exclusive": end,
                        "core_length": len(core),
                        "consensus": "".join(BASES[index] for index in np.argmax(core, axis=1)),
                    })
                    pwms.append(core)
                    task_index += 1
    return pd.DataFrame(rows), pwms


def axis_limits(coordinates, padding=0.12):
    x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()
    x_range = max(x_max - x_min, 1e-3)
    y_range = max(y_max - y_min, 1e-3)
    return (
        (x_min - padding * x_range, x_max + padding * x_range),
        (y_min - padding * y_range, y_max + padding * y_range),
    )


def plot_motif_atlas(patterns, pwms, output_pdf):
    n_columns = 8
    n_rows = int(np.ceil(len(patterns) / n_columns))
    figure = plt.figure(figsize=(13.6, 1.12 * n_rows + 0.75))
    grid = figure.add_gridspec(n_rows, n_columns, hspace=0.83, wspace=0.28)

    for index, row in patterns.reset_index(drop=True).iterrows():
        axis = figure.add_subplot(grid[index // n_columns, index % n_columns])
        matrix = logo_matrix(pwms[index])
        logomaker.Logo(
            pd.DataFrame(matrix, columns=BASES),
            ax=axis,
            color_scheme="classic",
            width=0.9,
        )
        axis.set_ylim(0, max(0.75, float(matrix.max()) * 1.13))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        sign = "+" if row.sign == "pos" else "-"
        axis.set_title(
            f"{row.label} ({sign}) | {row.core_length} bp | n={row.n_seqlets}",
            fontsize=5.4,
            color=SIGN_COLORS[row.sign],
            fontweight="bold",
            pad=2.0,
        )

    for index in range(len(patterns), n_rows * n_columns):
        figure.add_subplot(grid[index // n_columns, index % n_columns]).axis("off")

    figure.text(0.02, 0.992, "Updated count and profile TF-MoDISco PWM motif atlas",
                ha="left", va="top", fontsize=11, fontweight="bold")
    figure.text(0.02, 0.972,
                "C = count motif; P = profile motif; + / - denotes the TF-MoDISco contribution class.",
                ha="left", va="top", fontsize=6.6, color="0.30")
    figure.subplots_adjust(left=0.025, right=0.99, bottom=0.025, top=0.945)
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def plot_labeled_pca(patterns, variance, output_pdf):
    figure, axis = plt.subplots(figsize=(10.5, 8.4))
    for task in ("count", "profile"):
        subset = patterns.loc[patterns.task == task]
        axis.scatter(
            subset.PC1,
            subset.PC2,
            s=30,
            c=TASK_COLORS[task],
            edgecolors="black",
            linewidths=0.35,
            alpha=1.0,
            label=task,
            zorder=3,
        )

    (x_limits, y_limits) = axis_limits(patterns[["PC1", "PC2"]].to_numpy())
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    offset_x = (x_limits[1] - x_limits[0]) * 0.006
    offset_y = (y_limits[1] - y_limits[0]) * 0.006
    for _, row in patterns.iterrows():
        axis.text(
            row.PC1 + offset_x,
            row.PC2 + offset_y,
            row.label,
            fontsize=4.9,
            color="0.08",
            ha="left",
            va="bottom",
            zorder=4,
            path_effects=[path_effects.withStroke(linewidth=1.2, foreground="white")],
        )

    axis.axhline(0, color="0.88", linewidth=0.7, zorder=0)
    axis.axvline(0, color="0.88", linewidth=0.7, zorder=0)
    axis.set_xlabel(f"PC1 / PCo1 ({variance[0] * 100:.1f}% positive eig. variance)")
    axis.set_ylabel(f"PC2 / PCo2 ({variance[1] * 100:.1f}% positive eig. variance)")
    axis.set_title("Updated count and profile TF-MoDISco motifs by PWM similarity",
                   fontsize=11, fontweight="bold", pad=8)
    axis.legend(loc="upper right", fontsize=8, markerscale=1.1)
    figure.text(0.01, 0.012,
                "Pairwise PWM similarity is optimized over motif alignment and reverse-complement orientation.",
                ha="left", va="bottom", fontsize=6.6, color="0.35")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def ellipse_for_cluster(axis, coordinates, color):
    if len(coordinates) < 3:
        return
    covariance = np.cov(coordinates, rowvar=False)
    if not np.all(np.isfinite(covariance)):
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-8)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    scale = 2.15
    ellipse = Ellipse(
        xy=coordinates.mean(axis=0),
        width=2 * scale * np.sqrt(eigenvalues[0]),
        height=2 * scale * np.sqrt(eigenvalues[1]),
        angle=angle,
        facecolor=color,
        edgecolor="none",
        alpha=0.22,
        zorder=1,
    )
    axis.add_patch(ellipse)


def plot_clustered_pca(patterns, variance, output_pdf):
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    for cluster, color in zip(sorted(patterns.cluster.unique()), ELLIPSE_COLORS):
        cluster_coordinates = patterns.loc[patterns.cluster == cluster, ["PC1", "PC2"]].to_numpy()
        ellipse_for_cluster(axis, cluster_coordinates, color)

    for task in ("count", "profile"):
        subset = patterns.loc[patterns.task == task]
        axis.scatter(
            subset.PC1,
            subset.PC2,
            s=38,
            c=TASK_COLORS[task],
            edgecolors="black",
            linewidths=0.4,
            alpha=1.0,
            label="counts" if task == "count" else "profile",
            zorder=3,
        )

    (x_limits, y_limits) = axis_limits(patterns[["PC1", "PC2"]].to_numpy(), padding=0.15)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.axhline(0, color="0.88", linewidth=0.7, zorder=0)
    axis.axvline(0, color="0.88", linewidth=0.7, zorder=0)
    axis.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance explained)")
    axis.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance explained)")
    axis.legend(loc="upper right", fontsize=9.5, frameon=False)
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-h5", required=True, type=Path)
    parser.add_argument("--profile-h5", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })
    args.outdir.mkdir(parents=True, exist_ok=True)

    patterns, pwms = load_patterns({"count": args.count_h5, "profile": args.profile_h5})
    similarity, distance = compute_distance_matrix(pwms)
    coordinates, variance, eigenvalues = pcoa(distance)
    patterns["PC1"] = coordinates[:, 0]
    patterns["PC2"] = coordinates[:, 1]

    patterns["cluster"] = KMeans(
        n_clusters=4, random_state=17, n_init=50
    ).fit_predict(coordinates) + 1

    patterns.to_csv(args.outdir / "updated_tfmodisco_pwm_pcoa_motifs.tsv", sep="\t", index=False)
    np.save(args.outdir / "updated_tfmodisco_pwm_similarity.npy", similarity)
    np.save(args.outdir / "updated_tfmodisco_pwm_distance.npy", distance)
    np.save(args.outdir / "updated_tfmodisco_pwm_pcoa_eigenvalues.npy", eigenvalues)

    plot_motif_atlas(patterns, pwms, args.outdir / "updated_tfmodisco_motif_atlas.pdf")
    plot_labeled_pca(patterns, variance, args.outdir / "updated_tfmodisco_pwm_pcoa_labeled.pdf")
    plot_clustered_pca(patterns, variance, args.outdir / "updated_tfmodisco_pwm_pcoa_clusters.pdf")

    summary = {
        "n_motifs": int(len(patterns)),
        "n_count_motifs": int((patterns.task == "count").sum()),
        "n_profile_motifs": int((patterns.task == "profile").sum()),
        "n_pwm_clusters": 4,
        "cluster_method": "K-means (k=4) on the first two PCoA coordinates",
        "variance_fraction": [float(value) for value in variance],
        "method": (
            "Classical PCoA of pairwise PWM distances. Similarity is optimized over "
            "relative alignment and reverse-complement orientation of information-content PWMs."
        ),
        "figures": [
            "updated_tfmodisco_motif_atlas.pdf",
            "updated_tfmodisco_pwm_pcoa_labeled.pdf",
            "updated_tfmodisco_pwm_pcoa_clusters.pdf",
        ],
    }
    (args.outdir / "updated_tfmodisco_pwm_pcoa_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
