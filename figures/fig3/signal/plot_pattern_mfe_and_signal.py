#!/usr/bin/env python3
"""Mean DNA MFE + observed/predicted pausing signal around seqlets of one
TF-MoDISco pattern (HEK293T NET-seq PauseNet).

Works for BOTH TF-MoDISco tasks:
  - profile head patterns (only pos_patterns exist), e.g.
      python plot_profile_pattern_mfe_and_signal.py --task profile --pattern-index 37
  - count head patterns (pos_patterns and neg_patterns), e.g.
      python plot_profile_pattern_mfe_and_signal.py --task count --sign pos --pattern-index 3
      python plot_profile_pattern_mfe_and_signal.py --task count --sign neg --pattern-index 5

For every seqlet of the chosen pattern:

1. Bridge seqlet -> PauseNet test example via example_idx and
   <task>_selected_dataset_indices.npy.
2. Recover the full 2114-bp transcription-oriented input sequence from
   sequence_codes.npy, take +/-515 bp around the seqlet centre, fold all
   overlapping 30-nt windows with ViennaRNA DNA Mathews 2004 parameters, and
   assign each window MFE to the window centre (base 15) -> 1000 values at
   relative positions -500..+499.
3. Fetch observed/predicted 1000-bp profiles from test_profiles.npz;
   observed_profiles are already per-base counts (row sums == observed_counts),
   while predicted_profiles are probabilities (row sums == 1) and are
   multiplied by predicted_counts (already on count scale) to recover counts.
4. Average the three tracks across seqlets and plot them on one figure with
   dual y axes (G4-style).

Outputs (this folder):
  profile_pattern_<N>_mfe_and_signal.pdf / .png        (task=profile)
  count_<pos|neg>_pattern_<N>_mfe_and_signal.pdf/.png  (task=count)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from functools import lru_cache
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import RNA

HERE = Path(__file__).resolve().parent
RES = Path("/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/TFMoDISco/results/hek293t_netseq")
DEFAULT_CODES = Path("/mnt/HDD8TB/houruiyan/pausing_site/data/HEK293T_NETseq/dataset/test/sequence_codes.npy")
DEFAULT_PROFILES = Path("/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/hek293t_netseq/test_profiles.npz")

INPUT_LEN = 2114
CROP_LEN = 1000
CROP_START = (INPUT_LEN - CROP_LEN) // 2          # 557; modisco ran on this crop == output window
HALF = 500                                        # x axis -500..+499

CODE_TO_BASE = np.asarray(list("ACGT"))

WORKER_WINDOW_SIZE = 30
WORKER_MD: "RNA.md | None" = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["profile", "count"], default="profile",
                   help="which TF-MoDISco task the pattern comes from "
                        "(default: profile)")
    p.add_argument("--sign", choices=["pos", "neg"], default="pos",
                   help="pos_patterns or neg_patterns; only used for --task count "
                        "(profile only has pos_patterns)")
    p.add_argument("--patterns-h5", type=Path, default=None,
                   help="override patterns h5 (default: RES/<task>/<task>_tfmodisco_patterns.h5)")
    p.add_argument("--bridge", type=Path, default=None,
                   help="override bridge npy (default: RES/<task>/<task>_selected_dataset_indices.npy)")
    p.add_argument("--sequence-codes", type=Path, default=DEFAULT_CODES)
    p.add_argument("--test-profiles", type=Path, default=DEFAULT_PROFILES)
    p.add_argument("--pattern-index", type=int, required=True,
                   help="TF-MoDISco pattern index, e.g. --pattern-index 37")
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--temperature", type=float, default=37.0)
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--max-seqlets", type=int, default=None, help="testing option")
    p.add_argument("--normalize-signal", action="store_true",
                   help="scale each signal track to its own max (=1) for shape comparison")
    p.add_argument("--output-prefix", type=Path, default=None)
    return p.parse_args()


def initialize_fold_worker(window_size: int, temperature: float) -> None:
    global WORKER_WINDOW_SIZE, WORKER_MD
    WORKER_WINDOW_SIZE = window_size
    RNA.params_load_DNA_Mathews2004()
    model = RNA.md()
    model.temperature = temperature
    model.dangles = 2
    WORKER_MD = model


@lru_cache(maxsize=500_000)
def dna_window_mfe(sequence: str) -> float:
    if WORKER_MD is None:
        raise RuntimeError("DNA folding worker was not initialized")
    fc = RNA.fold_compound(sequence, WORKER_MD)
    _, mfe = fc.mfe()
    return float(mfe)


def mfe_track(sequence: str) -> np.ndarray:
    """sequence covers centre-(HALF+w//2) .. centre+(HALF+w//2); return 2*HALF MFE
    values assigned to window centres (relative -HALF..HALF-1)."""
    w = WORKER_WINDOW_SIZE
    n_windows = len(sequence) - w + 1
    out = np.full(2 * HALF, np.nan, dtype=np.float64)
    for i in range(n_windows):
        win = sequence[i:i + w]
        if "N" in win:
            continue
        out[i] = dna_window_mfe(win)
    return out


def fold_region(job: tuple) -> tuple:
    seqlet_index, sequence = job
    return seqlet_index, mfe_track(sequence)


def codes_to_sequence(codes: np.ndarray) -> str:
    codes = np.asarray(codes)
    bases = np.full(codes.shape, "N", dtype="<U1")
    valid = codes < 4
    bases[valid] = CODE_TO_BASE[codes[valid].astype(np.intp)]
    return "".join(bases.tolist())


def draw_figure(x: np.ndarray, mfe: np.ndarray, obs: np.ndarray, pred: np.ndarray,
                n_seqlets: int, title: str,
                pdf_path: Path, png_path: Path, sig_ylabel: str = "Mean pausing signal") -> None:
    # final-figure-small style, canvas 10x4 inch
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 12,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
    })
    C_MFE, C_OBS, C_PRED = "#008837", "#B2182B", "#2C7FB8"

    fig, ax_mfe = plt.subplots(figsize=(10, 4))
    ax_sig = ax_mfe.twinx()

    # line widths / zorder follow plot_profile_pattern_pausing_signal.py
    l_obs, = ax_sig.plot(x, obs, color=C_OBS, lw=1.05,
                         label="Observed Net-seq signal", zorder=3)
    l_pred, = ax_sig.plot(x, pred, color=C_PRED, lw=1.35,
                          label="PauseNet predicted signal", zorder=2)
    l_mfe, = ax_mfe.plot(x, mfe, color=C_MFE, lw=1.35,
                         label="ViennaRNA DNA MFE", zorder=2)

    ax_mfe.axvline(0, color="grey", lw=0.9, ls="--", zorder=1)
    ax_mfe.set_xlim(-HALF, HALF)
    ax_mfe.set_xlabel("Position relative to motif center (bp)")
    ax_mfe.set_ylabel("Mean DNA MFE (kcal/mol)", color="black")
    ax_sig.set_ylabel(sig_ylabel, color="#333333")
    ax_mfe.tick_params(axis="y", colors="black")
    ax_mfe.spines["top"].set_visible(False)
    ax_sig.spines["top"].set_visible(False)

    ax_mfe.text(0.02, 0.97, f"n = {n_seqlets} seqlets", transform=ax_mfe.transAxes,
                ha="left", va="top", fontsize=12, color="#333333")
    ax_mfe.legend(
        handles=[l_mfe, l_obs, l_pred],
        fontsize=12, frameon=False, loc="upper right",
    )
    ax_mfe.set_title(title, fontsize=12, pad=7)

    fig.tight_layout()
    # final-figure-small: keep the exact 10x4 canvas, no bbox_inches="tight"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    # resolve task-dependent inputs: <task>/<task>_tfmodisco_patterns.h5 + bridge
    task_dir = RES / args.task
    patterns_h5 = args.patterns_h5 or task_dir / f"{args.task}_tfmodisco_patterns.h5"
    bridge_path = args.bridge or task_dir / f"{args.task}_selected_dataset_indices.npy"
    group_name = "pos_patterns" if args.sign == "pos" else "neg_patterns"
    if args.task == "profile":
        label = f"profile pattern_{args.pattern_index}"
        default_prefix = HERE / f"profile_pattern_{args.pattern_index}_mfe_and_signal"
    else:
        label = f"count {args.sign} pattern_{args.pattern_index}"
        default_prefix = HERE / f"count_{args.sign}_pattern_{args.pattern_index}_mfe_and_signal"
    title = f"{label}: DNA MFE and pausing signal"

    output_prefix = args.output_prefix or default_prefix
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    bridge = np.load(bridge_path)
    codes = np.load(args.sequence_codes, mmap_mode="r")
    print(f"task={args.task} group={group_name}; bridge rows={bridge.shape[0]}, "
          f"sequence_codes shape={codes.shape}", flush=True)
    assert codes.shape[1] == INPUT_LEN, f"unexpected input length {codes.shape}"

    with h5py.File(patterns_h5, "r") as h:
        if group_name not in h:
            raise SystemExit(
                f"ERROR: {patterns_h5} has no {group_name} group "
                f"(available: {list(h.keys())})")
        pos = h[group_name]
        key = f"pattern_{args.pattern_index}"
        if key not in pos:
            available = sorted(int(k.split("_")[1]) for k in pos.keys()
                               if k.startswith("pattern_"))
            raise SystemExit(
                f"ERROR: {group_name}/{key} not found in {patterns_h5}\n"
                f"available pattern indices in {group_name}: {available}")
        g = pos[key]["seqlets"]
        example_idx = np.asarray(g["example_idx"], dtype=np.int64)
        starts = np.asarray(g["start"], dtype=np.int64)
        ends = np.asarray(g["end"], dtype=np.int64)
        is_rc = np.asarray(g["is_revcomp"], dtype=bool)
    n_all = len(example_idx)
    n_use = min(n_all, args.max_seqlets) if args.max_seqlets else n_all
    print(f"{label}: using {n_use}/{n_all} seqlets", flush=True)

    dataset_idx = bridge[example_idx[:n_use]]
    centers_crop = ((starts[:n_use] + ends[:n_use]) // 2).astype(np.int64)  # in 1000-bp crop == output coords
    centers_input = CROP_START + centers_crop
    print(f"seqlet centre (crop coords): min={centers_crop.min()} max={centers_crop.max()}", flush=True)

    # ---------- MFE tracks ----------
    flank = HALF + args.window_size // 2
    region_len = 2 * HALF + args.window_size - 1   # exactly 2*HALF windows
    jobs = []
    for i in range(n_use):
        c = int(centers_input[i])
        region = np.asarray(codes[dataset_idx[i], c - flank:c - flank + region_len])
        jobs.append((i, codes_to_sequence(region)))
    if args.jobs == 1:
        initialize_fold_worker(args.window_size, args.temperature)
        folded = [fold_region(j) for j in jobs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.jobs, initializer=initialize_fold_worker,
                      initargs=(args.window_size, args.temperature)) as pool:
            folded = list(pool.imap(fold_region, jobs, chunksize=1))
    mfe_tracks = np.full((n_use, 2 * HALF), np.nan)
    for i, tr in folded:
        mfe_tracks[i] = tr
    mean_mfe = np.nanmean(mfe_tracks, axis=0)
    print(f"MFE done: per-seqlet mean range "
          f"[{np.nanmin(np.nanmean(mfe_tracks, axis=1)):.2f}, "
          f"{np.nanmax(np.nanmean(mfe_tracks, axis=1)):.2f}] kcal/mol; "
          f"track min={np.nanmin(mean_mfe):.2f} at {np.nanargmin(mean_mfe) - HALF} bp, "
          f"max={np.nanmax(mean_mfe):.2f}", flush=True)

    # ---------- signal tracks ----------
    z = np.load(args.test_profiles)
    obs_prof = z["observed_profiles"][dataset_idx]          # (n,1000) per-base counts
    pred_prof = z["predicted_profiles"][dataset_idx]
    obs_counts = z["observed_counts"][dataset_idx].astype(np.float64)
    pred_counts = z["predicted_counts"][dataset_idx].astype(np.float64)
    print(f"obs_profiles row-sum ~ {obs_prof[0].sum():.3f}; "
          f"obs_counts range [{obs_counts.min():.1f}, {obs_counts.max():.1f}]; "
          f"pred_counts range [{pred_counts.min():.2f}, {pred_counts.max():.2f}] "
          f"(already count scale)", flush=True)
    # observed_profiles are already per-base counts (verified: row sums equal
    # observed_counts exactly), so multiplying by obs_counts again would
    # quadratically over-weight high-count examples in the cross-seqlet mean.
    obs_sig = obs_prof
    pred_sig = pred_prof * pred_counts[:, None]

    rel = np.arange(-HALF, HALF)
    obs_tracks = np.full((n_use, 2 * HALF), np.nan)
    pred_tracks = np.full((n_use, 2 * HALF), np.nan)
    for i in range(n_use):
        c = int(centers_crop[i])
        # profile position p maps to relative p - c; keep those inside [-HALF, HALF)
        p_lo = max(0, c - HALF)
        p_hi = min(CROP_LEN, c + HALF)
        obs_tracks[i, (p_lo - c) + HALF:(p_hi - c) + HALF] = obs_sig[i, p_lo:p_hi]
        pred_tracks[i, (p_lo - c) + HALF:(p_hi - c) + HALF] = pred_sig[i, p_lo:p_hi]
    mean_obs = np.nanmean(obs_tracks, axis=0)
    mean_pred = np.nanmean(pred_tracks, axis=0)
    finite_obs = np.isfinite(mean_obs)
    print(f"signal done: observed mean range [{np.nanmin(mean_obs):.3f}, {np.nanmax(mean_obs):.3f}] "
          f"(peak at {np.nanargmax(mean_obs) - HALF} bp); "
          f"predicted peak at {np.nanargmax(mean_pred) - HALF} bp; "
          f"covered relative span [{rel[finite_obs][0]}, {rel[finite_obs][-1]}]", flush=True)

    # ---------- plot ----------
    x = rel
    sig_ylabel = "Mean pausing signal"
    if args.normalize_signal:
        max_o, max_p = np.nanmax(mean_obs), np.nanmax(mean_pred)
        mean_obs = mean_obs / max_o
        mean_pred = mean_pred / max_p
        sig_ylabel = "Mean pausing signal (normalized to max)"
        print(f"normalize-signal: divided by obs max {max_o:.3f}, pred max {max_p:.3f}", flush=True)
    draw_figure(x, mean_mfe, mean_obs, mean_pred, n_use, title,
                Path(f"{output_prefix}.pdf"), Path(f"{output_prefix}.png"),
                sig_ylabel=sig_ylabel)
    print(f"Wrote: {output_prefix}.pdf")
    print(f"Wrote: {output_prefix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
