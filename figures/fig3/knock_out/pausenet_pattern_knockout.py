#!/usr/bin/env python3
"""PauseNet in-silico knock-out of ALL seqlets of one TF-MoDISco pattern.

For every seqlet of pattern N (from the TF-MoDISco h5 of the chosen branch:
--task profile -> profile_tfmodisco_patterns.h5, --task count ->
count_tfmodisco_patterns.h5), the
seqlet window in the corresponding 2114 bp input sequence is shuffled without
replacement (base composition / GC content exactly preserved), sampling
--n-shuffles permutations and keeping the mutant with the HIGHEST MFE (least
stable secondary structure; "destroy structure" = MFE goes UP / less
negative) under ViennaRNA DNA Mathews2004 parameters at 37 C, dangles=2.

PauseNet predictions (softmax(profile_logits) * expm1(log1p_count), count
scale -- verified against pausenet/predict.py and the evaluate scripts) for
WT and mutant sequences are aligned on each seqlet's crop center and averaged
across seqlets over relative positions -500..+499, then plotted (WT blue,
mutant orange).

Default = pattern-level mode. Pass --seqlet-index X to run only seqlet row X
and produce the single-seqlet figure (with observed NET-seq in red).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PAUSENET_REPO = Path("/mnt/HDD8TB/houruiyan/pausing_site/PauseNet")
sys.path.insert(0, str(PAUSENET_REPO))
from pausenet.model import PauseNet, PauseNetConfig  # noqa: E402

INPUT_LEN = 2114
CROP_LEN = 1000
CROP_START = (INPUT_LEN - CROP_LEN) // 2  # 557
HALF = 500
CODE_TO_BASE = np.array(list("ACGT"))

MODISCO_RES = (
    "/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/TFMoDISco/results/"
    "hek293t_netseq"
)
# per-task inputs: --task switches between profile/count TF-MoDISco branches
TASK_INPUTS = {
    "profile": {
        "h5": f"{MODISCO_RES}/profile/profile_tfmodisco_patterns.h5",
        "bridge": f"{MODISCO_RES}/profile/profile_selected_dataset_indices.npy",
    },
    "count": {
        "h5": f"{MODISCO_RES}/count/count_tfmodisco_patterns.h5",
        "bridge": f"{MODISCO_RES}/count/count_selected_dataset_indices.npy",
    },
}
MODEL_PATH = (
    "/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/train/hek293t_netseq/best_model.pt"
)
TEST_DIR = Path("/mnt/HDD8TB/houruiyan/pausing_site/data/HEK293T_NETseq/dataset/test")

COLOR_OBS = "#B2182B"
COLOR_WT = "#555555"        # same as pausenet_pattern_g4_knockout.py
COLOR_MUT = "#0072B2"       # same as pausenet_pattern_g4_knockout.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern-index", type=int, default=8)
    p.add_argument("--task", choices=["profile", "count"], default="profile",
                   help="Which TF-MoDISco branch to read seqlets from.")
    p.add_argument("--h5", type=Path, default=None,
                   help="Override patterns h5 (default: from --task).")
    p.add_argument("--bridge", type=Path, default=None,
                   help="Override bridge npy (default: from --task).")
    p.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    p.add_argument("--n-shuffles", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seqlets", type=int, default=None)
    p.add_argument("--seqlet-index", type=int, default=None,
                   help="Single-seqlet mode: run only this h5 row.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=16,
                   help="Processes for the ViennaRNA shuffle scan.")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return p.parse_args()


def configure_style() -> None:
    # same style as pausenet_pattern_g4_knockout.py
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.titlesize": 12,
        }
    )


def codes_to_str(codes: np.ndarray) -> str:
    return "".join(CODE_TO_BASE[np.asarray(codes, dtype=int)])


# ---------------- mutation scan (multiprocessing workers) -----------------

def _fold_mfe(sequence: str):
    import RNA

    md = RNA.md()
    md.temperature = 37.0
    md.dangles = 2
    fc = RNA.fold_compound(sequence, md)
    _structure, mfe = fc.mfe()
    return float(mfe)


def _scan_one_seqlet(task):
    """task = (row_index, wt_seqlet_seq, n_shuffles, child_seed)"""
    import RNA

    RNA.params_load_DNA_Mathews2004()
    row_index, wt_seq, n_shuffles, child_seed = task
    wt_codes = np.array([CODE_TO_BASE.tolist().index(b) for b in wt_seq], dtype=np.int64)
    wt_mfe = _fold_mfe(wt_seq)
    rng = np.random.default_rng(child_seed)
    best_codes = None
    best_mfe = -np.inf
    for _ in range(n_shuffles):
        shuffled = rng.permutation(wt_codes)
        if np.array_equal(shuffled, wt_codes):
            continue
        mfe = _fold_mfe(codes_to_str(shuffled))
        if mfe > best_mfe:
            best_mfe = mfe
            best_codes = shuffled
    failed = best_codes is None or best_mfe <= wt_mfe
    return (
        row_index,
        wt_mfe,
        best_mfe,
        best_codes if not failed else None,
        failed,
    )


# ---------------- model prediction ----------------------------------------

def load_model(model_path: Path, device: torch.device) -> PauseNet:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", {}).get("model", {})
    model = PauseNet(PauseNetConfig(**model_cfg)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_tracks(model: PauseNet, codes: np.ndarray, device: torch.device,
                   batch_size: int, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """codes: (n, 2114) int -> (tracks (n,1000) count scale, counts (n,))"""
    n = len(codes)
    tracks = np.zeros((n, CROP_LEN), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            batch = torch.from_numpy(codes[s:e]).long().to(device)
            logits, log1p_counts = model(batch)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            c = np.expm1(log1p_counts.float().cpu().numpy())
            tracks[s:e] = probs * c[:, None]
            counts[s:e] = c
            if (s // batch_size) % 4 == 0 or e == n:
                print(f"  [{tag}] {e}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return tracks, counts


# ---------------- alignment -----------------------------------------------

def align_tracks(tracks: np.ndarray, centers_crop: np.ndarray) -> np.ndarray:
    """Return (n, 2*HALF) array aligned so column j = rel position j-HALF.

    Profile position p (crop coords) maps to rel = p - center; positions
    outside [0, 1000) become NaN.
    """
    n = tracks.shape[0]
    out = np.full((n, 2 * HALF), np.nan, dtype=np.float64)
    for i in range(n):
        c = int(centers_crop[i])
        p_lo = max(0, c - HALF)
        p_hi = min(CROP_LEN, c + HALF)
        if p_hi > p_lo:
            out[i, p_lo - c + HALF : p_hi - c + HALF] = tracks[i, p_lo:p_hi]
    return out


# ---------------- main -----------------------------------------------------

def main() -> None:
    args = parse_args()
    configure_style()
    t_start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pattern_name = f"pattern_{args.pattern_index}"
    # resolve task-dependent inputs (profile/count TF-MoDISco branches)
    h5_path = args.h5 or Path(TASK_INPUTS[args.task]["h5"])
    bridge_path = args.bridge or Path(TASK_INPUTS[args.task]["bridge"])
    out_name = f"{args.task}_{pattern_name}"   # keeps profile/count outputs separate
    print(f"[input] task={args.task} h5={h5_path} bridge={bridge_path}", flush=True)

    # ---- load seqlets from h5 --------------------------------------------
    import h5py

    with h5py.File(h5_path, "r") as f:
        g = f[f"pos_patterns/{pattern_name}/seqlets"]
        example_idx = np.asarray(g["example_idx"], dtype=np.int64)
        starts = np.asarray(g["start"], dtype=np.int64)
        ends = np.asarray(g["end"], dtype=np.int64)
        revcomps = np.asarray(g["is_revcomp"]).astype(bool)
    n_all = len(example_idx)
    row_ids = np.arange(n_all)
    if args.seqlet_index is not None:
        sel = row_ids == args.seqlet_index
        if not sel.any():
            raise SystemExit(f"seqlet row {args.seqlet_index} not in {pattern_name}")
    else:
        sel = np.ones(n_all, dtype=bool)
        if args.max_seqlets:
            sel[args.max_seqlets:] = False
    rows = row_ids[sel]
    example_idx, starts, ends, revcomps = (
        example_idx[sel], starts[sel], ends[sel], revcomps[sel]
    )
    n_use = len(rows)
    print(f"[seqlets] {pattern_name}: using {n_use}/{n_all} seqlets "
          f"(mode={'single' if args.seqlet_index is not None else 'pattern'})",
          flush=True)

    # ---- bridge to dataset indices, fetch sequences ----------------------
    bridge = np.load(bridge_path)
    dataset_idx = bridge[example_idx]
    sequence_codes = np.load(TEST_DIR / "sequence_codes.npy", mmap_mode="r")
    wt_codes_all = np.stack(
        [np.array(sequence_codes[d], dtype=np.int64) for d in dataset_idx]
    )
    in_starts = CROP_START + starts
    in_ends = CROP_START + ends
    centers_crop = ((starts + ends) // 2).astype(np.int64)

    # ---- mutation scan ----------------------------------------------------
    seeds = np.random.SeedSequence(args.seed).spawn(n_use)
    tasks = [
        (int(rows[i]), codes_to_str(wt_codes_all[i][in_starts[i]:in_ends[i]]),
         args.n_shuffles, seeds[i])
        for i in range(n_use)
    ]
    t0 = time.time()
    if args.workers > 1 and n_use > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            results = pool.map(_scan_one_seqlet, tasks, chunksize=8)
    else:
        results = [_scan_one_seqlet(t) for t in tasks]
    print(f"[mfe] shuffle scan done: {n_use} seqlets x {args.n_shuffles} shuffles "
          f"in {time.time()-t0:.0f}s", flush=True)
    results.sort(key=lambda r: r[0])
    row_to_slot = {int(r): i for i, r in enumerate(rows)}
    wt_mfe = np.zeros(n_use)
    mut_mfe = np.zeros(n_use)
    failed = np.zeros(n_use, dtype=bool)
    mut_codes_all = wt_codes_all.copy()
    for row_index, w_mfe, m_mfe, best_codes, fail in results:
        i = row_to_slot[row_index]
        wt_mfe[i] = w_mfe
        mut_mfe[i] = m_mfe
        failed[i] = fail
        if not fail:
            mut_codes_all[i][in_starts[i]:in_ends[i]] = best_codes
    n_failed = int(failed.sum())
    print(f"[mfe] mean WT MFE = {wt_mfe.mean():.2f}; mean mutant MFE = "
          f"{mut_mfe.mean():.2f}; mean delta = {(mut_mfe-wt_mfe).mean():+.2f} "
          f"kcal/mol; failed (no destabilizing shuffle found): {n_failed}",
          flush=True)
    if n_failed == n_use:
        raise SystemExit("ERROR: every seqlet failed to find a shuffle with "
                         "higher MFE; increase --n-shuffles.")

    # ---- prediction -------------------------------------------------------
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"[device] WARNING: {args.device} unavailable; falling back to CPU",
              flush=True)
        requested = torch.device("cpu")
    device = requested
    torch.set_num_threads(max(1, mp.cpu_count()))
    print(f"[device] using {device} ({mp.cpu_count()} CPU threads)", flush=True)
    model = load_model(args.model, device)
    t0 = time.time()
    wt_tracks, wt_counts = predict_tracks(model, wt_codes_all, device,
                                          args.batch_size, "WT")
    mut_tracks, mut_counts = predict_tracks(model, mut_codes_all, device,
                                            args.batch_size, "MUT")
    print(f"[pred] inference done in {time.time()-t0:.0f}s", flush=True)

    # ---- per-seqlet stats --------------------------------------------------
    seq_len = (ends - starts).astype(int)
    win_sum_wt = np.array([wt_tracks[i][starts[i]:ends[i]].sum() for i in range(n_use)])
    win_sum_mut = np.array([mut_tracks[i][starts[i]:ends[i]].sum() for i in range(n_use)])
    win_max_wt = np.array([wt_tracks[i][starts[i]:ends[i]].max() for i in range(n_use)])
    win_max_mut = np.array([mut_tracks[i][starts[i]:ends[i]].max() for i in range(n_use)])
    peak_rel_wt = wt_tracks.argmax(axis=1) - centers_crop
    peak_rel_mut = mut_tracks.argmax(axis=1) - centers_crop

    summary = pd.DataFrame(
        {
            "seqlet_index": rows,
            "example_idx": example_idx,
            "dataset_idx": dataset_idx,
            "seqlet_start_crop": starts,
            "seqlet_end_crop": ends,
            "is_revcomp": revcomps,
            "mfe_wt": wt_mfe,
            "mfe_mut": mut_mfe,
            "delta_mfe": mut_mfe - wt_mfe,
            "mfe_scan_failed": failed,
            "pred_counts_wt": wt_counts,
            "pred_counts_mut": mut_counts,
            "counts_change_pct": (mut_counts / np.maximum(wt_counts, 1e-9) - 1) * 100,
            "window_sum_wt": win_sum_wt,
            "window_sum_mut": win_sum_mut,
            "window_sum_change_pct": (win_sum_mut / np.maximum(win_sum_wt, 1e-9) - 1) * 100,
            "window_max_wt": win_max_wt,
            "window_max_mut": win_max_mut,
            "peak_pos_rel_center_wt": peak_rel_wt,
            "peak_pos_rel_center_mut": peak_rel_mut,
        }
    )
    prefix = args.output_dir / f"{out_name}_knockout"
    summary_path = Path(f"{prefix}_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"[out] {summary_path}", flush=True)

    ok = ~failed
    print(f"[summary] n={n_use} (paired n={int(ok.sum())}, failed={n_failed})")
    print(f"[summary] mean delta MFE = {(mut_mfe[ok]-wt_mfe[ok]).mean():+.2f} kcal/mol")
    print(f"[summary] pred counts: mean WT {wt_counts[ok].mean():.1f} -> mut "
          f"{mut_counts[ok].mean():.1f} "
          f"({(mut_counts[ok].mean()/wt_counts[ok].mean()-1)*100:+.1f}%)")
    print(f"[summary] seqlet-window signal sum: mean WT {win_sum_wt[ok].mean():.3f} -> "
          f"mut {win_sum_mut[ok].mean():.3f} "
          f"({(win_sum_mut[ok].mean()/max(win_sum_wt[ok].mean(),1e-9)-1)*100:+.1f}%)")
    print(f"[summary] seqlet-window max: mean WT {win_max_wt[ok].mean():.3f} -> "
          f"mut {win_max_mut[ok].mean():.3f} "
          f"({(win_max_mut[ok].mean()/max(win_max_wt[ok].mean(),1e-9)-1)*100:+.1f}%)")
    print(f"[summary] peak pos rel center: WT mean {peak_rel_wt[ok].mean():+.1f} bp, "
          f"mut mean {peak_rel_mut[ok].mean():+.1f} bp")

    # ---- plots -------------------------------------------------------------
    if args.seqlet_index is not None:
        # single-seqlet figure (with observed track)
        i = 0
        observed = np.array(
            np.load(TEST_DIR / "profiles.npy", mmap_mode="r")[dataset_idx[i]],
            dtype=float,
        )
        x = np.arange(CROP_LEN) - centers_crop[i]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x, observed, color=COLOR_OBS, lw=0.9, label="Observed NET-seq")
        ax.plot(x, wt_tracks[i], color=COLOR_WT, lw=0.9, label="Original")
        ax.plot(x, mut_tracks[i], color=COLOR_MUT, lw=0.9, label="Structure-disrupted")
        for xv in (starts[i] - centers_crop[i], ends[i] - centers_crop[i]):
            ax.axvline(xv, color="0.35", lw=0.8, ls="--")
        ax.axvspan(starts[i] - centers_crop[i], ends[i] - centers_crop[i],
                   color="0.35", alpha=0.08, lw=0)
        ax.set_xlabel("Position relative to seqlet center (bp)")
        ax.set_ylabel("NET-seq signal (reads)")
        ax.set_xlim(-HALF, HALF)
        ax.set_xticks([-500, -250, 0, 250, 500])
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc="upper right")
        ax.set_title(
            f"{pattern_name} seqlet #{int(rows[i])} knock-out ({args.task})\n"
            f"MFE {wt_mfe[i]:.1f} -> {mut_mfe[i]:.1f} kcal/mol · "
            f"pred counts {wt_counts[i]:.0f} -> {mut_counts[i]:.0f}",
            pad=6,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        fig.tight_layout()
        single_prefix = args.output_dir / f"{out_name}_seqlet{int(rows[i])}_knockout_profile"
        fig.savefig(str(single_prefix) + ".pdf", bbox_inches="tight")
        fig.savefig(str(single_prefix) + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[out] {single_prefix}.pdf / .png", flush=True)
    else:
        # pattern-level mean figure, aligned on seqlet centers
        rel = np.arange(-HALF, HALF)
        aligned_wt = align_tracks(wt_tracks[ok], centers_crop[ok])
        aligned_mut = align_tracks(mut_tracks[ok], centers_crop[ok])
        mean_wt = np.nanmean(aligned_wt, axis=0)
        mean_mut = np.nanmean(aligned_mut, axis=0)
        n_cols = np.sum(np.isfinite(aligned_wt), axis=0)
        with np.errstate(invalid="ignore"):
            sem_wt = np.nanstd(aligned_wt, axis=0) / np.sqrt(np.maximum(n_cols, 1))
            sem_mut = np.nanstd(aligned_mut, axis=0) / np.sqrt(np.maximum(n_cols, 1))
        mean_dmfe = float((mut_mfe[ok] - wt_mfe[ok]).mean())
        median_dmfe = float(np.median(mut_mfe[ok] - wt_mfe[ok]))
        peak_wt = float(mean_wt.max())
        peak_mut = float(mean_mut.max())
        peak_rel = int(rel[int(np.argmax(mean_wt))])
        print(f"[mean-curve] WT peak {peak_wt:.3f} at rel {peak_rel:+d} bp; "
              f"mutant peak {peak_mut:.3f} "
              f"({(peak_mut/peak_wt-1)*100:+.1f}% vs WT peak)")

        # style follows pausenet_pattern_g4_knockout.py
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.fill_between(rel, np.maximum(0, mean_wt - sem_wt), mean_wt + sem_wt,
                        color=COLOR_WT, alpha=0.16, linewidth=0)
        ax.fill_between(rel, np.maximum(0, mean_mut - sem_mut), mean_mut + sem_mut,
                        color=COLOR_MUT, alpha=0.16, linewidth=0)
        ax.plot(rel, mean_wt, color=COLOR_WT, lw=1.4, label="Original")
        ax.plot(rel, mean_mut, color=COLOR_MUT, lw=1.4,
                label="Structure-disrupted")
        ax.axvline(0, color="#777777", lw=0.8, linestyle=(0, (3, 2)), zorder=1)
        ax.set_xlim(-HALF, HALF)
        ax.set_xticks([-500, -250, 0, 250, 500])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Position relative to seqlet center (bp)")
        ax.set_ylabel("Mean predicted pausing signal")
        ax.set_title(f"{pattern_name} ({args.task}) in-silico knock-out", pad=6)
        ax.text(0.03, 0.96,
                f"n = {int(ok.sum()):,}\nmedian ΔMFE = {median_dmfe:+.2f} kcal/mol",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color="#222222")
        ax.legend(frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        fig.tight_layout()
        mean_prefix = Path(f"{prefix}_mean_profile")
        fig.savefig(str(mean_prefix) + ".pdf", bbox_inches="tight")
        fig.savefig(str(mean_prefix) + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[out] {mean_prefix}.pdf / .png", flush=True)

    print(f"[done] total elapsed {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
