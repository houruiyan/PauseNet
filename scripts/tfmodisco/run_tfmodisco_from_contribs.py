#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
from modiscolite import io, tfmodisco, util


def one_hot_from_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int16)
    n, l = codes.shape
    one_hot = np.zeros((n, l, 4), dtype=np.float32)
    rows, cols = np.nonzero(codes < 4)
    one_hot[rows, cols, codes[rows, cols]] = 1.0
    return one_hot


def select_indices(codes, contrib_projected, max_samples, mode, seed):
    n = codes.shape[0]
    if max_samples is None or max_samples >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if mode == 'first':
        return np.arange(max_samples, dtype=np.int64)
    if mode == 'random':
        return np.sort(rng.choice(n, size=max_samples, replace=False).astype(np.int64))
    if mode == 'top_abs_contrib':
        score = np.mean(np.abs(np.asarray(contrib_projected, dtype=np.float32)), axis=1)
        idx = np.argpartition(score, -max_samples)[-max_samples:]
        return np.sort(idx.astype(np.int64))
    raise ValueError(f'Unknown selection mode: {mode}')


def pattern_summary_rows(patterns, sign):
    rows = []
    if patterns is None:
        return rows
    for i, pat in enumerate(patterns):
        seq = np.asarray(pat.sequence)
        contrib = np.asarray(pat.contrib_scores)
        hyp = np.asarray(pat.hypothetical_contribs)
        ppm = seq / np.maximum(seq.sum(axis=1, keepdims=True), 1e-8)
        ic = 2.0 + np.sum(ppm * np.log2(np.maximum(ppm, 1e-8)), axis=1)
        rows.append({
            'sign': sign,
            'pattern_index': i,
            'n_seqlets': len(getattr(pat, 'seqlets', [])),
            'length': int(getattr(pat, 'length', seq.shape[0])),
            'mean_abs_contrib': float(np.mean(np.abs(contrib))),
            'sum_abs_contrib': float(np.sum(np.abs(contrib))),
            'sum_contrib': float(np.sum(contrib)),
            'mean_abs_hypothetical': float(np.mean(np.abs(hyp))),
            'max_ic': float(np.max(ic)) if ic.size else 0.0,
            'mean_ic': float(np.mean(ic)) if ic.size else 0.0,
        })
    return rows


def write_summary(path, rows):
    fields = ['sign', 'pattern_index', 'n_seqlets', 'length', 'mean_abs_contrib',
              'sum_abs_contrib', 'sum_contrib', 'mean_abs_hypothetical', 'max_ic', 'mean_ic']
    with open(path, 'w') as f:
        f.write('\t'.join(fields) + '\n')
        for row in rows:
            f.write('\t'.join(str(row.get(k, '')) for k in fields) + '\n')


def main():
    p = argparse.ArgumentParser(description='Run modisco-lite TF-MoDISco from HEK293T attribution arrays.')
    p.add_argument('--task', choices=['count', 'profile'], required=True)
    p.add_argument('--data-dir', default='/mnt/HDD8TB/houruiyan/pausing_site/data/NET_seq/HEK293T/model_data/gene_structure_anchored_windows')
    p.add_argument('--contrib-dir', default='/mnt/HDD8TB/houruiyan/pausing_site/data/NET_seq/HEK293T/final/deepshap/test_bg4_steps8')
    p.add_argument('--split', default='test')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--sample-selection', choices=['all', 'first', 'random', 'top_abs_contrib'], default='all')
    p.add_argument('--seed', type=int, default=2026062402)
    p.add_argument('--sliding-window-size', type=int, default=21)
    p.add_argument('--flank-size', type=int, default=10)
    p.add_argument('--target-seqlet-fdr', type=float, default=0.2)
    p.add_argument('--min-passing-windows-frac', type=float, default=0.03)
    p.add_argument('--max-passing-windows-frac', type=float, default=0.2)
    p.add_argument('--max-seqlets-per-metacluster', type=int, default=20000)
    p.add_argument('--min-metacluster-size', type=int, default=100)
    p.add_argument('--n-leiden-runs', type=int, default=20)
    p.add_argument('--nearest-neighbors-to-compute', type=int, default=500)
    p.add_argument('--final-min-cluster-size', type=int, default=20)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) / args.split
    contrib_dir = Path(args.contrib_dir)
    codes_path = data_dir / 'sequence_codes.npy'
    full_path = contrib_dir / f'{args.task}_contribs_full_Lx4_{args.split}_start0_n57590_bg4_steps8.npy'
    proj_path = contrib_dir / f'{args.task}_contribs_projected_{args.split}_start0_n57590_bg4_steps8.npy'

    codes_mm = np.load(codes_path, mmap_mode='r')
    full_mm = np.load(full_path, mmap_mode='r')
    proj_mm = np.load(proj_path, mmap_mode='r')
    if codes_mm.shape[:2] != full_mm.shape[:2]:
        raise ValueError(f'shape mismatch: codes {codes_mm.shape}, full {full_mm.shape}')

    if args.sample_selection == 'all':
        indices = np.arange(codes_mm.shape[0], dtype=np.int64)
        if args.max_samples is not None and args.max_samples < len(indices):
            indices = select_indices(codes_mm, proj_mm, args.max_samples, 'top_abs_contrib', args.seed)
            effective_selection = 'top_abs_contrib_due_to_max_samples'
        else:
            effective_selection = 'all'
    else:
        indices = select_indices(codes_mm, proj_mm, args.max_samples, args.sample_selection, args.seed)
        effective_selection = args.sample_selection

    np.save(out / 'selected_indices.npy', indices)
    print(json.dumps({'event': 'selected_samples', 'n': int(len(indices)), 'selection': effective_selection}), flush=True)

    codes = np.asarray(codes_mm[indices], dtype=np.uint8)
    one_hot = one_hot_from_codes(codes)
    hypothetical_contribs = np.asarray(full_mm[indices], dtype=np.float32)
    print(json.dumps({
        'event': 'loaded_arrays',
        'one_hot_shape': list(one_hot.shape),
        'hypothetical_shape': list(hypothetical_contribs.shape),
        'one_hot_gb': one_hot.nbytes / 1e9,
        'hypothetical_gb': hypothetical_contribs.nbytes / 1e9,
    }), flush=True)

    pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
        one_hot=one_hot,
        hypothetical_contribs=hypothetical_contribs,
        sliding_window_size=args.sliding_window_size,
        flank_size=args.flank_size,
        min_metacluster_size=args.min_metacluster_size,
        max_seqlets_per_metacluster=args.max_seqlets_per_metacluster,
        target_seqlet_fdr=args.target_seqlet_fdr,
        min_passing_windows_frac=args.min_passing_windows_frac,
        max_passing_windows_frac=args.max_passing_windows_frac,
        n_leiden_runs=args.n_leiden_runs,
        nearest_neighbors_to_compute=args.nearest_neighbors_to_compute,
        final_min_cluster_size=args.final_min_cluster_size,
        verbose=args.verbose,
    )

    h5_path = out / f'{args.task}_tfmodisco_patterns.h5'
    io.save_hdf5(h5_path, pos_patterns, neg_patterns, window_size=args.sliding_window_size)

    for dtype in [util.MemeDataType.PFM, util.MemeDataType.CWM, util.MemeDataType.hCWM, util.MemeDataType.CWM_PFM, util.MemeDataType.hCWM_PFM]:
        meme_path = out / f'{args.task}_{dtype.value.replace("-", "_")}.meme'
        io.write_meme_from_h5(h5_path, datatype=dtype, output_filename=meme_path, is_quiet=True)

    rows = pattern_summary_rows(pos_patterns, 'pos') + pattern_summary_rows(neg_patterns, 'neg')
    write_summary(out / f'{args.task}_pattern_summary.tsv', rows)

    metadata = vars(args).copy()
    metadata.update({
        'effective_selection': effective_selection,
        'n_selected_samples': int(len(indices)),
        'input_length': int(codes_mm.shape[1]),
        'full_contrib_path': str(full_path),
        'sequence_codes_path': str(codes_path),
        'h5_output': str(h5_path),
        'n_pos_patterns': 0 if pos_patterns is None else len(pos_patterns),
        'n_neg_patterns': 0 if neg_patterns is None else len(neg_patterns),
        'elapsed_sec': time.time() - t0,
        'method': 'modisco-lite TFMoDISco using full_Lx4 DeepSHAP-style expected-gradient contributions as hypothetical_contribs',
    })
    (out / f'{args.task}_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'event': 'finished', **metadata}, indent=2), flush=True)


if __name__ == '__main__':
    main()
