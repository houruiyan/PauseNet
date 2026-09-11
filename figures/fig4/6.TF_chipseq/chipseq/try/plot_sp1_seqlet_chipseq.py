#!/usr/bin/env python3
"""SP1 count/pos_pattern_8 seqlet-centered ChIP-seq, hg19.

Question: what is the supplied ChIP-seq signal around SP1-matched seqlets?
Outputs: two 5x4 inch editable PDFs, mean curve and raw-signal heatmap.
No smoothing, log transform, row normalization or missing-to-zero conversion.
Every seqlet has equal weight (including repeated genomic occurrences).
Default orientation is motif, combining gene strand and seqlet is_revcomp.
The midpoint base is floor((start+end-1)/2) in transcription coordinates;
for even lengths this consistently chooses the transcription-upstream base.
Dependencies: numpy, pandas, h5py, pyBigWig, matplotlib.
"""
from pathlib import Path
import argparse
import json
import warnings
import h5py
import numpy as np
import pandas as pd
import pyBigWig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/HDD8TB/houruiyan/pausing_site')
MODISCO = ROOT / '3_model_explanation/TFMoDISco/results/hek293t_netseq/count'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bigwig', type=Path, default=ROOT / 'data/TF/SP1/ENCFF478WJL_SP1_HEK293T_hg19.bigWig')
    p.add_argument('--h5', type=Path, default=MODISCO / 'count_tfmodisco_patterns.h5')
    p.add_argument('--bridge', type=Path, default=MODISCO / 'count_selected_dataset_indices.npy')
    p.add_argument('--manifest', type=Path, default=ROOT / '3_model_explanation/DeepSHAP/hek293t_netseq/selected_manifest.tsv')
    p.add_argument('--flank', type=int, default=1000)
    p.add_argument('--orientation', choices=['motif', 'transcription', 'genomic'], default='motif')
    p.add_argument('--outdir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--overwrite', action='store_true', help='Explicitly replace existing output PDFs')
    args = p.parse_args()
    if args.flank < 1:
        p.error('--flank must be positive')
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f'SP1_count_pos_pattern_8_chipseq_{args.orientation}_pm{args.flank}'
    outputs = [args.outdir / f'{stem}_{suffix}.pdf' for suffix in ['mean', 'heatmap']]
    if not args.overwrite and any(x.exists() for x in outputs):
        raise FileExistsError('Output exists; use --overwrite explicitly or select another --outdir')
    manifest = pd.read_csv(args.manifest, sep='\t').set_index('array_index', verify_integrity=True)
    bridge = np.load(args.bridge)
    if bridge.ndim != 1:
        raise ValueError('Expected one-dimensional dataset-index bridge')
    with h5py.File(args.h5, 'r') as f:
        group = f['pos_patterns/pattern_8/seqlets']
        seq = {k: group[k][:] for k in ['example_idx', 'start', 'end', 'is_revcomp']}
    n = len(seq['start'])
    if n == 0 or any(len(a) != n for a in seq.values()):
        raise ValueError('Invalid seqlet arrays')
    if np.any(seq['example_idx'] < 0) or np.any(seq['example_idx'] >= len(bridge)):
        raise ValueError('Seqlet example index outside bridge')
    x = np.arange(-args.flank, args.flank + 1)
    matrix = np.full((n, len(x)), np.nan, dtype=np.float64)
    loci, centers = [], []
    bw = pyBigWig.open(str(args.bigwig))
    try:
        chroms = bw.chroms()
        for j, (ex, start, end, rc) in enumerate(zip(*(seq[k] for k in ['example_idx', 'start', 'end', 'is_revcomp']))):
            row = manifest.loc[int(bridge[int(ex)])]
            start, end = int(start), int(end)
            if not 0 <= start < end <= 1000:
                raise ValueError(f'Invalid crop coordinates at seqlet {j}')
            if int(row.output_end) - int(row.output_start) != 1000 or int(row.input_end) - int(row.input_start) != 2114:
                raise ValueError('Unexpected dataset window size')
            if int(row.output_start) - int(row.input_start) != 557 or int(row.input_end) - int(row.output_end) != 557:
                raise ValueError('Manifest output region is not the central 1000 bp')
            if row.strand not in ['+', '-']:
                raise ValueError('Unknown transcription strand')
            chrom = str(row.chrom)
            if chrom not in chroms:
                raise ValueError(f'{chrom} missing from bigWig; check genome/chromosome naming')
            offset = (start + end - 1) // 2
            if row.strand == '+':
                center = int(row.output_start) + offset
                gstart, gend = int(row.output_start) + start, int(row.output_start) + end
            else:
                center = int(row.output_end) - 1 - offset
                gstart, gend = int(row.output_end) - end, int(row.output_end) - start
            minus = (row.strand == '-')
            flip = (minus != bool(rc)) if args.orientation == 'motif' else minus if args.orientation == 'transcription' else False
            lo, hi = center - args.flank, center + args.flank + 1
            a, b = max(0, lo), min(chroms[chrom], hi)
            if a < b:
                values = np.asarray(bw.values(chrom, a, b), dtype=float)
                values[~np.isfinite(values)] = np.nan
                matrix[j, a-lo:b-lo] = values
            if flip:
                matrix[j] = matrix[j, ::-1]
            loci.append((chrom, gstart, gend))
            centers.append((chrom, center))
    finally:
        bw.close()
    if not np.isfinite(matrix).any():
        raise ValueError('No finite bigWig signal found')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
        central_mean = np.nanmean(matrix[:, np.abs(x) <= min(100, args.flank)], axis=1)
    counts = np.isfinite(matrix).sum(axis=0)
    order = np.argsort(-np.nan_to_num(central_mean, nan=-np.inf), kind='stable')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.labelsize': 14, 'axes.titlesize': 14,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'legend.fontsize': 12, 'pdf.fonttype': 42})
    xlabel = 'Position relative to seqlet center (bp)'
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, mean, color='#4E7FAF', linewidth=1.1)
    ax.axvline(0, color='0.5', ls='--', lw=0.8)
    ax.set(xlabel=xlabel, ylabel='Mean ChIP-seq signal', xlim=(-args.flank, args.flank),
           title='SP1 ChIP-seq around seqlets')
    ax.text(0.03, 0.97, f'n = {n} seqlets\n{args.orientation} orientation', transform=ax.transAxes,
            va='top', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(outputs[0])
    plt.close(fig)
    finite = matrix[np.isfinite(matrix)]
    vmin, vmax = float(min(0, finite.min())), float(np.percentile(finite, 99))
    if vmax <= vmin:
        vmax = max(float(finite.max()), vmin + 1)
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#D0D0D0')
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix[order], aspect='auto', interpolation='nearest', cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=[x[0]-.5, x[-1]+.5, n+.5, .5], rasterized=True)
    ax.axvline(0, color='0.4', ls='--', lw=0.7)
    ax.set(xlabel=xlabel, ylabel='Seqlets (sorted by central signal)',
           title=f'SP1 ChIP-seq ({args.orientation})', yticks=[1, n])
    bar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.05, extend='max')
    bar.set_label('ChIP-seq signal', fontsize=12)
    fig.tight_layout()
    fig.savefig(outputs[1])
    plt.close(fig)
    print(json.dumps({'pattern': 'count/pos_pattern_8', 'n_seqlets': n,
        'unique_genomic_seqlet_intervals': len(set(loci)), 'unique_genomic_centers': len(set(centers)),
        'fully_missing_rows': int(np.sum(~np.isfinite(matrix).any(axis=1))),
        'missing_fraction': float(np.mean(~np.isfinite(matrix))),
        'valid_seqlets_per_position_min_max': [int(counts.min()), int(counts.max())],
        'mean_profile_min_max': [float(np.nanmin(mean)), float(np.nanmax(mean))],
        'heatmap_upper_color_limit_99percentile': vmax,
        'heatmap_sort': 'descending raw mean within +/-100 bp (or flank if smaller)',
        'missing_policy': 'NaN omitted from mean; gray on heatmap',
        'signal': 'original bigWig values; no smoothing or normalization',
        'bigwig': str(args.bigwig), 'pdfs': [str(o) for o in outputs]}, indent=2))


if __name__ == '__main__':
    main()
