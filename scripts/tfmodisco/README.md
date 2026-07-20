# TF-MoDISco motif discovery

Run TF-MoDISco (modisco-lite) on DeepSHAP contribution scores of the two
PauseNet output heads (`profile` and `count`) to discover pausing-related
sequence motifs.

## Contents

| File | Purpose |
|---|---|
| `run_tfmodisco_from_contribs.py` | Core driver. Loads precomputed full `L x 4` expected-gradient contribution `.npy` files plus `sequence_codes.npy`, selects samples, runs modisco-lite TFMoDISco, and writes `*_tfmodisco_patterns.h5`, `*_PFM.meme`, `*_CWM.meme`, `*_pattern_summary.tsv`, `*_metadata.json`. |
| `run_tfmodisco_kimi_all_window.sh` | Full run over all 57,590 test windows for both tasks, serially (used for the final HEK293T all-window results; `n_leiden_runs=20`, `nearest_neighbors_to_compute=500`). |
| `run_tfmodisco_full_count_profile.sh` | Earlier full-data run with lighter clustering settings. |
| `run_tfmodisco_top5000_count_profile.sh` | Run on the top 5,000 windows by mean |contribution|. |
| `run_tfmodisco_final_top1000_count_profile.sh` | Run on the top 1,000 windows (fast iteration). |
| `run_tfmodisco_smoke_count.sh` | Small smoke test of the pipeline. |
| `install_tfmodisco_py310.sh` | Create the Python 3.10 conda env with modisco-lite and dependencies. |

## Inputs

The driver expects:

- contribution arrays produced by DeepSHAP on the test split
  (`<task>_contribs_full_Lx4_test_start0_n<N>_bg4_steps8.npy`), passed via
  `--contrib-dir`;
- `sequence_codes.npy` for the same windows (integer-encoded DNA, `0..3`),
  found under the dataset directory passed via `--data-dir`;
- `--task` selects `profile` or `count`.

## Example

```bash
PY=/path/to/env/bin/python
"$PY" run_tfmodisco_from_contribs.py \
    --task count \
    --output-dir results/count_all_test \
    --sample-selection all \
    --sliding-window-size 21 --flank-size 10 \
    --target-seqlet-fdr 0.2 \
    --min-passing-windows-frac 0.03 --max-passing-windows-frac 0.2 \
    --max-seqlets-per-metacluster 20000 --min-metacluster-size 100 \
    --n-leiden-runs 20 --nearest-neighbors-to-compute 500 \
    --final-min-cluster-size 20 --verbose
```

Key clustering parameters: `n_leiden_runs` and `nearest_neighbors_to_compute`
control the consensus Leiden clustering granularity (higher = fewer, more
robust pattern clusters, at roughly 3x runtime). Shell scripts hard-code
absolute paths from the original analysis machine; adapt `PY`, `SRC`, and
`BASE` before rerunning.

## Outputs (per task, per run)

- `<task>_tfmodisco_patterns.h5` — full modisco results (seqlets, patterns);
- `<task>_PFM.meme` / `<task>_CWM.meme` (+ `_PFM` / `hCWM` variants) — motif
  matrices in MEME format for Tomtom etc.;
- `<task>_pattern_summary.tsv` — per-pattern seqlet counts and stats;
- `<task>_metadata.json` — full parameter record, pattern counts, elapsed time;
- `selected_indices.npy` — window indices used in the run.

Final HEK293T all-window results (57,590 test windows): count 53 pos + 38 neg
patterns, profile 42 pos + 36 neg patterns.
