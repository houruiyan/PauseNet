# Figure 2: Model Performance and Motif Discovery

This directory contains the reproducible starting point for the Figure 2
interpretation workflow: sequence attributions followed by TF-MoDISco motif
discovery. It operates on a trained PauseNet checkpoint and the standard
PauseNet split-directory format.

## What this workflow does

1. `01_compute_deepshap.py` calculates DeepSHAP-style expected integrated
   gradients for the count and profile tasks.
2. `02_run_tfmodisco.py` uses those hypothetical contribution scores to
   discover positive and negative sequence motifs with modisco-lite.
3. `run_figure2_motif_discovery.sh` runs the two steps in sequence for both
   model tasks.
4. `03_plot_tfmodisco_pwm_pca.py` builds a count/profile motif atlas and
   PWM-similarity PCoA projections from the two TF-MoDISco HDF5 files.

The profile attribution target is a differentiable profile-shape score: the
mean-centered profile logits weighted by the detached predicted softmax
profile. The count attribution target is the predicted `log1p(counts)`.

## Requirements

Install PauseNet first, then install the Figure 2 extras:

```bash
pip install -e .
pip install -r paper/figures/fig2_performance/requirements.txt
```

## Run the full workflow

```bash
cd PauseNet
bash paper/figures/fig2_performance/run_figure2_motif_discovery.sh \
  /path/to/model_data \
  /path/to/best_model.pt \
  /path/to/figure2_motif_discovery \
  cuda:0
```

The data directory must contain the standard `train/`, `validation/`, and
`test/` directories, including `sequence_codes.npy`, `profiles.npy`, and
`manifest.tsv`. The default workflow attributes all test examples relative to
sampled training-sequence backgrounds, then runs TF-MoDISco separately for the
count and profile tasks.

For a faster exploratory run, add `--max-samples` directly to the Python
commands. Run `python ... --help` for all options.

## Plot the updated motif atlas and PWM projection

```bash
python paper/figures/fig2_performance/03_plot_tfmodisco_pwm_pca.py \
  --count-h5 /path/to/count_tfmodisco_patterns.h5 \
  --profile-h5 /path/to/profile_tfmodisco_patterns.h5 \
  --outdir /path/to/figure2_pwm_pcoa
```

This produces three vector PDFs: an atlas of all discovered motifs, a labelled
PWM-similarity projection, and an unlabelled projection with four motif
neighbourhoods. The projection is a classical PCoA of pairwise PWM distances;
each pair is compared after optimizing relative alignment and reverse-
complement orientation. The four visual neighbourhoods are K-means groups
on the first two PCoA coordinates.

## Outputs

```text
figure2_motif_discovery/
  deepshap/
    count_contribs_full_Lx4_*.npy
    count_contribs_projected_*.npy
    profile_contribs_full_Lx4_*.npy
    profile_contribs_projected_*.npy
    manifest_*.tsv
    metadata_*.json
  tfmodisco/
    count/
      count_tfmodisco_patterns.h5
      count_*.meme
      count_pattern_summary.tsv
    profile/
      profile_tfmodisco_patterns.h5
      profile_*.meme
      profile_pattern_summary.tsv
```

Attribution arrays, TF-MoDISco HDF5 files, motifs, and checkpoints are
deliberately not tracked in Git because they are generated results or large
binary artifacts. Store release-ready outputs in an archival repository or a
GitHub Release.

## Notes for the HEK293T model

The official HEK293T checkpoint was originally written before the final
`PauseNet` module names were cleaned up. `01_compute_deepshap.py` maps its
legacy profile/count-head parameter names when loading that checkpoint, so the
same script supports both the released checkpoint and new PauseNet checkpoints.
