# PauseNet

PauseNet is a sequence-based framework for modeling nascent transcription
profiles and pausing activity from NET-seq-like assays, including NET-seq,
GRO-seq and PRO-seq.

The repository is organized around three use cases:

1. Train a PauseNet model on a user-provided assay and cell type.
2. Evaluate user data with a released pretrained model, such as the
   HEK293T NET-seq model.
3. Reproduce the analysis and figure code used in the manuscript.

## Model Overview

PauseNet takes strand-oriented DNA sequence as input and predicts two outputs:

- a base-resolution profile distribution over a 1,000-bp prediction window;
- the total pausing activity as predicted `log1p(counts)`.

PauseNet uses a ProCapNet-style sequence-only architecture: a shared dilated
residual Conv1D backbone followed by one profile head and one count head.

```text
2,114-bp DNA sequence
        |
one-hot encoding
        |
initial Conv1D + 11 dilated residual Conv1D blocks
        |
shared sequence features
   |                       |
profile head              count head
Conv1D 128->1             mean pooling across sequence
softmax over 1,000 bp     linear 128->1 + softplus
   |                       |
1,000-bp profile          log1p(counts)
```

The default loss is the ProCapNet-style multi-task objective:

```text
L = MNLL(profile) + 100 * MSE(log1p counts)
```

Because the training examples are already strand-oriented, PauseNet uses a
single softmax over the 1,000-bp output region rather than a joint
strand-position softmax.

## Installation

```bash
git clone https://github.com/houruiyan/PauseNet.git
cd PauseNet
pip install -e .
```

For GPU training, install a PyTorch build matching your CUDA version first.
If you want to prepare datasets directly from strand-specific bigWig files,
install the optional genomics dependencies:

```bash
pip install -e ".[genomics]"
```

## Standard Data Format

Each dataset should be converted into split directories:

```text
my_dataset/
  train/
    sequence_codes.npy
    profiles.npy
  counts.npy
  profile_loss_mask.npy
  anchor_type_codes.npy
  manifest.tsv
  validation/
    ...
  test/
    ...
```

Required arrays:

- `sequence_codes.npy`: shape `(N, 2114)`, integer DNA codes
  `A=0, C=1, G=2, T=3, N=4`.
- `profiles.npy`: shape `(N, 1000)`, observed read-count profile in the
  prediction region.
- `counts.npy`: shape `(N,)`, total observed read counts in the prediction
  region.
- `profile_loss_mask.npy`: shape `(N,)`, 1 for examples used in the profile
  loss and profile JSD evaluation.
- `anchor_type_codes.npy`: shape `(N,)`, codes for `TSS`, `5SS`, `3SS` and
  `TES`, used for anchor-balanced sampling when requested.
- `original_strands.npy`: shape `(N,)`, `+1` or `-1` before the negative
  strand sequence/profile were reoriented to transcription direction.
- `manifest.tsv`: metadata table with genomic coordinates and annotations.

See [docs/data_format.md](docs/data_format.md) for details.

## Prepare Data from Strand-Specific bigWig Files

PauseNet constructs examples from gene structure rather than arbitrary
peak-centered windows. It selects one representative transcript per expressed
gene, defines TSS, 5'SS, 3'SS and TES landmarks, and expands each landmark
into five adjacent 1-kb prediction windows in transcription-oriented
coordinates:

```text
TSS:  -1000..0, 0..+1000, +1000..+2000, +2000..+3000, +3000..+4000
5'SS: -2500..-1500, -1500..-500, -500..+500, +500..+1500, +1500..+2500
3'SS: -2500..-1500, -1500..-500, -500..+500, +500..+1500, +1500..+2500
TES:  -4000..-3000, -3000..-2000, -2000..-1000, -1000..0, 0..+1000
```

Every 1-kb output window receives 557 bp of flanking context on both sides,
giving a 2,114-bp sequence input. Build the dataset directly from a GTF, an
optional expressed-gene list, and strand-specific bigWig files:

```bash
pausenet prepare-bigwig \
  --pos-bw /path/to/sample.pos.bw \
  --neg-bw /path/to/sample.neg.bw \
  --fasta /path/to/hg38.fa \
  --gtf /path/to/genes.gtf.gz \
  --expressed-genes-bed /path/to/expressed_genes.bed \
  --chrom-sizes /path/to/hg38.chrom.sizes \
  --output-dir /path/to/pausenet_dataset \
  --train-chroms chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr9,chr10 \
  --validation-chroms chr8 \
  --test-chroms chr11,chr12 \
  --cell-line HEK293T \
  --assay NET-seq
```

The converter writes `sequence_codes.npy`, `profiles.npy`, `counts.npy`,
`profile_loss_mask.npy`, `anchor_type_codes.npy`, `original_strands.npy` and
`manifest.tsv` for each split, plus a dataset-level `dataset_summary.json`.
The manifest retains the original landmark, window index and gene-oriented
window coordinates for every sample.
See [docs/bigwig_to_dataset.md](docs/bigwig_to_dataset.md) for the full input
format and options.

## Train a Model

```bash
pausenet train --config configs/hek293t_netseq.yaml
```

The same command can be used for NET-seq, GRO-seq, PRO-seq or other
strand-oriented profile assays once the data are converted into the standard
format.

## Evaluate a Checkpoint

```bash
pausenet evaluate \
  --data-dir /path/to/my_dataset \
  --checkpoint /path/to/checkpoint.pt \
  --split test \
  --output-dir outputs/my_test_eval
```

Evaluation reports:

- Pearson correlation for observed vs predicted `log1p(counts)`;
- raw count Pearson correlation;
- profile Jensen-Shannon distance at 1, 5, 10 and 20 bp resolution.

It also writes `test_predictions.tsv` and `test_profile_similarity.tsv`, which
can be visualized after installing the optional figure dependencies:

```bash
python -m pip install -e ".[figures]"

pausenet visualize \
  --evaluation-dir outputs/my_test_eval \
  --split test \
  --format png
```

This creates `figures/test_count_scatter.png` (observed versus predicted
`log1p(counts)`) and `figures/test_profile_similarity.png` (mean `1 - JSD` at
1, 5, 10 and 20 bp). The profile plot compares PauseNet with a binomially split
pseudoreplicate baseline and a within-profile random-permutation baseline. Use
`--output-dir /path/to/figures` to choose another destination, or `--format pdf`
or `--format svg` for vector output.

## Pretrained Models

Pretrained model weights are not stored directly in this repository. Model
cards and metrics live under [pretrained/](pretrained/). Large checkpoint files
should be distributed through GitHub Releases, Zenodo or Hugging Face.

## Manuscript Figure Code

Code used to generate manuscript figures is organized under [paper/](paper/).
The paper code is intentionally separated from the reusable model package.

## Citation

Citation information will be added after the manuscript/preprint is available.
