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

## Test Data

Test data are available on [Zenodo](https://doi.org/10.5281/zenodo.21371748).
Download the test data and follow the data preparation and train instructions to test model. And you can also follow evaluation instructions
below to test PauseNet with the pretrained model weights available from the same record.

## Standard Data Format

Each dataset should be converted into split directories:

```text
my_dataset/
  train/
    sequence_codes.npy
    profiles.npy
    counts.npy
    profile_loss_mask.npy
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
- `manifest.tsv`: metadata table with genomic coordinates and annotations.

See [docs/data_format.md](docs/data_format.md) for details.

## Prepare Data from Strand-Specific bigWig Files

If you only have positive- and negative-strand NET-seq, GRO-seq or PRO-seq
bigWig files, first define the genomic anchors/windows that should become
PauseNet examples. For example, these can be TSS, 5SS, 3SS, TES or
peak-centered anchors.

Then convert the signal tracks into the standard PauseNet format:

```bash
pausenet prepare-bigwig \
  --pos-bw /path/to/sample.pos.bw \
  --neg-bw /path/to/sample.neg.bw \
  --fasta /path/to/hg38.fa \
  --anchors-bed /path/to/anchors.tsv \
  --output-dir /path/to/pausenet_dataset \
  --cell-line HEK293T \
  --assay NET-seq
```

The converter writes `sequence_codes.npy`, `profiles.npy`, `counts.npy`,
`profile_loss_mask.npy` and `manifest.tsv` for each split.
See [docs/bigwig_to_dataset.md](docs/bigwig_to_dataset.md) for the full input
format and options.

## Train a Model

```bash
pausenet train --config configs/hek293t_netseq.yaml
```

The same command can be used for NET-seq, GRO-seq, PRO-seq or other
strand-oriented profile assays once the data are converted into the standard
format. The example configuration enables a `reduce_on_plateau` learning-rate
scheduler. Its `monitor` can be `val_loss`, `val_profile_loss` or
`val_count_loss`; checkpoint selection and early stopping continue to use the
total validation loss. `seed` controls model and library random state, while
`shuffle_seed` independently controls training-example order. Set
`device: auto` to use `cuda:0` when CUDA is available and otherwise fall back
to CPU. Explicit CUDA devices are validated and do not silently fall back.

## Evaluate a Checkpoint

```bash
pausenet evaluate \
  --data-dir /path/to/my_dataset \
  --checkpoint /path/to/checkpoint.pt \
  --split test \
  --output-dir outputs/my_test_eval
```

Evaluation writes:

- `test_profiles.npz`: observed/predicted counts and per-position profiles,
  profile masks, and manifest columns.
- `test_metrics.json`: loss summaries and evaluated window counts.

The evaluation command does not generate figures or similarity TSV tables.

## Pretrained Models

Pretrained PauseNet model weights are available on
[Zenodo](https://doi.org/10.5281/zenodo.21371748).
Model cards and evaluation metrics are provided under [pretrained/](pretrained/).

## Manuscript Figure Code

Code used to generate manuscript figures is organized under [paper/](paper/).
The paper code is intentionally separated from the reusable model package.

## Citation

Citation information will be added after the manuscript/preprint is available.
