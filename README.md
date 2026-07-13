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

## Standard Data Format

Each dataset should be converted into split directories:

```text
my_dataset/
  train/
    sequence_codes.npy
    profiles.npy
    counts.npy
    profile_loss_mask.npy
    sample_types.npy
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
- `sample_types.npy`: shape `(N,)`, optional sample type code. By convention,
  `0` denotes positive/high-confidence windows.
- `manifest.tsv`: metadata table with genomic coordinates and annotations.

See [docs/data_format.md](docs/data_format.md) for details.

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

## Pretrained Models

Pretrained model weights are not stored directly in this repository. Model
cards and metrics live under [pretrained/](pretrained/). Large checkpoint files
should be distributed through GitHub Releases, Zenodo or Hugging Face.

## Manuscript Figure Code

Code used to generate manuscript figures is organized under [paper/](paper/).
The paper code is intentionally separated from the reusable model package.

## Citation

Citation information will be added after the manuscript/preprint is available.
