# PauseNet-HEK293T-NETseq

## Model

PauseNet ProCapNet-style sequence-only model trained on HEK293T NET-seq
gene-structure anchored windows.

Architecture:

- 2,114-bp strand-oriented DNA sequence input.
- Initial Conv1D followed by 11 dilated residual Conv1D blocks.
- Profile head: central 1,000-bp hidden features, Conv1D `128 -> 1`,
  kernel size 75, softmax over positions.
- Count head: mean pooling across sequence positions, linear projection,
  softplus output.
- Loss: `MNLL(profile) + 100 * MSE(log1p counts)`.

## Input

- 2,114-bp strand-oriented DNA sequence.
- Central 1,000 bp is used as the prediction region.

## Output

- 1,000-bp NET-seq profile distribution.
- Predicted `log1p(counts)`.

## Training Data

- Cell line: HEK293T.
- Assay: NET-seq.
- Anchor types: TSS, 5SS, 3SS and TES.
- Split strategy: chromosome-based train/validation/test split.
- Split sizes: train `n = 695,060`, validation `n = 31,210`,
  test `n = 57,590`.

## Metrics

See `metrics.json`.

## Checkpoint

The checkpoint file should be distributed through a release or external data
repository, not committed directly to Git.
