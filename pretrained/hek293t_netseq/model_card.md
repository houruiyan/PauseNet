# PauseNet-HEK293T-NETseq

## Model

PauseNet sequence-only model trained on HEK293T NET-seq gene-structure anchored
windows.

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

## Metrics

See `metrics.json`.

## Checkpoint

The checkpoint file should be distributed through a release or external data
repository, not committed directly to Git.
