# PauseNet Architecture

PauseNet is a sequence-only multi-task CNN.

## Input

The input is a 2,114-bp strand-oriented DNA sequence encoded as a 4-channel
one-hot tensor. `N` bases are encoded as all zeros.

## Shared Backbone

PauseNet uses:

- initial Conv1D, kernel size 21;
- 11 dilated residual Conv1D blocks;
- dilations `1, 2, 4, ..., 1024`;
- 128 channels;
- dropout 0.1.

Dilated convolutions allow the model to integrate local motif information and
larger genomic context. Residual connections stabilize optimization and preserve
features learned in shallow layers.

## Profile Head

The profile head predicts a 1,000-bp probability distribution:

```text
shared hidden features over central 1,000 bp
  -> Conv1D(128 -> 1, kernel=75)
  -> softmax over 1,000 positions
```

Because samples are strand-oriented, PauseNet uses a single softmax over the
1,000 positions rather than a joint strand-position softmax.

## Count Head

The count head follows the ProCapNet-style count branch:

```text
shared hidden features
  -> mean pooling across sequence positions
  -> Linear(128 -> 1)
  -> softplus
```

The output is predicted `log1p(counts)`.

## Loss

The default ProCapNet-style loss is:

```text
L = L_profile + alpha * L_count
```

where:

- `L_profile` is multinomial negative log likelihood over the observed profile;
- `L_count` is MSE between observed and predicted `log1p(counts)`;
- `alpha` defaults to 100.
