# PauseNet Data Format

PauseNet expects a directory with `train`, `validation` and `test` splits.
Each split contains NumPy arrays and a metadata table.

```text
dataset/
  train/
    sequence_codes.npy
    profiles.npy
    counts.npy
    profile_loss_mask.npy
    sample_types.npy
    manifest.tsv
  validation/
  test/
```

## Arrays

### `sequence_codes.npy`

Shape: `(N, 2114)`

Integer encoded DNA sequence:

| Base | Code |
| --- | --- |
| A | 0 |
| C | 1 |
| G | 2 |
| T | 3 |
| N / padding | 4 |

All windows should be strand-oriented. For negative-strand genes, reverse
complement the sequence so the model always sees 5' to 3' transcriptional
orientation.

### `profiles.npy`

Shape: `(N, 1000)`

Observed read-count profile in the central prediction region.

### `counts.npy`

Shape: `(N,)`

Total observed counts in the 1,000-bp prediction region.

### `profile_loss_mask.npy`

Shape: `(N,)`

Boolean or `0/1` mask indicating which examples are used for the profile loss
and profile JSD evaluation. Count prediction still uses all examples.

### `sample_types.npy`

Shape: `(N,)`

Optional sample type code. The default convention is:

- `0`: positive/high-confidence windows;
- `1`: hard negative windows;
- `2`: zero-signal negative windows.

## Manifest

`manifest.tsv` should contain one row per sample. Recommended columns:

```text
sample_id
chrom
input_start
input_end
output_start
output_end
strand
gene_id
gene_name
transcript_id
region_type
cell_line
assay
split
```

`region_type` can be `TSS`, `5SS`, `3SS`, `TES`, or any user-defined anchor
type.
