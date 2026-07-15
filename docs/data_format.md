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
    anchor_type_codes.npy
    original_strands.npy
    manifest.tsv
  validation/
    ...
  test/
    ...
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

### `anchor_type_codes.npy`

Shape: `(N,)`

Integer codes identifying the landmark class for each sample:

| Code | Landmark |
| --- | --- |
| 0 | TSS |
| 1 | 5SS |
| 2 | 3SS |
| 3 | TES |

The file supports optional anchor-balanced sampling during training.

### `original_strands.npy`

Shape: `(N,)`

The genomic transcript strand before reorientation: `+1` for plus strand and
`-1` for minus strand. It is metadata only; the corresponding sequence and
profile arrays are already oriented in the 5' to 3' transcription direction.

## Manifest

`manifest.tsv` should contain one row per sample. Recommended columns:

```text
sample_id
chrom
input_start
input_end
output_start
output_end
original_strand
orientation
gene_id
gene_name
transcript_id
anchor_type
anchor_index
anchor_position
window_index
relative_start
relative_end
cell_line
assay
split
```

`anchor_position` is the original genomic coordinate of the TSS, splice site
or TES. `relative_start` and `relative_end` define the selected 1-kb output
window relative to that landmark in the transcription direction.

## Creating This Format from bigWig Files

If users start from strand-specific positive and negative NET-seq, GRO-seq or
PRO-seq bigWig files, use:

```bash
pausenet prepare-bigwig \
  --pos-bw sample.pos.bw \
  --neg-bw sample.neg.bw \
  --fasta hg38.fa \
  --gtf genes.gtf.gz \
  --chrom-sizes hg38.chrom.sizes \
  --output-dir pausenet_dataset
```

The converter derives TSS, 5SS, 3SS and TES landmarks from the GTF and creates
five predefined 1-kb windows around each landmark. See
[bigwig_to_dataset.md](bigwig_to_dataset.md) for the window definitions,
strand-orientation rules and split options.
