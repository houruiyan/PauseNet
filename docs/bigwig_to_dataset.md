# Prepare a PauseNet Dataset from Strand-Specific bigWig Files

This document describes how to convert strand-specific NET-seq, GRO-seq or
PRO-seq signal tracks into the standard PauseNet training format.

## Required Inputs

Strand-specific bigWig files alone are not sufficient, because bigWig files
store signal but do not define which genomic windows should become model
examples. To build a PauseNet dataset you need:

1. Positive-strand signal bigWig, for example `sample.pos.bw`.
2. Negative-strand signal bigWig, for example `sample.neg.bw`.
3. Reference genome FASTA, matching the coordinates of the bigWig files.
4. An anchor BED/TSV file defining one row per model example.

The anchor BED can contain TSS, 5SS, 3SS, TES, peak-centered windows, or any
other user-defined regions. PauseNet will create a 2,114-bp input sequence and
a 1,000-bp central prediction region centered on each anchor.

## Anchor BED/TSV Format

The file can be headered or BED-like without a header. The minimum required
columns are:

```text
chrom
start
end
strand
```

Recommended columns:

```text
chrom
start
end
sample_id
score
strand
region_type
gene_id
gene_name
transcript_id
split
```

`region_type` can be `TSS`, `5SS`, `3SS`, `TES`, `peak`, or any user-defined
label. `split` should usually be one of `train`, `validation`, or `test`.

Example:

```text
chrom	start	end	sample_id	score	strand	region_type	gene_id	gene_name	transcript_id	split
chr1	11873	11874	DDX11L1_TSS	0	+	TSS	ENSG00000223972	DDX11L1	ENST00000456328	train
chr1	29370	29371	WASH7P_TES	0	-	TES	ENSG00000227232	WASH7P	ENST00000488147	test
```

## Command

Install PauseNet with the optional genomics dependencies:

```bash
pip install -e ".[genomics]"
```

Then run:

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

This writes split directories such as:

```text
pausenet_dataset/
  train/
    sequence_codes.npy
    profiles.npy
    counts.npy
    profile_loss_mask.npy
    manifest.tsv
  validation/
  test/
```

## Strand Orientation

PauseNet expects every example in transcriptional 5' to 3' orientation.

- `+` strand anchors use the positive-strand bigWig and the reference sequence
  as-is.
- `-` strand anchors use the negative-strand bigWig; both the DNA sequence and
  the 1,000-bp profile are reversed into 5' to 3' orientation.

By default, the converter uses absolute bigWig values. This is useful when a
negative-strand bigWig stores signal as negative values for genome-browser
visualization. Use `--preserve-signed-signal` only if your bigWig values are
meaningful signed quantities rather than read counts.

## Split Assignment

If the anchor file has a `split` column, PauseNet uses it directly.

If the anchor file does not have a split column, you can assign splits by
chromosome:

```bash
pausenet prepare-bigwig \
  --pos-bw sample.pos.bw \
  --neg-bw sample.neg.bw \
  --fasta hg38.fa \
  --anchors-bed anchors.bed \
  --output-dir pausenet_dataset \
  --train-chroms chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr9,chr10 \
  --validation-chroms chr8 \
  --test-chroms chr11,chr12
```

Any anchor not assigned by a chromosome list is written to `--default-split`,
which is `train` by default.

## Profile Loss Mask

`profile_loss_mask.npy` is set to 1 when the total signal in the 1,000-bp
prediction region is at least `--profile-min-count`:

```bash
--profile-min-count 1
```

Count loss still uses all examples. The profile mask prevents zero-signal
profiles from contributing to the multinomial/profile loss.

## What the Converter Does

For each anchor row:

1. Center a 1,000-bp output region on the anchor midpoint.
2. Add 557 bp context on both sides to create the 2,114-bp input sequence.
3. Fetch and encode the reference DNA sequence as `A=0, C=1, G=2, T=3, N=4`.
4. Fetch the matching strand-specific bigWig signal over the 1,000-bp output
   region.
5. Reverse-complement negative-strand examples into transcriptional
   orientation.
6. Save the standard PauseNet arrays and `manifest.tsv`.
