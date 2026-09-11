#!/bin/bash
# Extract the representative (MANE Select) transcript GTF lines for the two
# genes shown in the chr12:9,880,000-10,080,000 track plot.
GTF=/mnt/HDD8TB/houruiyan/ref_data/hg19/gencode.v50lift37.annotation.gtf

# CD69 (-): ENST00000228434 = MANE_Select / Ensembl_canonical / appris_principal_1 / CCDS
grep "ENST00000228434" $GTF > CD69_transcript.gtf

# KLRF1 (+): ENST00000617889 = MANE_Select / Ensembl_canonical / appris_principal_1 / CCDS
grep "ENST00000617889" $GTF > KLRF1_transcript.gtf
