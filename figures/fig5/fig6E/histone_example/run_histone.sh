#!/bin/bash

make_tracks_file --trackFiles ./all_genes_transcripts.gtf \
	./K562_NETseq.hg19.bw \
	./HeLaS3_NETseq.hg19.bw \
	./K562_H3K36me3.hg19.bw \
	./HeLaS3_H3K36me3.hg19.bw \
	./K562_H3K79me2.hg19.bw \
	./HeLaS3_H3K79me2.hg19.bw \
	./K562_H3K9me3.hg19.bw \
	./HeLaS3_H3K9me3.hg19.bw \
       -o  histone_tracks.ini
