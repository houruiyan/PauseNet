#!/usr/bin/env python3
"""Matched-TF ChIP heatmaps around count TF-MoDISco seqlet midpoints.

One row per seqlet occurrence, including duplicate genomic loci. ±500 bp,
motif-oriented, sorted by center-base signal. Raw bigWig values:
no absolute value, zero imputation, smoothing or per-row normalization.
Independent 1st/99th percentile color limits (display only); missing=gray.
Motif matches are candidate TF associations, not verified binding labels.
Outputs: four 5x4 inch PDFs; previous outputs are never overwritten.
"""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import h5py
import pyBigWig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/HDD8TB/houruiyan/pausing_site')
MOD=ROOT/'3_model_explanation/TFMoDISco/results/hek293t_netseq/count'
SPECS=[('PATZ1',2,388,'ENCFF083UPA_PATZ1_HEK293_hg19.bigWig'),
       ('ZNF610',7,226,'ENCFF114HMS_ZNF610_HEK293_hg19.bigWig'),
       ('SP1',8,173,'ENCFF478WJL_SP1_HEK293T_hg19.bigWig'),
       ('SP2',10,107,'ENCFF763PKH_SP2_HEK293_hg19.bigWig')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flank',type=int,default=500)
    p.add_argument('--tfs', nargs='+', choices=[s[0] for s in SPECS], default=[s[0] for s in SPECS])
    p.add_argument('--outdir',type=Path,default=Path(__file__).resolve().parent)
    a=p.parse_args()
    assert a.flank>=1
    a.outdir.mkdir(parents=True,exist_ok=True)
    specs=[s for s in SPECS if s[0] in a.tfs]
    paths=[a.outdir/f'{tf}_P{pat}_seqlet_center_pm{a.flank}_chip_heatmap.pdf' for tf,pat,_,_ in specs]
    if any(x.exists() for x in paths): raise FileExistsError('Use another --outdir; existing PDFs preserved')
    selected=pd.read_csv(ROOT/'3_model_explanation/DeepSHAP/hek293t_netseq/selected_manifest.tsv',sep='\t')
    bridge=np.load(MOD/'count_selected_dataset_indices.npy')
    assert np.array_equal(selected.contribution_array_index,np.arange(len(selected)))
    assert np.array_equal(selected.array_index,bridge)
    manifest=pd.read_csv(ROOT/'data/HEK293T_NETseq/dataset/test/manifest.tsv',sep='\t')
    assert np.array_equal(manifest.iloc[bridge].sample_id,selected.sample_id)
    codes=np.load(ROOT/'data/HEK293T_NETseq/dataset/test/sequence_codes.npy',mmap_mode='r')
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans',
        'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12})
    with h5py.File(MOD/'count_tfmodisco_patterns.h5') as h:
        for (tf,pat,expected,filename),dest in zip(specs,paths):
            g=h[f'pos_patterns/pattern_{pat}/seqlets']
            n=len(g['start']); assert n==expected
            mat=np.full((n,2*a.flank+1),np.nan)
            loci=[]; negative=0; checked=0
            bwpath=ROOT/'data/TF'/tf/filename
            with pyBigWig.open(str(bwpath)) as bw:
                chroms=bw.chroms()
                for j,(ex,s,e,rc) in enumerate(zip(g['example_idx'][:],g['start'][:],g['end'][:],g['is_revcomp'][:])):
                    ex,s,e=int(ex),int(s),int(e)
                    r=selected.iloc[ex]
                    assert 0<=s<e<=1000 and r.strand in ('+','-')
                    assert r.output_end-r.output_start==1000 and r.output_start-r.input_start==557
                    offset=(s+e-1)//2
                    minus=r.strand=='-'; negative+=int(minus)
                    center=int(r.output_end)-1-offset if minus else int(r.output_start)+offset
                    gs,ge=(int(r.output_end)-e,int(r.output_end)-s) if minus else (int(r.output_start)+s,int(r.output_start)+e)
                    loci.append((r.chrom,gs,ge))
                    if 'sequence' in g:
                        encoded=np.asarray(codes[int(r.array_index),557+s:557+e])
                        onehot=np.eye(5,dtype=float)[encoded][:,:4]
                        if rc: onehot=onehot[::-1,::-1]
                        assert np.allclose(onehot,g['sequence'][j]), f'Seqlet sequence mismatch: {tf} {j}'
                        checked+=1
                    assert r.chrom in chroms, f'Chromosome missing: {r.chrom}'
                    lo,hi=center-a.flank,center+a.flank+1
                    left,right=max(0,lo),min(chroms[r.chrom],hi)
                    if right>left:
                        v=np.asarray(bw.values(r.chrom,left,right),dtype=float)
                        v[~np.isfinite(v)]=np.nan
                        mat[j,left-lo:right-lo]=v
                    # Align to de novo motif orientation, independent of gene strand.
                    if minus != bool(rc): mat[j]=mat[j,::-1]
            finite=mat[np.isfinite(mat)]
            assert finite.size, f'No signal coverage for {tf}'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                strength=mat[:,a.flank].copy()
            order=np.argsort(-np.nan_to_num(strength,nan=-np.inf),kind='stable')
            valid_strength=strength[order][np.isfinite(strength[order])]
            assert np.all(np.diff(valid_strength)<=0)
            low,high=np.percentile(finite,[1,99])
            if np.min(finite)>=0: low=0
            if high<=low: high=float(finite.max())
            if high<=low: high=low+1
            cmap=plt.get_cmap('Blues').copy(); cmap.set_bad('#D0D0D0')
            fig,ax=plt.subplots(figsize=(5,4))
            im=ax.imshow(np.ma.masked_invalid(mat[order]),aspect='auto',origin='upper',
                extent=[-a.flank-.5,a.flank+.5,n+.5,.5],cmap=cmap,vmin=low,vmax=high,
                interpolation='nearest',rasterized=True)
            ax.axvline(0,color='#666666',ls='--',lw=.6)
            ax.set_xticks([-a.flank,0,a.flank]); ax.set_yticks([1,n])
            ax.set_xlabel('From seqlet center (bp)'); ax.set_ylabel('Seqlets (center-sorted)')
            cell='HEK293T' if tf=='SP1' else 'HEK293'
            ax.set_title(f'{tf} ChIP-seq | {cell}',fontsize=14,pad=29)
            ax.text(.5,1.025,f'Count P{pat}; n={n}; motif-oriented',transform=ax.transAxes,
                ha='center',va='bottom',fontsize=10)
            cb=fig.colorbar(im,ax=ax,pad=.04,fraction=.05,extend='both' if low>finite.min() else 'max')
            cb.set_label('ChIP signal',fontsize=14)
            fig.subplots_adjust(left=.17,right=.81,bottom=.18,top=.79)
            fig.canvas.draw()
            for text in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts,cb.ax.yaxis.label]:
                b=text.get_window_extent(fig.canvas.get_renderer())
                assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,text.get_text()
            fig.savefig(dest,format='pdf'); plt.close(fig)
            print(f'{tf}: n={n}; unique_loci={len(set(loci))}; negative_strand={negative}; sequence_checked={checked}; missing={np.isnan(mat).mean():.4%}; no_data_rows={np.isnan(mat).all(axis=1).sum()}; signal_range={finite.min():.4g}..{finite.max():.4g}; color={low:.4g}..{high:.4g}\n{bwpath}\n{dest}',flush=True)

if __name__=='__main__': main()
