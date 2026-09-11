#!/usr/bin/env python3
"""Profile-only raincloud; reuse original 30-nt minimum DNA MFE values.
All observations used for distributions/boxes; <=180 points/pattern shown.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',default='/mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig2/second_structure/raincloud/cluster1_dna_mfe_scores.tsv')
p.add_argument('--output',default=str(Path(__file__).resolve().parent/'profile_mfe_raincloud.pdf'))
a=p.parse_args()
d=pd.read_csv(a.input,sep='\t'); d=d[d.source.eq('profile')]
patterns=[6,7,8,13,19,35,37]
assert sorted(d.pattern_number.unique())==patterns
assert not d.duplicated(['motif_id','seqlet_index']).any()
values=[d.loc[d.pattern_number.eq(k),'min_mfe_kcal_mol'].to_numpy() for k in patterns]
assert all(len(v)>1 and np.isfinite(v).all() for v in values)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.labelsize':14,
                     'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12})
fig,ax=plt.subplots(figsize=(5,4)); positions=np.arange(len(patterns))
color='#59ADA5'
vi=ax.violinplot(values,positions=positions,vert=False,widths=.8,
                 showmeans=False,showmedians=False,showextrema=False,bw_method=.25)
for body,pos in zip(vi['bodies'],positions):
    verts=body.get_paths()[0].vertices
    verts[:,1]=np.minimum(verts[:,1],pos)
    body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.3);body.set_linewidth(.7)
rng=np.random.default_rng(20260820)
for pos,v in zip(positions,values):
    shown=rng.choice(v,180,replace=False) if len(v)>180 else v
    ax.scatter(shown,pos+rng.uniform(.08,.27,len(shown)),s=5,color=color,alpha=.3,linewidths=0)
ax.boxplot(values,positions=positions,vert=False,widths=.14,patch_artist=True,showfliers=False,
           boxprops={'facecolor':'white','edgecolor':color,'linewidth':1},
           medianprops={'color':'#245953','linewidth':1.3},
           whiskerprops={'color':color,'linewidth':.8},capprops={'color':color,'linewidth':.8})
ax.set_yticks(positions,[f'P{k}' for k in patterns]); ax.set_ylim(len(patterns)-.5,-.6)
ax.set_xlabel('Minimum free energy (kcal/mol)');ax.set_title('Profile patterns',fontsize=14,pad=12)
ax.set_xlim(min(v.min() for v in values)-.8,max(0,max(v.max() for v in values))+.5)
ax.set_xticks([x for x in range(-40,1,5) if ax.get_xlim()[0]<=x<=ax.get_xlim()[1]])
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#888888')
ax.grid(axis='x',color='#EAEAEA',linewidth=.6);ax.set_axisbelow(True)
ax.tick_params(axis='y',length=0,pad=7)
fig.subplots_adjust(left=.13,right=.97,bottom=.18,top=.88)
fig.canvas.draw()
for t in [ax.title,ax.xaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
    b=t.get_window_extent(fig.canvas.get_renderer())
    assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height
if Path(a.output).exists():raise FileExistsError(a.output)
fig.savefig(a.output,format='pdf');plt.close(fig)
print(a.output)
print({f'P{k}':len(v) for k,v in zip(patterns,values)})
