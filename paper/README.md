# Manuscript Analysis Code

This directory contains scripts and notebooks used to generate manuscript
figures. These files are intentionally separated from the reusable `pausenet`
package.

Recommended organization:

```text
paper/
  figures/
    fig1_model_framework/
    fig2_performance/
    fig3_motifs/
    fig4_g4_secondary_structure/
    fig5_ablation_ism/
  notebooks/
  source_data/
```

Large intermediate arrays and raw sequencing tracks should not be committed to
Git. Store them in an external data repository and document how to download or
recreate them.
