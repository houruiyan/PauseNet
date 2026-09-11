"""final-figure-small style helper.

Import this module in the plotting script:

    import sys
    sys.path.insert(0, "<path-to-this-script>")
    from final_figure_small import new_figure, save_pdf

    fig, ax = new_figure()
    ax.plot(x, y, label="signal")
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("Signal")
    ax.legend()
    save_pdf(fig, "/server/output/dir/figure.pdf")

Enforced defaults (do not override unless the user asks):
  canvas 5x4 inch, ticks 12 pt, axis labels 14 pt, legend 12 pt,
  pdf.fonttype=42 (editable text), PDF only, no bbox_inches="tight".
"""
import matplotlib

matplotlib.use("Agg")

STYLE = {
    "pdf.fonttype": 42,          # TrueType, editable text in PDF
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
}

FIGSIZE = (5, 4)


def apply_style():
    matplotlib.rcParams.update(STYLE)


def new_figure(**subplot_kw):
    """Return (fig, ax) with the final-figure-small style applied."""
    import matplotlib.pyplot as plt
    apply_style()
    subplot_kw.setdefault("figsize", FIGSIZE)
    fig, ax = plt.subplots(**subplot_kw)
    return fig, ax


def save_pdf(fig, path):
    """Save PDF only. Never pass bbox_inches='tight' (keeps 5x4 canvas)."""
    if not str(path).lower().endswith(".pdf"):
        raise ValueError("final-figure-small outputs PDF only: " + str(path))
    fig.savefig(str(path))
    print("saved", path)
