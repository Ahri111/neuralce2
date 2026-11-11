"""Shared publication style of the quick-look figures (bold, high-contrast; Nature-Communications-like): one source of fonts, line widths, colours.
Imported as a sibling module (sys.path insert of scripts/figures/) by quick-look scripts; the paper figures use natstyle.py."""
import matplotlib as mpl
COL = dict(conv="#0b3fd1", icet="#5b2a9d", clamm="#8c4a1f", epi="#e05a00", gnn="#5a5a5a", dft="#000000", expt="#000000", linear="#1a8f3a")
SEQ_CONV = ["#a9cdf2", "#6ea6e8", "#3b7fd9", "#1656c0", "#062f86"]
SEQ_ICET = ["#cfc2ea", "#a88fdb", "#8460c8", "#6238ab", "#3f1680"]


def use():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "font.weight": "normal", "axes.labelsize": 14, "axes.labelweight": "bold", "axes.titlesize": 14, "axes.titleweight": "bold",
        "legend.fontsize": 11.5, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "axes.linewidth": 1.8, "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.major.size": 6.5, "ytick.major.size": 6.5,
        "xtick.major.width": 1.8, "ytick.major.width": 1.8, "xtick.minor.size": 3.5, "ytick.minor.size": 3.5, "xtick.minor.width": 1.3, "ytick.minor.width": 1.3,
        "lines.linewidth": 2.4, "lines.markersize": 7, "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "mathtext.default": "regular", "axes.unicode_minus": True})


def save(fig, stem):
    fig.savefig(stem + ".png"); fig.savefig(stem + ".pdf"); print("wrote", stem + ".png / .pdf")


def panel_label(ax, s, x=-0.20, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="left")
