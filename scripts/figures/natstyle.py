"""Nature-format figure style shared by the paper figure scripts (scripts/figures/*.py).
Arial when matplotlib can find it (Helvetica or Liberation Sans otherwise; else the bundled DejaVu Sans with one warning), 7 pt text / 6 pt ticks
and legend, panel letters 8 pt bold only, 0.6 pt spines and ticks, no top/right spines, TrueType-embedded PDF (pdf.fonttype 42).
Exponents and subscripts of 6 pt text are drawn at 5 pt.  Arial has no U+221D (proportional to); the figures write scaling as "~ N^b" with the
ASCII tilde so that every glyph is one typeface.
Used as a sibling module: the figure scripts insert their own directory into sys.path and `import natstyle as ns`."""
import warnings
import matplotlib as mpl
import matplotlib._mathtext as _mt
from matplotlib import font_manager as fm

MM = 1 / 25.4
PREFERRED = ("Arial", "Helvetica", "Liberation Sans")
FALLBACK = "DejaVu Sans"                                                                  # bundled with matplotlib, always available
_warned = False


def font_family():
    """First of Arial / Helvetica / Liberation Sans that matplotlib finds; otherwise DejaVu Sans (one warning, no crash)."""
    global _warned
    names = {f.name for f in fm.fontManager.ttflist}
    fam = next((n for n in PREFERRED if n in names), None)
    if fam is None:
        if not _warned:
            warnings.warn(f"none of {PREFERRED} is available to matplotlib; falling back to {FALLBACK} (the paper figures use Arial)")
            _warned = True
        fam = FALLBACK
    return fam


def use():
    _mt.SHRINK_FACTOR = 5.0 / 6.0                                                    # 6 pt -> 5 pt exponents (the minimum text size), not 0.7 x 6 = 4.2 pt
    fam = font_family()
    mpl.rcParams.update({"font.family": fam, "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.5, "font.weight": "normal", "axes.labelweight": "normal",
                         "axes.titleweight": "normal", "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
                         "mathtext.sf": fam, "mathtext.default": "regular", "mathtext.fallback": "stixsans", "savefig.bbox": None})
    return fam


def margins(fig):
    """Blank margin (mm) between the drawn ink and the canvas edges: (left, right, bottom, top).  Target: every one >= 2 mm."""
    fig.canvas.draw(); bb = fig.get_tightbbox(fig.canvas.get_renderer()); w, h = fig.get_size_inches()
    return tuple(round(v * 25.4, 2) for v in (bb.x0, w - bb.x1, bb.y0, h - bb.y1))


def text_check(fig, letters="abcde"):
    """All visible text 5-7 pt and regular, except the panel letters (8 pt bold).  Returns (sizes used, violations)."""
    bad, sizes = [], set()
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if not t.get_visible() or not s.strip(): continue
        fs_, wt = t.get_fontsize(), t.get_fontweight(); sizes.add(fs_); letter = s in list(letters) and fs_ == 8
        if (letter and wt not in ("bold", 700)) or (not letter and (not 5 <= fs_ <= 7 or wt not in ("normal", 400, "regular"))): bad.append((s, fs_, wt))
    return sorted(sizes), bad
