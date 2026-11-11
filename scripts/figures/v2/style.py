"""Shared style of the restyled paper figures (v2).  The v1 scripts (scripts/figures/fig*.py) and figures/paper/ are not touched.

Look: grouped bars with black outlines and rotated value labels, frameless legends with square swatches, left and bottom spines only,
a muted navy / slate / blue palette.
Nature rules kept from scripts/figures/natstyle.py: 183 mm wide, Arial (falls back to the default sans-serif with one warning), text 5-7 pt,
panel letters 8 pt bold, vector PDF + SVG + 600 dpi PNG.

ONE PANEL GRID FOR EVERY FIGURE, so panels of the same kind have the same size everywhere:
    three columns : axes 44 mm wide at x = 16 / 76 / 136 mm          (X3, W3)
    two columns   : axes 74 mm wide at x = 16 / 106 mm               (X2, W2)
    axes height 36 mm (HA); the 16 mm gutter left of every axes holds its tick labels, axis title and panel letter.
    Panels whose categories are long text (horizontal bars, beeswarms) keep the 60 mm pitch and give 10 mm of the axes to the label column (XL, WL).
All positions are in mm from the TOP-LEFT corner of the canvas.  Used as a sibling module by the scripts in scripts/figures/v2/."""
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib._mathtext as _mt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MM = 1 / 25.4
W = 183.0
X3, W3 = (16.0, 76.0, 136.0), 44.0
X2, W2 = (16.0, 106.0), 74.0
XL, WL = (26.0, 86.0, 146.0), 34.0
HA = 36.0
STRIP = 5.5            # band above an axes: panel letter and, if any, the panel title (one text line)
LINE1 = 1.9            # centre of that text line below the top of the band
LINE = 4.2             # one more text line in the band (a legend row under the letters)
FOOT = 9.0             # below an axes: tick labels (with exponents) and the axis title
FOOT_PLAIN = 7.6       # the same when the tick labels have no exponents
ROWGAP = 4.5           # blank between the axis title of one row and the letters of the next (rule: >= 4 mm)
EDGE = 2.0             # blank at every canvas edge (rule: >= 2 mm)
LETTER_DX = 14.0       # panel letter this far left of its axes: 2 mm from the canvas edge in the first column
YLAB_DX = 11.3         # centre of the y-axis title this far left of its axes (a two-line title then ends 2 mm from the canvas edge)

# ---- palette ---------------------------------------------------------------------------------------------------------------------------
BLUE = "#0272b2"       # this work: neural cluster expansion, GPU engine
NAVY = "#253247"       # orthonormal cluster expansion, icet on the CPU
OLIVE = "#625f4a"
SLATE = "#99a3b4"
PALE = "#e6e6ed"       # bands and pale fills
RED = "#9c2826"
TEAL = "#3d9da1"
GUIDE = "#6f7b91"      # dashed reference lines and their text
INK = "#1f1f1f"        # 0 K curves, DFT markers
DOT = "#d5d8e0"        # background lattice sites of the schematic
T_COLOR = {600: "#0272b2", 900: "#62b6cb", 1200: "#e08e6d", 1500: "#9c2826"}        # temperature: blue (cold) to red (hot), the same in every figure
T_CMAP = mpl.colors.LinearSegmentedColormap.from_list("t_blue_red", ["#0272b2", "#62b6cb", "#c9ced9", "#e08e6d", "#9c2826"])
LW, MS = 1.0, 3.6      # data lines (pt) and markers (pt)
DASH = (0, (3.5, 2.0))
PREFERRED = ("Arial", "Helvetica", "Liberation Sans"); FALLBACK = "DejaVu Sans"


def ramp(light, dark, n):
    """n colours from light to dark of one hue (cell-size series)."""
    cm = mpl.colors.LinearSegmentedColormap.from_list("r", [light, dark]); return [mpl.colors.to_hex(cm(v)) for v in np.linspace(0, 1, n)]


def font_family():
    """First of Arial / Helvetica / Liberation Sans that matplotlib finds; otherwise DejaVu Sans with one warning."""
    names = {f.name for f in fm.fontManager.ttflist}; fam = next((n for n in PREFERRED if n in names), None)
    if fam is None: warnings.warn(f"none of {PREFERRED} is available to matplotlib; falling back to {FALLBACK} (the paper figures use Arial)"); fam = FALLBACK
    return fam


def use():
    _mt.SHRINK_FACTOR = 5.0 / 6.0                                                    # exponents of 6 pt text at 5 pt (the minimum text size)
    fam = font_family()
    mpl.rcParams.update({
        "font.family": fam, "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "font.weight": "normal", "axes.labelweight": "normal", "axes.titleweight": "normal",
        "axes.linewidth": 0.7, "axes.edgecolor": "black", "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 2.0,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.55, "ytick.minor.width": 0.55,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.7, "ytick.minor.size": 1.7, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": LW, "lines.markersize": MS, "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt", "patch.linewidth": 0.5, "hatch.linewidth": 0.5,
        "legend.frameon": False, "legend.handletextpad": 0.5, "legend.columnspacing": 1.4, "legend.labelspacing": 0.3, "legend.borderpad": 0.0, "legend.borderaxespad": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
        "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold", "mathtext.sf": fam,
        "mathtext.default": "regular", "mathtext.fallback": "stixsans"})
    return fam


class Canvas:
    """A 183 mm wide figure addressed in mm from its top-left corner."""

    def __init__(self, H, Wd=W):
        import matplotlib.pyplot as plt
        self.W, self.H = Wd, H; self.fig = plt.figure(figsize=(Wd * MM, H * MM))

    def ax(self, x, y, w, h=HA, **kw):
        """Axes with its top-left corner at (x, y) mm."""
        a = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw); a._mm = (x, y, w, h); return a

    def text(self, x, y, s, **kw):
        kw.setdefault("ha", "left"); kw.setdefault("va", "center"); kw.setdefault("fontsize", 7)
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def letter(self, ax_or_x, y, s):
        """Bold panel letter, LETTER_DX left of the axes (or at x mm), vertically centred on y mm."""
        x = ax_or_x._mm[0] - LETTER_DX if hasattr(ax_or_x, "_mm") else ax_or_x
        return self.text(x, y, s, fontsize=8, fontweight="bold")

    def legend(self, handles, labels, x, y, loc="upper left", **kw):
        """Figure legend with the corner named by loc at (x, y) mm."""
        kw.setdefault("fontsize", 6)
        return self.fig.legend(handles, labels, loc=loc, bbox_to_anchor=(x / self.W, 1 - y / self.H), **kw)

    def mm(self, bbox):
        """Display bbox -> (x0, x1, y_top, y_bottom) in mm from the top-left corner."""
        d = self.fig.dpi; return bbox.x0 / d * 25.4, bbox.x1 / d * 25.4, self.H - bbox.y1 / d * 25.4, self.H - bbox.y0 / d * 25.4

    def save(self, stem):
        self.fig.savefig(stem + ".pdf"); self.fig.savefig(stem + ".svg"); self.fig.savefig(stem + ".png", dpi=600); print("wrote", stem + ".pdf / .svg / .png", flush=True)


def rows(n, strips=None, foots=None, top=EDGE, h=HA):
    """Top y (mm) of each of n panel rows and the canvas height.  strips / foots: height of the band above / below the axes of each row."""
    strips = strips or [STRIP] * n; foots = foots or [FOOT] * n; ys, y = [], top
    for s, f in zip(strips, foots): ys.append(y); y += s + h + f + ROWGAP
    return ys, y - ROWGAP + EDGE


def ylabel(ax, s, dx=YLAB_DX, **kw):
    """y-axis title centred dx mm left of the axes: the titles of a column line up whatever the tick labels are."""
    ax.set_ylabel(s, va="center", **kw); ax.yaxis.set_label_coords(-dx / ax._mm[2], 0.5)


def log_axis(axis, lo, hi, major_every=1, label_every=None):
    """Decade ticks from 10^lo to 10^hi: a major tick every major_every decades, minor ticks at the other decades (or at 2..9 when every decade is major)."""
    from matplotlib.ticker import FixedLocator, NullFormatter, LogFormatterMathtext, LogLocator
    axis.set_major_locator(FixedLocator([10.0 ** e for e in range(lo, hi + 1, major_every)])); axis.set_major_formatter(LogFormatterMathtext())
    if major_every > 1: axis.set_minor_locator(FixedLocator([10.0 ** e for e in range(lo, hi + 1) if (e - lo) % major_every]))
    else: axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=40))
    axis.set_minor_formatter(NullFormatter())


def bar_value(ax, x, y, s, color="black", box=False, **kw):
    """Value written vertically above a bar (the look of the reference bar chart)."""
    return ax.annotate(s, (x, y), xytext=(0, 1.6), textcoords="offset points", rotation=90, ha="center", va="bottom", fontsize=6, color=color, zorder=6,
                       bbox=dict(fc="white", ec="none", pad=0.5) if box else None, **kw)


def swatch(color, hatch=None, ec="black"):
    return Patch(facecolor=color, edgecolor=ec, linewidth=0.5, hatch=hatch)


def line(color, marker=None, ls="-", open_=False, lw=LW, ms=MS, **kw):
    kw.setdefault("mec", color); kw.setdefault("mew", 0.9 if open_ else 0.0)
    return Line2D([], [], color=color, lw=lw, ls=ls, marker=marker, ms=ms, mfc="white" if open_ else color, **kw)


def marker(color, m="o", open_=False, ms=MS, mec=None, mew=None):
    return Line2D([], [], ls="none", marker=m, ms=ms, mfc="white" if open_ else color, mec=mec or color, mew=(0.9 if open_ else 0.0) if mew is None else mew)


SWATCH_KW = dict(handlelength=1.15, handleheight=1.15)                               # square legend keys


def fit_label(fig, s, width_mm, wrapped, wrap_to):
    """Channel name on two lines (name / distance); a name wider than the label column breaks after its commas."""
    r = fig.canvas.get_renderer()
    def wmm(t):
        o = fig.text(0, 0, t, fontsize=6); w = o.get_window_extent(r).width / fig.dpi * 25.4; o.remove(); return w
    name = wrapped(s)
    if wmm(name) <= width_mm: return name
    for wc in range(34, 9, -1):
        name = wrap_to(s, wc, head_width=min(20, wc))
        if wmm(name) <= width_mm: return name
    return name


def check(C, letters="abcdef", name=""):
    """Blank margins, text sizes and weights, text outside the canvas and text overlapping other text."""
    fig = C.fig; fig.canvas.draw(); r = fig.canvas.get_renderer(); bb = fig.get_tightbbox(r); w, h = fig.get_size_inches()
    marg = tuple(round(v * 25.4, 2) for v in (bb.x0, w - bb.x1, bb.y0, h - bb.y1))
    ghost = set()                                                                        # labels of ticks outside the view limits are never drawn
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tk in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if not lo - 1e-9 * max(1, abs(lo)) <= tk.get_loc() <= hi + 1e-9 * max(1, abs(hi)): ghost |= {id(tk.label1), id(tk.label2)}
    texts = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip() and id(t) not in ghost]
    sizes, bad, boxes = set(), [], []
    for t in texts:
        fs_, wt, s = t.get_fontsize(), t.get_fontweight(), t.get_text(); sizes.add(round(fs_, 2)); letter = s in list(letters) and fs_ == 8
        if (letter and wt not in ("bold", 700)) or (not letter and (not 5 <= fs_ <= 7 or wt not in ("normal", 400, "regular"))): bad.append((s, fs_, wt))
        e = t.get_window_extent(r)
        if e.width > 0 and e.height > 0: boxes.append((s, e))
    out = [s for s, e in boxes if e.x0 < -0.5 or e.y0 < -0.5 or e.x1 > fig.bbox.x1 + 0.5 or e.y1 > fig.bbox.y1 + 0.5]
    px = fig.dpi / 72 * 0.6                                                              # boxes must overlap by more than 0.6 pt to count
    over = [(a[0].replace("\n", " "), b[0].replace("\n", " ")) for i, a in enumerate(boxes) for b in boxes[i + 1:]
            if min(a[1].x1, b[1].x1) - max(a[1].x0, b[1].x0) > px and min(a[1].y1, b[1].y1) - max(a[1].y0, b[1].y0) > px]
    print(f"[{name}] canvas {C.W:.0f} x {C.H:.1f} mm | margins (left, right, bottom, top) mm: {marg} | text sizes (pt): {sorted(sizes)} | size/weight violations: {bad}", flush=True)
    print(f"[{name}] text outside the canvas: {out} | overlapping text: {over}", flush=True)
    return marg
