"""Fig. 2, v2 style: ONE dense row of grouped bars (same data and numbers as scripts/figures/fig2_speed_memory.py).
(a) time per Monte Carlo step per temperature: the full step and its three stages side by side on one axis (v1 panels a, c, d, e);
(b) memory (v1 panel b).  Every pair = icet on one CPU core | this work on one GPU, at one cell size.
All numbers are written vertically above the bars; the ratio icet / this work stands above the GPU value of each pair.
Inputs:  results/bench_stage/gpu_icet_*H100*_R256.json and mchammer.json.
Output:  figures/v2/fig2_speed_memory.{pdf,svg,png}.  Source Data: that of v1 (figures/SourceData_Fig2.xlsx), the numbers are unchanged.
Run:     python scripts/figures/v2/fig2_speed.py"""
import os, sys, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import style as S
from neuralce.paths import REPO, RESULTS
import matplotlib as mpl; mpl.use("Agg"); FAM = S.use()
from matplotlib.lines import Line2D

B = f"{RESULTS}/bench_stage"; OUTDIR = f"{REPO}/figures/v2"; os.makedirs(OUTDIR, exist_ok=True); OUT = f"{OUTDIR}/fig2_speed_memory"
SIZES = [131072, 1048576, 10061824]; ST = ("propose", "energy", "accept"); FIT = [16384, 131072, 1048576]; NODE_GB = 515.0

# ---- data (identical to fig2_speed_memory.py) ------------------------------------------------------------------------------------------
G = {r["n_atoms"]: r for r in json.load(open(glob.glob(f"{B}/gpu_icet_*H100*_R256.json")[0]))}
gpu = {}
for N in SIZES:
    r = G[N]; raw = dict(propose=r["us_per_replica_step"]["propose"], energy=r["us_per_replica_step_direct"]["energy"], accept=r["us_per_replica_step_direct"]["accept"])
    tot = r["us_per_replica_step"]["total"]; s = tot / sum(raw.values())
    gpu[N] = dict(total=tot, mem=r["peak_MB"] / 1024, **{k: raw[k] * s for k in ST})          # stage times normalized to the total
Mc = {r["n_atoms"]: r for r in json.load(open(f"{B}/mchammer.json"))}
def powerlaw(vals):
    b, lna = np.polyfit(np.log(FIT), np.log(vals), 1); return float(b), float(np.exp(lna))
fits = {k: powerlaw([Mc[n][f"t_{k}_us"] for n in FIT]) for k in ST}; fits["mem"] = powerlaw([Mc[n]["mem_MB"] / 1024 for n in FIT])
cpu = {}
for N in SIZES:
    if N in Mc: c = dict(measured=True, mem=Mc[N]["mem_MB"] / 1024, **{k: Mc[N][f"t_{k}_us"] for k in ST})
    else: c = dict(measured=False, **{k: fits[k][1] * N ** fits[k][0] for k in ST + ("mem",)})   # does not fit in the 515 GB node: y = a N^b through the three largest measured sizes
    c["total"] = sum(c[k] for k in ST); cpu[N] = c


def ratio(r):
    if r < 1000: return f"{r:.0f}×"
    e = int(np.floor(np.log10(r))); m = round(r / 10 ** e, 1)
    if m >= 10: m, e = m / 10, e + 1
    return rf"${m:.1f}{{\times}}10^{{{e}}}$×"
def num(v, gpu_side, key):
    if gpu_side: return f"{v:,.1f}" if key == "mem" and v >= 1 else f"{v:,.2f}"
    return f"{v:,.0f}" if v >= 100 else f"{v:,.1f}"


# ---- layout (mm from the top-left corner) -------------------------------------------------------------------------------------------------
GROUPS = [("total", "Total"), ("propose", "Proposal"), ("energy", "Energy evaluation (ΔE)"), ("accept", "Accept + update")]
BW, GAP, PAD = 0.36, 0.5, 0.24                                                           # bar width, extra space between groups, space at the axis ends (units of one pair)
XA = lambda g, k: g * (len(SIZES) + GAP) + k                                             # centre of pair k of group g
XLIM_A = (-BW - PAD, XA(len(GROUPS) - 1, len(SIZES) - 1) + BW + PAD); XLIM_B = (-BW - PAD, len(SIZES) - 1 + BW + PAD)
X0, XR, GUT = 16.0, 180.0, 13.0                                                          # left edge of a, right edge of b, gutter between them (y axis of b)
UNIT = (XR - X0 - GUT) / (XLIM_A[1] - XLIM_A[0] + XLIM_B[1] - XLIM_B[0])                 # mm per pair: the same in a and b, so every bar has the same width
WA, WB = UNIT * (XLIM_A[1] - XLIM_A[0]), UNIT * (XLIM_B[1] - XLIM_B[0]); XB = XR - WB
Y_LEG, Y_HEAD, Y_AX, HAX = 3.9, 8.4, 10.6, 34.0                                           # legend line, group names, top of the axes, axes height
C = S.Canvas(Y_AX + HAX + 8.6 + S.EDGE); fig = C.fig
axa, axb = C.ax(X0, Y_AX, WA, HAX), C.ax(XB, Y_AX, WB, HAX)
XT = [rf"${N / 10 ** np.floor(np.log10(N)):.1f}{{\times}}10^{{{int(np.floor(np.log10(N)))}}}$" for N in SIZES]; PAIRS = []


def pair(ax, x, key, N, box):
    o, g = cpu[N], gpu[N]; xc, xg = x - BW / 2, x + BW / 2
    if o["measured"]: ax.bar(xc, o[key], BW, color=S.NAVY, ec="black", lw=0.6, zorder=3)
    else: ax.bar(xc, o[key], BW, color="white", ec=S.NAVY, lw=0.6, hatch="//////", zorder=3)
    ax.bar(xg, g[key], BW, color=S.BLUE, ec="black", lw=0.6, zorder=3)
    S.bar_value(ax, xc, o[key], num(o[key], False, key) + ("" if o["measured"] else "*"), box=box)
    PAIRS.append((ax, xg, S.bar_value(ax, xg, g[key], num(g[key], True, key), box=box), ratio(o[key] / g[key]), box))


def axis(ax, xlim, lo, hi, top, ticks):
    """Decades lo..hi carry the y axis; the space up to 10^top holds the vertical numbers."""
    ax.set_yscale("log"); ax.set_ylim(10.0 ** lo, 10.0 ** top); ax.set_xlim(*xlim); S.log_axis(ax.yaxis, lo, hi, major_every=2); ax.spines["left"].set_bounds(10.0 ** lo, 10.0 ** hi)
    ax.set_xticks(ticks); ax.set_xticklabels(XT * (len(ticks) // len(SIZES))); ax.tick_params(axis="x", length=2.2, pad=1.8); ax.set_xlabel("Number of atoms", labelpad=1.5)


axis(axa, XLIM_A, -2, 6, 8.55, [XA(g, k) for g in range(len(GROUPS)) for k in range(len(SIZES))])
axis(axb, XLIM_B, -1, 3, 4.25, list(range(len(SIZES))))
for g, (key, name) in enumerate(GROUPS):
    for k, N in enumerate(SIZES): pair(axa, XA(g, k), key, N, False)
    C.text(X0 + UNIT * (XA(g, 1) - XLIM_A[0]), Y_HEAD, name, ha="center")                # group names above the axis, like column heads
for k, N in enumerate(SIZES): pair(axb, k, "mem", N, True)
axb.axhline(NODE_GB, color=S.GUIDE, lw=0.8, ls=(0, (4, 2)), zorder=2)
S.ylabel(axa, "Time per Monte Carlo step\nper temperature (μs)", linespacing=1.15); S.ylabel(axb, "Memory (GB)", dx=9.3)

fig.canvas.draw(); rd = fig.canvas.get_renderer()                                         # ratio: vertical, above the GPU value of its pair
for ax, xg, tg, s, box in PAIRS:
    yd = ax.transData.inverted().transform((0, tg.get_window_extent(rd).y1))[1]
    ax.annotate(s, (xg, yd), xytext=(0, 2.6), textcoords="offset points", rotation=90, ha="center", va="bottom", fontsize=6, color=S.BLUE, zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.5) if box else None)

C.letter(axa, Y_LEG, "a"); C.letter(XB - 13.6, Y_LEG, "b")
C.legend([S.swatch(S.NAVY), S.swatch(S.BLUE), S.swatch("white", hatch="//////", ec=S.NAVY)], ["icet (CPU)", "This work (GPU)", "icet (CPU), extrapolated"],
         X0, Y_LEG, loc="center left", ncol=3, columnspacing=1.8, **S.SWATCH_KW)
C.legend([Line2D([], [], color=S.GUIDE, lw=0.8, ls=(0, (4, 2)))], ["largest node memory (515 GB)"], XR, Y_LEG, loc="center right", handlelength=2.2)
S.check(C, "ab", "Fig. 2"); C.save(OUT)
print(f"pair pitch {UNIT:.2f} mm, bar width {UNIT * BW:.2f} mm, a {WA:.1f} mm wide, b {WB:.1f} mm wide")
print("ratios:", {s: [f"{cpu[N][k] / gpu[N][k]:.3g}" for N in SIZES] for s, k in zip(("total", "mem") + ST, ("total", "mem") + ST)})
