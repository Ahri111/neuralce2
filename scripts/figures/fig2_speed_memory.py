"""Fig. 2 (performance): Monte Carlo speed and memory of the original icet (mchammer, 1 CPU core, one temperature) and of this work
(icet-GPU engine, NVIDIA H100, 256 temperatures in one compiled kernel), PtNi cluster expansion (same .ce file), 131,072 / 1,048,576 / 10,061,824 atoms.
Nature Communications format: 183 x 140 mm, Arial 5-7 pt (panel letters 8 pt bold), vector PDF drawn at its final size.
(a) time per MC step per temperature, (b) memory, (c-e) the three stages of a swap step: proposal, energy change, accept + update.

GPU stages: each stage is timed alone in its own compiled loop (the stage benchmark that writes results/bench_stage), so each carries the loop's fixed
per-step cost and the three do not add up to the fused step (a).  They are scaled by total / (sum of the three) so that c + d + e = a  (caption: "stage
times are normalized to the total").  CPU: total = sum of the three measured stage times (manual loop that reproduces ensemble.run within 2 %).
Ratios at 10,061,824 atoms are relative to the extrapolated CPU value (the asterisk marks only that value).
CPU at 10,061,824 atoms does not fit in the 515 GB node: y = a N^b, least squares in log-log through the three largest measured sizes
(16,384 / 131,072 / 1,048,576 atoms), fitted separately to each stage time and to the memory; the total is the sum of the extrapolated stages.

Inputs:  results/bench_stage/gpu_icet_*H100*_R256.json (GPU engine, 256 temperatures) and results/bench_stage/mchammer.json (icet on one CPU core).
Outputs: figures/fig2_speed_memory.{pdf,svg,png,json} and figures/SourceData_Fig2.xlsx (sheets a-e).  OUT_STEM overrides the output stem.
Run:     python scripts/figures/fig2_speed_memory.py"""
import os, sys, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import natstyle as ns
from neuralce.paths import REPO, RESULTS
import matplotlib as mpl; mpl.use("Agg")
import matplotlib._mathtext as _mt; _mt.SHRINK_FACTOR = 5.0 / 6.0      # exponents of 6 pt labels drawn at 5 pt (the minimum text size), not 0.7 x 6 = 4.2 pt
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, LogFormatterMathtext

FAM = ns.font_family()
mpl.rcParams.update({"font.family": FAM, "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "ytick.minor.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "ytick.minor.size": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.5,
                     "font.weight": "normal", "axes.labelweight": "normal", "axes.titleweight": "normal",
                     "mathtext.fontset": "custom", "mathtext.rm": FAM, "mathtext.it": f"{FAM}:italic", "mathtext.bf": f"{FAM}:bold", "mathtext.sf": FAM, "mathtext.default": "regular"})

B = f"{RESULTS}/bench_stage"; D = f"{REPO}/figures"; os.makedirs(D, exist_ok=True); OUT = os.environ.get("OUT_STEM", f"{D}/fig2_speed_memory")
SIZES = [131072, 1048576, 10061824]; ST = ("propose", "energy", "accept"); FIT = [16384, 131072, 1048576]; NODE_GB = 515.0
CPU_C, GPU_C, GREY = "#1a1a1a", "#5b2a9e", "#808080"

# ---- data -------------------------------------------------------------------------------------------------------------------------------
G = {r["n_atoms"]: r for r in json.load(open(glob.glob(f"{B}/gpu_icet_*H100*_R256.json")[0]))}
gpu = {}
for N in SIZES:
    r = G[N]; raw = dict(propose=r["us_per_replica_step"]["propose"], energy=r["us_per_replica_step_direct"]["energy"], accept=r["us_per_replica_step_direct"]["accept"])
    tot = r["us_per_replica_step"]["total"]; s = tot / sum(raw.values())
    gpu[N] = dict(total=tot, mem=r["peak_MB"] / 1024, raw=raw, raw_sum=sum(raw.values()), scale=s, **{k: raw[k] * s for k in ST})
Mc = {r["n_atoms"]: r for r in json.load(open(f"{B}/mchammer.json"))}
def powerlaw(vals):
    b, lna = np.polyfit(np.log(FIT), np.log(vals), 1); return float(b), float(np.exp(lna))
fits = {k: powerlaw([Mc[n][f"t_{k}_us"] for n in FIT]) for k in ST}; fits["mem"] = powerlaw([Mc[n]["mem_MB"] / 1024 for n in FIT])
fit_total = powerlaw([sum(Mc[n][f"t_{k}_us"] for k in ST) for n in FIT])       # check only: fitting the total directly
cpu = {}
for N in SIZES:
    if N in Mc: c = dict(measured=True, mem=Mc[N]["mem_MB"] / 1024, **{k: Mc[N][f"t_{k}_us"] for k in ST})
    else: c = dict(measured=False, **{k: fits[k][1] * N ** fits[k][0] for k in ST + ("mem",)})
    c["total"] = sum(c[k] for k in ST); cpu[N] = c

# ---- figure -----------------------------------------------------------------------------------------------------------------------------
FW_MM, FH_MM = 183.0, 140.0
fig = plt.figure(figsize=(FW_MM / 25.4, FH_MM / 25.4))
fig.subplots_adjust(left=16.0 / FW_MM, right=181.0 / FW_MM, top=1 - 6.5 / FH_MM, bottom=12.0 / FH_MM)   # the two-line y title needs 16 mm; >= 2 mm blank at every edge
outer = fig.add_gridspec(2, 1, hspace=0.35)                                          # >= 4 mm blank between the x title of row 1 and the titles of row 2
g1 = outer[0].subgridspec(1, 2, wspace=0.25); g2 = outer[1].subgridspec(1, 3, wspace=0.08)
axa, axb = fig.add_subplot(g1[0]), fig.add_subplot(g1[1])
axc = fig.add_subplot(g2[0]); axd = fig.add_subplot(g2[1], sharey=axc); axe = fig.add_subplot(g2[2], sharey=axc)
W = 0.38; XS = np.arange(len(SIZES)); LINE = 7.0                                   # bar width; one 6 pt text line in points
XT = [rf"${N / 10 ** np.floor(np.log10(N)):.1f}\times10^{{{int(np.floor(np.log10(N)))}}}$" for N in SIZES]


def ratio(r):
    """Speed-up / memory ratio: an integer below 1,000 (43x), else two significant figures as m x 10^e (2.5 x 10^4 x)."""
    if r < 1000: return f"{r:.0f}×"
    e = int(np.floor(np.log10(r))); m = round(r / 10 ** e, 1)
    if m >= 10: m, e = m / 10, e + 1
    return rf"${m:.1f}\times10^{{{e}}}$×"                                        # trailing x as plain text: no operator space before it


def num(v, gpu_side, key):
    if gpu_side: return f"{v:,.1f}" if key == "mem" and v >= 1 else f"{v:,.2f}"     # GPU: two decimals; memory >= 1 GB one decimal (0.53 / 3.1 / 29.9)
    return f"{v:,.0f}" if v >= 100 else f"{v:,.1f}"


def panel(ax, key, ylim, first_labels=False):
    for k, N in enumerate(SIZES):
        o, g = cpu[N], gpu[N]; xc, xg = k - W / 2, k + W / 2
        if o["measured"]: ax.bar(xc, o[key], W, color=CPU_C, ec="black", lw=0.5, zorder=3)
        else: ax.bar(xc, o[key], W, color="white", ec=CPU_C, lw=0.5, hatch="////", zorder=3)
        ax.bar(xg, g[key], W, color=GPU_C, ec="black", lw=0.5, zorder=3)
        ax.annotate(num(o[key], False, key) + ("" if o["measured"] else "*"), (xc, o[key]), xytext=(0, 1.2), textcoords="offset points", ha="center", va="bottom", fontsize=6, color=CPU_C)
        ax.annotate(num(g[key], True, key), (xg, g[key]), xytext=(0, 1.2), textcoords="offset points", ha="center", va="bottom", fontsize=6, color=GPU_C)
        up = 1.2 + LINE
        if first_labels and k == 0:
            ax.annotate("icet (CPU)", (xc, o[key]), xytext=(0, up), textcoords="offset points", ha="center", va="bottom", fontsize=6, color=CPU_C)
            ax.annotate("This work\n(GPU)", (k + 0.01, g[key]), xytext=(0, 1.2 + LINE), textcoords="offset points", ha="left", va="bottom", fontsize=6, color=GPU_C, linespacing=1.0)
            up += LINE
        hi_x, hi_v = (xc, o[key]) if o[key] >= g[key] else (xg, g[key])            # ratio one line above the number of the taller bar
        ax.annotate(ratio(o[key] / g[key]), (hi_x, hi_v), xytext=(0, up), textcoords="offset points", ha="center", va="bottom", fontsize=6, color=GPU_C,
                    zorder=5, bbox=dict(fc="white", ec="none", pad=0.4))            # white box: the 515 GB line in b stops behind the text
    ax.set_yscale("log"); ax.set_ylim(*ylim); ax.set_xlim(-0.6, len(SIZES) - 0.4)
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(-4, 10, 2)])); ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(FixedLocator([10.0 ** e for e in range(-3, 10, 2)])); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(XS); ax.set_xticklabels(XT); ax.set_xlabel("Number of atoms")


YT = "Time per Monte Carlo step\nper temperature (μs)"
panel(axa, "total", (1e-2, 1e7), first_labels=True); axa.set_ylabel(YT)
panel(axb, "mem", (1e-1, 1e4)); axb.set_ylabel("Memory (GB)")
axb.axhline(NODE_GB, color=GREY, lw=0.6, ls=(0, (4, 2)), zorder=2)
axb.annotate("largest node memory (515 GB)", (-0.55, NODE_GB), xytext=(0, 1.0), textcoords="offset points", ha="left", va="bottom", fontsize=6, color="#606060")
for ax, key, ttl in ((axc, "propose", "Proposal"), (axd, "energy", "Energy evaluation (ΔE)"), (axe, "accept", "Accept + update")):
    panel(ax, key, (1e-2, 1e7)); ax.set_title(ttl, loc="center", pad=3, fontsize=7)
axc.set_ylabel(YT)
for ax in (axd, axe): ax.tick_params(labelleft=False)
for s, ax, dx_mm in (("a", axa, 11.5), ("b", axb, 11.5), ("c", axc, 11.5), ("d", axd, 3.0), ("e", axe, 3.0)):   # letters outside the top-left corner
    p = ax.get_position(); fig.text((2.0 if s in "ac" else p.x0 * FW_MM - dx_mm) / FW_MM, p.y1 + 1.0 / FH_MM, s, fontsize=8, fontweight="bold", ha="left", va="bottom")   # a, c at x = 2 mm as in the other figures
fig.canvas.draw(); _r = fig.canvas.get_renderer(); _bb = fig.get_tightbbox(_r); _w, _h = fig.get_size_inches()
_gap = (min(a_.xaxis.label.get_window_extent(_r).y0 for a_ in (axa, axb)) - max([a_.title.get_window_extent(_r).y1 for a_ in (axc, axd, axe)] + [t_.get_window_extent(_r).y1 for t_ in fig.texts if t_.get_text() in list("cde")])) / fig.dpi * 25.4
print(f"canvas {FW_MM:.0f} x {FH_MM:.0f} mm | margins (left, right, bottom, top) mm: {[round(v * 25.4, 2) for v in (_bb.x0, _w - _bb.x1, _bb.y0, _h - _bb.y1)]} | blank between rows: {_gap:.2f} mm", flush=True)
fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".svg"); fig.savefig(OUT + ".png", dpi=600); print("wrote", OUT + ".pdf / .svg / .png")

# ---- source data + numbers --------------------------------------------------------------------------------------------------------------
import pandas as pd
with pd.ExcelWriter(f"{D}/SourceData_Fig2.xlsx", engine="openpyxl") as xw:
    for s, key, unit in (("a", "total", "µs"), ("b", "mem", "GB"), ("c", "propose", "µs"), ("d", "energy", "µs"), ("e", "accept", "µs")):
        pd.DataFrame({"Number of atoms": SIZES, f"CPU measured ({unit})": [cpu[N][key] if cpu[N]["measured"] else np.nan for N in SIZES],
                      f"CPU estimated ({unit})": [np.nan if cpu[N]["measured"] else cpu[N][key] for N in SIZES],
                      f"GPU ({unit})": [gpu[N][key] for N in SIZES]}).to_excel(xw, sheet_name=s, index=False)
json.dump(dict(gpu=gpu, cpu=cpu, fits={k: dict(b=v[0], a=v[1]) for k, v in fits.items()}, fit_total_direct=dict(b=fit_total[0], a=fit_total[1]), fit_sizes=FIT, font=FAM),
          open(OUT + ".json", "w"), indent=1)
print("font:", FAM)
print("\nsum check (us per MC step per temperature):")
for N in SIZES:
    o, g = cpu[N], gpu[N]
    print(f"  {N:>10,}  CPU c+d+e {sum(o[k] for k in ST):12,.2f}  a {o['total']:12,.2f}   | GPU raw c+d+e {g['raw_sum']:.4f}  a {g['total']:.4f}  (raw/a = {g['raw_sum'] / g['total']:.3f}, "
          f"fixed loop cost per stage (raw - a)/2 = {(g['raw_sum'] - g['total']) / 2:.4f})  normalized c+d+e {sum(g[k] for k in ST):.4f}")
print("\nextrapolation y = a N^b through", FIT)
for k in ST + ("mem",): print(f"  {k:8s} b = {fits[k][0]:.3f}  a = {fits[k][1]:.4g}  -> 10,061,824 atoms: {cpu[10061824][k]:,.1f}")
print(f"  total = sum of stages: {cpu[10061824]['total']:,.0f} us;  total fitted directly: b = {fit_total[0]:.3f} -> {fit_total[1] * 10061824 ** fit_total[0]:,.0f} us")
print("\nratios:", {s: [f"{cpu[N][k] / gpu[N][k]:.3g}" for N in SIZES] for s, k in zip("abcde", ("total", "mem") + ST)})
# text check on the figure itself (tick labels exist after the draw done by savefig)
bad, sizes = [], set()
for t in fig.findobj(mpl.text.Text):
    s = t.get_text()
    if not t.get_visible() or not s.strip(): continue
    fs_, wt = t.get_fontsize(), t.get_fontweight(); letter = s in list("abcde") and fs_ == 8; sizes.add(fs_)
    if (letter and wt not in ("bold", 700)) or (not letter and (not 5 <= fs_ <= 7 or wt not in ("normal", 400, "regular"))): bad.append((s, fs_, wt))
print("text sizes used (pt):", sorted(sizes), "| violations:", bad)
