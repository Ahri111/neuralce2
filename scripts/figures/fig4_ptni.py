"""Fig. 4 (Nature format, 183 x 165-170 mm): heat capacity of the PtNi order-disorder transition vs cell size, and free energy of mixing.
(a) orthonormal cluster expansion C_V(T), (b) neural cluster expansion C_V(T) at Pt3Ni / NiPt / Ni3Pt, every size.
Model names are row headers outside the axes and every row carries its own size legend (only the sizes that row actually has),
placed in blank space above the curves - no text or legend may sit on the data.
Solid = MBAR reweighting of all replicas of a parallel-tempering run (pymbar, robust solver).
Dashed = free-energy crossing of the ordered and disordered phases (two-phase analysis, results/ptni_two_phase/<fw>_<x>_M<M>.json):
  conv 131,072 atoms and icet 10,061,824 atoms are drawn from the phase data of a smaller run (N_run in the json: 16,384 and 1,048,576).
(c, d) Delta G_mix of the two models at the DFT compositions (108-atom DFT cell).
Supplementary Fig. S2: C_V-peak temperature vs number of atoms (a Pt3Ni, b NiPt, c Ni3Pt).

Inputs:  results/ptni_ladder/_fig_ptni_curves.pkl   the MBAR C_V curves (the MBAR step takes ~7 min).  If the cache exists it is used as is;
                                                     REFRESH=1 (or a missing cache) rebuilds it from the raw parallel-tempering runs under $NCE_RUNS/ptni_ladder/<fw>/<x>/<tag>/
                                                     (energies.npz for the neural cluster expansion, mcmc_result.npz for icet; not shipped) and rewrites the cache.
         results/ptni_two_phase/<fw>_<x>_M<M>.json  free-energy-crossing curves of the large cells (used only when rebuilding the cache).
         results/figure_data/fig_gmix_ptni_M3.json   Delta G_mix of both models;  data/benchmark_folds.json  the DFT formation energies.
Outputs in figures/: fig4_ptni.{pdf,svg,png,json}, figS2_ptni_tc.{pdf,svg,png}, SourceData_Fig4.xlsx (a-d, experiment), SourceData_FigS2.xlsx (a-c, experiment).
Run:     python scripts/figures/fig4_ptni.py"""
import os, sys, glob, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import natstyle as ns
from neuralce.paths import REPO, DATA, RUNS, RESULTS
import matplotlib as mpl; mpl.use("Agg"); FAM = ns.use()
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import logsumexp
from pymbar import MBAR

KB = 8.617333262e-5; D = f"{REPO}/figures"; os.makedirs(D, exist_ok=True); FD = f"{RESULTS}/figure_data"; TP = f"{RESULTS}/ptni_two_phase"; L = f"{RUNS}/ptni_ladder"; CAP = 8000
TP_SIZES = {"icet:64", "icet:136", "conv:32"}                                              # (framework:M) drawn from the two-phase analysis
COMP = [("x25", r"Pt$_3$Ni", (690, 790)), ("x50", "NiPt", (918, 930)), ("x75", r"Ni$_3$Pt", (788, 853))]; SIZES = [4, 8, 12, 16, 32, 64, 136]
# experimental T_c RANGES (K), not single values: Pt3Ni 690-790 [Schoenfeld 2015 (690); Camilos 2024 Table 3 (790); Fedorov 2022 Pt-rich dystectoid 783],
# NiPt 918-930 [Nash & Singleton 1989, Fedorov 2022 (918); Camilos 2024 (930)], Ni3Pt 788-853 [Fedorov 2022 Ni-rich dystectoid (788); Camilos 2024 (815);
# Nash & Singleton 1989 (853)].
EXPC = "#ececec"                                                                            # experiment band: this grey only, no edge, alpha 1, zorder 0
EXP_REF = {"x25": "Schoenfeld 2015; Camilos 2024; Fedorov 2022", "x50": "Nash & Singleton 1989; Fedorov 2022; Camilos 2024", "x75": "Fedorov 2022; Camilos 2024; Nash & Singleton 1989"}
XR = {"x25": (500, 900), "x50": (700, 1100), "x75": (650, 1050)}
TAGS = {4: ("M4",), 8: ("M8_long*", "M8_dense"), 12: ("M12_long*", "M12_dense"), 16: ("M16_T24*", "M16_dense10k", "M16_dense", "M16_long"), 32: ("M32_dense",), 64: ("M64_dense+M64_cp*",), 136: ("M136_dense",)}
PURPLE, ICET = "#5b2a9e", "#3a3a3a"
MODEL_NAME = {"conv": "Neural cluster expansion", "icet": "Orthonormal cluster expansion"}       # the only names used in the figures and the Source Data
RAMP = {"conv": [mpl.colors.to_hex(mpl.colormaps["Purples"](v)) for v in np.linspace(0.3, 0.95, len(SIZES))],   # paper-wide: purple = cluster convolution,
        "icet": [mpl.colors.to_hex(mpl.colormaps["Greys"](v)) for v in np.linspace(0.3, 0.95, len(SIZES))]}      # dark grey = orthonormal cluster expansion
NLAB = lambda N: f"{N:,}" if N < 10 ** 5 else rf"${N / 10 ** int(np.log10(N)):.1f}\times10^{{{int(np.log10(N))}}}$"
tp_file = lambda fw, x, M: f"{TP}/{fw}_{x}_M{M}.json"


def load(fw, x, M):
    N = 4 * M ** 3
    for tag in TAGS[M]:
        runs = sorted(r_ for part in tag.split("+") for r_ in (glob.glob(f"{L}/{fw}/{x}/{part}") if "*" in part else [f"{L}/{fw}/{x}/{part}"])); Es, T0 = [], None
        for run in runs:
            try:
                if fw == "icet":
                    z = np.load(f"{run}/mcmc_result.npz"); E = np.asarray(z["energies"], float); T = np.asarray(z["temperatures"], float); b = E.shape[0] // 10 if "_cp" in os.path.basename(run) else E.shape[0] // 2; E = E[b:].T
                else:
                    z = np.load(f"{run}/energies.npz"); tr = np.asarray(z["traj"], float); T = np.asarray(z["temps"], float); E = tr[:, tr.shape[1] // 2:] * N
            except (FileNotFoundError, KeyError):
                continue
            o = np.argsort(T); T, E = T[o], E[o]
            if T0 is None: T0 = T
            elif not np.allclose(T, T0): continue
            Es.append(E)
        if Es: return T0, [np.concatenate([e[k] for e in Es]) for k in range(len(T0))], N
    return None


def mbar_cv(E_list, T_k, N, T_target):
    unc, N_k = [], []
    for e in E_list:
        cap = CAP * max(1, len(e) // 20000) if N < 10 ** 6 else max(500, 150000 // len(T_k))
        idx = np.unique(np.linspace(0, len(e) - 1, min(cap, len(e))).astype(int)); unc.append(e[idx]); N_k.append(len(idx))
    E_n = np.concatenate(unc); N_k = np.asarray(N_k); Es = E_n - E_n.mean(); b_k = 1.0 / (KB * T_k); u_kn = b_k[:, None] * Es[None, :]
    mb = MBAR(u_kn, N_k, solver_protocol="robust", maximum_iterations=20000, relative_tolerance=1e-10); f_k = np.asarray(mb.f_k)
    log_den = logsumexp(np.log(N_k)[:, None] + f_k[:, None] - u_kn, axis=0); cv = np.empty(len(T_target))
    for t, Tt in enumerate(T_target):
        lw = -Es / (KB * Tt) - log_den; lw -= logsumexp(lw); w = np.exp(lw); m1 = (w * Es).sum(); cv[t] = ((w * Es ** 2).sum() - m1 ** 2) / (KB * Tt) ** 2 / N
    return cv


# ---- curves (cached: the cache is keyed on path, mtime and size of every input file) ---------------------------------------------------------
def _inputs():
    fs = []
    for fw in ("conv", "icet"):
        for x, _, _ in COMP:
            for M in SIZES:
                fs.append(tp_file(fw, x, M))
                for tag in TAGS[M]:
                    for part in tag.split("+"):
                        for run in (glob.glob(f"{L}/{fw}/{x}/{part}") if "*" in part else [f"{L}/{fw}/{x}/{part}"]):
                            fs.append(f"{run}/mcmc_result.npz" if fw == "icet" else f"{run}/energies.npz")
    return sorted((f, os.path.getmtime(f), os.path.getsize(f)) for f in set(fs) if os.path.exists(f))
CACHE = f"{RESULTS}/ptni_ladder/_fig_ptni_curves.pkl"; KEY = (_inputs(), CAP, sorted(TP_SIZES)); CUR, TC = {}, {}       # (fw, x, M) -> dict(T, cv, method, N_run); (fw, x, M) -> T_c
if os.path.exists(CACHE) and not int(os.environ.get("REFRESH", "0")):
    _c = pickle.load(open(CACHE, "rb")); CUR, TC = _c["CUR"], _c["TC"]
    print("curves: loaded from cache", CACHE, "" if _c["key"] == KEY else "(the raw inputs under NCE_RUNS are absent or differ from those the cache was built from; REFRESH=1 rebuilds)", flush=True)
if not CUR:
    for fw in ("conv", "icet"):
        for x, _, _ in COMP:
            for M in SIZES:
                tpf = tp_file(fw, x, M)
                if f"{fw}:{M}" in TP_SIZES and os.path.exists(tpf):
                    C = json.load(open(tpf)); T = np.asarray(C["T"]); cv = np.asarray(C["cv"]); w = np.abs(T - C["Tc_inf"]) <= 35.0
                    CUR[(fw, x, M)] = dict(T=T[w], cv=cv[w], method="free-energy crossing", N_run=int(C.get("N_run", C["N"]))); TC[(fw, x, M)] = float(C["Tc_peak"]); continue
                if 4 * M ** 3 >= 10 ** 6: continue
                d = load(fw, x, M)
                if d is None: continue
                T, E, N = d; Tt = np.linspace(T.min(), T.max(), 1201) if M > 4 else np.arange(T.min(), T.max() + 1e-9, 1.0)
                try: cv = mbar_cv(E, T, N, Tt)
                except Exception as ex: print("MBAR failed", fw, x, M, repr(ex), flush=True); continue
                CUR[(fw, x, M)] = dict(T=Tt, cv=cv, method="MBAR", N_run=N); k = int(np.nanargmax(cv))
                if 0 < k < len(Tt) - 1: TC[(fw, x, M)] = float(Tt[k])
                print(f"{fw} {x} N={N}: MBAR, peak {Tt[k]:.1f} K, Cv_max {cv.max():.1f}", flush=True)
    if not CUR: raise SystemExit(f"no C_V curves: the cache {CACHE} is missing and no parallel-tempering runs were found under {L}")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True); pickle.dump(dict(key=KEY, CUR=CUR, TC=TC), open(CACHE, "wb")); print("curves: computed and cached", flush=True)

# ---- Delta G_mix data ------------------------------------------------------------------------------------------------------------------------
Gm = json.load(open(f"{FD}/fig_gmix_ptni_M3.json")); ENi, EPt = -617.84377 / 108, -652.72821 / 108
it = pd.DataFrame(json.load(open(f"{DATA}/benchmark_folds.json"))["ptni"]["items"]); xq = it.n_ni / 108.0; Hq = (it.y - (xq * ENi + (1 - xq) * EPt)) * 1e3
TS = [600, 900, 1200, 1500]; VIR = [mpl.colors.to_hex(mpl.colormaps["viridis"](v)) for v in np.linspace(0.15, 0.85, len(TS))]
rr = {fw: sorted(Gm[fw], key=lambda r: r["x"]) for fw in ("conv", "icet")}; xs = [0.0] + [r["x"] for r in rr["conv"]] + [1.0]
assert [r["x"] for r in rr["conv"]] == [r["x"] for r in rr["icet"]]
ser = {fw: {"0 K": [0.0] + [r["H0"] for r in rr[fw]] + [0.0], **{f"{t} K": [0.0] + [r[f"G{t}"] for r in rr[fw]] + [0.0] for t in TS}} for fw in ("conv", "icet")}
diff = {k: np.array(ser["conv"][k]) - np.array(ser["icet"][k]) for k in ser["conv"]}        # not drawn; kept for fig4_ptni.json

# ==== Fig. 4 ================================================================================================================================
# Canvas 183 x 165 mm (up to 170 if a row gap needs it).  Axes fixed in mm from the TOP-LEFT corner of the canvas.
# a = orthonormal cluster expansion (top), b = neural cluster expansion; c, d = Delta G_mix of the same two models.
W = 183.0; XC = [(14.0, 63.0), (72.0, 121.0), (130.0, 179.0)]; YLIM = {"conv": (0.3, 3e3), "icet": (0.3, 1e5)}
ORDER = ("icet", "conv"); BIG = {"conv": r"1.3 × 10$^{5}$ atoms", "icet": r"1.0 × 10$^{7}$ atoms"}; LAB_UP = 4.0   # labels 4 mm above the axes top
def draw(s1, s2):
    """s1: extra mm above row b (shifts b, c, d down); s2: extra mm above c, d."""
    Hc = 165.0 + s1 + s2; fig = plt.figure(figsize=(W * ns.MM, Hc * ns.MM)); info = dict(H=Hc, xl=[[], []], lab=[[], [], []], txt=[], band={})
    axm = lambda x0, x1, y0, y1, **kw: fig.add_axes([x0 / W, 1 - y1 / Hc, (x1 - x0) / W, (y1 - y0) / Hc], **kw)
    put = lambda x, y, t, **kw: fig.text(x / W, 1 - y / Hc, t, ha="left", va="center", **kw)
    axes = {}
    for r, fw in enumerate(ORDER):
        y0, y1 = (7.0, 43.0) if r == 0 else (57.0 + s1, 93.0 + s1)
        for c, (x, lab, Texp) in enumerate(COMP):
            ax = axm(*XC[c], y0, y1, sharey=axes.get((fw, 0))); axes[(fw, c)] = ax
            for k, M in enumerate(SIZES):
                if (fw, x, M) not in CUR: continue
                q = CUR[(fw, x, M)]; m = (q["T"] >= XR[x][0]) & (q["T"] <= XR[x][1])
                ax.plot(q["T"][m], q["cv"][m], color=RAMP[fw][k], lw=0.75, ls="-" if q["method"] == "MBAR" else (0, (3, 2)), zorder=3 + k)
            ax.axvspan(*Texp, color=EXPC, lw=0, alpha=1, zorder=0); info["band"][ax] = Texp                    # literature RANGE of the experimental T_c
            t1 = ax.text(Texp[1] + 0.012 * (XR[x][1] - XR[x][0]), 0.97, "Expt.", transform=ax.get_xaxis_transform(), fontsize=6, color="k", ha="left", va="top")
            t2 = ax.text(0.03, 0.97, lab, transform=ax.transAxes, fontsize=7, ha="left", va="top"); info["txt"] += [(ax, t1), (ax, t2)]
            ax.set_yscale("log"); ax.set_ylim(*YLIM[fw]); ax.set_xlim(*XR[x]); ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100)); ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(50))
            ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=10)); ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=10)); ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            if c == 0: ax.set_ylabel(r"$C_V$ ($k_\mathrm{B}$ per atom)")
            else: ax.tick_params(labelleft=False)
            ax.tick_params(axis="x", pad=1.5); ax.set_xlabel("Temperature (K)", labelpad=0.5); info["xl"][r].append(ax.xaxis.label)
        have = [M for M in SIZES if any((fw, x, M) in CUR for x, _, _ in COMP)]                 # only the sizes this row has; entries are L (N = 4 L^3)
        meth = {M: next(CUR[(fw, x, M)]["method"] for x, _, _ in COMP if (fw, x, M) in CUR) for M in have}
        hs = [Line2D([], [], color=RAMP[fw][SIZES.index(M)], lw=0.75, ls="-" if meth[M] == "MBAR" else (0, (3, 2))) for M in have]
        lg = axes[(fw, 0)].legend(hs, [str(M) for M in have], loc="upper left", bbox_to_anchor=(0.0, 0.85), ncol=1, frameon=False, fontsize=6, handlelength=1.5,
                                  labelspacing=0.2, borderaxespad=0.3, title="L", title_fontsize=6); info["txt"].append((axes[(fw, 0)], lg))   # below the composition name
        big = max(M for M in SIZES if (fw, "x50", M) in CUR); q = CUR[(fw, "x50", big)]; m = (q["T"] >= XR["x50"][0]) & (q["T"] <= XR["x50"][1]); kp = int(np.argmax(q["cv"][m]))
        t3 = axes[(fw, 1)].text(q["T"][m][kp] - 6, q["cv"][m][kp], BIG[fw], fontsize=6, color=RAMP[fw][SIZES.index(big)], ha="right", va="center"); info["txt"].append((axes[(fw, 1)], t3))
        info["lab"][r] = [put(2.0, y0 - LAB_UP, "ab"[r], fontsize=8, fontweight="bold"), put(XC[0][0], y0 - LAB_UP, MODEL_NAME[fw], fontsize=7)]
    # c, d: Delta G_mix of each model
    yc0, yc1 = 110.0 + s1 + s2, 153.0 + s1 + s2; axg = {}
    for j, (fw, (x0, x1)) in enumerate(zip(ORDER, ((14.0, 95.0), (113.0, 179.0)))):
        ax = axm(x0, x1, yc0, yc1, sharey=axg.get(ORDER[0])); axg[fw] = ax
        ax.plot(xq, Hq, ls="none", marker="x", ms=3, color="#9a9a9a", mew=0.6, zorder=2)
        ax.plot(xs, ser[fw]["0 K"], color=ICET, lw=0.75, marker="o", ms=3, mfc=ICET, mec=ICET, zorder=5)
        for t, col in zip(TS, VIR): ax.plot(xs, ser[fw][f"{t} K"], color=col, lw=0.75, marker="o", ms=3, mfc=col, mec=col, zorder=4)
        ax.axhline(0, color="#9a9a9a", lw=0.6, zorder=1); ax.set_xlim(-0.02, 1.02); ax.set_xticks(TK); ax.set_xticklabels([TL.get(t, "") for t in TK])
        ax.set_xlabel(r"Ni fraction $\mathit{x}$ in Pt$_{1-\mathit{x}}$Ni$_{\mathit{x}}$")
        info["lab"][2] += [put(2.0 if j == 0 else x0 - 12.0, yc0 - LAB_UP, "cd"[j], fontsize=8, fontweight="bold"), put(x0, yc0 - LAB_UP, MODEL_NAME[fw], fontsize=7)]
    lo = min(min(v) for fw in ORDER for v in ser[fw].values()); axg[ORDER[0]].set_ylim(1.06 * min(lo, float(np.min(Hq))), 50)   # room above 0 for the two-row legend
    axg[ORDER[0]].set_ylabel(r"$\Delta G_\mathrm{mix}$ (meV per atom)")
    hl = [Line2D([], [], ls="none", marker="x", ms=3, color="#9a9a9a", mew=0.6), Line2D([], [], color=ICET, lw=0.75, marker="o", ms=3)] + [Line2D([], [], color=c_, lw=0.75, marker="o", ms=3) for c_ in VIR]
    info["gleg"] = (axg[ORDER[0]], axg[ORDER[0]].legend(hl, ["DFT", "0 K (Monte Carlo ground state)"] + [f"{t} K" for t in TS], loc="upper right", ncol=3, frameon=False, fontsize=6,
                                                        handlelength=2.2, columnspacing=1.0, labelspacing=0.25, borderaxespad=0.3))   # two rows in the blank strip above the zero line
    return fig, info
TK = [0, 1 / 9, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 5 / 6, 8 / 9, 1]; TL = {0: "0", 1 / 4: "1/4", 1 / 3: "1/3", 1 / 2: "1/2", 2 / 3: "2/3", 3 / 4: "3/4", 1: "1"}


def gaps(fig, info):
    """Blank space (mm): below the x-axis titles of row a to the top of the row-b labels, and below those of row b to the top of the c, d labels."""
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); mm = lambda px: px / fig.dpi * 25.4
    bot = lambda ts: max(info["H"] - mm(t.get_window_extent(rd).y0) for t in ts); top = lambda ts: min(info["H"] - mm(t.get_window_extent(rd).y1) for t in ts)
    return top(info["lab"][1]) - bot(info["xl"][0]), top(info["lab"][2]) - bot(info["xl"][1])


fig, info = draw(0.0, 0.0); g1, g2 = gaps(fig, info); print(f"row gaps at 165 mm: a-b {g1:.2f} mm, b-cd {g2:.2f} mm", flush=True)
n1, n2 = max(0.0, 4.0 - g1), max(0.0, 4.0 - g2)
if n1 + n2 > 0:
    f_ = min(1.0, 5.0 / (n1 + n2)); plt.close(fig); fig, info = draw(f_ * n1, f_ * n2); g1, g2 = gaps(fig, info)      # canvas may grow to 170 mm, not more
H = info["H"]; print(f"canvas 183 x {H:.1f} mm; row gaps: a-b {g1:.2f} mm, b-cd {g2:.2f} mm (required >= 4)", flush=True)

# overlaps: text / legend against curves, the experiment band and other text, panel by panel
rd = fig.canvas.get_renderer(); OVER = []
name = lambda o: "legend L" if isinstance(o, mpl.legend.Legend) else repr(o.get_text())
for i, (ax, o) in enumerate(info["txt"]):
    bb = o.get_window_extent(rd); tag = f"{ax.get_ylabel() or ''}".strip()
    for ln in ax.lines:
        xy = ax.transData.transform(np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)]))
        if np.any((xy[:, 0] > bb.x0) & (xy[:, 0] < bb.x1) & (xy[:, 1] > bb.y0) & (xy[:, 1] < bb.y1)): OVER.append(f"{name(o)} over a curve"); break
    bx = ax.transData.transform([(info["band"][ax][0], 1.0), (info["band"][ax][1], 1.0)])[:, 0]
    if bb.x1 > bx[0] and bb.x0 < bx[1]: OVER.append(f"{name(o)} over the experiment band")
    for ax2, o2 in info["txt"][i + 1:]:
        if ax2 is ax and bb.overlaps(o2.get_window_extent(rd)): OVER.append(f"{name(o)} over {name(o2)}")
axG, lgG = info["gleg"]; bbG = lgG.get_window_extent(rd); ptsG = np.vstack([axG.transData.transform(np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)])) for ln in axG.lines])
clrG = (bbG.y0 - ptsG[(ptsG[:, 0] > bbG.x0 - 3) & (ptsG[:, 0] < bbG.x1 + 3), 1].max()) / fig.dpi * 25.4 - 1.5 / 72 * 25.4   # legend bottom to the top of the highest marker under it (3 pt markers)
if clrG < 1.0: OVER.append(f"Delta G_mix legend only {clrG:.2f} mm above the data")
print("overlaps:", OVER if OVER else "none", f"| c: legend to the highest marker below it {clrG:.2f} mm | margins (left, right, bottom, top) mm: {ns.margins(fig)}", flush=True)
fig.savefig(f"{D}/fig4_ptni.pdf"); fig.savefig(f"{D}/fig4_ptni.svg"); fig.savefig(f"{D}/fig4_ptni.png", dpi=600); print("wrote fig4_ptni.pdf / .svg / .png", flush=True)
szP, badP = ns.text_check(fig, "abcd")

# ==== Supplementary Fig. S2: T_c vs N, 183 x 50 mm =========================================================================================
def add_ax(f, Wd, Hd, x, y, w, h, **kw): return f.add_axes([x / Wd, y / Hd, w / Wd, h / Hd], **kw)
def letter(f, Wd, Hd, x, y, s): f.text(x / Wd, y / Hd, s, fontsize=8, fontweight="bold", ha="left", va="bottom")
X0, PW, GAPX, YS, HS = 13.5, 52.5, 4.5, 12.0, 32.0                                              # right edge of c at 180 mm; >= 2 mm blank at every edge
W2, H2 = 183.0, 50.0; f2 = plt.figure(figsize=(W2 * ns.MM, H2 * ns.MM)); axc = []
YS2 = {"x25": (610, 850), "x50": (780, 1020), "x75": (730, 970)}                               # same 240 K span everywhere; a is shifted up so its blank top holds the legend
for c, (x, lab, Texp) in enumerate(COMP):
    ax = add_ax(f2, W2, H2, X0 + c * (PW + GAPX), YS, PW, HS); axc.append(ax)
    for fw, col, mk in (("conv", PURPLE, "o"), ("icet", ICET, "s")):
        pm = sorted((4 * M ** 3, TC[(fw, x, M)]) for M in SIZES if (fw, x, M) in TC and CUR[(fw, x, M)]["method"] == "MBAR")
        pf = sorted((4 * M ** 3, TC[(fw, x, M)]) for M in SIZES if (fw, x, M) in TC and CUR[(fw, x, M)]["method"] != "MBAR")
        if pm: ax.plot(*zip(*pm), color=col, lw=0.75, marker=mk, ms=3, mfc=col, mec=col, mew=0.5, zorder=4)
        if pf:
            ax.plot(*zip(*([pm[-1]] + pf)), color=col, lw=0.75, ls=(0, (3, 2)), zorder=3)
            ax.plot(*zip(*pf), ls="none", marker=mk, ms=3, mfc="white", mec=col, mew=0.6, zorder=4)
    ax.axhspan(*Texp, color=EXPC, lw=0, zorder=0.5)                                           # experimental range
    ax.set_xscale("log"); ax.set_xlim(150, 2e7); ax.set_ylim(*YS2[x]); ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50)); ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(25))
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=10)); ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=10)); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.text(0.03, 0.97, lab, transform=ax.transAxes, fontsize=7, ha="left", va="top"); ax.set_xlabel("Number of atoms")
    if c == 0: ax.set_ylabel(r"$T_\mathrm{c}$ (K)")
    letter(f2, W2, H2, 2.0 if c == 0 else X0 + c * (PW + GAPX) - 4.0, YS + HS + 0.8, "abc"[c])
hl = [Line2D([], [], color=PURPLE, lw=0.75, marker="o", ms=3, mfc=PURPLE, mec=PURPLE), Line2D([], [], color=ICET, lw=0.75, marker="s", ms=3, mfc=ICET, mec=ICET),
      Line2D([], [], color="0.45", lw=0.75, marker="o", ms=3, mfc="0.45", mec="0.45"), Line2D([], [], color="0.45", lw=0.75, ls=(0, (3, 2)), marker="o", ms=3, mfc="white", mec="0.45", mew=0.6),
      Patch(facecolor=EXPC, edgecolor="none")]
LKW = dict(loc="upper right", ncol=1, frameon=False, fontsize=6, handlelength=2.0, handletextpad=0.5, labelspacing=0.25, borderaxespad=0.3)
lgS = [axc[0].legend(hl[:2], ["Neural cluster expansion", "Orthonormal cluster expansion"], **LKW),                # models in a, methods in b: each sits in the blank strip
       axc[1].legend(hl[2:], ["MBAR reweighting", "Free-energy crossing", "Experiment"], **LKW)]                   # above the experimental band of its panel
f2.canvas.draw(); _r2 = f2.canvas.get_renderer()
for ax_, lg_, (_, _, Texp_) in zip(axc, lgS, COMP):
    print(f"Supp. Fig. S2 legend bottom to the top of the experimental band: {(lg_.get_window_extent(_r2).y0 - ax_.transData.transform((1e3, Texp_[1]))[1]) / f2.dpi * 25.4:.2f} mm", flush=True)
print("Supp. Fig. S2 margins (left, right, bottom, top) mm:", ns.margins(f2), flush=True)
f2.savefig(f"{D}/figS2_ptni_tc.pdf"); f2.savefig(f"{D}/figS2_ptni_tc.svg"); f2.savefig(f"{D}/figS2_ptni_tc.png", dpi=600); print("wrote figS2_ptni_tc.pdf / .svg / .png", flush=True)
szS, badS = ns.text_check(f2, "abc")

# ==== Source Data + numbers =================================================================================================================
with pd.ExcelWriter(f"{D}/SourceData_Fig4.xlsx", engine="openpyxl") as xw:
    for sh, fw in (("a", "icet"), ("b", "conv")):                                      # a = orthonormal cluster expansion, b = neural cluster expansion
        rows = [dict(Composition=x, **{"Number of atoms": 4 * M ** 3, "Method": CUR[(fw, x, M)]["method"], "Phase data from (atoms)": CUR[(fw, x, M)]["N_run"], "T (K)": float(t_), "C_V (k_B per atom)": float(c_)})
                for x, _, _ in COMP for M in SIZES if (fw, x, M) in CUR for t_, c_ in zip(CUR[(fw, x, M)]["T"], CUR[(fw, x, M)]["cv"]) if XR[x][0] <= t_ <= XR[x][1]]
        pd.DataFrame(rows).to_excel(xw, sheet_name=sh, index=False)
    pd.DataFrame({"Ni fraction x": xs, **{f"Orthonormal cluster expansion {k} (meV per atom)": v for k, v in ser["icet"].items()}}).to_excel(xw, sheet_name="c", index=False)
    pd.DataFrame({"Ni fraction x": list(xq), "DFT Delta H_mix (meV per atom)": list(Hq)}).to_excel(xw, sheet_name="c", index=False, startcol=7)
    pd.DataFrame({"Ni fraction x": xs, **{f"Neural cluster expansion {k} (meV per atom)": v for k, v in ser["conv"].items()}}).to_excel(xw, sheet_name="d", index=False)
    pd.DataFrame({"Ni fraction x": list(xq), "DFT Delta H_mix (meV per atom)": list(Hq)}).to_excel(xw, sheet_name="d", index=False, startcol=7)
    pd.DataFrame([{"Composition": x, "Experimental T_c low (K)": t[0], "Experimental T_c high (K)": t[1], "Sources": EXP_REF[x]} for x, _, t in COMP]).to_excel(xw, sheet_name="experiment", index=False)
with pd.ExcelWriter(f"{D}/SourceData_FigS2.xlsx", engine="openpyxl") as xw:
    for c, (x, _, _) in enumerate(COMP):
        pd.DataFrame([dict(Model=MODEL_NAME[fw], **{"Number of atoms": 4 * M ** 3, "T_c (K)": v, "Method": CUR[(fw, x_, M)]["method"]}) for (fw, x_, M), v in sorted(TC.items()) if x_ == x]).to_excel(xw, sheet_name="abc"[c], index=False)
    pd.DataFrame([{"Composition": x, "Experimental T_c low (K)": t[0], "Experimental T_c high (K)": t[1], "Sources": EXP_REF[x]} for x, _, t in COMP]).to_excel(xw, sheet_name="experiment", index=False)
kmax = max(diff, key=lambda k: np.abs(diff[k]).max()); jmax = int(np.argmax(np.abs(diff[kmax])))
json.dump(dict(Tc={f"{k[0]}_{k[1]}_N{4*k[2]**3}": v for k, v in TC.items()}, max_abs_diff=dict(value=float(diff[kmax][jmax]), T=kmax, x=xs[jmax])), open(f"{D}/fig4_ptni.json", "w"), indent=1)
print("font:", FAM); print("Fig. 4 text:", szP, badP); print("Fig. S2 text:", szS, badS)
print(f"d: max |conv - icet| = {abs(diff[kmax][jmax]):.2f} meV per atom ({diff[kmax][jmax]:+.2f}) at x = {xs[jmax]:.3f}, {kmax}")
