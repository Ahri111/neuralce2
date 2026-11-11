"""Fig. 4 and Supplementary Fig. S2, v2 style (same curves and numbers as scripts/figures/fig4_ptni.py).
Fig. 4   (a) orthonormal cluster expansion and (b) neural cluster expansion: C_V(T) at Pt3Ni / NiPt / Ni3Pt for every cell size (six 44 x 36 mm panels);
         (c, d) free energy of mixing of the same two models (two 74 x 36 mm panels).
Fig. S2  C_V-peak temperature vs number of atoms (three 44 x 36 mm panels).
The curves come from the cache written by the v1 script (results/ptni_ladder/_fig_ptni_curves.pkl: MBAR reweighting, ~7 min); this script does not recompute them.
Solid = MBAR reweighting of a parallel-tempering run, dashed = free-energy crossing of the ordered and disordered phases.
Inputs:  results/ptni_ladder/_fig_ptni_curves.pkl, results/figure_data/fig_gmix_ptni_M3.json, data/benchmark_folds.json.
Outputs: figures/v2/fig4_ptni.{pdf,svg,png}, figures/v2/figS2_ptni_tc.{pdf,svg,png}.
Run:     python scripts/figures/v2/fig4_ptni.py"""
import os, sys, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import style as S
from neuralce.paths import REPO, DATA, RESULTS
import matplotlib as mpl; mpl.use("Agg"); FAM = S.use()
import pandas as pd

FD = f"{RESULTS}/figure_data"; OUT = f"{REPO}/figures/v2"; os.makedirs(OUT, exist_ok=True); CACHE = f"{RESULTS}/ptni_ladder/_fig_ptni_curves.pkl"
COMP = [("x25", r"Pt$_3$Ni", (690, 790)), ("x50", "NiPt", (918, 930)), ("x75", r"Ni$_3$Pt", (788, 853))]; SIZES = [4, 8, 12, 16, 32, 64, 136]   # experimental T_c ranges (K), sources in the v1 script
XR = {"x25": (500, 900), "x50": (700, 1100), "x75": (650, 1050)}; YLIM = {"conv": (-0.5, 3.5), "icet": (-0.5, 5.0)}                               # decades
MODEL_NAME = {"conv": "Neural cluster expansion", "icet": "Orthonormal cluster expansion"}; MCOL = {"conv": S.BLUE, "icet": S.NAVY}; ORDER = ("icet", "conv")
BIG = {"conv": "1.3 × 10$^{5}$\natoms", "icet": "1.0 × 10$^{7}$\natoms"}
assert os.path.exists(CACHE), f"{CACHE} is missing: run scripts/figures/fig4_ptni.py once (it writes the cache)"
_c = pickle.load(open(CACHE, "rb")); CUR, TC = _c["CUR"], _c["TC"]                        # (fw, x, M) -> dict(T, cv, method, N_run); (fw, x, M) -> T_c
HAVE = {fw: [M for M in SIZES if any((fw, x, M) in CUR for x, _, _ in COMP)] for fw in ORDER}
RAMP = {"conv": dict(zip(HAVE["conv"], S.ramp("#bcdcee", "#015a8e", len(HAVE["conv"])))), "icet": dict(zip(HAVE["icet"], S.ramp("#c9ced9", "#111827", len(HAVE["icet"]))))}

# ---- Delta G_mix data (identical to fig4_ptni.py) -----------------------------------------------------------------------------------------
Gm = json.load(open(f"{FD}/fig_gmix_ptni_M3.json")); ENi, EPt = -617.84377 / 108, -652.72821 / 108
it = pd.DataFrame(json.load(open(f"{DATA}/benchmark_folds.json"))["ptni"]["items"]); xq = it.n_ni / 108.0; Hq = (it.y - (xq * ENi + (1 - xq) * EPt)) * 1e3
TS = [600, 900, 1200, 1500]; rr = {fw: sorted(Gm[fw], key=lambda r: r["x"]) for fw in ORDER}; xs = [0.0] + [r["x"] for r in rr["conv"]] + [1.0]
ser = {fw: {"0 K": [0.0] + [r["H0"] for r in rr[fw]] + [0.0], **{f"{t} K": [0.0] + [r[f"G{t}"] for r in rr[fw]] + [0.0] for t in TS}} for fw in ORDER}
TK = [0, 1 / 9, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 5 / 6, 8 / 9, 1]; TL = {0: "0", 1 / 4: "1/4", 1 / 3: "1/3", 1 / 2: "1/2", 2 / 3: "2/3", 3 / 4: "3/4", 1: "1"}

# ==== Fig. 4 ================================================================================================================================
(YA, YB, YC), H = S.rows(3, strips=[S.STRIP, S.STRIP, S.STRIP + S.LINE], foots=[S.FOOT_PLAIN] * 3); C = S.Canvas(H); fig = C.fig; axes = {}
for r, (fw, y0) in enumerate(zip(ORDER, (YA, YB))):
    lo, hi = YLIM[fw]
    for c, (x, lab, Texp) in enumerate(COMP):
        ax = C.ax(S.X3[c], y0 + S.STRIP, S.W3); axes[(fw, c)] = ax
        ax.axvspan(*Texp, color=S.PALE, lw=0, zorder=0)                                    # literature range of the experimental T_c
        for M in SIZES:
            if (fw, x, M) not in CUR: continue
            q = CUR[(fw, x, M)]; m = (q["T"] >= XR[x][0]) & (q["T"] <= XR[x][1])
            ax.plot(q["T"][m], q["cv"][m], color=RAMP[fw][M], lw=0.9, ls="-" if q["method"] == "MBAR" else (0, (3, 1.6)), zorder=3 + SIZES.index(M))
        ax.set_yscale("log"); ax.set_ylim(10.0 ** lo, 10.0 ** hi); ax.set_xlim(*XR[x]); S.log_axis(ax.yaxis, int(np.ceil(lo)), int(np.floor(hi)))
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100)); ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(50)); ax.set_xlabel("Temperature (K)")
        ax.text(Texp[1] + 0.015 * (XR[x][1] - XR[x][0]), 0.965, "Expt.", transform=ax.get_xaxis_transform(), fontsize=6, color=S.GUIDE, ha="left", va="top")
        ax.text(0.04, 0.965, lab, transform=ax.transAxes, fontsize=7, ha="left", va="top")
    S.ylabel(axes[(fw, 0)], r"$C_V$ ($k_\mathrm{B}$ per atom)")
    big = max(M for M in SIZES if (fw, "x50", M) in CUR); q = CUR[(fw, "x50", big)]; m = (q["T"] >= XR["x50"][0]) & (q["T"] <= XR["x50"][1]); kp = int(np.argmax(q["cv"][m]))
    axes[(fw, 1)].text(q["T"][m][kp] - 7, q["cv"][m][kp], BIG[fw], fontsize=6, color=RAMP[fw][big], ha="right", va="center", linespacing=1.1)
    C.letter(axes[(fw, 0)], y0 + S.LINE1, "ab"[r]); C.text(S.X3[0], y0 + S.LINE1, MODEL_NAME[fw])
    meth = {M: next(CUR[(fw, x, M)]["method"] for x, _, _ in COMP if (fw, x, M) in CUR) for M in HAVE[fw]}   # cell sizes of this row, on the row's title line
    lg = C.legend([S.line(RAMP[fw][M], lw=1.1, ls="-" if meth[M] == "MBAR" else (0, (3, 1.6))) for M in HAVE[fw]], [str(M) for M in HAVE[fw]], S.X3[2] + S.W3, y0 + S.LINE1,
                  loc="center right", ncol=len(HAVE[fw]), handlelength=1.7, columnspacing=1.2)
    fig.canvas.draw(); C.text(C.mm(lg.get_window_extent(fig.canvas.get_renderer()))[0] - 1.6, y0 + S.LINE1, "L", fontsize=6, ha="right")

lo = min(min(v) for fw in ORDER for v in ser[fw].values()); axg = {}
for j, fw in enumerate(ORDER):
    ax = C.ax(S.X2[j], YC + S.STRIP + S.LINE, S.W2); axg[fw] = ax
    ax.axhline(0, color=S.SLATE, lw=0.6, zorder=1); ax.plot(xq, Hq, ls="none", marker="x", ms=3.0, color=S.SLATE, mew=0.8, zorder=2)
    for t in TS: ax.plot(xs, ser[fw][f"{t} K"], color=S.T_COLOR[t], lw=S.LW, marker="o", ms=S.MS, mew=0, zorder=4)
    ax.plot(xs, ser[fw]["0 K"], color=S.INK, lw=S.LW, marker="o", ms=S.MS, mew=0, zorder=5)
    ax.set_xlim(-0.025, 1.025); ax.set_xticks(TK); ax.set_xticklabels([TL.get(t, "") for t in TK]); ax.set_ylim(1.06 * min(lo, float(np.min(Hq))), 12)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50)); ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(25))
    ax.set_xlabel(r"Ni fraction $\mathit{x}$ in Pt$_{1-\mathit{x}}$Ni$_{\mathit{x}}$"); C.letter(ax, YC + S.LINE1, "cd"[j]); C.text(S.X2[j], YC + S.LINE1, MODEL_NAME[fw])
S.ylabel(axg[ORDER[0]], r"$\Delta G_\mathrm{mix}$ (meV per atom)")
C.legend([S.marker(S.SLATE, "x", ms=3.0, mew=0.8), S.line(S.INK, "o")] + [S.line(S.T_COLOR[t], "o") for t in TS], ["DFT", "0 K (Monte Carlo ground state)"] + [f"{t} K" for t in TS],
         S.X2[0], YC + S.LINE1 + S.LINE, loc="center left", ncol=6, handlelength=2.0, columnspacing=1.8)
S.check(C, "abcd", "Fig. 4"); C.save(f"{OUT}/fig4_ptni")

# ==== Supplementary Fig. S2: T_c vs number of atoms ===========================================================================================
(Y0,), H2 = S.rows(1, strips=[S.STRIP + S.LINE]); C2 = S.Canvas(H2); YS2 = {"x25": (610, 850), "x50": (780, 1020), "x75": (730, 970)}   # the same 240 K span in the three panels
for c, (x, lab, Texp) in enumerate(COMP):
    ax = C2.ax(S.X3[c], Y0 + S.STRIP + S.LINE, S.W3); ax.axhspan(*Texp, color=S.PALE, lw=0, zorder=0.5)
    for fw, mk in (("conv", "o"), ("icet", "s")):
        col = MCOL[fw]
        pm = sorted((4 * M ** 3, TC[(fw, x, M)]) for M in SIZES if (fw, x, M) in TC and CUR[(fw, x, M)]["method"] == "MBAR")
        pf = sorted((4 * M ** 3, TC[(fw, x, M)]) for M in SIZES if (fw, x, M) in TC and CUR[(fw, x, M)]["method"] != "MBAR")
        if pf: ax.plot(*zip(*([pm[-1]] + pf)), color=col, lw=S.LW, ls=S.DASH, zorder=3)
        if pm: ax.plot(*zip(*pm), color=col, lw=S.LW, marker=mk, ms=S.MS, mew=0, zorder=4)
        if pf: ax.plot(*zip(*pf), ls="none", marker=mk, ms=S.MS, mfc="white", mec=col, mew=0.9, zorder=4)
    ax.set_xscale("log"); ax.set_xlim(150, 2e7); ax.set_ylim(*YS2[x]); ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50)); ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(25))
    S.log_axis(ax.xaxis, 3, 7); ax.set_xlabel("Number of atoms"); ax.text(0.04, 0.965, lab, transform=ax.transAxes, fontsize=7, ha="left", va="top")
    if c == 0: S.ylabel(ax, r"$T_\mathrm{c}$ (K)")
    C2.letter(ax, Y0 + S.LINE1, "abc"[c])
C2.legend([S.line(S.BLUE, "o"), S.line(S.NAVY, "s"), S.line(S.GUIDE, "o"), S.line(S.GUIDE, "o", ls=S.DASH, open_=True), S.swatch(S.PALE, ec="none")],
          ["Neural cluster expansion", "Orthonormal cluster expansion", "MBAR reweighting", "Free-energy crossing", "Experiment"],
          S.X3[0], Y0 + S.LINE1 + S.LINE, loc="center left", ncol=5, handlelength=2.0, columnspacing=1.8)
S.check(C2, "abc", "Fig. S2"); C2.save(f"{OUT}/figS2_ptni_tc")
print("T_c (K):", {f"{k[0]}_{k[1]}_N{4 * k[2] ** 3}": round(v, 1) for k, v in sorted(TC.items())})
