"""Fig. 5, v2 style (same data and numbers as scripts/figures/fig5_stf_fenicr.py), drawn from stored data only.  Six identical 44 x 36 mm panels.
(a) SrTi0.5Fe0.5O2.75: energy along the swap paths from the DFT ground state (line = mean over the paths, band = 25-75 % quantile);
    pale band = energy range of the DFT training structures at x = 0.5.
(b, c) Delta G_mix, orthonormal cluster expansion | neural cluster expansion; open markers = not equilibrated.
(d) FeNiCr Warren-Cowley alpha vs T: colour = shell (named on the curves), line style = source, this work drawn heavier.
(e, f) Shapley value of the spin-pair and chemical-pair channels vs alpha_1NN / alpha_2NN; one point per temperature; the legend gives the slope of each line.
Legends that do not fit in a blank part of a 44 mm panel sit in the band above it; no legend or text is placed on data.
Inputs:  results/figure_data/{fig7_stfo_extrap.json, fig_gmix_stfo.json, fig7_fenicr_shap2nn.json}, results/fenicr_sro/<tag>_dense_F6/sro.json,
         data/new_bench_aug467/detailed_info.csv; experimental SRO values from scripts/figures/fenicr_experiment.py.
Output:  figures/v2/fig5_stf_fenicr.{pdf,svg,png}.
Run:     python scripts/figures/v2/fig5_stf_fenicr.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above."""
import os, sys, json, numpy as np, pandas as pd
from math import lgamma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import style as S
from neuralce.paths import REPO, DATA, RESULTS
import matplotlib as mpl; mpl.use("Agg"); FAM = S.use()
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from fenicr_experiment import F6_T, F6_A1, F6_A2, PAPER6_T, PAPER6_A1, PAPER6_A2

FD = f"{RESULTS}/figure_data"; OUTDIR = f"{REPO}/figures/v2"; os.makedirs(OUTDIR, exist_ok=True); OUT = f"{OUTDIR}/fig5_stf_fenicr"; NCE, OCE = "Neural cluster expansion", "Orthonormal cluster expansion"
def require(paths, hint):                                                                 # the STFO / Fe-Ni-Cr inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] v2/{os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{FD}/fig7_stfo_extrap.json", f"{FD}/fig_gmix_stfo.json", f"{FD}/fig7_fenicr_shap2nn.json", f"{RESULTS}/fenicr_sro", f"{DATA}/new_bench_aug467/detailed_info.csv"],
        "$NCE_RESULTS/figure_data, $NCE_RESULTS/fenicr_sro, $NCE_DATA/new_bench_aug467")
C1NN, C2NN = S.TEAL, S.RED                                                              # d: colour = neighbour shell
BAND = 13.0                                                                              # band above each row: four legend lines (a, d); a title and the shared legend row (b, c); the shared colour bar (e, f)
(Y1, Y2), H = S.rows(2, strips=[BAND, BAND], foots=[S.FOOT_PLAIN, S.FOOT_PLAIN]); C = S.Canvas(H); fig = C.fig
axa, axb, axc = (C.ax(x, Y1 + BAND, S.W3) for x in S.X3); axd, axe, axf = (C.ax(x, Y2 + BAND, S.W3) for x in S.X3)
for s, ax, y0 in zip("abcdef", (axa, axb, axc, axd, axe, axf), (Y1,) * 3 + (Y2,) * 3): C.letter(ax, y0 + S.LINE1, s)
LEG = dict(ncol=1, handletextpad=0.5, labelspacing=0.3)                                 # legends in the band start on the letter line
def band_legend(ax, y0, handles, labels, dy=0.55, **kw): return C.legend(handles, labels, ax._mm[0], y0 + dy, **{**LEG, **kw})

# (a) swap paths -----------------------------------------------------------------------------------------------------------------------------
X = json.load(open(f"{FD}/fig7_stfo_extrap.json")); te = np.array([t[1] for t in X["train"]]); nstep = len(X["paths"][0]) - 1; xs = np.arange(nstep + 1)
axa.axhspan(te.min(), te.max(), color=S.PALE, lw=0, zorder=0); YP = {}
for key, col in (("icet", S.NAVY), ("conv", S.BLUE)):
    Y = np.array([[r[key] for r in p] for p in X["paths"]]); YP[key] = Y
    axa.fill_between(xs, np.quantile(Y, 0.25, axis=0), np.quantile(Y, 0.75, axis=0), color=col, alpha=0.18, lw=0, zorder=2)
    axa.plot(xs, Y.mean(0), color=col, lw=S.LW, marker="o", ms=3.0, mew=0, zorder=4)
e_dft = [e for k, (v, e) in X["dft"].items() if k != "A00_500"][0]; ei, ec = YP["icet"].mean(0)[-1], YP["conv"].mean(0)[-1]
axa.plot(0, 0, ls="none", marker="*", ms=7, color="black", mew=0, zorder=6); axa.plot(nstep, e_dft, ls="none", marker="*", ms=7, color=S.GUIDE, mew=0, zorder=6)
axa.annotate(f"{ei:.0f}".replace("-", "−"), (nstep, ei), xytext=(5, 0), textcoords="offset points", fontsize=6, color=S.NAVY, ha="left", va="center")   # end values, off the curves and their bands
axa.annotate(f"{e_dft:+.0f}", (nstep, e_dft), xytext=(5, 0.5), textcoords="offset points", fontsize=6, color=S.GUIDE, ha="left", va="center")
axa.annotate(f"{ec:+.0f}", (nstep, ec), xytext=(0, -4.5), textcoords="offset points", fontsize=6, color=S.BLUE, ha="right", va="top")
k_out = int(np.argmax(YP["icet"].mean(0) < te.min()))                                  # first swap at which the orthonormal cluster expansion leaves the band downwards
axa.annotate("", (k_out, YP["icet"].mean(0)[k_out]), xytext=(0, 14), textcoords="offset points", arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=2.5))
axa.set_xlim(-0.6, nstep + 2.6); axa.set_xticks(range(0, nstep + 1, 2)); axa.set_ylim(-300, 62); axa.yaxis.set_major_locator(mpl.ticker.MultipleLocator(100)); axa.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(50))
axa.set_xlabel("Swaps away from the DFT ground state"); S.ylabel(axa, "Energy relative to the DFT\nground state (meV per site)", linespacing=1.15)
axa.text(0.04, 0.05, r"SrTi$_{0.5}$Fe$_{0.5}$O$_{2.75}$", transform=axa.transAxes, fontsize=7, ha="left", va="bottom")
band_legend(axa, Y1, [S.line(S.BLUE, "o", ms=3.0), S.line(S.NAVY, "o", ms=3.0), S.marker("black", "*", ms=7), S.marker(S.GUIDE, "*", ms=7)],
            [NCE, OCE, "DFT ground state", "DFT of the final Monte Carlo state"], handlelength=1.9)

# (b, c) Delta G_mix ---------------------------------------------------------------------------------------------------------------------------
G = json.load(open(f"{FD}/fig_gmix_stfo.json")); assert int(G["K"]) == 1
E_STO, E_SFO = -1209.4315 / 160, -929.75272 / 160; ref = lambda x: (1 - x) * E_STO + x * E_SFO
dfb = pd.read_csv(f"{DATA}/new_bench_aug467/detailed_info.csv"); xd = dfb.comp / 1000.0; Hd = (dfb.total_energy / dfb.n_atoms - ref(xd)) * 1e3
def s_ideal(x):
    nFe, nV = round(32 * x), round(16 * x); return ((lgamma(33) - lgamma(nFe + 1) - lgamma(33 - nFe)) + (lgamma(97) - lgamma(nV + 1) - lgamma(97 - nV))) / 160.0
TS = [900, 1200, 1500]
for ax, fw, ylim, step, lab in ((axb, "icet", (-560, 100), 100, OCE), (axc, "conv", (-20, 63), 20, NCE)):
    rr = sorted(G["results"][fw], key=lambda r: r["x"]); xg = [0.0] + [r["x"] for r in rr] + [1.0]; h0 = [0.0] + [r["H0"] for r in rr] + [0.0]
    ax.axhline(0, color=S.SLATE, lw=0.6, zorder=1); ax.plot(xd, Hd, ls="none", marker="x", ms=3.0, color=S.SLATE, mew=0.8, zorder=2)
    ax.plot(xg, h0, color=S.INK, lw=S.LW, marker="x", ms=3.4, mew=1.0, zorder=5)
    for t in TS:
        g = [0.0] + [r[f"G{t}"] for r in rr] + [0.0]; ok = [True] + [(-1e-6 <= r[f"S{t}"] <= s_ideal(r["x"]) + 1e-6) and r[f"G{t}"] <= r["H0"] + 1e-6 for r in rr] + [True]
        ax.plot(xg, g, color=S.T_COLOR[t], lw=S.LW, zorder=4)
        for xx, gg, o in zip(xg, g, ok): ax.plot(xx, gg, ls="none", marker="o", ms=3.2, mfc=S.T_COLOR[t] if o else "white", mec=S.T_COLOR[t], mew=0.9, zorder=6)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(*ylim); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
    ax.set_xlabel(r"Fe fraction $\mathit{x}$"); C.text(ax._mm[0], Y1 + S.LINE1, lab)
    if fw == "icet":
        j = int(np.argmin([r["H0"] for r in rr])); ax.annotate("−505", (rr[j]["x"], rr[j]["H0"]), xytext=(-5, 0), textcoords="offset points", fontsize=6, ha="right", va="center")
S.ylabel(axb, r"$\Delta G_\mathrm{mix}$ (meV per site)")
C.legend([S.marker(S.SLATE, "x", ms=3.0, mew=0.8), S.line(S.INK, "x", ms=3.4, mew=1.0, mec=S.INK)] + [S.line(S.T_COLOR[t], "o", ms=3.2) for t in TS] + [S.marker(S.GUIDE, "o", open_=True, ms=3.2)],
         ["DFT", "0 K"] + [f"{t} K" for t in TS] + ["not equilibrated"], S.X3[1], Y1 + BAND - 2.6, loc="center left", ncol=6, handlelength=1.6, columnspacing=1.25)   # one row for b and c

# (d) FeNiCr short-range order vs T ---------------------------------------------------------------------------------------------------------
def sro(run):
    d = json.load(open(f"{RESULTS}/fenicr_sro/{run}/sro.json")); T = np.asarray(d["temps"], float); c = d["composition"]; w = c["Fe"] + c["Cr"]; o = np.argsort(T)
    avg = lambda a: ((c["Fe"] * np.asarray(a["Fe-Ni"]) + c["Cr"] * np.asarray(a["Cr-Ni"])) / w)[o]; return T[o], avg(d["alpha1"]), avg(d["alpha2"])
Tc_, c1, c2 = sro("fenicr_s2b_mm_c65_d1_dense_F6"); Ti_, i1, i2 = sro("fenicr_icet_heis_bench_dense_F6"); mc_, mi_ = Tc_ >= 495, Ti_ >= 495
SER = [(Tc_[mc_], c1[mc_], c2[mc_], "-", 1.5, NCE), (Ti_[mi_], i1[mi_], i2[mi_], (0, (3.5, 1.8)), 0.9, OCE + " + spin"),
       (np.asarray(PAPER6_T), np.asarray(PAPER6_A1), np.asarray(PAPER6_A2), (0, (1.2, 1.4)), 0.9, "Su et al.")]
for T_, a1_, a2_, ls, lw, lab in SER:
    axd.plot(T_, a1_, color=C1NN, lw=lw, ls=ls, solid_capstyle="butt"); axd.plot(T_, a2_, color=C2NN, lw=lw, ls=ls, solid_capstyle="butt")
EXP = dict(ls="none", ms=3.6, mec="black", mew=0.5, zorder=5)
axd.plot(F6_T, F6_A1, marker="s", mfc=C1NN, **EXP); axd.plot(F6_T, F6_A2, marker="v", mfc=C2NN, **EXP)
axd.text(1500, 0.115, "2NN", fontsize=6, color=C2NN, ha="right", va="bottom"); axd.text(1500, -0.105, "1NN", fontsize=6, color=C1NN, ha="right", va="top")
axd.set_xlim(450, 1550); axd.set_xticks([600, 900, 1200, 1500]); axd.set_ylim(-0.25, 0.8); axd.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2)); axd.set_xlabel("T (K)")
S.ylabel(axd, r"Warren–Cowley $\alpha$, (Fe,Cr)–Ni"); axd.text(0.96, 0.965, r"Fe$_{42.5}$Cr$_{7.5}$Ni$_{50}$", transform=axd.transAxes, fontsize=7, ha="right", va="top")
band_legend(axd, Y2, [Line2D([], [], color=S.INK, lw=lw, ls=ls) for *_, ls, lw, lab in SER]
            + [(Line2D([], [], marker="s", mfc=C1NN, **EXP), Line2D([], [], marker="v", mfc=C2NN, **EXP))], [lab for *_, lab in SER] + ["Experiment"],
            handlelength=2.3, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})

# (e, f) Shapley value vs alpha ------------------------------------------------------------------------------------------------------------------
Hh = json.load(open(f"{FD}/fig7_fenicr_shap2nn.json")); rows = [q for q in Hh["rows"] if q["label"] == "F6"]
Ts = np.array([q["T"] for q in rows]); norm = mpl.colors.Normalize(Ts.min(), Ts.max()); cm = S.T_CMAP; SLOPES = {}
for ax, sh, akey, xl in ((axe, "1NN", "a1", r"$\alpha_\mathrm{1NN}$, (Fe,Cr)–Ni"), (axf, "2NN", "a2", r"$\alpha_\mathrm{2NN}$, (Fe,Cr)–Ni")):
    xv = np.array([q[akey] for q in rows]); hl, ll = [], []
    for kind, g, filled in (("spin", f"spin pairs {sh}", True), ("chem", f"chem. pairs {sh}", False)):
        y = np.array([q[g] for q in rows]); y = y - y.mean(); ls_ = "-" if filled else (0, (3, 2))
        kf = np.polyfit(xv, y, 1); SLOPES[g] = float(kf[0]); xx = np.linspace(xv.min(), xv.max(), 10); ax.plot(xx, np.polyval(kf, xx), color="black", lw=0.8, ls=ls_, zorder=2)
        if filled: ax.scatter(xv, y, s=13, c=Ts, cmap=cm, norm=norm, edgecolors=S.INK, linewidths=0.25, zorder=3)
        else: ax.scatter(xv, y, s=11, facecolors="white", edgecolors=cm(norm(Ts)), linewidths=0.9, zorder=3)
        hl.append(Line2D([], [], color="black", lw=0.8, ls=ls_, marker="o", ms=3.4, mfc=S.GUIDE if filled else "white", mec=S.GUIDE, mew=0.9))
        ll.append(f"{'Spin' if kind == 'spin' else 'Chemical'} pairs  {kf[0]:+.0f}".replace("-", "−"))
    ax.legend(hl, ll, title="Slope", title_fontsize=6, loc="upper right", handlelength=2.0, borderaxespad=0.2)       # blank top-right corner of both panels
    ax.axhline(0, color=S.SLATE, lw=0.6, zorder=1); ax.set_xlabel(xl); ax.set_ylim(-40, 30); ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
S.ylabel(axe, "Shapley value (meV per atom)")
cax = C.ax((S.X3[1] + S.X3[2] + S.W3) / 2 - 15.0, Y2 + BAND - 3.3, 30.0, 1.2); cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, orientation="horizontal")   # one temperature scale for e and f
cb.outline.set_linewidth(0.5); cax.xaxis.set_ticks_position("top"); cb.set_ticks([400, 800, 1200]); cax.tick_params(labelsize=6, length=1.5, width=0.5, pad=1)
cax.text(-0.06, 0.5, "T (K)", transform=cax.transAxes, fontsize=6, ha="right", va="center")

S.check(C, "abcdef", "Fig. 5"); C.save(OUT)
fig.canvas.draw(); rd = fig.canvas.get_renderer()
for a_ in (axe, axf):                                                                    # no point under a slope legend
    b_ = a_.get_legend().get_window_extent(rd); pts = a_.transData.transform(np.vstack([c_.get_offsets() for c_ in a_.collections]))
    print("e/f: points under the slope legend:", int(np.sum((pts[:, 0] > b_.x0 - 3) & (pts[:, 0] < b_.x1 + 3) & (pts[:, 1] > b_.y0 - 3) & (pts[:, 1] < b_.y1 + 3))))
print("slopes:", {k: round(v, 1) for k, v in SLOPES.items()}, "| swap at which the orthonormal cluster expansion leaves the band:", k_out, "| end values:", round(float(ei)), round(float(ec)), round(float(e_dft)))
