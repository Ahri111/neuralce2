"""Fig. 5 (Nature format, 183 x 124 mm), drawn from stored data only.
(a) SrTi0.5Fe0.5O2.75: energy along the swap paths from the DFT ground state to the state the Monte Carlo of the orthonormal cluster expansion
    falls into (line = mean over the paths, band = 25-75 % quantile); grey band = energy range of the DFT training structures at x = 0.5.
(b, c) Delta G_mix, orthonormal cluster expansion | neural cluster expansion (K = 1); open markers: entropy (H - G)/T outside [0, ideal] or G above the 0 K energy.
(d) FeNiCr Warren-Cowley alpha vs T (Monte Carlo of both models, Su et al., experiment); colour = shell (named on the curves), line style = source,
    the neural cluster expansion drawn heavier.
Model names are written out in full everywhere, and no legend is placed over data.
(e, f) Shapley value of the spin-pair and chemical-pair channels vs alpha_1NN / alpha_2NN (F6 rows); one point per temperature,
    the legend gives the slope of each regression line (meV per atom per unit alpha).

Inputs:  results/figure_data/fig7_stfo_extrap.json (scripts/analysis/stfo_extrapolation.py), results/figure_data/fig_gmix_stfo.json,
         results/figure_data/fig7_fenicr_shap2nn.json (scripts/analysis/fenicr_shap_2nn.py), results/fenicr_sro/<tag>_dense_F6/sro.json,
         data/new_bench_aug467/detailed_info.csv (DFT mixing energies); experimental SRO values from scripts/figures/fenicr_experiment.py.
Outputs: figures/fig5_stf_fenicr.{pdf,svg,png,json}, figures/SourceData_Fig5.xlsx (a-f).
Run:     python scripts/figures/fig5_stf_fenicr.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above."""
import os, sys, json, numpy as np, pandas as pd
from math import lgamma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import natstyle as ns
from neuralce.paths import REPO, DATA, RESULTS
import matplotlib as mpl; mpl.use("Agg"); FAM = ns.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from fenicr_experiment import F6_T, F6_A1, F6_A2, PAPER6_T, PAPER6_A1, PAPER6_A2

D = f"{REPO}/figures"; os.makedirs(D, exist_ok=True); FD = f"{RESULTS}/figure_data"; PURPLE, ICET, DFTC, GREEN, ORANGE = "#5b2a9e", "#3a3a3a", "#9a9a9a", "#1b7837", "#d95f02"
def require(paths, hint):                                                                 # the STFO / Fe-Ni-Cr inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{FD}/fig7_stfo_extrap.json", f"{FD}/fig_gmix_stfo.json", f"{FD}/fig7_fenicr_shap2nn.json", f"{RESULTS}/fenicr_sro", f"{DATA}/new_bench_aug467/detailed_info.csv"],
        "$NCE_RESULTS/figure_data, $NCE_RESULTS/fenicr_sro, $NCE_DATA/new_bench_aug467")
W, H = 183.0, 124.0; fig = plt.figure(figsize=(W / 25.4, H / 25.4))                         # 124 mm: >= 2 mm blank at every edge, >= 4 mm between the rows
def add_ax(x, y, w, h, **kw): return fig.add_axes([x / W, y / H, w / W, h / H], **kw)
COL = ((16.0, 72.0), (100.5, 30.5), (141.5, 30.5))                                          # one column grid for both rows: a/d, b/e, c/f start at the same x and share a width
YB, YT, HA = 11.5, 72.5, 42.0                                                               # bottoms of the two rows and the axes height (mm): 19 mm between the rows
axa, axb, axc = (add_ax(x_, YT, w_, HA) for x_, w_ in COL)
axd, axe = add_ax(COL[0][0], YB, COL[0][1], HA), add_ax(COL[1][0], YB, COL[1][1], HA)
axf = add_ax(COL[2][0], YB, COL[2][1], HA, sharey=axe); cax = add_ax(173.5, YB, 1.5, HA)    # colour-bar labels end 2 mm inside the canvas
LET = [fig.text(x / W, y / H, s, fontsize=8, fontweight="bold", ha="left", va="bottom")
       for s, (x, y) in zip("abcdef", ((2, YT + HA + 2), (92.5, YT + HA + 2), (133.5, YT + HA + 2), (2, YB + HA + 2), (92.5, YB + HA + 2), (133.5, YB + HA + 2)))]
SD = {}

# (a) swap paths -----------------------------------------------------------------------------------------------------------------------------
X = json.load(open(f"{FD}/fig7_stfo_extrap.json")); te = np.array([t[1] for t in X["train"]]); nstep = len(X["paths"][0]) - 1; xs = np.arange(nstep + 1)
axa.axhspan(te.min(), te.max(), color="#000000", alpha=0.06, lw=0, zorder=0)
# the grey band (energy range of the DFT training structures at x = 0.5) is named in the caption, not in the panel
YP = {}
for key, col in (("icet", ICET), ("conv", PURPLE)):
    Y = np.array([[r[key] for r in p] for p in X["paths"]]); YP[key] = Y
    axa.fill_between(xs, np.quantile(Y, 0.25, axis=0), np.quantile(Y, 0.75, axis=0), color=col, alpha=0.15, lw=0, zorder=2)
    axa.plot(xs, Y.mean(0), color=col, lw=0.75, marker="o", ms=2.5, mfc=col, mec=col, zorder=4)
e_dft = [e for k, (v, e) in X["dft"].items() if k != "A00_500"][0]; ei, ec = YP["icet"].mean(0)[-1], YP["conv"].mean(0)[-1]
axa.plot(0, 0, ls="none", marker="*", ms=6, color="k", mec="k", zorder=6); axa.plot(nstep, e_dft, ls="none", marker="*", ms=6, color="#808080", mec="#808080", zorder=6)
axa.annotate(f"{ei:.0f}".replace("-", "−"), (nstep, ei), xytext=(5, 0), textcoords="offset points", fontsize=6, color=ICET, ha="left", va="center")   # end values only, right of the last point (off the curve and its band)
axa.annotate(f"{e_dft:+.0f}", (nstep, e_dft), xytext=(5, 0), textcoords="offset points", fontsize=6, color="#808080", ha="left", va="center")
axa.annotate(f"{ec:+.0f}", (nstep, ec), xytext=(0, -4), textcoords="offset points", fontsize=6, color=PURPLE, ha="right", va="top")
k_out = int(np.argmax(YP["icet"].mean(0) < te.min()))                                  # first swap at which the icet curve leaves the band downwards
axa.annotate("", (k_out, YP["icet"].mean(0)[k_out]), xytext=(0, 5.0 / 25.4 * 72), textcoords="offset points", arrowprops=dict(arrowstyle="-|>", color="k", lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=2))
axa.set_xlim(-0.5, nstep + 1.7); axa.set_xticks(range(0, nstep + 1, 2)); axa.set_ylim(-300, 60)   # 1.7 swaps of blank on the right: the end values stay inside the axes
axa.set_xlabel("Swaps away from the DFT ground state"); axa.set_ylabel("Energy relative to the DFT\nground state (meV per site)", linespacing=1.2)
axa.text(0.01, 0.995, r"SrTi$_{0.5}$Fe$_{0.5}$O$_{2.75}$", transform=axa.transAxes, fontsize=7, ha="left", va="top")
axa.legend([Line2D([], [], color=PURPLE, lw=0.75, marker="o", ms=2.5), Line2D([], [], color=ICET, lw=0.75, marker="o", ms=2.5), Line2D([], [], ls="none", marker="*", ms=6, color="k"),
            Line2D([], [], ls="none", marker="*", ms=6, color="#808080")], ["Neural cluster expansion", "Orthonormal cluster expansion", "DFT ground state", "DFT of the final Monte Carlo state"],
           loc="lower left", frameon=False, fontsize=6, handlelength=1.8, handletextpad=0.5, labelspacing=0.3)
SD["a"] = pd.DataFrame({"Swap": xs, "Orthonormal cluster expansion, mean (meV per site)": YP["icet"].mean(0), "Orthonormal cluster expansion, q25": np.quantile(YP["icet"], 0.25, 0),
                        "Orthonormal cluster expansion, q75": np.quantile(YP["icet"], 0.75, 0), "Neural cluster expansion, mean (meV per site)": YP["conv"].mean(0),
                        "Neural cluster expansion, q25": np.quantile(YP["conv"], 0.25, 0), "Neural cluster expansion, q75": np.quantile(YP["conv"], 0.75, 0)})

# (b, c) Delta G_mix ---------------------------------------------------------------------------------------------------------------------------
S = json.load(open(f"{FD}/fig_gmix_stfo.json")); assert int(S["K"]) == 1
E_STO, E_SFO = -1209.4315 / 160, -929.75272 / 160; ref = lambda x: (1 - x) * E_STO + x * E_SFO
dfb = pd.read_csv(f"{DATA}/new_bench_aug467/detailed_info.csv"); xd = dfb.comp / 1000.0; Hd = (dfb.total_energy / dfb.n_atoms - ref(xd)) * 1e3
def s_ideal(x):
    nFe, nV = round(32 * x), round(16 * x); return ((lgamma(33) - lgamma(nFe + 1) - lgamma(33 - nFe)) + (lgamma(97) - lgamma(nV + 1) - lgamma(97 - nV))) / 160.0
TS = [900, 1200, 1500]; VIR = [mpl.colors.to_hex(mpl.colormaps["viridis"](v)) for v in np.linspace(0.15, 0.85, len(TS))]
for ax, fw, ylim, lab in ((axb, "icet", (-790, 100), "Orthonormal cluster\nexpansion"), (axc, "conv", (-20, 63), "Neural cluster\nexpansion")):   # b reaches -505: the space below it holds the legend, 3 mm clear of the minimum
    rr = sorted(S["results"][fw], key=lambda r: r["x"]); xg = [0.0] + [r["x"] for r in rr] + [1.0]; h0 = [0.0] + [r["H0"] for r in rr] + [0.0]
    ax.plot(xd, Hd, ls="none", marker="x", ms=3, color=DFTC, mew=0.6, zorder=2)
    ax.plot(xg, h0, color=ICET, lw=0.75, marker="x", ms=3, mew=0.8, zorder=5); out = {"Fe fraction x": xg, "0 K": h0}
    for t, col in zip(TS, VIR):
        g = [0.0] + [r[f"G{t}"] for r in rr] + [0.0]; ok = [True] + [(-1e-6 <= r[f"S{t}"] <= s_ideal(r["x"]) + 1e-6) and r[f"G{t}"] <= r["H0"] + 1e-6 for r in rr] + [True]
        ax.plot(xg, g, color=col, lw=0.75, zorder=4)
        for xx, gg, o in zip(xg, g, ok): ax.plot(xx, gg, ls="none", marker="o", ms=2.5, mfc=col if o else "white", mec=col, mew=0.6, zorder=6)
        out[f"{t} K"] = g; out[f"{t} K equilibrated"] = ok
    SD["b" if fw == "icet" else "c"] = pd.DataFrame(out)
    ax.axhline(0, color="0.6", lw=0.5, zorder=1); ax.set_xlim(-0.03, 1.03); ax.set_ylim(*ylim); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(r"Fe fraction $\mathit{x}$"); ax.set_title(lab, fontsize=7, pad=6.0, linespacing=1.15)          # 2 mm above the axes, and the DFT points stop >= 2 mm below the axes top: text and data stay apart
    if fw == "icet":
        j = int(np.argmin([r["H0"] for r in rr])); ax.annotate("−505", (rr[j]["x"], rr[j]["H0"]), xytext=(-4, 0), textcoords="offset points", fontsize=6, ha="right", va="center")
axb.set_ylabel(r"$\Delta G_\mathrm{mix}$ (meV per site)")
axb.legend([Line2D([], [], ls="none", marker="x", ms=3, color=DFTC, mew=0.6), Line2D([], [], color=ICET, lw=0.75, marker="x", ms=3, mew=0.8)]
           + [Line2D([], [], color=c_, lw=0.75, marker="o", ms=2.5) for c_ in VIR] + [Line2D([], [], ls="none", marker="o", ms=2.5, mfc="white", mec="0.45", mew=0.6)],
           ["DFT", "0 K"] + [f"{t} K" for t in TS] + ["not equilibrated"], loc="lower left", ncol=2, borderaxespad=0.6, frameon=False, fontsize=6, handlelength=1.5,
           handletextpad=0.5, labelspacing=0.25, columnspacing=1.0)                        # in the blank strip below the -505 meV minimum, so no curve enters it

# (d) FeNiCr SRO vs T ----------------------------------------------------------------------------------------------------------------------------
def sro(run):
    d = json.load(open(f"{RESULTS}/fenicr_sro/{run}/sro.json")); T = np.asarray(d["temps"], float); c = d["composition"]; w = c["Fe"] + c["Cr"]; o = np.argsort(T)
    avg = lambda a: ((c["Fe"] * np.asarray(a["Fe-Ni"]) + c["Cr"] * np.asarray(a["Cr-Ni"])) / w)[o]; return T[o], avg(d["alpha1"]), avg(d["alpha2"])
Tc_, c1, c2 = sro("fenicr_s2b_mm_c65_d1_dense_F6"); Ti_, i1, i2 = sro("fenicr_icet_heis_bench_dense_F6"); mc_, mi_ = Tc_ >= 495, Ti_ >= 495
SER = [(Tc_[mc_], c1[mc_], c2[mc_], "-", "Neural cluster expansion"), (Ti_[mi_], i1[mi_], i2[mi_], (0, (4, 2)), "Orthonormal cluster expansion + spin"),
       (np.asarray(PAPER6_T), np.asarray(PAPER6_A1), np.asarray(PAPER6_A2), (0, (1.5, 1.5)), "Su et al.")]
LWD = {"Neural cluster expansion": 1.2}                                                       # this work drawn heavier than the two references
for T_, a1_, a2_, ls, lab in SER:
    axd.plot(T_, a1_, color=GREEN, lw=LWD.get(lab, 0.75), ls=ls); axd.plot(T_, a2_, color=ORANGE, lw=LWD.get(lab, 0.75), ls=ls)
axd.plot(F6_T, F6_A1, ls="none", marker="s", ms=3, mfc=GREEN, mec="k", mew=0.5, zorder=5); axd.plot(F6_T, F6_A2, ls="none", marker="v", ms=3, mfc=ORANGE, mec="k", mew=0.5, zorder=5)
hsty = [Line2D([], [], color="0.35", lw=LWD.get(lab, 0.75), ls=ls) for *_, ls, lab in SER] + [(Line2D([], [], ls="none", marker="s", ms=3, mfc=GREEN, mec="k", mew=0.5),
                                                                                              Line2D([], [], ls="none", marker="v", ms=3, mfc=ORANGE, mec="k", mew=0.5))]   # both experiment markers: square 1NN, triangle 2NN
axd.legend(hsty, [lab for *_, lab in SER] + ["Experiment"], ncol=1, loc="upper right", frameon=False, fontsize=6, handlelength=2.2,
           handletextpad=0.5, labelspacing=0.25, borderaxespad=0.3, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})   # line style = source; the top right is the only blank corner
axd.text(1480, 0.105, "2NN", fontsize=6, color=ORANGE, ha="right", va="bottom"); axd.text(1480, -0.095, "1NN", fontsize=6, color=GREEN, ha="right", va="top")   # colour = shell, named on the curves
AT = {lab: dict(a1=[float(np.interp(t, T_, a1_)) for t in F6_T], a2=[float(np.interp(t, T_, a2_)) for t in F6_T]) for T_, a1_, a2_, _, lab in SER}   # values at the experimental temperatures
AT["Experiment"] = dict(a1=list(F6_A1), a2=list(F6_A2))
axd.set_xlim(450, 1550); axd.set_xticks([600, 900, 1200, 1500]); axd.set_ylim(-0.25, 0.8); axd.set_xlabel("T (K)"); axd.set_ylabel(r"Warren–Cowley $\alpha$, (Fe,Cr)–Ni")
axd.text(0.02, 0.97, r"Fe$_{42.5}$Cr$_{7.5}$Ni$_{50}$", transform=axd.transAxes, fontsize=7, ha="left", va="top")
SD["d"] = pd.concat([pd.DataFrame({"Model": lab, "T (K)": T_, "alpha 1NN": a1_, "alpha 2NN": a2_}) for T_, a1_, a2_, _, lab in SER] + [pd.DataFrame({"Model": "Experiment", "T (K)": F6_T, "alpha 1NN": F6_A1, "alpha 2NN": F6_A2})])

# (e, f) Shapley value vs alpha ----------------------------------------------------------------------------------------------------------------------
Hh = json.load(open(f"{FD}/fig7_fenicr_shap2nn.json")); rows = [q for q in Hh["rows"] if q["label"] == "F6"]
Ts = np.array([q["T"] for q in rows]); norm = mpl.colors.Normalize(Ts.min(), Ts.max()); cm = mpl.colormaps["coolwarm"]; SLOPES = {}
for ax, sh, akey, xl in ((axe, "1NN", "a1", r"$\alpha_\mathrm{1NN}$, (Fe,Cr)–Ni"), (axf, "2NN", "a2", r"$\alpha_\mathrm{2NN}$, (Fe,Cr)–Ni")):
    xv = np.array([q[akey] for q in rows]); out = {"T (K)": Ts, f"alpha {sh}": xv}; hl, ll = [], []
    for kind, g, filled in (("spin", f"spin pairs {sh}", True), ("chem", f"chem. pairs {sh}", False)):
        y = np.array([q[g] for q in rows]); y = y - y.mean(); out[f"{g} (meV per atom)"] = y; ls_ = "-" if filled else (0, (3, 2))
        if filled: ax.scatter(xv, y, s=9, c=Ts, cmap=cm, norm=norm, edgecolors="none", zorder=3)
        else: ax.scatter(xv, y, s=9, facecolors="white", edgecolors=cm(norm(Ts)), linewidths=0.6, zorder=3)
        kf = np.polyfit(xv, y, 1); SLOPES[g] = float(kf[0]); xx = np.linspace(xv.min(), xv.max(), 10); ax.plot(xx, np.polyval(kf, xx), color="k", lw=0.6, ls=ls_, zorder=2)
        hl.append(Line2D([], [], color="k", lw=0.6, ls=ls_, marker="o", ms=3, mfc="0.45" if filled else "white", mec="0.45", mew=0.6))
        ll.append(f"{'Spin' if kind == 'spin' else 'Chemical'} pairs  {kf[0]:+.0f}".replace("-", "−"))
    ax.legend(hl, ll, title="Slope", title_fontsize=6, loc="upper right", frameon=False, fontsize=6, handlelength=1.6, handletextpad=0.5, labelspacing=0.25, borderaxespad=0.3)   # blank top-right corner
    ax.axhline(0, color="0.6", lw=0.5, zorder=1); ax.set_xlabel(xl); SD["e" if sh == "1NN" else "f"] = pd.DataFrame(out)
axe.set_ylim(-40, 30); axe.set_ylabel("Shapley value (meV per atom)"); axf.tick_params(labelleft=False)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax); cb.outline.set_linewidth(0.4); cb.set_ticks([400, 800, 1200])
cax.tick_params(labelsize=6, length=1.5, width=0.4, pad=1); cax.set_title("T (K)", fontsize=6, pad=3)

fig.canvas.draw(); _r = fig.canvas.get_renderer(); _mm = lambda px: px / fig.dpi * 25.4
_gap = _mm(min(a_.xaxis.label.get_window_extent(_r).y0 for a_ in (axa, axb, axc)) - max([t_.get_window_extent(_r).y1 for t_ in LET[3:]] + [cax.title.get_window_extent(_r).y1]))
_lb = axb.get_legend().get_window_extent(_r); _mn = axb.transData.transform((0.75, -505.0))[1]
for a_, n_ in ((axb, "b"), (axc, "c")):
    top_ = max(float(a_.transData.transform(np.column_stack([np.asarray(l_.get_xdata(), float), np.asarray(l_.get_ydata(), float)]))[:, 1].max()) for l_ in a_.lines if len(l_.get_xdata()) and l_.get_transform() == a_.transData)
    print(f"{n_}: blank between the highest data point and the title {_mm(a_.title.get_window_extent(_r).y0 - top_):.2f} mm", flush=True)
print(f"canvas {W:.0f} x {H:.0f} mm | margins (left, right, bottom, top) mm: {ns.margins(fig)} | blank between rows: {_gap:.2f} mm | b: legend top to the -505 minimum {_mm(_mn - _lb.y1):.2f} mm", flush=True)
fig.savefig(f"{D}/fig5_stf_fenicr.pdf"); fig.savefig(f"{D}/fig5_stf_fenicr.svg"); fig.savefig(f"{D}/fig5_stf_fenicr.png", dpi=600); print("wrote fig5_stf_fenicr.pdf / .svg / .png", flush=True)
with pd.ExcelWriter(f"{D}/SourceData_Fig5.xlsx", engine="openpyxl") as xw:
    for k in "abcdef": SD[k].to_excel(xw, sheet_name=k, index=False)
json.dump(dict(slopes=SLOPES, n_paths=len(X["paths"]), k_leaves_band=k_out, endpoints=dict(icet=float(ei), conv=float(ec), dft=float(e_dft)), alpha_at_experiment_T=dict(T=list(F6_T), **AT)),
          open(f"{D}/fig5_stf_fenicr.json", "w"), indent=1)
print("d, alpha at", list(F6_T), "K:", {k: {q: [round(v, 3) for v in w] for q, w in d_.items()} for k, d_ in AT.items()})
for q_ in ("a1", "a2"): print(f"d, mean |model - experiment| {q_}:", {k: round(float(np.mean([abs(a - b) for a, b in zip(AT[k][q_], AT['Experiment'][q_])])), 3) for k in AT if k != "Experiment"})
_le = [a_.get_legend().get_window_extent(fig.canvas.get_renderer()) for a_ in (axe, axf)]
for a_, b_ in zip((axe, axf), _le):
    pts = a_.transData.transform(np.vstack([c_.get_offsets() for c_ in a_.collections])); hit = int(np.sum((pts[:, 0] > b_.x0 - 3) & (pts[:, 0] < b_.x1 + 3) & (pts[:, 1] > b_.y0 - 3) & (pts[:, 1] < b_.y1 + 3)))
    print("e/f: points under the slope legend:", hit)
sz, bad = ns.text_check(fig, "abcdef"); print("font:", FAM, "| text sizes:", sz, "| violations:", bad)
fig.canvas.draw(); r_ = fig.canvas.get_renderer()
ghost = set()                                                                              # tick labels of ticks outside the view limits: matplotlib never draws them
for ax_ in fig.axes:
    for axis_, (lo_, hi_) in ((ax_.xaxis, sorted(ax_.get_xlim())), (ax_.yaxis, sorted(ax_.get_ylim()))):
        for tk in list(axis_.get_major_ticks()) + list(axis_.get_minor_ticks()):
            if not lo_ - 1e-9 <= tk.get_loc() <= hi_ + 1e-9: ghost |= {id(tk.label1), id(tk.label2)}
bb_ = lambda t: t.get_window_extent(r_)
off = [(t.get_text(), round(bb_(t).x0 / fig.dpi * 25.4, 1), round(bb_(t).x1 / fig.dpi * 25.4, 1), round(bb_(t).y0 / fig.dpi * 25.4, 1)) for t in fig.findobj(mpl.text.Text)
       if t.get_visible() and t.get_text().strip() and id(t) not in ghost
       and (bb_(t).x1 > fig.bbox.x1 + 0.5 or bb_(t).x0 < -0.5 or bb_(t).y1 > fig.bbox.y1 + 0.5 or bb_(t).y0 < -0.5)]
print("text outside the canvas (text, x0, x1, y0 in mm):", off)
