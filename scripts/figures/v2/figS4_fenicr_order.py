"""Supplementary Fig. S4, v2 style (same data and numbers as scripts/figures/figS4_fenicr_order.py): where the second-neighbour ordering energy of
Fe-Ni-Cr comes from, in DFT and in the two models.
(a) with magnetism, (b) without magnetism: two 44 x 36 mm panels; (c) the slope of panel a split into the cluster channels of each model (horizontal bars, same height).
Lines = least-squares fits through the random cells; the L1_0 cells (open markers) are not in the fits.
Input:   results/shap/fenicr_order_energy.json.
Output:  figures/v2/figS4_fenicr_order.{pdf,svg,png}.
Run:     python scripts/figures/v2/figS4_fenicr_order.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import style as S
from neuralce.paths import REPO, RESULTS
import matplotlib as mpl; mpl.use("Agg"); FAM = S.use()
from matplotlib.lines import Line2D
import channel_labels as CL

OUTDIR = f"{REPO}/figures/v2"; os.makedirs(OUTDIR, exist_ok=True); OUT = f"{OUTDIR}/figS4_fenicr_order"
if not os.path.exists(f"{RESULTS}/shap/fenicr_order_energy.json"):                       # the Fe-Ni-Cr results are not shipped: skip cleanly
    print(f"[skip] v2/{os.path.basename(__file__)}: {RESULTS}/shap/fenicr_order_energy.json is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_RESULTS/shap to run", flush=True); sys.exit(0)
J = json.load(open(f"{RESULTS}/shap/fenicr_order_energy.json"))
A = sorted([r for r in J["rows"] if r["set"] == "A"], key=lambda r: (r["kind"] != "random", r["a2"], r["name"])); RND = [r for r in A if r["kind"] == "random"]; L10 = [r for r in A if r["kind"] != "random"]
NCE, OCE = "Neural cluster expansion", "Orthonormal cluster expansion + spin"
STY = {"DFT": dict(color="black", marker="D", ms=3.1, ls="-"), NCE: dict(color=S.BLUE, marker="o", ms=3.6, ls="-"), OCE: dict(color=S.NAVY, marker="s", ms=3.2, ls=S.DASH)}
def series(mag):
    return {"DFT": [r["E_dft_spin" if mag else "E_dft_nm"] for r in A], NCE: [r["models"][NCE]["E_meV"] if mag else r["models_spin_off"][NCE] for r in A],
            OCE: [r["models"][OCE]["E_meV"] if mag else r["models_spin_off"][OCE] for r in A]}

BAND = S.STRIP + S.LINE; (Y0,), H = S.rows(1, strips=[BAND], foots=[11.5]); C = S.Canvas(H); fig = C.fig       # the x title of c has two lines
axa, axb = C.ax(S.X3[0], Y0 + BAND, S.W3), C.ax(S.X3[1], Y0 + BAND, S.W3); XC, WC = 156.0, 24.0; axc = C.ax(XC, Y0 + BAND, WC)   # c: the label column takes the rest of the 60 mm slot, so every channel name stays on two lines
a2 = np.array([r["a2"] for r in A]); nr = len(RND); xx = np.array([-0.62, 0.79]); JIT = (-0.04, 0.0, 0.04); SLOPE = {}
for ax, mag, key, ttl in ((axa, True, "a", "With magnetism"), (axb, False, "b", "Without magnetism")):
    Sr = series(mag); hl, ll = [], []
    for name, st in STY.items():
        e = np.array(Sr[name], float); e = e - e[:nr].mean(); k = np.polyfit(a2[:nr], e[:nr], 1); SLOPE[(key, name)] = float(k[0])
        ax.plot(xx, np.polyval(k, xx), color=st["color"], lw=S.LW, ls=st["ls"], zorder=3)
        ax.plot(a2[:nr], e[:nr], ls="none", marker=st["marker"], ms=st["ms"], mfc=st["color"], mew=0, zorder=4)
        ax.plot(a2[nr:] + np.array(JIT[:len(L10)]), e[nr:], ls="none", marker=st["marker"], ms=st["ms"], mfc="white", mec=st["color"], mew=0.9, zorder=4)   # the three L1_0 cells, side by side
        hl.append(Line2D([], [], color=st["color"], lw=S.LW, ls=st["ls"], marker=st["marker"], ms=st["ms"], mfc=st["color"], mew=0)); ll.append(f"{k[0]:+.0f}".replace("-", "−"))
    ax.legend(hl, ll, title="Slope", title_fontsize=6, loc="lower left" if mag else "upper left", handlelength=2.6, borderaxespad=0.3)
    ax.axhline(0, color=S.SLATE, lw=0.6, zorder=1); ax.set_xlim(-0.72, 1.12); ax.set_xticks([-0.5, 0.0, 0.5, 1.0]); ax.set_ylim(-58, 58); ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.set_xlabel(r"$\alpha_\mathrm{2NN}$, (Fe,Cr)–Ni"); C.text(ax._mm[0], Y0 + S.LINE1, ttl); C.letter(ax, Y0 + S.LINE1, key)
S.ylabel(axa, "Energy relative to the mean\nof the random cells (meV per atom)", linespacing=1.15)

# (c) the slope of panel a, channel by channel
GR = [g for g in J["models"][NCE]["groups"] if g != "singlets"]
sl = {m: {g: (float(np.polyfit(a2[:nr], [r["models"][m]["phi"][g] for r in RND], 1)[0]) if g in J["models"][m]["groups"] else np.nan) for g in GR} for m in (NCE, OCE)}
GR = sorted(GR, key=lambda g: -max(abs(sl[NCE][g]), abs(np.nan_to_num(sl[OCE][g])))); yy = np.arange(len(GR)); hb = 0.36
axc.barh(yy - hb / 2, [sl[NCE][g] for g in GR], height=hb, color=S.BLUE, ec="black", lw=0.5, zorder=3)
axc.barh(yy + hb / 2, [np.nan_to_num(sl[OCE][g]) for g in GR], height=hb, color=S.NAVY, ec="black", lw=0.5, zorder=3)
axc.axvline(0, color="black", lw=0.7, zorder=4); axc.set_yticks(yy); axc.spines["left"].set_visible(False)
axc.set_yticklabels([S.fit_label(fig, CL.label("fenicr", g), XC - 1.5 - (S.X3[1] + S.W3 + 1.5), CL.wrapped, CL.wrap_to) for g in GR], linespacing=1.0)
axc.tick_params(axis="y", length=0, pad=3); axc.set_ylim(len(GR) - 0.45, -0.55); axc.set_xlim(-24, 9); axc.set_xticks([-20, -10, 0])
xl = axc.set_xlabel("Contribution to the slope in a\n" + r"(meV per atom per unit $\alpha_\mathrm{2NN}$)", linespacing=1.15); C.text(XC, Y0 + S.LINE1, "By cluster channel")
fig.canvas.draw(); half = xl.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4 / 2                    # the two-line title is wider than the axes: keep it 2 mm inside the canvas
axc.xaxis.set_label_coords((min(XC + WC / 2, C.W - S.EDGE - 0.3 - half) - XC) / WC, -(S.FOOT_PLAIN - 2.6 - 0.3) / S.HA); xl.set_va("top"); C.letter(S.X3[2] - S.LETTER_DX, Y0 + S.LINE1, "c")

hd = [Line2D([], [], color=st["color"], lw=S.LW, ls=st["ls"], marker=st["marker"], ms=st["ms"], mfc=st["color"], mew=0) for st in STY.values()] + [S.marker(S.GUIDE, "o", open_=True)]
C.legend(hd, list(STY) + [r"L1$_0$-ordered cells (not in the fits)"], S.X3[0], Y0 + S.LINE1 + S.LINE, loc="center left", ncol=4, handlelength=2.6, columnspacing=1.8)
S.check(C, "abc", "Fig. S4"); C.save(OUT)
rd = fig.canvas.get_renderer()
for ax in (axa, axb):                                                                       # no marker under a slope legend
    bb = ax.get_legend().get_window_extent(rd); pts = np.vstack([ax.transData.transform(np.column_stack([np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float)])) for l in ax.lines if len(l.get_xdata()) > 2])
    print("markers under the slope legend:", int(np.sum((pts[:, 0] > bb.x0 - 3) & (pts[:, 0] < bb.x1 + 3) & (pts[:, 1] > bb.y0 - 3) & (pts[:, 1] < bb.y1 + 3))))
print("slopes (meV per atom per unit alpha_2NN):", {f"{k} {n}": round(v, 1) for (k, n), v in SLOPE.items()})
print("channel slopes:", {m: {g: round(v, 1) for g, v in d.items()} for m, d in sl.items()})
