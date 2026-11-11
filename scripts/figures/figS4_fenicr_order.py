"""Supplementary Fig. S4 (Nature format, 183 x 72 mm): where the second-neighbour ordering energy of Fe-Ni-Cr comes from, in DFT and in the two models.
Data: results/shap/fenicr_order_energy.json (scripts/analysis/fenicr_order_energy.py) - 13 DFT cells of 32 atoms (Fe14 Ni16 Cr2: 10 random cells spanning
alpha_2NN and 3 L1_0-ordered cells), computed after both models were fitted (held out).  Energies relative to the mean of the 10 random cells of the same series.
(a) with magnetism: spin-polarised DFT; models evaluated with the spin labels of the DFT moments.
(b) without magnetism: non-magnetic DFT; models evaluated with every spin label set to 0.
(c) the slope of a split into the cluster channels of each model (exact Shapley values over the channel groups; the two models share the cluster space
    and the spin-pair planes, so the groups are the same; the point clusters contribute < 0.1 and are left out).
Lines = least-squares fits through the random cells; the L1_0 cells (open markers) are not in the fits.
Inputs:  results/shap/fenicr_order_energy.json.
Outputs: figures/figS4_fenicr_order.{pdf,svg,png,json}, figures/SourceData_FigS4.xlsx (a, b, c).
Run:     python scripts/figures/figS4_fenicr_order.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import natstyle as ns
from neuralce.paths import REPO, RESULTS
import matplotlib as mpl; mpl.use("Agg"); FAM = ns.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import channel_labels as CL

D = f"{REPO}/figures"; os.makedirs(D, exist_ok=True)
if not os.path.exists(f"{RESULTS}/shap/fenicr_order_energy.json"):                       # the Fe-Ni-Cr results are not shipped: skip cleanly
    print(f"[skip] {os.path.basename(__file__)}: {RESULTS}/shap/fenicr_order_energy.json is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_RESULTS/shap to run", flush=True); sys.exit(0)
J = json.load(open(f"{RESULTS}/shap/fenicr_order_energy.json"))
A = sorted([r for r in J["rows"] if r["set"] == "A"], key=lambda r: (r["kind"] != "random", r["a2"], r["name"])); RND = [r for r in A if r["kind"] == "random"]; L10 = [r for r in A if r["kind"] != "random"]
NCE, OCE = "Neural cluster expansion", "Orthonormal cluster expansion + spin"; PURPLE, ICET = "#5b2a9e", "#3a3a3a"
STY = {"DFT": dict(color="k", marker="D", ms=2.8, ls="-"), NCE: dict(color=PURPLE, marker="o", ms=3.0, ls="-"), OCE: dict(color=ICET, marker="s", ms=2.8, ls=(0, (4, 2)))}
def series(mag):
    """name -> energies of the 13 cells (meV per atom), with magnetism (DFT spin-polarised, models with the DFT spin labels) or without (non-magnetic DFT, spin labels 0)"""
    return {"DFT": [r["E_dft_spin" if mag else "E_dft_nm"] for r in A], NCE: [r["models"][NCE]["E_meV"] if mag else r["models_spin_off"][NCE] for r in A],
            OCE: [r["models"][OCE]["E_meV"] if mag else r["models_spin_off"][OCE] for r in A]}

W, H = 183.0, 72.0; fig = plt.figure(figsize=(W * ns.MM, H * ns.MM))
def add_ax(x, y, w, h, **kw): return fig.add_axes([x / W, y / H, w / W, h / H], **kw)
Y0, HA = 14.0, 44.0; axa = add_ax(15.0, Y0, 47.0, HA); axb = add_ax(67.0, Y0, 47.0, HA, sharey=axa); axc = add_ax(152.0, Y0, 28.0, HA)
a2 = np.array([r["a2"] for r in A]); nr = len(RND); xx = np.array([-0.62, 0.79]); JIT = (-0.035, 0.0, 0.035); SLOPE, SD = {}, {}
for ax, mag, key, ttl in ((axa, True, "a", "With magnetism"), (axb, False, "b", "Without magnetism")):
    S = series(mag); out = {"Cell": [r["name"] for r in A], "Kind": [r["kind"] for r in A], "alpha 2NN": a2}; hl, ll = [], []
    for name, st in STY.items():
        e = np.array(S[name], float); e = e - e[:nr].mean(); k = np.polyfit(a2[:nr], e[:nr], 1); SLOPE[(key, name)] = float(k[0]); out[f"{name} (meV per atom)"] = e
        ax.plot(xx, np.polyval(k, xx), color=st["color"], lw=0.75, ls=st["ls"], zorder=3)
        ax.plot(a2[:nr], e[:nr], ls="none", marker=st["marker"], ms=st["ms"], mfc=st["color"], mec=st["color"], mew=0.5, zorder=4)
        ax.plot(a2[nr:] + np.array(JIT[:len(L10)]), e[nr:], ls="none", marker=st["marker"], ms=st["ms"], mfc="white", mec=st["color"], mew=0.6, zorder=4)   # the three L1_0 cells, side by side
        hl.append(Line2D([], [], color=st["color"], lw=0.75, ls=st["ls"], marker=st["marker"], ms=st["ms"], mfc=st["color"], mec=st["color"])); ll.append(f"{k[0]:+.0f}".replace("-", "−"))
    ax.legend(hl, ll, title="Slope", title_fontsize=6, loc="lower left" if mag else "upper left", frameon=False, fontsize=6, handlelength=2.6, handletextpad=0.5, labelspacing=0.25, borderaxespad=0.3)
    ax.axhline(0, color="0.6", lw=0.5, zorder=1); ax.set_xlim(-0.72, 1.12); ax.set_xticks([-0.5, 0.0, 0.5, 1.0]); ax.set_xlabel(r"$\alpha_\mathrm{2NN}$, (Fe,Cr)–Ni")
    ax.set_title(ttl, fontsize=7, pad=3.0); SD[key] = pd.DataFrame(out)
axa.set_ylim(-58, 58); axa.set_ylabel("Energy relative to the mean\nof the random cells (meV per atom)", linespacing=1.2); axb.tick_params(labelleft=False)

# (c) the slope of panel a, channel by channel
GR = [g for g in J["models"][NCE]["groups"] if g != "singlets"]
sl = {m: {g: (float(np.polyfit(a2[:nr], [r["models"][m]["phi"][g] for r in RND], 1)[0]) if g in J["models"][m]["groups"] else np.nan) for g in GR} for m in (NCE, OCE)}
GR = sorted(GR, key=lambda g: -max(abs(sl[NCE][g]), abs(np.nan_to_num(sl[OCE][g])))); yy = np.arange(len(GR)); hb = 0.36
axc.barh(yy - hb / 2, [sl[NCE][g] for g in GR], height=hb, color=PURPLE, lw=0, zorder=3); axc.barh(yy + hb / 2, [np.nan_to_num(sl[OCE][g]) for g in GR], height=hb, color=ICET, lw=0, zorder=3)
axc.axvline(0, color="k", lw=0.6, zorder=2); axc.set_yticks(yy); axc.set_yticklabels([CL.wrapped(CL.label("fenicr", g)) for g in GR]); axc.tick_params(axis="y", length=0, pad=2); axc.set_ylim(len(GR) - 0.5, -0.5)
axc.set_xlim(-24, 9); axc.set_xticks([-20, -10, 0]); axc.set_xlabel("Contribution to the slope in a\n" + r"(meV per atom per unit $\alpha_\mathrm{2NN}$)", linespacing=1.2); axc.set_title("By cluster channel", fontsize=7, pad=3.0)
SD["c"] = pd.DataFrame({"Cluster channel": [CL.plain(CL.label("fenicr", g)) for g in GR], f"{NCE} (meV per atom per unit alpha 2NN)": [sl[NCE][g] for g in GR], f"{OCE} (meV per atom per unit alpha 2NN)": [sl[OCE][g] for g in GR]})

# one legend for the three panels, in the band above them
hd = [Line2D([], [], color=st["color"], lw=0.75, ls=st["ls"], marker=st["marker"], ms=st["ms"], mfc=st["color"], mec=st["color"]) for st in STY.values()] + [Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec="0.35", mew=0.6)]
fig.legend(hd, list(STY) + [r"L1$_0$-ordered cells (not in the fits)"], loc="upper left", bbox_to_anchor=(15.0 / W, 1 - 2.5 / H), ncol=4, frameon=False, fontsize=6, handlelength=2.6, handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)
LET = [fig.text(x / W, (Y0 + HA + 1.5) / H, s, fontsize=8, fontweight="bold", ha="left", va="bottom") for s, x in zip("abc", (2.0, 63.0, 117.0))]

print(f"canvas {W:.0f} x {H:.0f} mm | margins (left, right, bottom, top) mm: {ns.margins(fig)}", flush=True)
fig.savefig(f"{D}/figS4_fenicr_order.pdf"); fig.savefig(f"{D}/figS4_fenicr_order.svg"); fig.savefig(f"{D}/figS4_fenicr_order.png", dpi=600); print("wrote figS4_fenicr_order.pdf / .svg / .png", flush=True)
with pd.ExcelWriter(f"{D}/SourceData_FigS4.xlsx", engine="openpyxl") as xw:
    for k in "abc": SD[k].to_excel(xw, sheet_name=k, index=False)
l10 = {key: {name: [float(v) for v in np.asarray(SD[key][f"{name} (meV per atom)"])[nr:]] for name in STY} for key in "ab"}
json.dump(dict(slopes={f"{k} {n}": v for (k, n), v in SLOPE.items()}, L10_minus_random_mean=l10, channel_slopes=sl, cells=[r["name"] for r in A]), open(f"{D}/figS4_fenicr_order.json", "w"), indent=1)
print("slopes (meV per atom per unit alpha_2NN):", {f"{k} {n}": round(v, 1) for (k, n), v in SLOPE.items()}); print("L1_0 cells minus the random mean:", {k: {n: [round(x, 1) for x in v] for n, v in d.items()} for k, d in l10.items()})
print("channel slopes:", {m: {g: round(v, 1) for g, v in d.items()} for m, d in sl.items()})
sz, bad = ns.text_check(fig, "abc"); print("font:", FAM, "| text sizes:", sz, "| violations:", bad)
r_ = fig.canvas.get_renderer()
for ax in (axa, axb):                                                                       # no marker under the slope legend
    bb = ax.get_legend().get_window_extent(r_); pts = np.vstack([ax.transData.transform(np.column_stack([np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float)])) for l in ax.lines if len(l.get_xdata()) > 2])
    print("markers under the slope legend:", int(np.sum((pts[:, 0] > bb.x0 - 3) & (pts[:, 0] < bb.x1 + 3) & (pts[:, 1] > bb.y0 - 3) & (pts[:, 1] < bb.y1 + 3))))
