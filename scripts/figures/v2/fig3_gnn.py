"""Fig. 3 and Supplementary Fig. S1, v2 style (same data and numbers as scripts/figures/fig3_gnn.py; Table S1 is not redrawn here).
Fig. 3   (a) time per Monte Carlo step per temperature and (b) GPU memory for 32 temperatures vs number of atoms, neural cluster expansion vs three GNNs;
         (c) mean |SHAP| of the cluster channels, PtNi / STFO / FeNiCr.  a, b: 74 x 36 mm (two-column grid); c: three 34 x 36 mm bar panels at the 60 mm pitch.
Fig. S1  beeswarm of the SHAP values, three panels at the same 60 mm pitch.
Filled markers + solid lines = measured, open markers + dashed = extrapolated.  Exponent at the end of each line: log-log slope over the measured points.
Inputs:  results/bench_stage/gnn_mc_step_H100.json, results/bench_stage/gpu_conv_NVIDIA_H100_80GB_HBM3*.json, results/shap/{ptni,stfo,fenicr}.json.
Outputs: figures/v2/fig3_gnn.{pdf,svg,png}, figures/v2/figS1_shap_beeswarm.{pdf,svg,png}.
Run:     python scripts/figures/v2/fig3_gnn.py"""
import os, sys, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import style as S
from neuralce.paths import REPO, RESULTS
import matplotlib as mpl; mpl.use("Agg"); FAM = S.use()
import channel_labels as CL

B = f"{RESULTS}/bench_stage"; SHAP = f"{RESULTS}/shap"; OUT = f"{REPO}/figures/v2"; os.makedirs(OUT, exist_ok=True); NT = 32; GPU_GB = 80.0
MODELS = [("conv", "Neural cluster expansion", S.BLUE, "o"), ("cgcnn", "CGCNN", S.SLATE, "v"), ("alignn", "ALIGNN", S.NAVY, "^"), ("disorder_gnn", "Disorder-GNN", S.OLIVE, "D")]
NEXT = [1048576, 10061824]
SYSTEMS = [("ptni", "PtNi", "meV per atom"), ("stfo", r"SrTi$_{1-\mathit{x}}$Fe$_{\mathit{x}}$O$_{3-\mathit{x}/2}$", "meV per site"), ("fenicr", "FeNiCr", "meV per atom")]
# panel c and Fig. S1 are drawn for the systems whose SHAP file exists; the others are skipped (the STFO / Fe-Ni-Cr results are not shipped)
def _have(s):
    if os.path.exists(f"{SHAP}/{s}.json"): return True
    print(f"[skip] v2/fig3_gnn.py: {SHAP}/{s}.json is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_RESULTS/shap/ to run", flush=True); return False
SYSTEMS = [t for t in SYSTEMS if _have(t[0])]

# ---- speed / memory data (identical to fig3_gnn.py) -----------------------------------------------------------------------------------------
G = json.load(open(f"{B}/gnn_mc_step_H100.json")); CV = sorted(json.load(open(f"{B}/gpu_conv_NVIDIA_H100_80GB_HBM3.json")), key=lambda r: r["n_atoms"])
D = {}
for key, name, col, mk in MODELS[1:]:
    rows = [r for r in G if r["model"] == key]; t, m = {"meas": {}, "ext": {}}, {"meas": {}, "ext": {}}
    for N in sorted({r["n_atoms"] for r in rows}):
        b32 = next((r for r in rows if r["n_atoms"] == N and r["batch"] == NT and r.get("ms_per_step")), None); b1 = next((r for r in rows if r["n_atoms"] == N and r["batch"] == 1 and r.get("ms_per_step")), None)
        if b32: t["meas"][N] = b32["ms_per_step"] / NT * 1e3; m["meas"][N] = b32["gpu_mem_MB"] / 1024
        elif b1: t["ext"][N] = b1["ms_per_step"] * 1e3; m["ext"][N] = b1["gpu_mem_MB"] / 1024 * NT
    last = max(list(t["meas"]) + list(t["ext"])); tl = {**t["meas"], **t["ext"]}[last]; ml = {**m["meas"], **m["ext"]}[last]
    for N in NEXT: t["ext"][N] = tl * N / last; m["ext"][N] = ml * N / last
    D[name] = dict(t=t, m=m)
t, m = {"meas": {}, "ext": {}}, {"meas": {}, "ext": {}}
for r in CV:
    if r["R"] == NT: t["meas"][r["n_atoms"]] = r["us_per_replica_step"]["total"]; m["meas"][r["n_atoms"]] = r["peak_MB"] / 1024
for fn_ in sorted(glob.glob(f"{B}/gpu_conv_NVIDIA_H100_80GB_HBM3_R*.json")):
    for r in json.load(open(fn_)):
        if r["n_atoms"] > max(t["meas"]) and r.get("us_per_replica_step"): t["ext"][r["n_atoms"]] = r["us_per_replica_step"]["total"]; m["ext"][r["n_atoms"]] = r["peak_MB"] / 1024 * NT / r["R"]
D["Neural cluster expansion"] = dict(t=t, m=m)
def slope(d):
    x = np.log(sorted(d)); return float(np.polyfit(x, np.log([d[k] for k in sorted(d)]), 1)[0])
EXPO = {name: dict(t=slope(D[name]["t"]["meas"]), m=slope(D[name]["m"]["meas"])) for _, name, _, _ in MODELS}


def shap_data(sysname):
    Dd = json.load(open(f"{SHAP}/{sysname}.json")); Gs = Dd["groups"]; Phi = np.array([[r["phi"][g] for g in Gs] for r in Dd["rows"]]); labs = np.array([str(r["label"]) for r in Dd["rows"]])
    for lb in set(labs): Phi[labs == lb] -= Phi[labs == lb].mean(0)                    # relative to the mean of the same composition
    Tt = np.array([r["T"] for r in Dd["rows"]]); imp = np.abs(Phi).mean(0); o = [g for g in np.argsort(imp)[::-1] if Gs[g] != "singlets"][:6]
    return dict(Gs=Gs, Phi=Phi, T=Tt, imp=imp, order=o)
SH = {s: shap_data(s) for s, _, _ in SYSTEMS}; NROW = max(len(SH[s]["order"]) for s, _, _ in SYSTEMS)   # the same row pitch in the three panels

# ==== Fig. 3 ================================================================================================================================
(Y1, Y2), H = S.rows(2, foots=[S.FOOT, S.FOOT_PLAIN]); C = S.Canvas(H); fig = C.fig
axa, axb = (C.ax(x, Y1 + S.STRIP, S.W2) for x in S.X2)


def end_labels(ax, items, gap_pt=11.5):
    """Exponent right of the last marker of each line; labels closer than gap_pt are pushed apart symmetrically."""
    tr = ax.transData; inv = tr.inverted(); px = fig.dpi / 72.0
    pts = sorted(((tuple(tr.transform((x, y))), txt, c) for x, y, txt, c in items), key=lambda p: p[0][1]); y0 = np.array([p[0][1] for p in pts]); up, dn = y0.copy(), y0.copy()
    for i in range(1, len(up)): up[i] = max(up[i], up[i - 1] + gap_pt * px)
    for i in range(len(dn) - 2, -1, -1): dn[i] = min(dn[i], dn[i + 1] - gap_pt * px)
    for (xy, txt, c), yd in zip(pts, (up + dn) / 2): ax.text(*inv.transform((xy[0] + 4.0 * px, yd)), txt, fontsize=6, color=c, ha="left", va="center")


def speed_panel(ax, q, lo, hi):
    items = []
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(150, 6e7); ax.set_ylim(10.0 ** lo, 10.0 ** hi)
    for key, name, col, mk in MODELS:
        d = D[name][q]; xm = sorted(d["meas"]); xe = sorted(d["ext"]); z = 6 if key == "conv" else 4
        if xe: ax.plot([xm[-1]] + xe, [d["meas"][xm[-1]]] + [d["ext"][x] for x in xe], color=col, lw=S.LW, ls=S.DASH, zorder=z - 1)
        ax.plot(xm, [d["meas"][x] for x in xm], color=col, lw=S.LW, marker=mk, ms=S.MS, mfc=col, mec=col, mew=0, zorder=z)
        if xe: ax.plot(xe, [d["ext"][x] for x in xe], ls="none", marker=mk, ms=S.MS, mfc="white", mec=col, mew=0.9, zorder=z)
        xl = max(xm + xe); below = 0.3 if (q == "m" and key == "conv") else 1.0                               # this label sits under the GPU-memory line, not on it
        items.append((xl, {**d["meas"], **d["ext"]}[xl] * below, rf"~$\mathit{{N}}^{{{EXPO[name][q]:.1f}}}$", col))     # ASCII tilde: Arial has no U+221D
    S.log_axis(ax.xaxis, 3, 7); S.log_axis(ax.yaxis, lo, hi); ax.set_xlabel("Number of atoms"); end_labels(ax, items)


speed_panel(axa, "t", 0, 9); S.ylabel(axa, "Time per Monte Carlo step\nper temperature (μs)", linespacing=1.15)
speed_panel(axb, "m", -2, 6); S.ylabel(axb, f"GPU memory, {NT} temperatures (GB)")
hl = {n: S.line(c, mk) for _, n, c, mk in MODELS}
axa.legend([hl["CGCNN"], hl["ALIGNN"], hl["Disorder-GNN"], hl["Neural cluster expansion"], S.line(S.GUIDE, "s", ls=S.DASH, open_=True)],
           ["CGCNN", "ALIGNN", "Disorder-GNN", "Neural cluster expansion (this work)", "Extrapolated (exceeds GPU memory)"], loc="upper left", handlelength=2.6, borderaxespad=0.2)
axb.axhline(GPU_GB, color=S.GUIDE, lw=0.7, ls=(0, (4, 2)), zorder=1)
axb.annotate(f"GPU memory ({GPU_GB:.0f} GB)", (170, GPU_GB), xytext=(0, 1.5), textcoords="offset points", ha="left", va="bottom", fontsize=6, color=S.GUIDE)
C.letter(axa, Y1 + S.LINE1, "a"); C.letter(axb, Y1 + S.LINE1, "b")

LABW = 22.0; axc = []                                                                     # label column (mm) left of each bar panel
for q, (sysname, title, unit) in enumerate(SYSTEMS):
    ax = C.ax(S.XL[q], Y2 + S.STRIP, S.WL); axc.append(ax); d = SH[sysname]; o = d["order"]; v = d["imp"][o]; yy = np.arange(len(o))
    ax.barh(yy, v, height=0.62, color=S.BLUE, ec="black", lw=0.5, zorder=3)
    for y_, v_ in zip(yy, v): ax.text(v_ + 0.025 * v.max(), y_, f"{v_:.1f}", fontsize=6, ha="left", va="center")
    ax.set_yticks(yy); ax.set_yticklabels([S.fit_label(fig, CL.label(sysname, d["Gs"][g]), LABW, CL.wrapped, CL.wrap_to) for g in o], linespacing=1.0)
    ax.tick_params(axis="y", length=0, pad=3); ax.set_ylim(NROW - 0.45, -0.55)
    ax.set_xlim(0, 1.24 * v.max()); ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4)); ax.set_xlabel(f"Mean |SHAP| ({unit})"); C.text(S.XL[q], Y2 + S.LINE1, title)
C.letter(2.0, Y2 + S.LINE1, "c")
S.check(C, "abc", "Fig. 3"); C.save(f"{OUT}/fig3_gnn")

# ==== Supplementary Fig. S1: SHAP beeswarm ===================================================================================================
def swarm(v, width, nb=42):                                                                # points of the same bin stacked symmetrically about the row
    q = np.round(nb * (v - v.min()) / (np.ptp(v) + 1e-12)); y = np.zeros(len(v)); last, layer = None, 0
    for j in np.argsort(q, kind="stable"):
        if q[j] != last: last, layer = q[j], 0
        y[j] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1); layer += 1
    return y * min(width / max(np.abs(y).max(), 1), 0.085)
HS, STRIP1, VCOL = 48.0, 11.0, 7.5; WS = S.WL - VCOL                                       # axes height, band above it (title + colour bar), column of the mean |SHAP| numbers
C1 = S.Canvas(S.EDGE + STRIP1 + HS + S.FOOT_PLAIN + S.EDGE); YA = S.EDGE + STRIP1
XLIM = 40.0 if max(np.abs(SH[s]["Phi"][:, SH[s]["order"]]).max() for s, _, _ in SYSTEMS) < 40.0 else None
for q, (sysname, title, unit) in enumerate(SYSTEMS):
    ax = C1.ax(S.XL[q], YA, WS, HS); d = SH[sysname]; o = d["order"]; norm = mpl.colors.Normalize(d["T"].min(), d["T"].max()); xm = XLIM or 1.12 * np.abs(d["Phi"][:, o]).max()
    for row, g in enumerate(o):
        ax.axhline(row, color=S.DOT, lw=0.5, zorder=1)
        ax.scatter(d["Phi"][:, g], row + swarm(d["Phi"][:, g], 0.36), c=d["T"], cmap=S.T_CMAP, norm=norm, s=4.5, ec=S.INK, lw=0.2, zorder=3)
        ax.text(1.04, row, f"{d['imp'][g]:.1f}", transform=ax.get_yaxis_transform(), fontsize=6, ha="left", va="center")
    ax.text(1.04, -0.62, "Mean\n|SHAP|", transform=ax.get_yaxis_transform(), fontsize=6, ha="left", va="bottom", linespacing=1.0)
    ax.axvline(0, color="black", lw=0.7, zorder=2); ax.set_xlim(-xm, xm); ax.set_ylim(NROW - 0.5, -0.5); ax.set_yticks(range(len(o)))
    ax.set_yticklabels([S.fit_label(C1.fig, CL.label(sysname, d["Gs"][g]), LABW, CL.wrapped, CL.wrap_to) for g in o], linespacing=1.0)
    ax.tick_params(axis="y", length=0, pad=3); ax.set_xlabel(f"SHAP value ({unit})"); ax.spines["left"].set_visible(False)
    cw = 12.0; cax = C1.ax(S.XL[q] + (WS - cw) / 2, YA - 2.6, cw, 1.1); cb = C1.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=S.T_CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.5); cax.xaxis.set_ticks_position("top"); cb.set_ticks([d["T"].min(), d["T"].max()]); cb.set_ticklabels([f"{d['T'].min():.0f}", f"{d['T'].max():.0f}"])
    cax.tick_params(labelsize=6, length=1.5, width=0.5, pad=1); cax.text(-0.08, 0.5, "T (K)", transform=cax.transAxes, fontsize=6, ha="right", va="center")
    C1.text(S.XL[q], S.EDGE + S.LINE1, title); C1.letter(S.XL[q] - 24.0, S.EDGE + S.LINE1, "abc"[q])
S.check(C1, "abc", "Fig. S1"); C1.save(f"{OUT}/figS1_shap_beeswarm")
print("exponents:", {n: {k: round(v, 3) for k, v in e.items()} for n, e in EXPO.items()})
print("mean |SHAP|:", {s: [(CL.plain(CL.label(s, SH[s]['Gs'][g])), round(float(SH[s]['imp'][g]), 2)) for g in SH[s]['order']] for s, _, _ in SYSTEMS})
