"""Fig. 3 (Nature format, 183 x 110 mm) + Supplementary Fig. S1 (SHAP beeswarm, 183 x 71 mm) + Supplementary Table S1 (ground-state Shapley split).
Fig. 3    (a) time per MC step per temperature and (b) GPU memory for 32 temperatures vs number of atoms, cluster convolution vs three GNNs
          (PtNi fcc cells, one NVIDIA H100 80GB HBM3, same GPU as Fig. 2); (c) mean |SHAP| of the cluster channels of the cluster convolution, PtNi / STFO / FeNiCr.
          Filled markers + solid lines = measured.  Open markers + dashed = extrapolated where 32 temperatures exceed the GPU memory:
          time from the measured single-temperature pass (GNNs) or the 8-temperature run (cluster conv. at 10M atoms), memory scaled to 32 temperatures,
          GNN sizes beyond the last run linear in N.  Exponent at the end of each line: log-log least-squares slope over the MEASURED points of that line.
Fig. S1   beeswarm of the SHAP values (exact Shapley values over cluster-channel groups, MC-sampled configurations, values relative to the mean of the
          same composition, point-cluster group left out; results/shap/<system>.json from scripts/analysis/shap_cluster_conv.py).
Table S1  exact Shapley contributions of the cluster channels to the energy of the known ground state relative to the random alloy of the same
          composition (results/shap/<system>_gs.json).  STFO energies are per site (160-site cell incl. oxygen vacancies); PtNi and FeNiCr per atom.
Inputs:  results/bench_stage/gnn_mc_step_H100.json, results/bench_stage/gpu_conv_NVIDIA_H100_80GB_HBM3*.json (R32 and the R128 / R65 / R8 runs),
         results/shap/{ptni,stfo,fenicr}.json and {ptni,stfo,fenicr}_gs.json.
Outputs in figures/: fig3_gnn.{pdf,svg,png,json}, figS1_shap_beeswarm.{pdf,svg,png}, SourceData_Fig3.xlsx (a b c), SourceData_FigS1.xlsx (a b c),
         TableS1_ground_state_shapley.{xlsx,tex} and, when pdflatex and PyMuPDF are available, TableS1_ground_state_shapley.png (skipped otherwise),
         channel_labels_used_gnn.json (the channel strings drawn / written, for the cross-figure consistency check).
Run:     python scripts/figures/fig3_gnn.py"""
import os, sys, glob, json, shutil, subprocess, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import natstyle as ns
from neuralce.paths import REPO, RESULTS
import matplotlib as mpl; mpl.use("Agg"); FAM = ns.use()
import channel_labels as CL                                                              # channel names shared with Fig. S3 and Table S1
import matplotlib.pyplot as plt
import pandas as pd

B = f"{RESULTS}/bench_stage"; SHAP = f"{RESULTS}/shap"; D = f"{REPO}/figures"; os.makedirs(D, exist_ok=True); NT = 32; GPU_GB = 80.0
PURPLE = "#5b2a9e"; GREY6 = "#606060"
MODELS = [("conv", "Neural cluster expansion", PURPLE, "o"), ("cgcnn", "CGCNN", "#8c8c8c", "v"), ("alignn", "ALIGNN", "#3a3a3a", "^"), ("disorder_gnn", "Disorder-GNN", "#c46a1a", "D")]
NEXT = [1048576, 10061824]
SYSTEMS = [("ptni", "PtNi", "meV per atom"), ("stfo", r"SrTi$_{1-\mathit{x}}$Fe$_{\mathit{x}}$O$_{3-\mathit{x}/2}$", "meV per site"), ("fenicr", "FeNiCr", "meV per atom")]
# panels c / Fig. S1 / Table S1 are drawn for the systems whose SHAP files exist; the others are skipped (the STFO / Fe-Ni-Cr results are not shipped)
def _have(s):
    for p in (f"{SHAP}/{s}.json", f"{SHAP}/{s}_gs.json"):
        if not os.path.exists(p):
            print(f"[skip] fig3_gnn.py: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_RESULTS/shap/ to run", flush=True); return False
    return True
SYSTEMS = [t for t in SYSTEMS if _have(t[0])]

# ---- speed / memory data (H100) -----------------------------------------------------------------------------------------------------------
G = json.load(open(f"{B}/gnn_mc_step_H100.json")); CV = sorted(json.load(open(f"{B}/gpu_conv_NVIDIA_H100_80GB_HBM3.json")), key=lambda r: r["n_atoms"])
S = {}                                                                                   # model -> {t|m: {"meas": {N: v}, "ext": {N: v}}, notes}
for key, name, col, mk in MODELS[1:]:
    rows = [r for r in G if r["model"] == key]; t, m, note = {"meas": {}, "ext": {}}, {"meas": {}, "ext": {}}, {}
    for N in sorted({r["n_atoms"] for r in rows}):
        b32 = next((r for r in rows if r["n_atoms"] == N and r["batch"] == NT and r.get("ms_per_step")), None); b1 = next((r for r in rows if r["n_atoms"] == N and r["batch"] == 1 and r.get("ms_per_step")), None)
        if b32: t["meas"][N] = b32["ms_per_step"] / NT * 1e3; m["meas"][N] = b32["gpu_mem_MB"] / 1024; note[N] = "measured, 32 temperatures"
        elif b1: t["ext"][N] = b1["ms_per_step"] * 1e3; m["ext"][N] = b1["gpu_mem_MB"] / 1024 * NT; note[N] = "32 temperatures exceed the GPU: time of the measured 1-temperature pass, memory x 32"
    last = max(list(t["meas"]) + list(t["ext"])); tl = {**t["meas"], **t["ext"]}[last]; ml = {**m["meas"], **m["ext"]}[last]
    for N in NEXT: t["ext"][N] = tl * N / last; m["ext"][N] = ml * N / last; note[N] = f"linear in N from {last:,} atoms"
    S[name] = dict(t=t, m=m, note=note)
t, m, note = {"meas": {}, "ext": {}}, {"meas": {}, "ext": {}}, {}
for r in CV:
    if r["R"] == NT: t["meas"][r["n_atoms"]] = r["us_per_replica_step"]["total"]; m["meas"][r["n_atoms"]] = r["peak_MB"] / 1024; note[r["n_atoms"]] = "measured, 32 temperatures"
for fn_ in sorted(glob.glob(f"{B}/gpu_conv_NVIDIA_H100_80GB_HBM3_R*.json")):
    for r in json.load(open(fn_)):
        if r["n_atoms"] > max(t["meas"]) and r.get("us_per_replica_step"):
            t["ext"][r["n_atoms"]] = r["us_per_replica_step"]["total"]; m["ext"][r["n_atoms"]] = r["peak_MB"] / 1024 * NT / r["R"]; note[r["n_atoms"]] = f"32 temperatures exceed the GPU: {r['R']}-temperature run, memory x {NT // r['R']}"
S["Neural cluster expansion"] = dict(t=t, m=m, note=note)
def slope(d):
    x = np.log(sorted(d)); return float(np.polyfit(x, np.log([d[k] for k in sorted(d)]), 1)[0])
EXPO = {name: dict(time=slope(S[name]["t"]["meas"]), memory=slope(S[name]["m"]["meas"])) for _, name, _, _ in MODELS}


def mmfig(w, h): return plt.figure(figsize=(w * ns.MM, h * ns.MM))
def add_ax(fig, W, H, x, y, w, h, **kw): return fig.add_axes([x / W, y / H, w / W, h / H], **kw)
def letter(fig, W, H, x, y, s): fig.text(x / W, y / H, s, fontsize=8, fontweight="bold", ha="left", va="bottom")


def end_labels(fig, ax, items, gap_pt=10.0):                                         # 10 pt: room for the exponent of the label below
    """Exponent text right of the last marker of each line, pushed apart vertically so that no two overlap."""
    tr = ax.transData; inv = tr.inverted(); px = fig.dpi / 72.0
    pts = sorted(((tuple(tr.transform((x, y))), txt, c) for x, y, txt, c in items), key=lambda p: p[0][1])
    ys = [p[0][1] for p in pts]
    for i in range(1, len(ys)): ys[i] = max(ys[i], ys[i - 1] + gap_pt * px)
    for (xy, txt, c), yd in zip(pts, ys): ax.text(*inv.transform((xy[0] + 3.0 * px, yd)), txt, fontsize=6, color=c, ha="left", va="center")


def speed_panel(fig, ax, q, ylabel, ylim):
    items = []
    for key, name, col, mk in MODELS:
        d = S[name][q]; xm = sorted(d["meas"]); xe = sorted(d["ext"]); z = 6 if key == "conv" else 4
        ax.plot(xm, [d["meas"][x] for x in xm], color=col, lw=0.75, marker=mk, ms=3, mfc=col, mec=col, mew=0.5, zorder=z)
        if xe:
            ax.plot([xm[-1]] + xe, [d["meas"][xm[-1]]] + [d["ext"][x] for x in xe], color=col, lw=0.75, ls=(0, (3, 2)), zorder=z - 1)
            ax.plot(xe, [d["ext"][x] for x in xe], ls="none", marker=mk, ms=3, mfc="white", mec=col, mew=0.6, zorder=z)
        xl = max(xm + xe); yl = {**d["meas"], **d["ext"]}[xl]; b = EXPO[name]["time" if q == "t" else "memory"]
        items.append((xl, yl * (0.3 if (q == "m" and key == "conv") else 1.0), rf"~$\mathit{{N}}^{{{b:.1f}}}$", col))                           # ASCII tilde: Arial has no U+221D, and a fallback glyph would be a second typeface
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(150, 6e7); ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=12)); ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=12))
    for a_ in (ax.xaxis, ax.yaxis): a_.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=12)); a_.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Number of atoms"); ax.set_ylabel(ylabel); end_labels(fig, ax, items)


def shap_data(sysname):
    Dd = json.load(open(f"{SHAP}/{sysname}.json")); Gs = Dd["groups"]; Phi = np.array([[r["phi"][g] for g in Gs] for r in Dd["rows"]]); labs = np.array([str(r["label"]) for r in Dd["rows"]])
    for lb in set(labs): Phi[labs == lb] -= Phi[labs == lb].mean(0)                    # relative to the mean of the same composition
    Tt = np.array([r["T"] for r in Dd["rows"]]); imp = np.abs(Phi).mean(0); o = [g for g in np.argsort(imp)[::-1] if Gs[g] != "singlets"][:6]
    return dict(Gs=Gs, Phi=Phi, labs=labs, T=Tt, imp=imp, order=o, files=[r["file"] for r in Dd["rows"]])


SH = {s: shap_data(s) for s, _, _ in SYSTEMS}
_meas_fig = plt.figure()                                                                 # text widths (mm) for the label columns
def text_mm(s, size=6):
    t = _meas_fig.text(0, 0, s, fontsize=size); w = t.get_window_extent(renderer=_meas_fig.canvas.get_renderer()).width / _meas_fig.dpi * 25.4; t.remove(); return w
LABW = [max(text_mm(CL.wrapped(CL.label(s, SH[s]["Gs"][g]))) for g in SH[s]["order"]) for s, _, _ in SYSTEMS]

# ==== Fig. 3 (183 x 110 mm) =================================================================================================================
W, H = 183.0, 110.0; fig = mmfig(W, H)                                                    # 110 mm: >= 2 mm blank at every edge and >= 4 mm between the rows
H1, H2, YA, YC = 42.0, 33.0, 61.0, 10.5                                                   # axes heights (mm) and bottoms of rows 1 and 2
axa = add_ax(fig, W, H, 15.5, YA, 74.0, H1); axb = add_ax(fig, W, H, 107.0, YA, 74.0, H1)
speed_panel(fig, axa, "t", "Time per Monte Carlo step\nper temperature (μs)", (1.0, 1e9))       # top decade left free for the legend
speed_panel(fig, axb, "m", f"GPU memory, {NT} temperatures (GB)", (0.01, 1e6))                   # H100 conv at 256-2,048 atoms is below 0.1 GB
from matplotlib.lines import Line2D
hl = [Line2D([], [], color=c, lw=0.75, marker=mk, ms=3, mfc=c, mec=c, mew=0.5) for _, _, c, mk in MODELS] + [Line2D([], [], color=GREY6, lw=0.75, ls=(0, (3, 2)), marker="s", ms=3, mfc="white", mec=GREY6, mew=0.6)]
LORDER = [1, 2, 3, 0, 4]                                                                      # CGCNN / ALIGNN / Disorder-GNN / Cluster conv. (this work) / Extrapolated
axa.legend([hl[k] for k in LORDER], [(["Neural cluster expansion (this work)"] + [n for _, n, _, _ in MODELS[1:]] + ["Extrapolated (exceeds GPU memory)"])[k] for k in LORDER], loc="upper left", frameon=False, fontsize=6, handlelength=2.6, borderaxespad=0.3)
axb.axhline(GPU_GB, color=GREY6, lw=0.6, ls=(0, (4, 2)), zorder=1)
axb.annotate(f"GPU memory ({GPU_GB:.0f} GB)", (170, GPU_GB), xytext=(0, 1.0), textcoords="offset points", ha="left", va="bottom", fontsize=6, color=GREY6)
letter(fig, W, H, 2.0, YA + H1 + 1.0, "a"); letter(fig, W, H, 94.5, YA + H1 + 1.0, "b")
# (c) mean |SHAP| bars, three sub-panels, each with its own x range
GAP = 3.0; wc = (W - 3.5 - 2.0 - sum(LABW) - 3 * 1.5 - 2 * GAP) / 3; x0 = 3.5; axc = []   # same left and right margins as row a, b
for q, ((sysname, title, unit), lw_) in enumerate(zip(SYSTEMS, LABW)):
    x0 += lw_ + 1.5; ax = add_ax(fig, W, H, x0, YC, wc, H2); axc.append(ax); x0 += wc + GAP
    d = SH[sysname]; o = d["order"]; v = d["imp"][o]; yy = np.arange(len(o))
    ax.barh(yy, v, height=0.6, color=PURPLE, lw=0, zorder=3)
    for y_, v_ in zip(yy, v): ax.text(v_ + 0.02 * v.max(), y_, f"{v_:.1f}", fontsize=6, color="black", ha="left", va="center")
    ax.set_yticks(yy); ax.set_yticklabels([CL.wrapped(CL.label(sysname, d["Gs"][g])) for g in o]); ax.tick_params(axis="y", length=0, pad=2); ax.set_ylim(len(o) - 0.5, -0.5)
    ax.set_xlim(0, 1.22 * v.max()); ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4)); ax.set_xlabel(f"Mean |SHAP| ({unit})"); ax.set_title(title, fontsize=7, pad=3)
letter(fig, W, H, 2.0, YC + H2 + 1.0, "c")
fig.canvas.draw(); _r = fig.canvas.get_renderer(); _mm = lambda px: px / fig.dpi * 25.4
_gap = _mm(min(a_.xaxis.label.get_window_extent(_r).y0 for a_ in (axa, axb)) - max([a_.title.get_window_extent(_r).y1 for a_ in axc] + [t_.get_window_extent(_r).y1 for t_ in fig.texts if t_.get_text() == "c"]))
print(f"Fig. 3 canvas {W:.0f} x {H:.0f} mm | margins (left, right, bottom, top) mm: {ns.margins(fig)} | blank between rows: {_gap:.2f} mm", flush=True)
fig.savefig(f"{D}/fig3_gnn.pdf"); fig.savefig(f"{D}/fig3_gnn.svg"); fig.savefig(f"{D}/fig3_gnn.png", dpi=600); print("wrote fig3_gnn.pdf / .svg / .png", flush=True)
sz6, bad6 = ns.text_check(fig)

# ==== Supplementary Fig. S1: SHAP beeswarm (183 x 71 mm) =================================================================================
def swarm(v, width, nb=42):                                                                # points of the same bin stacked symmetrically about the row
    q = np.round(nb * (v - v.min()) / (np.ptp(v) + 1e-12)); y = np.zeros(len(v)); last, layer = None, 0
    for j in np.argsort(q, kind="stable"):
        if q[j] != last: last, layer = q[j], 0
        y[j] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1); layer += 1
    return y * min(width / max(np.abs(y).max(), 1), 0.085)
W2, H2s = 183.0, 71.0; figs = mmfig(W2, H2s); YB, HB, MCOL = 9.5, 49.0, 7.0                # 71 mm and a 3 mm right reserve: >= 2 mm blank at every edge
XLIM = 40.0 if max(np.abs(SH[s]["Phi"][:, SH[s]["order"]]).max() for s, _, _ in SYSTEMS) < 40.0 else None
ws = (W2 - 3.5 - 3.0 - sum(LABW) - 3 * (1.5 + MCOL) - 2 * 3.0) / 3; x0 = 3.5
for q, ((sysname, title, unit), lw_) in enumerate(zip(SYSTEMS, LABW)):
    lx = x0; x0 += lw_ + 1.5; ax = add_ax(figs, W2, H2s, x0, YB, ws, HB); x0 += ws + MCOL + 3.0
    d = SH[sysname]; o = d["order"]; norm = mpl.colors.Normalize(d["T"].min(), d["T"].max()); xm = XLIM or 1.12 * np.abs(d["Phi"][:, o]).max()
    for row, g in enumerate(o):
        ax.axhline(row, color="0.85", lw=0.4, zorder=1)
        ax.scatter(d["Phi"][:, g], row + swarm(d["Phi"][:, g], 0.36), c=d["T"], cmap="coolwarm", norm=norm, s=4.0, ec="k", lw=0.2, zorder=3)
        ax.text(1.03, row, f"{d['imp'][g]:.1f}", transform=ax.get_yaxis_transform(), fontsize=6, ha="left", va="center")
    ax.text(1.03, -0.62, "Mean\n|SHAP|", transform=ax.get_yaxis_transform(), fontsize=6, ha="left", va="bottom", linespacing=1.0)
    ax.axvline(0, color="k", lw=0.6, zorder=2); ax.set_xlim(-xm, xm); ax.set_ylim(len(o) - 0.5, -0.5); ax.set_yticks(range(len(o)))
    ax.set_yticklabels([CL.wrapped(CL.label(sysname, d["Gs"][g])) for g in o]); ax.tick_params(axis="y", length=0, pad=2); ax.set_xlabel(f"SHAP value ({unit})")
    p = ax.get_position(); cw = 0.40 * ws; cx = x0 - MCOL - 3.0 - ws + (ws - cw) / 2
    cax = add_ax(figs, W2, H2s, cx, YB + HB + 1.6, cw, 1.1); cb = figs.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4); cax.xaxis.set_ticks_position("top"); cb.set_ticks([d["T"].min(), d["T"].max()]); cb.set_ticklabels([f"{d['T'].min():.0f}", f"{d['T'].max():.0f}"])
    cax.tick_params(labelsize=6, length=1.5, width=0.4, pad=1); cax.text(-0.04, 0.5, "T (K)", transform=cax.transAxes, fontsize=6, ha="right", va="center")
    figs.text((x0 - MCOL - 3.0 - ws / 2) / W2, (YB + HB + 6.2) / H2s, title, fontsize=7, ha="center", va="bottom")
    letter(figs, W2, H2s, lx, YB + HB + 6.2, "abc"[q])
print("Supp. Fig. S1 margins (left, right, bottom, top) mm:", ns.margins(figs), flush=True)
figs.savefig(f"{D}/figS1_shap_beeswarm.pdf"); figs.savefig(f"{D}/figS1_shap_beeswarm.svg"); figs.savefig(f"{D}/figS1_shap_beeswarm.png", dpi=600); print("wrote figS1_shap_beeswarm.pdf / .svg / .png", flush=True)
szS, badS = ns.text_check(figs, "abc")

# ==== Supplementary Table S1: ground-state Shapley split ===================================================================================
GS = {"ptni": ("x50", "L1$_0$ PtNi", "L1₀ PtNi"), "stfo": ("500", "DFT ground state, $x$ = 0.5", "DFT ground state, x = 0.5"), "fenicr": ("mag_data_0000", "Fe$_{12}$Ni$_{16}$Cr$_{4}$", "Fe₁₂Ni₁₆Cr₄")}
SYS_TEX = {"ptni": "PtNi", "stfo": r"SrTi$_{1-x}$Fe$_x$O$_{3-x/2}$\textsuperscript{a}", "fenicr": "FeNiCr"}; SYS_TXT = {"ptni": "PtNi", "stfo": "SrTi1−xFexO3−x/2 (a)", "fenicr": "FeNiCr"}
trows, tex = [], []
for sysname, _, _ in SYSTEMS:
    Dg = json.load(open(f"{SHAP}/{sysname}_gs.json")); lab, st_tex, st_txt = GS[sysname]; r = next(t_ for t_ in Dg["rows"] if str(t_["label"]) == lab)
    groups = sorted([g for g in Dg["groups"] if g != "singlets"], key=lambda g: -abs(r["phi"][g])); E = r["E_meV"] - (r["E_base_meV"] + r["phi"].get("singlets", 0.0))
    assert abs(E - sum(r["phi"][g] for g in groups)) < 1e-6                              # Shapley efficiency: the channels add up to the total
    for k, g in enumerate(groups):
        trows.append(dict(System=SYS_TXT[sysname] if k == 0 else "", **{"Ground-state structure": st_txt if k == 0 else ""}, **{"Cluster channel": CL.plain(CL.label(sysname, g))}, Contribution=round(r["phi"][g], 1)))
        tex.append(f"{SYS_TEX[sysname] if k == 0 else ''} & {st_tex if k == 0 else ''} & {CL.tex(CL.label(sysname, g))} & ${r['phi'][g]:+.1f}$ \\\\")
    trows.append({"System": "", "Ground-state structure": "", "Cluster channel": "Total (ground state relative to random alloy)", "Contribution": round(E, 1)})
    tex.append(f" & & Total (ground state relative to random alloy) & ${E:+.1f}$ \\\\"); tex.append(r"\midrule")
tex[-1] = r"\bottomrule"
UNIT_NOTE = "Contributions in meV per atom for PtNi and FeNiCr; (a) meV per site for SrTi1−xFexO3−x/2 (160-site cell including oxygen vacancies)."
T_df = pd.DataFrame(trows).rename(columns={"Ground-state structure": "Ground state", "Contribution": "Contribution (meV per atom; meV per site for SrTi1−xFexO3−x/2)"})
with pd.ExcelWriter(f"{D}/TableS1_ground_state_shapley.xlsx", engine="openpyxl") as xw:
    T_df.to_excel(xw, sheet_name="Table S", index=False); pd.DataFrame({"Note": [UNIT_NOTE]}).to_excel(xw, sheet_name="Table S", index=False, startrow=len(T_df) + 2)
body = "\n".join(tex)
TABLE = rf"""\begin{{table}}
\centering
\caption{{}}
\label{{tab:gs_shapley}}
\begin{{tabular}}{{llll}}
\toprule
System & Ground state & Cluster channel & Contribution (meV per atom) \\
\midrule
{body}
\end{{tabular}}

\smallskip
\footnotesize{{\textsuperscript{{a}}Contributions for SrTi$_{{1-x}}$Fe$_x$O$_{{3-x/2}}$ are in meV per site (160-site cell including oxygen vacancies); PtNi and FeNiCr in meV per atom.}}
\end{{table}}
"""
open(f"{D}/TableS1_ground_state_shapley.tex", "w").write(TABLE)
# optional: render the table to PNG (needs pdflatex and PyMuPDF); skipped with a message when either is missing
try:
    import pymupdf
except ImportError:
    pymupdf = None
if shutil.which("pdflatex") is None or pymupdf is None:
    print("TableS1 PNG skipped:", "pdflatex not found" if shutil.which("pdflatex") is None else "PyMuPDF (pymupdf) not installed", flush=True)
else:
    work = tempfile.mkdtemp(prefix="tableS1_render_")
    open(f"{work}/render.tex", "w").write(r"""\documentclass{article}
\usepackage[a4paper,landscape,margin=1cm]{geometry}\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage{booktabs}\usepackage{sansmath}\sansmath
\pagestyle{empty}\makeatletter\renewcommand{\caption}[1]{}\makeatother
\begin{document}
\input{""" + f"{D}/TableS1_ground_state_shapley.tex" + r"""}
\end{document}
""")
    rc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "render.tex"], cwd=work, capture_output=True, text=True)
    if rc.returncode == 0:
        pg = pymupdf.open(f"{work}/render.pdf")[0]; bb = pymupdf.Rect()                          # crop the page to the table (text + rules) with a 6 pt border
        for b_ in pg.get_text("blocks"): bb |= pymupdf.Rect(b_[:4])
        for d_ in pg.get_drawings(): bb |= d_["rect"]
        pg.get_pixmap(dpi=600, clip=bb + (-6, -6, 6, 6)).save(f"{D}/TableS1_ground_state_shapley.png"); print("wrote TableS1_ground_state_shapley.png", flush=True)
    else: print("pdflatex failed:\n", rc.stdout[-3000:], flush=True)

# ==== Source Data + numbers ================================================================================================================
def speed_sheet(q, unit):
    out = []
    for _, name, _, _ in MODELS:
        d = S[name][q]
        for N in sorted({**d["meas"], **d["ext"]}): out.append({"Model": name, "Number of atoms": N, f"Value ({unit})": d["meas"].get(N, d["ext"].get(N)),
                                                             "Status": "measured" if N in d["meas"] else "extrapolated", "How": S[name]["note"][N]})
    return pd.DataFrame(out)
with pd.ExcelWriter(f"{D}/SourceData_Fig3.xlsx", engine="openpyxl") as xw:
    speed_sheet("t", "µs per MC step per temperature").to_excel(xw, sheet_name="a", index=False)
    speed_sheet("m", "GB, 32 temperatures").to_excel(xw, sheet_name="b", index=False)
    pd.DataFrame([{"System": s, "Cluster channel": CL.plain(CL.label(s, SH[s]["Gs"][g])), f"Mean |SHAP| ({u})": float(SH[s]["imp"][g])} for s, _, u in SYSTEMS for g in SH[s]["order"]]).to_excel(xw, sheet_name="c", index=False)
with pd.ExcelWriter(f"{D}/SourceData_FigS1.xlsx", engine="openpyxl") as xw:
    for q, (s, _, u) in enumerate(SYSTEMS):
        d = SH[s]; pd.DataFrame({"Configuration": d["files"], "Composition": d["labs"], "T (K)": d["T"], **{f"{CL.plain(CL.label(s, d['Gs'][g]))} ({u})": d["Phi"][:, g] for g in d["order"]}}).to_excel(xw, sheet_name="abc"[q], index=False)
json.dump(dict(exponents=EXPO, speed={n: {q: {k: {str(N): v for N, v in S[n][q][k].items()} for k in ("meas", "ext")} for q in ("t", "m")} for n in S},
               mean_abs_shap={s: {SH[s]["Gs"][g]: float(SH[s]["imp"][g]) for g in SH[s]["order"]} for s, _, _ in SYSTEMS}, beeswarm_xlim=XLIM), open(f"{D}/fig3_gnn.json", "w"), indent=1)
print("font:", FAM, "| beeswarm x range:", XLIM)
print("fit exponents (log-log slope over the measured points):")
for _, n, _, _ in MODELS: print(f"  {n:14s} time {EXPO[n]['time']:.3f} (points {sorted(S[n]['t']['meas'])})   memory {EXPO[n]['memory']:.3f}")
print("Fig. 3 text sizes:", sz6, "violations:", bad6); print("Supp. Fig. text sizes:", szS, "violations:", badS)
print("mean |SHAP|:", {s: [(CL.plain(CL.label(s, SH[s]['Gs'][g])), round(float(SH[s]['imp'][g]), 2)) for g in SH[s]['order']] for s, _, _ in SYSTEMS})
print("table:\n", T_df.to_string(index=False))

# channel strings actually drawn / written, for the four-place match check (Fig. 3c, Fig. S1, Table S1; Fig. S3 writes its own)
USED = {"fig3_c": {s_: [CL.plain(t.get_text()).replace("\n", " ") for t in axc[q].get_yticklabels()] for q, (s_, _, _) in enumerate(SYSTEMS)},
        "figS1": {s_: [CL.plain(t.get_text()).replace("\n", " ") for t in figs.axes[2 * q].get_yticklabels()] for q, (s_, _, _) in enumerate(SYSTEMS)}}
tab = pd.read_excel(f"{D}/TableS1_ground_state_shapley.xlsx"); cur = None; USED["tableS1"] = {}
for _, r_ in tab.iterrows():
    sy = str(r_["System"]) if isinstance(r_["System"], str) and r_["System"].strip() else None
    if sy: cur = {"PtNi": "ptni", "FeNiCr": "fenicr"}.get(sy, "stfo")
    ch_ = r_["Cluster channel"]
    if cur and isinstance(ch_, str) and not ch_.startswith("Total"): USED["tableS1"].setdefault(cur, []).append(ch_)
json.dump(USED, open(f"{D}/channel_labels_used_gnn.json", "w"), indent=1, ensure_ascii=False); print("labels used:", json.dumps(USED, ensure_ascii=False))
