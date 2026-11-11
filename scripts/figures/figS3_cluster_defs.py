"""Supplementary Fig. S3 (cluster channel definitions), 2D schematic, 183 x 55 mm, lattice coordinates only (no calculation).
Groups side by side: (a) PtNi 5 channels, (b) SrTi1-xFexO3-x/2 6 channels, (c) FeNiCr 7 channels; within a group the cells sit in two rows
(3 / 3 / 4 columns).  One cell per channel of Table S1, one representative cluster drawn per channel; the full list is Supplementary Table S2
(scripts/figures/tableS2_cluster_catalogue.py).  Channel names from scripts/figures/channel_labels.py; labels are wrapped to the measured cell width.
fcc: (111) triangular 5 x 5 patch; 2NN and 3NN+ on a (100) 4 x 4 square patch.  Perovskite: (001) B-O plane, B on a 4 x 4 grid, O at the midpoints.
No per-cell plane note (every cell states only its channel name); the planes are given once in the caption.
Grey dots = surrounding lattice sites, coloured dots = sites of the cluster (black edge), black lines join them; arrows = spin channels.
Inputs:  none (channel names only).
Outputs: figures/figS3_cluster_defs.{pdf,svg,png}, figures/channel_labels_used_s3.json (the channel strings drawn, for the cross-figure consistency check).
Run:     python scripts/figures/figS3_cluster_defs.py"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import natstyle as ns
from neuralce.paths import REPO, MODELS
import matplotlib as mpl; mpl.use("Agg"); FAM = ns.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import channel_labels as CL                                                              # names shared with Fig. 3c, Fig. S1 and Table S1

D = f"{REPO}/figures"; os.makedirs(D, exist_ok=True); PURPLE, OGREY, BG = "#5b2a9e", "#9a9a9a", "#d0d0d0"
W, H = 183.0, 55.0; fig = plt.figure(figsize=(W / 25.4, H / 25.4))
PT = 72 / 25.4; S_BG, S_CL = (1.4 * PT) ** 2, (2.0 * PT) ** 2                         # scatter sizes: 1.4 mm / 2.0 mm diameters
VO = r"V$_\mathrm{O}$"


def tri(n=5):                                                                           # fcc (111) triangular patch, rows offset alternately
    return np.array([(i + 0.5 * (j % 2), j * np.sqrt(3) / 2) for j in range(n) for i in range(n)])
def sq(n=4): return np.array([(i, j) for j in range(n) for i in range(n)], float)
def pv(n=4):                                                                            # perovskite (001): B grid + O midpoints
    B = [(i, j) for j in range(n) for i in range(n)]; O = [(i + 0.5, j) for j in range(n) for i in range(n - 1)] + [(i, j + 0.5) for j in range(n - 1) for i in range(n)]
    return np.array(B + O, float)
T3 = np.sqrt(3) / 2
# channel -> (lattice, sites [(x, y, colour, spin arrow)], bonds [(i, j)], plane note)
P = lambda xy, c=PURPLE, s=False: (xy[0], xy[1], c, s)
def pair111(spin=False): return ("tri", [P((1, 2 * T3), s=spin), P((2, 2 * T3), s=spin)], [(0, 1)], "")
def pair2nn(spin=False): return ("sq", [P((0, 1), s=spin), P((2, 1), s=spin)], [(0, 1)], "")
def pair3nn(): return ("sq", [P((0, 0)), P((3, 1))], [(0, 1)], "")          # grid step a/2: (3,1) -> a*sqrt(10)/2 = 5NN (6.20 A PtNi, 5.69 A FeNiCr); (2,2) would be 4NN
def trip111(spin=False): return ("tri", [P((1, 2 * T3), s=spin), P((2, 2 * T3), s=spin), P((1.5, 3 * T3), s=spin)], [(0, 1), (1, 2), (2, 0)], "")
def quad111(): return ("tri", [P((1, 2 * T3)), P((2, 2 * T3)), P((2.5, 3 * T3)), P((1.5, 3 * T3))], [(0, 1), (1, 2), (2, 3), (3, 0)], "")
GROUPS = [
    ("a", "PtNi", [("pairs 2NN", pair2nn()), ("pairs 1NN", pair111()), ("triplets", trip111()), ("pairs 3NN+", pair3nn()), ("quadruplets", quad111())]),
    ("b", r"SrTi$_{1-\mathit{x}}$Fe$_{\mathit{x}}$O$_{3-\mathit{x}/2}$", [
        ("triplets", ("pv", [P((1, 1)), P((1.5, 1), OGREY), P((2, 1))], [(0, 1), (1, 2)], "")),
        # every drawn cluster is a member of its channel (Table S2): B site + 4 O in plane = a 5-site member [2.00 x4, 2.82 x4, 3.99 x2];
        # O-O a*sqrt(2) = 5.63 A is the commonest kind of "longer pair" (45 of 59 are O-O); the 3.45-4.03 A channel holds O-O pairs only (180 deg = 3.96 A)
        ("quadruplets", ("pv", [P((1, 1)), P((0.5, 1), OGREY), P((1.5, 1), OGREY), P((1, 0.5), OGREY), P((1, 1.5), OGREY)], [(0, 1), (0, 2), (0, 3), (0, 4)], "")),
        ("O-O pair 2.8 A  (V_O-V_O, 90 deg)", ("pv", [P((1.5, 1), "k"), P((1, 1.5), "k")], [(0, 1)], "")),
        ("longer pairs", ("pv", [P((0.5, 1), OGREY), P((1.5, 2), OGREY)], [(0, 1)], "")),
        ("B-B 1NN and O-O 180 deg pairs (3.96 A)", ("pv", [P((0.5, 1), OGREY), P((1.5, 1), OGREY)], [(0, 1)], "")),
        ("B-O pair 1.98 A  (Fe/Ti - V_O)", ("pv", [P((1, 1)), P((1.5, 1), "k")], [(0, 1)], ""))]),
    ("c", "FeNiCr", [("spin pairs 2NN", pair2nn(True)), ("other spin terms", trip111(True)), ("spin pairs 1NN", pair111(True)),
                     ("triplets", trip111()), ("chem. pairs 2NN", pair2nn()), ("chem. pairs 1NN", pair111()), ("chem. pairs 3NN+", pair3nn())]),
]                                                                                       # every channel of Table S1 has a cell
LAT = {"tri": tri(), "sq": sq(), "pv": pv()}
NCOL = [-(-len(cells) // 2) for _, _, cells in GROUPS]                                 # two rows per group -> 3 / 3 / 4 columns
GAP, X0 = 4.0, 2.0; CW = (W - X0 - 1.0 - (len(GROUPS) - 1) * GAP) / sum(NCOL)          # one cell width for all groups
ROWS = [(36.0, 13.0), (12.5, 13.0)]                                                     # (bottom, height) of the drawing area of rows 1 and 2 (mm); names up to 4 lines
SYSKEY = {"a": "ptni", "b": "stfo", "c": "fenicr"}; USED = {}
# the groups of the systems whose cluster-convolution model is present are drawn (same cells, same positions); the others are skipped
MODEL_OF = {"ptni": "nipt_v3_nomax_sel", "stfo": "stfo_dd_p10_w10_wd5_aug467_al1a_w0", "fenicr": "fenicr_s2b_mm_c65_d1"}   # as scripts/analysis/channel_table.py
def _have(sysk):
    p = f"{MODELS}/nce/{MODEL_OF[sysk]}/model.pkl"
    if os.path.exists(p): return True
    print(f"[skip] figS3_cluster_defs.py: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_MODELS/nce/ to run", flush=True); return False
DRAW = [g for g, (letter, _, _) in enumerate(GROUPS) if _have(SYSKEY[letter])]
_rend = fig.canvas.get_renderer()
def text_mm(s):
    t = fig.text(0, 0, s, fontsize=6); w = t.get_window_extent(_rend).width / fig.dpi * 25.4; t.remove(); return w
def cell_name(sysk, key, max_lines):                                                    # the shared string, wrapped as wide as the cell allows (break at " (" first)
    s = CL.label(sysk, key)
    for wc in range(22, 9, -1):
        name = CL.wrap_to(s, wc, head_width=min(16, wc))
        if max(text_mm(l) for l in name.split("\n")) <= CW - 0.8: break
    assert name.count("\n") + 1 <= max_lines, f"{key}: {name!r} needs more than {max_lines} lines in a {CW:.1f} mm cell"
    return name
for g, (letter, gname, cells) in enumerate(GROUPS):
    if g not in DRAW: continue
    nc = NCOL[g]; gx = X0 + sum(NCOL[:g]) * CW + g * GAP
    fig.text((gx + nc * CW / 2) / W, 49.5 / H, gname, fontsize=7, ha="center", va="bottom")
    fig.text(gx / W, 49.5 / H, letter, fontsize=8, fontweight="bold", ha="left", va="bottom")
    for k, (key, (lat, sites, bonds, note)) in enumerate(cells):
        name = cell_name(SYSKEY[letter], key, 4 - (1 if note else 0)); USED.setdefault(SYSKEY[letter], []).append(CL.plain(CL.label(SYSKEY[letter], key)))
        r, c = divmod(k, nc); yb, hh = ROWS[r]; ax = fig.add_axes([(gx + c * CW) / W, yb / H, CW / W, hh / H]); ax.set_axis_off(); ax.set_aspect("equal", adjustable="datalim")
        sb, sc = (S_BG, S_CL) if lat != "pv" else ((1.0 * PT) ** 2, (1.5 * PT) ** 2)   # perovskite: B-O is half a grid step, smaller dots so neighbours do not merge
        L = LAT[lat]; ax.scatter(L[:, 0], L[:, 1], s=sb, color=BG, lw=0, zorder=1)
        for i, j in bonds: ax.plot([sites[i][0], sites[j][0]], [sites[i][1], sites[j][1]], color="k", lw=0.6, zorder=2, solid_capstyle="round")
        for x, y, col, spin in sites:
            ax.scatter([x], [y], s=sc, color=col, edgecolors="k", linewidths=0.4, zorder=3)
            if spin: ax.annotate("", (x, y + 0.24), xytext=(x, y - 0.24), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.6, mutation_scale=3, shrinkA=0, shrinkB=0), zorder=4)
        pad = 0.45; ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].min() - pad, L[:, 1].max() + pad)
        fig.text((gx + (c + 0.5) * CW) / W, (yb - 0.8) / H, name + ("\n" + note if note else ""), fontsize=6, ha="center", va="top", linespacing=1.1)
hl = [Line2D([], [], ls="none", marker="o", ms=2.0 * PT, mfc=PURPLE, mec="k", mew=0.4), Line2D([], [], ls="none", marker="o", ms=2.0 * PT, mfc=OGREY, mec="k", mew=0.4),
      Line2D([], [], ls="none", marker="o", ms=2.0 * PT, mfc="k", mec="k", mew=0.4)]
if any(GROUPS[g][0] == "b" for g in DRAW):                                             # perovskite legend: only with the SrTi1-xFexO3-x/2 group
    fig.legend(hl, ["B site (Fe/Ti)", "O", VO], loc="lower left", bbox_to_anchor=(1.0 / W, 0.5 / H), ncol=3, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
fig.savefig(f"{D}/figS3_cluster_defs.pdf"); fig.savefig(f"{D}/figS3_cluster_defs.svg"); fig.savefig(f"{D}/figS3_cluster_defs.png", dpi=600); print("wrote figS3_cluster_defs.pdf / .svg / .png", flush=True)
sz, bad = ns.text_check(fig, "abc"); print("font:", FAM, "| text sizes:", sz, "| violations:", bad)
json.dump({"figS3": USED}, open(f"{D}/channel_labels_used_s3.json", "w"), indent=1, ensure_ascii=False); print("labels used:", json.dumps(USED, ensure_ascii=False))
