"""What physics separates the x=0.125 near-ground-state structures that a truncated CE cannot tell apart?
For the 19 structures (DFT within 3 meV/site of the comp-125 minimum): vacancy-vacancy and Fe-vacancy geometry
descriptors on the ideal lattice (PBC), the out-of-fold residuals of icet / linear-on-bank / cluster conv, and
Spearman correlations residual-vs-descriptor.  Hypothesis: the CE residual tracks LONG-RANGE vacancy arrangement
(beyond its cluster cutoff), which the two-hop convolution sees.  Also dumps the per-structure table.

Inputs : data/new_bench/{detailed_info_corrected.csv, detailed_info.csv, <id>.cif}, data/benchmark_folds_corrected.json,
         $RESULTS/cv/{stfo_lab12/oof_nested_rep0.npy, stfo_dd_p10_d0_lab12/oof_fixed.npy, stfo_dd_p10_w10_wd5_lab12/
         oof_fixed.npy}, $RESULTS/stfo/icet_basis/X_c6_5_4.5_4.npy (icet cluster vectors, detailed_info.csv order).
Outputs: figures/stfo_group_physics.png and the markdown table next to it (.md); --out overrides.
Run    : python scripts/analysis/stfo_group_physics.py [--out figures/stfo_group_physics.png]
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import argparse, csv, json, os
import numpy as np
from scipy.stats import spearmanr
from ase.io import read
from ase.neighborlist import neighbor_list
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neuralce.paths import REPO, DATA, RESULTS

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{REPO}/figures/stfo_group_physics.png"); a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
CV = f"{RESULTS}/cv"
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{DATA}/new_bench/detailed_info_corrected.csv", f"{DATA}/benchmark_folds_corrected.json", f"{CV}/stfo_lab12/oof_nested_rep0.npy", f"{CV}/stfo_dd_p10_d0_lab12/oof_fixed.npy",
         f"{CV}/stfo_dd_p10_w10_wd5_lab12/oof_fixed.npy", f"{RESULTS}/stfo/icet_basis/X_c6_5_4.5_4.npy"], "$NCE_DATA / $NCE_RESULTS")
rows = list(csv.DictReader(open(f"{DATA}/new_bench/detailed_info_corrected.csv")))
ids = [r["id"] for r in rows]; comp = np.array([int(r["comp"]) for r in rows]); y = np.array([float(r["total_energy"]) for r in rows]) / 160
fb = json.load(open(f"{DATA}/benchmark_folds_corrected.json"))["stfo"]["items"]; cids = [it["id"] for it in fb]
pred = {"icet CE": dict(zip(ids, np.load(f"{CV}/stfo_lab12/oof_nested_rep0.npy"))),
        "linear, 10 A bank": dict(zip(cids, np.load(f"{CV}/stfo_dd_p10_d0_lab12/oof_fixed.npy"))),
        "cluster conv": dict(zip(cids, np.load(f"{CV}/stfo_dd_p10_w10_wd5_lab12/oof_fixed.npy")))}
g = (comp == 125) & (y - y[comp == 125].min() < 0.003) & np.array([i in pred["cluster conv"] for i in ids])
gids = [i for i, k in zip(ids, g) if k]; yg = y[g]; ref = yg.min()


def descriptors(atoms):
    """vacancy = Xe placeholder.  Distances on the ideal perovskite lattice (a0 = 3.96 A): O-O 1NN 2.8, 2NN 3.96 (180 deg
    through a B cation), B-O 1.98.  Returns geometry counts + per-vacancy environment."""
    sym = np.array(atoms.get_chemical_symbols()); v = np.where(sym == "Xe")[0]; fe = np.where(sym == "Fe")[0]
    i, j, d = neighbor_list("ijd", atoms, 12.0)
    out = {}
    vv = [(d[k]) for k in range(len(i)) if sym[i[k]] == "Xe" and sym[j[k]] == "Xe" and i[k] < j[k]]
    out["dVV_min"] = min(vv) if vv else 12.0; out["dVV_mean"] = np.mean(vv) if vv else 12.0
    out["nVV<4.5"] = sum(x < 4.5 for x in vv); out["nVV<6"] = sum(x < 6.0 for x in vv); out["nVV<8"] = sum(x < 8.0 for x in vv); out["nVV<10"] = sum(x < 10.0 for x in vv)
    # Fe neighbours of each vacancy (B-O 1.98 A) and Fe-Fe 2NN-through-vacancy (Fe-V-Fe 180 deg)
    nfe = []
    for vi in v:
        m = (i == vi) & (d < 2.3); nfe.append(int((sym[j[m]] == "Fe").sum()))
    out["FeperV_mean"] = float(np.mean(nfe)); out["V_with_2Fe"] = int(sum(n == 2 for n in nfe)); out["V_with_0Fe"] = int(sum(n == 0 for n in nfe))
    # vacancy-vacancy 90 deg (2.8 A) vs 180 deg (3.96 A) pairs
    out["VV_90"] = sum(abs(x - 2.8) < 0.3 for x in vv); out["VV_180"] = sum(abs(x - 3.96) < 0.3 for x in vv)
    # Fe-Fe pairs within 4.2 (1NN B-B) and long-range Fe-V distances
    ff = [d[k] for k in range(len(i)) if sym[i[k]] == "Fe" and sym[j[k]] == "Fe" and i[k] < j[k]]
    out["nFeFe<4.2"] = sum(x < 4.2 for x in ff)
    fv = [d[k] for k in range(len(i)) if sym[i[k]] == "Xe" and sym[j[k]] == "Fe"]
    out["dFeV_mean"] = float(np.mean(fv)) if fv else 12.0
    return out


tab = []
for i, yv in zip(gids, yg):
    at = read(f"{DATA}/new_bench/{i}.cif"); dsc = descriptors(at)
    rec = dict(id=i, dE_DFT=(yv - ref) * 1e3, **dsc)
    for name, p in pred.items(): rec[f"res_{name}"] = (p[i] - yv) * 1e3
    tab.append(rec)
keys = [k for k in tab[0] if k not in ("id",)]
md = ["# x=0.125 near-GS group: geometry descriptors vs out-of-fold residuals (meV/site)\n\n| id | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1) + "\n"]
for r in tab: md.append(f"| {r['id']} | " + " | ".join(f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |\n")
md.append("\n## Spearman correlation of |residual| (and of DFT dE) with each descriptor\n\n| descriptor | dE_DFT | " + " | ".join(f"|res| {n}" for n in pred) + " |\n|---|---|" + "---|" * len(pred) + "\n")
desc_keys = [k for k in keys if not k.startswith("res_") and k != "dE_DFT"]
corr = {}
for k in desc_keys:
    xk = np.array([r[k] for r in tab], float)
    if np.ptp(xk) == 0: continue
    cells = [spearmanr(xk, [r["dE_DFT"] for r in tab])[0]] + [spearmanr(xk, [abs(r[f"res_{n}"]) for r in tab])[0] for n in pred]
    corr[k] = cells; md.append(f"| {k} | " + " | ".join(f"{c:+.2f}" for c in cells) + " |\n")
# identical cluster vectors: which structures share one, and how their DFT energies differ
X = np.load(f"{RESULTS}/stfo/icet_basis/X_c6_5_4.5_4.npy"); ref_ids = [r["id"] for r in csv.DictReader(open(f"{DATA}/new_bench/detailed_info.csv"))]
pos = {i: k for k, i in enumerate(ref_ids)}; groups = {}
for i, yv in zip(gids, yg): groups.setdefault(tuple(np.round(X[pos[i]], 8)), []).append((i, (yv - ref) * 1e3))
md.append("\n## structures sharing an identical cluster vector (icet basis 6/5/4.5/4 A)\n")
for k, mem in groups.items():
    if len(mem) > 1: md.append("- " + ", ".join(f"{i} ({e:+.2f})" for i, e in mem) + f"  -> DFT spread {max(e for _, e in mem) - min(e for _, e in mem):.2f} meV/site\n")
open(a.out.replace(".png", ".md"), "w").write("".join(md)); print("".join(md))
# figure: residual vs the descriptor with the strongest correlation for icet
best = max(corr, key=lambda k: abs(corr[k][1])); xk = np.array([r[best] for r in tab], float)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for name, c, m in (("icet CE", "0.35", "s"), ("linear, 10 A bank", "tab:green", "D"), ("cluster conv", "tab:blue", "o")):
    ax[0].scatter(xk, [r[f"res_{name}"] for r in tab], color=c, marker=m, label=name)
ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xlabel(best); ax[0].set_ylabel("OOF residual (meV/site)"); ax[0].legend(fontsize=8, frameon=False)
ax[0].set_title(f"x=0.125 near-GS group: residual vs {best}")
ax[1].scatter([r["dVV_min"] for r in tab], [r["dE_DFT"] for r in tab], c="k"); ax[1].set_xlabel("min V_O-V_O distance (A)"); ax[1].set_ylabel("DFT dE from group minimum (meV/site)")
fig.tight_layout(); fig.savefig(a.out, dpi=150); print("wrote", a.out)
