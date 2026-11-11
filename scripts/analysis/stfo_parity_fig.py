"""STFO parity: (A) mixing-enthalpy parity of the out-of-fold predictions of every benchmark-table model (icet heavy fit,
CLAMM occupation CE, EPI, Disorder_GNN, CGCNN, ALIGNN, linear model on the 10 A bank, cluster convolution), coloured by
Fe ratio; (B) zoom on the comp-125 near-ground-state group (DFT within 3 meV/site of the comp-125 minimum), where the
standard cluster basis gives identical cluster vectors for many structures so a CE cannot tell them apart, plus the
group MAE of every model.  All on the corrected labels (lab12 runs).  End members from the same DFT protocol.

Inputs : data/new_bench/{detailed_info_corrected.csv, detailed_info.csv}, data/benchmark_folds_corrected.json,
         $RESULTS/cv/{stfo_lab12/oof_nested_rep0.npy, clamm/stfo_lab12/oof.csv, stfo_dd_p10_d0_lab12/oof_fixed.npy,
         stfo_dd_p10_w10_wd5_lab12/oof_fixed.npy, external/{epi,disorder_gnn}/stfo_lab12/oof.csv,
         external/disorder_gnn/gate_cgcnn_stfo_lab12/cv10_results.csv}, $RESULTS/stfo/icet_basis/X_c6_5_4.5_4.npy.
Outputs: <out>_all_models.png (A), <out>_group.png (B) and the markdown table <out>.md; --out default
         figures/stfo_parity.png.
Run    : python scripts/analysis/stfo_parity_fig.py [--out figures/stfo_parity.png] [--E_STO ...] [--E_SFO ...]
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import argparse, csv, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neuralce.paths import REPO, DATA, RESULTS

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{REPO}/figures/stfo_parity.png")
ap.add_argument("--E_STO", type=float, default=-1209.4315, help="eV per 160-site cell, SrTiO3 (ISIF=3 run, same protocol)")
ap.add_argument("--E_SFO", type=float, default=-929.75272, help="eV per 160-site cell, brownmillerite Sr2Fe2O5")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
CV = f"{RESULTS}/cv"
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{DATA}/new_bench/detailed_info_corrected.csv", f"{DATA}/new_bench/detailed_info.csv", f"{DATA}/benchmark_folds_corrected.json", f"{CV}/stfo_lab12/oof_nested_rep0.npy",
         f"{CV}/clamm/stfo_lab12/oof.csv", f"{CV}/stfo_dd_p10_d0_lab12/oof_fixed.npy", f"{CV}/stfo_dd_p10_w10_wd5_lab12/oof_fixed.npy", f"{CV}/external/epi/stfo_lab12/oof.csv",
         f"{CV}/external/disorder_gnn/stfo_lab12/oof.csv"], "$NCE_DATA / $NCE_RESULTS/cv")
rows = list(csv.DictReader(open(f"{DATA}/new_bench/detailed_info_corrected.csv")))
ids = [r["id"] for r in rows]; comp = np.array([int(r["comp"]) for r in rows]); y = np.array([float(r["total_energy"]) for r in rows]) / 160
x = comp / 1000.0; lin = ((1 - x) * a.E_STO + x * a.E_SFO) / 160
dH = lambda e, l: (e - l) * 1e3                                # meV per site (160-site cell)
# out-of-fold predictions (eV/site), aligned by id
pred = {}
pred["icet CE"] = dict(zip(ids, np.load(f"{CV}/stfo_lab12/oof_nested_rep0.npy")))   # one rep (table protocol), not the 3-rep mean
cl = list(csv.DictReader(open(f"{CV}/clamm/stfo_lab12/oof.csv")))
pred["CLAMM CE"] = {r["id"]: float(r["oof_nested"]) for r in cl}
fb = json.load(open(f"{DATA}/benchmark_folds_corrected.json"))["stfo"]["items"]; cids = [it["id"] for it in fb]
pred["linear, 10 A bank"] = dict(zip(cids, np.load(f"{CV}/stfo_dd_p10_d0_lab12/oof_fixed.npy")))   # depth 0 on the conv's cluster bank
pred["cluster conv"] = dict(zip(cids, np.load(f"{CV}/stfo_dd_p10_w10_wd5_lab12/oof_fixed.npy")))
ext = {}
for name, path in (("EPI", f"{CV}/external/epi/stfo_lab12/oof.csv"), ("Disorder_GNN", f"{CV}/external/disorder_gnn/stfo_lab12/oof.csv")):
    o = list(csv.DictReader(open(path))); acc = {}
    for r in o: acc.setdefault(r["id"], []).append(float(r["pred"]))
    ext[name] = {k: float(np.mean(v)) for k, v in acc.items()}
try:
    import ast
    for r in csv.DictReader(open(f"{CV}/external/disorder_gnn/gate_cgcnn_stfo_lab12/cv10_results.csv")):
        nm = r["method"].upper(); ext.setdefault(nm, {})
        for i, pv in zip(ast.literal_eval(r["test_ids"]), ast.literal_eval(r["test_pred"])): ext[nm][i] = float(pv) / 160.0
except Exception as ex: print("cgcnn/alignn skipped:", ex)
keep = np.array([i in pred["cluster conv"] for i in ids])        # the dedup 387
ids_k = [i for i, k in zip(ids, keep) if k]; y_k = y[keep]; x_k = x[keep]; lin_k = lin[keep]
# ---- figure A: mixing-enthalpy parity of EVERY table model (out-of-fold, corrected labels) ----
order = [("icet CE", pred["icet CE"]), ("CLAMM CE", pred["CLAMM CE"]), ("EPI", ext.get("EPI")), ("Disorder_GNN", ext.get("Disorder_GNN")),
         ("CGCNN", ext.get("CGCNN")), ("ALIGNN", ext.get("ALIGNN")), ("linear, 10 A bank", pred["linear, 10 A bank"]), ("cluster conv (ours)", pred["cluster conv"])]
order = [(n, p) for n, p in order if p]
cmap = plt.cm.viridis; norm = plt.Normalize(0.125, 0.75)
figA, axA = plt.subplots(2, 4, figsize=(17, 8.2)); axA = axA.ravel()
md = ["# STFO parity (corrected labels, out-of-fold)\n", f"end members per 160-site cell: SrTiO3 {a.E_STO} eV, Sr2Fe2O5 {a.E_SFO} eV\n",
      "| model | MAE all (meV/site) | comp-125 near-GS group: n, MAE, spread(pred)/spread(DFT) |\n|---|---|---|\n"]
g = (comp == 125) & (y - y[comp == 125].min() < 0.003) & keep
gids = [i for i, k in zip(ids, g) if k]; yg = y[g]
lo, hi = dH(y_k, lin_k).min() - 3, dH(y_k, lin_k).max() + 3
for ax, (name, p) in zip(axA, order):
    have = np.array([i in p for i in ids_k]); pk = np.array([p[i] for i, h in zip(ids_k, have) if h]); yk = y_k[have]; lk = lin_k[have]; xk = x_k[have]
    e = np.abs(pk - yk) * 1e3
    sc = ax.scatter(dH(yk, lk), (pk - lk) * 1e3, c=xk, cmap=cmap, norm=norm, s=12, alpha=0.85, edgecolor="none")
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.7); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f"{name}   MAE {e.mean():.2f}", fontsize=11, color="tab:blue" if "ours" in name else "k"); ax.set_xlabel(r"$\Delta H_{mix}$ DFT (meV/site)")
    pg = np.array([p[i] for i in gids if i in p]); ygg = np.array([v for i, v in zip(gids, yg) if i in p])
    md.append(f"| {name} | {e.mean():.2f} (n={int(have.sum())}) | n={len(pg)}, MAE {np.abs(pg-ygg).mean()*1e3:.2f}, spread {np.ptp(pg)*1e3:.2f}/{np.ptp(ygg)*1e3:.2f} meV |\n")
for ax in axA[::4]: ax.set_ylabel(r"$\Delta H_{mix}$ predicted (meV/site)")
cb = figA.colorbar(sc, ax=axA.tolist(), fraction=0.015, pad=0.01); cb.set_label("Fe ratio x")
figA.savefig(a.out.replace(".png", "_all_models.png"), dpi=150, bbox_inches="tight"); print("wrote", a.out.replace(".png", "_all_models.png"))
# ---- figure B: the x = 0.125 near-ground-state group ----
fig, axes_b = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.05, 0.9])); axes = [None, None, None, axes_b[0], axes_b[1]]
# (b) zoom: near-GS group of comp 125, energies relative to the group minimum
ax = axes[3]; ref = yg.min(); cols = {"icet CE": "0.35", "CLAMM CE": "tab:orange", "linear, 10 A bank": "tab:green", "cluster conv": "tab:blue"}; mk = {"icet CE": "s", "CLAMM CE": "^", "linear, 10 A bank": "D", "cluster conv": "o"}
for name, p in pred.items():
    pg = np.array([p[i] for i in gids])
    ax.scatter((yg - ref) * 1e3, (pg - ref) * 1e3, s=34, marker=mk[name], color=cols[name], label=name, alpha=0.9)
ax.plot([-0.5, 3.5], [-0.5, 3.5], "k-", lw=0.7); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-1.0, 5.5); ax.set_xlabel("DFT, relative to group minimum (meV/site)"); ax.set_ylabel("predicted (meV/site)")   # one linear-model point at +13 is clipped (say so in the caption)
ax.set_title(f"x = 0.125 near-ground-state group (n={len(gids)})"); ax.legend(fontsize=8, frameon=False)
# how many distinct cluster vectors in the standard basis
try:
    X = np.load(f"{RESULTS}/stfo/icet_basis/X_c6_5_4.5_4.npy"); ref_ids = [r["id"] for r in csv.DictReader(open(f"{DATA}/new_bench/detailed_info.csv"))]
    pos = {i: k for k, i in enumerate(ref_ids)}; Xg = np.round(X[[pos[i] for i in gids]], 8)
    nd = len({tuple(r) for r in Xg}); ax.text(0.03, 0.93, f"distinct cluster vectors\n(icet basis 6/5/4.5/4 A): {nd}/{len(gids)}", transform=ax.transAxes, fontsize=8, va="top")
    md.append(f"\ndistinct cluster vectors in the icet basis [6,5,4.5,4] within the group: {nd}/{len(gids)}\n")
except Exception as ex:
    md.append(f"\n(cluster-vector count skipped: {ex})\n")
# (c) group MAE of every model in the table
allm = dict(pred); allm.update(ext); names, vals = [], []
for name, p in allm.items():
    pg = np.array([p[i] for i in gids if i in p]); yy = np.array([yv for i, yv in zip(gids, yg) if i in p])
    if len(pg) < len(gids) // 2: continue
    names.append(name); vals.append(np.abs(pg - yy).mean() * 1e3)
    md.append(f"| {name} (group only) | - | n={len(pg)}, MAE {vals[-1]:.2f} |\n")
ax = axes[4]; order = np.argsort(vals)[::-1]
ax.barh([names[i] for i in order], [vals[i] for i in order], color=["tab:blue" if names[i] == "cluster conv" else "0.6" for i in order])
ax.set_xlabel("MAE within the x=0.125 near-GS group (meV/site)"); ax.set_title("all table models")
fig.tight_layout(); fig.savefig(a.out.replace(".png", "_group.png"), dpi=150); open(a.out.replace(".png", ".md"), "w").write("".join(md)); print("wrote", a.out)
print("".join(md))
