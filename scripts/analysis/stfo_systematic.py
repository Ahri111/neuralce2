"""Systematic (not anecdotal) evidence on STFO, corrected labels, all out-of-fold:
(a) pairwise ordering accuracy: for every pair of structures at the same composition, how often does the model put them
    in the DFT order, as a function of the DFT energy gap.  This is what Metropolis sampling consumes; no GS needed.
(b) the CE dilemma: as the cluster basis grows, fewer structures are indistinguishable (top) but the cross-validated
    error rises (bottom); the cluster convolution uses the large bank without that penalty.

Inputs : data/new_bench/detailed_info_corrected.csv, data/benchmark_folds_corrected.json;
         out-of-fold predictions in $RESULTS/cv/: stfo_lab12/oof_nested_rep0.npy (icet CE), clamm/stfo_lab12/oof.csv,
         stfo_dd_p10_d0_lab12/oof_fixed.npy (linear model on the 10 A bank), stfo_dd_p10_w10_wd5_lab12/oof_fixed.npy
         (cluster conv), external/{epi,disorder_gnn}/stfo_lab12/oof.csv, external/disorder_gnn/gate_cgcnn_stfo_lab12/
         cv10_results.csv (CGCNN / ALIGNN); icet cluster-vector matrices $RESULTS/stfo/icet_basis/X_c*.npy (389 rows in
         the order of data/new_bench/detailed_info.csv).
Outputs: $RESULTS/stfo/stfo_systematic.json and figures/stfo_systematic.png (--out overrides the figure path).
Run    : python scripts/analysis/stfo_systematic.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import argparse, csv, json, os, ast, glob
import numpy as np
from sklearn.linear_model import RidgeCV
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neuralce.paths import REPO, DATA, RESULTS

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{REPO}/figures/stfo_systematic.png"); a = ap.parse_args()
CV = f"{RESULTS}/cv"
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{DATA}/new_bench/detailed_info_corrected.csv", f"{DATA}/benchmark_folds_corrected.json", f"{CV}/stfo_lab12/oof_nested_rep0.npy", f"{CV}/clamm/stfo_lab12/oof.csv",
         f"{CV}/stfo_dd_p10_d0_lab12/oof_fixed.npy", f"{CV}/stfo_dd_p10_w10_wd5_lab12/oof_fixed.npy", f"{CV}/external/epi/stfo_lab12/oof.csv", f"{CV}/external/disorder_gnn/stfo_lab12/oof.csv",
         f"{CV}/external/disorder_gnn/gate_cgcnn_stfo_lab12/cv10_results.csv"], "$NCE_DATA / $NCE_RESULTS")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); os.makedirs(f"{RESULTS}/stfo", exist_ok=True)
rows = list(csv.DictReader(open(f"{DATA}/new_bench/detailed_info_corrected.csv"))); ids_all = [r["id"] for r in rows]
y = {r["id"]: float(r["total_energy"]) / 160 for r in rows}; comp = {r["id"]: int(r["comp"]) for r in rows}
fb = json.load(open(f"{DATA}/benchmark_folds_corrected.json"))["stfo"]["items"]; cids = [it["id"] for it in fb]; fold = {it["id"]: it["fold"] for it in fb}
P = {"icet CE": dict(zip(ids_all, np.load(f"{CV}/stfo_lab12/oof_nested_rep0.npy"))),
     "CLAMM CE": {r["id"]: float(r["oof_nested"]) for r in csv.DictReader(open(f"{CV}/clamm/stfo_lab12/oof.csv"))},
     "linear, same 10 A bank": dict(zip(cids, np.load(f"{CV}/stfo_dd_p10_d0_lab12/oof_fixed.npy"))),
     "cluster conv": dict(zip(cids, np.load(f"{CV}/stfo_dd_p10_w10_wd5_lab12/oof_fixed.npy")))}
for name, path in (("EPI (pair only)", "epi/stfo_lab12/oof.csv"), ("Disorder_GNN", "disorder_gnn/stfo_lab12/oof.csv")):
    acc = {}
    for r in csv.DictReader(open(f"{CV}/external/{path}")): acc.setdefault(r["id"], []).append(float(r["pred"]))
    P[name] = {k: float(np.mean(v)) for k, v in acc.items()}
for r in csv.DictReader(open(f"{CV}/external/disorder_gnn/gate_cgcnn_stfo_lab12/cv10_results.csv")):
    nm = r["method"].upper(); P.setdefault(nm, {})
    for i, pv in zip(ast.literal_eval(r["test_ids"]), ast.literal_eval(r["test_pred"])): P[nm][i] = float(pv) / 160.0
# ---------- (a) pairwise ordering accuracy ----------
bins = [0, 1, 2, 3, 5, 8, 12, 20, 60]; ctr = [0.5 * (bins[k] + bins[k + 1]) for k in range(len(bins) - 1)]
res = {}
for name, p in P.items():
    ok = [[] for _ in ctr]
    for c in sorted(set(comp.values())):
        g = [i for i in cids if comp[i] == c and i in p]; yy = np.array([y[i] for i in g]) * 1e3; pp = np.array([p[i] for i in g]) * 1e3
        a_, b_ = np.triu_indices(len(g), 1); dy = yy[a_] - yy[b_]; dp = pp[a_] - pp[b_]; gap = np.abs(dy)
        for k in range(len(ctr)):
            m = (gap >= bins[k]) & (gap < bins[k + 1]) & (gap > 1e-6); ok[k] += list((np.sign(dy[m]) == np.sign(dp[m])).astype(float))
    res[name] = ([np.mean(o) if o else np.nan for o in ok], [len(o) for o in ok])
print("pairs per gap bin:", res["cluster conv"][1])
print("| model | " + " | ".join(f"{bins[k]}-{bins[k+1]}" for k in range(len(ctr))) + " | (ordering accuracy by DFT gap, meV/site)\n|---|" + "---|" * len(ctr))
for name, (acc_, n_) in res.items(): print(f"| {name} | " + " | ".join(f"{v:.2f}" for v in acc_) + " |")
# ---------- (b) CE dilemma: basis size vs indistinguishable structures vs CV error ----------
keep = [i for i in ids_all if i in fold]; idx = np.array([ids_all.index(i) for i in keep]); yk = np.array([y[i] for i in keep]); fk = np.array([fold[i] for i in keep]); ck = np.array([comp[i] for i in keep])
basis = []
for f in sorted(glob.glob(f"{RESULTS}/stfo/icet_basis/X_c*.npy")):
    X = np.load(f)[idx]; tag = os.path.basename(f)[3:-4]
    g = {}
    for i, x, c in zip(keep, np.round(X, 8), ck): g.setdefault((c,) + tuple(x), []).append(i)
    n_deg = sum(len(v) for v in g.values() if len(v) > 1)
    oof = np.zeros(len(keep))
    for k in sorted(set(fk)):
        tr, te = fk != k, fk == k; oof[te] = RidgeCV(alphas=np.logspace(-6, 4, 40)).fit(X[tr], yk[tr]).predict(X[te])
    basis.append((X.shape[1], tag, n_deg, float(np.abs(oof - yk).mean() * 1e3))); print(f"basis {tag:14s} K={X.shape[1]:5d}  indistinguishable {n_deg:3d}  ridge CV MAE {basis[-1][3]:.2f}", flush=True)
basis.sort()
json.dump(dict(ordering={k: dict(acc=v[0], n=v[1]) for k, v in res.items()}, bins=bins, basis=basis), open(f"{RESULTS}/stfo/stfo_systematic.json", "w"), indent=1)
# ---------- figure ----------
fig = plt.figure(figsize=(14.5, 5.6)); gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.12, wspace=0.22)
ax = fig.add_subplot(gs[:, 0]); col = {"cluster conv": "#1f4fd8", "icet CE": "#7570b3", "linear, same 10 A bank": "#2ca02c", "CLAMM CE": "#8c564b", "EPI (pair only)": "#d95f02", "ALIGNN": "0.45", "CGCNN": "0.6", "Disorder_GNN": "0.75"}
for name in ["EPI (pair only)", "Disorder_GNN", "CGCNN", "ALIGNN", "CLAMM CE", "icet CE", "linear, same 10 A bank", "cluster conv"]:
    if name in res: ax.plot(range(len(ctr)), res[name][0], marker="o", ms=5, lw=2.6 if name == "cluster conv" else 1.6, color=col[name], label=name)
ax.axhline(0.5, color="k", ls=":", lw=1); ax.text(len(ctr) - 1, 0.505, "coin flip", ha="right", fontsize=9)
ax.set_xticks(range(len(ctr))); ax.set_xticklabels([f"{bins[k]}-{bins[k+1]}" for k in range(len(ctr))]); ax.set_xlabel("DFT energy gap between two structures of the same composition (meV/site)")
ax.set_ylabel("fraction of pairs the model orders like DFT"); ax.set_ylim(0.42, 1.01); ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.grid(color="0.9"); ax.set_axisbelow(True)
ax.set_title(f"(a)  which of two structures is lower?  all {sum(res['cluster conv'][1])} same-composition pairs, out of fold", fontsize=11.5)
K = [b[0] for b in basis]; ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax1.plot(K, [b[2] for b in basis], "o-", color="#7570b3"); ax1.set_xscale("log"); ax1.set_ylabel("structures the basis\ncannot tell apart"); ax1.grid(color="0.9"); plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_title("(b)  the cluster-expansion dilemma (linear model, growing basis)", fontsize=11.5)
ax2.plot(K, [b[3] for b in basis], "o-", color="#7570b3", label="linear CE (ridge), 10-fold CV")
conv_mae = float(np.mean([abs(P["cluster conv"][i] - y[i]) for i in cids]) * 1e3); lin_mae = float(np.mean([abs(P["linear, same 10 A bank"][i] - y[i]) for i in cids]) * 1e3)
ax2.axhline(conv_mae, color="#1f4fd8", lw=2, label=f"cluster conv on a 10 A bank ({conv_mae:.2f})"); ax2.set_xlabel("number of cluster functions in the basis"); ax2.set_ylabel("CV MAE (meV/site)")
ax2.legend(frameon=False, fontsize=9); ax2.grid(color="0.9")
fig.savefig(a.out, dpi=150, bbox_inches="tight"); print("wrote", a.out)
