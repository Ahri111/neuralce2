"""STFO benchmark accuracy table, original labels vs corrected labels (data/new_bench/label_corrections.json,
non-FM rule, 12 ids).  Each column keeps its own protocol (the one used for the original table); only the labels
changed.  MAE in meV/atom (per 160 sites), 10-fold mean +- fold std unless noted.

Inputs (results/cv/<tag>/):
  neural CE   : <tag>/{oof_fixed.npy, oof_fixed_y.npy} with tag stfo_dd_p10_w10_wd5 (original) and *_lab12 (corrected)
  icet CE     : stfo_orig3/oof_nested_rep*.npy (original), stfo_lab12/ (corrected); labels from data/new_bench/
                detailed_info.csv or detailed_info_corrected.csv (scripts/train/stfo_label_corrections.py)
  external    : CLAMM (clamm/<tag>/results.json), EPI, Disorder_GNN, CGCNN/ALIGNN (external/<d>/<tag>/*.json) are not
                part of this repository; a missing file prints "n/a" / "pending".
Output : figures/cv/benchmark_table_stfo.md (--out overrides)
Run    : python scripts/analysis/cv_stfo_table.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import argparse, glob, json, os, numpy as np, pandas as pd
from neuralce.paths import REPO, DATA, RESULTS

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{REPO}/figures/cv/benchmark_table_stfo.md"); a = ap.parse_args()
CV = f"{RESULTS}/cv"; J = lambda p: json.load(open(p)) if os.path.exists(p) else None
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{DATA}/new_bench/detailed_info.csv", f"{CV}/stfo_dd_p10_w10_wd5_lab12/oof_fixed.npy"], "$NCE_DATA/new_bench, $NCE_RESULTS/cv")
rows = []
def add(name, note, old, new): rows.append((name, note, old, new))
def fmt(m, s=None): return "pending" if m is None else (f"{m:.2f} ± {s:.2f}" if s is not None else f"{m:.2f}")
# neural CE (final recipe, OOF_FIXED 10-fold, dedup 387)
def conv(tag):
    p = f"{CV}/{tag}/oof_fixed.npy"
    if not os.path.exists(p): return None
    o = np.load(p); y = np.load(f"{CV}/{tag}/oof_fixed_y.npy"); e = np.abs(o - y) * 1e3
    bf = J(f"{DATA}/benchmark_folds.json")["stfo"]["items"]; fold = {it["id"]: it["fold"] for it in bf}
    ids = [it["id"] for it in bf]                          # same order as the training set (dedup 387)
    f = np.array([fold[i] for i in ids]) if len(ids) == len(e) else None
    per = [e[f == k].mean() for k in range(10)] if f is not None else None
    return (float(e.mean()), float(np.std(per)) if per else None)
c_old, c_new = conv("stfo_dd_p10_w10_wd5"), conv("stfo_dd_p10_w10_wd5_lab12")
add("Neural CE (ours)", "pairs 10 A, depth 2, OOF 10-fold, 387", c_old, c_new)
def icet(tag):
    # results.json "nested_mean" is the nested RMSE; the table is MAE, so recompute MAE per repeat from the saved
    # out-of-fold predictions (mean +- std over the 3 repeats).  Labels: corrected csv for *_lab12, original otherwise.
    fs = sorted(glob.glob(f"{CV}/{tag}/oof_nested_rep*.npy"))
    if not fs: return None
    csv = f"{DATA}/new_bench/detailed_info_corrected.csv" if "lab12" in tag else f"{DATA}/new_bench/detailed_info.csv"
    df = pd.read_csv(csv); y = (df.total_energy / df.n_atoms).to_numpy()
    m = [float(np.abs(np.load(f) - y).mean() * 1e3) for f in fs]
    return (float(np.mean(m)), float(np.std(m)))
add("Orthonormal CE (icet, heavy fit)", "nested cutoff x regressor, 3 reps, MAE of OOF", (icet("stfo_orig3") or icet("stfo")), icet("stfo_lab12"))
def clamm(tag):
    r = J(f"{CV}/clamm/{tag}/results.json"); return None if r is None else (r["nested"]["all"]["mae_fold_mean"], r["nested"]["all"]["mae_fold_std"])
add("Occupation CE (CLAMM, nested regressor)", "StratifiedKFold(10) on comp, 389", clamm("stfo"), clamm("stfo_lab12"))
def epi(tag):
    r = J(f"{CV}/external/epi/{tag}/summary.json"); return None if r is None else (r["mae_mean"], r["mae_std"])
add("EPI (Liu 2025, pair-only, BIC)", "benchmark folds", epi("stfo"), epi("stfo_lab12"))
def dgnn(tag):
    r = J(f"{CV}/external/disorder_gnn/{tag}/summary.json"); return None if r is None else (r["mae_mean"], r["mae_std"])
add("Disorder_GNN (Fang & Yan 2024, Transformer)", "benchmark folds, 3-seed ens", dgnn("stfo"), dgnn("stfo_lab12"))
def cg(tag, m):
    r = J(f"{CV}/external/disorder_gnn/{tag}/cv10_summary.json"); return None if r is None or m not in r else (r[m]["test_mae_mean_meV_per_160"], r[m]["test_mae_std_meV_per_160"])
add("CGCNN (fair budget)", "benchmark folds, MAE/160", (cg("gate_cgcnn_stfo_orig", "cgcnn") or cg("gate_cgcnn_stfo", "cgcnn")), cg("gate_cgcnn_stfo_lab12", "cgcnn"))
add("ALIGNN (fair budget)", "benchmark folds, MAE/160", cg("gate_cgcnn_stfo_orig", "alignn"), cg("gate_cgcnn_stfo_lab12", "alignn"))
lines = ["# STFO benchmark table — original vs corrected labels", "",
         "Corrected labels: 12 structures whose original DFT run ended in a non-FM state were re-relaxed FM (same protocol); the lower",
         "energy replaces the label (data/new_bench/label_corrections.json; mean shift -7.5 meV/site). MAE meV/atom (160 sites).", "",
         "| model | protocol | original labels | corrected labels |", "|---|---|---|---|"]
for name, note, old, new in rows:
    lines.append(f"| {name} | {note} | {fmt(*old) if old else 'n/a'} | {fmt(*new) if new else 'pending'} |")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
open(a.out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print("wrote", a.out)
