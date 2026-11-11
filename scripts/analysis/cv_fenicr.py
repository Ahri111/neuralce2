"""FeNiCr energy parity and accuracy table (DFT vs model energy, eV/atom, formation energy w.r.t. the pure elements of
the same data set), out-of-fold predictions on the benchmark folds, coloured by Cr content.

Models: icet + spin CE (lasso on the site-mean conv features = icet cluster vector + 12 Heisenberg-pair features,
refitted here on the benchmark folds), the neural CE seed ensemble (results/cv/fenicr_spin3_v3_sel/
oof_nested_ensemble.npy), and optional external baselines whose out-of-fold csv files are not part of this
repository (CLAMM spin CE, EPI, Disorder_GNN): included when the file given by --clamm / --epi / --dgnn exists.

Inputs : data/benchmark_folds.json, results/cv/fenicr_spin3_v3_sel/{features.npz, oof_nested_ensemble.npy}
Outputs: figures/cv/fenicr_parity.png and fenicr_parity.md (MAE / RMSE table, meV/atom, plus the 108-atom slice)
Run    : python scripts/analysis/cv_fenicr.py [--out figures/cv/fenicr_parity.png]
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import argparse, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from neuralce.paths import REPO, DATA, RESULTS

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{REPO}/figures/cv/fenicr_parity.png")
ap.add_argument("--cv_tag", default="fenicr_spin3_v3_sel", help="results/cv/<tag> of the neural CE nested CV")
ap.add_argument("--clamm", default=f"{RESULTS}/cv/clamm/fenicr/oof.csv", help="CLAMM spin-CE OOF csv (id, oof_nested); optional")
ap.add_argument("--epi", default=f"{RESULTS}/cv/external/epi/fenicr/oof.csv", help="EPI OOF csv (id, pred); optional")
ap.add_argument("--dgnn", default=f"{RESULTS}/cv/external/disorder_gnn/fenicr/oof.csv", help="Disorder_GNN OOF csv (id, pred); optional")
a = ap.parse_args()
def require(paths, hint):                                                                 # the Fe-Ni-Cr inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{RESULTS}/cv/{a.cv_tag}/oof_nested_ensemble.npy"], "$NCE_RESULTS/cv (--cv_tag)")
if "fenicr" not in json.load(open(f"{DATA}/benchmark_folds.json")):
    print(f"[skip] {os.path.basename(__file__)}: {DATA}/benchmark_folds.json has no 'fenicr' entry (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_DATA to run", flush=True); raise SystemExit(0)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
it = json.load(open(f"{DATA}/benchmark_folds.json"))["fenicr"]["items"]
df = pd.DataFrame(it); comp = df.comp.apply(lambda s: {el: int(n) for el, n in re.findall(r"([A-Z][a-z]?)(\d+)", s)})
for el in ("Fe", "Ni", "Cr"): df[el] = [c.get(el, 0) for c in comp]
df["c_Cr"] = df.Cr / df.n_atoms
ref = {el: df[(df[el] == df.n_atoms)].y.min() for el in ("Fe", "Ni", "Cr")}; print("pure-element references (lowest entry, eV/atom):", ref)
df["Ef"] = df.y - sum(df[el] / df.n_atoms * ref[el] for el in ref)
lin = df.y - df.Ef                                                     # composition baseline to subtract from predictions too
pred = {}
CV = f"{RESULTS}/cv/{a.cv_tag}"
# site-mean layer-0 features of every structure (the cluster-expansion design matrix): the shipped
# features_sitemean.npz holds the mean already; the full per-site features.npz (100 MB) is used when present.
if os.path.exists(f"{CV}/features.npz"):
    z = np.load(f"{CV}/features.npz", allow_pickle=True); H, M = z["H"], z["M"]; phi = (H * M[..., None]).sum(1) / M.sum(1)[:, None]
else:
    z = np.load(f"{CV}/features_sitemean.npz", allow_pickle=True); phi = z["phi"]
ids_f = list(z["ids"]); idx = np.array([ids_f.index(i) for i in df.id]); phi = phi[idx]
oof = np.full(len(df), np.nan)
for f in sorted(set(df.fold[df.fold >= 0])):
    tr, te = np.where(df.fold.values != f)[0], np.where(df.fold.values == f)[0]
    oof[te] = make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-8, 0, 41), cv=5, max_iter=50000)).fit(phi[tr], df.y.values[tr]).predict(phi[te])
pred["icet + spin CE"] = pd.Series(oof, index=df.id)
if os.path.exists(a.clamm): pred["CLAMM spin CE"] = pd.read_csv(a.clamm).set_index("id").oof_nested
else: print(f"[skip] CLAMM spin CE: {a.clamm} not found")
for nm, p in (("EPI", a.epi), ("Disorder_GNN", a.dgnn)):
    if os.path.exists(p): pred[nm] = pd.read_csv(p).groupby("id").pred.mean()
    else: print(f"[skip] {nm}: {p} not found")
pred["neural CE (ours)"] = pd.Series(np.load(f"{CV}/oof_nested_ensemble.npy"), index=ids_f)
fig, axes = plt.subplots(1, len(pred), figsize=(4.1 * len(pred), 4.3), squeeze=False); axes = axes[0]; md = ["| model | n | MAE | RMSE (meV/atom) | MAE, 108-atom cells |\n|---|---|---|---|---|\n"]
lo, hi = df.Ef.min() - 0.02, df.Ef.max() + 0.02
for ax, (name, p) in zip(axes, pred.items()):
    d = df.copy(); d["p"] = p.reindex(d.id).values; d = d[d.p.notna() & (d.fold >= 0)]; pf = d.p - (d.y - d.Ef)
    e = (d.p - d.y) * 1e3; big = d.n_atoms == 108
    sc = ax.scatter(d.Ef, pf, c=d.c_Cr, cmap="viridis", s=12, alpha=0.85, edgecolor="none", vmin=0, vmax=0.5)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("DFT energy (eV/atom)")
    ax.set_title(f"{name}\nRMSE {np.sqrt((e**2).mean()):.2f} / MAE {e.abs().mean():.2f} meV/atom", fontsize=10, color="tab:blue" if "ours" in name else "k")
    md.append(f"| {name} | {len(d)} | {e.abs().mean():.2f} | {np.sqrt((e**2).mean()):.2f} | {e[big].abs().mean():.2f} (n={int(big.sum())}) |\n")
axes[0].set_ylabel("model energy (eV/atom)"); cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.012, pad=0.01); cb.set_label("Cr fraction")
fig.savefig(a.out, dpi=150, bbox_inches="tight"); open(a.out.replace(".png", ".md"), "w").write("".join(md)); print("".join(md)); print("wrote", a.out)
