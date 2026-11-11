"""PtNi mixing (formation) enthalpy parity and MAE table, out-of-fold on the benchmark folds (all cells 108 atoms).

Pure-element references recomputed with the data set's protocol (ISIF=3 relax -> fresh single point, ISPIN=1):
Ni -617.8438 eV / 108, Pt -652.7282 eV / 108.
Models: icet CE (ridge on the site-mean conv features = icet cluster vectors, refitted here on the benchmark folds),
the neural CE seed ensemble (results/cv/nipt_v3_nomax_sel/oof_nested_ensemble.npy), and optional external baselines
whose out-of-fold csv files are not part of this repository (CLAMM occupation CE, EPI, Disorder_GNN): they are
included when the file given by --clamm / --epi / --dgnn exists and skipped otherwise.

Inputs : data/benchmark_folds.json, results/cv/nipt_v3_nomax_sel/{features.npz, oof_nested_ensemble.npy}
Outputs: figures/cv/ptni_parity.png and ptni_parity.md (MAE table, meV/atom)
Run    : python scripts/analysis/cv_ptni.py [--out figures/cv/ptni_parity.png]
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from neuralce.paths import REPO, DATA, RESULTS

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{REPO}/figures/cv/ptni_parity.png")
ap.add_argument("--E_Ni", type=float, default=-617.84377 / 108); ap.add_argument("--E_Pt", type=float, default=-652.72821 / 108)
ap.add_argument("--cv_tag", default="nipt_v3_nomax_sel", help="results/cv/<tag> of the neural CE nested CV")
ap.add_argument("--clamm", default=f"{RESULTS}/cv/clamm/nipt/oof.csv", help="CLAMM occupation-CE OOF csv (id, y, oof_nested); optional")
ap.add_argument("--epi", default=f"{RESULTS}/cv/external/epi/ptni/oof.csv", help="EPI OOF csv (id, pred); optional")
ap.add_argument("--dgnn", default=f"{RESULTS}/cv/external/disorder_gnn/ptni/oof.csv", help="Disorder_GNN OOF csv (id, pred); optional")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
it = json.load(open(f"{DATA}/benchmark_folds.json"))["ptni"]["items"]; df = pd.DataFrame(it); df["x"] = df.n_ni / 108.0
df["lin"] = df.x * a.E_Ni + (1 - df.x) * a.E_Pt; df["Hf"] = (df.y - df.lin) * 1e3
pred = {}
CV = f"{RESULTS}/cv/{a.cv_tag}"
z = np.load(f"{CV}/features.npz", allow_pickle=True); ids_f = list(z["ids"]); H, M = z["H"], z["M"]
phi = (H * M[..., None]).sum(1) / M.sum(1)[:, None]; phi = phi[np.array([ids_f.index(i) for i in df.id])]; oof = np.full(len(df), np.nan)
for f in sorted(set(df.fold[df.fold >= 0])):
    tr, te = np.where(df.fold.values != f)[0], np.where(df.fold.values == f)[0]
    oof[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-8, 4, 80))).fit(phi[tr], df.y.values[tr]).predict(phi[te])
pred["icet CE"] = pd.Series(oof, index=df.id)
if os.path.exists(a.clamm):
    o = pd.read_csv(a.clamm); suf = {i.split("__", 1)[1]: i for i in df.id}; m = {}
    for r in o.itertuples():
        if r.id in suf: m[suf[r.id]] = r.oof_nested
        else:
            k = df[np.isclose(df.y, r.y, atol=1e-9)]
            if len(k) == 1: m[k.id.iloc[0]] = r.oof_nested
    pred["CLAMM CE"] = pd.Series(m)
else:
    print(f"[skip] CLAMM CE: {a.clamm} not found")
for nm, p in (("EPI", a.epi), ("Disorder_GNN", a.dgnn)):
    if os.path.exists(p): pred[nm] = pd.read_csv(p).groupby("id").pred.mean()
    else: print(f"[skip] {nm}: {p} not found")
pred["neural CE (ours)"] = pd.Series(np.load(f"{CV}/oof_nested_ensemble.npy"), index=ids_f)
fig, axes = plt.subplots(1, len(pred), figsize=(4.1 * len(pred), 4.3), squeeze=False); axes = axes[0]; md = ["| model | n | MAE (meV/atom) |\n|---|---|---|\n"]
lo, hi = df.Hf.min() - 8, df.Hf.max() + 8
for ax, (name, p) in zip(axes, pred.items()):
    d = df.copy(); d["p"] = p.reindex(d.id).values; d = d[d.p.notna() & (d.fold >= 0)]; e = (d.p - d.y).abs() * 1e3
    sc = ax.scatter(d.Hf, (d.p - d.lin) * 1e3, c=d.x, cmap="viridis", s=16, alpha=0.9, edgecolor="none", vmin=0, vmax=1)
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.7); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel(r"$\Delta H_{mix}$ DFT (meV/atom)")
    ax.set_title(f"{name}   MAE {e.mean():.2f}", fontsize=11, color="tab:blue" if "ours" in name else "k"); md.append(f"| {name} | {len(d)} | {e.mean():.2f} |\n")
axes[0].set_ylabel(r"$\Delta H_{mix}$ predicted (meV/atom)"); cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.012, pad=0.01); cb.set_label("Ni fraction")
fig.savefig(a.out, dpi=150, bbox_inches="tight"); open(a.out.replace(".png", ".md"), "w").write("".join(md)); print("".join(md)); print("Hf range", round(df.Hf.min(), 1), round(df.Hf.max(), 1)); print("wrote", a.out)
