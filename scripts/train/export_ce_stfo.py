"""STFO orthonormal CE (icet) for the MC framework, fitted on the augmented set (data/new_bench_aug467: 465 structures
after dedup, 17 label corrections) plus optional extra DFT structures (env EXTRA = pickle {atoms, y}).
Lasso (default) or ridge on the standardised cluster vectors, cutoffs CUT (default 6,5,4.5,4); the scaler is folded
into the parameters.  Written in the MC framework's convention: parameters x 160 = total energy of the 160-site
base cell (the framework multiplies by the supercell repeats).

Inputs : data/new_bench_aug467/{detailed_info.csv, mapped.pkl, label_corrections_al.json},
         data/new_bench/{dedup_stfo.json, label_corrections.json}, models/ce/stfo/icet_q56.ce (primitive; convention check).
Output : <out .ce>  (the shipped file is models/ce/stfo/icet_aug467_al2b_c6_5_4.5_4.ce)
Env    : CUT (6,5,4.5,4), REG (lasso | ridge), EXTRA (none; extra structures pickle, see the docstring of train_nce.py)
Run    : python scripts/train/export_ce_stfo.py models/ce/stfo/<name>.ce
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, sys, json, pickle, numpy as np, pandas as pd
from icet import ClusterExpansion, ClusterSpace
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from neuralce.paths import DATA, MODELS

for _p in (f"{DATA}/new_bench_aug467/detailed_info.csv", f"{DATA}/new_bench_aug467/mapped.pkl", f"{DATA}/new_bench_aug467/label_corrections_al.json", f"{DATA}/new_bench/dedup_stfo.json",
           f"{DATA}/new_bench/label_corrections.json", f"{MODELS}/ce/stfo/icet_q56.ce"):   # the STFO inputs are not shipped: skip cleanly when they are missing
    if not os.path.exists(_p):
        print(f"[skip] {os.path.basename(__file__)}: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_DATA/new_bench*, $NCE_MODELS/ce/stfo to run", flush=True); raise SystemExit(0)
out = sys.argv[1]; CUT = [float(c) for c in os.environ.get("CUT", "6,5,4.5,4").split(",")]
df = pd.read_csv(f"{DATA}/new_bench_aug467/detailed_info.csv"); atoms = pickle.load(open(f"{DATA}/new_bench_aug467/mapped.pkl", "rb")); df["y"] = df.total_energy / df.n_atoms
dd = json.load(open(f"{DATA}/new_bench/dedup_stfo.json"))
for grp in dd["groups"]: df.loc[df.id.isin(grp), "y"] = df.loc[df.id.isin(grp), "y"].mean()
keep = ~df.id.isin(dd["drop"]); atoms = [a for a, k in zip(atoms, keep) if k]; df = df[keep].reset_index(drop=True); fx = dict(json.load(open(f"{DATA}/new_bench/label_corrections.json"))["corrections"]); fx.update(json.load(open(f"{DATA}/new_bench_aug467/label_corrections_al.json"))["corrections"])
for sid, e in fx.items(): df.loc[df.id.astype(str) == sid, "y"] = float(e)
y = df.y.to_numpy()
if os.environ.get("EXTRA"): ex = pickle.load(open(os.environ["EXTRA"], "rb")); atoms = list(atoms) + list(ex["atoms"]); y = np.r_[y, np.asarray(ex["y"], float)]; print(f"+{len(ex['y'])} extra structures")
cs0 = ClusterExpansion.read(f"{MODELS}/ce/stfo/icet_q56.ce").get_cluster_space_copy(); cs = ClusterSpace(cs0.primitive_structure, CUT, cs0.chemical_symbols)
X = np.array([cs.get_cluster_vector(a) for a in atoms]); assert np.allclose(X[:, 0], 1.0)
REG = os.environ.get("REG", "lasso")                                                          # same pipeline as scripts/train/fit_ce_stfo.py (winner for [6,5,4.5,4]: StandardScaler + LassoCV)
mk = {"lasso": lambda: LassoCV(alphas=np.logspace(-6, 0, 41), cv=5, max_iter=50000, n_jobs=1), "ridge": lambda: RidgeCV(alphas=np.logspace(-6, 4, 60))}[REG]
oof = np.full(len(y), np.nan)
for tr, te in KFold(10, shuffle=True, random_state=42).split(X): oof[te] = make_pipeline(StandardScaler(), mk()).fit(X[tr], y[tr]).predict(X[te])
print(f"10-fold CV ({REG}, standardized): RMSE {np.sqrt(np.mean((oof - y) ** 2)) * 1e3:.2f}  MAE {np.abs(oof - y).mean() * 1e3:.2f} meV/site")
pipe = make_pipeline(StandardScaler(), mk()).fit(X, y); sc, reg = pipe[0], pipe[1]
coef = np.asarray(reg.coef_, float) / sc.scale_; coef[0] = 0.0; coef[0] = float(reg.intercept_) - float((np.asarray(reg.coef_, float) * sc.mean_ / sc.scale_)[1:].sum())   # fold the scaler in: E = sum_i coef_i (x_i - mu_i)/s_i + b ; x_0 = 1
assert np.allclose(X @ coef, pipe.predict(X), atol=1e-8), "folded parameters != pipeline prediction"
print(f"{len(y)} structures, K = {X.shape[1]}, {REG} alpha = {getattr(reg, 'alpha_', np.nan):.3g}, nonzero {int((np.abs(coef[1:]) > 1e-12).sum())}, train MAE {np.abs(X @ coef - y).mean()*1e3:.2f} meV/site, RMSE {np.sqrt(((X @ coef - y)**2).mean())*1e3:.2f}, max|ECI| {np.abs(coef[1:]).max()*1e3:.0f} meV/site")
ref = ClusterExpansion.read(f"{MODELS}/ce/stfo/icet_q56.ce"); print("convention check: existing file zerolet parameter", ref.parameters[0], "; this fit zerolet x 160 =", coef[0] * 160)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True); ClusterExpansion(cs, coef * 160.0).write(out); print("wrote", out)
