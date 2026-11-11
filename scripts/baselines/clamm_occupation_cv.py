"""Occupation CE (CLAMM decoration counts) 10-fold CV, fitted with the same folds, regressor menu and nested selection
as the icet heavy fit, so the "Occupation CE" column of the benchmark table is comparable with the "Orthonormal CE
(icet)" column.

Features = CLAMM count_out (one column per cluster decoration, the software's own CLUSTERS file); rows aligned to the
labelled csv by structure id and checked against the energy CLAMM stored in str_out.  Regressors: ridge / lasso / ARD
(threshold 1e4 and 1e5), all after StandardScaler, intercept fitted.  Per fold: the regressor is picked by an inner
5-fold on the outer train only (nested); the per-regressor table is also reported.

SYSTEM=nipt   icet_csv_FINAL_dedup (121), StratifiedKFold(10, seed 42) on n_ni, anchors pinned.  (default; the only
              system whose labelled csv is shipped)
SYSTEM=stfo   new_bench (389 rows with CLAMM counts), StratifiedKFold(10, seed 42) on comp,
              single-member compositions pinned to train.  Code path kept for reference; data not included.
SYSTEM=fenicr data/feni_cr dedup (533), decoration GroupKFold folds from data/feni_cr/folds_decoration.json;
              MAE reported on all / ge32 / eq108.  Code path kept for reference; data not included.
Metric: MAE per fold -> mean +- std (ddof=1), meV/atom.  Also pooled RMSE.

External requirements: the CLAMM toolkit outputs count_out (json cluster counts) and str_out (json structure list)
for the system, produced by CLAMM (Ertekin group, MIT licence) from its CLUSTERS / mag_data inputs; they are NOT part
of this repository (count_out of STFO alone is 30 MB).  Directory: env CLAMM_DIR, default
$DATA/nipt/clamm_q4_FINAL | $DATA/clamm_stfo | $DATA/feni_cr/clamm_fit.  count_out is read with the CLAMM parser
copied next to this script (clamm_parse.py, MIT).  The shipped result of this script is results/cv/clamm/nipt/.
Env    : SYSTEM (nipt|stfo|fenicr, default nipt), CLAMM_DIR, NJOBS (8), LIMIT (smoke subset), REGS (comma subset of the
         regressors), STFO_DETAILED_CSV (STFO label csv), STFO_LABEL_TAG (output-dir suffix, e.g. _lab12).
Outputs: $RUNS/external/clamm/<SYSTEM>[_smoke][<STFO_LABEL_TAG>]/{oof.csv, oof_*.npy, results.json, log.txt};
         oof.csv and results.json are also copied to $RESULTS/cv/clamm/<same name>/.
Run    : SYSTEM=nipt python scripts/baselines/clamm_occupation_cv.py
"""
from __future__ import annotations
import os, json, time, shutil
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import RidgeCV, LassoCV, ARDRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from neuralce.paths import DATA, RUNS, RESULTS
import clamm_parse as parse                      # scripts/baselines/clamm_parse.py (CLAMM_Fit/parse.py, MIT)

SYSTEM = os.environ.get("SYSTEM", "nipt"); NJOBS = int(os.environ.get("NJOBS", 8)); LIMIT = int(os.environ.get("LIMIT", 0))


def require(paths, hint, why="STFO / Fe-Ni-Cr data are not public"):                      # inputs not in the public repository: skip cleanly when missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository ({why}); provide it under {hint} to run", flush=True); raise SystemExit(0)


_D = Path(DATA)
if SYSTEM == "stfo": require([os.environ.get("STFO_DETAILED_CSV", str(_D / "new_bench/detailed_info.csv"))], "$NCE_DATA/new_bench (STFO_DETAILED_CSV)")
elif SYSTEM != "nipt": require([str(_D / "feni_cr/detailed_info.csv"), str(_D / "feni_cr/folds_decoration.json")], "$NCE_DATA/feni_cr")
_FIT = Path(os.environ.get("CLAMM_DIR", {"stfo": _D / "clamm_stfo", "nipt": _D / "nipt/clamm_q4_FINAL"}.get(SYSTEM, _D / "feni_cr/clamm_fit")))
require([str(_FIT / "count_out"), str(_FIT / "str_out")], "CLAMM_DIR", why="the CLAMM count_out / str_out outputs are not part of this repository")
NAME = SYSTEM + ("_smoke" if LIMIT else "") + os.environ.get("STFO_LABEL_TAG", "")
OUT = Path(RUNS) / "external" / "clamm" / NAME; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "log.txt", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

REG = {
    "ridge": lambda: RidgeCV(alphas=np.logspace(-6, 4, 60)),
    "lasso": lambda: LassoCV(alphas=np.logspace(-6, 0, 41), cv=5, max_iter=50000, n_jobs=1),
    "ardr1e4": lambda: ARDRegression(max_iter=300, threshold_lambda=1e4),
    "ardr1e5": lambda: ARDRegression(max_iter=300, threshold_lambda=1e5),
}
if os.environ.get("REGS"): REG = {k: REG[k] for k in os.environ["REGS"].split(",")}

# ── CLAMM counts + labelled rows, aligned by id ────────────────────────────────
D = Path(DATA)
if SYSTEM == "stfo":
    FIT = Path(os.environ.get("CLAMM_DIR", D / "clamm_stfo")); df = pd.read_csv(os.environ.get("STFO_DETAILED_CSV", str(D / "new_bench/detailed_info.csv")))
    df["key"] = df.id.astype(str); df["y"] = df.total_energy / df.n_atoms; df["strat"] = df.comp.astype(int)
elif SYSTEM == "nipt":
    FIT = Path(os.environ.get("CLAMM_DIR", D / "nipt/clamm_q4_FINAL")); df = pd.read_csv(D / "nipt/icet_csv_FINAL_dedup.csv")
    df["key"] = df.id.astype(str).str.split("__", n=1).str[-1]; df["y"] = df.energy_per_atom; df["strat"] = df.n_ni
else:
    FIT = Path(os.environ.get("CLAMM_DIR", D / "feni_cr/clamm_fit")); df = pd.read_csv(D / "feni_cr/detailed_info.csv")
    spec = json.load(open(D / "feni_cr/folds_decoration.json"))
    df = df[~df.id.astype(str).isin(set(map(str, spec["dropped_exact_duplicates"])))].reset_index(drop=True)
    df["key"] = df.id.astype(str); df["y"] = df.total_energy / df.n_atoms; df["strat"] = df.comp.astype(str)
count, _ = parse.parse_count(str(FIT / "count_out")); Xc = np.asarray(count, float)
strs = json.load(open(FIT / "str_out")); assert len(strs) == Xc.shape[0], (len(strs), Xc.shape)
cell = [s["CellName"] for s in strs]; e_clamm = np.array([s["Enrg"] for s in strs]); nat_clamm = np.array([s["AtomSum"] for s in strs])
if SYSTEM == "fenicr":                      # CLAMM rows follow the deduped csv order
    assert len(df) == len(strs) and (df.n_atoms.to_numpy() == nat_clamm).all(); row_of = {k: i for i, k in enumerate(df.key)}
else:
    row_of = {c: i for i, c in enumerate(cell)}
    if SYSTEM == "nipt":                    # CLAMM CellName is the csv id or its part after "__"
        df["key"] = [i if i in row_of else k for i, k in zip(df.id.astype(str), df.key)]
        # dedup groups: the csv carries one member's name, CLAMM another's (identical cluster vector,
        # labels within ~1e-4 eV/atom) -> match the leftovers by energy (require < 5e-4 eV/atom)
        left_c = [i for i, c in enumerate(cell) if c not in set(df.key)]
        for r in np.where(~df.key.isin(row_of))[0]:
            i = min(left_c, key=lambda i: abs(e_clamm[i] - df.y[r])); assert abs(e_clamm[i] - df.y[r]) < 5e-4, (df.id[r], cell[i])
            log(f"  [nipt] {df.id[r]} <- CLAMM row {cell[i]} (|dE| {abs(e_clamm[i]-df.y[r])*1e3:.2f} meV/atom)"); df.loc[r, "key"] = cell[i]; left_c.remove(i)
have = df.key.isin(row_of); log(f"[{SYSTEM}] labelled {len(df)} rows, CLAMM counts {Xc.shape}, matched {int(have.sum())}, missing {sorted(df.key[~have])[:5]}")
df = df[have].reset_index(drop=True); rows = np.array([row_of[k] for k in df.key]); X = Xc[rows]
y = df.y.to_numpy(); n_atoms = df.n_atoms.to_numpy(); strat = df.strat.to_numpy(); ids = df.key.tolist()
dE = np.abs(e_clamm[rows] - y).max(); log(f"[{SYSTEM}] max |E_clamm(str_out) - E_csv| = {dE:.2e} eV/atom" + ("  (rescaled target in CLAMM; csv label used)" if dE > 1e-4 else ""))
if LIMIT:
    X, y, n_atoms, strat, ids = X[:LIMIT], y[:LIMIT], n_atoms[:LIMIT], strat[:LIMIT], ids[:LIMIT]
n_all = len(y); idx_all = np.arange(n_all)
nzc = int((X.std(0) > 0).sum()); X = X[:, X.std(0) > 0]
_, uniq = np.unique(X.T, axis=0, return_index=True); X = X[:, np.sort(uniq)]
log(f"[{SYSTEM}] {Xc.shape[1]} decoration columns, {nzc} non-constant, {X.shape[1]} after removing exact-duplicate columns")

# ── folds (identical to the icet heavy fit / PtNi protocol) ───────────────────
if SYSTEM in ("stfo", "nipt"):
    cnt = pd.Series(strat).value_counts(); ANCH = np.isin(strat, cnt[cnt < 2].index.to_numpy()); idx_ev = idx_all[~ANCH]
    skf = StratifiedKFold(10, shuffle=True, random_state=42)
    if SYSTEM == "stfo":
        folds = [(np.concatenate([idx_ev[a], idx_all[ANCH]]), idx_ev[b]) for a, b in skf.split(idx_ev, strat[~ANCH])]
    else:                                   # PtNi protocol: split all rows, then move anchors to train
        folds = []
        for tr, te in skf.split(np.zeros(n_all), strat):
            folds.append((np.union1d(tr, idx_all[ANCH]), np.setdiff1d(te, idx_all[ANCH])))
    def inner_splits(tr):
        skf5 = StratifiedKFold(5, shuffle=True, random_state=0); st = strat[tr]; c = pd.Series(st).value_counts()
        st2 = np.where(np.isin(st, c[c < 5].index.to_numpy()), -1, st); return list(skf5.split(tr, st2))
    groups = None
else:
    FOLD = np.array([int(spec["folds"][i]) for i in ids])
    folds = [(idx_all[FOLD != k], idx_all[FOLD == k]) for k in range(10) if (FOLD == k).any()]
    dig = {}
    for k in ids:                          # decoration digest for the inner GroupKFold, from the fold spec's grouping if present
        dig[k] = spec.get("group_of", {}).get(k, k)
    groups = np.array([dig[k] for k in ids])
    def inner_splits(tr): return list(GroupKFold(5).split(tr, groups=groups[tr]))
log(f"[{SYSTEM}] {len(folds)} outer folds, test sizes {[len(te) for _, te in folds]}")

def fitpred(mk, tr, te):
    m = make_pipeline(StandardScaler(), mk()).fit(X[tr], y[tr]); nz = int((np.abs(m[-1].coef_) > 1e-12).sum()); return m.predict(X[te]), nz
def mae(p, t): return float(np.mean(np.abs(p - t)) * 1e3)
def rmse(p, t): return float(np.sqrt(np.mean(((p - t) * 1e3) ** 2)))
def inner_score(rn, tr, isp):
    oof = np.full(len(tr), np.nan)
    try:
        for a, b in isp: oof[b] = fitpred(REG[rn], tr[a], tr[b])[0]
    except Exception as e:
        return rn, np.inf
    return rn, rmse(oof, y[tr])

def run_fold(f, tr, te):
    pr, nz = {}, {}
    for rn in REG: pr[rn], nz[rn] = fitpred(REG[rn], tr, te)
    isp = inner_splits(tr); scored = [inner_score(rn, tr, isp) for rn in REG]
    rn, sc = min(scored, key=lambda s: s[1])
    return f, pr, nz, rn, sc
t0 = time.time()
res = Parallel(n_jobs=min(NJOBS, len(folds)))(delayed(run_fold)(f, tr, te) for f, (tr, te) in enumerate(folds))
per_reg = {rn: np.full(n_all, np.nan) for rn in REG}; nz_reg = {rn: [None] * len(folds) for rn in REG}
oof_nested = np.full(n_all, np.nan); picks = [None] * len(folds)
for f, pr, nz, rn, sc in res:
    te = folds[f][1]
    for r in REG: per_reg[r][te] = pr[r]; nz_reg[r][f] = nz[r]
    oof_nested[te] = pr[rn]; picks[f] = rn
    log(f"  fold {f}: " + "  ".join(f"{r} {mae(per_reg[r][te], y[te]):.2f}" for r in REG) + f"  | picked {rn} (inner rmse {sc:.2f})")
log(f"  all folds done in {time.time()-t0:.0f}s")

def slices(oof):
    res = {}
    for nm, msk in (("all", np.ones(n_all, bool)), ("ge32", n_atoms >= 32), ("eq108", n_atoms == 108)):
        if SYSTEM != "fenicr" and nm != "all": continue
        fm = [mae(oof[te][msk[te]], y[te][msk[te]]) for _, te in folds if (msk[te] & ~np.isnan(oof[te])).any()]
        ok = msk & ~np.isnan(oof)
        res[nm] = dict(mae_fold_mean=float(np.mean(fm)), mae_fold_std=float(np.std(fm, ddof=1)), n_folds=len(fm),
                       mae_pooled=mae(oof[ok], y[ok]), rmse_pooled=rmse(oof[ok], y[ok]), n=int(ok.sum()))
    return res
results = dict(system=SYSTEM, n=n_all, n_cols=int(X.shape[1]), n_cols_nonconst=nzc, picks=picks,
               nested=slices(oof_nested), per_regressor={rn: slices(per_reg[rn]) for rn in REG},
               nonzero_per_fold={rn: nz_reg[rn] for rn in REG})
json.dump(results, open(OUT / "results.json", "w"), indent=2)
np.save(OUT / "oof_nested.npy", oof_nested); [np.save(OUT / f"oof_{rn}.npy", per_reg[rn]) for rn in REG]
pd.DataFrame(dict(id=ids, y=y, n_atoms=n_atoms, oof_nested=oof_nested, **{f"oof_{rn}": per_reg[rn] for rn in REG})).to_csv(OUT / "oof.csv", index=False)
log(f"\n[SUMMARY] {SYSTEM} Occupation CE (CLAMM counts) 10-fold MAE, fold mean ± std (meV/atom):")
for rn in list(REG) + ["nested"]:
    r = results["per_regressor"][rn] if rn != "nested" else results["nested"]
    log(f"  {rn:9s} " + "  ".join(f"{nm} {v['mae_fold_mean']:.2f} ± {v['mae_fold_std']:.2f} (pooled MAE {v['mae_pooled']:.2f}, RMSE {v['rmse_pooled']:.2f}, n={v['n']})" for nm, v in r.items())
        + ("" if rn == "nested" else f"  nonzero {int(np.median(nz_reg[rn]))}/{X.shape[1]}"))
log(f"  nested picks: {picks}")
RES = Path(RESULTS) / "cv" / "clamm" / NAME; RES.mkdir(parents=True, exist_ok=True)
for fn in ("oof.csv", "results.json"): shutil.copy2(OUT / fn, RES / fn)
log(f"  copied oof.csv, results.json -> {RES}")
log("CLAMM_OCC_CV10_DONE")
