"""Orthonormal-CE (icet) heavy fit: cutoff x regressor sweep with NESTED selection, per system; also builds mapped.pkl.

SYSTEM=stfo   : data/new_bench (389 x 160-atom), primitive/sublattices taken from the trained models/ce/stfo/icet_q56.ce
                (never re-detected from a decorated cif), folds = StratifiedKFold(10, seed 42+rep) on comp, anchors
                pinned, N_REP repeats -- the unified-benchmark protocol.
SYSTEM=fenicr : data/feni_cr, exact-duplicate manifest consumed from data/feni_cr/folds_decoration.json, outer folds =
                that file's decoration GroupKFold(10); inner = GroupKFold(5) on the decoration digest.
                Sizes reported all / ge32 / eq108.

For every outer fold the (cutoff set, regressor) pair is chosen on the outer TRAIN only (inner CV); the outer test is
used once.  Pooled OOF RMSE is the headline.  A per-cutoff oracle table (regressor chosen on the test score) is
written for diagnostics only.  The winning cutoff set is what becomes the kernel bank of the neural CE.

Outputs: $RUNS/icet_heavy/<SYSTEM>[_smoke][STFO_LABEL_TAG]/{fit_log.txt, X_<cutoffs>.npy, oof_nested_rep*.npy, results.json}.
         The structures mapped onto the primitive are cached as data/<set>/mapped.pkl (set = new_bench | feni_cr),
         which is the file train_nce.py reads; an existing cache is reused.
Env    : SYSTEM (stfo), N_REP (3, stfo only), LIMIT (0; smoke subset), CUTS (comma-separated indices into the cutoff
         grid; smoke), NJOBS (8), STFO_DETAILED_CSV (label csv override, e.g. the corrected one),
         STFO_LABEL_TAG ("" ; suffix of the output directory for corrected labels).

Run:  SYSTEM=stfo NJOBS=16 N_REP=3 python scripts/train/fit_ce_stfo.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
from __future__ import annotations
import os as _os
SYSTEM = _os.environ.get("SYSTEM", "stfo")
N_REP = int(_os.environ.get("N_REP", 3))
LIMIT = int(_os.environ.get("LIMIT", 0))
NJOBS = int(_os.environ.get("NJOBS", 8))
import json, time, pickle
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import RidgeCV, LassoCV, ARDRegression, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import ase.io
from icet import ClusterSpace, ClusterExpansion
from icet.tools import map_structure_to_reference
from neuralce.paths import DATA, MODELS, RUNS


def require(paths, hint):                                                                 # the STFO / Fe-Ni-Cr inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not _os.path.exists(p): print(f"[skip] {_os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)


if SYSTEM == "stfo": require([_os.environ.get("STFO_DETAILED_CSV", f"{DATA}/new_bench/detailed_info.csv"), f"{MODELS}/ce/stfo/icet_q56.ce"], "$NCE_DATA/new_bench (STFO_DETAILED_CSV), $NCE_MODELS/ce/stfo")
else: require([_os.environ.get("STFO_DETAILED_CSV", f"{DATA}/feni_cr/detailed_info.csv"), f"{DATA}/feni_cr/folds_decoration.json"], "$NCE_DATA/feni_cr (STFO_DETAILED_CSV)")
OUT = Path(RUNS) / "icet_heavy" / (SYSTEM + ("_smoke" if LIMIT else "") + _os.environ.get("STFO_LABEL_TAG", ""))   # STFO_LABEL_TAG: separate output for corrected labels
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "fit_log.txt", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ── cutoff grids ──────────────────────────────────────────────────────────────
GRID = {
    "stfo": [[5.0, 4.5, 4.0], [6.0, 5.0, 4.5], [6.0, 5.0, 5.6], [7.0, 5.0, 4.5],
             [7.0, 6.0, 5.0], [6.0, 6.0, 4.5], [8.0, 5.0, 4.5], [8.0, 6.5, 5.5],
             [6.0, 5.0, 4.5, 4.0], [7.0, 6.0, 5.0, 4.0],
             # "more diverse clusters" extension (indices 10-14)
             [9.0, 7.0, 6.0], [10.0, 8.0, 6.0], [8.0, 6.5, 5.5, 5.0],
             [9.0, 7.0, 6.0, 5.5], [7.0, 7.0, 6.0, 5.5]],
    # fcc a=3.60: shells 2.546 / 3.600 / 4.409 / 5.091 / 5.692 / 6.235 / 6.735
    "fenicr": [[5.2, 4.5], [6.0, 5.0], [6.5, 5.2], [7.0, 5.0], [7.0, 6.0],
               [6.0, 5.0, 4.5], [7.0, 5.2, 4.5], [8.0, 5.2, 4.5], [7.0, 6.0, 5.2]],
}[SYSTEM]
if _os.environ.get("CUTS"):
    GRID = [GRID[int(i)] for i in _os.environ["CUTS"].split(",")]
MAX_PARAMS = 4000                         # skip a cluster space bigger than this

REG = {
    "ridge": lambda: RidgeCV(alphas=np.logspace(-6, 4, 60)),
    "lasso": lambda: LassoCV(alphas=np.logspace(-6, 0, 41), cv=5, max_iter=50000, n_jobs=1),
    "enet":  lambda: ElasticNetCV(l1_ratio=[.3, .5, .7], alphas=np.logspace(-6, 0, 31),
                                  cv=5, max_iter=50000, n_jobs=1),
    "ardr":  lambda: ARDRegression(max_iter=300),
}


# ── data ──────────────────────────────────────────────────────────────────────
if SYSTEM == "stfo":
    CIF_DIR = Path(DATA) / "new_bench"
    df = pd.read_csv(_os.environ.get("STFO_DETAILED_CSV", str(CIF_DIR / "detailed_info.csv")))
    ce0 = ClusterExpansion.read(str(Path(MODELS) / "ce" / "stfo" / "icet_q56.ce"))
    cs0 = ce0.get_cluster_space_copy()
    PRIM, CHEM, INERT = cs0.primitive_structure, cs0.chemical_symbols, ["Sr"]
    ids = df.id.astype(str).tolist()
    y = (df.total_energy / df.n_atoms).to_numpy()
    strat = df.comp.astype(int).to_numpy()
    n_atoms = df.n_atoms.to_numpy()
    dropped = set()
    log(f"[stfo] {len(df)} rows | primitive {len(PRIM)} atoms {PRIM.get_chemical_symbols()}")
else:
    from neuralce.ce.spin_pairs import reembed_to_reference_frame, decoration_digest
    from neuralce.ce.primitive_no_idealize import detect_primitive_no_idealize
    CIF_DIR = Path(DATA) / "feni_cr"
    df = pd.read_csv(_os.environ.get("STFO_DETAILED_CSV", str(CIF_DIR / "detailed_info.csv")))
    spec = json.load(open(CIF_DIR / "folds_decoration.json"))
    dropped = set(map(str, spec["dropped_exact_duplicates"]))
    keep = ~df.id.astype(str).isin(dropped)
    df = df[keep].reset_index(drop=True)
    ids = df.id.astype(str).tolist()
    y = (df.total_energy / df.n_atoms).to_numpy()
    n_atoms = df.n_atoms.to_numpy()
    strat = df.comp.astype(str).to_numpy()
    FOLD = np.array([int(spec["folds"][i]) for i in ids])
    ref_row = int(np.where(n_atoms == 32)[0][0])
    PRIM = detect_primitive_no_idealize(str(CIF_DIR / f"{ids[ref_row]}.cif"),
                                        {"Ni": "Fe", "Cr": "Fe"}, 1e-2)
    assert len(PRIM) == 1
    CHEM, INERT = [["Fe", "Ni", "Cr"]], None
    log(f"[fenicr] {len(df)} rows after dedup ({len(dropped)} dropped) | "
        f"primitive a_conv={np.cbrt(abs(np.linalg.det(PRIM.cell.array))*4):.4f}")

if LIMIT:
    sel = np.arange(len(ids))[:LIMIT]
    ids = [ids[i] for i in sel]; y = y[sel]; strat = strat[sel]; n_atoms = n_atoms[sel]
    if SYSTEM == "fenicr": FOLD = FOLD[sel]
n_all = len(ids)


# ── map every structure onto the primitive once ──────────────────────────────
def _map(i):
    a = ase.io.read(str(CIF_DIR / f"{i}.cif"))
    if SYSTEM == "fenicr":
        a = reembed_to_reference_frame(a, PRIM.cell.array)
    m, info = map_structure_to_reference(a, PRIM, inert_species=INERT,
                                         assume_no_cell_relaxation=False)
    return m, float(info.get("drmax", 0.0))

t0 = time.time()
cache = CIF_DIR / "mapped.pkl"                 # data/<set>/mapped.pkl: the structure list train_nce.py reads
if cache.exists() and not LIMIT:
    mapped = pickle.load(open(cache, "rb"))
else:
    res = Parallel(n_jobs=NJOBS)(delayed(_map)(i) for i in ids)
    mapped = [r[0] for r in res]
    drmax = max(r[1] for r in res)
    log(f"[map] {n_all} structures in {time.time()-t0:.0f}s, drmax={drmax:.2e}")
    if not LIMIT: pickle.dump(mapped, open(cache, "wb")); log(f"[map] cached {cache}")

groups = None
if SYSTEM == "fenicr":
    groups = np.array([decoration_digest(m) for m in mapped])
    log(f"[fenicr] {len(set(groups))} decorations among {n_all} rows")


# ── cluster vectors per cutoff set ───────────────────────────────────────────
FEATS, NPAR = {}, {}
for cut in GRID:
    key = "c" + "_".join(f"{c:g}" for c in cut)
    f = OUT / f"X_{key}.npy"
    cs = ClusterSpace(PRIM, cut, CHEM)
    NPAR[key] = len(cs)
    if len(cs) > MAX_PARAMS:
        log(f"[cs] {key}: {len(cs)} params > {MAX_PARAMS}, skipped"); continue
    if f.exists() and not LIMIT:
        FEATS[key] = np.load(f); continue
    t0 = time.time()
    # a ClusterSpace cannot be pickled to loky workers: each worker rebuilds it
    # from (primitive, cutoffs, symbols) and does a contiguous chunk.
    def _cv_chunk(cut_, lo, hi):
        cs_ = ClusterSpace(PRIM, cut_, CHEM)
        return [cs_.get_cluster_vector(m) for m in mapped[lo:hi]]
    step = max(1, int(np.ceil(n_all / (NJOBS * 4))))
    parts = Parallel(n_jobs=NJOBS)(delayed(_cv_chunk)(cut, lo, min(lo + step, n_all))
                                   for lo in range(0, n_all, step))
    X = np.array([row for p in parts for row in p])
    FEATS[key] = X
    if not LIMIT: np.save(f, X)
    log(f"[cs] {key}: {len(cs)} params, X {X.shape} in {time.time()-t0:.0f}s")


# ── folds ────────────────────────────────────────────────────────────────────
idx_all = np.arange(n_all)
if SYSTEM == "stfo":
    cnt = pd.Series(strat).value_counts()
    ANCH = np.isin(strat, cnt[cnt < 2].index.to_numpy())
    idx_ev = idx_all[~ANCH]
    def outer_folds(rep):
        skf = StratifiedKFold(10, shuffle=True, random_state=42 + rep)
        return [(np.concatenate([idx_ev[a], idx_all[ANCH]]), idx_ev[b])
                for a, b in skf.split(idx_ev, strat[~ANCH])]
    def inner_splits(tr):
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        st = strat[tr]; c = pd.Series(st).value_counts()
        st2 = np.where(np.isin(st, c[c < 5].index.to_numpy()), -1, st)   # tiny classes pooled
        return list(skf.split(tr, st2))
    REPS = range(N_REP)
else:
    ANCH = np.zeros(n_all, bool)
    def outer_folds(rep):
        return [(idx_all[FOLD != k], idx_all[FOLD == k]) for k in range(10) if (FOLD == k).any()]
    def inner_splits(tr):
        return list(GroupKFold(5).split(tr, groups=groups[tr]))
    REPS = [0]


def rmse(p, t): return float(np.sqrt(np.mean(((p - t) * 1000) ** 2)))
def fitpred(mk, X, tr, te):
    return make_pipeline(StandardScaler(), mk()).fit(X[tr], y[tr]).predict(X[te])


def inner_score(key, rn, tr, isp):
    X = FEATS[key]; oof = np.full(len(tr), np.nan)
    try:
        for a, b in isp:
            oof[b] = fitpred(REG[rn], X, tr[a], tr[b])
    except Exception as e:
        return (key, rn, np.inf)
    return (key, rn, rmse(oof, y[tr]))


def nested(folds):
    oof = np.full(n_all, np.nan); picks = []
    for f, (tr, te) in enumerate(folds):
        isp = inner_splits(tr)
        scored = Parallel(n_jobs=NJOBS, backend="threading")(
            delayed(inner_score)(k, r, tr, isp) for k in FEATS for r in REG)
        key, rn, sc = min(scored, key=lambda s: s[2])
        oof[te] = fitpred(REG[rn], FEATS[key], tr, te)
        picks.append((key, rn))
        log(f"    fold {f}: picked {key}/{rn} (inner {sc:.3f})  outer {rmse(oof[te], y[te]):.3f}")
    return oof, picks


def slices(oof):
    ok = ~np.isnan(oof); out = {"all": rmse(oof[ok], y[ok])}
    if SYSTEM == "fenicr":
        for nm, msk in (("ge32", n_atoms >= 32), ("eq108", n_atoms == 108)):
            m = ok & msk
            if m.any(): out[nm] = rmse(oof[m], y[m])
    return out


results = {"system": SYSTEM, "n": n_all, "grid": GRID, "n_params": NPAR, "reps": []}
log(f"\n== nested selection over {len(FEATS)} cutoff sets x {len(REG)} regressors ==")
for rep in REPS:
    folds = outer_folds(rep)
    t0 = time.time()
    oof, picks = nested(folds)
    r = slices(oof)
    top = Counter(picks).most_common(3)
    results["reps"].append({"rep": rep, "rmse": r, "picks": [f"{k}/{v}" for k, v in picks],
                            "top": [(f"{k}/{v}", c) for (k, v), c in top]})
    np.save(OUT / f"oof_nested_rep{rep}.npy", oof)
    log(f"[rep {rep}] nested {r}  top picks {top}  ({time.time()-t0:.0f}s)")

log("\n== per-cutoff oracle (regressor chosen on test; diagnostics only) ==")
oracle = {}
folds0 = outer_folds(0)
for key in FEATS:
    best = None
    for rn in REG:
        oof = np.full(n_all, np.nan)
        try:
            for tr, te in folds0:
                oof[te] = fitpred(REG[rn], FEATS[key], tr, te)
        except Exception:
            continue
        r = slices(oof)
        if best is None or r["all"] < best[1]["all"]:
            best = (rn, r)
    if best:
        oracle[key] = {"reg": best[0], **best[1], "n_params": NPAR[key]}
        log(f"  {key:22s} n={NPAR[key]:5d}  {best[0]:6s}  " +
            "  ".join(f"{k} {v:.3f}" for k, v in best[1].items()))
results["oracle"] = oracle

allp = Counter(p for rep in results["reps"] for p in rep["picks"])
best_key = Counter(p.split("/")[0] for rep in results["reps"] for p in rep["picks"]).most_common(1)[0][0]
results["nested_mean"] = float(np.mean([rep["rmse"]["all"] for rep in results["reps"]]))
results["nested_std"] = float(np.std([rep["rmse"]["all"] for rep in results["reps"]]))
results["best_cutoffs"] = [c for c in GRID if "c" + "_".join(f"{x:g}" for x in c) == best_key][0]
results["best_cutoffs_key"] = best_key
json.dump(results, open(OUT / "results.json", "w"), indent=2)
log(f"\n[SUMMARY] {SYSTEM}: nested RMSE = {results['nested_mean']:.3f} +- {results['nested_std']:.3f} "
    f"meV/atom | most-picked cutoffs {best_key} ({allp.most_common(1)[0]})")
log("ICET_HEAVY_DONE")
