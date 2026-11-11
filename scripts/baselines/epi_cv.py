"""EPI (effective pair interaction) model of Liu et al., npj Comput. Mater. 11 (2025) as an external benchmark column.
Protocol: folds / labels from data/benchmark_folds.json, per-fold MAE in meV/atom, FeNiCr metric on the 108-atom test
structures only.

Model (paper eq. 3):   E/N = sum_{p'<=p, m} V_m^{pp'} Pi_m^{pp'} + sum_p V^p x_p + V0
    Pi_m^{pp'} = n_m^{pp'} / sum_{pp'} n_m^{pp'}   (fraction of pp' pairs in coordination shell m)
    x_p        = species concentration (single-site term; the paper absorbs it into the constant because its
                 datasets have ONE composition, ours mix compositions inside a fold)
Fit: Bayesian regression (sklearn BayesianRidge) as in the paper; interaction range (number of shells) chosen by
BIC inside the outer training fold.  Reference variant: fixed 2nd-NN range (the paper's FeCoNiAlTi setting).
FeNiCr: the paper's EPI has no spin.  'chem' = paper model; 'chem+spin' adds per-shell spin-pair fractions
    S_m^{pp'} = sum_{(i,j) in (pp',m)} s_i s_j / sum n_m  and the |s| fraction per species (Ising pair).

External requirements: pymatgen, scikit-learn (both in the base install); no external code.
Inputs : data/benchmark_folds.json (env STFO_FOLDS_JSON overrides), data/nipt/cifs/<id>.cif (PtNi, env NIPT_CIF_DIR).
         Only the PtNi inputs are shipped; the stfo / fenicr code paths (data/new_bench/<id>.cif, data/feni_cr/{<id>.cif,
         detailed_info.csv}) are kept for reference and need data that is not included.
Outputs: $RUNS/external/epi/<system>[_smoke]/{oof.csv, oof_<variant>_<range>.csv, summary.json, features.npz}
         (--out overrides); oof.csv, summary.json and features.npz are also copied to
         $RESULTS/cv/external/epi/<name>/ where <name> is the output directory's basename (--results-tag overrides).
Run    : python scripts/baselines/epi_cv.py [--system ptni] [--cutoff 10] [--smoke] [--out DIR]
"""
import os, json, time, argparse, shutil
from collections import Counter
import numpy as np
import pandas as pd
from neuralce.paths import DATA, RUNS, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="ptni", choices=["stfo", "ptni", "fenicr"], help="only the ptni inputs are shipped")
ap.add_argument("--cutoff", type=float, default=10.0, help="max pair distance considered (A)")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--results-tag", default=None, help="name under $RESULTS/cv/external/epi/ (default: basename of --out)")
args = ap.parse_args()
SYSTEM = args.system
# the STFO / Fe-Ni-Cr inputs are not shipped: skip cleanly when they are missing (only the PtNi inputs are in the repository)
_FJ = os.environ.get("STFO_FOLDS_JSON", f"{DATA}/benchmark_folds.json")
_need = {"stfo": [f"{DATA}/new_bench"], "fenicr": [f"{DATA}/feni_cr/detailed_info.csv"]}.get(SYSTEM, [])
for _p in _need:
    if not os.path.exists(_p):
        print(f"[skip] {os.path.basename(__file__)}: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_DATA to run", flush=True); raise SystemExit(0)
if SYSTEM not in json.load(open(_FJ)):
    print(f"[skip] {os.path.basename(__file__)}: {_FJ} has no '{SYSTEM}' entry (STFO / Fe-Ni-Cr data are not public); provide it under STFO_FOLDS_JSON to run", flush=True); raise SystemExit(0)
OUT = args.out or f"{RUNS}/external/epi/{SYSTEM}{'_smoke' if args.smoke else ''}"
os.makedirs(OUT, exist_ok=True)

from pymatgen.core import Structure
from sklearn.linear_model import BayesianRidge

folds_json = json.load(open(os.environ.get("STFO_FOLDS_JSON", f"{DATA}/benchmark_folds.json")))[SYSTEM]
items = folds_json["items"]
if args.smoke:
    items = items[:80]
ids = [it["id"] for it in items]
y = np.asarray([it["y"] for it in items], float)
fold = np.asarray([it["fold"] for it in items], int)
n_atoms_arr = np.asarray([it.get("n_atoms", 160 if SYSTEM == "stfo" else 108) for it in items])
print(f"=== EPI | system={SYSTEM} | N={len(ids)} | folds={sorted((int(k), int(v)) for k, v in Counter(fold).items())}")

if SYSTEM == "fenicr":
    _df = pd.read_csv(f"{DATA}/feni_cr/detailed_info.csv")
    SPINS = dict(zip(_df["id"].astype(str), _df["spins"]))


def load_structure(i):
    sid = ids[i]
    if SYSTEM == "stfo":
        st = Structure.from_file(f"{DATA}/new_bench/{sid}.cif"); assert st.num_sites == 160
        return st, None
    if SYSTEM == "ptni":
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor
        from neuralce.nce.features import field_from_atoms, atoms_from_field, CIF_DIR
        field, _ = field_from_atoms(read(f"{CIF_DIR}/{sid}.cif"), m=3)
        ideal = atoms_from_field(field, a=3.924)
        assert Counter(ideal.get_chemical_symbols()).get("Ni", 0) == items[i]["n_ni"], sid
        return AseAtomsAdaptor.get_structure(ideal), None
    st = Structure.from_file(f"{DATA}/feni_cr/{sid}.cif")
    s = np.asarray(json.loads(SPINS[sid]), float); assert len(s) == st.num_sites == items[i]["n_atoms"], sid
    return st, s


# ── pass 1: pair lists (distance, species pair, spin product) per structure ────────────────────────
t0 = time.time()
raw = []          # per structure: (dist array, pair-type array, sisj array, species counts, |s| counts)
species_all = set()
for i in range(len(ids)):
    st, s = load_structure(i)
    nbrs = st.get_neighbor_list(args.cutoff)               # (center, neighbor, image, dist), both directions
    ci, cj, _, d = nbrs
    sym = np.asarray([site.specie.symbol for site in st])
    species_all |= set(sym.tolist())
    pt = np.array(["|".join(sorted((a, b))) for a, b in zip(sym[ci], sym[cj])])
    ss = (s[ci] * s[cj]) if s is not None else None
    raw.append((d.astype(np.float32), pt, ss, Counter(sym.tolist()), Counter(sym[np.abs(s) > 0].tolist()) if s is not None else None))
    if i % 100 == 0:
        print(f"  pairs {i}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)
SPECIES = sorted(species_all)
PAIRS = sorted({"|".join(sorted((a, b))) for a in SPECIES for b in SPECIES})

# ── shells: distinct pair distances over the whole dataset (ideal lattices -> discrete set) ─────────
alld = np.concatenate([r[0] for r in raw])
dr = np.round(alld, 2)
uniq = np.unique(dr)
shell_d = [uniq[0]]
for u in uniq[1:]:                                       # merge distances closer than 0.05 A
    if u - shell_d[-1] > 0.05:
        shell_d.append(u)
shell_d = np.asarray(shell_d)
N_SHELLS = len(shell_d)
print(f"  species={SPECIES}  pair types={len(PAIRS)}  shells within {args.cutoff} A: {N_SHELLS}  d={np.round(shell_d[:8],3)}...")


def shell_of(d):
    return np.abs(d[:, None] - shell_d[None, :]).argmin(1)


# ── pass 2: feature matrix  [x_p] + [Pi_m^{pp'} for m, pp'] (+ spin blocks) ─────────────────────────
pair_idx = {p: k for k, p in enumerate(PAIRS)}
X_point, X_pair, X_spair, X_spoint = [], [], [], []
for (d, pt, ss, cnt, mcnt) in raw:
    n = sum(cnt.values())
    X_point.append([cnt.get(p, 0) / n for p in SPECIES])
    m = shell_of(d)
    k = np.array([pair_idx[p] for p in pt])
    npair = np.zeros((N_SHELLS, len(PAIRS)))
    np.add.at(npair, (m, k), 1.0)
    tot = npair.sum(1, keepdims=True); tot[tot == 0] = 1
    X_pair.append((npair / tot).ravel())                 # row-major: shell-major, pair-minor
    if ss is not None:
        sp = np.zeros((N_SHELLS, len(PAIRS)))
        np.add.at(sp, (m, k), ss)
        X_spair.append((sp / tot).ravel())
        X_spoint.append([mcnt.get(p, 0) / n for p in SPECIES])
X_point = np.asarray(X_point); X_pair = np.asarray(X_pair)
X_spair = np.asarray(X_spair) if X_spair else None; X_spoint = np.asarray(X_spoint) if X_spoint else None
np.savez(f"{OUT}/features.npz", ids=np.array(ids), y=y, fold=fold, X_point=X_point, X_pair=X_pair,
         shell_d=shell_d, species=np.array(SPECIES), pairs=np.array(PAIRS),
         **({"X_spair": X_spair, "X_spoint": X_spoint} if X_spair is not None else {}))
print(f"  features built in {time.time()-t0:.0f}s: point {X_point.shape}, pair {X_pair.shape}")


def design(K, variant):
    """Feature matrix with the first K shells; drop constant columns later (fold-wise)."""
    cols = [X_point, X_pair[:, :K * len(PAIRS)]]
    if variant == "chem+spin":
        cols += [X_spoint, X_spair[:, :K * len(PAIRS)]]
    return np.hstack(cols)


def fit_predict(Xtr, ytr, Xte):
    keep = Xtr.std(0) > 1e-12                              # constant columns carry nothing in this fold
    mu, sd = Xtr[:, keep].mean(0), Xtr[:, keep].std(0)
    A = (Xtr[:, keep] - mu) / sd; B = (Xte[:, keep] - mu) / sd
    mdl = BayesianRidge(max_iter=1000, tol=1e-6, fit_intercept=True).fit(A, ytr)
    return mdl.predict(B), mdl.predict(A), int(keep.sum()), mdl


def bic_select(tr, variant, Kmax):
    """BIC over K = 1..Kmax on the TRAINING fold only (paper's range selection)."""
    n = len(tr); best = None; table = []
    for K in range(1, Kmax + 1):
        X = design(K, variant)
        ptr, _, kfeat, _ = fit_predict(X[tr], y[tr], X[tr])
        rss = float(np.sum((ptr - y[tr]) ** 2))
        if kfeat + 1 >= n:
            break
        bic = n * np.log(max(rss, 1e-30) / n) + (kfeat + 1) * np.log(n)
        table.append((K, kfeat, rss / n, bic))
        if best is None or bic < best[1]:
            best = (K, bic)
    return best[0], table


FOLDS = sorted(set(int(f) for f in fold if f >= 0))
KMAX = N_SHELLS
VARIANTS = ["chem"] + (["chem+spin"] if SYSTEM == "fenicr" else [])
RANGES = ["bic", "2nn"]
results = {}
for variant in VARIANTS:
    for rng in RANGES:
        rows, per_fold = [], []
        for f in FOLDS:
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            assert not set(te) & set(tr) and len(te) == sum(1 for it in items if it["fold"] == f)
            if rng == "bic":
                K, table = bic_select(tr, variant, KMAX)
            else:
                K, table = 2, None
            X = design(K, variant)
            pte, ptr, kfeat, mdl = fit_predict(X[tr], y[tr], X[te])
            e = np.abs(pte - y[te]) * 1000
            sel = (n_atoms_arr[te] == 108) if SYSTEM == "fenicr" else np.ones(len(te), bool)
            per_fold.append(dict(fold=f, K=int(K), n_feat=kfeat, n_train=len(tr), n_test=len(te),
                                 train_rmse_meV=float(np.sqrt(np.mean((ptr - y[tr]) ** 2)) * 1000),
                                 test_mae_meV=float(e[sel].mean()) if sel.any() else None,
                                 test_mae_all_meV=float(e.mean()), bic_table=table))
            for i, p in zip(te, pte):
                rows.append(dict(id=ids[i], y=y[i], pred=float(p), fold=f, seed=0, n_atoms=int(n_atoms_arr[i]),
                                 variant=variant, range=rng, K=int(K)))
            print(f"  [{variant}/{rng}] fold {f}: K={K} shells, {kfeat} feats, train RMSE {per_fold[-1]['train_rmse_meV']:.2f}, "
                  f"test MAE {per_fold[-1]['test_mae_meV']} (all {per_fold[-1]['test_mae_all_meV']:.2f})", flush=True)
        oof = pd.DataFrame(rows)
        # gate: every test id predicted by a model fitted without it (fold membership == JSON fold)
        jf = {it["id"]: it["fold"] for it in items}
        assert all(jf[r.id] == r.fold for r in oof.itertuples())
        fm = [pf["test_mae_meV"] for pf in per_fold if pf["test_mae_meV"] is not None]
        mask = (oof.n_atoms == 108) if SYSTEM == "fenicr" else np.ones(len(oof), bool)
        res = dict(fold_mae=[pf["test_mae_meV"] for pf in per_fold], mae_mean=float(np.mean(fm)) if fm else None,
                   mae_std=float(np.std(fm, ddof=1)) if len(fm) > 1 else None, K_per_fold=[pf["K"] for pf in per_fold],
                   n_feat_per_fold=[pf["n_feat"] for pf in per_fold],
                   train_rmse_meV=[pf["train_rmse_meV"] for pf in per_fold],
                   pooled_rmse=float(np.sqrt(np.mean((oof[mask].pred - oof[mask].y) ** 2)) * 1000),
                   pooled_mae=float(np.mean(np.abs(oof[mask].pred - oof[mask].y)) * 1000),
                   per_fold=per_fold)
        if SYSTEM == "stfo":
            comp = {it["id"]: it["comp"] for it in items}
            oof["comp"] = oof.id.map(comp)
            res["comp_mae"] = {int(c): float(np.mean(np.abs(s.pred - s.y)) * 1000) for c, s in oof.groupby("comp")}
        if SYSTEM == "fenicr":
            res["fold_mae_all_sizes"] = [pf["test_mae_all_meV"] for pf in per_fold]
        results[f"{variant}/{rng}"] = res
        oof.to_csv(f"{OUT}/oof_{variant.replace('+', '_')}_{rng}.csv", index=False)
        print(f"  ==> [{variant}/{rng}] MAE = {res['mae_mean']} ± {res['mae_std']} meV/atom  K={res['K_per_fold']}")

# table entry: paper model = chem (STFO/PtNi), chem+spin for FeNiCr (the FeNiCr rows carry spin); BIC range
main = f"{'chem+spin' if SYSTEM == 'fenicr' else 'chem'}/bic"
os.replace(f"{OUT}/oof_{main.split('/')[0].replace('+', '_')}_bic.csv", f"{OUT}/oof.csv")
summary = dict(system=SYSTEM, model="EPI (Liu et al. 2025 eq. 3, BayesianRidge, BIC range in training fold)",
               table_entry=main, cutoff=args.cutoff, n_shells_available=int(N_SHELLS), shell_d=shell_d.tolist(),
               species=SPECIES, pair_types=PAIRS, n_structures=len(ids), hp=dict(fit="BayesianRidge", range="BIC in-fold; ref 2nd-NN",
               point_terms=True, normalisation="pair fraction per shell", cutoff=args.cutoff),
               params=results[main]["n_feat_per_fold"], wall_time_sec=round(time.time() - t0),
               fold_mae=results[main]["fold_mae"], mae_mean=results[main]["mae_mean"], mae_std=results[main]["mae_std"],
               variants=results)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print(json.dumps({k: (v["mae_mean"], v["mae_std"], v["K_per_fold"]) for k, v in results.items()}, indent=1))
print(f"saved {OUT}/oof.csv, {OUT}/summary.json  ({(time.time()-t0)/60:.1f} min)")

# small outputs -> tracked results tree (read by scripts/analysis/*)
RES = f"{RESULTS}/cv/external/epi/{args.results_tag or os.path.basename(os.path.normpath(OUT))}"
os.makedirs(RES, exist_ok=True)
for fn in ("oof.csv", "summary.json", "features.npz"):
    shutil.copy2(f"{OUT}/{fn}", f"{RES}/{fn}")
print(f"copied oof.csv, summary.json, features.npz -> {RES}")
