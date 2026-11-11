"""Disorder_GNN (qmatyanlab/Configurational-Disorder, Fang & Yan, npj Comput. Mater. 10, 91, 2024) as an external
benchmark column, on the benchmark folds.

  data / folds : data/benchmark_folds.json (id, y [eV/atom], fold; fold=-1 = anchor -> always train);
                 env STFO_FOLDS_JSON overrides
  inputs       : PtNi relaxed cif -> snapped to ideal fcc (neuralce.nce.features field_from_atoms -> atoms_from_field).
                 Only the PtNi inputs are shipped; the stfo (160 sites incl. Xe, Z=54 vacancy channel) and fenicr
                 (spins -1/0/+1 folded into the embedding index) code paths are kept for reference and need data
                 that is not included.
  graph        : repo utilities.structureToData defaults (r_cutoff 10 A, max_neighbors 20, dr 0.1 -> 100 edge feats)
  model        : repo model_embedding.{Transformer,GAT,CGNN,MPNN}; ONE fixed hyper-parameter set for all
                 systems / folds (no optuna).  Repo bug fixed: Conv_list is a python list in the original
                 code, so conv parameters never reach the optimizer -> wrapped in torch.nn.ModuleList.
  training     : Adam + weight decay, MSE on standardised per-atom energy (train-fold mean/std only),
                 outer training fold split 85/15 (stratified by the fold key / decoration-grouped for FeNiCr)
                 for early stopping (ReduceLROnPlateau on val + best-val checkpoint), test fold predicted once.
  metric       : per-fold MAE (meV/atom); FeNiCr on 108-atom test structures only.

External requirements: the optional `gnn` extra (pip install -e ".[gnn]": torch, torch-geometric) and the git
submodule external/Configurational-Disorder (git submodule update --init), whose Disorder_GNN/GNN/utilities.py and
Disorder_GNN/GNN/model/model_embedding.py are loaded by file path (the submodule is not a package).
Outputs: $RUNS/external/disorder_gnn/<system>[_smoke][<tag>]/{oof.csv, runs.json, summary.json, graphs.pt} (--out
         overrides); oof.csv and summary.json are also copied to $RESULTS/cv/external/disorder_gnn/<name>/ where
         <name> is the output directory's basename (--results-tag overrides).
Run    : python scripts/baselines/disorder_gnn_cv.py [--system ptni] [--smoke] [--model Transformer]
         [--seeds 0,1,2] [--folds 0,1] [--max-epochs 2000] [--patience 300] [--tag ''] [--out DIR]
"""
import os, sys, json, time, argparse, hashlib, copy, shutil, importlib.util
from collections import Counter
import numpy as np
import pandas as pd
from neuralce.paths import REPO, DATA, RUNS, RESULTS

EXT = f"{REPO}/external/Configurational-Disorder/Disorder_GNN/GNN"


def ext_module(name, relpath):
    """Import a module of the Disorder-GNN submodule by file path (the submodule is not a package)."""
    path = os.path.join(EXT, relpath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: initialise the submodule first (git submodule update --init external/Configurational-Disorder)")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


ap = argparse.ArgumentParser()
ap.add_argument("--system", default="ptni", choices=["stfo", "ptni", "fenicr"], help="only the ptni inputs are shipped")
ap.add_argument("--model", default="Transformer", choices=["Transformer", "GAT", "CGNN", "MPNN"])
ap.add_argument("--smoke", action="store_true", help="folds 0-1, 1 seed, 30-epoch cap, separate out dir")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--folds", default=None, help="comma list of folds to run (default all 10)")
ap.add_argument("--out", default=None)
ap.add_argument("--max-epochs", type=int, default=2000)
ap.add_argument("--patience", type=int, default=300)
ap.add_argument("--tag", default="")
ap.add_argument("--results-tag", default=None, help="name under $RESULTS/cv/external/disorder_gnn/ (default: basename of --out)")
args = ap.parse_args()

# the STFO / Fe-Ni-Cr inputs are not shipped: skip cleanly when they are missing (only the PtNi inputs are in the repository)
_FJ = os.environ.get("STFO_FOLDS_JSON", f"{DATA}/benchmark_folds.json")
_need = {"stfo": [f"{DATA}/new_bench"], "fenicr": [f"{DATA}/feni_cr/detailed_info.csv"]}.get(args.system, [])
for _p in _need:
    if not os.path.exists(_p):
        print(f"[skip] {os.path.basename(__file__)}: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_DATA to run", flush=True); sys.exit(0)
if args.system not in json.load(open(_FJ)):
    print(f"[skip] {os.path.basename(__file__)}: {_FJ} has no '{args.system}' entry (STFO / Fe-Ni-Cr data are not public); provide it under STFO_FOLDS_JSON to run", flush=True); sys.exit(0)

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pymatgen.core import Structure
dg_util = ext_module("utilities", "utilities.py")                                   # repo graph builder
dg_models = ext_module("model_embedding", os.path.join("model", "model_embedding.py"))   # repo models

# ── ONE fixed hyper-parameter set (identical for stfo / ptni / fenicr, all folds, all seeds) ──────────
# Published pieces: model family = Transformer (paper's choice for the main results); 1-2 conv layers
# (SI §2.2: "the selected best models are typically very shallow, containing only one or two convolution
# layers"); channel/head grid {8..128}/{1,2,4,8} (GNN.py optuna space); weight_decay 5e-4 (GNN.py default);
# graph r_cutoff 10 / max_neighbors 20 / dr 0.1 (utilities.py defaults).  The optimum lr / batch / channels
# found by the paper's optuna run are NOT published (main text + SI checked); the values below were fixed
# a priori, before any run, and never changed: channels 64 / heads 4 = centre of the published grid,
# lr 1e-3 (the constructor default 0.1 is the top of the search range and unusable with Adam), batch 32
# (inside the published range 2-144; the in-house CGCNN/ALIGNN rows used 32).
HP = dict(
    model=args.model, num_hidden_layers=2, num_hidden_channels=64, num_heads=4,
    lr=1e-3, weight_decay=5e-4, batch_size=32,
    max_epochs=args.max_epochs, patience=args.patience,
    sched=dict(factor=0.5, patience=50, min_lr=1e-5),
    r_cutoff=10.0, max_neighbors=20, dr=0.1,
    val_frac=0.15, loss="mse_standardised", checkpoint="best_val_mae",
)
if args.smoke:
    HP["max_epochs"] = 30; HP["patience"] = 30

SEEDS = [int(s) for s in args.seeds.split(",")]
if args.smoke:
    SEEDS = SEEDS[:1]

SYSTEM = args.system
OUT = args.out or f"{RUNS}/external/disorder_gnn/{SYSTEM}{'_smoke' if args.smoke else ''}{args.tag}"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Disorder_GNN {HP['model']} | system={SYSTEM} | device={device} | out={OUT}")
print("HP:", json.dumps(HP))

# ── data: ids / y / fold from the exported JSON ──────────────────────────────────────────────────────
folds_json = json.load(open(os.environ.get("STFO_FOLDS_JSON", f"{DATA}/benchmark_folds.json")))[SYSTEM]
items = folds_json["items"]
ids = [it["id"] for it in items]
y = np.asarray([it["y"] for it in items], dtype=np.float64)       # eV/atom
fold = np.asarray([it["fold"] for it in items], dtype=int)
assert len(set(ids)) == len(ids) == folds_json["n"]
if SYSTEM == "stfo":
    strat = np.asarray([it["comp"] for it in items])
elif SYSTEM == "ptni":
    strat = np.asarray([it["n_ni"] for it in items])             # never parse comp from the id
else:
    strat = None                                                  # decoration-grouped (below)
n_atoms_arr = np.asarray([it.get("n_atoms", 160 if SYSTEM == "stfo" else 108) for it in items])
print(f"  N={len(ids)}  folds={sorted((int(k), int(v)) for k, v in Counter(fold).items())}")


def decoration_key(st):
    """Decoration digest (spins excluded) used to group the FeNiCr inner split."""
    lat = (np.round(st.lattice.matrix, 4) + 0.0).ravel().tolist()
    sites = sorted((site.specie.symbol, tuple((np.round(site.frac_coords % 1.0, 4) + 0.0).tolist()))
                   for site in st)
    return hashlib.md5(repr((lat, sites)).encode()).hexdigest()


def load_structure(i):
    """pymatgen Structure on the IDEAL lattice + per-site embedding index."""
    sid = ids[i]
    if SYSTEM == "stfo":
        st = Structure.from_file(f"{DATA}/new_bench/{sid}.cif")     # Xe kept (Z=54 = vacancy channel)
        assert st.num_sites == 160, (sid, st.num_sites)
        idx = [site.specie.Z for site in st]
    elif SYSTEM == "ptni":
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor
        from neuralce.nce.features import field_from_atoms, atoms_from_field, CIF_DIR
        atoms = read(f"{CIF_DIR}/{sid}.cif")
        field, dev = field_from_atoms(atoms, m=3)
        ideal = atoms_from_field(field, a=3.924)                          # snapped ideal fcc, a = 3.924 A
        assert len(ideal) == 108 and Counter(ideal.get_chemical_symbols()).get("Ni", 0) == items[i]["n_ni"], sid
        st = AseAtomsAdaptor.get_structure(ideal)
        idx = [site.specie.Z for site in st]
    else:
        st = Structure.from_file(f"{DATA}/feni_cr/{sid}.cif")
        spins = np.asarray(json.loads(FENICR_SPINS[sid]), dtype=float)
        assert len(spins) == st.num_sites == items[i]["n_atoms"], (sid, len(spins), st.num_sites)
        assert set(np.unique(spins)).issubset({-1.0, 0.0, 1.0}), sid
        # spin folded into the embedding index: Z + 30*(s+1)  (Cr 24/54/84, Fe 26/56/86, Ni 28/58/88 < 100)
        idx = [int(site.specie.Z) + 30 * (int(s) + 1) for site, s in zip(st, spins)]
    return st, idx


if SYSTEM == "fenicr":
    _df = pd.read_csv(f"{DATA}/feni_cr/detailed_info.csv")
    FENICR_SPINS = dict(zip(_df["id"].astype(str), _df["spins"]))

# ── graphs (cached) ─────────────────────────────────────────────────────────────────────────────────
gpath = f"{OUT}/graphs.pt"
if os.path.exists(gpath):
    graphs, groups = torch.load(gpath, weights_only=False)
    print(f"  graphs loaded from {gpath}")
else:
    t0 = time.time()
    graphs, groups = [], []
    for i in range(len(ids)):
        st, idx = load_structure(i)
        d = dg_util.structureToData(st, E=float(y[i]), normalize_E=False,
                                    r_cutoff=HP["r_cutoff"], dr=HP["dr"], max_neighbors=HP["max_neighbors"])
        d.x = torch.tensor(idx, dtype=torch.float)          # embedding index per site (repo: atomic number)
        d.sid = ids[i]
        graphs.append(d)
        groups.append(decoration_key(st) if SYSTEM == "fenicr" else str(strat[i]))
        if i % 100 == 0:
            print(f"    graph {i}/{len(ids)}  nodes={d.num_nodes} edges={d.num_edges}  ({time.time()-t0:.0f}s)")
    torch.save((graphs, groups), gpath)
    print(f"  graphs built in {time.time()-t0:.0f}s -> {gpath}")
groups = np.asarray(groups)
num_edge_features = graphs[0].edge_attr.shape[1]
assert num_edge_features == int(round(HP["r_cutoff"] / HP["dr"])), num_edge_features
assert all(abs(float(g.y) - y[i]) < 1e-6 for i, g in enumerate(graphs))   # y stored as float32 in Data
assert all(g.sid == ids[i] for i, g in enumerate(graphs))


# ── inner 85/15 split (train / early-stopping val) ─────────────────────────────────────────────────
def inner_split(trainval_idx, seed):
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    if SYSTEM == "fenicr":
        gss = GroupShuffleSplit(n_splits=1, test_size=HP["val_frac"], random_state=seed)
        tr, va = next(gss.split(trainval_idx, groups=groups[trainval_idx]))
        return trainval_idx[tr], trainval_idx[va]
    keys = strat[trainval_idx]
    cnt = Counter(keys)
    forced = trainval_idx[np.array([cnt[k] < 2 for k in keys])]      # single-member keys -> train
    splittable = trainval_idx[np.array([cnt[k] >= 2 for k in keys])]
    try:
        tr, va = train_test_split(splittable, test_size=HP["val_frac"], random_state=seed,
                                  stratify=strat[splittable])
    except ValueError:
        tr, va = train_test_split(splittable, test_size=HP["val_frac"], random_state=seed)
    return np.concatenate([tr, forced]), np.asarray(va)


# ── model ───────────────────────────────────────────────────────────────────────────────────────────
def make_model():
    cls = getattr(dg_models, HP["model"])
    if HP["model"] in ("Transformer", "GAT"):
        m = cls(num_hidden_layers=HP["num_hidden_layers"], num_hidden_channels=HP["num_hidden_channels"],
                num_heads=HP["num_heads"], num_edge_features=num_edge_features, device=device)
    else:
        m = cls(num_hidden_layers=HP["num_hidden_layers"], num_hidden_channels=HP["num_hidden_channels"],
                num_edge_features=num_edge_features, device=device)
    m.Conv_list = torch.nn.ModuleList(m.Conv_list)   # repo bug fix: register conv layers as parameters
    return m.to(device)


@torch.no_grad()
def predict(model, idx, mu, sd):
    model.eval()
    out = []
    for b in DataLoader([graphs[i] for i in idx], batch_size=128, shuffle=False):
        b = b.to(device)
        out.append((model(b).view(-1) * sd + mu).cpu().numpy())
    return np.concatenate(out)


def train_one(train_idx, val_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    ytr = y[train_idx]
    mu, sd = float(ytr.mean()), float(ytr.std())
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", **HP["sched"])
    loader = DataLoader([graphs[i] for i in train_idx], batch_size=HP["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best = (np.inf, 0, None); bad = 0
    t0 = time.time()
    for ep in range(1, HP["max_epochs"] + 1):
        model.train()
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            loss = torch.mean((model(b).view(-1) - (b.y.view(-1) - mu) / sd) ** 2)
            loss.backward()
            opt.step()
        val_mae = float(np.mean(np.abs(predict(model, val_idx, mu, sd) - y[val_idx])))
        sched.step(val_mae)
        if val_mae < best[0]:
            best = (val_mae, ep, copy.deepcopy(model.state_dict())); bad = 0
        else:
            bad += 1
        if ep % 100 == 0 or ep == 1:
            print(f"      ep {ep:4d} train_loss={loss.item():.4f} val_mae={val_mae*1000:.2f} meV "
                  f"best={best[0]*1000:.2f}@{best[1]} lr={opt.param_groups[0]['lr']:.1e}", flush=True)
        if bad >= HP["patience"]:
            break
    model.load_state_dict(best[2])
    return model, mu, sd, dict(best_val_mae_meV=best[0] * 1000, best_epoch=best[1], epochs_run=ep,
                               n_params=n_params, wall_sec=time.time() - t0)


# ── outer CV ────────────────────────────────────────────────────────────────────────────────────────
FOLDS = [int(f) for f in args.folds.split(",")] if args.folds else list(range(10))
if args.smoke:
    FOLDS = FOLDS[:2]
rows, runs = [], []
train_sets = {}
T0 = time.time()
for f in FOLDS:
    test_idx = np.where(fold == f)[0]
    trainval_idx = np.where(fold != f)[0]                 # includes anchors (fold = -1)
    assert len(test_idx) == sum(1 for it in items if it["fold"] == f)
    assert not set(test_idx) & set(trainval_idx)
    for seed in SEEDS:
        tr, va = inner_split(trainval_idx, seed=1000 * seed + f)
        assert not set(tr) & set(va) and not (set(tr) | set(va)) & set(test_idx)
        assert len(tr) + len(va) == len(trainval_idx)
        print(f"--- fold {f} seed {seed}: train/val/test = {len(tr)}/{len(va)}/{len(test_idx)}")
        model, mu, sd, info = train_one(tr, va, seed)
        pred = predict(model, test_idx, mu, sd)
        train_sets[(f, seed)] = set(ids[i] for i in tr) | set(ids[i] for i in va)
        for i, p in zip(test_idx, pred):
            rows.append(dict(id=ids[i], y=y[i], pred=float(p), fold=f, seed=seed, n_atoms=int(n_atoms_arr[i])))
        e = np.abs(pred - y[test_idx]) * 1000
        sel = n_atoms_arr[test_idx] == 108 if SYSTEM == "fenicr" else np.ones(len(test_idx), bool)
        info.update(fold=f, seed=seed, n_test=int(len(test_idx)), n_train=int(len(tr)), n_val=int(len(va)),
                    test_mae_meV=float(e[sel].mean()) if sel.any() else None,
                    test_mae_all_meV=float(e.mean()))
        runs.append(info)
        print(f"    fold {f} seed {seed}: test MAE = {info['test_mae_meV']} meV/atom "
              f"(all sizes {info['test_mae_all_meV']:.2f}), best_ep={info['best_epoch']}, "
              f"epochs={info['epochs_run']}, params={info['n_params']}, wall={info['wall_sec']:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/oof.csv", index=False)
        json.dump(runs, open(f"{OUT}/runs.json", "w"), indent=1)

# ── gates + aggregation ─────────────────────────────────────────────────────────────────────────────
oof = pd.DataFrame(rows)
json_fold = dict(zip(ids, fold))
for r in oof.itertuples():                                   # each prediction from a model that never saw the id
    assert r.id not in train_sets[(r.fold, r.seed)], r.id
    assert json_fold[r.id] == r.fold, r.id
for f in FOLDS:
    n_json = sum(1 for it in items if it["fold"] == f)
    for seed in SEEDS:
        assert (oof[(oof.fold == f) & (oof.seed == seed)].shape[0] == n_json), (f, seed)
print("GATES PASS: fold sizes match JSON; every oof prediction is out-of-training by id")

metric_mask = (oof.n_atoms == 108) if SYSTEM == "fenicr" else np.ones(len(oof), bool)


def fold_mae(df):
    out = []
    for f in FOLDS:
        s = df[(df.fold == f)]
        out.append(float(np.mean(np.abs(s.pred - s.y)) * 1000) if len(s) else None)
    return out


ens = oof[metric_mask].groupby(["id", "fold"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
fm_ens = fold_mae(ens)
per_seed = {}
for seed in SEEDS:
    fm = fold_mae(oof[metric_mask & (oof.seed == seed)])
    v = [x for x in fm if x is not None]
    per_seed[seed] = dict(fold_mae=fm, mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else None)
v = [x for x in fm_ens if x is not None]
seed_avg_fold = [float(np.mean([per_seed[s]["fold_mae"][k] for s in SEEDS])) if fm_ens[k] is not None else None
                 for k in range(len(FOLDS))]
sv = [x for x in seed_avg_fold if x is not None]
pooled = ens
summary = dict(
    system=SYSTEM, model=HP["model"], hp=HP, seeds=SEEDS, folds=FOLDS,
    n_structures=len(ids), n_test_metric=int(metric_mask.sum() // len(SEEDS)),
    fold_mae=fm_ens, mae_mean=float(np.mean(v)), mae_std=float(np.std(v, ddof=1)) if len(v) > 1 else None,
    per_seed=per_seed,
    seed_average_fold_mae=seed_avg_fold, seed_average_mean=float(np.mean(sv)),
    seed_average_std=float(np.std(sv, ddof=1)) if len(sv) > 1 else None,
    pooled_rmse_ens=float(np.sqrt(np.mean((pooled.pred - pooled.y) ** 2)) * 1000),
    pooled_mae_ens=float(np.mean(np.abs(pooled.pred - pooled.y)) * 1000),
    params=int(runs[0]["n_params"]),
    epochs=dict(best_epoch=[r["best_epoch"] for r in runs], epochs_run=[r["epochs_run"] for r in runs],
                best_epoch_mean=float(np.mean([r["best_epoch"] for r in runs]))),
    wall_time_sec=dict(per_run=[round(r["wall_sec"]) for r in runs], total=round(time.time() - T0),
                       mean_per_run=float(np.mean([r["wall_sec"] for r in runs]))),
    graph=dict(r_cutoff=HP["r_cutoff"], max_neighbors=HP["max_neighbors"], dr=HP["dr"],
               num_edge_features=int(num_edge_features), nodes_example=int(graphs[0].num_nodes)),
    repo_bugfix="Conv_list wrapped in torch.nn.ModuleList (original: plain list -> conv params untrained)",
    device=str(device),
)
if SYSTEM == "stfo":
    comp = dict(zip(ids, strat))
    ens["comp"] = ens.id.map(comp)
    summary["comp_mae_ens"] = {int(c): float(np.mean(np.abs(s.pred - s.y)) * 1000) for c, s in ens.groupby("comp")}
    summary["comp_n"] = {int(c): int(len(s)) for c, s in ens.groupby("comp")}
    summary["comp_mae_per_seed"] = {
        int(seed): {int(c): float(np.mean(np.abs(s.pred - s.y)) * 1000)
                    for c, s in oof[oof.seed == seed].assign(comp=lambda d: d.id.map(comp)).groupby("comp")}
        for seed in SEEDS}
if SYSTEM == "fenicr":
    ens_all = oof.groupby(["id", "fold"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
    fa = fold_mae(ens_all)
    summary["fold_mae_all_sizes"] = fa
    summary["mae_all_sizes_mean"] = float(np.mean(fa))
    summary["n_108_per_fold"] = [int(((ens.fold == f)).sum()) for f in FOLDS]
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("fold_mae", "mae_mean", "mae_std", "seed_average_mean",
                                           "seed_average_std", "pooled_rmse_ens", "params")}, indent=1))
print({s: (round(p["mean"], 2), None if p["std"] is None else round(p["std"], 2)) for s, p in per_seed.items()})
if SYSTEM == "stfo":
    print("comp MAE (ens):", {c: round(m, 2) for c, m in summary["comp_mae_ens"].items()})
print(f"saved {OUT}/oof.csv, {OUT}/summary.json  total {(time.time()-T0)/60:.1f} min")

# small outputs -> tracked results tree (read by scripts/analysis/*)
RES = f"{RESULTS}/cv/external/disorder_gnn/{args.results_tag or os.path.basename(os.path.normpath(OUT))}"
os.makedirs(RES, exist_ok=True)
for fn in ("oof.csv", "summary.json"):
    shutil.copy2(f"{OUT}/{fn}", f"{RES}/{fn}")
print(f"copied oof.csv, summary.json -> {RES}")
