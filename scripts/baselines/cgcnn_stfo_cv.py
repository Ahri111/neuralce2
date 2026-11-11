"""CGCNN / ALIGNN / SchNet graph-network baselines on STFO with the folds of data/benchmark_folds.json (387 dedup ids,
JSON y labels) instead of a StratifiedKFold of their own.  Reports the test RMSE per 128 graph atoms (Sr excluded
from the graph, energy / 128) and the MAE per 160 sites (the benchmark-table metric).

Graph  : Sr (Z=38) removed, Xe read as the vacancy dummy species (atom_init feature row = 0); cutoff 4.08 A,
         shell one-hot edges (edges 0 / 2.41 / 3.135 / 3.705 / 4.08), 16 nearest neighbours.
Node   : the 92-dim CGCNN paper features of data/atom_features/atom_init.json.
Models : original paper hyper-parameters (CGCNN Xie & Grossman 2018: atom_fea 64, 3 conv, h 128, edge 41;
         ALIGNN Choudhary & DeCost 2021: 64 / 4 / 256 / 80; SchNet Schuett 2018: 128 / 6 / 128 / 20, n_filters 128),
         padded sum readout.
Train  : AdamW (lr 2e-3, wd 1e-4), grad clip 1, batch 32, 20 % stratified validation inside the training fold
         (seed 42 + fold), early stopping on validation MAE, best checkpoint evaluated on the test fold.

External requirements: the optional `gnn` extra (pip install -e ".[gnn]": flax) and the frozen JAX/flax ports in
scripts/bench/gnn_baselines/ (create_neuralce, build_graph_lite, get_specie_number); that sibling directory is put on
sys.path below.  JAX_DEFAULT_MATMUL_PRECISION=highest is recommended (source env.sh).
Env    : STFO_FOLDS_JSON (default data/benchmark_folds.json; e.g. data/benchmark_folds_corrected.json),
         STFO_LABEL_TAG (output-dir suffix, e.g. _lab12), BL_OUT_SUFFIX, BL_MAX_EPOCHS (300), BL_PATIENCE (50),
         BL_LR (2e-3), BL_MODELS (comma subset of cgcnn,alignn,schnet).
Outputs: $RUNS/external/disorder_gnn/gate_cgcnn_stfo<BL_OUT_SUFFIX><STFO_LABEL_TAG>/{cv10_results.csv, cv10_summary.json};
         both are also copied to $RESULTS/cv/external/disorder_gnn/<same name>/.
Run    : python scripts/baselines/cgcnn_stfo_cv.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, sys, re, time, copy, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from neuralce.paths import DATA, RUNS, RESULTS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))   # frozen GNN ports (sibling dir)

_FJ = os.environ.get("STFO_FOLDS_JSON", f"{DATA}/benchmark_folds.json")                   # the STFO inputs are not shipped: skip cleanly when they are missing
if not os.path.exists(f"{DATA}/new_bench/detailed_info.csv"):
    print(f"[skip] {os.path.basename(__file__)}: {DATA}/new_bench/detailed_info.csv is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_DATA/new_bench to run", flush=True); sys.exit(0)
if "stfo" not in json.load(open(_FJ)):
    print(f"[skip] {os.path.basename(__file__)}: {_FJ} has no 'stfo' entry (STFO / Fe-Ni-Cr data are not public); provide it under STFO_FOLDS_JSON to run", flush=True); sys.exit(0)

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from pymatgen.core import Structure, DummySpecies

from gnn_baselines.graph_lite import build_graph_lite, get_specie_number
from gnn_baselines.neuralce_gnn_jax import create_neuralce

ATOM_INIT_JSON = f"{DATA}/atom_features/atom_init.json"   # 92-dim CGCNN paper features

OUT_DIR = Path(f"{RUNS}/external/disorder_gnn/gate_cgcnn_stfo" + os.environ.get("BL_OUT_SUFFIX", "")
               + os.environ.get("STFO_LABEL_TAG", ""))      # separate output for corrected labels
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "fold_ckpts").mkdir(exist_ok=True)

# ── training settings ──
SEED = 42
N_FOLDS = 10
MAX_EPOCHS = int(os.environ.get("BL_MAX_EPOCHS", 300))
PATIENCE = int(os.environ.get("BL_PATIENCE", 50))
BATCH_SIZE = 32
VAL_FRAC_INSIDE_FOLD = 0.20
LR = float(os.environ.get("BL_LR", 2.0e-3))
GRAD_CLIP = 1.0

# ── 3 baseline models, original paper hyper-parameters ──
BASELINES = {
    'cgcnn': {
        'atom_fea_len': 64,
        'n_conv': 3,
        'h_fea_len': 128,
        'edge_fea_len': 41,
    },
    'alignn': {
        'atom_fea_len': 64,
        'n_conv': 4,
        'h_fea_len': 256,
        'edge_fea_len': 80,
    },
    'schnet': {
        'atom_fea_len': 128,
        'n_conv': 6,
        'h_fea_len': 128,
        'edge_fea_len': 20,
        'n_filters': 128,
    },
}

# ── Data load ──
print("=== CV10 baselines: CGCNN / ALIGNN / SchNet ===")
t0 = time.time()

NEWBENCH_DIR = f"{DATA}/new_bench"
NEWBENCH_CSV = f"{DATA}/new_bench/detailed_info.csv"

# Shared graph HP (matches the stable NeuralCE graph)
GRAPH_HP = {
    'cutoff': 4.08, 'n_shells': 4,
    'shell_edges': [0.0, 2.41, 3.135, 3.705, 4.08],
    'max_num_nbr': 16,
    'edge_basis': 'shell_onehot',
    'conv_type': 'product',
    'n_bessel': None,
}


def load_cif_safe(path):
    """Load a CIF and restore Xe sites to DummySpecies('X0+') (pymatgen writes the vacancy dummy as 'X' = Xe)."""
    struct = Structure.from_file(path)
    for i in range(len(struct)):
        if str(struct[i].specie) == 'Xe':
            struct[i] = DummySpecies('X0+'), struct[i].frac_coords
    return struct


def load_data(cif_dir, csv_path, id_col, comp_regex, exclude_z=None):
    """Structures of cif_dir with a label in csv_path -> list of dicts (sorted by cif file name); no spins."""
    exclude_z = exclude_z or set()
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        id_col = 'id' if 'id' in df.columns else 'cif_id'
    energy_map = dict(zip(df[id_col].astype(str), df['total_energy'].values))
    structures = []
    cif_files = sorted([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
    n_skip = 0
    for cif_file in cif_files:
        cif_id = cif_file.replace('.cif', '')
        if cif_id not in energy_map:
            n_skip += 1
            continue
        crystal = load_cif_safe(os.path.join(cif_dir, cif_file))
        spins = np.zeros(len(crystal), dtype=np.float32)
        if exclude_z:
            keep_idx = [i for i, site in enumerate(crystal) if get_specie_number(site.specie) not in exclude_z]
            crystal = Structure.from_sites([crystal[i] for i in keep_idx])
            spins = spins[keep_idx]
        comp_match = re.search(comp_regex, cif_id)
        comp_code = int(comp_match.group(1)) if comp_match else 0
        structures.append({'cif_id': cif_id, 'total_energy': energy_map[cif_id], 'crystal': crystal,
                           'spin_states': spins, 'comp_code': comp_code, 'n_atoms': len(crystal)})
    print(f"Loaded {len(structures)} structures ({n_skip} skipped)")
    if exclude_z:
        print(f"  Excluded Z: {sorted(exclude_z)}; atoms per crystal: {sorted(set(s['n_atoms'] for s in structures))}")
    return structures


def build_all_graphs(structures, cutoff, n_shells, max_num_nbr, include_sisj=False, shell_edges=None,
                     edge_basis='shell_onehot', n_bessel=8, n_atoms_pad=None):
    """Build and batch all graphs (zero-row padding to n_atoms_pad, atom_mask 1 = real site)."""
    raw = {'atom_fea': [], 'nbr_fea': [], 'nbr_fea_idx': [], 'spin_fea': [], 'energies': [], 'comps': [], 'n_atoms_list': []}
    for s in structures:
        af, nf, nfi, sf = build_graph_lite(s, cutoff, n_shells, max_num_nbr, include_sisj,
                                           shell_edges=shell_edges, edge_basis=edge_basis, n_bessel=n_bessel)
        raw['atom_fea'].append(af); raw['nbr_fea'].append(nf); raw['nbr_fea_idx'].append(nfi); raw['spin_fea'].append(sf)
        raw['energies'].append(s['total_energy']); raw['comps'].append(s['comp_code']); raw['n_atoms_list'].append(af.shape[0])
    n_atoms_arr = np.array(raw['n_atoms_list'])
    if n_atoms_pad is None:
        n_atoms_pad = int(n_atoms_arr.max())
    n_species = raw['atom_fea'][0].shape[-1]
    edge_dim = raw['nbr_fea'][0].shape[-1]
    out = {'atom_fea': [], 'nbr_fea': [], 'nbr_fea_idx': [], 'spin_fea': [], 'atom_mask': []}
    n_padded = 0
    for i in range(len(raw['atom_fea'])):
        n_at = raw['atom_fea'][i].shape[0]
        if n_at == n_atoms_pad:
            out['atom_fea'].append(raw['atom_fea'][i]); out['nbr_fea'].append(raw['nbr_fea'][i])
            out['nbr_fea_idx'].append(raw['nbr_fea_idx'][i]); out['spin_fea'].append(raw['spin_fea'][i])
            out['atom_mask'].append(np.ones((n_atoms_pad,), dtype=np.float32))
        elif n_at < n_atoms_pad:
            n_pad = n_atoms_pad - n_at
            out['atom_fea'].append(np.concatenate([raw['atom_fea'][i], np.zeros((n_pad, n_species), dtype=np.float32)], axis=0))
            out['nbr_fea'].append(np.concatenate([raw['nbr_fea'][i], np.zeros((n_pad, max_num_nbr, edge_dim), dtype=np.float32)], axis=0))
            out['nbr_fea_idx'].append(np.concatenate([raw['nbr_fea_idx'][i], np.zeros((n_pad, max_num_nbr), dtype=np.int32)], axis=0))
            out['spin_fea'].append(np.concatenate([raw['spin_fea'][i], np.zeros((n_pad, 1), dtype=np.float32)], axis=0))
            out['atom_mask'].append(np.concatenate([np.ones((n_at,), dtype=np.float32), np.zeros((n_pad,), dtype=np.float32)], axis=0))
            n_padded += 1
        else:
            raise ValueError(f"Structure {i} has {n_at} atoms > n_atoms_pad={n_atoms_pad}")
    if n_padded > 0:
        print(f"    Padded {n_padded}/{len(raw['atom_fea'])} structures to {n_atoms_pad} atoms")
    return {'atom_fea': jnp.array(np.stack(out['atom_fea'])), 'nbr_fea': jnp.array(np.stack(out['nbr_fea'])),
            'nbr_fea_idx': jnp.array(np.stack(out['nbr_fea_idx'])), 'spin_fea': jnp.array(np.stack(out['spin_fea'])),
            'atom_mask': jnp.array(np.stack(out['atom_mask'])), 'energies': jnp.array(raw['energies']),
            'comps': raw['comps'], 'n_atoms_list': n_atoms_arr}


print("[1] Loading newbench DFT")
data = load_data(cif_dir=NEWBENCH_DIR, csv_path=NEWBENCH_CSV, id_col='id', comp_regex=r'_(\d+)$', exclude_z={38})
# ── benchmark_folds.json: ids / labels / folds (dedup 387, group-mean labels for B04_125 / B07_125) ──
_bf = json.load(open(os.environ.get("STFO_FOLDS_JSON", f"{DATA}/benchmark_folds.json")))["stfo"]
_bf_items = {it["id"]: it for it in _bf["items"]}
data = [d for d in data if d["cif_id"] in _bf_items]
for d in data:
    d["total_energy"] = _bf_items[d["cif_id"]]["y"] * 160.0     # JSON y is per 160 sites
    d["json_fold"] = _bf_items[d["cif_id"]]["fold"]
assert len(data) == _bf["n"] == 387, len(data)
n = len(data)
print(f"  N = {n} (benchmark_folds.json)")

print("[2] Building graphs")
dataset = build_all_graphs(
    data, GRAPH_HP['cutoff'], GRAPH_HP['n_shells'],
    max_num_nbr=GRAPH_HP['max_num_nbr'],
    include_sisj=False, edge_basis=GRAPH_HP['edge_basis'],
    n_bessel=8, shell_edges=GRAPH_HP['shell_edges'],
)
N_ATOMS = int(dataset['atom_fea'].shape[1])

# ── Override atom_fea with the atom_init.json 92-dim CGCNN features (paper convention) ──
print(f"[2b] Loading atom_init.json (CGCNN paper 92-dim features) ...")
with open(ATOM_INIT_JSON) as f:
    atom_emb_map = {int(k): np.asarray(v, dtype=np.float32) for k, v in json.load(f).items()}
N_crystals = len(data)
atom_fea_92 = np.zeros((N_crystals, N_ATOMS, 92), dtype=np.float32)
for i, struct in enumerate(data):
    crystal = struct['crystal']
    n_real = len(crystal)
    for j in range(n_real):
        z = get_specie_number(crystal[j].specie)
        if z in atom_emb_map:
            atom_fea_92[i, j] = atom_emb_map[z]
        # else: leave zero (vacancy dummy species, Z = 0)
dataset['atom_fea'] = jnp.asarray(atom_fea_92)
print(f"  atom_fea shape replaced: {dataset['atom_fea'].shape}  (92-dim CGCNN/ALIGNN paper features)")

comps_arr = np.asarray([d['comp_code'] for d in data])
print(f"  N_ATOMS = {N_ATOMS}, comps = {sorted(set(comps_arr))}")


def _flatten_batch(af, nf, nfi, msk):
    """Flatten (B, N, ...) -> (B*N, ...) with index offsets for nbr_fea_idx."""
    B = af.shape[0]
    af_flat  = af.reshape(-1, af.shape[-1])
    offsets  = jnp.arange(B)[:, None, None] * N_ATOMS
    nfi_flat = (nfi + offsets).reshape(-1, nfi.shape[-1])
    nf_flat  = nf.reshape(-1, nf.shape[-2], nf.shape[-1])
    return af_flat, nf_flat, nfi_flat, B


def make_model_and_state(model_name, hp, rng):
    kwargs = dict(
        atom_fea_len=hp['atom_fea_len'],
        n_conv=hp['n_conv'],
        h_fea_len=hp['h_fea_len'],
        edge_fea_len=hp['edge_fea_len'],
    )
    if model_name == 'schnet':
        kwargs['n_filters'] = hp.get('n_filters', hp['atom_fea_len'])
    model = create_neuralce(
        model_type=model_name, pool_mode='padded',
        readout_type='sum', **kwargs,
    )
    # Init with a FLATTENED single-sample input (B=1 -> 128 atoms)
    af1 = dataset['atom_fea'][0:1]
    nf1 = dataset['nbr_fea'][0:1]
    nfi1 = dataset['nbr_fea_idx'][0:1]
    msk1 = dataset['atom_mask'][0:1]
    af_f, nf_f, nfi_f, B = _flatten_batch(af1, nf1, nfi1, msk1)
    init_kw = dict(
        atom_fea=af_f, nbr_fea=nf_f, nbr_fea_idx=nfi_f,
        batch_size=B, n_atoms_per_crystal=N_ATOMS,
        atom_mask=msk1,
    )
    params = model.init(rng, **init_kw)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    return model, params, n_params


def train_one_fold(model_name, hp, train_idx, val_idx, test_idx, fold_seed):
    rng = jax.random.PRNGKey(fold_seed)
    model, init_params, n_params = make_model_and_state(model_name, hp, rng)

    tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adamw(LR, weight_decay=1e-4),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=init_params, tx=tx)

    @jax.jit
    def train_step(state, batch):
        af, nf, nfi, msk, y = batch
        def loss_fn(params):
            af_f, nf_f, nfi_f, B = _flatten_batch(af, nf, nfi, msk)
            pred = model.apply(
                params, atom_fea=af_f, nbr_fea=nf_f, nbr_fea_idx=nfi_f,
                batch_size=B, n_atoms_per_crystal=N_ATOMS,
                atom_mask=msk,
            )
            return jnp.mean((pred.ravel() - y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_batch(params, af, nf, nfi, msk):
        af_f, nf_f, nfi_f, B = _flatten_batch(af, nf, nfi, msk)
        return model.apply(
            params, atom_fea=af_f, nbr_fea=nf_f, nbr_fea_idx=nfi_f,
            batch_size=B, n_atoms_per_crystal=N_ATOMS,
            atom_mask=msk,
        )

    def eval_indices(state, idx):
        preds_list, targs_list = [], []
        for start in range(0, len(idx), 64):
            b = idx[start:start + 64]
            p = eval_batch(state.params, dataset['atom_fea'][b], dataset['nbr_fea'][b],
                           dataset['nbr_fea_idx'][b], dataset['atom_mask'][b])
            preds_list.append(np.asarray(p).ravel())
            targs_list.append(np.asarray(dataset['energies'][b]).ravel())
        return np.concatenate(preds_list), np.concatenate(targs_list)

    best_val = float('inf')
    best_params = None
    pat = 0
    best_epoch = 0

    for ep in range(1, MAX_EPOCHS + 1):
        perm = np.array(jax.random.permutation(
            jax.random.PRNGKey(fold_seed + ep), jnp.array(train_idx)))
        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2: continue
            batch = (dataset['atom_fea'][idx], dataset['nbr_fea'][idx],
                     dataset['nbr_fea_idx'][idx], dataset['atom_mask'][idx],
                     dataset['energies'][idx])
            state, _ = train_step(state, batch)

        val_pred, val_targ = eval_indices(state, val_idx)
        val_mae = float(np.mean(np.abs(val_pred - val_targ)))

        if val_mae < best_val:
            best_val = val_mae
            best_params = jax.tree_util.tree_map(lambda x: np.asarray(x), state.params)
            pat = 0
            best_epoch = ep
        else:
            pat += 1
        if pat >= PATIENCE:
            break

    # eval test/train with the best checkpoint
    state = state.replace(params=best_params)
    test_pred, test_targ = eval_indices(state, test_idx)
    train_pred, train_targ = eval_indices(state, train_idx)
    test_rmse = float(np.sqrt(np.mean((test_pred - test_targ) ** 2)))
    train_rmse = float(np.sqrt(np.mean((train_pred - train_targ) ** 2)))
    test_srcc = float(spearmanr(test_pred, test_targ)[0])
    test_mae = float(np.mean(np.abs(test_pred - test_targ)))
    return {
        'train_rmse_meV_per_atom': float(train_rmse / N_ATOMS * 1000),
        'test_rmse_meV_per_atom':  float(test_rmse / N_ATOMS * 1000),
        'test_mae_meV_per_atom':   float(test_mae / N_ATOMS * 1000),
        'test_pred': test_pred.tolist(), 'test_targ': test_targ.tolist(),
        'test_srcc': test_srcc,
        'best_epoch': best_epoch,
        'best_val_mae': best_val,
        'n_params': int(n_params),
    }


# ── 10-fold CV loop ──
print(f"\n[3] {N_FOLDS}-fold CV (benchmark folds)")
json_fold = np.asarray([d["json_fold"] for d in data])
ids_arr = np.asarray([d["cif_id"] for d in data])
print(f"  fold sizes (JSON): {sorted(__import__('collections').Counter(json_fold).items())}")

results = []
for fold_idx in range(N_FOLDS):
    test_idx = np.where(json_fold == fold_idx)[0]
    trainval_idx = np.where(json_fold != fold_idx)[0]        # anchors (-1) always train
    assert len(test_idx) == sum(1 for it in _bf["items"] if it["fold"] == fold_idx)
    assert not set(test_idx) & set(trainval_idx)
    # stratified val split
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=VAL_FRAC_INSIDE_FOLD,
        random_state=SEED + fold_idx, stratify=comps_arr[trainval_idx])
    train_idx = np.asarray(train_idx); val_idx = np.asarray(val_idx)

    print(f"\n--- fold {fold_idx}: train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)} ---")
    _sel = os.environ.get("BL_MODELS")
    _items = [(k, v) for k, v in BASELINES.items()
              if (not _sel) or k in _sel.split(",")]
    for model_name, hp in _items:
        t_fold = time.time()
        info = train_one_fold(model_name, hp, train_idx, val_idx, test_idx, SEED + fold_idx)
        info.update(fold=fold_idx, method=model_name, wall_sec=float(time.time() - t_fold),
                    test_ids=ids_arr[test_idx].tolist(),
                    test_mae_meV_per_160=float(info['test_mae_meV_per_atom'] * N_ATOMS / 160),
                    test_rmse_meV_per_160=float(info['test_rmse_meV_per_atom'] * N_ATOMS / 160))
        assert not set(ids_arr[test_idx]) & (set(ids_arr[train_idx]) | set(ids_arr[val_idx]))
        results.append(info)
        print(f"  [{model_name}]  test_RMSE={info['test_rmse_meV_per_atom']:.2f} meV/atom, "
              f"SRCC={info['test_srcc']:.4f}, best_ep={info['best_epoch']}, "
              f"params={info['n_params']:,}, wall={info['wall_sec']:.0f}s")

# Aggregate
df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "cv10_results.csv", index=False)

summary = {}
for m in df['method'].unique():
    sub = df[df['method'] == m]
    summary[m] = {
        'test_rmse_mean_meV_per_atom': float(sub['test_rmse_meV_per_atom'].mean()),
        'test_rmse_std_meV_per_atom':  float(sub['test_rmse_meV_per_atom'].std()),
        'test_mae_mean_meV_per_atom':  float(sub['test_mae_meV_per_atom'].mean()),
        'test_mae_std_meV_per_atom':   float(sub['test_mae_meV_per_atom'].std()),
        'test_mae_mean_meV_per_160':   float(sub['test_mae_meV_per_160'].mean()),
        'test_mae_std_meV_per_160':    float(sub['test_mae_meV_per_160'].std()),
        'test_rmse_mean_meV_per_160':  float(sub['test_rmse_meV_per_160'].mean()),
        'test_rmse_std_meV_per_160':   float(sub['test_rmse_meV_per_160'].std()),
        'fold_rmse_meV_per_atom': sub.sort_values('fold')['test_rmse_meV_per_atom'].tolist(),
        'fold_mae_meV_per_160': sub.sort_values('fold')['test_mae_meV_per_160'].tolist(),
        'n_atoms_graph': int(N_ATOMS),
        'reference_row': 'cv10_baselines_fair cgcnn: test RMSE/128 = 6.81 +- 0.99 (fold std)',
        'best_epochs': sub.sort_values('fold')['best_epoch'].tolist(),
        'test_srcc_mean': float(sub['test_srcc'].mean()),
        'test_srcc_std':  float(sub['test_srcc'].std()),
        'n_params': int(sub['n_params'].iloc[0]),
        'wall_sec_mean': float(sub['wall_sec'].mean()),
    }
with open(OUT_DIR / "cv10_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== SUMMARY ===")
for m, s in summary.items():
    print(f"  {m:8s}:  Test RMSE/128 = {s['test_rmse_mean_meV_per_atom']:.2f} ± {s['test_rmse_std_meV_per_atom']:.2f}, "
          f"MAE/160 = {s['test_mae_mean_meV_per_160']:.2f} ± {s['test_mae_std_meV_per_160']:.2f} meV/atom, "
          f"SRCC = {s['test_srcc_mean']:.4f}, n_params = {s['n_params']:,}")

RES = Path(f"{RESULTS}/cv/external/disorder_gnn/{OUT_DIR.name}")
RES.mkdir(parents=True, exist_ok=True)
for fn in ("cv10_results.csv", "cv10_summary.json"):
    shutil.copy2(OUT_DIR / fn, RES / fn)
print(f"\nsaved: {OUT_DIR}  (copied cv10_results.csv, cv10_summary.json -> {RES})")
print(f"Total: {(time.time()-t0)/60:.1f} min")
