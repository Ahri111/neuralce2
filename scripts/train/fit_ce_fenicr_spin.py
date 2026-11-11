"""Package the benchmark FeNiCr "icet + Heisenberg pair" model as a depth-0 neural-CE model.pkl.

The benchmark model (models/ce/fenicr: icet [6,5] chemical CE per atom + 12 species-resolved 1NN/2NN s_i s_j
couplings, one Lasso fit) is put into the depth-0 cluster-convolution format, so the validated conv MC
(species-resolved spin pairs recomputed from the current configuration, spins move with the atoms, DFT spin rule,
snapshots) samples EXACTLY that energy.
Depth 0:  E = site-mean(h . lin) + b  with h = [local cluster features, spin-pair planes]; site-mean(h) =
[cluster vector, X_spin] (the training code asserts both identities), so lin = [ECI (without the zerolet), J per bond],
b = zerolet.
Gate: on every 108-atom training structure, the new model's energy (training path) == ce.predict + X_spin . J
(and the MC path via gate_final).

Inputs : models/ce/fenicr/{feni_cr_icet_chemspin.ce, feni_cr_icet_spin_orbits.pkl}, data/feni_cr/design_matrix.npz,
         template models/nce/fenicr_s1_mean_c65_d0/model.pkl, data/feni_cr/{detailed_info.csv, folds_decoration.json,
         mapped.pkl, *.cif}.
Output : models/nce/fenicr_icet_heis_bench/model.pkl (--out overrides).
Run    : python scripts/train/fit_ce_fenicr_spin.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, ast, json, pickle, argparse, numpy as np, pandas as pd
from icet import ClusterExpansion, ClusterSpace
from ase.io import read
from neuralce.paths import DATA, MODELS
from neuralce.nce.mc_model import load_model, train_path_energy, gate_final
from neuralce.ce.spin_pairs import reembed_to_reference_frame, pbc_perm_raw_to_mapped, spin_features, PAIRS

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{MODELS}/nce/fenicr_icet_heis_bench", help="output directory of model.pkl")
ap.add_argument("--template", default=f"{MODELS}/nce/fenicr_s1_mean_c65_d0/model.pkl", help="depth-0 model.pkl whose metadata is reused")
args = ap.parse_args()
for _p in (args.template, f"{MODELS}/ce/fenicr/feni_cr_icet_chemspin.ce", f"{MODELS}/ce/fenicr/feni_cr_icet_spin_orbits.pkl", f"{DATA}/feni_cr/design_matrix.npz",
           f"{DATA}/feni_cr/detailed_info.csv", f"{DATA}/feni_cr/folds_decoration.json", f"{DATA}/feni_cr/mapped.pkl"):   # the Fe-Ni-Cr inputs are not shipped: skip cleanly
    if not os.path.exists(_p):
        print(f"[skip] {os.path.basename(__file__)}: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_MODELS/ce/fenicr, $NCE_MODELS/nce (--template), $NCE_DATA/feni_cr to run", flush=True); raise SystemExit(0)
OUT = args.out; os.makedirs(OUT, exist_ok=True)
T = load_model(args.template)
print("template keys/values:", {k: (T[k] if not hasattr(T[k], "shape") and not isinstance(T[k], (list, dict)) else (np.shape(T[k]) if hasattr(T[k], "shape") else type(T[k]).__name__)) for k in T if k not in ("params", "train_ids", "train_y", "train_pred", "primitive")}, flush=True)
print("template spin fields:", T["spin"], T["spin_edges"], T["spin_pairs"], T["spin_species"], T.get("spin_node_lo"), T.get("spin_values"), "keep", len(T["keep"]), T["keep"][:5], "...", T["keep"][-5:], flush=True)
ce = ClusterExpansion.read(f"{MODELS}/ce/fenicr/feni_cr_icet_chemspin.ce"); SP = pickle.load(open(f"{MODELS}/ce/fenicr/feni_cr_icet_spin_orbits.pkl", "rb"))
z = np.load(f"{DATA}/feni_cr/design_matrix.npz", allow_pickle=True)
print("pkl keys:", list(SP.keys()), "| design keys:", list(z.files), flush=True)
names = [po.get("name", po.get("label", str(k))) for k, po in enumerate(SP["pair_orbits"])]
Jb = np.array([jv / len(po["instances"]) for jv, po in zip(SP["j_values"], SP["pair_orbits"])])   # J per bond (eV)
print("spin couplings (per bond, meV):", {n: round(1e3 * j, 3) for n, j in zip(names, Jb)}, flush=True)
spin_names = [str(s) for s in z[[f for f in z.files if "spin" in f and "name" in f][0]]] if any("spin" in f and "name" in f for f in z.files) else None
print("design X_spin names:", spin_names, flush=True)
if spin_names is not None:                                                 # order J like the X_spin columns
    Jb = np.array([Jb[names.index(n)] if n in names else np.nan for n in spin_names]); assert np.all(np.isfinite(Jb)), (names, spin_names)
cs_ce = ce.get_cluster_space_copy(); cs = ClusterSpace(T["primitive"], list(T["cutoffs"]), T["chemical_symbols"])
assert len(cs) == len(cs_ce) == len(ce.parameters), (len(cs), len(cs_ce), len(ce.parameters))
# data: the 516 rows of the spin design, spins in mapped order (same path as train_nce.py)
CIF = f"{DATA}/feni_cr"; df = pd.read_csv(f"{CIF}/detailed_info.csv"); spec = json.load(open(f"{DATA}/feni_cr/folds_decoration.json"))
df = df[~df.id.astype(str).isin(set(map(str, spec["dropped_exact_duplicates"])))].reset_index(drop=True)
atoms_all = pickle.load(open(f"{DATA}/feni_cr/mapped.pkl", "rb")); keep_ids = set(str(i) for i in z["ids"])
sel = [k for k, i in enumerate(df.id.astype(str)) if i in keep_ids]; df = df.iloc[sel].reset_index(drop=True); atoms = [atoms_all[k] for k in sel]
PRIM = T["primitive"]; spins = []
for i, a in zip(df.id.astype(str), atoms):
    raw = reembed_to_reference_frame(read(f"{CIF}/{i}.cif"), PRIM.cell.array); s_raw = np.array(ast.literal_eval(df.spins[df.id.astype(str) == i].iloc[0]), float)
    perm = pbc_perm_raw_to_mapped(raw, a); s_m = np.empty(len(a)); s_m[perm] = s_raw; spins.append(s_m)
y = (df.total_energy / df.n_atoms).to_numpy()
E_b = np.array([float(ce.predict(a)) + float(spin_features(a, s)[:12] @ Jb) for a, s in zip(atoms, spins)])
print(f"benchmark model vs DFT on {len(y)} structures: RMSE {1e3*np.sqrt(np.mean((E_b-y)**2)):.2f}  (>=32 atoms {1e3*np.sqrt(np.mean(((E_b-y)[df.n_atoms>=32])**2)):.2f}) meV/atom", flush=True)
# depth-0 model: channels = all cluster-vector columns except the zerolet (constant), then the 12 spin planes; mu = 0, sd = 1, ys = 1, ym = 0
nchem = len(cs); keep = list(range(1, nchem)) + list(range(nchem, nchem + 12))
w = np.concatenate([np.asarray(ce.parameters)[1:], Jb]); b0 = float(ce.parameters[0])
M = dict(T); M.update(keep=np.asarray(keep), mu=np.zeros(len(keep)), sd=np.ones(len(keep)), ys=1.0, ym=0.0, depth=0, seeds=[0], lasso_coef=w, lasso_b=b0,
         params=[dict(layers=[], head=dict(w=np.zeros(0), b=np.asarray(b0)), lin=w)], train_pred=None, val_mae=None,
         note="benchmark icet [6,5] + Heisenberg-pair Lasso fit (models/ce/fenicr) in depth-0 form; built by scripts/train/fit_ce_fenicr_spin.py")
pickle.dump(M, open(f"{OUT}/model.pkl", "wb")); print("wrote", f"{OUT}/model.pkl", flush=True)
big = [k for k in range(len(atoms)) if len(atoms[k]) >= 108]
E_n = np.array([train_path_energy(M, cs, atoms[k], spins[k], 0) for k in big])
print(f"GATE training path: {len(big)} structures (108 atoms), max |E_d0 - E_benchmark| = {1e3*np.abs(E_n - E_b[big]).max():.2e} meV/atom", flush=True)
mae_mc, dmax = gate_final(M, [atoms[k] for k in big], y[big], 0, spins_list=[spins[k] for k in big])
print(f"GATE MC path: MAE vs DFT {1e3*mae_mc:.2f} meV/atom, max |MC - training path| {dmax:.2e}", flush=True)
