"""Independent cross-checks of the FeNiCr SRO result (is the pipeline wrong?).

V1 snapshot == final cif species (order mapping), composition;  V2 alpha recomputed straight from the final structure with
ase, independent code path;  V3 MC-path energy of the final 1300 K structure == training-path prediction (icet features
-> model) and both vs a random configuration;  V4 what the run did: fraction of accepted spin vs swap moves is not
logged, so report the spin state (mean spin per species) of the final structure.

Argument: a run tag under $RUNS/fenicr_select/<tag>/multiT (the model tag, $MODELS/nce/<tag>/model.pkl), or the full path
of the <tag> directory.  Env: MODEL (model.pkl override), FENICR_SELECT_ROOT (default $RUNS/fenicr_select).
Run: python scripts/mc/fenicr_verify_sro.py <tag | /path/to/tag_dir>   (evaluates the model: GPU/CPU compute node)
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, sys, numpy as np, jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
from ase.io import read
from ase.neighborlist import neighbor_list
from icet import ClusterSpace
from neuralce.nce.mc_model import load_model, setup, train_path_energy
from neuralce.paths import MODELS, RUNS

if len(sys.argv) < 2:
    sys.exit(f"usage: python {sys.argv[0]} <run tag under $NCE_RUNS/fenicr_select, or a run directory>")
arg = sys.argv[1].rstrip("/")
ROOT = os.environ.get("FENICR_SELECT_ROOT", f"{RUNS}/fenicr_select")
TAGDIR = arg if os.path.isdir(arg) else f"{ROOT}/{arg}"; tag = os.path.basename(TAGDIR); D = f"{TAGDIR}/multiT"
for _p in (f"{D}/snapshots.npz", os.environ.get("MODEL", f"{MODELS}/nce/{tag}/model.pkl")):   # the Fe-Ni-Cr runs and models are not shipped: skip cleanly
    if not os.path.exists(_p):
        print(f"[skip] {os.path.basename(__file__)}: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under FENICR_SELECT_ROOT/<tag>/multiT, $NCE_MODELS/nce (MODEL) to run", flush=True); sys.exit(0)
z = np.load(f"{D}/snapshots.npz"); S = z["sigma"]; temps = z["temps"]; k = int(np.argmin(np.abs(temps - 1300)))
fin = read(f"{D}/final_T{temps[k]:.0f}_r{k}.extxyz", format="extxyz"); sym = np.array(fin.get_chemical_symbols())
name = {0: "Cr", 1: "Fe", 2: "Ni"}; snap_sym = np.array([name[v] for v in S[-1, k]])
print(f"[V1] last snapshot == final cif species: {np.array_equal(snap_sym, sym)}   composition {dict(zip(*np.unique(sym, return_counts=True)))}")
i, j, d = neighbor_list("ijd", fin, 2.7); c = {s: (sym == s).mean() for s in ("Fe", "Ni", "Cr")}
for A, B in (("Fe", "Ni"), ("Fe", "Cr"), ("Cr", "Ni"), ("Ni", "Fe"), ("Cr", "Fe"), ("Ni", "Cr")):
    m = sym[i] == A; pB = (sym[j[m]] == B).mean(); print(f"[V2] alpha1({A}-{B}) from the cif, ase neighbours: {1 - pB / c[B]:+.3f}   (P({B}|{A}) = {pB:.3f}, c_{B} = {c[B]:.3f})")
mdl = load_model(os.environ.get("MODEL", f"{MODELS}/nce/{tag}/model.pkl")); cs = ClusterSpace(mdl['primitive'], list(mdl['cutoffs']), mdl['chemical_symbols'])
sp = fin.get_initial_magnetic_moments(); mc = setup(mdl, fin, dtype=None)
e_mc = mc['energy'](mc['sig0'], sp[mc['perm']]); e_tr = train_path_energy(mdl, cs, fin, sp)
print(f"[V3] final 1300 K structure: MC path {e_mc:.6f}  training path (icet features -> model) {e_tr:.6f} eV/atom  diff {abs(e_mc-e_tr):.2e}")
rng = np.random.RandomState(3); rnd = fin.copy(); Z = rnd.numbers.copy(); rng.shuffle(Z); rnd.numbers = Z; sp_r = rng.choice([-1.0, 0.0, 1.0], len(rnd))
mc2 = setup(mdl, rnd, dtype=None); e_r = mc2['energy'](mc2['sig0'], sp_r[mc2['perm']]); e_r_tr = train_path_energy(mdl, cs, rnd, sp_r)
print(f"[V3] random structure: MC path {e_r:.6f}  training path {e_r_tr:.6f}  -> segregation lowers E by {(e_r - e_mc)*1e3:.1f} meV/atom according to the model")
for s in ("Fe", "Ni", "Cr"): print(f"[V4] final spins {s}: mean {sp[sym == s].mean():+.3f}, |mean| of +1/-1/0 counts {np.unique(sp[sym == s], return_counts=True)[1].tolist()}")
print("VERIFY_DONE")
