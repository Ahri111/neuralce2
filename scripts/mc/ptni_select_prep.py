"""PtNi model selection, step 1: initial structures + ideal ordered references.

Writes (OUTDIR, default $RUNS/ptni_select):
    init_x{25,50,75}.cif      random fcc Ni/Pt, M x M x M conventional cell (M = 4: 256 atoms, a = 3.924), Ni listed first
    ref_L12_Ni3Pt.cif, ref_L10_NiPt.cif, ref_L12_Pt3Ni.cif   ideal ordered structures, same cell
                              (the M = 4 versions are shipped as models/structures/ptni/ref_*.cif)
    ref_energies.json         model energy (eV/atom) of the ideal structures for every saved seed
The ordered references are what the MC must find at low T (experiment: L1_2 Ni3Pt, L1_0 NiPt, L1_2 Pt3Ni).

Env: M (default 4), OUTDIR, MODEL (default $MODELS/nce/nipt_v3_nomax_sel/model.pkl).
Run: python scripts/mc/ptni_select_prep.py   (GPU/CPU node; evaluates the model)
"""
import os, json, numpy as np
from ase.build import bulk
from ase.io import write
from neuralce.paths import MODELS, RUNS

M = int(os.environ.get("M", "4")); OUT = os.environ.get("OUTDIR", f"{RUNS}/ptni_select"); os.makedirs(OUT, exist_ok=True)
A = 3.924
conv = bulk("Ni", "fcc", a=A, cubic=True).repeat((M, M, M))            # 256 atoms
frac = conv.get_scaled_positions() * M                                # conventional-cell units
sub = ((np.round(frac * 2).astype(int) % 2) * np.array([1, 2, 4])).sum(1)   # 0:(0,0,0) 3:(.5,.5,0) 5:(.5,0,.5) 6:(0,.5,.5)
sub = {0: 0, 3: 1, 5: 2, 6: 3}.__getitem__; sub = np.array([sub(int(s)) for s in ((np.round(frac * 2).astype(int) % 2) * np.array([1, 2, 4])).sum(1)])
assert all((sub == k).sum() == M ** 3 for k in range(4))
rng = np.random.RandomState(0)


def save(atoms_syms, name):
    a = conv.copy(); a.set_chemical_symbols(atoms_syms)
    order = np.argsort([0 if s == "Ni" else 1 for s in atoms_syms], kind="stable")   # Ni first (cif species order)
    a = a[order]; write(f"{OUT}/{name}.cif", a); return a


refs = {}
for x, nm in ((0.25, "x25"), (0.5, "x50"), (0.75, "x75")):
    syms = np.array(["Pt"] * len(conv)); syms[rng.choice(len(conv), int(round(x * len(conv))), replace=False)] = "Ni"
    save(list(syms), f"init_{nm}")
# ideal ordered structures on the 4 fcc sublattices
syms = np.where(sub == 0, "Pt", "Ni"); refs["L12_Ni3Pt"] = save(list(syms), "ref_L12_Ni3Pt")          # Pt on one sublattice
syms = np.where(np.isin(sub, [0, 1]), "Pt", "Ni"); refs["L10_NiPt"] = save(list(syms), "ref_L10_NiPt")  # Pt on two (layers along z)
syms = np.where(sub == 0, "Ni", "Pt"); refs["L12_Pt3Ni"] = save(list(syms), "ref_L12_Pt3Ni")
# model energies of the references, per seed
import jax; jax.config.update("jax_enable_x64", True)
from neuralce.nce.mc_model import load_model, setup
mdl = load_model(os.environ.get("MODEL", f"{MODELS}/nce/nipt_v3_nomax_sel/model.pkl"))
res = {}
for k, a in refs.items():
    res[k] = {}
    for si in range(len(mdl['params'])):
        mc = setup(mdl, a, seed_idx=si, dtype=None); res[k][f"seed_idx{si}"] = mc['energy'](mc['sig0'])
    print(k, {s: round(v, 5) for s, v in res[k].items()}, flush=True)
json.dump(res, open(f"{OUT}/ref_energies.json", "w"), indent=1)
print("PREP_DONE")
