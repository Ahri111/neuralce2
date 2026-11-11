"""Model energies (eV/atom, 256-atom cell, no aliasing) of competing ordered PtNi structures per saved seed:
A3B: L1_2 vs DO22 (L1_2 with an antiphase boundary every 2 cubic cells along z);  AB: L1_0 vs L1_1 (CuPt, (111) layers)
plus the low-T final configurations of the 32-atom PT runs (repeated 2x2x2) that showed 0.5/0.5 sublattice patterns.

Env: MODEL (default $MODELS/nce/nipt_v3_nomax_sel/model.pkl), PT32_DIR (root of the 32-atom PT runs, pt_<x>_seed*;
default $RUNS/ptni_select_m2; skipped when absent).   Run: python scripts/mc/ptni_candidates.py
"""
import os, glob, numpy as np
from ase.build import bulk
from ase.io import read
from neuralce.paths import MODELS, RUNS
import jax; jax.config.update("jax_enable_x64", True)
from neuralce.nce.mc_model import load_model, setup

M = 4; conv = bulk("Ni", "fcc", a=3.924, cubic=True).repeat((M, M, M)); frac = conv.get_scaled_positions() * M
code = ((np.round(frac * 2).astype(int) % 2) * np.array([1, 2, 4])).sum(1); sub = np.array([{0: 0, 3: 1, 5: 2, 6: 3}[int(c)] for c in code])
h = np.round(frac.sum(1) * 2).astype(int)                       # x+y+z in a/2 units (even on fcc sites)
zcell = np.floor(frac[:, 2] + 1e-6).astype(int)                 # which cubic cell along z
cands = {}
def mk(mask_pt, name): a = conv.copy(); a.set_chemical_symbols(np.where(mask_pt, "Pt", "Ni").tolist()); cands[name] = a
mk(sub == 0, "A3B L1_2 Ni3Pt (Pt on 1 sublattice)")
mk(np.where((zcell // 2) % 2 == 0, sub == 0, sub == 1), "A3B DO22 Ni3Pt (APB every 2 cells)")
mk(sub != 0, "AB3 L1_2 Pt3Ni")
mk(~np.where((zcell // 2) % 2 == 0, sub == 0, sub == 1), "AB3 DO22 Pt3Ni")
mk(np.isin(sub, [0, 1]), "AB L1_0 NiPt")
mk((h // 2) % 2 == 0, "AB L1_1 NiPt (CuPt, (111) layers)")
PT32 = os.environ.get("PT32_DIR", f"{RUNS}/ptni_select_m2")
for x in ("x50", "x75"):
    for run in sorted(glob.glob(f"{PT32}/pt_{x}_seed*")):
        z = np.load(f"{run}/energies.npz"); a = read(f"{run}/final_T{z['temps'][0]:.0f}_r0.cif").repeat((2, 2, 2)); cands[f"32-atom PT low-T final {os.path.basename(run)} x8"] = a
mdl = load_model(os.environ.get("MODEL", f"{MODELS}/nce/nipt_v3_nomax_sel/model.pkl"))
print(f"{'structure':58s}" + "".join(f"{'seed'+str(s):>12s}" for s in range(len(mdl['params']))) + "   x_Ni")
for name, a in cands.items():
    es = []
    for s in range(len(mdl['params'])):
        mc = setup(mdl, a, seed_idx=s, dtype=None); es.append(mc['energy'](mc['sig0']))
    print(f"{name:58s}" + "".join(f"{e:12.5f}" for e in es) + f"   {np.mean(np.array(a.get_chemical_symbols())=='Ni'):.3f}", flush=True)
print("CAND_DONE")
