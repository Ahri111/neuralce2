"""Decode the lowest-energy state of the icet PT runs for every (composition, K) and describe it: composition conserved per
sublattice, CE energy of the decoded structure == E_min, vacancy-vacancy pairs (90 deg 2.8 A / 180 deg 3.96 A) and Fe
coordination, per 160 sites, compared with A00.  Same decoding as stfo_icet_min_check.py (K=1), generalised: the
framework tiles the input cif with atoms.repeat((K,K,K)) when CE_SUPERCELL_PERIODIC=1 and writes sigma in that input order.

Inputs : $RUNS/stfo_mc/<NAME>/c<comp>_K<K>/mcmc_result.npz (env MC_DIR overrides the run directory), the icet CE (env CE),
         data/new_bench/A00_<comp>.cif.
Env    : CE (required), NAME (required), COMPS (default 125:250:375:500:750), KS (default 1:2:3), MC_DIR.
Outputs: report on stdout; <run dir>/c<comp>_K<K>/icet_min.cif.
Run    : CE=models/ce/stfo/<x>.ce NAME=<run> python scripts/analysis/stfo_icet_min_describe.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from icet import ClusterExpansion
from neuralce.paths import DATA, RUNS
from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{DATA}/new_bench"] + ([os.environ["CE"]] if os.environ.get("CE") else []), "$NCE_DATA/new_bench, $NCE_MODELS/ce/stfo (CE), $NCE_RUNS (NAME, MC_DIR)")
CE = os.environ["CE"]; NAME = os.environ["NAME"]; MC = os.environ.get("MC_DIR", f"{RUNS}/stfo_mc/{NAME}")
ce = ClusterExpansion.read(CE)

def describe(a):
    i, j, d = neighbor_list("ijd", a, 4.2); s = np.array(a.get_chemical_symbols()); n160 = len(a) / 160
    isX = np.isin(s, ["X", "Xe"]); vv = isX[i] & isX[j]; n90 = ((d < 3.2) & vv).sum() / 2 / n160; n180 = ((d >= 3.2) & vv).sum() / 2 / n160
    fe = np.where(s == "Fe")[0]; co = np.array([((i == k) & (s[j] == "O") & (d < 2.4)).sum() for k in fe])
    return dict(VV90=round(n90, 1), VV180=round(n180, 1), Fe_coord={c: round(float((co == c).sum() / n160), 1) for c in (3, 4, 5, 6)}, comp={q: int((s == q).sum()) for q in ("Ti", "Fe", "O", "Xe", "X")})

for comp in [int(c) for c in os.environ.get("COMPS", "125:250:375:500:750").split(":")]:
    for K in [int(k) for k in os.environ.get("KS", "1:2:3").split(":")]:
        f = f"{MC}/c{comp}_K{K}/mcmc_result.npz"
        if not os.path.exists(f): continue
        small = read(f"{DATA}/new_bench/A00_{comp}.cif")
        if K > 1:                                                      # exactly what run_pt_mc does for --supercell K K K (no ideal_map / inert given)
            from neuralce.ce.icet_mc import _rebuild_as_clean_supercell
            atoms = _rebuild_as_clean_supercell(small, ce._cluster_space.primitive_structure).repeat((K, K, K))
        else: atoms = small
        gce = GPUClusterExpansion(ce, atoms, table_mode="implicit", energy_scale=float(K ** 3)); sig0 = np.asarray(gce.encode(atoms))
        perm = np.asarray(gce.perm) if hasattr(gce, "perm") else np.arange(len(atoms)); inv_perm = np.argsort(perm); sig0 = sig0[inv_perm]
        sp_of = {}
        for sl in gce.sublattices:
            for c in np.asarray(sl["indices"]): sp_of[int(perm[c])] = list(sl["chemical_symbols"])
        def decode(sig):
            b = atoms.copy(); syms = b.get_chemical_symbols()
            for o, sp in sp_of.items(): syms[o] = sp[int(sig[o])]
            b.set_chemical_symbols(syms); return b
        z = np.load(f); smin = np.asarray(z["sigma_min"]); N = len(atoms); a0, am = decode(sig0), decode(smin)
        assert a0.get_chemical_symbols() == atoms.get_chemical_symbols(), "decode(sig0) != input"
        e0 = float(ce.predict(a0)) * K ** 3 / N; em = float(ce.predict(am)) * K ** 3 / N          # ce value = total E of 160-site cell -> x K^3 / N = eV/site
        ok_E = abs(em * N - float(z["E_min"])) < 1e-2
        print(f"x={comp/1000:<6} K={K} N={N:5d}  E_min-A00 {(em - e0) * 1e3:+6.1f} meV/site  (decoded CE energy == MC E_min: {ok_E}, |diff| {abs(em * N - float(z['E_min'])):.1e} eV)")
        print(f"      A00   : {describe(a0)}"); print(f"      MC-min: {describe(am)}")
        write(f"{MC}/c{comp}_K{K}/icet_min.cif", am)
print("MIN_DESCRIBE_DONE")
