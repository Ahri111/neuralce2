"""STFO: what is the state the orthonormal-CE (icet) MC falls into?  The icet PT at x = 0.5 reaches ~300 meV/site below the
DFT-lowest structure A00 -- far outside anything a 3.7 meV/site CE can be trusted for.  Decode sigma_min of the icet runs,
then check (1) the sublattices are intact (Fe/Ti only on B sites, O/vacancy only on O sites), (2) the icet energy of the
decoded structure equals E_min, (3) the vacancy arrangement (V_O-V_O pairs at 2.8 A / 3.96 A) and the Fe coordination,
compared with A00, (4) the cluster-conv energy of the same structure, and write the structure as cif for a DFT check.

Inputs : the icet PT run $RUNS/stfo_mc/<NAME>/c<comp>_K1/mcmc_result.npz (python -m neuralce.mc.run, K = 1; env MC_DIR
         overrides the run directory), data/new_bench/A00_<comp>.cif, the icet CE (env CE, default
         models/ce/stfo/icet_aug467_al1a_c6_5_4.5_4.ce) and the cluster-conv model models/nce/<TAG>/model.pkl.
Env    : CE, NAME (default icet_aug467_al1a), TAG (default stfo_dd_p10_w10_wd5_aug467_al1a_w0), COMPS (default 125:500), MC_DIR.
Outputs: report on stdout; <run dir>/c<comp>_K1/icet_min.cif.
Run    : python scripts/analysis/stfo_icet_min_check.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from icet import ClusterSpace
from icet import ClusterExpansion
from neuralce.paths import DATA, MODELS, RUNS
from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
from neuralce.nce.mc_model import load_model, train_path_energy
CE = os.environ.get("CE", f"{MODELS}/ce/stfo/icet_aug467_al1a_c6_5_4.5_4.ce"); NAME = os.environ.get("NAME", "icet_aug467_al1a")
TAG = os.environ.get("TAG", "stfo_dd_p10_w10_wd5_aug467_al1a_w0"); MC = os.environ.get("MC_DIR", f"{RUNS}/stfo_mc/{NAME}")
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([CE, f"{MODELS}/nce/{TAG}/model.pkl", f"{DATA}/new_bench", MC], "$NCE_DATA/new_bench, $NCE_MODELS (CE, TAG), $NCE_RUNS (MC_DIR)")
mdl = load_model(f"{MODELS}/nce/{TAG}/model.pkl"); csc = ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"])


def describe(a):
    """vacancy pairs and Fe coordination of a 160-site structure (X = vacancy)"""
    i, j, d = neighbor_list("ijd", a, 4.2); s = np.array(a.get_chemical_symbols())
    isX = np.isin(s, ["X", "Xe"]); vv = isX[i] & isX[j]; n90 = int(((d < 3.2) & vv).sum() // 2); n180 = int(((d >= 3.2) & vv).sum() // 2)
    fe = [k for k in range(len(a)) if s[k] == "Fe"]; co = [int(((i == k) & (s[j] == "O") & (d < 2.4)).sum()) for k in fe]
    return dict(n_X=int(isX.sum()), VV_90=n90, VV_180=n180, Fe_coord={c: co.count(c) for c in (3, 4, 5, 6)}, comp={q: int((s == q).sum()) for q in ("Sr", "Ti", "Fe", "O", "X", "Xe")})


for comp in [int(c) for c in os.environ.get("COMPS", "125:500").split(":")]:
    ce = ClusterExpansion.read(CE); atoms = read(f"{DATA}/new_bench/A00_{comp}.cif")                 # same as run_pt_mc for K = 1: the cif is already an ideal cell -> used as is (its atom order)
    gce = GPUClusterExpansion(ce, atoms, table_mode="implicit"); sig0 = gce.encode(atoms)
    perm = np.asarray(gce.perm) if hasattr(gce, "perm") else np.arange(len(atoms)); inv_perm = np.argsort(perm)      # kernel order is canonical (sigma_canon = sigma_orig[perm]);
    sp_of = {}                                                                                                            # the framework writes sigma_min / sigma_final back in the INPUT order (pt_mcmc_ce.py)
    for sl in gce.sublattices:
        for c in np.asarray(sl["indices"]): sp_of[int(perm[c])] = list(sl["chemical_symbols"])
    SL = [(np.array([o for o, sp in sp_of.items() if sp == list(sl["chemical_symbols"])]), list(sl["chemical_symbols"])) for sl in gce.sublattices]   # original-order indices per active sublattice
    sig0 = np.asarray(sig0)[inv_perm]                                                                                     # encode() gives canonical order -> input order
    z = np.load(f"{MC}/c{comp}_K1/mcmc_result.npz"); smin = np.asarray(z["sigma_min"]); Emin = float(z["E_min"]); N = len(atoms)
    print(f"\n==== x = {comp/1000}: {N} sites, icet E_min {Emin:.3f} eV = {Emin/N:.4f} eV/site; sublattices {[sl['chemical_symbols'] for sl in gce.sublattices]}")
    def decode(sig):
        b = atoms.copy(); syms = b.get_chemical_symbols()
        for o, sp in sp_of.items(): syms[o] = sp[int(sig[o])]
        b.set_chemical_symbols(syms); return b
    a0, am = decode(np.asarray(sig0)), decode(smin); assert a0.get_chemical_symbols() == atoms.get_chemical_symbols(), "decode(sig0) != input structure"
    print(f"  ones in sigma: start {int(np.asarray(sig0).sum())}  MC-min {int(smin.sum())}   (must be equal: swaps conserve the composition)")
    # (1) sublattices intact?  per-sublattice species counts of A00 vs the MC minimum
    for lab, s_ in (("A00", np.asarray(sig0)), ("MC-min", smin)):
        print(f"  {lab}: per-sublattice counts", {"".join(sp): np.bincount(s_[idx], minlength=len(sp)).tolist() for idx, sp in SL}, " (Sr sites untouched)")
    # (2) energies: icet CE on the decoded structures (parameters x 160 -> total energy of the 160-site cell) and the cluster conv
    e0, em = float(ce.predict(a0)), float(ce.predict(am)); print(f"  icet predict: A00 {e0:.3f}  MC-min {em:.3f} eV (MC reported {Emin:.3f})  ->  MC-min - A00 = {(em - e0) / N * 1e3:+.1f} meV/site")
    try:
        c0, cm = train_path_energy(mdl, csc, a0, None, 0), train_path_energy(mdl, csc, am, None, 0); print(f"  cluster conv ({TAG}): A00 {c0:.4f}  MC-min {cm:.4f} eV/site  ->  {(cm - c0) * 1e3:+.1f} meV/site")
    except Exception as e: print("  cluster conv scoring failed:", repr(e)[:200])
    # (3) structure
    print("  A00   :", describe(a0)); print("  MC-min:", describe(am))
    out = f"{MC}/c{comp}_K1/icet_min.cif"; write(out, am); print("  wrote", out)
print("ICET_MIN_CHECK_DONE")
