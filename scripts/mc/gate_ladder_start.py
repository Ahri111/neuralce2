"""Production-size start gate for the tiled ladder runs: check that a tiled start npz (scripts/mc/ptni_ladder_tile_init.py)
really carries the source structures' energy into the cell the run will use (131k, 1M atoms), so nothing above the
smoke-tested size goes unchecked.

Every replica's starting energy per atom must equal the source structure's energy per atom (the npz records both
src_E_check, the model's own re-evaluation, and src_E_run, what the source run reported).  A wrong site mapping
between the source cell and the tiled cell shows up as ~1e-3 eV/atom; float64 summation over the extra sites gives
~1e-8 (nce) / ~1e-7 (icet, float32 tables).

  icet: --fw icet --npz N --init_cif C --supercell K --ce CE.ce
  nce:  --fw conv --npz N --init_cif C --supercell K --ckpt model.pkl      (``--fw nce`` = ``conv``)
K is the run's --supercell (the target cell is the init cif repeated K^3), i.e. exactly what the launcher passes.
Needs a GPU node for the large cells; submit it, do not run it on the login node.
Run: python scripts/mc/gate_ladder_start.py ...   (exit 0 = PASS)
"""
import argparse, os, sys, numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fw", required=True, choices=["icet", "conv", "nce"]); ap.add_argument("--npz", required=True)
    ap.add_argument("--init_cif", required=True); ap.add_argument("--supercell", type=int, required=True)
    ap.add_argument("--ce"); ap.add_argument("--ckpt")
    ap.add_argument("--tol", type=float, default=None, help="eV/atom; default 2e-6 icet, 1e-6 conv")
    a = ap.parse_args()
    if a.fw == "nce": a.fw = "conv"
    from ase.io import read
    from neuralce.mc.init_structures import load_init_npz
    z = np.load(a.npz); K = int(a.supercell); tol = a.tol if a.tol is not None else (2e-6 if a.fw == "icet" else 1e-6)
    raw = read(a.init_cif)
    if a.fw == "icet":
        from icet import ClusterExpansion
        from neuralce.ce.icet_mc import _rebuild_as_clean_supercell
        from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
        ce = ClusterExpansion.read(a.ce); prim = ce._cluster_space.primitive_structure
        atoms = _rebuild_as_clean_supercell(raw, prim).repeat((K, K, K)) if K > 1 else _rebuild_as_clean_supercell(raw, prim)
        gce = GPUClusterExpansion(ce, atoms, table_mode="implicit", energy_scale=float(K ** 3))
        nums, T = load_init_npz(a.npz, prim, atoms, z["temps"])
        b = atoms.copy(); E = np.zeros(len(T))
        for r in range(len(T)):
            b.numbers = nums[r]; E[r] = float(gce.energy(gce.encode(b))) / len(atoms)
    else:
        from neuralce.nce.mc_model import load_model, setup
        from neuralce.nce.mcmc import sigma_from_numbers
        mdl = load_model(a.ckpt); prim = mdl['primitive']
        atoms = raw.repeat((K, K, K)) if K > 1 else raw
        mc = setup(mdl, atoms, dtype=None)
        nums, T = load_init_npz(a.npz, prim, atoms, z["temps"])
        sig = sigma_from_numbers(mc, nums)
        E = np.array([float(mc['energy'](sig[r])) for r in range(len(T))])
    d_chk = float(np.max(np.abs(E - z["src_E_check"]))); d_run = float(np.max(np.abs(E - z["src_E_run"])))
    print(f"[start] {a.fw} {os.path.basename(a.npz)}: {len(atoms)} atoms, {len(T)} replicas, source {z['src_dir']}")
    print(f"[start] max |E_start/atom - E_source/atom| = {d_chk:.2e} eV (tol {tol:g}) | vs the source run's own energies {d_run:.2e}")
    ok = d_chk <= tol
    print("[SUMMARY] LADDER START GATE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
