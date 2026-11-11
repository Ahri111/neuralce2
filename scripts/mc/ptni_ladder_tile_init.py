"""Per-replica starting structures for a bigger cell, built by tiling the FINAL structures of a finished ladder run
(e.g. 131k -> 1M atoms with --tile 2 2 2).  The output is the --init_npz input of ``python -m neuralce.mc.run``
(format: neuralce/mc/init_structures.py).  Every target temperature takes the source replica at the nearest temperature.

Energy check: the source structures are re-evaluated with the model and compared with the run's own final energies,
which catches any atom-order mix-up before a long run starts.

  icet source: --fw icet --src_dir <run with mcmc_result.npz> --ce <.ce> --src_init_cif <that run's init cif> --src_supercell K
  nce source:  --fw conv --src_dir <run with final_sigma.npy> --ckpt <model.pkl> --src_init_cif <that run's init cif> --src_supercell K
               (without --src_init_cif the run's cifs are read instead, which is only viable for small cells)
  (``--fw nce`` is accepted as an alias of ``conv``, the folder name used by the ladder runs)

Run: python scripts/mc/ptni_ladder_tile_init.py --fw icet|conv --src_dir D ... --tile 2 2 2 --T_grid_file T.txt --out init.npz
"""
import os, re, glob, argparse, numpy as np


def icet_source(src_dir, ce_path, init_cif, K, check_idx):
    """Source run's MC cell (atom order of neuralce.ce.icet_mc.run_pt_mc), atomic numbers per replica, temperatures,
    the run's final energies (eV/atom) and the re-evaluated energies of the chosen replicas."""
    from ase.io import read
    from ase.data import atomic_numbers
    from icet import ClusterExpansion
    from neuralce.ce.icet_mc import _rebuild_as_clean_supercell
    from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
    from neuralce.nce.gpu_layer0_tiled import site_decompose
    ce = ClusterExpansion.read(ce_path); prim = ce._cluster_space.primitive_structure; raw = read(init_cif)
    if K > 1:                                                            # run_pt_mc: clean supercell of the primitive, then tiled
        atoms = _rebuild_as_clean_supercell(raw, prim).repeat((K, K, K))
    else:                                                                # run_pt_mc (implicit, supercell 1): the cif as is if ideal
        try:
            site_decompose(prim, raw); atoms = raw
        except AssertionError:
            atoms = _rebuild_as_clean_supercell(raw, prim)
    z = np.load(f"{src_dir}/mcmc_result.npz"); S = np.asarray(z["sigma_final"]); T = np.asarray(z["temperatures"], float)
    if S.shape[1] != len(atoms):
        raise ValueError(f"sigma_final has {S.shape[1]} sites, rebuilt cell {len(atoms)}: wrong --src_init_cif / --src_supercell")
    gce = GPUClusterExpansion(ce, atoms, table_mode="implicit", energy_scale=float(K ** 3)); info = gce.info
    site_sl_orig = info['site_to_sl'][info['inv_perm']]; Z = np.zeros(S.shape, np.uint8)
    for sl, species in info['sl_species'].items():                      # sigma_final is in input atom order
        m = site_sl_orig == sl; zs = np.array([atomic_numbers[x] for x in species], np.uint8); Z[:, m] = zs[S[:, m]]
    E_run = np.asarray(z["energies"], float)[-1] / len(atoms)            # last sample = end of the last chunk
    E_chk = np.full(len(S), np.nan); a = atoms.copy()
    for r in check_idx:
        a.numbers = Z[r]; E_chk[r] = float(gce.energy(gce.encode(a))) / len(atoms)
    return atoms, Z, T, E_run, E_chk


CIF_READ_MAX = 4096          # ASE's cif reader is O(N^2) in the number of sites: a 16k-atom cif takes hours


def conv_source(src_dir, ckpt, check_idx, init_cif=None, K=1):
    """Source run's MC cell (atom order of the nce runner), atomic numbers per replica, temperatures, the run's final
    energies (eV/atom) and the re-evaluated energies of the chosen replicas.

    Preferred route (--src_init_cif): rebuild the cell the way the run built it (its init cif, tiled K^3) and take the
    configurations from final_sigma.npy, which the runner writes in that same atom order.  Exact and instant.
    Without it the run's final_T*_r*.cif files are read instead, which only works for small cells (CIF_READ_MAX)."""
    from ase.io import read
    from neuralce.nce.mc_model import load_model, setup
    from neuralce.nce.mcmc import sigma_from_numbers, numbers_from_sigma
    e = np.load(f"{src_dir}/energies.npz"); T = np.asarray(e["temps"], float); E_run = np.asarray(e["E_chunk"], float)[-1]
    sig_p = f"{src_dir}/final_sigma.npy"
    if init_cif and os.path.exists(sig_p):
        atoms = read(init_cif)
        if int(K) > 1: atoms = atoms.repeat((int(K), int(K), int(K)))
        mc = setup(load_model(ckpt), atoms, dtype=None)
        if not mc.get('implicit'):
            raise ValueError("conv source needs the implicit geometry")
        S = np.load(sig_p)
        if S.shape[1] != len(atoms):
            raise ValueError(f"final_sigma.npy has {S.shape[1]} sites, rebuilt cell {len(atoms)}: wrong --src_init_cif / --src_supercell")
        Z = numbers_from_sigma(mc, S, atoms)
        print(f"[src] {len(atoms)} atoms, {len(S)} replicas from final_sigma.npy ({init_cif} tiled {K}^3)", flush=True)
    else:
        files = sorted(glob.glob(f"{src_dir}/final_T*_r*.cif"), key=lambda f: int(re.search(r"_r(\d+)\.cif$", f).group(1)))
        if not files:
            raise FileNotFoundError(f"no final_sigma.npy (with --src_init_cif) and no final_T*_r*.cif in {src_dir}")
        atoms = read(files[0]); N = len(atoms)
        if N > CIF_READ_MAX:
            raise ValueError(f"{files[0]} has {N} sites: reading {len(files)} cifs that size takes hours "
                             f"(ASE's reader is O(N^2)).  Pass --src_init_cif / --src_supercell to use final_sigma.npy instead.")
        Z = np.zeros((len(files), N), np.uint8)
        for k, f in enumerate(files):
            a = read(f)
            if int(re.search(r"_r(\d+)\.cif$", f).group(1)) != k or len(a) != N or not np.allclose(a.positions, atoms.positions, atol=1e-4):
                raise ValueError(f"{f}: replica cifs do not share one geometry")
            Z[k] = a.numbers
        mc = None
    E_chk = np.full(len(Z), np.nan)
    if len(check_idx):
        if mc is None:
            mc = setup(load_model(ckpt), atoms, dtype=None)
            if not mc.get('implicit'):
                raise ValueError("conv energy check needs the implicit geometry")
        sig = sigma_from_numbers(mc, Z[list(check_idx)])                 # numbers -> sigma round trip: also checks the decoding
        for j, r in enumerate(check_idx):
            E_chk[r] = mc['energy'](sig[j])
    return atoms, Z, T, E_run, E_chk


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fw", required=True, choices=["icet", "conv", "nce"]); ap.add_argument("--src_dir", required=True)
    ap.add_argument("--ce"); ap.add_argument("--src_init_cif"); ap.add_argument("--src_supercell", type=int, default=1); ap.add_argument("--ckpt")
    ap.add_argument("--tile", nargs=3, type=int, default=[2, 2, 2]); ap.add_argument("--T_grid_file", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--check_replicas", default="all", help="'all', 'none' or comma list of source replica indices to re-evaluate")
    ap.add_argument("--tol", type=float, default=None, help="max |re-evaluated - run final| energy (eV/atom); default 1e-6 icet, 1e-4 conv (float32 run)")
    a = ap.parse_args()
    if a.fw == "nce": a.fw = "conv"
    # number of source replicas: read cheaply before the (possibly expensive) re-evaluation
    n_src = np.load(f"{a.src_dir}/mcmc_result.npz")["temperatures"].shape[0] if a.fw == "icet" else np.load(f"{a.src_dir}/energies.npz")["temps"].shape[0]
    chk = list(range(n_src)) if a.check_replicas == "all" else ([] if a.check_replicas == "none" else [int(x) for x in a.check_replicas.split(",")])
    if a.fw == "icet":
        if not (a.ce and a.src_init_cif): ap.error("--fw icet needs --ce and --src_init_cif (and --src_supercell)")
        atoms, Z, T_src, E_run, E_chk = icet_source(a.src_dir, a.ce, a.src_init_cif, a.src_supercell, chk)
    else:
        if not a.ckpt: ap.error("--fw conv needs --ckpt (model of the source run)")
        atoms, Z, T_src, E_run, E_chk = conv_source(a.src_dir, a.ckpt, chk, a.src_init_cif, a.src_supercell)
    tol = a.tol if a.tol is not None else (1e-6 if a.fw == "icet" else 1e-4)
    if chk:
        pair = float(np.nanmax(np.abs(E_chk - E_run)))                    # replica by replica
        mset = float(np.max(np.abs(np.sort(E_chk[~np.isnan(E_chk)]) - np.sort(E_run[list(chk)]))))   # as a set
        n_off = int(np.sum(np.abs(E_chk - E_run) > tol))
        print(f"[check] {len(chk)} source replicas re-evaluated: max |E_model - E_run| = {pair:.2e} eV/atom replica by replica "
              f"({n_off} above {tol:g}), {mset:.2e} comparing the two as sets (tol {tol:g})")
        if not mset <= tol:
            raise SystemExit("energy check FAILED: the rebuilt source structures do not reproduce the run's energies")
        if n_off:
            print(f"[check] note: {n_off} replicas differ in place.  The nce runner records the chunk energies BEFORE the "
                  f"replica exchange of that chunk, so a run's last energies are one exchange behind its final structures; "
                  f"structure r still belongs to temperature r and the set of energies matches.")
    T_tgt = np.atleast_1d(np.loadtxt(a.T_grid_file, comments="#", dtype=float))
    src_of = np.array([int(np.argmin(np.abs(T_src - t))) for t in T_tgt])
    tile = tuple(int(x) for x in a.tile); big = atoms.repeat(tile); nums = np.zeros((len(T_tgt), len(big)), np.uint8); b = atoms.copy()
    for r, s in enumerate(src_of):
        b.numbers = Z[s]; nums[r] = b.repeat(tile).numbers
    np.savez(a.out, numbers=nums, positions=big.positions, cell=big.cell.array, temps=T_tgt, src_temps=T_src[src_of], src_replica=src_of,
             src_E_run=E_run[src_of], src_E_check=E_chk[src_of], tile=np.array(tile), src_dir=str(a.src_dir), fw=a.fw)
    print(f"[tile] {a.fw} {a.src_dir}: {len(atoms)} -> {len(big)} atoms, {len(T_tgt)} target temperatures "
          f"(source T used: {np.round(T_src[src_of]).astype(int).tolist()}) -> {a.out}")


if __name__ == "__main__":
    main()
