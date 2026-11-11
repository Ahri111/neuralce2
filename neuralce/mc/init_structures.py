"""Per-replica starting structures for the central MC entry (--init_npz); shared by the icet PT and ce_cnn paths.

npz format (written e.g. by scripts/mc/ptni_ladder_tile_init.py):
    numbers   (R, N) uint8   atomic numbers of replica r's starting structure
    positions (N, 3) float   Cartesian positions (A) of the N sites, one geometry for all replicas
    cell      (3, 3) float   cell of that geometry; must equal the MC cell
    temps     (R,)   float   temperature (K) each structure belongs to; must equal the run's temperature list
The sites may be listed in ANY order: they are matched onto the MC cell by lattice position (basis index of the
primitive + primitive-cell offset reduced into the supercell), never by atom index."""
import numpy as np


def _site_keys(prim, atoms):
    """one int64 per site identifying its lattice site (basis, reduced offset); equal for the same site in any atom order."""
    from neuralce.nce.gpu_layer0_tiled import site_decompose
    basis, off_red, T, _, _ = site_decompose(prim, atoms)
    S = 1 << 17
    if np.abs(off_red).max() >= S or basis.max() >= 256:
        raise ValueError("cell too large for the lattice-site key")
    o = (np.asarray(off_red, np.int64) + S)
    keys = (np.asarray(basis, np.int64) << 54) | (o[:, 0] << 36) | (o[:, 1] << 18) | o[:, 2]
    if len(np.unique(keys)) != len(keys):
        raise ValueError("two atoms occupy the same lattice site")
    return keys, np.asarray(T)


def map_to_reference(prim, ref_atoms, cell, positions):
    """idx (N,): site j = idx[i] of the given geometry sits on the lattice site of ref_atoms[i]."""
    from ase import Atoms
    cell = np.asarray(cell, float); positions = np.asarray(positions, float)
    if cell.shape != (3, 3) or not np.allclose(cell, ref_atoms.cell.array, rtol=0, atol=1e-3):
        raise ValueError(f"init_npz cell differs from the MC cell:\n{cell}\nvs\n{ref_atoms.cell.array}")
    if len(positions) != len(ref_atoms):
        raise ValueError(f"init_npz has {len(positions)} sites, the MC cell {len(ref_atoms)}")
    other = Atoms(numbers=np.ones(len(positions), int), positions=positions, cell=ref_atoms.cell.array, pbc=True)
    k_ref, T_ref = _site_keys(prim, ref_atoms); k_oth, T_oth = _site_keys(prim, other)
    if not np.array_equal(T_ref, T_oth):
        raise ValueError(f"supercell matrices differ: {T_ref.tolist()} vs {T_oth.tolist()}")
    o_ref, o_oth = np.argsort(k_ref, kind="stable"), np.argsort(k_oth, kind="stable")
    if not np.array_equal(k_ref[o_ref], k_oth[o_oth]):
        raise ValueError("init_npz sites do not coincide with the MC cell's lattice sites")
    idx = np.empty(len(k_ref), np.int64); idx[o_ref] = o_oth
    return idx


def load_init_npz(path, prim, ref_atoms, temps, tol_T=0.5):
    """-> numbers (R, N) uint8 in the atom order of ref_atoms, and the npz temperatures.  Checks: cell, lattice sites,
    one structure per run temperature (|dT| <= tol_T), composition of every replica == composition of ref_atoms."""
    z = np.load(path)
    numbers = np.asarray(z["numbers"]); T_npz = np.asarray(z["temps"], float).reshape(-1); temps = np.asarray(temps, float).reshape(-1)
    if numbers.ndim != 2 or numbers.shape[0] != len(T_npz):
        raise ValueError(f"init_npz numbers {numbers.shape} / temps {T_npz.shape} inconsistent")
    if len(temps) != len(T_npz) or np.any(np.abs(temps - T_npz) > np.maximum(tol_T, 1e-5 * T_npz)):   # relative at high T (ce_cnn carries betas in float32)
        raise ValueError(f"init_npz temperatures {np.round(T_npz, 1).tolist()} != run temperatures {np.round(temps, 1).tolist()}")
    idx = map_to_reference(prim, ref_atoms, z["cell"], z["positions"])
    out = numbers[:, idx]
    comp = np.bincount(np.asarray(ref_atoms.numbers), minlength=256)
    for r in range(out.shape[0]):
        if not np.array_equal(np.bincount(out[r].astype(np.int64), minlength=256), comp):
            raise ValueError(f"init_npz replica {r}: composition differs from the MC cell")
    return out, T_npz
