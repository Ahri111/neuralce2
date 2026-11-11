"""Heisenberg pair features for the Fe-Ni-Cr spin cluster expansion.

Shared by the orthonormal-CE + spin baseline and the neural cluster expansion
spin channels (``neuralce.nce.spin_features``): per-structure pair counts
0.5 * s_i s_j per (neighbour shell, species pair), and the helpers that put a
relaxed fcc cell back onto the reference lattice frame.
"""
from __future__ import annotations
import hashlib
import numpy as np
from ase.neighborlist import neighbor_list

SPECIES = ["Fe", "Ni", "Cr"]
PAIRS = [("Cr", "Cr"), ("Cr", "Fe"), ("Cr", "Ni"),
         ("Fe", "Fe"), ("Fe", "Ni"), ("Ni", "Ni")]
PAIR_IDX = {p: k for k, p in enumerate(PAIRS)}
SHELL_EDGES = [0.0, 3.075, 3.65]      # A: 1NN (2.546) and 2NN (3.600) shells of fcc a = 3.60
N_SHELLS = len(SHELL_EDGES) - 1
NBR_CUTOFF = SHELL_EDGES[-1] + 1e-4


def reembed_to_reference_frame(raw, prim_cell, atol=1e-4):
    """Rotate ``raw`` into the Cartesian frame of the reference lattice.

    A cif stores only cell lengths and angles, so every structure is read back
    in its own orientation, while icet's ``map_structure_to_reference`` does no
    rotation search.  The cells are exact fcc lattice cells, so the integer
    supercell matrix P (rows m1, m2, m3 in the primitive basis) is found from
    the cell metric and the proper rotation M = inv(raw_cell) @ (P @ prim_cell)
    is applied to the positions.  Atom order is unchanged.
    """
    cell = raw.cell.array
    G_raw = cell @ cell.T
    if np.abs(cell @ np.linalg.inv(prim_cell)
              - np.round(cell @ np.linalg.inv(prim_cell))).max() < 1e-6:
        return raw.copy()

    nmax = int(np.ceil(np.sqrt(G_raw.diagonal().max())
                       / np.linalg.norm(prim_cell, axis=1).min())) + 1
    rng = np.arange(-nmax, nmax + 1)
    grid = np.array(np.meshgrid(rng, rng, rng)).reshape(3, -1).T
    V = grid @ prim_cell
    L2 = (V ** 2).sum(axis=1)
    cands = [grid[np.abs(L2 - G_raw[i, i]) < atol] for i in range(3)]
    for m1 in cands[0]:
        v1 = m1 @ prim_cell
        for m2 in cands[1]:
            v2 = m2 @ prim_cell
            if abs(v1 @ v2 - G_raw[0, 1]) > atol:
                continue
            for m3 in cands[2]:
                v3 = m3 @ prim_cell
                if (abs(v1 @ v3 - G_raw[0, 2]) > atol
                        or abs(v2 @ v3 - G_raw[1, 2]) > atol):
                    continue
                P = np.array([m1, m2, m3])
                if np.linalg.det(P) <= 0:
                    continue
                new_cell = P @ prim_cell
                M = np.linalg.inv(cell) @ new_cell
                assert np.allclose(M @ M.T, np.eye(3), atol=1e-6), "re-embed rotation not orthogonal"
                out = raw.copy()
                out.set_cell(new_cell, scale_atoms=False)
                out.set_positions(raw.positions @ M)
                out.wrap()
                return out
    raise ValueError("no integer fcc re-embedding found: structure is not commensurate "
                     f"with the reference lattice (G_raw diag {G_raw.diagonal()})")


def pbc_perm_raw_to_mapped(raw, mapped, tol=1e-3):
    """perm[i] = index in ``mapped`` of raw atom i (periodic position match, same species)."""
    cell = raw.cell.array
    inv_cell = np.linalg.inv(cell)
    mpos = mapped.positions
    msym = np.array(mapped.get_chemical_symbols())
    perm = np.full(len(raw), -1, dtype=int)
    used = np.zeros(len(mapped), dtype=bool)
    for i in range(len(raw)):
        df = (mpos - raw.positions[i]) @ inv_cell
        df -= np.round(df)
        d = np.linalg.norm(df @ cell, axis=1)
        d[msym != raw.symbols[i]] = np.inf
        d[used] = np.inf
        k = int(np.argmin(d))
        assert d[k] < tol, (f"raw atom {i} ({raw.symbols[i]}): no unused mapped site within "
                            f"{tol} A (min {d[k]:.4g})")
        perm[i] = k
        used[k] = True
    assert used.all()
    return perm


def spin_features(atoms, spins):
    """(N_SHELLS * len(PAIRS),) per-atom Heisenberg pair sums; each bond counted once."""
    n = len(atoms)
    i_idx, j_idx, dists = neighbor_list("ijd", atoms, NBR_CUTOFF)
    syms = np.array(atoms.get_chemical_symbols())
    x = np.zeros(N_SHELLS * len(PAIRS), dtype=np.float64)
    if len(i_idx) == 0:
        return x
    shell = np.digitize(dists, SHELL_EDGES[1:-1], right=True)
    contrib = spins[i_idx] * spins[j_idx] * 0.5
    for s in range(N_SHELLS):
        m = shell == s
        for a, b, c in zip(i_idx[m], j_idx[m], contrib[m]):
            p = PAIR_IDX[tuple(sorted((syms[a], syms[b])))]
            x[s * len(PAIRS) + p] += c
    return x / n


def pair_instances_by_feature(atoms):
    """Undirected bond instances per (shell, species pair) column, with their distances."""
    i_idx, j_idx, dists, S = neighbor_list("ijdS", atoms, NBR_CUTOFF)
    syms = np.array(atoms.get_chemical_symbols())
    shell = np.digitize(dists, SHELL_EDGES[1:-1], right=True)
    inst = {k: [] for k in range(N_SHELLS * len(PAIRS))}
    rads = {k: [] for k in range(N_SHELLS * len(PAIRS))}
    seen = set()
    for a, b, d, s, sv in zip(i_idx, j_idx, dists, shell, S):
        key = (min(a, b), max(a, b), tuple(sv if a <= b else -sv))
        if key in seen:
            continue
        seen.add(key)
        col = int(s) * len(PAIRS) + PAIR_IDX[tuple(sorted((syms[a], syms[b])))]
        inst[col].append((int(a), int(b)))
        rads[col].append(float(d))
    return inst, rads


def decoration_digest(atoms):
    """md5 of (lattice, sorted (species, fractional position)) at 4 decimals; spins excluded."""
    cell = np.round(atoms.cell.array, 4) + 0.0
    frac = np.round(atoms.get_scaled_positions() % 1.0, 4) + 0.0
    lat = ";".join(f"{v:.4f}" for v in cell.ravel())
    sites = sorted(f"{sp},{x:.4f},{y:.4f},{z:.4f}"
                   for sp, (x, y, z) in zip(atoms.get_chemical_symbols(), frac))
    return hashlib.md5((lat + "|" + ";".join(sites)).encode()).hexdigest()
