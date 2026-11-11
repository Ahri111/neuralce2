"""Occupation fields + cluster-conv site features for the NiPt 3x3x3 conventional cell."""
from __future__ import annotations
import os, numpy as np
from ase.io import read
from ase.build import bulk
from .lattice import FCC_BASIS, convert_bank
from .kernels import extract_kernels

from neuralce.paths import DATA
CIF_DIR = os.environ.get("NIPT_CIF_DIR", os.path.join(DATA, "nipt", "cifs"))
_BASIS_ID = {b: i for i, b in enumerate(FCC_BASIS)}


def field_from_atoms(atoms, m=3, species=('Ni', 'Pt'), tol=0.25):
    """Snap a relaxed cell onto the ideal 3x3x3 conventional fcc grid -> (m,m,m,4)."""
    f = atoms.get_scaled_positions() * (2 * m)
    u = np.rint(f)
    dev = np.abs(f - u).max()
    if dev > tol:
        raise ValueError(f"site {dev:.3f} (of half-spacing) off the ideal lattice")
    u = u.astype(int) % (2 * m)
    cell, half = u // 2, u % 2
    sym = atoms.get_chemical_symbols()
    field = np.full((m, m, m, 4), -1, dtype=np.int8)
    for p in range(len(atoms)):
        sub = _BASIS_ID[tuple(half[p])]
        c = cell[p]
        if field[c[0], c[1], c[2], sub] != -1:
            raise ValueError("two atoms snapped onto the same lattice site")
        field[c[0], c[1], c[2], sub] = species.index(sym[p])
    if (field == -1).any():
        raise ValueError("unfilled lattice sites")
    return field, dev


def atoms_from_field(field, a=3.924, species=('Ni', 'Pt')):
    """Inverse: ideal Atoms, for cross-checking against icet."""
    m = field.shape[0]
    conv = bulk(species[0], 'fcc', a=a, cubic=True).repeat((m, m, m))
    pos = conv.get_scaled_positions() * (2 * m)
    u = np.rint(pos).astype(int) % (2 * m)
    cell, half = u // 2, u % 2
    sym = []
    for p in range(len(conv)):
        sub = _BASIS_ID[tuple(half[p])]
        c = cell[p]
        sym.append(species[int(field[c[0], c[1], c[2], sub])])
    conv.set_chemical_symbols(sym)
    return conv


def load_fields(ids, cif_dir=CIF_DIR, m=3):
    fields, devs = [], []
    for i in ids:
        f, d = field_from_atoms(read(f"{cif_dir}/{i}.cif"), m=m)
        fields.append(f); devs.append(d)
    return np.asarray(fields), float(np.max(devs))


def build_bank(a=3.924, cutoffs=(10.0, 8.0, 6.0, 5.0), species=('Ni', 'Pt')):
    """Kernel bank on the IDEAL lattice (never primitive-detect a decorated cell)."""
    from icet import ClusterSpace, ClusterExpansion
    prim = bulk(species[0], 'fcc', a=a)
    cs = ClusterSpace(prim, list(cutoffs), list(species))
    ce = ClusterExpansion(cs, np.zeros(len(cs)))
    return cs, convert_bank(extract_kernels(ce))


def site_features(field, bank):
    """(m,m,m,4) occupation -> (m,m,m,K) per-site cluster-match features."""
    import jax.numpy as jnp
    from .conv import cluster_conv, theta_binary
    x = jnp.asarray(theta_binary(np.asarray(field, dtype=np.float64)))
    return jnp.stack([cluster_conv(x, k['clusters']) / k['norm']
                      for k in bank['kernels']], axis=-1)


# ── fast path: precomputed site-index tensors instead of per-cluster rolls ──
def build_index(bank, m=3):
    """For every kernel, site indices of all its cluster copies at every anchor cell.

    idx[k] : (n_cells, n_copies, order) int32 into the flattened (m,m,m,4) field.
    Identical maths to `cluster_conv`, ~100x cheaper (one gather per kernel).
    """
    cells = np.array([(i, j, l) for i in range(m) for j in range(m) for l in range(m)])

    def flat(cell, sub):
        c = np.asarray(cell) % m
        return ((c[..., 0] * m + c[..., 1]) * m + c[..., 2]) * 4 + sub

    idx = []
    for k in bank['kernels']:
        cl = k['clusters']
        order = len(cl[0])
        off = np.array([[d for (s, d) in c] for c in cl])          # (n_copies, order, 3)
        sub = np.array([[s for (s, d) in c] for c in cl])          # (n_copies, order)
        a = cells[:, None, None, :] + off[None]                    # (n_cells, cp, ord, 3)
        idx.append(flat(a, sub[None]).astype(np.int32))
    return idx


def site_features_fast(field, index, bank):
    """(m,m,m,4) occupation -> (n_cells, K) per-anchor-cell features."""
    from .conv import theta_binary
    sig = theta_binary(np.asarray(field, dtype=np.float64)).reshape(-1)
    return np.stack([sig[ix].prod(axis=2).sum(axis=1) / k['norm']
                     for ix, k in zip(index, bank['kernels'])], axis=-1)


# ── GPU path: one padded index tensor, one jitted gather+product+segment-sum ──
def build_flat_index(bank, m=3):
    """Pack every kernel's cluster copies into one tensor so the whole feature
    map is a single jitted op (no Python loop over kernels -> GPU friendly).

    returns idx (n_cells, n_inst, max_order) int32, mask (n_inst, max_order),
            kid (n_inst,) int32, norm (K,) float64
    """
    per_kernel = build_index(bank, m=m)                 # list of (n_cells, cp, ord)
    n_cells = per_kernel[0].shape[0]
    max_order = max(a.shape[2] for a in per_kernel)
    n_inst = sum(a.shape[1] for a in per_kernel)
    idx = np.zeros((n_cells, n_inst, max_order), dtype=np.int32)
    mask = np.zeros((n_inst, max_order), dtype=bool)
    kid = np.zeros(n_inst, dtype=np.int32)
    p = 0
    for k, a in enumerate(per_kernel):
        cp, order = a.shape[1], a.shape[2]
        idx[:, p:p + cp, :order] = a
        mask[p:p + cp, :order] = True
        kid[p:p + cp] = k
        p += cp
    norm = np.array([kk['norm'] for kk in bank['kernels']], dtype=np.float64)
    return idx, mask, kid, norm


def make_site_feature_fn(bank, m=3):
    """-> jitted f(field_occupancy (…,m,m,m,4)) = site features (…, n_cells, K)."""
    import jax, jax.numpy as jnp
    idx, mask, kid, norm = build_flat_index(bank, m=m)
    K = len(bank['kernels'])
    idx_j = jnp.asarray(idx); mask_j = jnp.asarray(mask)
    kid_j = jnp.asarray(kid); norm_j = jnp.asarray(norm)

    @jax.jit
    def f(occ):
        sig = 2.0 * occ.reshape(occ.shape[:-4] + (-1,)) - 1.0     # theta, flattened
        g = jnp.take(sig, idx_j, axis=-1)                          # (…, cells, inst, ord)
        g = jnp.where(mask_j, g, 1.0)                              # padded slots -> x1
        prod = jnp.prod(g, axis=-1)                                # (…, cells, inst)
        prod = jnp.moveaxis(prod, -1, 0)                           # segment axis first
        h = jax.ops.segment_sum(prod, kid_j, num_segments=K)       # (K, …, cells)
        return jnp.moveaxis(h, 0, -1) / norm_j                     # (…, cells, K)
    return f
