"""Turn an icet ClusterExpansion into a sparse-convolution kernel bank.

An icet orbit is exactly a convolution kernel: `orbit.clusters` lists every
symmetry-equivalent cluster as (site_index_in_primitive, unitcell_offset), all
anchored so that one member sits in cell (0,0,0).  Translating that set over
every cell of a supercell enumerates the whole orbit -- i.e. a sparse, dilated
3-D convolution with circular padding (PBC).

    kernel[a] = { clusters : [[(sub, (di,dj,dk)), ...], ...],   # equivalent orientations
                  mult     : int,                               # icet multiplicity
                  order    : int,
                  eci      : float }

The weight tying across `clusters` is the space-group equivariance; icet has
already done that grouping for us.
"""
from __future__ import annotations
import numpy as np


def extract_kernels(ce) -> dict:
    cs = ce._cluster_space
    prim = cs.primitive_structure
    params = np.asarray(ce.parameters, dtype=float)
    out = []
    cv_idx = 1
    for o_idx in range(len(cs.orbit_list)):
        orbit = cs.orbit_list.get_orbit(o_idx)
        clusters = [
            [(int(ls.index), tuple(int(v) for v in ls.unitcell_offset))
             for ls in c.lattice_sites]
            for c in orbit.clusters
        ]
        for cv_elem in orbit.cluster_vector_elements:
            out.append(dict(
                cv_idx=cv_idx,
                orbit_idx=o_idx,
                order=int(orbit.order),
                radius=float(orbit.radius),
                mult=int(cv_elem.get('multiplicity', len(clusters))),
                mc_vector=[int(v) for v in cv_elem['multicomponent_vector']],
                clusters=clusters,
                eci=float(params[cv_idx]),
            ))
            cv_idx += 1
    assert cv_idx == len(params), f"cv mismatch {cv_idx} vs {len(params)}"
    return dict(zerolet=float(params[0]),
                n_prim=len(prim),
                species=list(cs.chemical_symbols[0]),
                kernels=out)


def supercell_index(m, n_prim):
    """ase `Atoms.repeat(m)` ordering: idx = ((i0*m1 + i1)*m2 + i2)*n_prim + sub."""
    m0, m1, m2 = m
    idx = np.empty((m0, m1, m2, n_prim), dtype=np.int64)
    for i0 in range(m0):
        for i1 in range(m1):
            for i2 in range(m2):
                base = ((i0 * m1 + i1) * m2 + i2) * n_prim
                idx[i0, i1, i2, :] = base + np.arange(n_prim)
    return idx
