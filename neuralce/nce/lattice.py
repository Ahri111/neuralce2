"""Map icet primitive-cell cluster offsets onto a diagonal conventional grid.

icet gives cluster members as (sublattice_index, unitcell_offset) in units of
the PRIMITIVE cell.  `jnp.roll` needs a diagonal m0 x m1 x m2 grid, and the NiPt
data lives on a cubic 3x3x3 CONVENTIONAL fcc cell (108 sites), which is not a
diagonal repeat of the 1-atom primitive.  This module converts.

fcc primitive vectors in conventional units:
    a1 = (0, 1/2, 1/2)   a2 = (1/2, 0, 1/2)   a3 = (1/2, 1/2, 0)
so the primitive offset (n1,n2,n3) sits at conventional fractional position
    v = ( (n2+n3)/2, (n1+n3)/2, (n1+n2)/2 )
whose fractional part is one of the four fcc basis vectors (the sublattice) and
whose integer part is the conventional cell offset.
"""
from __future__ import annotations
import numpy as np

# the four fcc sites inside the conventional cell, in half-integer units
FCC_BASIS = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
_BASIS_ID = {b: i for i, b in enumerate(FCC_BASIS)}


def prim_offset_to_conv(n, sub_in=0):
    """(n1,n2,n3) primitive offset -> (conventional cell offset, sublattice id)."""
    if sub_in != 0:
        raise NotImplementedError("multi-site primitive not supported yet")
    n1, n2, n3 = n
    twice = np.array([n2 + n3, n1 + n3, n1 + n2])      # 2 * conventional fractional
    cell = np.floor_divide(twice, 2)
    half = tuple(int(t - 2 * c) for t, c in zip(twice, cell))
    if half not in _BASIS_ID:
        raise ValueError(f"offset {n} -> non-fcc residue {half}")
    return tuple(int(c) for c in cell), _BASIS_ID[half]


# the four primitive offsets that represent the cosets of the conventional
# lattice inside the fcc primitive lattice (anchor -> sublattice 0,1,2,3)
COSETS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


def convert_bank(bank):
    """Rewrite kernels into conventional (sublattice, cell_offset) form.

    Sliding over CONVENTIONAL cells only realises a quarter of the primitive
    translations (fcc has 4 primitive cells per conventional cell), so each
    icet cluster is expanded into its 4 coset copies -- anchored on each of the
    four sublattices.  Rolling those over the conventional grid then enumerates
    every primitive translation exactly once, and the normalisation picks up the
    extra factor n_sub.
    """
    out = dict(bank)
    out['n_sub'] = len(COSETS)
    out['kernels'] = []
    for k in bank['kernels']:
        kk = dict(k)
        expanded = []
        for cl in k['clusters']:
            for t in COSETS:
                shifted = [(s, tuple(d[a] + t[a] for a in range(3))) for (s, d) in cl]
                expanded.append([(sub, cell) for (cell, sub) in
                                 (prim_offset_to_conv(d, s) for (s, d) in shifted)])
        kk['clusters'] = expanded
        kk['norm'] = k['mult'] * len(COSETS)
        out['kernels'].append(kk)
    return out


def field_from_fractional(frac, occ, m=3):
    """108-site cubic cell -> occupation field (m,m,m,4).

    Fractional coordinates on a 3x3x3 conventional fcc cell are multiples of
    1/(2m); u = round(frac * 2m) gives cell = u//2 and sublattice = u%2.
    """
    u = np.rint(np.asarray(frac) * (2 * m)).astype(int) % (2 * m)
    cell = u // 2
    half = tuple(map(tuple, (u - 2 * cell)))
    field = np.full((m, m, m, 4), -99, dtype=np.int8)
    for p, h, c, o in zip(range(len(occ)), half, cell, occ):
        sub = _BASIS_ID[h]
        field[c[0], c[1], c[2], sub] = o
    if (field == -99).any():
        raise ValueError("some lattice sites were not filled")
    return field
