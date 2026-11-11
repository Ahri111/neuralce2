"""Translation-template layer-0 table: same table as gpu_layer0.build_table, built WITHOUT enumerating the
supercell's clusters.  icet's PRIMITIVE orbit list already lists every cluster instance once with its sites as
(basis index, unit-cell offset); a supercell that is an integer multiple of the primitive is exactly that template
translated to every primitive cell it contains.  So:
    template rows  (R_t, mo): basis b, offset o (3), point-function index p, column, order
    supercell rows (n_cells * R_t, mo): site index of (b, t_c + o) reduced modulo the supercell lattice
All of it is numpy integer arithmetic — O(N) memory as int32 arrays, no python objects, no per-supercell icet call
(PtNi 4,000 atoms killed the old builder at 16 GB CPU RAM; this one is ~R x mo x 12 bytes).
Balls (receptive fields) are built with scipy.sparse in local_update.build_balls_sparse.
Gate: identical cluster vector / h0 / local dE to build_table (gate_tiled_table.py).
"""
from __future__ import annotations
import numpy as np


def site_decompose(prim, atoms, tol=1e-3):
    """Every supercell site -> (basis index in prim, integer primitive-cell offset).  Also the integer supercell
    matrix T (S = T @ P) and the reduction of an offset modulo the supercell lattice."""
    P = prim.cell.array; S = atoms.cell.array; T = S @ np.linalg.inv(P)
    assert np.allclose(T, np.round(T), atol=1e-4), f"supercell is not an integer multiple of the primitive: T={T}"
    T = np.round(T).astype(np.int64); n_cells = int(round(abs(np.linalg.det(T))))
    F = atoms.positions @ np.linalg.inv(P); pb = prim.get_scaled_positions()
    basis = np.full(len(atoms), -1, np.int64); off = np.zeros((len(atoms), 3), np.int64); best = np.full(len(atoms), np.inf)
    for b in range(len(prim)):
        d = F - pb[b]; t = np.round(d); err = np.abs(d - t).max(1)
        take = err < best; basis[take] = b; off[take] = t[take]; best[take] = err[take]
    assert best.max() < tol, f"site does not sit on the primitive lattice (max err {best.max():.2e})"
    Tinv = np.linalg.inv(T)

    def reduce(t):                                   # integer offsets -> canonical representative inside the supercell
        f = t @ Tinv; return t - np.floor(f + 1e-9).astype(np.int64) @ T
    off_red = reduce(off)
    return basis, off_red, T, n_cells, reduce


def build_template(cs):
    """rows of the primitive orbit list: (basis, offset, pidx) per member, column, order, per-column count."""
    ol = cs.orbit_list; tb, to, tp, tcol, torder, cnt = [], [], [], [], [], []
    col = 0
    for oi in range(len(ol)):
        o = ol.get_orbit(oi)
        insts = [[(int(ls.index), tuple(int(x) for x in ls.unitcell_offset)) for ls in c.lattice_sites] for c in o.clusters]
        for e in o.cluster_vector_elements:
            mc = e['multicomponent_vector']; perms = e['site_permutations']; n = 0
            for inst in insts:
                for pm in perms:
                    tb.append([s[0] for s in inst]); to.append([s[1] for s in inst]); tp.append([mc[pm[j]] for j in range(len(inst))]); tcol.append(col); torder.append(len(inst)); n += 1
            cnt.append(n); col += 1
    mo = max(len(r) for r in tb); Rt = len(tb)
    B = np.zeros((Rt, mo), np.int64); O = np.zeros((Rt, mo, 3), np.int64); Pp = np.zeros((Rt, mo), np.int32); M = np.zeros((Rt, mo), bool)
    for r in range(Rt):
        k = len(tb[r]); B[r, :k] = tb[r]; O[r, :k] = to[r]; Pp[r, :k] = tp[r]; M[r, :k] = True
    return dict(basis=B, off=O, pidx=Pp, mask=M, col=np.asarray(tcol, np.int32), order=np.asarray(torder, np.int32), count=np.asarray(cnt, float), K=col, mo=mo)


def build_table_tiled(cs, atoms, template=None, active=None):
    """Same dict as gpu_layer0.build_table (idx, pidx, mask, col, order, norm, K, sig, M, theta, active, site_rows,
    site_rows_mask, N) plus 'basis'/'cell' per site — built by tiling the template."""
    from .gpu_layer0 import species_index, theta_np
    prim = cs.primitive_structure; tpl = template or build_template(cs)
    basis, off_red, T, n_cells, reduce = site_decompose(prim, atoms)
    N = len(atoms); n_p = len(prim)
    # site lookup: key(b, reduced offset) -> site.  Reduced offsets live in a bounded box; encode as int64.
    lo = off_red.min(0); span = off_red.max(0) - lo + 1
    def key(b, t): d = t - lo; return ((b * span[0] + d[..., 0]) * span[1] + d[..., 1]) * span[2] + d[..., 2]
    keys = key(basis, off_red); order_k = np.argsort(keys); ks = keys[order_k]
    assert len(np.unique(ks)) == N, "duplicate (basis, cell) keys — supercell sites are not a clean tiling"
    cells = off_red[basis == 0] if (basis == 0).any() else np.unique(off_red, axis=0); assert len(cells) == n_cells, (len(cells), n_cells)
    Rt, mo = tpl['basis'].shape
    # members of every translated row: (n_cells, Rt, mo, 3) offsets -> reduce -> lookup
    t_all = reduce((cells[:, None, None, :] + tpl['off'][None]).reshape(-1, 3)).reshape(n_cells, Rt, mo, 3)
    b_all = np.broadcast_to(tpl['basis'][None], (n_cells, Rt, mo))
    kq = key(b_all, t_all); pos = np.searchsorted(ks, kq); pos = np.clip(pos, 0, N - 1)
    hit = ks[pos] == kq; hit |= ~np.broadcast_to(tpl['mask'][None], hit.shape)          # padded members are irrelevant
    assert hit.all(), f"{(~hit).sum()} template members do not map onto supercell sites (supercell too small / wrong primitive?)"
    idx = order_k[pos].reshape(n_cells * Rt, mo).astype(np.int32)
    pidx = np.broadcast_to(tpl['pidx'][None], (n_cells, Rt, mo)).reshape(-1, mo).astype(np.int32)
    mask = np.broadcast_to(tpl['mask'][None], (n_cells, Rt, mo)).reshape(-1, mo).copy()
    col = np.broadcast_to(tpl['col'][None], (n_cells, Rt)).reshape(-1).astype(np.int32)
    order = np.broadcast_to(tpl['order'][None], (n_cells, Rt)).reshape(-1).astype(np.int32)
    idx[~mask] = 0
    norm = tpl['count'] * n_cells; K = int(tpl['K'])
    sig, Msite, sub_of, maps = species_index(cs, atoms)
    maxM = int(Msite.max()); maxP = max(int(pidx.max()) + 1, 1); theta = np.zeros((maxM + 1, maxP, maxM))
    for Mv in range(1, maxM + 1):
        for p in range(maxP):
            for s in range(Mv):
                theta[Mv, p, s] = theta_np(Mv, p, s)
    # reverse index site -> rows (CSR by sorting the (site, row) pairs), padded
    R = idx.shape[0]; rr = np.repeat(np.arange(R, dtype=np.int32), mo).reshape(R, mo)[mask]; ss = idx[mask]
    o = np.argsort(ss, kind="stable"); ss, rr = ss[o], rr[o]
    counts = np.bincount(ss, minlength=N); Rmax = int(counts.max()); starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    site_rows = np.full((N, Rmax), -1, np.int32)
    slot = np.arange(len(ss)) - np.repeat(starts, counts); site_rows[ss, slot] = rr
    if active is None:
        active = np.where(counts > 0)[0]
    return dict(idx=idx, pidx=pidx, mask=mask, col=col, order=order, norm=norm, K=K, sig=sig, M=Msite, theta=theta,
                site_rows=site_rows, site_rows_mask=site_rows >= 0, active=np.asarray(active, np.int32), N=N,
                basis=basis, cell=off_red, n_cells=n_cells, template_rows=Rt)
