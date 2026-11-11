"""General GPU layer-0 for the cluster convolution: ANY icet cluster space, ANY supercell.

The supercell's complete instance list comes from icet itself
(`_icet.LocalOrbitListGenerator(...).generate_full_orbit_list()`), so periodic images,
multiple sublattices, inert species and multicomponent decorations are all icet's
business, not ours.  Every (column k, instance, symmetric permutation) becomes one ROW:

    row r :  sites  idx[r, j]      point-function index  pidx[r, j]      column col[r]
    value :  pi_r = prod_j theta_{pidx[r,j]}( sigma[idx[r,j]] ; M of that sublattice )
    Phi_k = (1/norm_k) * sum_{r: col[r]=k} pi_r          (norm_k = number of rows of k)
    h0[i, k] = N_active * (1/norm_k) * sum_{r: col[r]=k, i in r} pi_r / order_r

Gates (proto, 2026-09-04): max|Phi - icet| = 0.0 on STFO 160-atom (2 active sublattices +
inert Sr), 1e-16 on FeNiCr ternary incl. 1- and 4-atom self-interacting cells.

Species index rule (icet): rank of the atomic number among the sublattice's allowed
species.  `cs.species_maps` is indexed by PRIMITIVE site, not by sublattice -- using it
per sublattice silently maps STFO to the Sr map (everything 0) -- so the map is rebuilt
here from `cs.get_sublattices(atoms)`.

Point functions (verified M=2,3,4):  theta_p(s) = -cos(2 pi k s / M) (p even),
-sin(2 pi k s / M) (p odd), k = (p+2)//2.

Delta:  `site_rows` is the reverse index (site -> rows containing it); flipping site i
changes only those rows, so d(Phi) and d(h0) are sums over |site_rows[i]| rows and h0
changes only on the member sites of those rows (the "ball0").
"""
from __future__ import annotations
import numpy as np


def theta_np(M, p, s):
    k = (p + 2) // 2
    x = 2 * np.pi * k * s / M
    return np.where(p % 2 == 0, -np.cos(x), -np.sin(x))


def species_index(cs, atoms):
    """(sig, M) per site: icet species index and number of allowed species of its sublattice."""
    from ase.data import atomic_numbers
    sub = cs.get_sublattices(atoms); Z = atoms.numbers; N = len(atoms)
    maps = [{atomic_numbers[sym]: k for k, sym in enumerate(sorted(sl.chemical_symbols, key=lambda x: atomic_numbers[x]))}
            for sl in sub]
    sig = np.zeros(N, np.int32); Msite = np.ones(N, np.int32); sub_of = np.zeros(N, np.int32)
    for i in range(N):
        si = sub.get_sublattice_index_from_site_index(i); m = maps[si]
        sub_of[i] = si; Msite[i] = len(m); sig[i] = m[Z[i]]
    return sig, Msite, sub_of, maps


def build_table(cs, atoms, active=None):
    """Instance table of `cs` on the supercell `atoms`.

    returns dict: idx (R,mo) int32, pidx (R,mo) int32, mask (R,mo) bool, col (R,) int32,
                  order (R,) int32, norm (K,) float64 [K excludes the zerolet],
                  sig (N,), M (N,), theta (n_M+1, max_p, max_M) lookup, active (n_active,) ,
                  site_rows (N, Rmax) int32 + site_rows_mask, order of the whole table."""
    import _icet
    from icet.core.structure import Structure
    full = _icet.LocalOrbitListGenerator(cs.orbit_list, Structure.from_atoms(atoms),
                                         cs.fractional_position_tolerance).generate_full_orbit_list()
    sig, Msite, sub_of, maps = species_index(cs, atoms)
    rows_idx, rows_p, rows_col, norm = [], [], [], []
    col = 0
    for oi in range(len(full)):
        o = full.get_orbit(oi)
        insts = [[ls.index for ls in c.lattice_sites] for c in o.clusters]
        for e in o.cluster_vector_elements:
            mc = e['multicomponent_vector']; perms = e['site_permutations']; cnt = 0
            for inst in insts:
                for pm in perms:
                    rows_idx.append(inst); rows_p.append([mc[pm[j]] for j in range(len(inst))]); rows_col.append(col); cnt += 1
            norm.append(cnt); col += 1
    K = col; mo = max(len(r) for r in rows_idx); R = len(rows_idx)
    idx = np.zeros((R, mo), np.int32); pidx = np.zeros((R, mo), np.int32); mask = np.zeros((R, mo), bool)
    order = np.zeros(R, np.int32)
    for r, (ii, pp) in enumerate(zip(rows_idx, rows_p)):
        idx[r, :len(ii)] = ii; pidx[r, :len(ii)] = pp; mask[r, :len(ii)] = True; order[r] = len(ii)
    # theta lookup table indexed [M, p, s]
    maxM = int(Msite.max()); maxP = max(int(pidx.max()) + 1, 1)
    theta = np.zeros((maxM + 1, maxP, maxM))
    for M in range(1, maxM + 1):
        for p in range(maxP):
            for s in range(M):
                theta[M, p, s] = theta_np(M, p, s)
    # reverse index: site -> rows containing it
    N = len(atoms)
    lists = [[] for _ in range(N)]
    for r in range(R):
        for j in range(order[r]):
            lists[idx[r, j]].append(r)
    Rmax = max(len(l) for l in lists)
    site_rows = np.full((N, Rmax), -1, np.int32)
    for i, l in enumerate(lists):
        site_rows[i, :len(l)] = l
    if active is None:
        active = np.where(np.array([len(l) > 0 for l in lists]))[0]      # sites that belong to some cluster
    return dict(idx=idx, pidx=pidx, mask=mask, col=np.asarray(rows_col, np.int32), order=order,
                norm=np.asarray(norm, float), K=K, sig=sig, M=Msite, theta=theta,
                site_rows=site_rows, site_rows_mask=site_rows >= 0, active=np.asarray(active, np.int32), N=N)


# ── numpy reference ──────────────────────────────────────────────────────────
def cv_np(T, sig):
    g = np.where(T['mask'], T['theta'][T['M'][T['idx']], T['pidx'], sig[T['idx']]], 1.0)
    Z = np.bincount(T['col'], weights=g.prod(1), minlength=T['K'])
    return np.concatenate([[1.0], Z / T['norm']])


def h0_np(T, sig):
    """site-resolved layer 0 on the ACTIVE sites: mean over active sites == cv"""
    g = np.where(T['mask'], T['theta'][T['M'][T['idx']], T['pidx'], sig[T['idx']]], 1.0)
    pr = g.prod(1) / T['order'] / T['norm'][T['col']] * len(T['active'])
    h = np.zeros((T['N'], T['K']))
    for j in range(T['idx'].shape[1]):
        m = T['mask'][:, j]
        np.add.at(h, (T['idx'][m, j], T['col'][m]), pr[m])
    out = np.ones((len(T['active']), T['K'] + 1)); out[:, 1:] = h[T['active']]
    return out


# ── jax ──────────────────────────────────────────────────────────────────────
def layer0_arrays(T):
    """the table as a pytree of device arrays — passed as an ARGUMENT to the jitted functions (never closed over:
    a closed-over 10M-row table becomes an XLA constant -> minutes of compile time and duplicated memory)."""
    import jax.numpy as jnp
    N = int(T['N']); nA = len(T['active']); inv_active = np.full(N, -1, np.int32); inv_active[T['active']] = np.arange(nA)
    return dict(idx=jnp.asarray(T['idx']), pidx=jnp.asarray(T['pidx']), mask=jnp.asarray(T['mask']), col=jnp.asarray(T['col']),
                order=jnp.asarray(T['order'], jnp.float64), norm=jnp.asarray(T['norm']), M=jnp.asarray(T['M']), theta=jnp.asarray(T['theta']),
                active=jnp.asarray(T['active']), inv_active=jnp.asarray(inv_active), site_rows=jnp.asarray(T['site_rows']), srm=jnp.asarray(T['site_rows_mask']))


def _species_at(sig, idx, ov):
    """species on the member sites `idx`, with the pending flips `ov` = ((site, species), ...) applied WITHOUT
    materialising a modified copy of sig (a (N,) copy per flip per replica was the last O(N) per-step cost)."""
    import jax.numpy as jnp
    sp = sig[idx]
    for site, species in (ov or ()):
        sp = jnp.where(idx == site, species.astype(sp.dtype), sp)
    return sp


def _row_products(A, sig, rows, ov=None):               # rows: (R',) -> (R',)
    import jax.numpy as jnp
    idx = A['idx'][rows]
    g = A['theta'][A['M'][idx], A['pidx'][rows], _species_at(sig, idx, ov)]
    g = jnp.where(A['mask'][rows], g, 1.0)
    return jnp.prod(g, axis=-1)


def cv_a(A, sig, K):
    import jax, jax.numpy as jnp
    pr = _row_products(A, sig, jnp.arange(A['idx'].shape[0]))
    Z = jax.ops.segment_sum(pr, A['col'], num_segments=K)
    return jnp.concatenate([jnp.ones(1, Z.dtype), Z / A['norm']])


def h0_a(A, sig, K, N, nA):
    """(n_active, K+1) site-resolved layer 0; static K, N, nA (python ints)."""
    import jax.numpy as jnp
    idx = A['idx']; pr = _row_products(A, sig, jnp.arange(idx.shape[0])) / A['order'] / A['norm'][A['col']] * nA
    h = jnp.zeros((N, K), pr.dtype)
    for j in range(idx.shape[1]):
        h = h.at[idx[:, j], A['col']].add(jnp.where(A['mask'][:, j], pr, 0.0))
    return jnp.concatenate([jnp.ones((nA, 1), pr.dtype), h[A['active']]], axis=1)


def delta_rows_a(A, sig, i, s_new, ov=None):
    """rows touched by flipping site i (with earlier flips `ov` already in force) -> (rows, mask, d pi)."""
    import jax.numpy as jnp
    rows = jnp.where(A['srm'][i], A['site_rows'][i], 0); rm = A['srm'][i]
    old = _row_products(A, sig, rows, ov); new = _row_products(A, sig, rows, tuple(ov or ()) + ((i, s_new),))
    return rows, rm, jnp.where(rm, new - old, 0.0)


def delta_cv_a(A, sig, i, s_new, K, ov=None):
    import jax, jax.numpy as jnp
    rows, rm, dpi = delta_rows_a(A, sig, i, s_new, ov)
    dZ = jax.ops.segment_sum(dpi, A['col'][rows], num_segments=K)
    return jnp.concatenate([jnp.zeros(1, dZ.dtype), dZ / A['norm']])


def delta_h0_a(A, sig, i, s_new, K, nA, ov=None):
    """-> (flat (Rmax*mo,), valid, value, channel): flat = active site * (K+1) + channel; scatter-add `value`."""
    import jax.numpy as jnp
    rows, rm, dpi = delta_rows_a(A, sig, i, s_new, ov)
    w = dpi / A['order'][rows] / A['norm'][A['col'][rows]] * nA
    sites = A['inv_active'][A['idx'][rows]]
    valid = A['mask'][rows] & rm[:, None] & (sites >= 0)
    chan = (A['col'][rows] + 1)[:, None] * jnp.ones_like(sites)
    flat = jnp.where(valid, sites * (K + 1) + chan, 0)
    value = jnp.where(valid, w[:, None] * jnp.ones_like(sites, dtype=w.dtype), 0.0)
    return flat.reshape(-1), valid.reshape(-1), value.reshape(-1), jnp.where(valid, chan, 0).reshape(-1)


def make_jax_fns(T):
    """L0 dict: closures cv/h0/delta_cv/delta_h0 (gates, benches: table closed over, fine for small cells) plus
    the argument-passing versions (h0_a, delta_h0_a, cv_a, delta_cv_a; static K/N/nA in 'K','N','n_active') and
    'A' = layer0_arrays(T).  The MC path (local_update.make_local_fns_v2) takes A through `tab` as a jit argument."""
    import jax
    A = layer0_arrays(T); K = int(T['K']); N = int(T['N']); nA = len(T['active'])
    cv = jax.jit(lambda sig: cv_a(A, sig, K)); h0 = jax.jit(lambda sig: h0_a(A, sig, K, N, nA))
    delta_cv = jax.jit(lambda sig, i, s_new: delta_cv_a(A, sig, i, s_new, K)); delta_h0 = jax.jit(lambda sig, i, s_new: delta_h0_a(A, sig, i, s_new, K, nA))
    return dict(cv=cv, h0=h0, delta_cv=delta_cv, delta_h0=delta_h0, n_active=nA, K=K + 1, N=N, A=A,
                h0_a=lambda A_, sig: h0_a(A_, sig, K, N, nA), delta_h0_a=lambda A_, sig, i, s_new, ov=None: delta_h0_a(A_, sig, i, s_new, K, nA, ov),
                cv_a=lambda A_, sig: cv_a(A_, sig, K), delta_cv_a=lambda A_, sig, i, s_new: delta_cv_a(A_, sig, i, s_new, K))
