"""Cell-size-independent geometry for the cluster convolution MC path.

Nothing proportional to the number of sites is built for the geometry: the primitive orbit list (template),
the shell partner lists and the receptive-field balls are all stored PER BASIS SITE as (basis, offset) pairs,
and the sites of a cluster instance / shell / ball around site i are computed on the fly from
    i = cell * n_basis + basis,   cell <-> k = U n mod D   (Smith normal form of the supercell matrix,
    CE/ce_gpu_implicit.py),   partner = site(basis', (k + U off') mod D).
Only the STATE is O(N): sigma (int32), the layer-0 cache h0 (K floats) or the projection cache, the site
energies e and two int32 slot scratches per replica.  Sites are in CANONICAL order (perm / inv_perm map to
the input atom order).  Layer-0 values, h0, ΔE identical to gpu_layer0 / gpu_layer0_tiled
(gate: gates/gate_nce_implicit.py).
"""
from __future__ import annotations
import numpy as np


def build_geometry(cs, atoms, shell_edges, depth, template=None, spin=False):
    from ase.data import atomic_numbers
    from ase.neighborlist import neighbor_list
    from .gpu_layer0_tiled import site_decompose, build_template
    from .gpu_layer0 import theta_np
    from neuralce.ce.ce_gpu_implicit import smith_normal_form, canonical_cells
    prim = cs.primitive_structure; nB = len(prim)
    basis, off_red, T, n_cells, _ = site_decompose(prim, atoms)
    N = len(atoms); assert N == nB * n_cells, (N, nB, n_cells)
    U, d, _ = smith_normal_form(T.T)                # rows of T = supercell vectors -> lattice T^T Z^3
    assert int(np.prod(d)) == n_cells
    _, cell_idx = canonical_cells(off_red, U, d)
    canon = cell_idx * nB + basis; assert len(np.unique(canon)) == N, "sites do not form a clean tiling of the primitive"
    perm = np.argsort(canon).astype(np.int64); inv_perm = np.empty(N, np.int64); inv_perm[perm] = np.arange(N)
    uo = lambda off: np.mod(np.asarray(off, np.int64) @ U.T, d).astype(np.int32)
    # sublattices / species index per basis (icet rule: rank of Z among the sublattice's species)
    sub = cs.get_sublattices(prim)
    maps_sl = [{atomic_numbers[s]: k for k, s in enumerate(sorted(sl.chemical_symbols, key=lambda x: atomic_numbers[x]))} for sl in sub]
    b_sl = np.array([sub.get_sublattice_index_from_site_index(p) for p in range(nB)], np.int32)
    Mb = np.array([len(maps_sl[s]) for s in b_sl], np.int32); active_b = np.where(Mb > 1)[0]; nA = int(len(active_b) * n_cells)
    Z = atoms.numbers; sig_orig = np.zeros(N, np.int32)
    for b in range(nB):
        m = basis == b; lut = maps_sl[b_sl[b]]; u, inv = np.unique(Z[m], return_inverse=True)
        assert all(int(z) in lut for z in u), f"basis {b}: species {u.tolist()} not in {lut}"
        sig_orig[m] = np.array([lut[int(z)] for z in u], np.int32)[inv]
    sig = sig_orig[perm]
    # ── layer-0 template ──
    tpl = template or build_template(cs); Rt, mo = tpl['basis'].shape; K = int(tpl['K'])
    msk = tpl['mask']; T_b = tpl['basis'].astype(np.int32).copy(); T_off = tpl['off'].astype(np.int64).copy()
    T_b[~msk] = np.broadcast_to(T_b[:, :1], T_b.shape)[~msk]; T_off[~msk] = np.broadcast_to(T_off[:, :1], T_off.shape)[~msk]
    maxM = int(Mb.max()); maxP = max(int(tpl['pidx'].max()) + 1, 1); theta = np.zeros((maxM + 1, maxP, maxM))
    for Mv in range(1, maxM + 1):
        for p in range(maxP):
            for s in range(Mv): theta[Mv, p, s] = theta_np(Mv, p, s)
    ent = [[] for _ in range(nB)]
    for r in range(Rt):
        for m in range(mo):
            if msk[r, m]: ent[T_b[r, m]].append((r, m))
    J = max(len(e) for e in ent)
    Rv_b = np.zeros((nB, J, mo), np.int32); Rv_uo = np.zeros((nB, J, mo, 3), np.int32); Rv_p = np.zeros((nB, J, mo), np.int32)
    Rv_m = np.zeros((nB, J, mo), bool); Rv_col = np.zeros((nB, J), np.int32); Rv_ord = np.ones((nB, J), np.float64); Rv_v = np.zeros((nB, J), bool)
    ball0 = [set() for _ in range(nB)]
    for b in range(nB):
        Rv_b[b] = b
        for j, (r, m) in enumerate(ent[b]):
            rel = T_off[r] - T_off[r, m]
            Rv_b[b, j] = T_b[r]; Rv_uo[b, j] = uo(rel); Rv_p[b, j] = tpl['pidx'][r]; Rv_m[b, j] = msk[r]
            Rv_col[b, j] = tpl['col'][r]; Rv_ord[b, j] = tpl['order'][r]; Rv_v[b, j] = True
            for mm in range(mo):
                if msk[r, mm]: ball0[b].add((int(T_b[r, mm]), tuple(int(x) for x in rel[mm])))
    # ── shells (active bases, active partners, periodic images as separate entries) ──
    P = len(shell_edges) - 1
    i, j, D, S = neighbor_list('ijdS', prim, float(shell_edges[-1]) + 1e-6)
    keep = (Mb[i] > 1) & (Mb[j] > 1) & (D > 1e-6); i, j, D, S = i[keep], j[keep], D[keep], S[keep]
    shell = np.digitize(D, shell_edges[1:-1], right=True)
    part = [[[] for _ in range(P)] for _ in range(nB)]
    for a, bb, sh, s in zip(i, j, shell, S): part[int(a)][int(sh)].append((int(bb), tuple(int(x) for x in s)))
    W = max(max(len(part[b][p]) for p in range(P)) for b in range(nB))
    Sh_b = np.zeros((nB, P, W), np.int32); Sh_off = np.zeros((nB, P, W, 3), np.int64); Sh_m = np.zeros((nB, P, W), bool)
    for b in range(nB):
        Sh_b[b] = b
        for p in range(P):
            for w, (bb, o) in enumerate(part[b][p]): Sh_b[b, p, w] = bb; Sh_off[b, p, w] = o; Sh_m[b, p, w] = True
    partners = [set((bb, o) for p in range(P) for (bb, o) in part[b][p]) for b in range(nB)]
    # ── balls (relative offsets on the infinite lattice; duplicates after reduction are handled by the slot maps) ──
    if spin:
        for b in range(nB):
            if Mb[b] > 1: ball0[b] |= {(b, (0, 0, 0))} | partners[b]
    balls = [ball0]
    for _ in range(depth):
        prev = balls[-1]; nxt = []
        for b in range(nB):
            s = set(prev[b])
            for (bb, o) in prev[b]:
                for (b2, o2) in partners[bb]: s.add((b2, (o[0] + o2[0], o[1] + o2[1], o[2] + o2[2])))
            nxt.append(s)
        balls.append(nxt)
    ball_b, ball_uo = [], []
    for bl in balls:
        w = max(len(s) for s in bl); Bb = np.zeros((nB, w), np.int32); Bo = np.zeros((nB, w, 3), np.int64)
        for b in range(nB):
            Bb[b] = b; items = sorted(bl[b])
            for q, (bb, o) in enumerate(items): Bb[b, q] = bb; Bo[b, q] = o
        ball_b.append(Bb); ball_uo.append(uo(Bo))
    # ── swap groups: active sublattices with their bases ──
    act_sl = sorted(set(int(b_sl[b]) for b in active_b)); maxb = max(int((b_sl[active_b] == sl).sum()) for sl in act_sl)
    grp_b = np.zeros((len(act_sl), maxb), np.int32); grp_nb = np.zeros(len(act_sl), np.int32)
    for g, sl in enumerate(act_sl):
        bs = [int(b) for b in active_b if b_sl[b] == sl]; grp_b[g, :len(bs)] = bs; grp_nb[g] = len(bs)
    import jax.numpy as jnp
    A = lambda x: jnp.asarray(x)
    G = dict(nB=jnp.int32(nB), d=A(d.astype(np.int32)), Mb=A(Mb), theta=A(theta), norm=A(tpl['count'] * n_cells),
             T_b=A(T_b), T_uo=A(uo(T_off)), T_p=A(tpl['pidx'].astype(np.int32)), T_m=A(msk), T_col=A(tpl['col'].astype(np.int32)), T_ord=A(tpl['order'].astype(np.float64)),
             Rv_b=A(Rv_b), Rv_uo=A(Rv_uo), Rv_p=A(Rv_p), Rv_m=A(Rv_m), Rv_col=A(Rv_col), Rv_ord=A(Rv_ord), Rv_v=A(Rv_v),
             Sh_b=A(Sh_b), Sh_uo=A(uo(Sh_off)), Sh_m=A(Sh_m), ball_b=[A(x) for x in ball_b], ball_uo=[A(x) for x in ball_uo],
             grp_b=A(grp_b), grp_nb=A(grp_nb))
    stat = dict(N=N, nA=nA, nB=nB, n_cells=n_cells, K=K, mo=mo, Rt=Rt, J=J, P=P, W=W, depth=depth,
                active_b=[int(b) for b in active_b], ball_w=[x.shape[1] for x in ball_b], block=int(max(64, min(n_cells, 2_000_000 // max(Rt * mo, 1)))))
    stat['n_blocks'] = -(-n_cells // stat['block'])
    return dict(G=G, stat=stat, sig=sig, perm=perm, inv_perm=inv_perm, basis=basis, active_b=active_b, Mb=Mb, b_sl=b_sl,
                maps_sl=maps_sl, U=U, d=d, T=T, n_cells=n_cells, nB=nB, N=N, nA=nA, K=K, template=tpl,
                active_orig=np.where(Mb[basis] > 1)[0])


# ── jax geometry helpers (G = jit argument) ─────────────────────────────────────────────────────────
def k_of(G, i):
    import jax.numpy as jnp
    nB = G['nB']; d = G['d']; b = i % nB; c = i // nB
    return b, jnp.stack([c // (d[1] * d[2]), (c // d[2]) % d[1], c % d[2]], axis=-1)


def site_of(G, b, k):
    d = G['d']; return ((k[..., 0] * d[1] + k[..., 1]) * d[2] + k[..., 2]) * G['nB'] + b


def ball_sites(G, l, i):
    """sites of ball_l(i) (w_l,), duplicates possible in small cells (idempotent for the slot maps)."""
    b, k = k_of(G, i)
    return site_of(G, G['ball_b'][l][b], (k[None, :] + G['ball_uo'][l][b]) % G['d'])


def shells_of(G, sites):
    """(P, B, W) partner sites and (P, B, W) float mask for `sites` (B,)."""
    import jax.numpy as jnp
    b, k = k_of(G, sites)                                                       # (B,), (B,3)
    si = site_of(G, G['Sh_b'][b], (k[:, None, None, :] + G['Sh_uo'][b]) % G['d'])   # (B, P, W)
    return jnp.transpose(si, (1, 0, 2)), jnp.transpose(G['Sh_m'][b], (1, 0, 2))


def active_mask(G, N):
    import jax.numpy as jnp
    return G['Mb'][jnp.arange(N, dtype=jnp.int32) % G['nB']] > 1


def h0_full(G, sig, stat):
    """(N, K+1) site-resolved layer 0 in canonical order (inactive rows 0); lax.scan over cell blocks."""
    import jax, jax.numpy as jnp
    N, nA, K, block, n_blocks = stat['N'], stat['nA'], stat['K'], stat['block'], stat['n_blocks']
    d = G['d']; nB = G['nB']; n_cells = jnp.int32(stat['n_cells'])
    Mrow = G['Mb'][G['T_b']]                                                   # (Rt, mo)

    def body(h, bi):
        c = bi * block + jnp.arange(block, dtype=jnp.int32)
        k = jnp.stack([c // (d[1] * d[2]), (c // d[2]) % d[1], c % d[2]], axis=-1)              # (B, 3)
        sites = site_of(G, G['T_b'][None], (k[:, None, None, :] + G['T_uo'][None]) % d)          # (B, Rt, mo)
        sites = jnp.minimum(sites, N - 1)
        g = jnp.where(G['T_m'][None], G['theta'][Mrow[None], G['T_p'][None], sig[sites]], 1.0)
        pr = jnp.prod(g, axis=-1) / G['T_ord'][None] / G['norm'][G['T_col']][None] * nA          # (B, Rt)
        pr = jnp.where((c < n_cells)[:, None], pr, 0.0)
        for m in range(sites.shape[-1]):
            h = h.at[sites[:, :, m], G['T_col'][None]].add(jnp.where(G['T_m'][None, :, m], pr, 0.0))
        return h, None
    h, _ = jax.lax.scan(body, jnp.zeros((N, K), G['theta'].dtype), jnp.arange(n_blocks, dtype=jnp.int32))
    act = active_mask(G, N)
    return jnp.concatenate([act[:, None].astype(h.dtype), h], axis=1)


def rows_of(G, i):
    """cluster instances containing site i: member sites (J, mo), pidx, mask, col, order, valid."""
    b, k = k_of(G, i)
    sites = site_of(G, G['Rv_b'][b], (k[None, None, :] + G['Rv_uo'][b]) % G['d'])
    return sites, G['Rv_p'][b], G['Rv_m'][b], G['Rv_col'][b], G['Rv_ord'][b], G['Rv_v'][b]


def _species_at(sig, idx, ov):
    import jax.numpy as jnp
    sp = sig[idx]
    for site, species in (ov or ()):
        sp = jnp.where(idx == site, species.astype(sp.dtype), sp)
    return sp


def delta_h0(G, sig, i, s_new, stat, ov=None):
    """-> (flat (J*mo,), valid, value, chan): flat = canonical site * (K+1) + channel; scatter-add `value`."""
    import jax.numpy as jnp
    K, nA = stat['K'], stat['nA']
    sites, pidx, m, col, order, valid = rows_of(G, i); Mv = G['Mb'][G['Rv_b'][i % G['nB']]]
    def prod(ovx):
        g = jnp.where(m, G['theta'][Mv, pidx, _species_at(sig, sites, ovx)], 1.0); return jnp.prod(g, axis=-1)
    dpi = prod(tuple(ov or ()) + ((i, s_new),)) - prod(ov)
    w = jnp.where(valid, dpi / order / G['norm'][col] * nA, 0.0)                  # (J,)
    ok = m & valid[:, None]
    chan = (col + 1)[:, None] * jnp.ones_like(sites)
    flat = jnp.where(ok, sites * (K + 1) + chan, 0)
    value = jnp.where(ok, w[:, None] * jnp.ones_like(sites, dtype=w.dtype), 0.0)
    return flat.reshape(-1), ok.reshape(-1), value.reshape(-1), jnp.where(ok, chan, 0).reshape(-1)


def make_propose(G, stat):
    """O(1) table-free swap proposal: sublattice group ∝ size, uniform cell + uniform basis of the group, twice."""
    import jax, jax.numpy as jnp
    nB = stat['nB']; n_cells = stat['n_cells']; ng = int(G['grp_nb'].shape[0]); p = np.asarray(G['grp_nb'], float); p = jnp.asarray(p / p.sum())

    def draw(key, g):
        k1, k2 = jax.random.split(key)
        c = jax.random.randint(k1, (), 0, n_cells); b = G['grp_b'][g, jax.random.randint(k2, (), 0, G['grp_nb'][g])]
        return c * nB + b

    def propose(key):
        k0, k1, k2 = jax.random.split(key, 3)
        g = jax.random.choice(k0, ng, p=p)
        return draw(k1, g), draw(k2, g)

    act_b = jnp.asarray(np.asarray(stat['active_b'], np.int32)); n_act_b = int(act_b.shape[0])

    def site_uniform(key):                                     # uniform over the ACTIVE sites (every basis has n_cells sites)
        k1, k2 = jax.random.split(key)
        return jax.random.randint(k1, (), 0, n_cells) * nB + act_b[jax.random.randint(k2, (), 0, n_act_b)]
    return propose, site_uniform
