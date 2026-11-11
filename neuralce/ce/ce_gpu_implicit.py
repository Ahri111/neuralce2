"""Cell-size-independent GPU memory for the orthonormal (icet) cluster expansion.

Nothing proportional to the number of atoms is stored on the GPU except the state itself (sigma, one int
per site per replica).  The cluster table is NOT tiled out: the PRIMITIVE orbit list (template, a few
hundred rows) is kept, and the member sites of every cluster instance are computed on the fly from
integer arithmetic on (basis index, primitive-cell coordinate).

Sites are stored in CANONICAL order  i = cell * n_basis + basis,  cell = (k0 * d1 + k1) * d2 + k2, where
k = U n mod D is the Smith-normal-form coordinate of the primitive-cell offset n (U T V = D = diag(d),
T = integer supercell matrix).  Any integer supercell (diagonal or not) is a clean group Z^3 / T Z^3 in
these coordinates, so "site + template offset" is   k' = (k + U off) mod D   — three adds, three mods.

Template tables (all small, jit ARGUMENTS):
    rows      T_b (Rt,mo) member basis, T_uo (Rt,mo,3) U*offset mod D, T_bf / T_st point-function and
              sublattice index, T_eci (Rt,) eci / n_instances(supercell)      -> energy_a (lax.map over cell blocks)
    reverse   per basis b: the (row, slot) pairs that contain a site of basis b, written relative to that
              site: Rv_b / Rv_uo / Rv_bf / Rv_st (nB, J, mo[,3]), Rv_eci (nB, J)   -> delta_energy_a
Padded members point at the anchor site with point-function index 0 (pf == 1), padded reverse entries
carry eci 0, so no masks are needed in the kernels.
Energy convention, point functions, eci normalisation: identical to ce_gpu_tiled / classical_ce_gpu
(gate: CE/gate_ce_implicit.py, energies equal to the tiled tables to float32 round-off).
"""
from __future__ import annotations
import numpy as np


# ── integer lattice algebra ───────────────────────────────────────────────────────────────────────
def smith_normal_form(T):
    """U, d, V with U @ T @ V = diag(d), U and V unimodular (integer, det ±1)."""
    import sympy
    from sympy.matrices.normalforms import smith_normal_decomp
    M = sympy.Matrix(np.asarray(T, np.int64).tolist())
    S, U, V = smith_normal_decomp(M, domain=sympy.ZZ)
    U = np.array(U.tolist(), np.int64); V = np.array(V.tolist(), np.int64); S = np.array(S.tolist(), np.int64)
    assert np.array_equal(U @ np.asarray(T, np.int64) @ V, S), "SNF decomposition does not reproduce T"
    d = np.abs(np.diag(S)); assert (d > 0).all() and abs(round(np.linalg.det(U))) == 1 and abs(round(np.linalg.det(V))) == 1
    if (np.diag(S) < 0).any():                       # make the diagonal positive (flip rows of U)
        U = U * np.sign(np.diag(S))[:, None]
    return U, d, V


def canonical_cells(off, U, d):
    """integer offsets (..., 3) -> SNF coordinates k = U n mod d in [0, d) and cell index.  U from smith_normal_form(T.T)
    (site_decompose convention: rows of T are the supercell vectors), so that n -> k is a bijection Z^3 / T^T Z^3 -> prod Z_d."""
    k = np.mod(np.asarray(off, np.int64) @ U.T, d)
    return k, (k[..., 0] * d[1] + k[..., 1]) * d[2] + k[..., 2]


# ── setup (numpy; only O(N) work is the per-site basis/cell decomposition, int arrays) ───────────
def build_ce_gpu_implicit(ce, supercell, block=None, energy_scale=1.0):
    """Returns dict(tp=<jit pytree>, n_atoms, n_cells, n_basis, perm, inv_perm, site_to_sl (canonical order),
    basis_to_sl, sl_species, zerolet, n_instances, n_rows_template)."""
    import jax.numpy as jnp
    from .classical_ce_gpu import extract_point_functions
    from neuralce.nce.gpu_layer0_tiled import site_decompose
    cs = ce._cluster_space; pol = cs._orbit_list; prim = cs.primitive_structure; nB = len(prim)
    # energy_scale: the CE value (zerolet + Σ eci <Π>) is the TOTAL energy of the cell the .ce was scaled for; a cell
    # tiled n_reps times from it needs x n_reps (E ∝ N).  Without it every N > N_ce ran at an effectively hotter T.
    eci_full = np.asarray(ce.parameters, float) * float(energy_scale); zerolet = float(eci_full[0])
    pf_table, n_subs = extract_point_functions(cs)
    prim_subs = cs.get_sublattices(prim)
    basis_to_sl = np.array([prim_subs.get_sublattice_index_from_site_index(p) for p in range(nB)], np.int32)
    chem = cs.chemical_symbols; sl_species = {}
    for p in range(nB):
        sl_species.setdefault(int(basis_to_sl[p]), list(chem[p]))
    basis, off_red, T, n_cells, _ = site_decompose(prim, supercell)
    N = len(supercell); assert N == nB * n_cells, (N, nB, n_cells)
    U, d, _ = smith_normal_form(T.T)                # rows of T = supercell vectors -> lattice T^T Z^3
    assert int(np.prod(d)) == n_cells
    _, cell_idx = canonical_cells(off_red, U, d)
    canon = cell_idx * nB + basis                                      # canonical index of every original site
    assert len(np.unique(canon)) == N, "sites do not form a clean tiling of the primitive"
    perm = np.argsort(canon).astype(np.int64)                          # sigma_canon = sigma_orig[perm]
    inv_perm = np.empty(N, np.int64); inv_perm[perm] = np.arange(N)    # sigma_orig = sigma_canon[inv_perm]
    # template rows (as in ce_gpu_tiled): members (basis, offset), point-function index, eci / n_inst(supercell)
    tb, to, tbf, teci = [], [], [], []
    eci_idx = 1
    for oi in range(len(pol)):
        o = pol.get_orbit(oi)
        insts = [[(int(ls.index), tuple(int(x) for x in ls.unitcell_offset)) for ls in c.lattice_sites] for c in o.clusters]
        n_inst_sc = len(insts) * n_cells
        for e in o.cluster_vector_elements:
            mc = e['multicomponent_vector']; ev = float(eci_full[eci_idx]) / n_inst_sc
            for inst in insts:
                tb.append([s[0] for s in inst]); to.append([s[1] for s in inst]); tbf.append([mc[k] + 1 for k in range(len(inst))]); teci.append(ev)
            eci_idx += 1
    Rt = len(tb); mo = max(len(r) for r in tb)
    T_b = np.zeros((Rt, mo), np.int32); T_off = np.zeros((Rt, mo, 3), np.int64); T_bf = np.zeros((Rt, mo), np.int32); T_msk = np.zeros((Rt, mo), bool)
    for r in range(Rt):
        n = len(tb[r]); T_b[r, :n] = tb[r]; T_off[r, :n] = to[r]; T_bf[r, :n] = tbf[r]; T_msk[r, :n] = True
    # padded members: anchor basis of the row's first member, offset 0, bf 0 -> pf 1
    T_b[~T_msk] = np.broadcast_to(T_b[:, :1], (Rt, mo))[~T_msk]; T_off[~T_msk] = T_off[:, 0][np.where(~T_msk)[0]]
    T_uo = np.mod(T_off @ U.T, d).astype(np.int32)                     # U*offset mod D  (rows anchored at cell 0)
    T_st = basis_to_sl[T_b].astype(np.int32); T_eci = np.asarray(teci, np.float32)
    # reverse template: for basis b, every (row, slot) with T_b[row, slot] == b, members relative to that slot
    ent = [[] for _ in range(nB)]
    for r in range(Rt):
        for m in range(mo):
            if T_msk[r, m]: ent[T_b[r, m]].append((r, m))
    J = max(len(e) for e in ent)
    Rv_b = np.zeros((nB, J, mo), np.int32); Rv_uo = np.zeros((nB, J, mo, 3), np.int32); Rv_bf = np.zeros((nB, J, mo), np.int32); Rv_eci = np.zeros((nB, J), np.float32)
    for b in range(nB):
        Rv_b[b] = b                                                   # padded entries: anchor itself, bf 0, eci 0
        for j, (r, m) in enumerate(ent[b]):
            rel = np.mod((T_off[r] - T_off[r, m]) @ U.T, d)
            Rv_b[b, j] = T_b[r]; Rv_uo[b, j] = rel; Rv_bf[b, j] = T_bf[r]; Rv_eci[b, j] = T_eci[r]
    Rv_st = basis_to_sl[Rv_b].astype(np.int32)
    # active sublattices (basis members) for the O(1) proposal
    act = sorted(sl for sl, sp in sl_species.items() if len(sp) > 1)
    maxb = max(int((basis_to_sl == sl).sum()) for sl in act)
    act_b = np.zeros((len(act), maxb), np.int32); act_nb = np.zeros(len(act), np.int32)
    for a, sl in enumerate(act):
        bs = np.where(basis_to_sl == sl)[0]; act_b[a, :len(bs)] = bs; act_nb[a] = len(bs)
    act_p = (act_nb / act_nb.sum()).astype(np.float32)
    if block is None:
        block = int(max(64, min(n_cells, 2_000_000 // max(Rt * mo, 1))))
    n_blocks = -(-n_cells // block)
    # tp = jit ARGUMENT: arrays only (python scalars would become tracers); block/n_blocks/zerolet stay in `info`
    tp = dict(n_basis=jnp.int32(nB), n_cells=jnp.int32(n_cells), d=jnp.asarray(d, jnp.int32),
              T_b=jnp.asarray(T_b), T_uo=jnp.asarray(T_uo), T_bf=jnp.asarray(T_bf), T_st=jnp.asarray(T_st), T_eci=jnp.asarray(T_eci),
              Rv_b=jnp.asarray(Rv_b), Rv_uo=jnp.asarray(Rv_uo), Rv_bf=jnp.asarray(Rv_bf), Rv_st=jnp.asarray(Rv_st), Rv_eci=jnp.asarray(Rv_eci),
              pf=jnp.asarray(pf_table, jnp.float32), act_b=jnp.asarray(act_b), act_nb=jnp.asarray(act_nb), act_p=jnp.asarray(act_p))
    site_to_sl = basis_to_sl[np.arange(N) % nB]                       # canonical order: basis = i % nB
    return dict(tp=tp, block=block, n_blocks=n_blocks, n_atoms=N, n_cells=n_cells, n_basis=nB, perm=perm, inv_perm=inv_perm, site_to_sl=site_to_sl,
                basis_to_sl=basis_to_sl, sl_species=sl_species, act_sl=act, zerolet=zerolet, n_instances=Rt * n_cells,
                n_rows_template=Rt, max_size=mo, J=J, U=U, d=d, T=T)


# ── kernels ───────────────────────────────────────────────────────────────────────────────────────
def _k_of(tp, i):
    """canonical site -> (basis, SNF cell coordinate k (3,))"""
    import jax.numpy as jnp
    nB = tp['n_basis']; d = tp['d']; b = i % nB; c = i // nB
    return b, jnp.stack([c // (d[1] * d[2]), (c // d[2]) % d[1], c % d[2]])


def _site(tp, b, k):
    """(basis, reduced k (...,3)) -> canonical site index"""
    d = tp['d']; return ((k[..., 0] * d[1] + k[..., 1]) * d[2] + k[..., 2]) * tp['n_basis'] + b


def _pf_prod(tp, st, bf, sp):
    import jax.numpy as jnp
    return jnp.prod(tp['pf'][st, bf, sp], axis=-1)


def energy_a(tp, sigma, block, n_blocks, zerolet):
    """total CE energy (zerolet + Σ eci Π pf), lax.map over blocks of cells; O(block * Rt * mo) transient memory.
    block / n_blocks / zerolet are python constants (closed over by GPUClusterExpansion.energy_a)."""
    import jax, jax.numpy as jnp
    n_cells = tp['n_cells']; d = tp['d']

    def block_sum(bi):
        c = bi * block + jnp.arange(block, dtype=jnp.int32)                              # (B,)
        k = jnp.stack([c // (d[1] * d[2]), (c // d[2]) % d[1], c % d[2]], axis=-1)        # (B, 3)
        kk = (k[:, None, None, :] + tp['T_uo'][None]) % d                                # (B, Rt, mo, 3)
        sites = _site(tp, tp['T_b'][None], kk)                                            # (B, Rt, mo)
        sp = sigma[jnp.minimum(sites, sigma.shape[0] - 1)]
        rows = tp['T_eci'][None] * _pf_prod(tp, tp['T_st'][None], tp['T_bf'][None], sp)   # (B, Rt)
        return jnp.sum(jnp.where(c < n_cells, rows.sum(1), 0.0)).astype(jnp.float32)
    s = jax.lax.map(block_sum, jnp.arange(n_blocks, dtype=jnp.int32))
    acc = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32          # E carried in float64 when x64 is on
    return jnp.asarray(zerolet, acc) + jnp.sum(s.astype(acc))


def _entries(tp, i):
    """all cluster instances containing canonical site i: member sites (J, mo) + their tables."""
    b, k = _k_of(tp, i)
    kk = (k[None, None, :] + tp['Rv_uo'][b]) % tp['d']                                    # (J, mo, 3)
    return _site(tp, tp['Rv_b'][b], kk), tp['Rv_st'][b], tp['Rv_bf'][b], tp['Rv_eci'][b]


def _species_at(sigma, sites, ov):
    import jax.numpy as jnp
    sp = sigma[sites]
    for site, species in ov:
        sp = jnp.where(sites == site, jnp.asarray(species, sp.dtype), sp)
    return sp


def delta_energy_a(tp, i_a, i_b, sigma, s_a_new, s_b_new):
    """E(after: sigma[i_a]=s_a_new, sigma[i_b]=s_b_new) - E(before); instances of i_a plus instances of i_b
    not containing i_a.  No copy of sigma, no O(N) table."""
    import jax.numpy as jnp
    ov = ((i_a, s_a_new), (i_b, s_b_new))
    sa, st_a, bf_a, ec_a = _entries(tp, i_a)
    da = ec_a * (_pf_prod(tp, st_a, bf_a, _species_at(sigma, sa, ov)) - _pf_prod(tp, st_a, bf_a, sigma[sa]))
    sb, st_b, bf_b, ec_b = _entries(tp, i_b)
    db = ec_b * (_pf_prod(tp, st_b, bf_b, _species_at(sigma, sb, ov)) - _pf_prod(tp, st_b, bf_b, sigma[sb]))
    both = jnp.any(sb == i_a, axis=1)
    return jnp.sum(da) + jnp.sum(jnp.where(both, 0.0, db))


def make_fast_swap_implicit(tp):
    """O(1), table-free proposal: active sublattice ∝ its size, uniform cell + uniform basis of that sublattice, twice."""
    import jax.numpy as jnp
    from jax import random as jrandom
    nB = tp['n_basis']; n_cells = tp['n_cells']

    def draw(key, sl):
        k1, k2 = jrandom.split(key)
        c = (jrandom.uniform(k1) * n_cells).astype(jnp.int32)
        b = tp['act_b'][sl, (jrandom.uniform(k2) * tp['act_nb'][sl]).astype(jnp.int32)]
        return c * nB + b

    def propose(key, sigma):
        k0, k1, k2 = jrandom.split(key, 3)
        sl = jrandom.choice(k0, tp['act_b'].shape[0], p=tp['act_p'])
        i_a = draw(k1, sl); i_b = draw(k2, sl)
        return i_a, i_b, sigma[i_a] != sigma[i_b]
    return propose


def encode_canonical(info, atoms, basis=None):
    """species index per site in canonical order (sigma_canon = sigma_orig[perm]) from chemical symbols; vectorised."""
    from ase.data import atomic_numbers, chemical_symbols
    Z = np.asarray(atoms.numbers); N = len(Z)
    sigma = np.zeros(N, np.int32)
    site_sl_orig = info['site_to_sl'][info['inv_perm']]                 # sublattice of each ORIGINAL site
    for sl, species in info['sl_species'].items():
        m = site_sl_orig == sl
        if not m.any(): continue
        lut = {atomic_numbers[s]: i for i, s in enumerate(species)}
        u, inv = np.unique(Z[m], return_inverse=True)
        bad = [chemical_symbols[z] for z in u if z not in lut]
        if bad: raise ValueError(f"sublattice {sl}: species {bad} not in {species}")
        sigma[m] = np.array([lut[z] for z in u], np.int32)[inv]
    return sigma[info['perm']]
