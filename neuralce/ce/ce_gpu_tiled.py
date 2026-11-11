"""Large-supercell path for the orthonormal (icet) GPU cluster expansion.

Three O(N) traps of the legacy path (classical_ce_gpu / pt_mcmc_ce) are removed here, exactly as in neuralce.nce:
  1. table build: icet's get_supercell_orbit_list + python loops (memory ∝ N python objects) → the PRIMITIVE
     orbit list is a translation template (basis, unit-cell offset); the supercell table is that template
     tiled over all primitive cells with integer arithmetic (numpy only).
  2. jit constants: the flat tables were closed over by the jitted energy/delta functions → here they are a
     pytree ARGUMENT (`tables_pytree`) so the scan carries no O(N) constants.
  3. per-step copies: sigma_new = sigma.at[..].set(..) and jnp.concatenate([sigma, [0]]) copied the (N,)
     state per replica per step; the delta here reads the two new species as scalars (`delta_energy_a`), and
     the O(N) Gumbel-max site selection is replaced by an O(1) draw from the sublattice site table
     (same-species draw = rejected proposal; symmetric proposal, valid MCMC).
The energy convention is unchanged: E(sigma) = zerolet + Σ_rows eci_row * Π pf  (= the CE value; the repo's
.ce files are fitted on TOTAL energies of the training cell, see memory icet-framework-total-energy-convention).
"""
from __future__ import annotations
import numpy as np


def build_ce_gpu_tables_tiled(ce, supercell, energy_scale=1.0):
    """CEGPUTables identical (up to row order) to classical_ce_gpu.build_ce_gpu_tables."""
    import jax.numpy as jnp
    from .classical_ce_gpu import CEGPUTables, extract_point_functions
    from neuralce.nce.gpu_layer0_tiled import site_decompose
    cs = ce._cluster_space; pol = cs._orbit_list; prim = cs.primitive_structure
    eci_full = np.asarray(ce.parameters, float) * float(energy_scale); zerolet = float(eci_full[0])
    n_atoms = len(supercell); pad_index = n_atoms
    pf_table, n_subs = extract_point_functions(cs)
    prim_subs = cs.get_sublattices(prim); basis_to_sl = np.array([prim_subs.get_sublattice_index_from_site_index(p) for p in range(len(prim))], np.int32)
    basis, off_red, T, n_cells, reduce = site_decompose(prim, supercell)
    # template rows: (basis, offset) per member, basis-function index, eci / n_instances
    tb, to, tbf, teci, tsz = [], [], [], [], []
    eci_idx = 1
    for oi in range(len(pol)):
        o = pol.get_orbit(oi)
        insts = [[(int(ls.index), tuple(int(x) for x in ls.unitcell_offset)) for ls in c.lattice_sites] for c in o.clusters]
        n_inst_sc = len(insts) * n_cells                              # instances in the supercell
        for e in o.cluster_vector_elements:
            mc = e['multicomponent_vector']; ev = float(eci_full[eci_idx]) / n_inst_sc
            for inst in insts:
                tb.append([s[0] for s in inst]); to.append([s[1] for s in inst]); tbf.append([mc[k] + 1 for k in range(len(inst))]); teci.append(ev); tsz.append(len(inst))
            eci_idx += 1
    Rt = len(tb); mo = max(tsz)
    B = np.zeros((Rt, mo), np.int64); O = np.zeros((Rt, mo, 3), np.int64); BF = np.zeros((Rt, mo), np.int32); Mk = np.zeros((Rt, mo), bool)
    for r in range(Rt):
        k = tsz[r]; B[r, :k] = tb[r]; O[r, :k] = to[r]; BF[r, :k] = tbf[r]; Mk[r, :k] = True
    # tile: (n_cells, Rt, mo) member sites via key lookup
    lo = off_red.min(0); span = off_red.max(0) - lo + 1
    def key(b, t): d = t - lo; return ((b * span[0] + d[..., 0]) * span[1] + d[..., 1]) * span[2] + d[..., 2]
    keys = key(basis, off_red); order_k = np.argsort(keys); ks = keys[order_k]; assert len(np.unique(ks)) == n_atoms
    cells = off_red[basis == 0] if (basis == 0).any() else np.unique(off_red, axis=0); assert len(cells) == n_cells
    t_all = reduce((cells[:, None, None, :] + O[None]).reshape(-1, 3)).reshape(n_cells, Rt, mo, 3)
    kq = key(np.broadcast_to(B[None], (n_cells, Rt, mo)), t_all); pos = np.clip(np.searchsorted(ks, kq), 0, n_atoms - 1)
    hit = (ks[pos] == kq) | ~np.broadcast_to(Mk[None], (n_cells, Rt, mo)); assert hit.all(), "template member off the supercell lattice"
    sites = order_k[pos].reshape(n_cells * Rt, mo).astype(np.int32); mask = np.broadcast_to(Mk[None], (n_cells, Rt, mo)).reshape(-1, mo)
    sites[~mask] = pad_index
    bf = np.broadcast_to(BF[None], (n_cells, Rt, mo)).reshape(-1, mo).astype(np.int32); bf[~mask] = 0
    site_to_sl = np.zeros(n_atoms + 1, np.int32); site_to_sl[:n_atoms] = basis_to_sl[basis]
    st = site_to_sl[sites]
    eci = np.broadcast_to(np.asarray(teci, np.float32)[None], (n_cells, Rt)).reshape(-1)
    # inverted index site -> rows (CSR, padded)
    R = sites.shape[0]; rr = np.repeat(np.arange(R, dtype=np.int32), mo).reshape(R, mo)[mask]; ss = sites[mask]
    # dedupe (row, site) pairs (a cluster may touch a site twice under PBC in tiny cells)
    pair = np.unique(ss.astype(np.int64) * R + rr); ss, rr = (pair // R).astype(np.int32), (pair % R).astype(np.int32)
    counts = np.bincount(ss, minlength=n_atoms); Rmax = int(counts.max()); starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    site_to_row = np.full((n_atoms, Rmax), -1, np.int32); site_to_row[ss, np.arange(len(ss)) - np.repeat(starts, counts)] = rr
    return CEGPUTables(cluster_sites=jnp.asarray(sites), basis_fn_indices=jnp.asarray(bf), sublattice_types=jnp.asarray(st),
                       pf_table=jnp.asarray(pf_table), eci_per_instance=jnp.asarray(eci), zerolet=zerolet, n_atoms=n_atoms,
                       pad_index=pad_index, max_size=mo, site_to_row=jnp.asarray(site_to_row), max_per_site=Rmax)


def tables_pytree(tables):
    """the big arrays as a dict → passed as a jit ARGUMENT (never closed over)."""
    return dict(cs=tables.cluster_sites, bf=tables.basis_fn_indices, st=tables.sublattice_types, pf=tables.pf_table,
                eci=tables.eci_per_instance, site_to_row=tables.site_to_row)


def _species_at(tp, sigma, sites, n_atoms, ov=()):
    """species on `sites` (pad sites -> 0, whose point function is 1), with pending flips `ov` applied on the fly."""
    import jax.numpy as jnp
    sp = jnp.where(sites < n_atoms, sigma[jnp.minimum(sites, n_atoms - 1)], 0)
    for site, species in ov:
        sp = jnp.where(sites == site, jnp.asarray(species, sp.dtype), sp)
    return sp


def energy_a(tp, sigma, zerolet, n_atoms):
    import jax.numpy as jnp
    pf_vals = tp['pf'][tp['st'], tp['bf'], _species_at(tp, sigma, tp['cs'], n_atoms)]
    return zerolet + jnp.sum(tp['eci'] * jnp.prod(pf_vals, axis=-1))


def delta_energy_a(tp, i_a, i_b, sigma, s_a_new, s_b_new, n_atoms):
    """E(after swap) - E(before): rows touching i_a plus rows touching i_b that do not touch i_a; no copy of sigma."""
    import jax, jax.numpy as jnp
    ov = ((i_a, s_a_new), (i_b, s_b_new))

    def row_delta(row):
        sites = tp['cs'][row]; bfs = tp['bf'][row]; sts = tp['st'][row]
        c_old = tp['eci'][row] * jnp.prod(tp['pf'][sts, bfs, _species_at(tp, sigma, sites, n_atoms)])
        c_new = tp['eci'][row] * jnp.prod(tp['pf'][sts, bfs, _species_at(tp, sigma, sites, n_atoms, ov)])
        return c_new - c_old, jnp.any(sites == i_a)
    la = tp['site_to_row'][i_a]; lb = tp['site_to_row'][i_b]
    da, _ = jax.vmap(row_delta)(jnp.where(la >= 0, la, 0)); db, tb = jax.vmap(row_delta)(jnp.where(lb >= 0, lb, 0))
    return jnp.sum(jnp.where(la >= 0, da, 0.0)) + jnp.sum(jnp.where((lb >= 0) & (~tb), db, 0.0))


def make_fast_swap(mcmc_tables):
    """O(1) proposal: sublattice ∝ size, two uniform sites on it. Returns (i_a, i_b, valid=different species)."""
    import jax, jax.numpy as jnp
    from jax import random as jrandom
    sub_idx = mcmc_tables.sub_indices; sub_len = jnp.sum(mcmc_tables.sub_mask, axis=1); probs = sub_len / jnp.sum(sub_len)

    def propose(key, sigma):
        k1, k2, k3 = jrandom.split(key, 3)
        sl = jrandom.choice(k1, sub_idx.shape[0], p=probs)
        i_a = sub_idx[sl, (jrandom.uniform(k2) * sub_len[sl]).astype(jnp.int32)]
        i_b = sub_idx[sl, (jrandom.uniform(k3) * sub_len[sl]).astype(jnp.int32)]
        return i_a, i_b, sigma[i_a] != sigma[i_b]
    return propose
