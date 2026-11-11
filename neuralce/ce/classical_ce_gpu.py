"""
classical_ce_gpu.py
═══════════════════════════════════════════════════════════════════════════════

GPU-accelerated classical Cluster Expansion (CE) evaluator.
Supports arbitrary species count per sublattice (binary, ternary, HEA, etc.)
and optional Heisenberg spin pair interactions.

KEY CHANGES FROM BINARY-ONLY VERSION
─────────────────────────────────────────────────────────────────────────────
σ encoding changed from ±1 float to integer species index.
Point function values are extracted from icet at build time and stored as
a lookup table. The runtime kernel is the same structure:
gather → point function lookup → product → weighted sum.

DATA STRUCTURES
─────────────────────────────────────────────────────────────────────────────
    cluster_sites      (n_flat, max_size) int32    — site indices, padded
    basis_fn_indices   (n_flat, max_size) int32    — which φ to apply (0=φ₀=1 for pad)
    sublattice_types   (n_flat, max_size) int32    — sublattice index per position (via icet)
    pf_table           (n_subs, max_basis, max_sp) — point function values
    eci_per_instance   (n_flat,)          float32  — J / multiplicity
    zerolet            float                       — constant CE term

Runtime: species = σ[cluster_sites]
         pf_vals = pf_table[sublattice_types, basis_fn_indices, species]
         correlations = prod(pf_vals, axis=-1)
         E = zerolet + sum(eci * correlations)

For binary CE this reduces to the previous implementation: pf_table
contains ±1 values, species indices are 0/1, and each orbit has one
cv_element.

SPIN SUPPORT
─────────────────────────────────────────────────────────────────────────────
Optional Heisenberg spin pair interactions loaded from train_ce.py output:
    E(σ, s) = E_CE(σ) + Σ_k j_k · s[pair_k[0]] · s[pair_k[1]]
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap


# ═══════════════════════════════════════════════════════════════════════════════
# Tables container
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class CEGPUTables:
    cluster_sites: jnp.ndarray       # (n_flat, max_size) int32
    basis_fn_indices: jnp.ndarray    # (n_flat, max_size) int32
    sublattice_types: jnp.ndarray    # (n_flat, max_size) int32
    pf_table: jnp.ndarray            # (n_sublattices, max_basis, max_species) float32
    eci_per_instance: jnp.ndarray    # (n_flat,) float32
    zerolet: float
    n_atoms: int
    pad_index: int
    max_size: int
    spin_pair_sites: Optional[jnp.ndarray] = None   # (n_pairs, 2) int32
    j_per_pair: Optional[jnp.ndarray] = None         # (n_pairs,) float32
    # delta-energy mode (opt-in); full-energy path ignores these. Inverted index
    # site → flat-row indices touching that site, -1 padded to a fixed width so
    # it is a dense JAX array. Mirrors CE_clamm gpu_clamm.build_jax_tables.
    site_to_row: Optional[jnp.ndarray] = None        # (n_atoms, max_per_site) int32
    max_per_site: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Point function extraction
# ═══════════════════════════════════════════════════════════════════════════════
def extract_point_functions(cs):
    """Extract point function lookup table by probing icet's get_cluster_vector.

    Uses icet's get_sublattices() for sublattice grouping — robust to
    non-diagonal supercell transformations.

    Returns
    -------
    pf_table : (n_sublattices, max_basis+1, max_species) float32
    n_sublattices : int
    """
    prim = cs.primitive_structure
    n_prim = len(prim)
    chem_syms = cs.chemical_symbols
    subs = cs.get_sublattices(prim)

    prim_to_sl = np.array([subs.get_sublattice_index_from_site_index(p)
                           for p in range(n_prim)], dtype=np.int32)
    n_subs = max(prim_to_sl) + 1

    sl_species = {}
    for p in range(n_prim):
        sl_idx = int(prim_to_sl[p])
        if sl_idx not in sl_species:
            sl_species[sl_idx] = list(chem_syms[p])

    active_sls = {sl: sp for sl, sp in sl_species.items() if len(sp) > 1}
    if not active_sls:
        return np.ones((n_subs, 1, 1), dtype=np.float32), n_subs

    max_n_species = max(len(sp) for sp in sl_species.values())
    max_n_basis = max(len(sp) - 1 for sp in active_sls.values())

    pf_table = np.zeros((n_subs, max_n_basis + 1, max_n_species), dtype=np.float64)
    pf_table[:, 0, :] = 1.0

    singlet_map = {}
    global_idx = 1
    for orbit_idx in range(len(cs.orbit_list)):
        orbit = cs.orbit_list.get_orbit(orbit_idx)
        for cv_elem in orbit.cluster_vector_elements:
            if orbit.representative_cluster.order == 1:
                ps = orbit.representative_cluster.lattice_sites[0].index
                bf = cv_elem['multicomponent_vector'][0]
                singlet_map[global_idx] = (ps, bf)
            global_idx += 1

    for sl_idx, allowed in active_sls.items():
        prim_sites_in_sl = [p for p in range(n_prim) if prim_to_sl[p] == sl_idx]
        for sp_idx, sp_sym in enumerate(allowed):
            test = prim.copy()
            for q in range(n_prim):
                if prim_to_sl[q] == sl_idx:
                    test[q].symbol = sp_sym
                else:
                    test[q].symbol = chem_syms[q][0]
            cv = cs.get_cluster_vector(test)
            for cv_gidx, (ps, bf) in singlet_map.items():
                if ps in prim_sites_in_sl:
                    pf_table[sl_idx, bf + 1, sp_idx] = cv[cv_gidx]

    return pf_table.astype(np.float32), n_subs


# ═══════════════════════════════════════════════════════════════════════════════
# Table builder
# ═══════════════════════════════════════════════════════════════════════════════
def build_ce_gpu_tables(ce, supercell, fractional_position_tolerance=None) -> CEGPUTables:
    cs = ce._cluster_space
    pol = cs._orbit_list

    if fractional_position_tolerance is None:
        fractional_position_tolerance = cs.fractional_position_tolerance

    sup_ol = pol.get_supercell_orbit_list(
        supercell,
        fractional_position_tolerance=fractional_position_tolerance,
    )

    eci_full = ce.parameters
    zerolet = float(eci_full[0])
    n_atoms = len(supercell)
    pad_index = n_atoms

    pf_table, n_subs = extract_point_functions(cs)

    sup_subs = cs.get_sublattices(supercell)
    site_to_sl = np.zeros(n_atoms + 1, dtype=np.int32)
    for site in range(n_atoms):
        site_to_sl[site] = sup_subs.get_sublattice_index_from_site_index(site)

    all_sites = []
    all_basis_fns = []
    all_eci = []

    eci_idx = 1
    for orbit_idx in range(len(sup_ol)):
        sup_orbit = sup_ol.get_orbit(orbit_idx)
        prim_orbit = pol.get_orbit(orbit_idx)

        instances = []
        for cluster in sup_orbit.clusters:
            sites = [s.index for s in cluster.lattice_sites]
            instances.append(sites)
        n_instances = len(instances)

        for cv_elem in prim_orbit.cluster_vector_elements:
            mc_vec = cv_elem['multicomponent_vector']
            eci_val = float(eci_full[eci_idx])

            for inst in instances:
                all_sites.append(inst)
                all_basis_fns.append([mc_vec[k] + 1 for k in range(len(inst))])
                all_eci.append(eci_val / n_instances)

            eci_idx += 1

    n_flat = len(all_sites)
    max_size = max(len(x) for x in all_sites) if all_sites else 1

    cluster_sites_np = np.full((n_flat, max_size), pad_index, dtype=np.int32)
    basis_fn_np = np.zeros((n_flat, max_size), dtype=np.int32)

    for i in range(n_flat):
        k = len(all_sites[i])
        cluster_sites_np[i, :k] = all_sites[i]
        basis_fn_np[i, :k] = all_basis_fns[i]

    sublattice_types_np = site_to_sl[cluster_sites_np]

    eci_np = np.array(all_eci, dtype=np.float32)

    # ── Inverted index for delta-energy MC mode (full path never reads it) ──
    # site_to_row[s] = flat-row indices whose cluster touches real site s.
    # Pad slots hold pad_index (= n_atoms) and are excluded; rows are deduped
    # per site (a cluster can touch a site in >1 slot under PBC in small cells —
    # the delta recompute is per row, so each row is listed once per site).
    per_site = [[] for _ in range(n_atoms)]
    for i in range(n_flat):
        touched = {int(cluster_sites_np[i, k]) for k in range(max_size)
                   if int(cluster_sites_np[i, k]) < n_atoms}
        for s in touched:
            per_site[s].append(i)
    max_per_site = max((len(lst) for lst in per_site), default=0)
    site_to_row_np = np.full((n_atoms, max_per_site), -1, dtype=np.int32)
    for s, lst in enumerate(per_site):
        site_to_row_np[s, : len(lst)] = lst

    return CEGPUTables(
        cluster_sites=jnp.array(cluster_sites_np),
        basis_fn_indices=jnp.array(basis_fn_np),
        sublattice_types=jnp.array(sublattice_types_np),
        pf_table=jnp.array(pf_table),
        eci_per_instance=jnp.array(eci_np),
        zerolet=zerolet,
        n_atoms=n_atoms,
        pad_index=pad_index,
        max_size=max_size,
        site_to_row=jnp.array(site_to_row_np),
        max_per_site=int(max_per_site),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Energy functions
# ═══════════════════════════════════════════════════════════════════════════════
def make_energy_fn(tables: CEGPUTables):
    cs = tables.cluster_sites
    bf = tables.basis_fn_indices
    st = tables.sublattice_types
    pf = tables.pf_table
    eci = tables.eci_per_instance
    z = tables.zerolet
    has_spin = tables.spin_pair_sites is not None

    if has_spin:
        sp_sites = tables.spin_pair_sites
        jp = tables.j_per_pair

        @jit
        def single_energy(sigma, spins):
            sigma_ext = jnp.concatenate([sigma, jnp.array([0], dtype=jnp.int32)])
            species = sigma_ext[cs]
            pf_vals = pf[st, bf, species]
            corr = jnp.prod(pf_vals, axis=-1)
            E = z + jnp.sum(eci * corr)
            si = spins[sp_sites[:, 0]]
            sj = spins[sp_sites[:, 1]]
            E = E + jnp.sum(jp * si * sj)
            return E

        @jit
        def batched_energy(sigma_batch, spins_batch):
            return vmap(single_energy)(sigma_batch, spins_batch)
    else:
        @jit
        def single_energy(sigma):
            sigma_ext = jnp.concatenate([sigma, jnp.array([0], dtype=jnp.int32)])
            species = sigma_ext[cs]
            pf_vals = pf[st, bf, species]
            corr = jnp.prod(pf_vals, axis=-1)
            return z + jnp.sum(eci * corr)

        @jit
        def batched_energy(sigma_batch):
            return vmap(single_energy)(sigma_batch)

    return single_energy, batched_energy


def make_delta_energy_fn(tables: CEGPUTables):
    """ΔE for a two-site chem swap, compositional CE only (no spin).

    Returns delta_energy(i_a, i_b, sigma_old, sigma_new) → scalar
    ΔE = E(sigma_new) − E(sigma_old), computed by recomputing ONLY the flat
    rows that touch site i_a or i_b (via tables.site_to_row) instead of all
    n_flat rows.

    Correctness: a swap changes sigma only at i_a, i_b. Rows touching neither
    have identical correlations old/new → zero delta. So Σ over the affected
    union == E_full(new) − E_full(old), up to float-summation order (the carried
    e_curr drifts ~meV over a long run; verify post-hoc with a full recompute).

    Sort-free dedup mirrors CE_clamm.gpu_clamm.make_delta_energy_fn: sum ALL of
    list_a, then add only the list_b rows that do NOT also touch i_a (those are
    already in list_a). A row in list_b touches i_a iff it is in list_a, so this
    is an exact set union with no jnp.sort in the scan body. For a degenerate
    move (i_a == i_b) every list_b row touches i_a, so the second sum is 0.

    The per-row math is duplicated verbatim from make_energy_fn so the full
    production fast path can never be perturbed by delta-mode changes. Spin pair
    interactions (tables.spin_pair_sites) are NOT handled here: an atom swap
    leaves spins untouched so the pair term is unchanged; spin-flip deltas are a
    separate (Phase 2) concern. Callers must restrict delta mode to no-spin CEs.
    """
    cs = tables.cluster_sites
    bf = tables.basis_fn_indices
    st = tables.sublattice_types
    pf = tables.pf_table
    eci = tables.eci_per_instance
    site_to_row = tables.site_to_row
    if site_to_row is None:
        raise ValueError("tables.site_to_row is None — rebuild CEGPUTables with "
                         "the current build_ce_gpu_tables (delta-energy mode).")

    def _row_delta(row_idx, i_a, sigma_old_ext, sigma_new_ext):
        # Per-row (c_new − c_old) plus whether the row touches i_a (sort-free
        # dedup). Math matches make_energy_fn.single_energy per row. Pad slots
        # gather sigma_ext[pad_index]=0 and pf[0,0,0]=1.0, so they cancel in the
        # delta and never equal i_a (i_a < n_atoms = pad_index).
        sites = cs[row_idx]
        bfs = bf[row_idx]
        sts = st[row_idx]
        c_old = eci[row_idx] * jnp.prod(pf[sts, bfs, sigma_old_ext[sites]])
        c_new = eci[row_idx] * jnp.prod(pf[sts, bfs, sigma_new_ext[sites]])
        touches_ia = jnp.any(sites == i_a)
        return c_new - c_old, touches_ia

    @jit
    def delta_energy(i_a, i_b, sigma_old, sigma_new):
        sigma_old_ext = jnp.concatenate([sigma_old, jnp.array([0], dtype=jnp.int32)])
        sigma_new_ext = jnp.concatenate([sigma_new, jnp.array([0], dtype=jnp.int32)])
        list_a = site_to_row[i_a]                # (max_per_site,), -1 padded
        list_b = site_to_row[i_b]
        da, _ = jax.vmap(_row_delta, in_axes=(0, None, None, None))(
            list_a, i_a, sigma_old_ext, sigma_new_ext)
        db, db_touches_ia = jax.vmap(_row_delta, in_axes=(0, None, None, None))(
            list_b, i_a, sigma_old_ext, sigma_new_ext)
        sum_a = jnp.sum(jnp.where(list_a >= 0, da, 0.0))
        sum_b = jnp.sum(jnp.where((list_b >= 0) & (~db_touches_ia), db, 0.0))
        return sum_a + sum_b

    return delta_energy


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════════════════
def make_sigma_encoder(cs):
    """Build Atoms → integer species index encoder.

    Uses icet's get_sublattices() for site-to-sublattice mapping,
    robust to non-diagonal supercell transformations.
    """
    chem_syms = cs.chemical_symbols
    n_prim = len(chem_syms)

    sl_species_map = {}
    prim_subs = cs.get_sublattices(cs.primitive_structure)
    for p in range(n_prim):
        sl_idx = prim_subs.get_sublattice_index_from_site_index(p)
        if sl_idx not in sl_species_map:
            sl_species_map[sl_idx] = {sym: idx for idx, sym in enumerate(chem_syms[p])}

    def encode(struct) -> np.ndarray:
        subs = cs.get_sublattices(struct)
        sigma = np.zeros(len(struct), dtype=np.int32)
        for i in range(len(struct)):
            sl_idx = subs.get_sublattice_index_from_site_index(i)
            mapping = sl_species_map[sl_idx]
            sym = struct[i].symbol
            if sym not in mapping:
                raise ValueError(
                    f"Site {i} (sublattice {sl_idx}): got '{sym}', "
                    f"expected one of {list(mapping.keys())}")
            sigma[i] = mapping[sym]
        return sigma

    return encode, sl_species_map


# ═══════════════════════════════════════════════════════════════════════════════
# Sublattice auto-detection
# ═══════════════════════════════════════════════════════════════════════════════
def build_sublattices(cs, supercell):
    """Extract sublattice info from icet for Gumbel-max swap.

    Species ordering matches cs.chemical_symbols (same as make_sigma_encoder),
    NOT subs.active_sublattices[i].chemical_symbols (which uses alphabetical).
    This ensures sigma encoding from make_initial_state / swap is consistent
    with the GPU energy evaluator's ECI table.
    """
    chem_syms = cs.chemical_symbols
    n_prim = len(chem_syms)

    # Same mapping as make_sigma_encoder
    prim_subs = cs.get_sublattices(cs.primitive_structure)
    sl_species_canonical = {}   # sl_idx → canonical species list (cs.chemical_symbols order)
    for p in range(n_prim):
        sl_idx = prim_subs.get_sublattice_index_from_site_index(p)
        if sl_idx not in sl_species_canonical:
            sl_species_canonical[sl_idx] = list(chem_syms[p])

    subs = cs.get_sublattices(supercell)
    result = []
    for sl in subs.active_sublattices:
        # Find which sublattice index this is by matching against primitive sublattices
        # (active_sublattices may reorder vs prim_subs)
        sl_species_set = set(sl.chemical_symbols)
        match_idx = None
        for prim_sl_idx, canonical in sl_species_canonical.items():
            if set(canonical) == sl_species_set:
                match_idx = prim_sl_idx
                break
        if match_idx is None:
            raise RuntimeError(
                f"Cannot match active sublattice {sl.symbol} ({sl.chemical_symbols}) "
                f"to primitive sublattices {sl_species_canonical}")

        canonical_symbols = sl_species_canonical[match_idx]
        result.append({
            'indices': jnp.array(sl.indices, dtype=jnp.int32),
            'n_species': len(canonical_symbols),
            'chemical_symbols': canonical_symbols,   # cs.chemical_symbols order
            'name': sl.symbol,
        })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Spin loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_spin_tables(spin_orbits_pkl_path: str):
    with open(spin_orbits_pkl_path, 'rb') as f:
        data = pickle.load(f)

    all_sites = []
    all_j = []
    for po, j_val in zip(data['pair_orbits'], data['j_values']):
        n_inst = len(po['instances'])
        for (i, j) in po['instances']:
            all_sites.append([i, j])
            all_j.append(j_val / n_inst)

    return (jnp.array(all_sites, dtype=jnp.int32),
            jnp.array(all_j, dtype=jnp.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# sigma → raw-lattice Atoms (inverse of mapped supercell encoding)
# ═══════════════════════════════════════════════════════════════════════════════
def sigma_to_raw_atoms(sigma, gce, reference_cif_path,
                       ideal_map, inert_species, symprec=1e-2):
    """Convert mapped-supercell sigma back to an Atoms object in the raw
    (training-data) lattice.

    icet's map_structure_to_reference attaches an 'IndexMapping' array to
    the mapped supercell such that mapped[i] originated from raw[idx_map[i]].
    This routine reads the reference cif as raw_atoms, recomputes that
    mapping, then writes sigma's per-site species back into the raw cell
    using the inverse index, so the output preserves the raw coordinates
    and atom ordering used during training.
    """
    from ase.io import read
    from icet.tools import map_structure_to_reference
    from neuralce.ce.primitive_no_idealize import detect_primitive_no_idealize

    raw_atoms = read(reference_cif_path)
    primitive = detect_primitive_no_idealize(reference_cif_path, ideal_map, symprec)
    mapped, _ = map_structure_to_reference(
        raw_atoms, primitive,
        inert_species=inert_species,
        assume_no_cell_relaxation=False)
    idx_map = mapped.arrays['IndexMapping']
    sigma_np = np.asarray(sigma)
    raw_out = raw_atoms.copy()
    for sl in gce.sublattices:
        symbols = sl['chemical_symbols']
        for i in np.array(sl['indices']):
            raw_out[int(idx_map[i])].symbol = symbols[int(sigma_np[i])]
    return raw_out


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience class
# ═══════════════════════════════════════════════════════════════════════════════
class GPUClusterExpansion:
    def __init__(self, ce, supercell, fractional_position_tolerance=None, table_mode=None, energy_scale=1.0):
        """fractional_position_tolerance: forwarded to build_ce_gpu_tables for
        the icet supercell→primitive position matcher. None → cs default
        (tight). Loosen (e.g. 1e-4) when a relaxed supercell fails the
        position match with the default tolerance.
        energy_scale: the CE value is the TOTAL energy of the cell the .ce was scaled for (repo convention, memory
        icet-framework-total-energy-convention); for a cell tiled n_reps times from that cell pass energy_scale =
        n_reps so that E and dE are total energies of the big cell (otherwise the Boltzmann factor is n_reps x too weak).
        table_mode: 'tiled' (default; full instance table, O(N) GPU memory), 'legacy' (icet supercell
        orbit list), or 'implicit' (CE/ce_gpu_implicit.py: only the primitive template on the GPU, member
        sites computed on the fly; memory independent of the cell size; delta-energy MC only, sigma in
        canonical order — see .perm / .inv_perm / .encode).  Env CE_TABLE overrides the default.
        """
        self.ce = ce
        self.supercell = supercell
        import os as _os
        mode = table_mode or _os.environ.get("CE_TABLE", "tiled")
        self.implicit = mode == "implicit"
        if self.implicit:
            from .ce_gpu_implicit import build_ce_gpu_implicit, energy_a as _ea, delta_energy_a as _da, encode_canonical
            import jax
            info = build_ce_gpu_implicit(ce, supercell, energy_scale=energy_scale); self.info = info; self.tables = None; self.energy_scale = float(energy_scale)
            self.tp = info['tp']; self.perm = info['perm']; self.inv_perm = info['inv_perm']; self.site_to_sl = info['site_to_sl']
            self._n_atoms = info['n_atoms']; self._n_instances = info['n_instances']; self._max_size = info['max_size']
            _b, _nb, _z = info['block'], info['n_blocks'], info['zerolet']
            self.energy_a = lambda tp, sigma: _ea(tp, sigma, _b, _nb, _z)
            self.delta_energy_a = lambda tp, i_a, i_b, sigma, s_a, s_b: _da(tp, i_a, i_b, sigma, s_a, s_b)
            self._encode = lambda struct: encode_canonical(info, struct); self.site_mapping = None
            # sublattices in CANONICAL site order (basis = i % n_basis)
            nB = info['n_basis']; N = info['n_atoms']; self.sublattices = []
            for sl in info['act_sl']:
                bs = np.where(info['basis_to_sl'] == sl)[0]
                idx = (np.arange(info['n_cells'])[:, None] * nB + bs[None]).reshape(-1).astype(np.int32)
                self.sublattices.append({'indices': np.sort(idx), 'n_species': len(info['sl_species'][sl]),
                                         'chemical_symbols': list(info['sl_species'][sl]), 'name': "".join(info['sl_species'][sl])})
            _tp = self.tp; self.energy = jax.jit(lambda sigma: self.energy_a(_tp, jnp.asarray(sigma)))
            self.batched_energy = jax.jit(lambda sb: jax.lax.map(lambda s_: self.energy_a(_tp, s_), sb))
            self.delta_energy = None; self.spin_tables = None; self._has_spin = False
            return
        if mode == "legacy":
            self.tables = build_ce_gpu_tables(
                ce, supercell, fractional_position_tolerance=fractional_position_tolerance)
        else:
            # translation-template table (CE/ce_gpu_tiled.py): no per-supercell icet orbit list, numpy only.
            # Falls back to the legacy builder if the cell is not an integer multiple of the primitive.
            try:
                from .ce_gpu_tiled import build_ce_gpu_tables_tiled
                self.tables = build_ce_gpu_tables_tiled(ce, supercell, energy_scale=energy_scale)
            except AssertionError as e:
                print(f"[GPUClusterExpansion] tiled table not applicable ({e}); legacy builder", flush=True)
                self.tables = build_ce_gpu_tables(
                    ce, supercell, fractional_position_tolerance=fractional_position_tolerance)
        from .ce_gpu_tiled import tables_pytree, energy_a, delta_energy_a
        self.tp = tables_pytree(self.tables)                     # jit ARGUMENT for the large-cell scan path
        _z, _n = self.tables.zerolet, self.tables.n_atoms
        self.energy_a = lambda tp, sigma: energy_a(tp, sigma, _z, _n)
        self.delta_energy_a = lambda tp, i_a, i_b, sigma, s_a, s_b: delta_energy_a(tp, i_a, i_b, sigma, s_a, s_b, _n)
        self._encode, self.site_mapping = make_sigma_encoder(ce._cluster_space)
        self.sublattices = build_sublattices(ce._cluster_space, supercell)
        self.energy, self.batched_energy = make_energy_fn(self.tables)
        # Opt-in delta-energy for atom-swap MC (compositional/no-spin). The @jit
        # closure is cheap to build and only traces on first call, so full-mode
        # runs that never call it pay nothing.
        self.delta_energy = make_delta_energy_fn(self.tables)

    def load_spin(self, spin_orbits_pkl_path: str):
        sp_sites, jp = load_spin_tables(spin_orbits_pkl_path)
        self.tables.spin_pair_sites = sp_sites
        self.tables.j_per_pair = jp
        self.energy, self.batched_energy = make_energy_fn(self.tables)

    @property
    def n_atoms(self) -> int:
        return self._n_atoms if self.tables is None else self.tables.n_atoms

    @property
    def n_instances(self) -> int:
        return self._n_instances if self.tables is None else int(self.tables.cluster_sites.shape[0])

    @property
    def max_size(self) -> int:
        return self._max_size if self.tables is None else self.tables.max_size

    @property
    def has_spin(self) -> bool:
        return False if self.tables is None else self.tables.spin_pair_sites is not None

    def encode(self, struct) -> np.ndarray:
        return self._encode(struct)

    def __repr__(self) -> str:
        s = (f"GPUClusterExpansion(n_atoms={self.n_atoms}, "
             f"n_instances={self.n_instances}, max_size={self.max_size}")
        if self.has_spin:
            s += f", spin_pairs={int(self.tables.spin_pair_sites.shape[0])}"
        s += f", sublattices={len(self.sublattices)})"
        return s


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import time
    from ase.build import make_supercell
    from icet import ClusterExpansion
    from jax import random as jrandom

    CE_PATH = os.environ.get('CE_PATH', 'models/stfo.ce')
    REPEAT = (2, 2, 2)

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading: {CE_PATH}")

    ce = ClusterExpansion.read(CE_PATH)
    cs = ce._cluster_space
    prim = cs.primitive_structure
    supercell = make_supercell(prim, np.diag(REPEAT))

    gce = GPUClusterExpansion(ce, supercell)
    print(gce)

    print(f"\nSublattices:")
    for sl in gce.sublattices:
        print(f"  {sl['name']}: {sl['chemical_symbols']} ({len(sl['indices'])} sites)")

    print(f"\nPoint function table shape: {gce.tables.pf_table.shape}")
    print(f"  (n_sublattices={gce.tables.pf_table.shape[0]}, "
          f"max_basis={gce.tables.pf_table.shape[1]}, "
          f"max_species={gce.tables.pf_table.shape[2]})")

    sigma = jnp.array(gce.encode(supercell))
    E_gpu = float(gce.energy(sigma))
    E_icet = float(ce.predict(supercell))
    print(f"\nAccuracy:")
    print(f"  icet:  {E_icet:.6f}")
    print(f"  GPU:   {E_gpu:.6f}")
    print(f"  diff:  {abs(E_icet - E_gpu):.2e}")

    max_species = max(len(s) for s in cs.chemical_symbols)
    print(f"\nSpeed benchmark:")
    for R in [1, 32, 128, 1024, 3000]:
        key = jrandom.PRNGKey(R)
        sb = jrandom.randint(key, shape=(R, gce.n_atoms), minval=0, maxval=max_species)
        E = gce.batched_energy(sb); E.block_until_ready()
        t0 = time.time()
        for _ in range(1000):
            E = gce.batched_energy(sb)
        E.block_until_ready()
        dt_ms = (time.time() - t0) / 1000 * 1000
        print(f"  R={R:>5d}: {dt_ms:>8.4f} ms/call,  {dt_ms*1000/R:>8.3f} μs/replica")
