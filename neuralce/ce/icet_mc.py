"""MCMC entry for icet (orthonormal) CE.

Single-T → run_mc  = GPU lax.scan via gpu_mc.run_gpu_mc        (production)
Multi-T → run_multiT_mc = GPU + in-memory chain via gpu_mc      (production)
PT      → run_pt_mc → wraps CE/pt_mcmc_ce.run_pt_mcmc (GPU, lax.scan, vmap replicas)

CPU debug loops still available as run_mc_cpu / run_multiT_mc_cpu in
.acceptance_ratio (small jobs / cross-check only).
"""
from __future__ import annotations
import os
import hashlib
import numpy as np
from pathlib import Path
from .gpu_mc import run_gpu_mc as run_mc, run_multiT_gpu_mc as run_multiT_mc
from .acceptance_ratio import (
    run_mc as run_mc_cpu, run_multiT_mc as run_multiT_mc_cpu)


def _rebuild_as_clean_supercell(mapped_atoms, primitive, cell_atol=0.1):
    """Build an idealized integer supercell of `primitive` and transfer
    `mapped_atoms`' species onto it by PBC-aware nearest-neighbor matching.

    Why this exists: icet's `get_supercell_orbit_list` (called inside
    GPUClusterExpansion → build_ce_gpu_tables) requires both the supercell
    *cell* AND the per-atom *positions* to correspond exactly to an integer
    transformation of the primitive's lattice. `map_structure_to_reference`
    preserves the raw cif's relaxed positions (e.g. STFO new_bench cifs are
    relaxed off-ideal even though P=round(cell @ inv(prim)) is integer), so
    `findLatticeSiteByPosition` fails with 'Failed to find site by position'.

    Implementation:
      1. P_int = round(mapped.cell @ inv(prim.cell))  (must be integer)
      2. ideal = make_supercell(primitive, P_int)     (clean lattice positions)
      3. For each ideal site, take the species of the PBC-nearest mapped atom.
      Composition is preserved exactly when |mapped_atoms| == |ideal|.
    """
    from ase.build import make_supercell
    P = mapped_atoms.cell.array @ np.linalg.inv(primitive.cell.array)
    P_int = np.round(P).astype(int)
    residual = np.max(np.abs(P - P_int))
    if residual > cell_atol:
        raise ValueError(
            f"Mapped supercell cell deviates too far from an integer "
            f"transformation of CE primitive (max residual {residual:.4f} > {cell_atol}). "
            f"cif and CE primitive geometries are incompatible. "
            f"mapped_cell={mapped_atoms.cell.array.tolist()}, "
            f"primitive_cell={primitive.cell.array.tolist()}")
    ideal = make_supercell(primitive, P_int)
    if len(ideal) != len(mapped_atoms):
        raise ValueError(
            f"Clean supercell atom count ({len(ideal)}) ≠ mapped "
            f"({len(mapped_atoms)}). primitive may not match the mapping primitive.")
    inv_cell = np.linalg.inv(ideal.cell.array)
    mapped_pos = mapped_atoms.positions
    mapped_syms = mapped_atoms.get_chemical_symbols()
    new_syms = []
    used = np.zeros(len(mapped_atoms), dtype=bool)
    for ipos in ideal.positions:
        delta = mapped_pos - ipos
        frac = delta @ inv_cell
        frac -= np.round(frac)
        cart = frac @ ideal.cell.array
        d2 = (cart ** 2).sum(axis=1)
        d2[used] = np.inf  # bijection: each mapped atom assigned to exactly one ideal site
        j = int(np.argmin(d2))
        used[j] = True
        new_syms.append(mapped_syms[j])
    ideal.set_chemical_symbols(new_syms)
    if residual > 1e-4 or not np.all(used):
        print(f"  rebuilt clean supercell from CE primitive: P_int={P_int.tolist()}, "
              f"max cell residual={residual:.4f}, n_atoms={len(ideal)}", flush=True)
    return ideal


def run_pt_mc(ce_path, init_cif, n_steps, T_min=100.0, T_max=2000.0,
              n_replicas=32, n_ladders=1, swap_mode="uniform", seed=0, log_every=0,
              n_equilib=0, exchange_every=100, chunk_size=1000,
              sample_interval=100, out_dir=".",
              ideal_map=None, inert_species=None, symprec=1e-2,
              swap_pairs=None, ref_cif=None,
              fractional_position_tolerance=None,
              energy_mode="full",
              spin_move_prob=0.0, spin_orbits_pkl="", spin_states=None,
              supercell_reps=None, table_mode=None, checkpoint_every=0, T_grid=None, init_npz=None):
    """Run icet PT-MCMC via CE/pt_mcmc_ce. Fully GPU-batched.

    supercell_reps — (nx, ny, nz): tile the (ideal) init_cif in memory and re-shuffle the species within
                     every sublattice (random configuration at the cif's composition; no periodic-image start,
                     no giant cif on disk).  table_mode — GPUClusterExpansion table mode; delta mode defaults
                     to 'implicit' (template-only GPU tables, memory independent of the cell size).

    T_grid — explicit temperatures (K), one replica each (overrides T_min/T_max/n_replicas; n_ladders must be 1).
    init_npz — per-replica starting structures (neuralce/models/init_structures.py): replica r starts from numbers[r],
               matched onto this MC cell by lattice position; the npz temperatures must equal the run's.  Needs the
               implicit tables (energy_mode delta).

    System config (same semantics as single-T `run_mc`):
        ideal_map / inert_species / symprec — primitive detection
        ref_cif — cif for primitive detection (defaults to init_cif)
        swap_pairs — list of symbol pairs (e.g. [["Ti","Fe"],["O","Xe"]]).
                     Each pair must lie within one icet sublattice;
                     pt_mcmc_ce's sublattice-based swap_fn auto-enforces
                     same-sublattice swaps. Pairs that cross sublattices raise.
        fractional_position_tolerance — forwarded to GPUClusterExpansion for
                     the supercell→primitive position match. None → icet
                     default; loosen (e.g. 1e-4) for relaxed cells that fail
                     the default tolerance.

    swap_mode is accepted for API symmetry but not consumed: pt_mcmc_ce's
    swap_fn picks uniformly over (sublattice, different-species pair); a
    'weighted' variant would require a new swap_fn (not in this path).

    init_cif is mapped to the icet reference primitive before passing to
    GPUClusterExpansion so PT initial state matches single-T initial state.
    """
    from icet import ClusterExpansion
    from ase.io import read as ase_read
    from icet.tools import map_structure_to_reference
    from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
    from neuralce.ce.pt_mcmc_ce import MCMCConfig, run_pt_mcmc
    from neuralce.ce.primitive_no_idealize import detect_primitive_no_idealize

    if swap_mode not in ("uniform", "weighted"):
        raise ValueError(f"swap_mode must be 'uniform' or 'weighted'; got {swap_mode!r}")
    if swap_mode == "weighted":
        print("PT-MC: warning — icet PT swap_fn is sublattice-uniform; "
              "'weighted' falls back to uniform.", flush=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ce = ClusterExpansion.read(ce_path)
    raw = ase_read(init_cif)
    ce_primitive = ce._cluster_space.primitive_structure
    prim_src = ref_cif if ref_cif else init_cif
    if ideal_map or inert_species:
        # Detect primitive from cif for the *mapping* (so map_structure_to_reference
        # finds atoms correctly); the resulting `atoms` is then snapped to CE's
        # primitive cell (next step) so build_ce_gpu_tables works.
        primitive = detect_primitive_no_idealize(prim_src, ideal_map or {}, symprec)
        atoms, _ = map_structure_to_reference(
            raw, primitive, inert_species=list(inert_species or []),
            assume_no_cell_relaxation=False)
        print(f"PT-MC: mapped to primitive  ideal_map={ideal_map}  "
              f"inert={list(inert_species or [])}  prim_src={prim_src}", flush=True)
    else:
        atoms = raw
    # Rebuild as a clean integer supercell of CE primitive so the GPU path's
    # get_supercell_orbit_list succeeds. map_structure_to_reference preserves
    # the cif's relaxed positions, which fall outside icet's site-finding
    # tolerance even when the cell matrix happens to be exactly integer.
    if table_mode is None and energy_mode == "delta":
        table_mode = os.environ.get("CE_TABLE", "implicit")
    energy_scale = 1.0
    if supercell_reps is not None and tuple(supercell_reps) != (1, 1, 1):
        atoms = _rebuild_as_clean_supercell(atoms, ce_primitive)
        small = atoms; atoms = small.repeat(tuple(int(r) for r in supercell_reps))
        energy_scale = float(np.prod([int(r) for r in supercell_reps]))          # CE value = total E of the base cell -> x n_reps
        rng_sc = np.random.default_rng(seed); Z = atoms.numbers.copy()
        periodic = os.environ.get("CE_SUPERCELL_PERIODIC", "0") == "1"                     # ordered start: keep the images
        from neuralce.ce.ce_gpu_implicit import build_ce_gpu_implicit as _bci
        # shuffle species within each sublattice (basis sets of the primitive); sublattice of a site from the
        # primitive decomposition (numpy, no icet call on the big cell)
        from neuralce.nce.gpu_layer0_tiled import site_decompose as _sd
        basis, _, _, _, _ = _sd(ce_primitive, atoms)
        prim_subs = ce._cluster_space.get_sublattices(ce_primitive)
        b2sl = np.array([prim_subs.get_sublattice_index_from_site_index(p) for p in range(len(ce_primitive))])
        for sl in ([] if periodic else np.unique(b2sl)):
            m = np.isin(basis, np.where(b2sl == sl)[0]); Z[m] = rng_sc.permutation(Z[m])
        atoms.numbers = Z
        print(f"PT-MC: supercell {tuple(supercell_reps)} of the init cif -> {len(atoms)} atoms, " + ("periodic images kept (ordered start)" if periodic else f"species re-shuffled per sublattice (seed {seed})"), flush=True)
    else:
        ideal = False
        if table_mode == "implicit":
            try:
                from neuralce.nce.gpu_layer0_tiled import site_decompose as _sd
                _sd(ce_primitive, atoms); ideal = True                 # already a clean integer supercell: no rebuild needed
            except AssertionError:
                ideal = False
        if not ideal:
            atoms = _rebuild_as_clean_supercell(atoms, ce_primitive)
    gce = GPUClusterExpansion(
        ce, atoms,
        fractional_position_tolerance=fractional_position_tolerance, table_mode=table_mode, energy_scale=energy_scale)
    print(f"PT-MC: {gce}  energy_scale {energy_scale:g} (CE value = total energy of the {len(atoms)/energy_scale:.0f}-atom base cell)", flush=True)
    # Encode the cif's configuration as the PT initial state (otherwise PT
    # starts from a random per-sublattice configuration with E far from the
    # cif's actual energy).
    initial_sigma = gce.encode(atoms)
    # icet GPUClusterExpansion.energy takes only sigma for compositional CE
    # (no spin); for spin CE it expects (sigma, spins) — handle both.
    try:
        init_E = float(gce.energy(initial_sigma))
    except TypeError:
        init_E = float(gce.energy(
            initial_sigma, np.zeros(len(atoms), dtype=np.float32)))
    print(f"PT-MC: init E (from cif) = {init_E:.4f} eV", flush=True)
    if init_npz:
        if not getattr(gce, "implicit", False):
            raise ValueError("--init_npz needs the implicit tables (--energy_mode delta, CE_TABLE=implicit)")
        if n_ladders != 1:
            raise ValueError("--init_npz assigns one structure per temperature of a single ladder; n_ladders must be 1")
        from neuralce.mc.init_structures import load_init_npz
        run_T = np.asarray(T_grid, float) if T_grid is not None else np.geomspace(T_min, T_max, n_replicas)
        nums, _ = load_init_npz(init_npz, ce_primitive, atoms, run_T)
        a_r = atoms.copy(); rows = []
        for r in range(nums.shape[0]):
            a_r.numbers = nums[r]; rows.append(gce.encode(a_r))            # atom order of `atoms` -> canonical sigma
        initial_sigma = np.stack(rows)
        print(f"PT-MC: per-replica starting structures from {init_npz} ({initial_sigma.shape[0]} replicas, matched by lattice position)", flush=True)

    # Fingerprint of what the run actually integrates: the CE parameters and the starting configuration (which already
    # carries the init cif, the supercell shuffle seed and --init_npz).  Stored in the checkpoint; a restart in the same
    # folder with a different CE or different starting structures is then refused instead of resuming that state.
    _h = hashlib.md5(np.ascontiguousarray(ce.parameters, dtype=np.float64).tobytes())
    _h.update(np.ascontiguousarray(initial_sigma, dtype=np.int32).tobytes())
    run_id = _h.hexdigest()[:16]

    cfg = MCMCConfig(
        T_min=T_min, T_max=T_max, n_replicas=(len(T_grid) if T_grid is not None else n_replicas),
        n_ladders=n_ladders,
        n_steps=n_steps, n_equilib=n_equilib,
        exchange_interval=exchange_every,
        chunk_size=chunk_size, sample_interval=sample_interval,
        output_dir=out_dir, seed=seed,
        swap_pairs=swap_pairs,
        initial_sigma=initial_sigma,
        energy_mode=energy_mode,
        spin_move_prob=spin_move_prob,
        spin_orbits_pkl=spin_orbits_pkl or "",
        spin_states=spin_states or {},
        checkpoint_every=checkpoint_every,
        T_grid=(np.asarray(T_grid, float) if T_grid is not None else None),
        run_id=run_id,
    )
    return run_pt_mcmc(gce, cfg)
