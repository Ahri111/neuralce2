"""
pt_mcmc_ce.py
═══════════════════════════════════════════════════════════════════════════════

GPU-accelerated Parallel Tempering MCMC for classical Cluster Expansion.
Supports arbitrary species count per sublattice and optional discrete spin.

Two modes (auto-detected from config):
    Compositional only:  swap moves on σ
    Compositional + spin: swap moves on σ + spin flip moves on s

All inner-loop operations are JIT-compiled. Replicas are vmapped.
No Python in the hot path.
"""

from __future__ import annotations

import os
import sys
import time
import faulthandler
import signal
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, random as jrandom

from neuralce.ce.classical_ce_gpu import GPUClusterExpansion


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class MCMCConfig:
    # CE source (only used by the __main__ self-test; production callers
    # supply gce directly to run_pt_mcmc and these stay empty).
    ce_path: str = ''
    supercell_matrix: list = field(default_factory=list)

    # Temperature
    T_min: float = 100.0
    T_max: float = 2000.0
    n_replicas: int = 64

    # MCMC
    n_steps: int = 100000
    n_equilib: int = 10000
    sample_interval: int = 100
    exchange_interval: int = 10
    chunk_size: int = 1000          # jax.lax.scan body length (must divide
                                    # both n_equilib and (n_steps − n_equilib),
                                    # and be a multiple of sample_interval)

    # Spin (optional)
    spin_orbits_pkl: str = ''
    spin_states: dict = field(default_factory=dict)
    # Probability of attempting a spin flip (vs a chem swap) per step, when the
    # CE has spin. 0.0 (default) = never flip (pure compositional MC); mirrors
    # CLAMM's --spin_move_prob. Ignored when the CE has no spin.
    spin_move_prob: float = 0.0

    # Initial composition — fraction per sublattice
    # e.g. {'A': [0.5, 0.5]} for 50/50 binary
    # e.g. {'A': [0.6, 0.2, 0.2]} for ternary
    # If empty, defaults to uniform distribution
    initial_composition: dict = field(default_factory=dict)

    # Optional chemistry constraint: list of symbol pairs (e.g.
    # [("Ti","Fe"),("O","Xe")]) that swaps must respect. Each pair must lie
    # within a single sublattice — the sublattice-based swap_fn already
    # restricts swaps to same-sublattice different-species pairs, so this is
    # a verification hook (catches user input that crosses sublattices).
    swap_pairs: Optional[list] = None

    # Optional initial sigma (shape (n_atoms,) int32). If provided, broadcast
    # to all replicas as the starting configuration — overrides random init
    # from initial_composition. Used when the caller wants PT to start from a
    # specific cif's encoded sigma instead of random fractions.
    initial_sigma: Optional[np.ndarray] = None

    # Number of independent ladders for interleaved multi-ladder PT.
    # 1 = standard PT (default, backward compat). n_ladders > 1 splits
    # replicas into block-diagonal exchange groups; each ladder spans
    # T_min~T_max via every n_ladders-th element of a global geomspace.
    # Total replicas = n_replicas (per-ladder) × n_ladders.
    n_ladders: int = 1

    seed: int = 42
    output_dir: str = 'mcmc_results'

    # MC energy update. 'full' (default): recompute the full CE energy on every
    # proposed swap (unchanged production path). 'delta': recompute only the
    # flat rows touching the swapped sites and carry E as a running sum
    # (compositional/no-spin CEs only). See make_delta_energy_fn.
    energy_mode: str = 'full'
    checkpoint_every: int = 0        # chunks; 0 = off.  Saves state to output_dir/checkpoint.npz and resumes from it on restart
    T_grid: Optional[np.ndarray] = None   # explicit temperatures (K), one replica each; overrides T_min/T_max/n_replicas (n_ladders 1)
    run_id: str = ''                 # fingerprint of the CE parameters + starting configuration (set by the caller); stored in the
                                     # checkpoint, so a restart in the same folder with a different CE or different starting
                                     # structures is refused instead of silently resuming someone else's state


# ═══════════════════════════════════════════════════════════════════════════════
# MCMC tables
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class MCMCTables:
    # Sublattice info for swap moves
    sub_indices: jnp.ndarray     # (n_active_subs, max_sub_size) int32 padded with -1
    sub_mask: jnp.ndarray        # (n_active_subs, max_sub_size) bool
    sub_n_species: jnp.ndarray   # (n_active_subs,) int32
    sub_log_probs: jnp.ndarray   # (n_active_subs,) float32 — log selection probability

    # Site info
    site_to_sl: jnp.ndarray      # (n_atoms,) int32 — sublattice index per site
    n_atoms: int
    n_active_subs: int

    # Spin tables (None if no spin)
    has_spin: bool
    spin_allowed: Optional[jnp.ndarray] = None   # (n_subs, max_sp, max_spin_states) float32
    n_spin_states: Optional[jnp.ndarray] = None   # (n_subs, max_sp) int32
    max_spin_states: int = 1


def build_mcmc_tables(gce: GPUClusterExpansion, spin_states_config: dict = None) -> MCMCTables:
    cs = gce.ce._cluster_space
    supercell = gce.supercell
    n_atoms = gce.n_atoms

    # Sublattice info
    active_subs = gce.sublattices
    n_active = len(active_subs)
    max_sub_size = max(len(sl['indices']) for sl in active_subs)

    sub_indices_np = np.full((n_active, max_sub_size), -1, dtype=np.int32)
    sub_mask_np = np.zeros((n_active, max_sub_size), dtype=bool)
    sub_n_species_np = np.zeros(n_active, dtype=np.int32)
    sub_sizes = np.zeros(n_active, dtype=np.float32)

    for i, sl in enumerate(active_subs):
        idx = np.array(sl['indices'])
        n = len(idx)
        sub_indices_np[i, :n] = idx
        sub_mask_np[i, :n] = True
        sub_n_species_np[i] = sl['n_species']
        sub_sizes[i] = n

    sub_probs = sub_sizes / sub_sizes.sum()
    sub_log_probs = np.log(sub_probs).astype(np.float32)

    # site_to_sl — full supercell (implicit tables: precomputed in canonical order, no icet call on the supercell)
    if getattr(gce, "site_to_sl", None) is not None:
        site_to_sl_np = np.asarray(gce.site_to_sl, dtype=np.int32)
    else:
        sup_subs = cs.get_sublattices(supercell)
        site_to_sl_np = np.zeros(n_atoms, dtype=np.int32)
        for i in range(n_atoms):
            site_to_sl_np[i] = sup_subs.get_sublattice_index_from_site_index(i)

    # Spin tables
    has_spin = bool(spin_states_config) and gce.has_spin
    spin_allowed = None
    n_spin_states_arr = None
    max_spin_states = 1

    if spin_states_config:
        species_to_sl_sp = {}
        for sl_idx, sl in enumerate(active_subs):
            for sp_idx, sym in enumerate(sl['chemical_symbols']):
                species_to_sl_sp[sym] = (sl_idx, sp_idx)

        all_spin_lists = {}
        for sym, states in spin_states_config.items():
            if sym in species_to_sl_sp:
                sl_i, sp_i = species_to_sl_sp[sym]
                all_spin_lists[(sl_i, sp_i)] = list(states)

        max_sp = max(sl['n_species'] for sl in active_subs)
        max_spin_states = max(len(v) for v in all_spin_lists.values()) if all_spin_lists else 1
        max_spin_states = max(max_spin_states, 1)

        spin_allowed_np = np.zeros((n_active, max_sp, max_spin_states), dtype=np.float32)
        n_spin_states_np = np.ones((n_active, max_sp), dtype=np.int32)

        for sl_i in range(n_active):
            for sp_i in range(active_subs[sl_i]['n_species']):
                if (sl_i, sp_i) in all_spin_lists:
                    states = all_spin_lists[(sl_i, sp_i)]
                    n_s = len(states)
                    spin_allowed_np[sl_i, sp_i, :n_s] = states
                    n_spin_states_np[sl_i, sp_i] = n_s

        spin_allowed = jnp.array(spin_allowed_np)
        n_spin_states_arr = jnp.array(n_spin_states_np)
        has_spin = True

    return MCMCTables(
        sub_indices=jnp.array(sub_indices_np),
        sub_mask=jnp.array(sub_mask_np),
        sub_n_species=jnp.array(sub_n_species_np),
        sub_log_probs=jnp.array(sub_log_probs),
        site_to_sl=jnp.array(site_to_sl_np),
        n_atoms=n_atoms,
        n_active_subs=n_active,
        has_spin=has_spin,
        spin_allowed=spin_allowed,
        n_spin_states=n_spin_states_arr,
        max_spin_states=max_spin_states,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Temperature ladder
# ═══════════════════════════════════════════════════════════════════════════════
_KB_EV_PER_K = 8.617333e-5


def make_temperature_ladder(T_min, T_max, n_replicas):
    if n_replicas == 1:
        temps = np.array([T_min])
    else:
        temps = np.geomspace(T_min, T_max, n_replicas)
    betas = 1.0 / (_KB_EV_PER_K * temps)  # 1/(kB*T), kB in eV/K
    return jnp.array(temps, dtype=jnp.float32), jnp.array(betas, dtype=jnp.float32)


def _build_interleaved_ladders(T_min, T_max, n_per_ladder, n_ladders):
    """Multi-ladder interleaved T schedule. See CE_clamm/pt_mc.py for full doc.

    Replicas laid out CONTIGUOUSLY per ladder (replica i → ladder
    i//n_per_ladder, pos i%n_per_ladder) so parity-based exchange picks
    within-ladder pairs by default; ladder_id mask only blocks boundary
    parity-1 pairs between adjacent ladder blocks.

    n_ladders=1 reduces to make_temperature_ladder semantics exactly.

    Returns:
        temps:      (n_total,) jnp.float32
        betas:      (n_total,) jnp.float32
        ladder_id:  (n_total,) jnp.int32
        ladder_pos: (n_total,) jnp.int32
    """
    n_total = n_per_ladder * n_ladders
    if n_total == 1:
        t_global_np = np.array([T_min], dtype=np.float64)
    else:
        t_global_np = np.geomspace(T_min, T_max, n_total, dtype=np.float64)
    i_np = np.arange(n_total, dtype=np.int64)
    ladder_id_np = (i_np // n_per_ladder).astype(np.int32)
    ladder_pos_np = (i_np % n_per_ladder).astype(np.int32)
    t_index = ladder_id_np.astype(np.int64) + ladder_pos_np.astype(np.int64) * n_ladders
    temps_np = t_global_np[t_index].astype(np.float32)
    betas_np = (1.0 / (_KB_EV_PER_K * temps_np)).astype(np.float32)
    return (jnp.asarray(temps_np), jnp.asarray(betas_np),
            jnp.asarray(ladder_id_np), jnp.asarray(ladder_pos_np))


# ═══════════════════════════════════════════════════════════════════════════════
# Initial state
# ═══════════════════════════════════════════════════════════════════════════════
def make_initial_state(key, gce, mcmc_tables, initial_composition, n_replicas):
    n_atoms = gce.n_atoms
    active_subs = gce.sublattices

    # Derive a numpy RNG from the JAX PRNGKey so initial state respects cfg.seed.
    seed_int = int(np.asarray(key, dtype=np.uint32).view(np.uint32)[0])
    rng = np.random.default_rng(seed_int)

    # Build one template sigma
    sigma_template = np.zeros(n_atoms, dtype=np.int32)

    for sl_idx, sl in enumerate(active_subs):
        indices = np.array(sl['indices'])
        n_sites = len(indices)
        n_species = sl['n_species']
        sl_name = sl['name']

        if sl_name in initial_composition:
            fracs = initial_composition[sl_name]
            if not isinstance(fracs, list):
                fracs = [1.0 - fracs, fracs]
        else:
            fracs = [1.0 / n_species] * n_species

        counts = np.round(np.array(fracs) * n_sites).astype(int)
        diff = n_sites - counts.sum()
        counts[0] += diff

        species_arr = np.concatenate([
            np.full(c, sp_i, dtype=np.int32)
            for sp_i, c in enumerate(counts)
        ])
        rng.shuffle(species_arr)
        sigma_template[indices] = species_arr

    # Replicate and shuffle independently per replica
    sigma_batch = np.tile(sigma_template, (n_replicas, 1))
    for r in range(n_replicas):
        for sl in active_subs:
            indices = np.array(sl['indices'])
            sub_sigma = sigma_batch[r, indices].copy()
            rng.shuffle(sub_sigma)
            sigma_batch[r, indices] = sub_sigma

    sigma_batch = jnp.array(sigma_batch)

    # Initial spins
    if mcmc_tables.has_spin:
        spins_batch = jnp.zeros((n_replicas, n_atoms), dtype=jnp.float32)
        spin_allowed_np = np.array(mcmc_tables.spin_allowed)
        n_spin_states_np = np.array(mcmc_tables.n_spin_states)
        site_to_sl_np = np.array(mcmc_tables.site_to_sl)
        sigma_np = np.array(sigma_batch)

        spins_np = np.zeros((n_replicas, n_atoms), dtype=np.float32)
        for r in range(n_replicas):
            for i in range(n_atoms):
                sl_i = site_to_sl_np[i]
                found = False
                for a_sl_idx, sl in enumerate(active_subs):
                    if any(i == idx for idx in np.array(sl['indices'])):
                        sp_i = sigma_np[r, i]
                        n_s = n_spin_states_np[a_sl_idx, sp_i]
                        if n_s > 1:
                            spins_np[r, i] = spin_allowed_np[
                                a_sl_idx, sp_i, rng.integers(0, n_s)]
                        found = True
                        break

        spins_batch = jnp.array(spins_np)
    else:
        spins_batch = None

    return sigma_batch, spins_batch


# ═══════════════════════════════════════════════════════════════════════════════
# Move proposals
# ═══════════════════════════════════════════════════════════════════════════════
def make_swap_fn(mcmc_tables):
    sub_idx = mcmc_tables.sub_indices
    sub_mask = mcmc_tables.sub_mask
    sub_log_probs = mcmc_tables.sub_log_probs
    n_active = mcmc_tables.n_active_subs

    def propose_swap(key, sigma):
        k_sub, k_s1, k_s2 = jrandom.split(key, 3)

        # Pick sublattice (Gumbel-max categorical)
        sl_choice = jnp.argmax(sub_log_probs + jrandom.gumbel(k_sub, (n_active,)))

        # Get sites on chosen sublattice
        sites = sub_idx[sl_choice]
        mask = sub_mask[sl_choice]
        sub_sigma = sigma[sites]

        # Pick site1 (uniform among valid sites)
        noise1 = jnp.where(mask, jrandom.gumbel(k_s1, sites.shape), -jnp.inf)
        i1_local = jnp.argmax(noise1)
        site1 = sites[i1_local]
        species1 = sigma[site1]

        # Pick site2 (different species, Gumbel-max)
        diff_mask = mask & (sub_sigma != species1)
        noise2 = jnp.where(diff_mask, jrandom.gumbel(k_s2, sites.shape), -jnp.inf)
        i2_local = jnp.argmax(noise2)
        site2 = sites[i2_local]

        # Swap
        sigma_new = sigma.at[site1].set(sigma[site2])
        sigma_new = sigma_new.at[site2].set(species1)

        # valid only if a different-species site exists
        valid = jnp.any(diff_mask)
        # site1, site2 returned for delta-energy mode (full mode ignores them).
        return sigma_new, valid, site1, site2

    return propose_swap


def make_spin_flip_fn(mcmc_tables):
    if not mcmc_tables.has_spin:
        return None

    site_to_sl = mcmc_tables.site_to_sl
    spin_allowed = mcmc_tables.spin_allowed
    n_spin_st = mcmc_tables.n_spin_states
    max_ss = mcmc_tables.max_spin_states
    sub_indices = mcmc_tables.sub_indices
    sub_mask = mcmc_tables.sub_mask
    n_active = mcmc_tables.n_active_subs
    n_atoms = mcmc_tables.n_atoms

    # Build active_sl_index: site → index into active sublattice list, or -1
    # We need this because site_to_sl gives the global sublattice index,
    # but spin_allowed is indexed by active sublattice index
    # For now, assume active sublattice indices are contiguous and match
    # the ordering in gce.sublattices. We build a lookup.

    def propose_spin_flip(key, sigma, spins, active_sl_lookup):
        k_site, k_spin = jrandom.split(key)

        asl = active_sl_lookup
        asl_safe = jnp.maximum(asl, 0)  # clamp -1 → 0 for safe indexing

        is_on_active = asl >= 0
        n_ss = jnp.where(is_on_active, n_spin_st[asl_safe, sigma], 1)
        is_magnetic = is_on_active & (n_ss > 1)

        # Pick a magnetic site (Gumbel-max)
        noise = jnp.where(is_magnetic, jrandom.gumbel(k_site, (n_atoms,)), -jnp.inf)
        site = jnp.argmax(noise)
        valid = jnp.any(is_magnetic)

        # Get allowed spins for this site (safe indexing)
        sl_i = jnp.maximum(asl[site], 0)
        sp_i = sigma[site]
        allowed = spin_allowed[sl_i, sp_i]
        n_allowed = n_spin_st[sl_i, sp_i]
        current_spin = spins[site]

        # Pick new spin != current (Gumbel-max with mask)
        spin_mask = (jnp.arange(max_ss) < n_allowed) & (allowed != current_spin)
        noise_spin = jnp.where(spin_mask, jrandom.gumbel(k_spin, (max_ss,)), -jnp.inf)
        new_spin_idx = jnp.argmax(noise_spin)
        new_spin = allowed[new_spin_idx]

        spins_new = spins.at[site].set(new_spin)
        return spins_new, valid

    return propose_spin_flip


# ═══════════════════════════════════════════════════════════════════════════════
# Replica exchange
# ═══════════════════════════════════════════════════════════════════════════════
def make_exchange_fn(betas, ladder_id=None):
    """Block-diagonal permutation-based replica exchange — no Python for-loop.

    ladder_id=None ⇒ single-ladder behavior (all replicas in one block,
    identical to standard PT). For multi-ladder, ladder_id[i]==ladder_id[i+1]
    is required for the parity-pair (i, i+1) to swap.
    """
    n_replicas = len(betas)
    if ladder_id is None:
        ladder_id_arr = jnp.zeros(n_replicas, dtype=jnp.int32)
    else:
        ladder_id_arr = jnp.asarray(ladder_id, dtype=jnp.int32)

    @jit
    def replica_exchange(key, sigma_batch, spins_batch, E_batch):
        k_parity, k_accept = jrandom.split(key)
        parity = jrandom.randint(k_parity, (), 0, 2)

        i_indices = jnp.arange(0, n_replicas - 1, 2) + parity
        j_indices = i_indices + 1
        valid_mask = j_indices < n_replicas

        i_safe = jnp.where(valid_mask, i_indices, 0)
        j_safe = jnp.where(valid_mask, j_indices, 0)

        same_ladder = ladder_id_arr[i_safe] == ladder_id_arr[j_safe]

        dBeta = jnp.where(valid_mask, betas[j_safe] - betas[i_safe], 0.0)
        dE = jnp.where(valid_mask, E_batch[j_safe] - E_batch[i_safe], 0.0)

        log_accept = dBeta * dE
        u = jrandom.uniform(k_accept, shape=i_indices.shape)
        do_swap = valid_mask & same_ladder & (jnp.log(u) < log_accept)

        new_perm = jnp.arange(n_replicas)
        new_perm = new_perm.at[i_safe].set(jnp.where(do_swap, j_safe, i_safe))
        new_perm = new_perm.at[j_safe].set(jnp.where(do_swap, i_safe, j_safe))

        sigma_batch = sigma_batch[new_perm]
        spins_batch = spins_batch[new_perm]
        E_batch = E_batch[new_perm]

        return sigma_batch, spins_batch, E_batch, jnp.sum(do_swap).astype(jnp.int32)

    return replica_exchange


# ═══════════════════════════════════════════════════════════════════════════════
# MC step (single replica)
# ═══════════════════════════════════════════════════════════════════════════════
def make_mc_step_fn(energy_fn, swap_fn, spin_flip_fn, has_spin, active_sl_lookup,
                    delta_energy_fn=None, energy_mode="full", spin_move_prob=0.0):
    """Build a single-replica MC step function.

    Returns a function with signature:
        (key, sigma, spins, E, beta) → (sigma, spins, E, accepted_swap, accepted_flip)

    spin_move_prob (has_spin only): probability of attempting a single spin flip
    vs a chem swap on a given step (CLAMM-style mutually-exclusive move choice;
    both outcomes are computed and selected with `where` to stay jit-friendly).
    0.0 → pure compositional MC. Ignored when has_spin is False.

    energy_mode (closure, Python-level branch; default "full"):
      - "full"  → ΔE from a fresh full energy_fn eval each step (current
                  behavior; default leaves every code path bit-identical).
      - "delta" → ΔE recomputed only over flat rows touching the swapped sites
                  via delta_energy_fn(i_a, i_b, sigma_old, sigma_new). E is
                  carried as a running sum (drifts ~meV over a long run; verify
                  post-hoc with a full recompute). Compositional/no-spin only —
                  run_pt_mcmc gates has_spin+delta out before reaching here.
    """
    if energy_mode not in ("full", "delta"):
        raise ValueError(f"energy_mode must be 'full' or 'delta'; got {energy_mode!r}")
    if energy_mode == "delta":
        if has_spin:
            raise ValueError(
                "delta energy_mode is compositional-only; spin CEs must use "
                "energy_mode='full' (spin-flip delta not implemented).")
        if delta_energy_fn is None:
            raise ValueError("energy_mode='delta' requires delta_energy_fn")

    if has_spin:
        asl = active_sl_lookup

        def mc_step(key, sigma, spins, E, beta):
            k_choice, k_swap, k_flip, k_a1, k_a2 = jrandom.split(key, 5)
            use_spin = jrandom.uniform(k_choice) < spin_move_prob

            # Chem-swap outcome (spins unchanged)
            sigma_sw, swap_valid, _, _ = swap_fn(k_swap, sigma)
            E_sw = energy_fn(sigma_sw, spins)
            acc_sw = swap_valid & (jnp.log(jrandom.uniform(k_a1)) < -beta * (E_sw - E))
            sigma_a = jnp.where(acc_sw, sigma_sw, sigma)
            E_a = jnp.where(acc_sw, E_sw, E)

            # Spin-flip outcome (sigma unchanged)
            spins_fl, flip_valid = spin_flip_fn(k_flip, sigma, spins, asl)
            E_fl = energy_fn(sigma, spins_fl)
            acc_fl = flip_valid & (jnp.log(jrandom.uniform(k_a2)) < -beta * (E_fl - E))
            spins_b = jnp.where(acc_fl, spins_fl, spins)
            E_b = jnp.where(acc_fl, E_fl, E)

            # Mutually-exclusive move selection (CLAMM-style): spin move leaves
            # sigma untouched; swap move leaves spins untouched.
            sigma_out = jnp.where(use_spin, sigma, sigma_a)
            spins_out = jnp.where(use_spin, spins_b, spins)
            E_out = jnp.where(use_spin, E_b, E_a)
            accept_swap = (~use_spin) & acc_sw
            accept_flip = use_spin & acc_fl

            return sigma_out, spins_out, E_out, accept_swap, accept_flip

    elif energy_mode == "delta":
        # large-cell path (CE/ce_gpu_tiled.py): tables come in as the argument `tp`, the proposal is O(1)
        # (two uniform sites of one sublattice; same species = rejected), the delta reads the two new
        # species as scalars and sigma is updated in place on the two sites — nothing O(N) per step.
        def mc_step(key, sigma, spins, E, beta, tp):
            k_swap, k_a1 = jrandom.split(key)
            i_a, i_b, swap_valid = swap_fn(k_swap, sigma)                    # make_fast_swap
            s_a, s_b = sigma[i_a], sigma[i_b]
            dE = delta_energy_fn(tp, i_a, i_b, sigma, s_b, s_a)
            accept_swap = swap_valid & (jnp.log(jrandom.uniform(k_a1)) < -beta * dE)
            sigma = sigma.at[jnp.array([i_a, i_b])].set(jnp.where(accept_swap, jnp.array([s_b, s_a]), jnp.array([s_a, s_b])))
            E = jnp.where(accept_swap, E + dE, E)   # running sum (carried)
            return sigma, spins, E, accept_swap, jnp.bool_(False)

    else:
        def mc_step(key, sigma, spins, E, beta):
            k_swap, k_a1 = jrandom.split(key)

            sigma_new, swap_valid, _, _ = swap_fn(k_swap, sigma)
            E_new = energy_fn(sigma_new)
            dE = E_new - E
            accept_swap = swap_valid & (jnp.log(jrandom.uniform(k_a1)) < -beta * dE)
            sigma = jnp.where(accept_swap, sigma_new, sigma)
            E = jnp.where(accept_swap, E_new, E)

            return sigma, spins, E, accept_swap, jnp.bool_(False)

    return mc_step


# ═══════════════════════════════════════════════════════════════════════════════
# Main PT-MCMC loop
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class MCMCResult:
    energies: np.ndarray           # (n_samples, n_replicas)
    temperatures: np.ndarray       # (n_replicas,)
    swap_accept_rate: float
    flip_accept_rate: float
    exchange_accept_rate: float
    sigma_final: np.ndarray        # (n_replicas, n_atoms)
    sigma_min: np.ndarray          # (n_atoms,) — argmin across production phase
    E_min: float                   # lowest E observed in the production phase
    spins_final: Optional[np.ndarray]
    wall_time: float


def run_pt_mcmc(gce, cfg):
    """Run PT-MCMC simulation.

    Inner loop is a ``jax.lax.scan`` of length ``cfg.chunk_size``, JIT-
    compiled.  Replica exchange is fired conditionally inside the scan
    using the global step counter, so it stays correct across chunk
    boundaries.  The lowest-energy ``sigma`` over the production phase
    is tracked inside the scan carry (no host-side trajectory needed).
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    has_spin = bool(cfg.spin_orbits_pkl) and bool(cfg.spin_states)

    energy_mode = getattr(cfg, "energy_mode", "full")
    if energy_mode not in ("full", "delta"):
        raise ValueError(f"cfg.energy_mode must be 'full' or 'delta'; got {energy_mode!r}")
    if energy_mode == "delta" and has_spin:
        raise ValueError(
            "energy_mode='delta' supports compositional (no-spin) icet CEs only; "
            "spin-flip delta is not implemented. Use energy_mode='full' for spin CEs.")

    if has_spin and not gce.has_spin:
        gce.load_spin(cfg.spin_orbits_pkl)

    mcmc_tables = build_mcmc_tables(gce, cfg.spin_states if has_spin else None)
    n_ladders_local = max(int(cfg.n_ladders), 1)
    n_per_ladder = int(cfg.n_replicas)
    n_total = n_per_ladder * n_ladders_local
    T_grid = getattr(cfg, "T_grid", None)
    if T_grid is not None:
        if n_ladders_local != 1:
            raise ValueError("cfg.T_grid is one ladder; n_ladders must be 1")
        T_grid = np.asarray(T_grid, dtype=np.float64).reshape(-1)
        n_per_ladder = n_total = int(len(T_grid))
        temps = jnp.array(T_grid, dtype=jnp.float32)                        # same casts as make_temperature_ladder
        betas = jnp.array(1.0 / (_KB_EV_PER_K * T_grid), dtype=jnp.float32)
        ladder_id  = jnp.zeros(n_per_ladder, dtype=jnp.int32)
        ladder_pos = jnp.arange(n_per_ladder, dtype=jnp.int32)
    elif n_ladders_local == 1:
        temps, betas = make_temperature_ladder(cfg.T_min, cfg.T_max, n_per_ladder)
        ladder_id  = jnp.zeros(n_per_ladder, dtype=jnp.int32)
        ladder_pos = jnp.arange(n_per_ladder, dtype=jnp.int32)
    else:
        temps, betas, ladder_id, ladder_pos = _build_interleaved_ladders(
            cfg.T_min, cfg.T_max, n_per_ladder, n_ladders_local)

    chunk_size = int(cfg.chunk_size)
    sample_interval = int(cfg.sample_interval)
    exch_interval = int(cfg.exchange_interval)
    if n_total > 1 and exch_interval % chunk_size != 0:
        import math
        if exch_interval < chunk_size:                       # frequent exchanges (small cells): shorten the chunk
            new_chunk = exch_interval
            if new_chunk % sample_interval != 0: sample_interval = math.gcd(new_chunk, sample_interval)
            print(f"  chunk_size {chunk_size} -> {new_chunk} (= exchange interval) so exchanges fall on chunk boundaries; sample_interval {sample_interval}", flush=True)
            chunk_size = new_chunk
        else:                                                # rare exchanges (large cells): round the interval to whole chunks
            new_ex = int(round(exch_interval / chunk_size)) * chunk_size
            print(f"  exchange_interval {exch_interval} -> {new_ex} (whole chunks of {chunk_size})", flush=True); exch_interval = new_ex
    n_replicas_local = n_total
    n_production = cfg.n_steps - cfg.n_equilib

    # Validity checks for chunk arithmetic — keep behaviour deterministic
    if cfg.n_equilib % chunk_size != 0:
        raise ValueError(
            f"n_equilib ({cfg.n_equilib}) must be a multiple of "
            f"chunk_size ({chunk_size})")
    if n_production % chunk_size != 0:
        raise ValueError(
            f"(n_steps − n_equilib) = {n_production} must be a multiple "
            f"of chunk_size ({chunk_size})")
    if chunk_size % sample_interval != 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be a multiple of "
            f"sample_interval ({sample_interval})")

    if n_ladders_local == 1:
        print(f"PT-MCMC: {n_per_ladder} replicas, "
              f"T=[{float(jnp.min(temps)):.0f}, {float(jnp.max(temps)):.0f}] K" + (" (explicit list)" if T_grid is not None else ""))
    else:
        print(f"PT-MCMC: Multi-ladder = {n_ladders_local} ladders × "
              f"{n_per_ladder} replicas = {n_total} total, "
              f"T_global=[{cfg.T_min:.0f}, {cfg.T_max:.0f}] K "
              f"(block-diagonal within ladders)")
    print(f"  n_steps={cfg.n_steps:,}, n_equilib={cfg.n_equilib:,}, "
          f"chunk_size={chunk_size:,}")
    print(f"  spin: {'yes' if has_spin else 'no'}"
          f"{f' (spin_move_prob={cfg.spin_move_prob})' if has_spin else ''}, "
          f"exchange_interval={exch_interval}, "
          f"sample_interval={sample_interval}, "
          f"energy_mode={energy_mode}")

    # Sublattice / swap_pairs report + (optional) consistency check.
    sl_species = [sl['chemical_symbols'] for sl in gce.sublattices]
    print(f"  sublattices: {sl_species}")
    if cfg.swap_pairs:
        sl_sets = [set(sp) for sp in sl_species]
        bad = [(a, b) for (a, b) in cfg.swap_pairs
               if not any({a, b} <= s for s in sl_sets)]
        if bad:
            raise ValueError(
                f"swap_pairs {bad} do not lie within any single icet sublattice. "
                f"pt_mcmc_ce's swap_fn enforces same-sublattice swaps only "
                f"(sublattices: {sl_species}). Cross-sublattice pairs are not "
                f"physically permitted by the active sublattice structure.")
        print(f"  swap_pairs (verified within sublattices): {cfg.swap_pairs}")

    # active_sl_lookup: site → active sublattice index (or -1 for inactive)
    active_subs = gce.sublattices
    asl_np = np.zeros(gce.n_atoms, dtype=np.int32)
    mask_np = np.zeros(gce.n_atoms, dtype=bool)
    for a_idx, sl in enumerate(active_subs):
        idx = np.asarray(sl['indices']); asl_np[idx] = a_idx; mask_np[idx] = True
    asl_np[~mask_np] = -1
    active_sl_lookup = jnp.array(asl_np)

    if energy_mode == "delta":
        if getattr(gce, "implicit", False):                 # template-only tables: table-free O(1) proposal
            from .ce_gpu_implicit import make_fast_swap_implicit
            swap_fn = make_fast_swap_implicit(gce.tp)
        else:
            from .ce_gpu_tiled import make_fast_swap
            swap_fn = make_fast_swap(mcmc_tables)
        delta_energy_fn = gce.delta_energy_a; tp = gce.tp
    else:
        swap_fn = make_swap_fn(mcmc_tables); delta_energy_fn = None; tp = None
    spin_flip_fn = make_spin_flip_fn(mcmc_tables) if has_spin else None
    spin_move_prob = float(getattr(cfg, "spin_move_prob", 0.0))
    mc_step = make_mc_step_fn(gce.energy, swap_fn, spin_flip_fn,
                                has_spin, active_sl_lookup,
                                delta_energy_fn=delta_energy_fn,
                                energy_mode=energy_mode,
                                spin_move_prob=spin_move_prob)
    mc_step_batched = jit(vmap(mc_step, in_axes=(0, 0, 0, 0, 0, None))) if energy_mode == "delta" else jit(vmap(mc_step, in_axes=(0, 0, 0, 0, 0)))
    # PT exchange as a TEMPERATURE swap: replicas stay in place; rep_of_slot[t] = replica holding temperature slot t,
    # beta_r[r] = beta of replica r.  Nothing O(N) moves per step (the previous configuration swap inside lax.cond copied
    # the whole sigma batch every step: 128 MB per step at 1M atoms x 32 replicas).
    ladder_id_arr = jnp.zeros(n_total, dtype=jnp.int32) if n_ladders_local == 1 else jnp.asarray(ladder_id, dtype=jnp.int32)
    betas_slot = jnp.asarray(betas, jnp.float32)
    # host copies of the small per-slot arrays, taken now: the end of the run must not need the device for them.
    # (1M-atom ladders deadlocked at the very end -- every thread in futex_wait, right after the last print -- inside
    #  the first device->host conversion after the loop; the energies must reach disk without touching the GPU.)
    temps_np = np.asarray(temps); ladder_id_np = np.asarray(ladder_id); ladder_pos_np = np.asarray(ladder_pos)
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)   # `kill -USR1 <pid>` prints every thread's Python stack
    except (AttributeError, ValueError, RuntimeError):
        pass

    def exchange_slots(key, E_b, rep_of_slot):
        k_parity, k_accept = jrandom.split(key)
        parity = jrandom.randint(k_parity, (), 0, 2)
        i_idx = jnp.arange(0, n_total - 1, 2) + parity; j_idx = i_idx + 1; valid = j_idx < n_total
        i_s = jnp.where(valid, i_idx, 0); j_s = jnp.where(valid, j_idx, 0)
        same = ladder_id_arr[i_s] == ladder_id_arr[j_s]
        E_slot = E_b[rep_of_slot]
        log_acc = jnp.where(valid, (betas_slot[j_s] - betas_slot[i_s]) * (E_slot[j_s] - E_slot[i_s]), 0.0)
        do_swap = valid & same & (jnp.log(jrandom.uniform(k_accept, shape=i_idx.shape)) < log_acc)
        new_perm = jnp.arange(n_total).at[i_s].set(jnp.where(do_swap, j_s, i_s)).at[j_s].set(jnp.where(do_swap, i_s, j_s))
        rep_of_slot = rep_of_slot[new_perm]                                      # slot t now holds the replica that was in slot new_perm[t]
        slot_of_rep = jnp.zeros(n_total, jnp.int32).at[rep_of_slot].set(jnp.arange(n_total, dtype=jnp.int32))
        return rep_of_slot, betas_slot[slot_of_rep], jnp.sum(do_swap).astype(jnp.int32)

    # ── Build the lax.scan body ──
    # State: (sigma_batch, spins_batch, E_batch, sigma_min, E_min,
    #         swap_cum, flip_cum, ex_cum, step_count)
    def micro(state, key, tp_):
        (sigma_b, spins_b, E_b,
         sigma_min, E_min,
         swap_cum, flip_cum, ex_cum, step, rep_of_slot, beta_r) = state

        k_step, k_ex = jrandom.split(key)
        # per-SLOT random keys handed to the replica holding that slot: bit-identical trajectories to the old
        # configuration-swap driver (keys were indexed by slot there), so the two schemes can be compared exactly
        slot_of_rep = jnp.zeros(n_total, jnp.int32).at[rep_of_slot].set(jnp.arange(n_total, dtype=jnp.int32))
        keys = jrandom.split(k_step, n_replicas_local)[slot_of_rep]
        if energy_mode == "delta":
            sigma_b, spins_b, E_b, acc_swap, acc_flip = mc_step_batched(keys, sigma_b, spins_b, E_b, beta_r, tp_)
        else:
            sigma_b, spins_b, E_b, acc_swap, acc_flip = mc_step_batched(keys, sigma_b, spins_b, E_b, beta_r)

        # the exchange is applied BETWEEN chunks (run_chunk below): a lax.cond inside the scan cost 2x the whole
        # step (bench_pt_components: 534k -> 974k steps/s at 1M when removed).  chunk_size is made to divide
        # exchange_interval so the exchange timing is unchanged.
        n_ex = jnp.int32(0)

        # Running min of the energy per step (scalar).  sigma_min is NOT copied per step (an O(N) copy per step cost
        # 8x at 10M atoms); run_chunk refreshes it from the chunk's final configuration when the chunk lowered E_min.
        cE = jnp.min(E_b)
        E_min = jnp.where(cE < E_min, cE, E_min)

        new_state = (sigma_b, spins_b, E_b,
                      sigma_min, E_min,
                      swap_cum + jnp.sum(acc_swap).astype(swap_cum.dtype),
                      flip_cum + jnp.sum(acc_flip).astype(flip_cum.dtype),
                      ex_cum + n_ex.astype(ex_cum.dtype),
                      step + 1, rep_of_slot, beta_r)
        return new_state, E_b[rep_of_slot]                                 # energies ordered by temperature slot (as before)

    @jit
    def run_chunk(state, key, tp_):
        keys = jrandom.split(key, chunk_size)
        k_ex = jrandom.split(keys[-1])[1]            # the exchange key the old per-step driver used at this step (bit-exact)
        E_min_before = state[4]
        final_state, E_traj = jax.lax.scan(lambda st, k: micro(st, k, tp_), state, keys)
        sb_, sp_, Eb_, smin_, Emin_, a_, b_, c_, step_, ros_, br_ = final_state
        do_ex = (step_ % exch_interval) == 0                                        # chunk boundary that is an exchange step
        ros_, br_, n_ex = jax.lax.cond(do_ex, lambda: exchange_slots(k_ex, Eb_, ros_), lambda: (ros_, br_, jnp.int32(0)))
        final_state = (sb_, sp_, Eb_, smin_, Emin_, a_, b_, c_ + n_ex.astype(c_.dtype), step_, ros_, br_)
        # the last sample of the chunk was recorded before the exchange: re-order it by the post-exchange slots
        # (the old per-step driver recorded it after the exchange) -> energies per temperature bit-identical
        E_traj = E_traj.at[-1].set(jnp.where(do_ex, Eb_[ros_], E_traj[-1]))
        # sigma_min: the lowest-energy replica at the END of a chunk that improved E_min (chunk-granular, O(N) once per chunk)
        sigma_b, spins_b, E_b, sigma_min, E_min, a, b, c, d, ros, br = final_state
        improved = E_min < E_min_before
        sigma_min = jnp.where(improved, sigma_b[jnp.argmin(E_b)], sigma_min)
        final_state = (sigma_b, spins_b, E_b, sigma_min, E_min, a, b, c, d, ros, br)
        # Subsample on device — transfer only sample_interval-th samples
        E_sub = E_traj[sample_interval - 1::sample_interval]
        return final_state, E_sub

    # ── Initial state ──
    key = jrandom.PRNGKey(cfg.seed)
    key, k_init = jrandom.split(key)
    if cfg.initial_sigma is not None:
        init_sigma = np.asarray(cfg.initial_sigma, dtype=np.int32)
        if init_sigma.shape == (gce.n_atoms,):                              # one configuration for every replica
            sigma_batch = jnp.broadcast_to(jnp.asarray(init_sigma),
                                            (n_total, gce.n_atoms))
        elif init_sigma.shape == (n_total, gce.n_atoms):                    # one configuration per temperature slot
            sigma_batch = jnp.asarray(init_sigma)
        else:
            raise ValueError(
                f"initial_sigma shape {init_sigma.shape} must be (n_atoms,)=({gce.n_atoms},) "
                f"or (n_replicas, n_atoms)=({n_total}, {gce.n_atoms})")
        spins_batch = None
        print(f"  initial sigma: from caller-provided array (shape={init_sigma.shape})",
              flush=True)
    else:
        sigma_batch, spins_batch = make_initial_state(
            k_init, gce, mcmc_tables, cfg.initial_composition, n_total)
    if spins_batch is None:                                # spin-free CE: width-1 placeholder — the no-spin mc_step
        spins_batch = jnp.zeros((n_total, 1), jnp.float32) # branches pass `spins` through untouched, so carrying (R, n_atoms)
                                                           # zeros would only waste 128 MB per state and per checkpoint at 1M
    if has_spin:
        E_batch = gce.batched_energy(sigma_batch, spins_batch)
    elif tp is not None:                                   # delta mode: per-replica full energy, O(rows) memory (no vmap over replicas)
        E_batch = jax.lax.map(lambda s_: gce.energy_a(tp, s_), sigma_batch)
    else:
        E_batch = gce.batched_energy(sigma_batch)
    cnt = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32          # 2e8 steps x 32 replicas overflow int32
    state = (sigma_batch, spins_batch, E_batch,
              sigma_batch[0], E_batch[0],
              jnp.zeros((), cnt), jnp.zeros((), cnt), jnp.zeros((), cnt),
              jnp.int32(0), jnp.arange(n_total, dtype=jnp.int32), betas_slot)

    n_chunks_eq = cfg.n_equilib // chunk_size
    n_chunks_prod = n_production // chunk_size
    n_chunks_tot = n_chunks_eq + n_chunks_prod
    samples_per_chunk = chunk_size // sample_interval

    print(f"\nRunning {cfg.n_steps:,} steps  "
          f"({n_chunks_eq} equilib + {n_chunks_prod} production chunks)")
    t0 = time.time()
    progress_every = max(1, n_chunks_tot // 50)

    # ── Production state / resume ──
    # Read before equilibration on purpose: a checkpoint is written only during production, so a resumed run is already
    # past equilibration and must not repeat it (the old order re-ran every equilibration chunk and then threw the result
    # away when it loaded the checkpoint -- pure waste on every preemption).
    energy_samples = np.zeros(
        (n_chunks_prod * samples_per_chunk, n_total),
        dtype=np.float64)                                  # total energies of large cells exceed float32 resolution
    si = 0; c0 = 0
    ck_every = int(getattr(cfg, "checkpoint_every", 0) or 0); ck_path = os.path.join(cfg.output_dir, "checkpoint.npz")
    if ck_every and os.path.exists(ck_path):               # resume (preemption / time limit): state + samples so far
        ck = np.load(ck_path)
        want = dict(n_steps=cfg.n_steps, chunk=chunk_size, n_total=n_total, n_atoms=gce.n_atoms, sample_interval=sample_interval, seed=cfg.seed)
        have = {k: int(ck["cfg_" + k]) for k in want}
        if "cfg_run_id" in ck.files:                                      # CE parameters + starting configuration (caller-supplied)
            want["run_id"] = str(getattr(cfg, "run_id", "") or ""); have["run_id"] = str(ck["cfg_run_id"])
        else:                                                             # written before the fingerprint existed: cannot check it
            print(f"  warning: {ck_path} predates the CE fingerprint — resuming without verifying the CE / starting structures", flush=True)
        if not np.allclose(np.asarray(ck["cfg_temps"]), np.asarray(temps), rtol=0, atol=1e-6): have["temps"] = "differ"; want["temps"] = "same"
        if have != want:
            raise RuntimeError(f"checkpoint {ck_path} belongs to a different run (checkpoint {have} vs this run {want}); delete it to start fresh")
        c0 = int(ck["chunk"]); si = int(ck["si"]); energy_samples[:si] = ck["energy_samples"]
        key = jnp.asarray(ck["key"]); cnt = state[5].dtype
        state = (jnp.asarray(ck["sigma"]), jnp.asarray(ck["spins"]) if has_spin else spins_batch, jnp.asarray(ck["E"]),
                 jnp.asarray(ck["sigma_min"]), jnp.asarray(ck["E_min"]), jnp.asarray(ck["swap_cum"], cnt), jnp.asarray(ck["flip_cum"], cnt),
                 jnp.asarray(ck["ex_cum"], cnt), jnp.asarray(ck["step"], jnp.int32), jnp.asarray(ck["rep_of_slot"], jnp.int32), jnp.asarray(ck["beta_r"], jnp.float32))
        print(f"  resumed from {ck_path}: production chunk {c0}/{n_chunks_prod}, {si} samples", flush=True)
    chunks_this_run = (n_chunks_eq if c0 == 0 else 0) + (n_chunks_prod - c0)   # what THIS process still has to run (ETA base)

    # ── Equilibration (skipped on a resume: the checkpoint state is past it) ──
    if c0 == 0:
        for c in range(n_chunks_eq):
            key, k = jrandom.split(key)
            state, _ = run_chunk(state, k, tp)
            state[2].block_until_ready()
            if (c + 1) % progress_every == 0 or c + 1 == n_chunks_eq:
                elapsed = time.time() - t0
                eta = elapsed / (c + 1) * (chunks_this_run - (c + 1))
                print(f"  [equil  {c+1:>5d}/{n_chunks_tot}]  "
                      f"E_min(curr)={float(state[2].min()):>10.2f}  "
                      f"elapsed={elapsed:>7.1f}s  ETA={eta:>7.1f}s")

        # Reset min tracker so reported E_min reflects production phase only
        sigma_b, spins_b, E_b, _, _, swap_c, flip_c, ex_c, step_c, ros_, br_ = state
        idx0 = jnp.argmin(E_b)
        state = (sigma_b, spins_b, E_b, sigma_b[idx0], E_b[idx0],
                  swap_c, flip_c, ex_c, step_c, ros_, br_)
        if n_chunks_eq > 0:
            print(f"  → Equilibration complete; reset E_min for production.")

    # ── Production ──
    for c in range(c0, n_chunks_prod):
        key, k = jrandom.split(key)
        state, E_sub = run_chunk(state, k, tp)
        state[2].block_until_ready()
        E_sub_np = np.asarray(E_sub)
        n_take = E_sub_np.shape[0]
        energy_samples[si:si + n_take] = E_sub_np
        si += n_take
        if ck_every and (c + 1) % ck_every == 0 and c + 1 < n_chunks_prod:
            sg, sp, Eb, smin, Emin, sw, fl, ex, st, ros, br = state
            np.savez(ck_path + ".tmp.npz", chunk=c + 1, si=si, energy_samples=energy_samples[:si], key=np.asarray(key), sigma=np.asarray(sg),
                     rep_of_slot=np.asarray(ros), beta_r=np.asarray(br),
                     cfg_n_steps=cfg.n_steps, cfg_chunk=chunk_size, cfg_n_total=n_total, cfg_n_atoms=gce.n_atoms, cfg_sample_interval=sample_interval, cfg_seed=cfg.seed, cfg_temps=np.asarray(temps),
                     cfg_run_id=str(getattr(cfg, "run_id", "") or ""),
                     spins=(np.zeros(0) if sp is None else np.asarray(sp)), E=np.asarray(Eb), sigma_min=np.asarray(smin), E_min=np.asarray(Emin),
                     swap_cum=np.asarray(sw), flip_cum=np.asarray(fl), ex_cum=np.asarray(ex), step=np.asarray(st))
            os.replace(ck_path + ".tmp.npz", ck_path)
        global_chunk = n_chunks_eq + c + 1
        if global_chunk % progress_every == 0 or c + 1 == n_chunks_prod:
            elapsed = time.time() - t0
            done = (n_chunks_eq if c0 == 0 else 0) + (c + 1 - c0)              # chunks run by THIS process, not since step 0
            eta = elapsed / done * (chunks_this_run - done) if done else 0.0
            print(f"  [prod   {global_chunk:>5d}/{n_chunks_tot}]  "
                  f"E_min(run)={float(state[4]):>10.4f}  "
                  f"elapsed={elapsed:>7.1f}s  ETA={eta:>7.1f}s")

    wall_time = time.time() - t0
    try:                                                   # if the tail hangs again, dump every thread's stack every 15 min
        faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
    except (AttributeError, ValueError, RuntimeError):
        pass
    (sigma_batch, spins_batch, E_batch,
     sigma_min_arr, E_min_val,
     swap_cum, flip_cum, ex_cum, _step, rep_of_slot, _beta_r) = state

    n_swap_attempts = cfg.n_steps * n_total
    # Block-diagonal: each step proposes (n_per_ladder//2) within-ladder pairs
    # per ladder × n_ladders ladders. (Cross-ladder boundary pairs are masked
    # out; they contribute to denominator only when single-ladder.)
    pairs_per_step = (n_per_ladder // 2) * n_ladders_local
    n_ex_attempts = (cfg.n_steps // exch_interval) * pairs_per_step
    swap_accept_rate = float(swap_cum) / max(1, n_swap_attempts)
    flip_accept_rate = (float(flip_cum) / max(1, n_swap_attempts)
                         if has_spin else 0.0)
    exchange_accept_rate = float(ex_cum) / max(1, n_ex_attempts)

    print(f"\nDone in {wall_time:.1f}s")
    print(f"  swap acceptance     : {swap_accept_rate:.4f}")
    if has_spin:
        print(f"  flip acceptance     : {flip_accept_rate:.4f}")
    print(f"  exchange acceptance : {exchange_accept_rate:.4f}")
    E_min_f = float(E_min_val)
    print(f"  E_min (production)  : {E_min_f:.4f}", flush=True)

    # The energies are the physics; the configurations are 25 x N int32 and everything below them is O(N).  Write the
    # energies first, so a run killed in the O(N) tail (a 1M x 25 job lost 19.6 h of sampling that way) still lands its
    # temperature curve on disk.
    np.savez(os.path.join(cfg.output_dir, 'energies.npz'),
             energies=energy_samples[:si], temperatures=temps_np, ladder_id=ladder_id_np,
             n_atoms=np.int32(gce.n_atoms), swap_accept_rate=swap_accept_rate, exchange_accept_rate=exchange_accept_rate,
             flip_accept_rate=flip_accept_rate, E_min=E_min_f, wall_time=wall_time)
    print(f"  wrote {os.path.join(cfg.output_dir, 'energies.npz')} ({si} samples x {n_total} T)", flush=True)

    # configurations: ONE device->host copy each, then reorder by temperature slot in numpy (no new device kernels)
    t_h = time.time()
    ros = np.asarray(rep_of_slot)
    sigma_batch = np.asarray(sigma_batch)[ros]; E_batch = np.asarray(E_batch)[ros]
    if spins_batch is not None: spins_batch = np.asarray(spins_batch)[ros]
    sigma_min_arr = np.asarray(sigma_min_arr)
    print(f"  configurations to host in {time.time()-t_h:.1f}s", flush=True)

    if getattr(gce, "inv_perm", None) is not None:        # implicit tables run in canonical site order -> back to the input order
        t_g = time.time(); ip = np.asarray(gce.inv_perm)
        sigma_batch = np.ascontiguousarray(sigma_batch)[:, ip]; sigma_min_arr = sigma_min_arr[ip]
        print(f"  reordered to the input atom order in {time.time()-t_g:.1f}s", flush=True)
    result = MCMCResult(
        energies=energy_samples[:si],
        temperatures=np.array(temps_np),
        swap_accept_rate=swap_accept_rate,
        flip_accept_rate=flip_accept_rate,
        exchange_accept_rate=exchange_accept_rate,
        sigma_final=np.array(sigma_batch),
        sigma_min=np.array(sigma_min_arr),
        E_min=E_min_f,
        spins_final=np.array(spins_batch) if has_spin else None,
        wall_time=wall_time,
    )

    np.savez(os.path.join(cfg.output_dir, 'mcmc_result.npz'),
              energies=result.energies,
              temperatures=result.temperatures,
              ladder_id=ladder_id_np,
              ladder_pos=ladder_pos_np,
              n_ladders=np.int32(n_ladders_local),
              sigma_final=result.sigma_final,
              sigma_min=result.sigma_min,
              E_min=result.E_min,
              spins_final=(result.spins_final
                           if result.spins_final is not None
                           else np.array([])),
              swap_accept_rate=result.swap_accept_rate,
              flip_accept_rate=result.flip_accept_rate,
              exchange_accept_rate=result.exchange_accept_rate,
              wall_time=result.wall_time)
    print(f"  wrote {os.path.join(cfg.output_dir, 'mcmc_result.npz')}", flush=True)
    try:
        faulthandler.cancel_dump_traceback_later()
    except (AttributeError, ValueError, RuntimeError):
        pass
    if ck_every and os.path.exists(ck_path):
        os.remove(ck_path)      # only now: a run killed between the last chunk and this line must still be resumable
                                # (a finished run leaves no checkpoint, so a rerun in the same dir starts fresh)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from icet import ClusterExpansion
    from ase.build import make_supercell

    CE_PATH = os.environ.get('CE_PATH', 'models/stfo_comp_test.ce')
    SUPERCELL_MATRIX = [[0, 2, 2], [2, 0, 2], [2, 2, 0]]

    cfg = MCMCConfig(
        ce_path=CE_PATH,
        supercell_matrix=SUPERCELL_MATRIX,
        T_min=300.0, T_max=2000.0, n_replicas=32,
        n_steps=10000, n_equilib=2000,
        sample_interval=10, exchange_interval=5,
        initial_composition={'A': [0.5, 0.5], 'B': [0.75, 0.25]},
    )

    ce = ClusterExpansion.read(cfg.ce_path)
    prim = ce._cluster_space.primitive_structure
    supercell = make_supercell(prim, np.array(cfg.supercell_matrix))
    gce = GPUClusterExpansion(ce, supercell)
    print(gce)

    result = run_pt_mcmc(gce, cfg)
    print(f"\nEnergy samples shape: {result.energies.shape}")
