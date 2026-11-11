"""JAX/JIT vectorized icet (orthonormal) CE single-T + multi-T MC.

Mirrors CE_clamm/gpu_clamm.py:
    - run_gpu_mc(...): single-T loop, jit + lax.scan, GPU-accelerated
    - run_multiT_gpu_mc(...): in-memory anneal chain (no cif round-trip)

icet specifics vs CLAMM:
  - Energy is gce.energy(sigma) — one positional arg (compositional CE, no spin).
  - Swap proposal uses pt_mcmc_ce.make_swap_fn (sublattice-aware): pick a
    sublattice via Gumbel-max over sub_log_probs, then two distinct-species
    sites within it. swap_mode {uniform, weighted} controls sub_log_probs
    construction.
  - sigma encoding via gce.encode(atoms); CLAMM's species_order pattern not used.

Multi-T contract (matches CLAMM run_multiT_gpu_mc):
  atoms / sigma chained in-memory between T's; cif written as snapshot only.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
from neuralce.ce.pt_mcmc_ce import build_mcmc_tables, make_swap_fn

KB = 8.617333e-5


# ──────────────────────────────────────────────────────────────────────
# Sublattice log-probs for swap_mode {uniform, weighted}
# ──────────────────────────────────────────────────────────────────────
def _build_sub_log_probs(mcmc_tables, gce, swap_mode: str):
    """Return jnp array of shape (n_active_subs,) — log-probabilities for
    sublattice selection in make_swap_fn.

    uniform  → log(1/n_active) for every sublattice (Gumbel-max picks uniformly).
    weighted → log(n_sites_in_sublattice / n_total_active_sites). Larger
               sublattices proposed more often, matching the explicit/icet CPU
               'weighted' convention.
    """
    if swap_mode not in ("uniform", "weighted"):
        raise ValueError(f"swap_mode must be 'uniform' or 'weighted'; got {swap_mode!r}")
    n_active = mcmc_tables.n_active_subs
    if swap_mode == "uniform":
        return jnp.full((n_active,), -float(np.log(n_active)), dtype=jnp.float32)
    sizes = np.array([len(np.array(sl['indices'])) for sl in gce.sublattices],
                     dtype=np.float64)
    total = float(sizes.sum())
    return jnp.asarray(np.log(sizes / total), dtype=jnp.float32)


# ──────────────────────────────────────────────────────────────────────
# Single MC step (uses make_swap_fn for sublattice-aware swap)
# ──────────────────────────────────────────────────────────────────────
def make_step_fn(gce, mcmc_tables, kBT: float, energy_mode: str = "full"):
    """Build a JIT single-T MC step. mcmc_tables.sub_log_probs should already
    be set per swap_mode (see _build_sub_log_probs).

    energy_mode='delta': ΔE from the instance-local recompute
    (classical_ce_gpu.make_delta_energy_fn) — E carried as e_curr+ΔE, so the
    caller must refresh with a full energy periodically (chunk boundaries)."""
    energy_fn = gce.energy   # already jit-compiled inside GPUClusterExpansion
    swap_fn = make_swap_fn(mcmc_tables)
    if energy_mode not in ("full", "delta"):
        raise ValueError(f"energy_mode must be full|delta; got {energy_mode!r}")
    delta_fn = gce.delta_energy if energy_mode == "delta" else None

    @jax.jit
    def step(key, sigma, e_curr):
        k_swap, k_acc = jrandom.split(key, 2)
        sigma_new, valid, site1, site2 = swap_fn(k_swap, sigma)
        if delta_fn is None:
            e_new = energy_fn(sigma_new)
            de = e_new - e_curr
        else:
            de = delta_fn(site1, site2, sigma, sigma_new)
            e_new = e_curr + de
        u = jrandom.uniform(k_acc)
        accept = valid & ((de <= 0) | (u < jnp.exp(-de / kBT)))
        sigma_out = jnp.where(accept, sigma_new, sigma)
        e_out = jnp.where(accept, e_new, e_curr)
        return sigma_out, e_out, accept, de

    return step


def make_run_chunk(step_fn):
    """Compile N-step trajectory as one lax.scan (no Python per-step overhead)."""
    def body(carry, _):
        key, sigma, e = carry
        key, sub = jrandom.split(key)
        sigma, e, accept, de = step_fn(sub, sigma, e)
        return (key, sigma, e), (de, accept)

    @partial(jax.jit, static_argnums=(3,))
    def run_chunk(key, sigma, e_init, n_steps):
        (key_out, sigma_out, e_out), (de_traj, acc_traj) = jax.lax.scan(
            body, (key, sigma, e_init), None, length=n_steps)
        return sigma_out, e_out, key_out, de_traj, acc_traj
    return run_chunk


# ──────────────────────────────────────────────────────────────────────
# cif → CE-primitive supercell setup
# ──────────────────────────────────────────────────────────────────────
def _setup(ce_path, init_cif, init_atoms=None, init_sigma=None,
           ideal_map=None, inert_species=None, symprec=1e-2,
           ref_cif=None, fractional_position_tolerance=None):
    """Build gce + mcmc_tables + sigma from cif (or from in-memory atoms/sigma).

    Mirrors CE_clamm/gpu_clamm._setup. If init_atoms+init_sigma are provided
    (multi-T chaining), skips cif read/mapping and uses them directly.
    """
    from icet import ClusterExpansion
    from ase.io import read as ase_read
    from icet.tools import map_structure_to_reference
    from neuralce.ce.primitive_no_idealize import detect_primitive_no_idealize
    from .mcmc import _rebuild_as_clean_supercell

    ce = ClusterExpansion.read(ce_path)
    ce_primitive = ce._cluster_space.primitive_structure

    if init_atoms is not None and init_sigma is not None:
        atoms = init_atoms
        sigma = init_sigma
    else:
        raw = ase_read(init_cif)
        prim_src = ref_cif if ref_cif else init_cif
        if ideal_map or inert_species:
            primitive = detect_primitive_no_idealize(
                prim_src, ideal_map or {}, symprec)
            mapped, _ = map_structure_to_reference(
                raw, primitive,
                inert_species=list(inert_species or []),
                assume_no_cell_relaxation=False)
            print(f"  mapped to primitive  ideal_map={ideal_map}  "
                  f"inert={list(inert_species or [])}  prim_src={prim_src}",
                  flush=True)
        else:
            mapped = raw
        atoms = _rebuild_as_clean_supercell(mapped, ce_primitive)
        sigma = None  # will encode after gce built

    gce = GPUClusterExpansion(
        ce, atoms,
        fractional_position_tolerance=fractional_position_tolerance)
    if sigma is None:
        sigma = gce.encode(atoms)
    sigma = jnp.asarray(sigma, dtype=jnp.int32)

    mcmc_tables = build_mcmc_tables(gce, None)
    return ce, gce, mcmc_tables, atoms, sigma


# ──────────────────────────────────────────────────────────────────────
# Production entry: single-T
# ──────────────────────────────────────────────────────────────────────
def run_gpu_mc(ce_path: str, init_cif: str, n_steps: int, temperature: float,
               seed: int = 0, log_every: int = 0, swap_mode: str = "uniform",
               ideal_map=None, inert_species=None, symprec: float = 1e-2,
               swap_pairs=None, ref_cif=None,
               fractional_position_tolerance=None,
               init_atoms=None, init_sigma=None,
               energy_mode: str = "full",
               out_dir: str = "."):
    """Single-T icet GPU MC via jit + lax.scan.

    System config (mirrors neuralce.ce.icet_mc.run_pt_mc):
        ideal_map, inert_species, symprec, ref_cif → primitive detection
          + map_structure_to_reference → rebuild as integer supercell of CE prim.
        swap_pairs → verified against icet sublattices (sublattice-uniform swap
          already restricts pairs to same-sublattice different-species).
        fractional_position_tolerance → forwarded to GPUClusterExpansion.

    Chained-T contract: if init_atoms+init_sigma given, skip cif read and use
    in-memory state (used by run_multiT_gpu_mc to avoid cif round-trip).

    swap_mode: "uniform" → all sublattices picked with equal prob.
               "weighted" → ∝ sublattice size.

    Returns dict with final_atoms, final_sigma, final_E, acceptance_ratio,
    n_steps, T, dt_s, sublattices.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    ce, gce, mcmc_tables, atoms, sigma = _setup(
        ce_path, init_cif,
        init_atoms=init_atoms, init_sigma=init_sigma,
        ideal_map=ideal_map, inert_species=inert_species, symprec=symprec,
        ref_cif=ref_cif,
        fractional_position_tolerance=fractional_position_tolerance,
    )

    # Replace mcmc_tables.sub_log_probs per swap_mode.
    sub_log_probs = _build_sub_log_probs(mcmc_tables, gce, swap_mode)
    mcmc_tables.sub_log_probs = sub_log_probs

    sl_species = [sl['chemical_symbols'] for sl in gce.sublattices]
    print(f"  sublattices: {sl_species}, swap_mode={swap_mode}", flush=True)
    if swap_pairs:
        sl_sets = [set(sp) for sp in sl_species]
        bad = [(a, b) for (a, b) in swap_pairs
               if not any({a, b} <= s for s in sl_sets)]
        if bad:
            raise ValueError(
                f"swap_pairs {bad} cross sublattices (sublattices={sl_species}). "
                f"icet sublattice-uniform swap_fn cannot enforce cross-sublattice pairs.")
        print(f"  swap_pairs (verified within sublattices): {swap_pairs}", flush=True)

    energy_fn = gce.energy
    kBT = max(KB * temperature, 1e-9)
    e_curr = energy_fn(sigma)
    print(f"  init E = {float(e_curr):.4f} eV  (T={temperature:.0f} K)", flush=True)

    if energy_mode == "delta":
        print("  energy_mode=delta (instance-local ΔE; full refresh per chunk)",
              flush=True)
    step_fn = make_step_fn(gce, mcmc_tables, kBT, energy_mode=energy_mode)
    run_chunk = make_run_chunk(step_fn)
    key = jrandom.PRNGKey(seed)

    chunk = log_every if log_every else n_steps
    n_chunks = max(n_steps // chunk, 1)
    if n_chunks * chunk != n_steps:
        chunk = n_steps; n_chunks = 1
    n_acc = 0
    t0 = time.time()
    for c in range(n_chunks):
        sigma, e_curr, key, de_t, acc_t = run_chunk(
            key, sigma, e_curr, chunk)
        de_t.block_until_ready()
        acc_arr = np.asarray(acc_t)
        n_acc += int(acc_arr.sum())
        done = (c + 1) * chunk
        ar = n_acc / done
        drift_msg = ""
        if energy_mode == "delta":
            e_exact = energy_fn(sigma)
            drift_msg = f"  drift={float(jnp.abs(e_exact - e_curr)):.2e}"
            e_curr = e_exact                     # anchor: kill f32 ΔE drift
        print(f"  [{done}/{n_steps}] E={float(e_curr):.4f}  AR={ar:.3f}{drift_msg}",
              flush=True)
    dt = time.time() - t0
    print(f"  final E={float(e_curr):.4f}  AR={n_acc/n_steps:.4f}  "
          f"time={dt:.1f}s ({n_steps/dt:.0f} steps/s)", flush=True)

    # Reconstruct atoms with final sigma's species labels.
    sigma_np = np.asarray(sigma)
    sl_species_lists = [sl['chemical_symbols'] for sl in gce.sublattices]
    site_to_sl_np = np.asarray(mcmc_tables.site_to_sl)
    syms_out = []
    inert_set = set(inert_species or [])
    raw_syms = atoms.get_chemical_symbols()
    for i, s in enumerate(sigma_np):
        sl_i = int(site_to_sl_np[i])
        if 0 <= sl_i < len(sl_species_lists):
            syms_out.append(sl_species_lists[sl_i][int(s)])
        else:
            syms_out.append(raw_syms[i])  # inert site, keep original
    atoms_out = atoms.copy()
    atoms_out.set_chemical_symbols(syms_out)

    return {
        "acceptance_ratio": n_acc / n_steps,
        "final_E": float(e_curr),
        "n_steps": n_steps,
        "T": temperature,
        "dt_s": dt,
        "final_atoms": atoms_out,
        "final_sigma": sigma_np,
        "sublattices": sl_species_lists,
    }


# ──────────────────────────────────────────────────────────────────────
# Production entry: multi-T anneal (in-memory chain — no cif round-trip)
# ──────────────────────────────────────────────────────────────────────
def run_multiT_gpu_mc(ce_path, init_cif, temps, n_steps,
                       seed=0, log_every=0, swap_mode="uniform", out_dir=".",
                       ideal_map=None, inert_species=None, symprec=1e-2,
                       swap_pairs=None, ref_cif=None,
                       fractional_position_tolerance=None,
                       energy_mode="full"):
    """Anneal scan over `temps` (high → low typically). atoms/sigma chained
    in-memory between T's; cif/sigma saved per T as snapshots only (never read
    back). Mirrors CE_clamm/gpu_clamm.run_multiT_gpu_mc.
    """
    from ase.io import write as ase_write
    import json

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    results = {}
    cur_atoms = None; cur_sigma = None
    for T in temps:
        print(f"\n=== T = {T:.0f} K ===", flush=True)
        res = run_gpu_mc(
            ce_path=ce_path, init_cif=init_cif,
            n_steps=n_steps, temperature=T, energy_mode=energy_mode,
            seed=seed + int(T), log_every=log_every, swap_mode=swap_mode,
            ideal_map=ideal_map, inert_species=inert_species, symprec=symprec,
            swap_pairs=swap_pairs, ref_cif=ref_cif,
            fractional_position_tolerance=fractional_position_tolerance,
            init_atoms=cur_atoms, init_sigma=cur_sigma,
            out_dir=out_dir,
        )
        cur_atoms = res.pop("final_atoms")
        cur_sigma = res.pop("final_sigma")
        ase_write(str(out / f"final_T{int(T)}.cif"), cur_atoms)
        np.save(out / f"final_sigma_T{int(T)}.npy", cur_sigma)
        results[float(T)] = res
    with open(out / "multiT_result.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {out / 'multiT_result.json'}", flush=True)
    return results
