"""Single-swap MC acceptance ratio for icet (orthonormal CE) — generic over system.

Loads fitted ClusterExpansion (.ce) + init cif, maps to lattice (with optional
ideal_map and inert_species), performs swaps between every (sp_a, sp_b) in
swap_pairs, applies Metropolis at given T. All system-specific configuration
is passed in via args — no module-level hardcoding.
"""
from __future__ import annotations
import argparse
import random
import time
import math
import os
from itertools import combinations

# (platform selection is done by the command line: neuralce-mc --platform)

import numpy as np
from ase.io import read
from icet import ClusterExpansion
from icet.tools import map_structure_to_reference

from neuralce.ce.primitive_no_idealize import detect_primitive_no_idealize


KB = 8.617333e-5  # eV/K


def map_to_lattice(raw_atoms, primitive, inert_species):
    mapped, info = map_structure_to_reference(
        raw_atoms, primitive, inert_species=list(inert_species),
        assume_no_cell_relaxation=False)
    return mapped


def find_swap_indices(symbols: np.ndarray, sp_a: str, sp_b: str):
    idx_a = np.where(symbols == sp_a)[0]
    idx_b = np.where(symbols == sp_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None, None
    return int(np.random.choice(idx_a)), int(np.random.choice(idx_b))


def _propose_pair_swap(syms: np.ndarray, sp_pair):
    """Propose one swap on (sp_a, sp_b) (no accept/reject). Returns (new_syms, valid)."""
    sp_a, sp_b = sp_pair
    i_a, i_b = find_swap_indices(syms, sp_a, sp_b)
    if i_a is None:
        return syms, False
    new_syms = syms.copy()
    new_syms[i_a], new_syms[i_b] = new_syms[i_b], new_syms[i_a]
    return new_syms, True


def _attempt_swap(syms: np.ndarray, atoms_template, sp_pair, ce, e_curr, kBT):
    """Try one swap on given (sp_a, sp_b) pair. Returns (syms_new, e_new, accept, de)."""
    new_syms, valid = _propose_pair_swap(syms, sp_pair)
    if not valid:
        return syms, e_curr, False, 0.0
    atoms_new = atoms_template.copy()
    atoms_new.set_chemical_symbols(new_syms.tolist())
    e_new = float(ce.predict(atoms_new))
    de = e_new - e_curr
    accept = de <= 0 or random.random() < math.exp(-de / kBT)
    if accept:
        return new_syms, e_new, True, de
    return syms, e_curr, False, de


def run_mc(ce_path: str, init_cif: str, n_steps: int, temperature: float,
           seed: int = 0, log_every: int = 0, swap_mode: str = "uniform",
           ref_cif: str | None = None, ideal_map: dict | None = None,
           inert_species=None, swap_pairs=None, symprec: float = 1e-2):
    """Generic single-T MC for icet CE.

    swap_pairs: list of (sp_a, sp_b) tuples. If None, auto-derived as all unique
                pairs of species present in init_cif (excluding inert_species).
    ref_cif: cif used for primitive detection; defaults to init_cif.
    ideal_map: optional species→species map for primitive idealization (e.g.
               {"Fe": "Ti"} for STFO). Default: {} (no idealization).
    inert_species: list of species treated as inert (excluded from swaps).
                   Default: [] (none).
    """
    random.seed(seed)
    np.random.seed(seed)

    if ref_cif is None:
        ref_cif = init_cif
    if ideal_map is None:
        ideal_map = {}
    if inert_species is None:
        inert_species = []

    ce = ClusterExpansion.read(ce_path)
    print(f"loaded icet CE: {len(ce.parameters)} parameters", flush=True)

    primitive = detect_primitive_no_idealize(ref_cif, ideal_map, symprec)
    raw = read(init_cif)
    atoms = map_to_lattice(raw, primitive, inert_species)
    print(f"mapped init: {len(atoms)} atoms", flush=True)

    syms_all = np.array(atoms.get_chemical_symbols())
    if swap_pairs is None:
        # Derive swap_pairs from icet's sublattice structure so we never propose
        # cross-sublattice swaps (e.g. Ti↔O in STFO is physically nonsense).
        # cs.chemical_symbols is a list-of-lists: one entry per primitive site
        # giving allowed species at that site. Sites with identical allowed-
        # species sets form a sublattice; combine within-sublattice only.
        # This is more robust than cs.get_sublattices(atoms) which can fail on
        # midpoint coordinates due to position tolerance.
        cs = ce._cluster_space
        inert_set = set(inert_species or [])
        unique_sublattices = set()
        for site_species in cs.chemical_symbols:
            active_on_site = tuple(sorted(s for s in site_species if s not in inert_set))
            if len(active_on_site) >= 2:
                unique_sublattices.add(active_on_site)
        pair_set = set()
        for sl_species in unique_sublattices:
            for a, b in combinations(sl_species, 2):
                pair_set.add(tuple(sorted([a, b])))
        if not pair_set:
            print(f"  warning: no within-sublattice swap pairs derived from "
                  f"cs.chemical_symbols={cs.chemical_symbols}; falling back to "
                  f"all-active-pair combinations", flush=True)
            active = [s for s in sorted(set(syms_all.tolist())) if s not in inert_set]
            swap_pairs = list(combinations(active, 2))
        else:
            swap_pairs = sorted(pair_set)
    print(f"swap_pairs = {swap_pairs}  inert = {list(inert_species)}", flush=True)

    e_curr = float(ce.predict(atoms))
    print(f"init E = {e_curr:.4f} eV  (T = {temperature:.0f} K), swap_mode={swap_mode}",
          flush=True)

    kBT = max(KB * temperature, 1e-9)
    n_accept = 0
    delta_e_list = []
    t0 = time.time()
    syms = syms_all

    # weighted prob: per-pair weight ∝ (count(a) + count(b)) in the system
    if swap_mode == "weighted":
        pair_weights = np.array([(np.isin(syms, list(p))).sum()
                                 for p in swap_pairs], dtype=float)
        pair_weights /= pair_weights.sum()
    else:
        pair_weights = None

    for step in range(1, n_steps + 1):
        if swap_mode == "all":
            # Namhoon-style joint move: propose ONE swap on every pair, then a single
            # accept/reject on the cumulative ΔE for the joint proposal.
            syms_proposed = syms.copy()
            valid_all = True
            for sp_pair in swap_pairs:
                syms_proposed, valid = _propose_pair_swap(syms_proposed, sp_pair)
                valid_all = valid_all and valid
            if not valid_all:
                delta_e_list.append(0.0)
            else:
                atoms_new = atoms.copy()
                atoms_new.set_chemical_symbols(syms_proposed.tolist())
                e_new = float(ce.predict(atoms_new))
                de = e_new - e_curr
                delta_e_list.append(de)
                if de <= 0 or random.random() < math.exp(-de / kBT):
                    syms = syms_proposed
                    e_curr = e_new
                    atoms.set_chemical_symbols(syms.tolist())
                    n_accept += 1
        else:
            if swap_mode == "weighted":
                idx = int(np.random.choice(len(swap_pairs), p=pair_weights))
                sp_pair = swap_pairs[idx]
            else:  # uniform
                sp_pair = random.choice(swap_pairs)
            syms, e_curr, acc, de = _attempt_swap(syms, atoms, sp_pair, ce, e_curr, kBT)
            delta_e_list.append(de)
            if acc:
                atoms.set_chemical_symbols(syms.tolist())
                n_accept += 1

        if log_every and step % log_every == 0:
            ar = n_accept / step
            print(f"  [{step}/{n_steps}] E={e_curr:.4f}  AR={ar:.3f}  "
                  f"<|ΔE|>={np.mean(np.abs(delta_e_list[-log_every:])):.4f}", flush=True)

    dt = time.time() - t0
    ar = n_accept / n_steps
    print(f"\nfinal E = {e_curr:.4f} eV", flush=True)
    print(f"acceptance ratio = {n_accept}/{n_steps} = {ar:.4f}", flush=True)
    print(f"<ΔE> = {np.mean(delta_e_list):+.4f}  σ(ΔE) = {np.std(delta_e_list):.4f} eV", flush=True)
    print(f"time = {dt:.1f}s ({n_steps/dt:.1f} steps/s)", flush=True)
    return {"acceptance_ratio": ar, "final_E": e_curr, "n_steps": n_steps,
            "T": temperature, "dt_s": dt, "final_atoms": atoms}


def run_multiT_mc(ce_path, init_cif, temps, n_steps, swap_mode="uniform",
                  seed=0, log_every=0, out_dir=".",
                  ref_cif=None, ideal_map=None, inert_species=None,
                  swap_pairs=None, symprec=1e-2):
    """Annealing scan: each T uses previous T's final atoms as init.

    System-config args (ref_cif/ideal_map/inert_species/swap_pairs/symprec)
    are passed through to each run_mc call. None values are auto-defaulted.
    """
    from pathlib import Path
    from ase.io import write as ase_write
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    results = {}
    cur_cif = init_cif
    for T in temps:
        print(f"\n=== T = {T:.0f} K ===", flush=True)
        res = run_mc(ce_path=ce_path, init_cif=cur_cif, n_steps=n_steps,
                     temperature=T, seed=seed + int(T), log_every=log_every,
                     swap_mode=swap_mode,
                     ref_cif=ref_cif, ideal_map=ideal_map,
                     inert_species=inert_species, swap_pairs=swap_pairs,
                     symprec=symprec)
        atoms = res.pop("final_atoms")
        cif_out = out / f"final_T{int(T)}.cif"
        ase_write(str(cif_out), atoms)
        cur_cif = str(cif_out)
        results[float(T)] = res
    import json
    with open(out / "multiT_result.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {out/'multiT_result.json'}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", required=True, help="path to icet .ce file")
    ap.add_argument("--init_cif", required=True, help="initial structure cif")
    ap.add_argument("--n_steps", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--swap_mode", default="uniform",
                    choices=["uniform", "weighted"])
    args = ap.parse_args()
    run_mc(args.ce, args.init_cif, args.n_steps, args.temperature, args.seed,
           args.log_every, args.swap_mode)


if __name__ == "__main__":
    main()
