"""Central Monte Carlo entry point (console script ``neuralce-mc``).

Two energy models share one command line:

    nce   neural cluster expansion (``--ckpt models/nce/<tag>/model.pkl``);
          receptive-field local energy update, float32, ``--energy_mode local``
    icet  orthonormal cluster expansion (``--ce <file>.ce`` fitted with icet);
          template-only GPU tables, ``--energy_mode delta``

Examples::

    neuralce-mc --framework nce --ckpt models/nce/nipt_v3_nomax_sel/model.pkl \
        --init_cif models/structures/ptni/ref_L10_NiPt.cif --supercell 2 2 2 \
        --pt --T_min 300 --T_max 1400 --n_replicas 32 --n_steps 200000 \
        --energy_mode local --out_dir runs/demo_nce

    neuralce-mc --framework icet --ce models/ce/ptni/icet_c8_6_5.ce \
        --init_cif models/structures/ptni/ref_L10_NiPt.cif --supercell 2 2 2 \
        --pt --T_min 300 --T_max 1400 --n_replicas 32 --n_steps 200000 \
        --energy_mode delta --out_dir runs/demo_icet

``ce_cnn`` is accepted as an alias of ``nce`` (name used by older launch scripts).
"""
from __future__ import annotations
import os
import json
import argparse
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Central MC / parallel-tempering entry for the cluster-expansion models.")
    ap.add_argument("--framework", required=True, choices=["nce", "ce_cnn", "icet"],
                    help="nce = neural cluster expansion (alias ce_cnn); icet = orthonormal cluster expansion")

    # energy model
    ap.add_argument("--ce", default=None, help="(icet) fitted .ce file")
    ap.add_argument("--ckpt", default=None, help="(nce) models/nce/<tag>/model.pkl")
    ap.add_argument("--init_cif", required=True, help="initial structure (cif)")

    # single-T / multi-T
    ap.add_argument("--T", type=float, default=300.0, help="single-replica temperature in K")
    ap.add_argument("--temps", nargs="+", type=float, default=None,
                    help="list of temperatures: one independent chain per temperature (overrides --T)")
    ap.add_argument("--n_steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=0)
    ap.add_argument("--swap_mode", default="uniform", choices=["uniform", "weighted"],
                    help="(icet) sublattice sampling: 50:50 / proportional to size")

    # parallel tempering
    ap.add_argument("--pt", action="store_true", help="parallel tempering over a temperature ladder")
    ap.add_argument("--n_replicas", type=int, default=32, help="(PT) replicas per ladder")
    ap.add_argument("--n_ladders", type=int, default=1, help="(icet PT) independent interleaved ladders")
    ap.add_argument("--T_min", type=float, default=100.0)
    ap.add_argument("--T_max", type=float, default=2000.0)
    ap.add_argument("--T_grid_file", default=None,
                    help="(PT) text file with one temperature (K) per line, ascending; overrides T_min/T_max/n_replicas")
    ap.add_argument("--init_npz", default=None,
                    help="(PT) per-replica starting structures: npz with numbers (R,N), positions (N,3), cell (3,3), "
                         "temps (R,) (see neuralce/mc/init_structures.py)")
    ap.add_argument("--exchange_every", type=int, default=100, help="(PT) replica-exchange interval in steps")
    ap.add_argument("--sample_interval", type=int, default=100, help="energy sub-sampling interval of the trajectory")
    ap.add_argument("--chunk", type=int, default=1000,
                    help="steps per jitted chunk; the carried energy is re-anchored with a full recompute at chunk boundaries")
    ap.add_argument("--checkpoint_every", type=int, default=0,
                    help="save the MC state to out_dir/checkpoint.npz every K production chunks and resume from it "
                         "when restarted (0 = off)")
    ap.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 1], metavar=("NX", "NY", "NZ"),
                    help="tile init_cif NX*NY*NZ times in memory (species re-shuffled unless "
                         "CE_SUPERCELL_PERIODIC=1 / CE_CNN_SUPERCELL_PERIODIC=1)")
    ap.add_argument("--out_dir", default=".", help="directory for trajectory and summary outputs")

    # lattice / species configuration (icet)
    ap.add_argument("--ideal_map", default=None,
                    help='(icet) JSON mapping species -> idealised species for primitive detection, e.g. \'{"Fe":"Ti","Xe":"O"}\'')
    ap.add_argument("--inert_species", nargs="*", default=None, help='(icet) species excluded from swaps, e.g. "Sr"')
    ap.add_argument("--swap_pairs", default=None, help='(icet) JSON list of swap pairs, e.g. \'[["Ti","Fe"],["O","Xe"]]\'')
    ap.add_argument("--ref_cif", default=None, help="(icet) reference cif for primitive detection (default: init_cif)")
    ap.add_argument("--symprec", type=float, default=1e-2)
    ap.add_argument("--frac_tol", type=float, default=None,
                    help="(icet) fractional position tolerance of the supercell -> primitive matcher")

    # spins
    ap.add_argument("--spin_move_prob", type=float, default=0.5,
                    help="probability of a spin move (vs a swap) per step; 0 for spin-free models")
    ap.add_argument("--spin_orbits_pkl", default=None, help="(icet PT) Heisenberg pair-orbit pkl (pair_orbits / j_values)")
    ap.add_argument("--spin_states", default=None,
                    help='(icet PT) JSON dict species -> allowed spin values, e.g. \'{"Fe":[-1,0,1]}\'')
    ap.add_argument("--spin_allowed_file", default="",
                    help="(nce) JSON {species: [spin values]} restricting spin proposals per species")
    ap.add_argument("--spin_init", choices=["random", "fm", "zero"], default="random",
                    help="(nce) initial spin state: random draw, all +1, or all 0")

    # numerics
    ap.add_argument("--energy_mode", choices=["full", "delta", "local"], default="full",
                    help="full = recompute the whole energy every step; delta (icet) = cluster-instance local update; "
                         "local (nce) = receptive-field local update")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--anchor_every", type=int, default=None,
                    help="(nce) full cache/energy recompute every K chunks (default: every chunk while chunk >= N, "
                         "else about once per sweep)")
    ap.add_argument("--model_seed", type=int, default=0,
                    help="(nce) which saved parameter set to use; -1 = ensemble mean over all seeds")
    ap.add_argument("--platform", choices=["auto", "cpu", "cuda"], default="auto",
                    help="JAX platform; auto = GPU if visible, else CPU (overrides JAX_PLATFORMS)")
    return ap.parse_args(argv)


def _load_T_grid(path):
    import numpy as np
    return np.atleast_1d(np.loadtxt(path, comments="#", dtype=np.float64))


def dispatch_icet(args):
    if args.energy_mode == "local":
        raise ValueError("--energy_mode local is the nce path; icet uses --energy_mode delta (or full)")
    from neuralce.ce.icet_mc import run_mc, run_multiT_mc, run_pt_mc
    ideal_map = json.loads(args.ideal_map) if args.ideal_map else None
    swap_pairs = json.loads(args.swap_pairs) if args.swap_pairs else None
    icet_kw = dict(ideal_map=ideal_map, inert_species=args.inert_species, symprec=args.symprec,
                   ref_cif=args.ref_cif, swap_pairs=swap_pairs, fractional_position_tolerance=args.frac_tol)
    if args.pt:
        spin_states = json.loads(args.spin_states) if args.spin_states else None
        if args.energy_mode == "delta":
            # tables stay float32; the carried total energy is float64 (large cells)
            import jax
            jax.config.update("jax_enable_x64", True)
        reps = tuple(int(x) for x in args.supercell) if list(args.supercell) != [1, 1, 1] else None
        T_grid = _load_T_grid(args.T_grid_file) if args.T_grid_file else None
        return run_pt_mc(
            supercell_reps=reps, chunk_size=args.chunk, sample_interval=args.sample_interval,
            checkpoint_every=args.checkpoint_every, T_grid=T_grid, init_npz=args.init_npz,
            ce_path=args.ce, init_cif=args.init_cif, n_steps=args.n_steps,
            T_min=args.T_min, T_max=args.T_max, n_replicas=args.n_replicas, n_ladders=args.n_ladders,
            swap_mode=args.swap_mode, seed=args.seed, log_every=args.log_every,
            exchange_every=args.exchange_every, out_dir=args.out_dir, energy_mode=args.energy_mode,
            spin_move_prob=args.spin_move_prob, spin_orbits_pkl=args.spin_orbits_pkl, spin_states=spin_states,
            **icet_kw,
        )
    if args.temps:
        return run_multiT_mc(ce_path=args.ce, init_cif=args.init_cif, temps=args.temps, n_steps=args.n_steps,
                             swap_mode=args.swap_mode, seed=args.seed, log_every=args.log_every,
                             out_dir=args.out_dir, energy_mode=args.energy_mode, **icet_kw)
    return run_mc(ce_path=args.ce, init_cif=args.init_cif, n_steps=args.n_steps, temperature=args.T,
                  seed=args.seed, log_every=args.log_every, swap_mode=args.swap_mode,
                  energy_mode=args.energy_mode, **icet_kw)


def dispatch_nce(args):
    if not args.ckpt:
        raise ValueError("--framework nce requires --ckpt models/nce/<tag>/model.pkl")
    if args.energy_mode not in ("local", "full"):
        raise ValueError("nce supports --energy_mode local (recommended) or full")
    from neuralce.nce.mcmc import run_mc, run_multiT_mc, run_pt_mc
    common = dict(model_path=args.ckpt, init_cif=args.init_cif, seed=args.seed, log_every=args.log_every,
                  out_dir=args.out_dir, chunk=args.chunk, sample_interval=args.sample_interval,
                  supercell=tuple(args.supercell), seed_idx=args.model_seed, dtype=args.dtype,
                  spin_init=args.spin_init, p_spin=args.spin_move_prob, anchor_every=args.anchor_every,
                  spin_allowed=(json.load(open(args.spin_allowed_file)) if args.spin_allowed_file else None),
                  init_npz=args.init_npz, checkpoint_every=args.checkpoint_every)
    if args.pt:
        T_grid = _load_T_grid(args.T_grid_file) if args.T_grid_file else None
        return run_pt_mc(n_steps=args.n_steps, T_min=args.T_min, T_max=args.T_max, n_replicas=args.n_replicas,
                         T_grid=T_grid, exchange_every=args.exchange_every, **common)
    if args.temps:
        return run_multiT_mc(temps=args.temps, n_steps=args.n_steps, **common)
    return run_mc(n_steps=args.n_steps, temperature=args.T, n_replicas=args.n_replicas, **common)


def main(argv=None):
    args = parse_args(argv)
    if args.framework == "ce_cnn":
        args.framework = "nce"
    if args.framework == "icet" and not args.ce:
        raise SystemExit("--ce is required for --framework icet")
    # the platform must be fixed before any JAX import (the dispatch imports happen below)
    if args.platform != "auto":
        os.environ["JAX_PLATFORMS"] = args.platform
        print(f"== platform set to {args.platform}", flush=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"== mcmc: framework={args.framework}, pt={args.pt}, energy_mode={args.energy_mode} ==", flush=True)

    result = dispatch_icet(args) if args.framework == "icet" else dispatch_nce(args)

    if result is not None and not isinstance(result, dict) and hasattr(result, "__dataclass_fields__"):
        result = {k: getattr(result, k) for k in result.__dataclass_fields__}      # icet PT: MCMCResult
    if isinstance(result, dict):
        summary = {k: (float(v) if hasattr(v, "shape") and getattr(v, "shape", None) == () else v)
                   for k, v in result.items()}
        summary = {k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool, list))}
        out_json = Path(args.out_dir) / "mcmc_summary.json"
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nwrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
