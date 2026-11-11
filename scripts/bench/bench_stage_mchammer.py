"""Per-stage timing of one canonical swap MC step in the ORIGINAL icet/mchammer (CPU).

Stages (exactly the calls mchammer's ThermodynamicBaseEnsemble.do_canonical_swap makes):
  (1) propose : ensemble.get_random_sublattice_index + configuration.get_swapped_state
  (2) energy  : ensemble._get_property_change -> calculator.calculate_change
                (np.array copy of occupations + 2x C++ get_cluster_vector_change + dot with ECIs)
  (3) accept  : ensemble._acceptance_condition, and if accepted ensemble.update_occupations
                (ConfigurationManager.update_occupations + calculator.accept_change)

One cell size per process (so ru_maxrss is the peak of that size only); results are
merged into a JSON list keyed by n_atoms.

Inputs: --ce (default $MODELS/ce/ptni/icet_c8_6_5.ce).  Output: --out (default $RESULTS/bench_stage/mchammer.json).
Usage (inside a batch job, see slurm/bench_stage_mchammer.sh):
  python scripts/bench/bench_stage_mchammer.py --M 4 --steps 2000
"""
import argparse
import json
import os
import resource
import time

import numpy as np

from neuralce.paths import MODELS, RESULTS


def rss_now_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024.0
    return float('nan')


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # linux: kB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ce', default=f'{MODELS}/ce/ptni/icet_c8_6_5.ce')
    ap.add_argument('--M', type=int, required=True, help='MxMxM conventional cubic cells')
    ap.add_argument('--a', type=float, default=3.924)
    ap.add_argument('--T', type=float, default=900.0)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--warmup_sweeps', type=float, default=0.0,
                    help='extra untimed equilibration, in units of n_atoms trial steps')
    ap.add_argument('--fine_steps', type=int, default=500)
    ap.add_argument('--ce_cell_atoms', type=int, default=256,
                    help='the .ce parameters are total energies of a cell of this many atoms')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=f'{RESULTS}/bench_stage/mchammer.json')
    args = ap.parse_args()

    from ase.build import bulk
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator
    from mchammer.ensembles import CanonicalEnsemble

    pc = time.perf_counter

    # ---- structure: fcc Ni conventional cubic cell, MxMxM, 50/50 random Ni/Pt
    atoms = bulk('Ni', 'fcc', a=args.a, cubic=True).repeat(args.M)
    n = len(atoms)
    rng = np.random.default_rng(args.seed)
    Z = np.full(n, 28)
    Z[rng.permutation(n)[: n // 2]] = 78
    atoms.set_atomic_numbers(Z)
    print(f'[M={args.M}] n_atoms={n}  Ni={int((Z == 28).sum())} Pt={int((Z == 78).sum())}', flush=True)

    ce = ClusterExpansion.read(args.ce)
    print(f'n_parameters={len(ce.parameters)}', flush=True)
    rss0 = rss_now_mb()

    # ---- BUILD (setup)
    # .ce params = total energy of a 256-atom cell -> scaling n/256 gives the true total energy
    t0 = pc()
    calc = ClusterExpansionCalculator(atoms, ce, scaling=n / float(args.ce_cell_atoms))
    t_calc = pc() - t0
    t0 = pc()
    ens = CanonicalEnsemble(structure=atoms, calculator=calc, temperature=args.T,
                            dc_filename=None, random_seed=args.seed,
                            ensemble_data_write_interval=10 ** 9,
                            trajectory_write_interval=10 ** 9)
    t_ens = pc() - t0
    build_s = t_calc + t_ens
    rss_build = rss_now_mb()
    print(f'build: calculator {t_calc:.1f} s + ensemble {t_ens:.2f} s ; RSS {rss0:.0f} -> {rss_build:.0f} MB',
          flush=True)

    conf = ens.configuration
    probs = ens._swap_sublattice_probabilities

    # ---- manual replica of do_canonical_swap, staged timing
    def manual(nsteps, record=True):
        tp = te = ta = 0.0
        ta_acc = ta_rej = 0.0
        nacc = 0
        for _ in range(nsteps):
            t_a = pc()
            sl = ens.get_random_sublattice_index(probs)
            sites, species = conf.get_swapped_state(sl, None, None)
            t_b = pc()
            dE = ens._get_property_change(sites, species)
            t_c = pc()
            acc = ens._acceptance_condition(dE)
            if acc:
                ens.update_occupations(sites, species)
            t_d = pc()
            tp += t_b - t_a
            te += t_c - t_b
            ta += t_d - t_c
            if acc:
                nacc += 1
                ta_acc += t_d - t_c
            else:
                ta_rej += t_d - t_c
        return tp, te, ta, ta_acc, ta_rej, nacc

    n_warm = args.warmup + int(args.warmup_sweeps * n)
    manual(n_warm)
    t0 = pc()
    tp, te, ta, ta_acc, ta_rej, nacc = manual(args.steps)
    wall_manual = pc() - t0
    S = args.steps

    # ---- fine breakdown of the energy stage (replica of calculate_change internals)
    t_copy = t_cpp = t_dot = 0.0
    params = calc.cluster_expansion.parameters
    for _ in range(args.fine_steps):
        sl = ens.get_random_sublattice_index(probs)
        sites, species = conf.get_swapped_state(sl, None, None)
        t_a = pc()
        occ = np.array(conf.occupations)
        t_b = pc()
        t_copy += t_b - t_a
        change = 0.0
        for idx, newZ in zip(sites, species):
            t_a = pc()
            cv = calc.cpp_calc.get_cluster_vector_change(occupations=occ, flip_index=idx,
                                                         new_occupation=newZ)
            t_b = pc()
            change += np.dot(cv, params)
            t_c = pc()
            occ[idx] = newZ
            t_cpp += t_b - t_a
            t_dot += t_c - t_b
        dE = change * calc._property_scaling
        if ens._acceptance_condition(dE):
            ens.update_occupations(sites, species)
    F = max(args.fine_steps, 1)

    # ---- plain ensemble.run cross-check
    t0 = pc()
    ens.run(args.steps)
    wall_run = pc() - t0

    res = dict(
        n_atoms=n, M=args.M, T_K=args.T, n_parameters=int(len(ce.parameters)),
        build_s=build_s, build_calculator_s=t_calc, build_ensemble_s=t_ens,
        mem_MB=peak_mb(), rss_before_build_MB=rss0, rss_after_build_MB=rss_build,
        steps=S, warmup_steps=n_warm,
        t_propose_us=1e6 * tp / S, t_energy_us=1e6 * te / S, t_accept_us=1e6 * ta / S,
        accept_rate=nacc / S,
        t_accept_us_when_accepted=(1e6 * ta_acc / nacc) if nacc else None,
        t_accept_us_when_rejected=(1e6 * ta_rej / (S - nacc)) if S > nacc else None,
        energy_fine_steps=args.fine_steps,
        t_energy_occcopy_us=1e6 * t_copy / F,
        t_energy_cpp_cvchange_us=1e6 * t_cpp / F,
        t_energy_dot_us=1e6 * t_dot / F,
        steps_per_s_manual=S / wall_manual,
        steps_per_s_run=S / wall_run,
        host=os.uname().nodename, slurm_job=os.environ.get('SLURM_JOB_ID'),
    )
    print(json.dumps(res, indent=1), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import fcntl
    with open(args.out + '.lock', 'w') as lk:   # two jobs may merge into the same file
        fcntl.flock(lk, fcntl.LOCK_EX)
        data = []
        if os.path.exists(args.out):
            with open(args.out) as f:
                data = json.load(f)
        data = [d for d in data if d.get('n_atoms') != n] + [res]
        data.sort(key=lambda d: d['n_atoms'])
        tmp = args.out + f'.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, args.out)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
