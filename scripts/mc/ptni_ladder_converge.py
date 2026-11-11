#!/usr/bin/env python
"""Convergence + Tc readout for the PtNi temperature-ladder runs (icet PT and nce PT).

One run directory per ladder.  The framework is detected from the files it left:
  icet  -> mcmc_result.npz  (energies = TOTAL eV of the cell, one column per temperature slot)
  conv  -> energies.npz     (traj = eV/atom, one row per temperature slot; written by the nce runner)
A run that is still going has neither: its checkpoint.npz is read instead and the kind is printed with a * (the
numbers are then a partial readout of an unfinished ladder).  conv's checkpoint drift is in the model's normalised
units, so that column is only comparable within one model.

What it reports per run, and why (each column answers one question about whether the number can be trusted):

  drift        largest |carried E - full recompute| at a re-anchor, in meV/atom.  Cv from fluctuations is
               Var(E_total)/(kB T^2), i.e. N x Var(e_per_atom)/(kB T^2): a per-atom energy error that does not
               shrink with N is amplified by N in Cv.  At 1M atoms the bias stays below a few percent of the
               peak only if the drift is under ~0.1 meV/atom, so that is the threshold flagged here.
               (conv records it every re-anchor; icet's delta mode carries E as a running sum and stores no
               drift log -> use --recompute, which reloads the CE and re-evaluates sigma_final.)

  Tc(dE/dT)    temperature of the maximum of dE/dT on the dense part of the grid.  dE/dT is a first derivative
               of the mean energy: a constant energy offset (drift) cancels, so it survives a drift that would
               visibly bias Cv.  This is the estimator to quote at 1M.

  Tc(Cv)       temperature of the maximum of Var(E)/(kB T^2).  Reported for comparison only; it is the one that
               the drift bias hits.

  split-half   the same Tc(dE/dT) computed from the first and the second half of the production samples.  If the
               two halves disagree by more than one grid spacing, or the mean energy of the two halves differs by
               more than the tolerance below, the ladder has not equilibrated and no Tc from it means anything yet
               -- that is the actual convergence test, not the smoothness of the curve.

  E drift/2    mean energy of the second half minus the first half, per temperature (meV/atom, max over the
               dense block).  A ladder still relaxing downhill shows a systematic negative value here.

Usage
  python scripts/mc/ptni_ladder_converge.py RUN_DIR [RUN_DIR ...]
        [--burn 0.5] [--json out.json] [--png out.png] [--recompute CE_PATH INIT_CIF NREP]

  --burn      fraction of the samples discarded before averaging (default 0.5).
  --recompute icet only: reload the CE, rebuild the cell (init cif tiled NREP^3) and re-evaluate the energy of
              every sigma_final, to measure the delta-mode drift that mcmc_result.npz does not record.  Needs a
              GPU node (submit it, do not run it on the login node).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

KB = 8.617333262e-5          # eV/K


# -- loading -------------------------------------------------------------------
def load_run(d):
    """-> dict(kind, temps (R,), e (n_samples, R) eV/atom, n_atoms, drift_meV or None)."""
    icet_p, conv_p = os.path.join(d, "mcmc_result.npz"), os.path.join(d, "energies.npz")
    if os.path.exists(icet_p):
        z = np.load(icet_p)
        n_atoms = int(z["sigma_final"].shape[1])
        return dict(kind="icet", dir=d, temps=np.asarray(z["temperatures"], float),
                    e=np.asarray(z["energies"], float) / n_atoms,          # total eV -> eV/atom
                    n_atoms=n_atoms, drift_meV=None,
                    exch=float(z["exchange_accept_rate"]), swap=float(z["swap_accept_rate"]))
    if os.path.exists(conv_p):
        z = np.load(conv_p)
        if "energies" in z.files:                                           # icet's energies-first file (same folder name)
            n_atoms = int(z["n_atoms"])
            return dict(kind="icet", dir=d, temps=np.asarray(z["temperatures"], float),
                        e=np.asarray(z["energies"], float) / n_atoms, n_atoms=n_atoms, drift_meV=None,
                        exch=float(z["exchange_accept_rate"]), swap=float(z["swap_accept_rate"]))
        n_atoms = None
        js = os.path.join(d, "mcmc_summary.json")                           # written by neuralce.mc.run
        if os.path.exists(js):
            try: n_atoms = int(json.load(open(js)).get("n_atoms") or 0) or None
            except Exception: pass
        if n_atoms is None and os.path.exists(os.path.join(d, "final_sigma.npy")):
            n_atoms = int(np.load(os.path.join(d, "final_sigma.npy"), mmap_mode="r").shape[1])
        drift = z["drift_eV"] if "drift_eV" in z.files else None            # older runs: only the normalised drift
        return dict(kind="conv", dir=d, temps=np.asarray(z["temps"], float),
                    e=np.asarray(z["traj"], float).T,                       # (R, n_samples) -> (n_samples, R)
                    n_atoms=n_atoms,
                    drift_meV=(None if drift is None else float(np.max(np.abs(drift))) * 1e3),
                    exch=None, swap=float(np.mean(z["acc_chunk"])))
    ck = os.path.join(d, "checkpoint.npz")                                  # still running: read what it has so far
    if os.path.exists(ck):
        z = np.load(ck)
        if "energy_samples" in z.files:                                     # icet
            n_atoms = int(z["cfg_n_atoms"]); si = int(z["si"])
            return dict(kind="icet*", dir=d, temps=np.asarray(z["cfg_temps"], float),
                        e=np.asarray(z["energy_samples"], float)[:si] / n_atoms, n_atoms=n_atoms,
                        drift_meV=None, exch=None, swap=float("nan"))
        return dict(kind="conv*", dir=d, temps=1.0 / (KB * np.asarray(z["cfg_betas"], float)),   # conv
                    e=np.asarray(z["traj"], float).T, n_atoms=int(z["cfg_n_atoms"]),
                    drift_meV=float(np.max(np.abs(z["drift_log"]))) * 1e3 if "drift_log" in z.files else None,
                    exch=None, swap=float(np.mean(z["acc_chunk"])))
    raise FileNotFoundError(f"{d}: no mcmc_result.npz (icet), energies.npz (conv) or checkpoint.npz")


# -- estimators ----------------------------------------------------------------
def dense_block(T):
    """Indices of the longest stretch of temperatures with no big gap in it.  The explicit ladders carry a dense window
    around Tc plus a few far-away anchor temperatures (300, 500, 1400 K), and a derivative taken across such a gap is
    meaningless.  A gap is a spacing above 3x the median spacing, which leaves an evenly OR geometrically spaced
    ladder (the older runs) whole."""
    o = np.argsort(T); Ts = T[o]; d = np.diff(Ts)
    if len(d) < 2: return o
    ok = d <= 3.0 * float(np.median(d))
    best = 0; bi = bj = 0; i = 0
    while i < len(ok):
        if not ok[i]: i += 1; continue
        j = i
        while j + 1 < len(ok) and ok[j + 1]: j += 1
        if (j - i + 1) > best: best = j - i + 1; bi, bj = i, j
        i = j + 1
    return o if best == 0 else o[bi:bj + 2]


def peak(T, y):
    """Temperature of the maximum of y(T), refined by a parabola through the top point and its neighbours.
    Returns (T_peak, index, at_edge); at_edge means the maximum sits on the first or last sampled temperature,
    i.e. the peak is not bracketed by the ladder and the number is a lower/upper bound, not a Tc."""
    k = int(np.argmax(y))
    if k == 0 or k == len(y) - 1: return float(T[k]), k, True
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    den = y0 - 2 * y1 + y2
    dt = 0.0 if den == 0 else 0.5 * (y0 - y2) / den                          # in units of the grid spacing
    dt = float(np.clip(dt, -1.0, 1.0))
    return float(T[k] + dt * (T[k + 1] - T[k - 1]) / 2.0), k, False


def tc_dedt(T, e_mean):
    """Tc from the maximum of dE/dT (central differences on the dense block)."""
    idx = dense_block(T)
    Td, Ed = T[idx], e_mean[idx]
    o = np.argsort(Td); Td, Ed = Td[o], Ed[o]
    if len(Td) < 5: return float("nan"), Td, np.full(len(Td), np.nan), True
    d = np.gradient(Ed, Td)
    Tc, _, edge = peak(Td[1:-1], d[1:-1])                                    # ends of a gradient are one-sided
    return Tc, Td, d, edge


def tc_cv(T, e, n_atoms):
    """Tc from the maximum of Cv = N Var(e)/(kB T^2) (kB per atom), on the dense block."""
    idx = dense_block(T)
    Td = T[idx]; o = np.argsort(Td); idx = idx[o]; Td = T[idx]
    N = n_atoms or 1
    cv = N * np.var(e[:, idx], axis=0) / (KB * Td ** 2) / KB                 # kB/atom
    Tc, _, edge = peak(Td, cv)
    return Tc, Td, cv, edge


def analyse(run, burn=0.5):
    T, e = run["temps"], run["e"]
    n0 = int(len(e) * burn)
    prod = e[n0:]
    if len(prod) < 8: raise ValueError(f"{run['dir']}: only {len(prod)} samples after burn-in")
    em = prod.mean(axis=0)
    Tc_d, Td, dEdT, edge_d = tc_dedt(T, em)
    Tc_c, Tcv, cv, edge_c = tc_cv(T, prod, run["n_atoms"])
    h = len(prod) // 2
    Tc_1, _, _, _ = tc_dedt(T, prod[:h].mean(axis=0))
    Tc_2, _, _, _ = tc_dedt(T, prod[h:].mean(axis=0))
    idx = dense_block(T)
    dE_half = (prod[h:].mean(axis=0) - prod[:h].mean(axis=0))[idx] * 1e3     # meV/atom
    # tolerance on that difference: 3x its own (naive, correlation-free) standard error, but never below
    # 0.1 meV/atom -- the same bar the Cv drift has to clear at 1M, and at 1M it is the one that binds.
    sem = np.sqrt(prod[:h].var(axis=0) / h + prod[h:].var(axis=0) / (len(prod) - h))[idx] * 1e3
    dE_tol = float(max(0.1, 3.0 * np.max(sem)))
    spacing = float(np.median(np.diff(np.sort(T[idx])))) if len(idx) > 2 else float("nan")
    ok_tc = bool(abs(Tc_1 - Tc_2) <= spacing); ok_dE = bool(np.max(np.abs(dE_half)) <= dE_tol)
    return dict(dir=run["dir"], kind=run["kind"], n_atoms=run["n_atoms"], n_T=len(T),
                n_samples=int(len(e)), n_prod=int(len(prod)), spacing=spacing,
                Tc_dEdT=Tc_d, Tc_Cv=Tc_c, Tc_half1=Tc_1, Tc_half2=Tc_2,
                dTc_half=abs(Tc_1 - Tc_2), equilibrated=(ok_tc and ok_dE), ok_tc=ok_tc, ok_dE=ok_dE,
                bracketed=bool(not edge_d and not edge_c),                   # the peak sits inside the sampled window
                sd_meV=float(np.max(prod[:, idx].std(axis=0)) * 1e3),        # per-sample scatter: ~0 means a frozen ladder
                dE_half_max=float(np.max(np.abs(dE_half))), dE_tol=dE_tol,
                drift_meV=run["drift_meV"],
                drift_ok=(None if run["drift_meV"] is None else bool(run["drift_meV"] < 0.1)),
                swap_acc=run["swap"], exch_acc=run["exch"],
                T_dense=[float(x) for x in Td], dEdT=[float(x) for x in dEdT],
                T_cv=[float(x) for x in Tcv], Cv=[float(x) for x in cv],
                T_all=[float(x) for x in T], E_mean=[float(x) for x in em])


# -- icet drift, measured (needs the CE and a GPU) -----------------------------
def icet_drift(run_dir, ce_path, init_cif, nrep):
    """max |E(sigma_final) recomputed - last recorded sample| in meV/atom, for the delta-mode running sum."""
    from icet import ClusterExpansion
    from ase.io import read as ase_read
    from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
    from neuralce.ce.icet_mc import _rebuild_as_clean_supercell
    z = np.load(os.path.join(run_dir, "mcmc_result.npz"))
    ce = ClusterExpansion.read(ce_path)
    prim = ce._cluster_space.primitive_structure
    atoms = _rebuild_as_clean_supercell(ase_read(init_cif), prim).repeat((nrep, nrep, nrep))
    gce = GPUClusterExpansion(ce, atoms, table_mode="implicit", energy_scale=float(nrep ** 3))
    sf = np.asarray(z["sigma_final"]); N = sf.shape[1]
    perm = getattr(gce, "perm", None)
    last = np.asarray(z["energies"])[-1] / N
    err = []
    for r in range(sf.shape[0]):
        s = sf[r] if perm is None else sf[r][perm]                            # input order -> canonical
        err.append(abs(float(gce.energy_a(gce.tp, s)) / N - last[r]))
    return float(max(err)) * 1e3


# -- report --------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--burn", type=float, default=0.5)
    ap.add_argument("--json", default="")
    ap.add_argument("--png", default="")
    ap.add_argument("--recompute", nargs=3, metavar=("CE_PATH", "INIT_CIF", "NREP"), default=None)
    a = ap.parse_args(argv)

    rows = []
    for d in a.run_dirs:
        run = load_run(d)
        if a.recompute and run["kind"] == "icet":
            run["drift_meV"] = icet_drift(d, a.recompute[0], a.recompute[1], int(a.recompute[2]))
        rows.append(analyse(run, a.burn))

    hdr = (f"{'run':<30} {'kind':<5} {'N':>9} {'nT':>3} {'dT':>5} {'Tc(dE/dT)':>10} {'Tc(Cv)':>8} "
           f"{'half1':>7} {'half2':>7} {'|d|':>6} {'eq?':>4} {'brk?':>5} {'sd':>6} {'dE/2':>7} {'drift':>7} {'ok?':>4}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        dr = "  n/a" if r["drift_meV"] is None else f"{r['drift_meV']:7.3f}"
        ok = " n/a" if r["drift_ok"] is None else ("  ok" if r["drift_ok"] else "FAIL")
        print(f"{os.path.basename(r['dir'].rstrip('/')):<30} {r['kind']:<5} {r['n_atoms'] or -1:>9} {r['n_T']:>3} "
              f"{r['spacing']:>5.0f} {r['Tc_dEdT']:>10.1f} {r['Tc_Cv']:>8.1f} {r['Tc_half1']:>7.1f} {r['Tc_half2']:>7.1f} "
              f"{r['dTc_half']:>6.1f} {'yes' if r['equilibrated'] else 'NO':>4} {'yes' if r['bracketed'] else 'NO':>5} "
              f"{r['sd_meV']:>6.3f} {r['dE_half_max']:>7.3f} {dr} {ok}")
    print("\ndT = grid spacing (K) of the dense block; |d| = |half1 - half2| (K); eq? = |d| <= dT AND dE/2 within tolerance")
    print("brk? = the peak lies inside the sampled temperatures (NO -> the ladder does not bracket it: the number is a bound)")
    print("sd = largest per-sample energy scatter over the dense block (meV/atom); ~0 means a frozen ladder, not a converged one")
    print("dE/2 = max |E(2nd half) - E(1st half)| (meV/atom), tolerance " +
          ", ".join(f"{r['dE_tol']:.2f}" for r in rows) + "; drift in meV/atom, ok? = < 0.1 (the 1M threshold).")
    bad = [r for r in rows if not r["equilibrated"]]
    unbr = [r for r in rows if not r["bracketed"]]
    if bad: print(f"\nNOT EQUILIBRATED: {', '.join(os.path.basename(r['dir'].rstrip('/')) for r in bad)}")
    if unbr: print(f"PEAK NOT BRACKETED: {', '.join(os.path.basename(r['dir'].rstrip('/')) for r in unbr)}")
    if len(rows) > 1:
        tcs = [r["Tc_dEdT"] for r in rows]
        print(f"\nTc(dE/dT) across the runs: mean {np.mean(tcs):.1f} K, spread {max(tcs) - min(tcs):.1f} K")

    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1); print(f"\nwrote {a.json}")
    if a.png:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for r in rows:
            lb = os.path.basename(r["dir"].rstrip("/"))
            ax[0].plot(r["T_all"], r["E_mean"], "o-", ms=3, label=lb)
            ax[1].plot(r["T_dense"], r["dEdT"], "o-", ms=3, label=lb)
            ax[2].plot(r["T_cv"], r["Cv"], "o-", ms=3, label=lb)
        for k, (t, y) in enumerate([("T (K)", "E (eV/atom)"), ("T (K)", "dE/dT (eV/atom/K)"), ("T (K)", "Cv (kB/atom)")]):
            ax[k].set_xlabel(t); ax[k].set_ylabel(y); ax[k].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.png, dpi=150); print(f"wrote {a.png}")
    return 0 if (not bad and not unbr) else 1


if __name__ == "__main__":
    sys.exit(main())
