"""Cv peak temperature vs cell size for the PtNi ladder.

Reads every finished run under <ladder root>/<icet|conv>/<x>/M<M>[_tag] (icet: mcmc_result.npz; nce: energies.npz +
mcmc_summary.json).  Cv per replica from the second half of the energy samples, peak refined by a parabola through
the maximum and its neighbours in log T.  Prints a table, writes $RESULTS/ptni_ladder/ladder_summary.json and
$REPO/figures/ptni_ladder_cv_peaks.png.

Env: LADDER_ROOT (default $RUNS/ptni_ladder).   Run: python scripts/mc/ptni_ladder_analyze.py
"""
import os, glob, json, numpy as np
from neuralce.paths import REPO, RUNS, RESULTS

KB = 8.617333e-5; L = os.environ.get("LADDER_ROOT", f"{RUNS}/ptni_ladder"); EXPT = {"x25": 670, "x50": 918, "x75": 853}
SUMMARY = f"{RESULTS}/ptni_ladder/ladder_summary.json"; FIG = f"{REPO}/figures/ptni_ladder_cv_peaks.png"


def peak_interp(T, cv):
    k = int(np.argmax(cv));
    if k == 0 or k == len(T) - 1: return float(T[k])
    x = np.log(T[k-1:k+2]); y = cv[k-1:k+2]; d = (y[0] - 2*y[1] + y[2]);
    return float(np.exp(x[1] - 0.5 * (x[2] - x[0]) / 2 * (y[2] - y[0]) / d)) if d < 0 else float(T[k])


rows = []
for fw in ("icet", "conv"):
    for run in sorted(glob.glob(f"{L}/{fw}/x*/M*")):
        x = run.split("/")[-2]; M = int(run.split("/")[-1].split("_")[0][1:]); N = 4 * M ** 3; tag = run.split("/")[-1]
        try:
            if fw == "icet":
                z = np.load(f"{run}/mcmc_result.npz"); E = z["energies"]; T = z["temperatures"]; half = E.shape[0] // 2; e = E[half:] / N   # eV/atom
                acc = float(z["swap_accept_rate"]); exch = float(z["exchange_accept_rate"])
            else:
                z = np.load(f"{run}/energies.npz"); tr = z["traj"]; T = z["temps"]; half = tr.shape[1] // 2; e = tr[:, half:].T                   # (samples, R) eV/atom
                s = json.load(open(f"{run}/mcmc_summary.json")); acc = float(s["acc_mean"]); exch = float(s["exchange_acc"] or 0)
        except FileNotFoundError:
            continue
        cv = N * e.var(0) / (KB * T) ** 2                # kB per atom
        q = e.shape[0] // 4; drift = float(np.abs(e[-q:].mean(0) - e[:q].mean(0)).max()) * 1e3        # meV/atom change over the production half (max over T)
        if fw == "icet" and acc > 0.2: continue                                                          # pre-fix run (energy not scaled): skip
        Tp = peak_interp(np.asarray(T), cv); rows.append(dict(fw=fw, x=x, N=N, tag=tag, Tpeak=Tp, cv_max=float(cv.max()), acc=acc, exch=exch, n_samples=int(e.shape[0]), drift_meV=drift, at_edge=bool(int(np.argmax(cv)) in (0, len(T) - 1))))
rows.sort(key=lambda r: (r["fw"], r["x"], r["N"])); os.makedirs(os.path.dirname(SUMMARY), exist_ok=True); json.dump(rows, open(SUMMARY, "w"), indent=1)
print(f"{'fw':5s} {'comp':5s} {'N':>9s} {'Cv peak T (K)':>14s} {'expt':>5s} {'Cv max':>7s} {'acc':>6s} {'exch':>6s} {'E drift':>8s} {'ok':>3s}")
for r in rows: print(f"{r['fw']:5s} {r['x']:5s} {r['N']:9d} {r['Tpeak']:14.0f} {EXPT[r['x']]:5d} {r['cv_max']:7.2f} {r['acc']:6.3f} {r['exch']:6.3f} {r['drift_meV']:8.2f} {'' if (r['at_edge'] or r['drift_meV'] > 1.0) else 'ok':>3s}   {r['tag']}")
print("ok = Cv peak inside the T grid and production energies drifted < 1 meV/atom (equilibrated); otherwise the run is not equilibrated at low T")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
for fw, mk in (("icet", "o"), ("conv", "s")):
    for x, col in (("x25", "C0"), ("x50", "C1"), ("x75", "C2")):
        rr = [r for r in rows if r["fw"] == fw and r["x"] == x and "_" not in r["tag"]]
        if rr: ax.plot([r["N"] for r in rr], [r["Tpeak"] for r in rr], marker=mk, color=col, ls="-" if fw == "icet" else "--", label=f"{fw} {x}")
for x, col in (("x25", "C0"), ("x50", "C1"), ("x75", "C2")): ax.axhline(EXPT[x], color=col, ls=":", lw=1)
ax.set_xscale("log"); ax.set_xlabel("atoms"); ax.set_ylabel("Cv peak T (K)"); ax.legend(fontsize=7, ncol=2); ax.set_title("PtNi ladder (dotted: experiment)")
os.makedirs(os.path.dirname(FIG), exist_ok=True); fig.tight_layout(); fig.savefig(FIG, dpi=130); print(f"wrote {SUMMARY} and {FIG}"); print("LADDER_ANALYZE_DONE")
