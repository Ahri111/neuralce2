"""PtNi model selection, step 2: Cv(T) peaks and low-T ground state per (composition, seed) PT run.

For every OUTDIR/pt_<x>_seed<i>/ (OUTDIR default $RUNS/ptni_select; runs of slurm/mc_ptni_select_pt.sh):
    Cv per atom(T) = N * Var(e) / (kB T^2) from the second half of the sampled trajectory (e = eV/atom),
    T_peak = temperature of the Cv maximum (with the two neighbouring grid points quoted as the resolution),
    GS check at the lowest T: Pt occupation of the 4 fcc sublattices of the final configuration
    (L1_2: one sublattice ~1 (Ni3Pt) or ~0 (Pt3Ni), L1_0: two sublattices ~1) and E_min(MC) vs E_model(ideal).
Experiment: NiPt L1_0 918 K, Ni3Pt L1_2 853 K, Pt3Ni L1_2 ~670 K.
Writes OUTDIR/summary.json + $REPO/figures/<FIG>.png.

Env: OUTDIR, M (default 4), FIG (default ptni_select_cv).   Run: python scripts/mc/ptni_select_analyze.py
"""
import os, json, glob, numpy as np
from neuralce.paths import REPO, RUNS
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from ase.io import read
KB = 8.617333262e-5


def peak_interp(temps, cv):
    """Cv maximum refined by a parabola through the max and its two neighbours (in log T): sub-grid peak temperature."""
    k = int(np.argmax(cv))
    if k == 0 or k == len(cv) - 1: return float(temps[k])
    x = np.log(temps[k - 1:k + 2]); y = cv[k - 1:k + 2]; a, b, c = np.polyfit(x, y, 2)
    return float(np.exp(-b / (2 * a))) if a < 0 else float(temps[k])


D = os.environ.get("OUTDIR", f"{RUNS}/ptni_select"); MM = int(os.environ.get("M", "4")); FIG = os.environ.get("FIG", "ptni_select_cv")
EXP = {"x25": ("Pt3Ni L1_2", 670.0, "L12_Pt3Ni"), "x50": ("NiPt L1_0", 918.0, "L10_NiPt"), "x75": ("Ni3Pt L1_2", 853.0, "L12_Ni3Pt")}
refE = json.load(open(f"{D}/ref_energies.json"))


def sublattice_pt(atoms):
    frac = atoms.get_scaled_positions() * MM
    code = ((np.round(frac * 2).astype(int) % 2) * np.array([1, 2, 4])).sum(1)
    sub = np.array([{0: 0, 3: 1, 5: 2, 6: 3}[int(c)] for c in code]); sym = np.array(atoms.get_chemical_symbols())
    return [float((sym[sub == k] == "Pt").mean()) for k in range(4)]


def order_label(occ, x):
    occ = np.sort(occ)[::-1]
    if x == "x50":  score = (occ[0] + occ[1]) / 2 - (occ[2] + occ[3]) / 2          # L1_0: two full, two empty
    elif x == "x75": score = occ[0] - np.mean(occ[1:])                              # Ni3Pt: one Pt sublattice
    else: score = np.mean(occ[:3]) - occ[3]                                          # Pt3Ni: one Ni sublattice
    return float(score)


summary = {}; fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, x in zip(axes, ("x25", "x50", "x75")):
    name, Texp, refkey = EXP[x]; summary[x] = {}
    for run in sorted(glob.glob(f"{D}/pt_{x}_seed*")):
        if not os.path.exists(f"{run}/mcmc_summary.json"): continue                     # still running
        seed = run.split("seed")[-1].split("_")[0]; z = np.load(f"{run}/energies.npz"); s = json.load(open(f"{run}/mcmc_summary.json"))
        seed_label = run.split("seed")[-1]
        temps = z["temps"]; traj = z["traj"]; N = s["n_atoms"]; half = traj.shape[1] // 2
        e = traj[:, half:]; cv = N * e.var(1) / (KB * temps ** 2) / KB                      # in units of kB per atom
        k = int(np.argmax(cv)); Tpk = peak_interp(temps, cv); Tlo = float(temps[max(k - 1, 0)]); Thi = float(temps[min(k + 1, len(temps) - 1)])
        low = read(f"{run}/final_T{temps[0]:.0f}_r0.cif")
        occ = sublattice_pt(low); sc = order_label(occ, x)
        Emin = float(np.min(traj[0])); Eref = refE[refkey][f"seed_idx{seed}"]
        summary[x][f"seed{seed_label}"] = dict(T_peak=Tpk, T_peak_range=[Tlo, Thi], Cv_peak_kB=float(cv[k]), T_exp=Texp, exchange_acc=s["exchange_acc"],
                                        acc=s["acc_mean"], drift=s["drift_max"], steps_per_s=s["steps_per_s"], sublattice_Pt_lowT=occ, order_score=sc,
                                        E_lowT_final=float(z["E_chunk"][-1][0]), E_min_traj_lowT=Emin, E_model_ideal=Eref, dE_min_minus_ideal_meV=(Emin - Eref) * 1e3)
        ax.plot(temps, cv, "o-", ms=3, label=f"seed {seed_label}: peak {Tpk:.0f} K, order {sc:.2f}")
        print(f"{x} {name:12s} seed{seed_label}: Cv peak {Tpk:.0f} K [{Tlo:.0f},{Thi:.0f}] (exp {Texp:.0f})  lowT sublattice Pt {np.round(occ, 2).tolist()} order {sc:.2f}  "
              f"E_min(MC) - E(ideal) = {(Emin - Eref) * 1e3:+.2f} meV/atom  exch {s['exchange_acc']:.2f} acc {s['acc_mean']:.3f} drift {s['drift_max']:.1e}", flush=True)
    ax.axvline(Texp, color="k", ls="--", lw=1, label=f"experiment {Texp:.0f} K"); ax.set_title(f"{name}  (x_Ni = {int(x[1:])/100:.2f}, {4*MM**3} atoms)")
    ax.set_xlabel("T (K)"); ax.set_ylabel("Cv per atom (kB)"); ax.legend(fontsize=8)
fig.suptitle("PtNi neural cluster expansion (nipt_v3_nomax_sel): Cv from PT-MC, per saved seed"); fig.tight_layout()
os.makedirs(f"{REPO}/figures", exist_ok=True)
fig.savefig(f"{REPO}/figures/{FIG}.png", dpi=140); json.dump(summary, open(f"{D}/summary.json", "w"), indent=1)
print(f"saved figures/{FIG}.png"); print("ANALYZE_DONE")
