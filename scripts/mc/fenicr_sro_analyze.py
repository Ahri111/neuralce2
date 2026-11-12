"""Warren-Cowley SRO parameters + Cv(T) of a FeNiCr model's multi-T MC run (slurm/mc_fenicr.sh), against experiment.

alpha_l(AB) = 1 - P_l(B | A) / c_B on the 1NN / 2NN shells of the ideal fcc cell, averaged over the production
snapshots (first SW_EQ sweeps discarded).  Experiment: Cenedese et al. 1984 (Fe56Cr21Ni23 annealed 1273 K), values as
tabulated in Wrobel et al. PRB 91, 024108 (2015) Table VII.

Argument: a run tag under $RUNS/fenicr_select/<tag>/multiT, or the full path of the <tag> directory.
Env: SW_EQ (equilibration sweeps discarded, default 1000), FENICR_SELECT_ROOT (default $RUNS/fenicr_select).
Writes <tag dir>/sro.json and $REPO/figures/fenicr_select_<tag>.png.
Run: python scripts/mc/fenicr_sro_analyze.py <tag | /path/to/tag_dir>
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, sys, glob, json, numpy as np
from neuralce.paths import REPO, RUNS

KB = 8.617333e-5
EXPT = {"Fe-Ni": (0.017, -0.002), "Fe-Cr": (-0.009, 0.043), "Cr-Ni": (-0.113, 0.148)}      # (alpha1, alpha2)
SPECIES = {0: "Cr", 1: "Fe", 2: "Ni"}                                                         # sigma = rank of Z within the sublattice
if len(sys.argv) < 2:
    sys.exit(f"usage: python {sys.argv[0]} <run tag under $NCE_RUNS/fenicr_select, or a run directory>")
arg = sys.argv[1].rstrip("/")
ROOT = os.environ.get("FENICR_SELECT_ROOT", f"{RUNS}/fenicr_select")
TAGDIR = arg if os.path.isdir(arg) else f"{ROOT}/{arg}"; tag = os.path.basename(TAGDIR); D = f"{TAGDIR}/multiT"
SW_EQ = int(os.environ.get("SW_EQ", "1000"))
if not os.path.exists(f"{D}/snapshots.npz"):                                               # the Fe-Ni-Cr Monte Carlo runs are not shipped: skip cleanly
    print(f"[skip] {os.path.basename(__file__)}: {D}/snapshots.npz is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under FENICR_SELECT_ROOT/<tag>/multiT (or give the tag directory) to run", flush=True); sys.exit(0)
from ase.io import read
from ase.neighborlist import neighbor_list
z = np.load(f"{D}/snapshots.npz"); S = z["sigma"]; temps = z["temps"]; chunk_steps = int(z["chunk_steps"]); chunks = z["chunk"]
N = S.shape[2]
a = read(sorted(glob.glob(f"{D}/final_T*_r0.cif"))[0]); assert len(a) == N            # the run's own atoms (input order = snapshot order)
i, j, d = neighbor_list("ijd", a, 3.7); sh1 = (d < 2.7); sh2 = (~sh1) & (d < 3.7)
assert abs((sh1.sum() / N) - 12) < 1e-6 and abs((sh2.sum() / N) - 6) < 1e-6, (sh1.sum() / N, sh2.sum() / N)
sweeps_done = chunks * chunk_steps / N; keep = sweeps_done > SW_EQ
c = {k: float((S[0, 0] == k).mean()) for k in SPECIES}
def alpha(sig, mask):
    ii, jj = i[mask], j[mask]; out = {}
    for A, B, name in ((1, 2, "Fe-Ni"), (1, 0, "Fe-Cr"), (0, 2, "Cr-Ni")):
        selA = sig[ii] == A; pB = float((sig[jj[selA]] == B).mean()); out[name] = 1.0 - pB / c[B]
    return out
res = {"temps": temps.tolist(), "composition": {SPECIES[k]: v for k, v in c.items()}, "n_snapshots_used": int(keep.sum()), "alpha1": {}, "alpha2": {}}
for t_i, T in enumerate(temps):
    a1 = [alpha(S[s, t_i], sh1) for s in np.where(keep)[0]]; a2 = [alpha(S[s, t_i], sh2) for s in np.where(keep)[0]]
    for name in EXPT:
        res["alpha1"].setdefault(name, []).append(float(np.mean([x[name] for x in a1]))); res["alpha2"].setdefault(name, []).append(float(np.mean([x[name] for x in a2])))
# Cv from the energy trajectory (production half), per temperature
e = np.load(f"{D}/energies.npz"); traj = e["traj"]; half = traj.shape[1] // 2; E = traj[:, half:] * N          # total eV
cv = E.var(1) / (KB * temps) ** 2 / N; res["cv_kB_per_atom"] = cv.tolist(); res["E_mean_eV_per_atom"] = (E.mean(1) / N).tolist()
Ec = e["E_chunk"]; q = Ec.shape[0] // 5; res["E_drift_last20pct_meV_per_atom"] = ((Ec[-q:].mean(0) - Ec[-2 * q:-q].mean(0)) * 1e3).tolist()   # equilibration check
print("E drift over the last 20% of the run (meV/atom, per T):", np.round(res["E_drift_last20pct_meV_per_atom"], 2).tolist())
k13 = int(np.argmin(np.abs(temps - 1300)))
res["at_1300K"] = {name: {"alpha1": res["alpha1"][name][k13], "alpha2": res["alpha2"][name][k13], "expt_alpha1": EXPT[name][0], "expt_alpha2": EXPT[name][1]} for name in EXPT}
res["score_1300K"] = float(np.mean([abs(res["alpha1"][n][k13] - EXPT[n][0]) + abs(res["alpha2"][n][k13] - EXPT[n][1]) for n in EXPT]) / 2)
res["Tc_cv_peak"] = float(temps[int(np.argmax(cv))])
json.dump(res, open(f"{TAGDIR}/sro.json", "w"), indent=1)
print(f"[{tag}] 1300 K:  " + "  ".join(f"{n}: a1 {res['at_1300K'][n]['alpha1']:+.3f} (expt {EXPT[n][0]:+.3f}) a2 {res['at_1300K'][n]['alpha2']:+.3f} (expt {EXPT[n][1]:+.3f})" for n in EXPT)
      + f"  | mean |dalpha| {res['score_1300K']:.3f}  | Cv peak {res['Tc_cv_peak']:.0f} K")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for name, col in zip(EXPT, ("C0", "C1", "C2")):
    ax[0].plot(temps, res["alpha1"][name], "-o", color=col, label=name); ax[0].plot([1273], [EXPT[name][0]], "*", color=col, ms=12)
    ax[1].plot(temps, res["alpha2"][name], "-o", color=col, label=name); ax[1].plot([1273], [EXPT[name][1]], "*", color=col, ms=12)
ax[0].set_title("alpha 1NN (stars: expt 1273 K)"); ax[1].set_title("alpha 2NN"); ax[2].plot(temps, cv, "-o"); ax[2].set_title("Cv (kB/atom)")
for x in ax: x.set_xlabel("T (K)")
ax[0].legend(); fig.suptitle(f"{tag}  Fe56Cr21Ni23, 4000 atoms"); fig.tight_layout(); os.makedirs(f"{REPO}/figures", exist_ok=True)
fig.savefig(f"{REPO}/figures/fenicr_select_{tag}.png", dpi=130); print(f"wrote {TAGDIR}/sro.json and figures/fenicr_select_{tag}.png"); print("ANALYZE_DONE")
