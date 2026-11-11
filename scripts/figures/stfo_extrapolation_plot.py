"""Quick-look plot of the STFO extrapolation paths (the data behind Fig. 5a): the same clusters, two models, one path out of the training data.
x = number of V_O-V_O pairs within 4.1 A in the 160-site cell (vacancy clustering), y = energy relative to the DFT ground state A00 (x = 0.5).
Grey band: energy range of the DFT training structures at x = 0.5.  Lines: the icet CE and the cluster convolution along swap paths from
A00 to the state the icet MC fell into (mean and range over random swap orders).
Inputs:  results/figure_data/fig7_stfo_extrap.json (scripts/analysis/stfo_extrapolation.py).
Outputs: figures/quicklook/fig7_stfo_extrap.{png,pdf}.
Run:     python scripts/figures/stfo_extrapolation_plot.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import figstyle as fs; fs.use()
from neuralce.paths import REPO, RESULTS
import matplotlib.pyplot as plt
if not os.path.exists(f"{RESULTS}/figure_data/fig7_stfo_extrap.json"):                   # the STFO results are not shipped: skip cleanly
    print(f"[skip] {os.path.basename(__file__)}: {RESULTS}/figure_data/fig7_stfo_extrap.json is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_RESULTS/figure_data to run", flush=True); sys.exit(0)
D = json.load(open(f"{RESULTS}/figure_data/fig7_stfo_extrap.json")); OUTDIR = f"{REPO}/figures/quicklook"; os.makedirs(OUTDIR, exist_ok=True); OUT = f"{OUTDIR}/fig7_stfo_extrap"
fig, ax = plt.subplots(figsize=(8.8, 7.0)); fig.subplots_adjust(left=0.15, right=0.97, top=0.9, bottom=0.13)
te = np.array([t[1] for t in D["train"]]); nstep = len(D["paths"][0]) - 1
ax.axhspan(te.min(), te.max(), color="0.92", zorder=0); ax.text(0.5, te.max() + 6, f"energy range of the {len(te)} DFT training structures at x = 0.5", fontsize=11.5, color="0.35", va="bottom")
for key, col, lab in (("icet", fs.COL["icet"], "icet cluster expansion (linear)"), ("conv", fs.COL["conv"], "Cluster convolution")):
    Y = np.array([[r[key] for r in p] for p in D["paths"]]); xs = np.arange(nstep + 1)
    ax.fill_between(xs, Y.min(0), Y.max(0), color=col, alpha=0.18, lw=0); ax.plot(xs, Y.mean(0), "-o", color=col, lw=3.0, ms=7, mec="k", mew=0.8, label=lab, zorder=4)
ax.plot(0, 0, "*", color="k", ms=20, zorder=6, label="DFT ground state A00")
for name, (v, e) in D["dft"].items():
    if name != "A00_500": ax.plot(nstep, e, "*", color="k", ms=20, zorder=6, label=f"DFT: {name}")
ax.axhline(0, color="k", lw=1.0); ax.set_xlabel("Swaps away from the ground state (Fe$\\leftrightarrow$Ti and O$\\leftrightarrow$V$_\\mathrm{O}$, composition fixed)"); ax.set_ylabel("Energy relative to the DFT ground state (meV/site)")
ax.set_title("x = 0.5: same clusters, two models, leaving the training data", fontsize=14); ax.legend(loc="lower left", fontsize=11); ax.set_xlim(-0.5, nstep + 0.5)
ax.annotate(f"state the icet MC falls into\n{D['endpoint']['icet']:+.0f} meV/site  (10 B-site + 4 O-site swaps)", xy=(nstep, D["endpoint"]["icet"]), xytext=(-250, 30), textcoords="offset points", fontsize=11, color=fs.COL["icet"], arrowprops=dict(arrowstyle="->", color=fs.COL["icet"]))
ax.annotate(f"cluster conv., same structure: {D['endpoint']['conv']:+.0f}", xy=(nstep, D["endpoint"]["conv"]), xytext=(-210, 30), textcoords="offset points", fontsize=11, color=fs.COL["conv"], arrowprops=dict(arrowstyle="->", color=fs.COL["conv"]))
fs.save(fig, OUT)
