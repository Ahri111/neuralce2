"""Data of Fig. 5e, f: why the cluster convolution gets the 2NN short-range order of Fe42.5Cr7.5Ni50 right.  For the MC configurations already explained
by exact Shapley values (results/shap/fenicr.json), compute their Warren-Cowley (Fe,Cr)-Ni SRO at 1NN and 2NN and plot the Shapley contribution of the
2NN spin-pair channels and of the 2NN chemical-pair channels against alpha_2NN (colour = temperature).  If the energy gain that drives the 2NN order
comes from the spin channels (slope in the spin panel, none in the chemical panel), a chemistry-only CE, or a CE with a fixed 2NN spin coupling, cannot
reproduce it.
Inputs:  results/shap/fenicr.json; the configurations it names, from $FENICR_MC_DIR/<model>_dense_<label>/multiT/ (default results/fenicr_sro; both F5 and F6 are shipped;
         missing files are skipped with a message).  COMP selects the composition of the quick-look plot (default F6).
Outputs: results/figure_data/fig7_fenicr_shap2nn.json (read by scripts/figures/fig5_stf_fenicr.py) and a quick-look figures/quicklook/fig7_fenicr_shap2nn.{png,pdf}.
Run:     python scripts/analysis/fenicr_shap_2nn.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")); import figstyle as fs; fs.use()
from neuralce.paths import REPO, RESULTS
def require(paths, hint):                                                                 # the Fe-Ni-Cr inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{RESULTS}/shap/fenicr.json", os.environ.get("FENICR_MC_DIR", f"{RESULTS}/fenicr_sro")], "$NCE_RESULTS/shap, FENICR_MC_DIR")
import matplotlib.pyplot as plt, matplotlib as mpl
from ase.io import read
from ase.neighborlist import neighbor_list
D = json.load(open(f"{RESULTS}/shap/fenicr.json")); TAG = D["model"]; base = f"{os.environ.get('FENICR_MC_DIR', f'{RESULTS}/fenicr_sro')}/{TAG}_dense"
OUT_JSON = f"{RESULTS}/figure_data/fig7_fenicr_shap2nn.json"; QL = f"{REPO}/figures/quicklook"; os.makedirs(QL, exist_ok=True); os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
def alphas(a):
    """(Fe,Cr)-Ni Warren-Cowley SRO at 1NN (2.55 A) and 2NN (3.60 A), composition-weighted over Fe and Cr as in Su et al."""
    i, j, d = neighbor_list("ijd", a, 4.0); s = np.array(a.get_chemical_symbols()); cNi = (s == "Ni").mean(); out = []
    for lo, hi in ((2.2, 3.1), (3.1, 4.0)):
        m = (d > lo) & (d < hi); al = {}
        for X in ("Fe", "Cr"):
            mi = m & (s[i] == X); al[X] = 1.0 - (s[j][mi] == "Ni").mean() / cNi
        cF, cC = (s == "Fe").mean(), (s == "Cr").mean(); out.append((cF * al["Fe"] + cC * al["Cr"]) / (cF + cC))
    return out
rows = []
for r in D["rows"]:
    f = f"{base}_{r['label']}/multiT/{r['file']}"
    if not os.path.exists(f): print("missing", f); continue
    a1, a2 = alphas(read(f)); rows.append(dict(label=r["label"], T=r["T"], a1=a1, a2=a2, **{g: r["phi"][g] for g in D["groups"]})); print(r["label"], int(r["T"]), f"a1 {a1:+.3f} a2 {a2:+.3f}", flush=True)
json.dump(dict(model=TAG, groups=D["groups"], rows=rows), open(OUT_JSON, "w"), indent=1); print("wrote", OUT_JSON)
rows = [q for q in rows if q["label"] == os.environ.get("COMP", "F6")]                                              # one composition (F6 = Fe42.5Cr7.5Ni50, the Su et al. Fig. 6 alloy)
T = np.array([q["T"] for q in rows]); a2 = np.array([q["a2"] for q in rows]); a1 = np.array([q["a1"] for q in rows]); norm = mpl.colors.Normalize(T.min(), T.max())
fig, axs = plt.subplots(1, 4, figsize=(22, 5.6)); fig.subplots_adjust(wspace=0.32, left=0.05, right=0.97, bottom=0.15, top=0.88)
for ax, g, x, xl in zip(axs, ("spin pairs 2NN", "chem. pairs 2NN", "spin pairs 1NN", "chem. pairs 1NN"), (a2, a2, a1, a1), (r"$\alpha_{2NN}$", r"$\alpha_{2NN}$", r"$\alpha_{1NN}$", r"$\alpha_{1NN}$")):
    y = np.array([q[g] for q in rows]); y = y - y.mean(); ax.scatter(x, y, c=T, cmap="coolwarm", norm=norm, s=70, ec="k", lw=0.5); k = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 10); ax.plot(xx, np.polyval(k, xx), "k--", lw=1.5)
    ax.set_xlabel(xl + "  ((Fe,Cr)–Ni)"); ax.set_ylabel(f"SHAP: {g} (meV/atom, centred)"); ax.set_title(f"{g}: slope {k[0]:.0f} meV/atom per unit " + r"$\alpha$", fontsize=12); ax.axhline(0, color="0.6", lw=1)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=axs, fraction=0.015, pad=0.01, label="T (K)"); fs.save(fig, f"{QL}/fig7_fenicr_shap2nn")
