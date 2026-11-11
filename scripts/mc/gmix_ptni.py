"""Free energy of mixing of Pt(1-x)Ni(x) vs composition at several temperatures (same kind of figure as Fig. 6(a) of
N. Kim et al. 2022 for STF): DFT structures as crosses, the 0 K line = lowest energy found by the Monte Carlo at each
composition, coloured lines = Delta G_mix(x, T).

G(T) comes from parallel tempering (4 M^3 atoms, env M; M=3 = the 108-atom DFT cell at the DFT compositions; ladder
300 K ... infinity, slurm/mc_ptni_gmix.sh) + MBAR, anchored by the exact infinite-temperature entropy
S = k_B ln C(N, n_Ni):  beta F(beta) = [f(beta) - f(0)] - ln W.  Reference = DFT energies of pure Ni and Pt (same
protocol as the data).

Inputs: $RUNS/ptni_gmix/<icet|conv>[/M<M>]/n<n_Ni>/ (mcmc_result.npz or energies.npz), $DATA/benchmark_folds.json.
Outputs: $REPO/figures/fig_gmix_ptni[_M<M>].{png,pdf} and $RESULTS/figure_data/fig_gmix_ptni[_M<M>].json.
Env: M (default 4), TS (default "600,900,1200,1500"), GMIX_ROOT (default $RUNS/ptni_gmix).
Run: python scripts/mc/gmix_ptni.py
"""
import os, glob, json, numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import logsumexp, gammaln
from pymbar import MBAR
from neuralce.paths import REPO, DATA, RUNS, RESULTS


def _style():
    """Publication style shared by the figure scripts (bold, high-contrast)."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "font.weight": "normal", "axes.labelsize": 14, "axes.labelweight": "bold", "axes.titlesize": 14, "axes.titleweight": "bold",
        "legend.fontsize": 11.5, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "axes.linewidth": 1.8, "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.major.size": 6.5, "ytick.major.size": 6.5,
        "xtick.major.width": 1.8, "ytick.major.width": 1.8, "xtick.minor.size": 3.5, "ytick.minor.size": 3.5, "xtick.minor.width": 1.3, "ytick.minor.width": 1.3,
        "lines.linewidth": 2.4, "lines.markersize": 7, "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "mathtext.default": "regular", "axes.unicode_minus": True})


def _save(fig, stem):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png"); fig.savefig(stem + ".pdf"); print("wrote", stem + ".png / .pdf")


def _panel_label(ax, s, x=-0.20, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="left")


_style()
M = int(os.environ.get("M", "4")); N = 4 * M ** 3                                          # M = conventional cells per side (3: the 108-atom DFT cell)
G = os.environ.get("GMIX_ROOT", f"{RUNS}/ptni_gmix"); STEM = "fig_gmix_ptni" + ("" if M == 4 else f"_M{M}")
FIG = f"{REPO}/figures/{STEM}"; OUT_JSON = f"{RESULTS}/figure_data/{STEM}.json"; KB = 8.617333262e-5
SUB = "" if M == 4 else f"/M{M}"
ENi, EPt = -617.84377 / 108, -652.72821 / 108; TS = [float(t) for t in os.environ.get("TS", "600,900,1200,1500").split(",")]; CAP = 4000


def free_energy(E, T):
    """E: (R, samples) total energies (eV) at the ladder temperatures T.  Returns dict T_target -> (G, H) total eV, from MBAR anchored at beta = 0."""
    h = E.shape[1] // 2; E = E[:, h:]; idx = np.unique(np.linspace(0, E.shape[1] - 1, min(CAP, E.shape[1])).astype(int)); E = E[:, idx]
    En = E.ravel(); c = En.mean(); Es = En - c; b = 1.0 / (KB * T); Nk = np.full(len(T), E.shape[1]); u = b[:, None] * Es[None]
    f_k = np.asarray(MBAR(u, Nk, solver_protocol="robust").f_k); ld = logsumexp(np.log(Nk)[:, None] + f_k[:, None] - u, axis=0)
    f = lambda beta: -logsumexp(-beta * Es - ld); f0 = f(0.0); out = {}
    for t in TS:
        beta = 1.0 / (KB * t); lw = -beta * Es - ld; w = np.exp(lw - logsumexp(lw)); out[t] = (((f(beta) - f0) + beta * c) / beta, float((w * En).sum()))     # beta*F + ln W -> F added below
    return out


res = {}
for fw in ("icet", "conv"):
    for d in sorted(glob.glob(f"{G}/{fw}{SUB}/n*"), key=lambda p: int(p.split("/n")[-1])):
        n = int(d.split("/n")[-1]); x = n / N; lnW = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1); ref = x * ENi + (1 - x) * EPt
        try:
            if fw == "icet": z = np.load(f"{d}/mcmc_result.npz"); E = np.asarray(z["energies"], float).T; T = np.asarray(z["temperatures"], float); emin = float(z["E_min"]) if "E_min" in z.files else float(E.min())
            else: z = np.load(f"{d}/energies.npz"); E = np.asarray(z["traj"], float) * N; T = np.asarray(z["temps"], float); emin = float(E.min())
        except Exception as ex: print("skip", d, repr(ex)); continue
        o = np.argsort(T); fe = free_energy(E[o], T[o]); r = dict(x=x, H0=(min(emin, E.min()) / N - ref) * 1e3)
        for t, (fb, Ht) in fe.items(): r[f"G{t:.0f}"] = ((fb - KB * t * lnW) / N - ref) * 1e3; r[f"H{t:.0f}"] = (Ht / N - ref) * 1e3
        res.setdefault(fw, []).append(r); print(fw, n, {k: round(v, 2) for k, v in r.items()}, flush=True)
it = pd.DataFrame(json.load(open(f"{DATA}/benchmark_folds.json"))["ptni"]["items"]); xd = it.n_ni / 108.0; Hd = (it.y - (xd * ENi + (1 - xd) * EPt)) * 1e3
fws = [f for f in ("icet", "conv") if f in res]; fig, axs = plt.subplots(1, max(len(fws), 1), figsize=(8.6 * max(len(fws), 1), 7.4), squeeze=False); fig.subplots_adjust(wspace=0.22, left=0.09, right=0.98, top=0.92, bottom=0.12)
cols = ["#1a8f3a", "#f08c00", "#d62728", "#7a1fa2"]; mks = ["<", "^", "v", "D"]
for ax, fw, pl in zip(axs[0], fws, "ab"):
    rr = sorted(res[fw], key=lambda r: r["x"]); xs = [0.0] + [r["x"] for r in rr] + [1.0]
    ax.plot(xd, Hd, "x", color="0.35", ms=7, mew=1.3, label="DFT structures", zorder=2)
    ax.plot(xs, [0.0] + [r["H0"] for r in rr] + [0.0], "-X", color="k", lw=2.6, ms=9, label="0 K (lowest energy found by MC)", zorder=5)
    for t, col, mk in zip(TS, cols, mks): ax.plot(xs, [0.0] + [r[f"G{t:.0f}"] for r in rr] + [0.0], marker=mk, color=col, lw=2.6, ms=8, mec="k", mew=0.7, label=f"{t:.0f} K", zorder=4)
    ax.axhline(0, color="k", lw=1.0); ax.set_xlim(-0.02, 1.02); ax.set_xlabel(r"Ni fraction $x$, Pt$_{1-x}$Ni$_x$"); ax.set_ylabel(r"Free energy of mixing $\Delta G_{mix}$ (meV/atom)")
    ax.set_title({"icet": "icet cluster expansion", "conv": "Neural cluster expansion"}[fw], fontsize=15); ax.legend(loc="lower left", fontsize=11); _panel_label(ax, pl, x=-0.13)
_save(fig, FIG); os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True); json.dump(res, open(OUT_JSON, "w"), indent=1); print("wrote", OUT_JSON)
