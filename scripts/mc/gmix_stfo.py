"""Free energy of mixing of SrTi(1-x)Fe(x)O(3-x/2) vs the Fe fraction at several temperatures (the figure of Fig. 6(a) of
N. Kim et al. 2022): DFT structures as crosses, the 0 K line = lowest energy found by the Monte Carlo at each composition,
coloured lines = Delta G_mix(x, T).

G(T) comes from parallel tempering (K^3 x 160-site cells, ladder 300 K ... infinity, slurm/mc_stfo_gmix.sh) + MBAR,
anchored by the exact infinite-temperature entropy of the two active sublattices,
S = k_B ln [C(32K^3, n_Fe) C(96K^3, n_V)]:   beta F(beta) = [f(beta) - f(0)] - ln W.
Reference = DFT end members (SrTiO3 and Sr2Fe2O5, same protocol as the data), per 160-site cell.

Inputs: $RUNS/stfo_gmix/icet/<ICET>/c<comp>_K<K>/ and $RUNS/stfo_gmix/conv/<CONV>/c<comp>_K<K>/, $DATA/new_bench_aug467/detailed_info.csv.
Outputs: $REPO/figures/fig_gmix_stfo.{png,pdf} and $RESULTS/figure_data/fig_gmix_stfo.json.
Env: ICET (folder under stfo_gmix/icet, default icet_aug467_al2b), CONV (folder under stfo_gmix/conv, default
stfo_dd_p10_w10_wd5_aug467_al1a_w0), K (default 1), TS (default "900,1200,1500"), GMIX_ROOT (default $RUNS/stfo_gmix).
Run: python scripts/mc/gmix_stfo.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
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
G = os.environ.get("GMIX_ROOT", f"{RUNS}/stfo_gmix"); FIG = f"{REPO}/figures/fig_gmix_stfo"; OUT_JSON = f"{RESULTS}/figure_data/fig_gmix_stfo.json"; KB = 8.617333262e-5
E_STO, E_SFO = -1209.4315 / 160, -929.75272 / 160                                         # eV per site (160-site cell = 32 formula units)
TS = [float(t) for t in os.environ.get("TS", "900,1200,1500").split(",")]; CAP = 4000; K = int(os.environ.get("K", "1")); N = 160 * K ** 3
RUNS_ = {"icet": os.environ.get("ICET", "icet_aug467_al2b"), "conv": os.environ.get("CONV", "stfo_dd_p10_w10_wd5_aug467_al1a_w0")}
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{DATA}/new_bench_aug467/detailed_info.csv"] + ([] if any(os.path.isdir(f"{G}/{fw}/{n}") for fw, n in RUNS_.items()) else [f"{G}/conv/{RUNS_['conv']}"]),
        "$NCE_DATA/new_bench_aug467, GMIX_ROOT (ICET, CONV)")
ref = lambda x: (1 - x) * E_STO + x * E_SFO                                                # eV per site


def free_energy(E, T):
    """E: (R, samples) total energies (eV) at the ladder temperatures T.  Returns dict T_target -> (beta*F/beta without the ln W term, <H>) total eV, from MBAR anchored at beta = 0."""
    h = E.shape[1] // 2; E = E[:, h:]; idx = np.unique(np.linspace(0, E.shape[1] - 1, min(CAP, E.shape[1])).astype(int)); E = E[:, idx]
    En = E.ravel(); c = En.mean(); Es = En - c; b = 1.0 / (KB * T); Nk = np.full(len(T), E.shape[1]); u = b[:, None] * Es[None]
    f_k = np.asarray(MBAR(u, Nk, solver_protocol="robust").f_k); ld = logsumexp(np.log(Nk)[:, None] + f_k[:, None] - u, axis=0)
    f = lambda beta: -logsumexp(-beta * Es - ld); f0 = f(0.0); out = {}
    for t in TS:
        beta = 1.0 / (KB * t); lw = -beta * Es - ld; w = np.exp(lw - logsumexp(lw)); out[t] = (((f(beta) - f0) + beta * c) / beta, float((w * En).sum()))
    return out


res = {}
for fw, name in RUNS_.items():
    for d in sorted(glob.glob(f"{G}/{fw}/{name}/c*_K{K}"), key=lambda p: int(os.path.basename(p)[1:].split("_")[0])):
        comp = int(os.path.basename(d)[1:].split("_")[0]); x = comp / 1000.0; nFe = round(32 * K ** 3 * x); nV = round(16 * K ** 3 * x)
        lnW = (gammaln(32 * K ** 3 + 1) - gammaln(nFe + 1) - gammaln(32 * K ** 3 - nFe + 1)) + (gammaln(96 * K ** 3 + 1) - gammaln(nV + 1) - gammaln(96 * K ** 3 - nV + 1))
        try:
            if fw == "icet": z = np.load(f"{d}/mcmc_result.npz"); E = np.asarray(z["energies"], float).T; T = np.asarray(z["temperatures"], float); emin = float(z["E_min"])
            else:
                z = np.load(f"{d}/energies.npz"); E = np.asarray(z["traj"], float) * N; T = np.asarray(z["temps"], float) if "temps" in z.files else 1.0 / (KB * np.asarray(z["betas"], float)); emin = float(E.min())
        except Exception as ex: print("skip", d, repr(ex)); continue
        o = np.argsort(T); fe = free_energy(E[o], T[o]); r = dict(x=x, H0=(min(emin, E.min()) / N - ref(x)) * 1e3, n_samples=int(E.shape[1]))
        for t, (fb, Ht) in fe.items(): r[f"G{t:.0f}"] = ((fb - KB * t * lnW) / N - ref(x)) * 1e3; r[f"H{t:.0f}"] = (Ht / N - ref(x)) * 1e3
        for t in TS:                                                                            # sanity gate: S = (H - G)/T must lie in [0, ideal mixing entropy]; a violation means the ladder is not equilibrated (drifting minimum)
            S = (r[f"H{t:.0f}"] - r[f"G{t:.0f}"]) / (KB * t * 1e3); r[f"S{t:.0f}"] = S
            if not (-1e-6 <= S <= lnW / N + 1e-6): print(f"  WARNING {fw} x={x}: S({t:.0f} K) = {S:.3f} k_B/site outside [0, {lnW / N:.3f}] -> not equilibrated, do not use", flush=True)
            if r[f"G{t:.0f}"] > r["H0"] + 1e-6: print(f"  WARNING {fw} x={x}: G({t:.0f} K) = {r[f'G{t:.0f}']:.1f} above the lowest energy found {r['H0']:.1f} (F <= E_min always) -> not equilibrated, do not use", flush=True)
        res.setdefault(fw, []).append(r); print(fw, comp, {k: round(v, 2) for k, v in r.items()}, flush=True)
df = pd.read_csv(f"{DATA}/new_bench_aug467/detailed_info.csv"); xd = df.comp / 1000.0; Hd = (df.total_energy / df.n_atoms - ref(xd)) * 1e3     # DFT structures (all compositions of the data set)
fws = [f for f in ("icet", "conv") if f in res]; fig, axs = plt.subplots(1, max(len(fws), 1), figsize=(8.6 * max(len(fws), 1), 7.4), squeeze=False); fig.subplots_adjust(wspace=0.22, left=0.09, right=0.98, top=0.92, bottom=0.12)
cols = ["#f08c00", "#d62728", "#7a1fa2", "#1a8f3a"]; mks = ["^", "v", "D", "<"]
for ax, fw, pl in zip(axs[0], fws, "ab"):
    rr = sorted(res[fw], key=lambda r: r["x"]); xs = [0.0] + [r["x"] for r in rr] + [1.0]
    ax.plot(xd, Hd, "x", color="0.35", ms=7, mew=1.3, label="DFT structures", zorder=2)
    ax.plot(xs, [0.0] + [r["H0"] for r in rr] + [0.0], "-X", color="k", lw=2.6, ms=9, label="0 K (lowest energy found by MC)", zorder=5)
    for t, col, mk in zip(TS, cols, mks): ax.plot(xs, [0.0] + [r[f"G{t:.0f}"] for r in rr] + [0.0], marker=mk, color=col, lw=2.6, ms=8, mec="k", mew=0.7, label=f"{t:.0f} K", zorder=4)
    ax.axhline(0, color="k", lw=1.0); ax.set_xlim(-0.02, 1.02); ax.set_xlabel(r"Fe fraction $x$, SrTi$_{1-x}$Fe$_x$O$_{3-x/2}$"); ax.set_ylabel(r"Free energy of mixing $\Delta G_{mix}$ (meV/site)")
    ax.set_title({"icet": "icet cluster expansion", "conv": "Neural cluster expansion"}[fw] + f"  ({N} sites)", fontsize=15); _panel_label(ax, pl, x=-0.13)
h, l = axs[0][0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(l), fontsize=12)   # one legend above the panels, off the data
_save(fig, FIG); os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True); json.dump(dict(runs=RUNS_, K=K, N=N, results=res), open(OUT_JSON, "w"), indent=1); print("wrote", OUT_JSON)
