"""Analysis of a two-phase PtNi run (slurm/mc_ptni_two_phase.sh): free energy of each branch, T_c, finite-N C_v.

Per replica (no exchange: temperature slot = replica): burn-in BURN of the samples, then the samples that are in the replica's own phase
(in the window where both branches exist, a sample counts as ordered when its energy is below the midpoint of the two branch curves; a replica
that spends most of the run in the other phase is dropped).  Per-atom energy e(T) and variance of each branch.
Free energy per atom, g = F / (N k_B T), relative to c = e of the ordered branch at its lowest temperature (~ the ordered ground state):
    ordered    g_o(b) = b c - int_b^{b_max} (e_o - c) db' - ln(q)/N          q = ordered variants (L1_0: 6, L1_2: 4)
    disordered g_d(b) = b c - s_conf + int_0^b (e_d - c) db'                 s_conf = ln C(N, N_Ni) / N  (exact random limit)
(b = 1/k_B T; piecewise-cubic monotone interpolation of e(b), integrated exactly; trapezoid given for comparison.)
T_c(inf) = where g_o = g_d without the 1/N terms.  Finite N (interface states neglected, exact for large cells):
    p_o = 1 / (1 + exp(N (g_o - g_d))),  C_v/(N k_B) = [p_o Var_o + p_d Var_d + p_o p_d N^2 (e_d - e_o)^2] / (N (k_B T)^2).
Errors: the whole analysis repeated on NB contiguous time blocks.

Env: RUN (run dir with branch.json + energies.npz or checkpoint.npz), BURN (0.2), NB (4), PT_CACHE (optional json of the PT MBAR curve for
the check plot), ANCHOR (json of a full-range analysis of the same composition, for WINDOW_ONLY runs: T_c(inf) taken from it and g_d - g_o
integrated from there with this run's energies) or ANCHOR_T + ANCHOR_N (a finite-N T_c), N_CURVE (draw the curve for another N),
CV_MODE (derivative | variance), CV_LAM, CV_SLOPE_W, OUT (stem; default RUN/two_phase).
Outputs: OUT.json + OUT.png, and a copy of the json at $RESULTS/ptni_two_phase/<fw>_<x>_<M dir>.json.
Run: RUN=... python scripts/mc/ptni_two_phase_analyze.py
"""
import os, json, numpy as np
from math import lgamma, log
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from neuralce.paths import RESULTS

KB = 8.617333262e-5
RUN = os.environ["RUN"]; BURN = float(os.environ.get("BURN", "0.2")); NB = int(os.environ.get("NB", "4")); OUT = os.environ.get("OUT", f"{RUN}/two_phase")
B = json.load(open(f"{RUN}/branch.json")); z = np.load(f"{RUN}/energies.npz" if os.path.exists(f"{RUN}/energies.npz") else f"{RUN}/checkpoint.npz"); N = int(B["N"])
if "traj" in z.files:                                                              # nce: traj (R, samples) in eV/atom
    E = np.asarray(z["traj"], float).T * N; T = np.asarray(z["temps"], float)
else:                                                                              # icet: energies (samples, R) total eV
    E = np.asarray(z["energies"] if "energies" in z.files else z["energy_samples"], float); T = np.asarray(z["temperatures"] if "temperatures" in z.files else z["cfg_temps"], float)
br = np.array(B["branch"]); assert np.allclose(T, B["temps"]), "temperatures differ from branch.json"
q = 6 if B["x"] == "x50" else 4; nNi = int(B["n_Ni"]); x = nNi / N; TC0 = float(B["TC0"])
s_N = (lgamma(N + 1) - lgamma(nNi + 1) - lgamma(N - nNi + 1)) / N; s_inf = -(x * log(x) + (1 - x) * log(1 - x))
e_all = E[int(E.shape[0] * BURN):] / N                                           # (samples, replicas), eV/atom
S = e_all.shape[0]; print(f"{RUN}: N={N}, {E.shape[0]} samples, {S} after burn-in, {len(T)} replicas", flush=True)
io, id_ = np.where(br == "ordered")[0], np.where(br == "disordered")[0]
lo_w, hi_w = T[id_].min(), T[io].max()                                           # window where both branches exist

def branch_fit(idx, med):
    """robust smooth curve through the replica medians of one branch inside the window (quadratic, 3 rejection passes)"""
    k = idx[(T[idx] >= lo_w - 1) & (T[idx] <= hi_w + 1)]; keep = np.ones(len(k), bool)
    for _ in range(3):
        c = np.polyfit(T[k][keep], med[k][keep], 2); r = med[k] - np.polyval(c, T[k]); s = 1.4826 * np.median(np.abs(r[keep])) + 1e-7; keep = np.abs(r) < 5 * s
    return c

def analyse(e, tag):
    med = np.median(e, 0)
    co, cd = branch_fit(io, med), branch_fit(id_, med)
    mid = lambda t: 0.5 * (np.polyval(co, t) + np.polyval(cd, t))
    own = np.ones_like(e, bool); win = (T >= lo_w) & (T <= hi_w)
    for r in np.where(win)[0]:
        own[:, r] = (e[:, r] < mid(T[r])) if br[r] == "ordered" else (e[:, r] > mid(T[r]))
    frac = own.mean(0); ok = frac > 0.5
    m = np.array([e[own[:, r], r].mean() if ok[r] else np.nan for r in range(len(T))])
    v = np.array([e[own[:, r], r].var() if ok[r] else np.nan for r in range(len(T))])
    c = m[io][np.argmin(T[io])]                                                  # ordered branch at its lowest temperature
    bo, yo = 1 / (KB * T[io][ok[io]]), m[io][ok[io]] - c; o = np.argsort(bo); bo, yo = bo[o], yo[o]
    bd, yd = 1 / (KB * T[id_][ok[id_]]), m[id_][ok[id_]] - c; o = np.argsort(bd); bd, yd = bd[o], yd[o]
    y0 = yd[0] - (yd[1] - yd[0]) / (bd[1] - bd[0]) * bd[0]                       # e_d at b = 0 (linear from the two hottest replicas)
    bd0, yd0 = np.concatenate([[0.0], bd]), np.concatenate([[y0], yd])
    Po, Pd = PchipInterpolator(bo, yo).antiderivative(), PchipInterpolator(bd0, yd0).antiderivative()
    I_o = lambda b: Po(bo[-1]) - Po(b); I_d = lambda b: Pd(b) - Pd(0.0)
    ct_o = np.concatenate([np.cumsum((0.5 * (yo[1:] + yo[:-1]) * np.diff(bo))[::-1])[::-1], [0.0]])       # trapezoid, from b_max down
    ct_d = np.concatenate([[0.0], np.cumsum(0.5 * (yd0[1:] + yd0[:-1]) * np.diff(bd0))])[1:]
    Tr_o, Tr_d = PchipInterpolator(bo, ct_o), PchipInterpolator(bd, ct_d)
    dg_inf = lambda b: -s_inf + I_d(b) + I_o(b)
    dg_N = lambda b: -s_N + I_d(b) + I_o(b) + log(q) / N
    b_lo, b_hi = max(bo.min(), bd.min()), min(bo.max(), bd.max())                 # overlap of the two branches in b
    def root(f):
        bb = np.linspace(b_lo, b_hi, 2001); fb = np.array([f(t) for t in bb]); s = np.where(np.diff(np.sign(fb)) != 0)[0]
        return (1 / (KB * brentq(f, bb[s[0]], bb[s[0] + 1]))) if len(s) else np.nan
    Tc_inf, Tc_N = root(dg_inf), root(dg_N); Tc_trap = root(lambda b: -s_inf + Tr_d(b) + Tr_o(b))
    if tag == "all":                                                              # quadrature check: T_c(inf) with other interpolants and with half of the coarse points
        from scipy.interpolate import CubicSpline, Akima1DInterpolator
        def tc_with(kind, keep_o=None, keep_d=None):
            bo_, yo_ = (bo, yo) if keep_o is None else (bo[keep_o], yo[keep_o]); bd_, yd_ = (bd0, yd0) if keep_d is None else (bd0[keep_d], yd0[keep_d])
            mk = {"pchip": PchipInterpolator, "cubic": lambda x_, y_: CubicSpline(x_, y_, bc_type="natural"), "akima": Akima1DInterpolator}
            if kind == "trapz":
                co_ = np.concatenate([np.cumsum((0.5 * (yo_[1:] + yo_[:-1]) * np.diff(bo_))[::-1])[::-1], [0.0]]); cd_ = np.concatenate([[0.0], np.cumsum(0.5 * (yd_[1:] + yd_[:-1]) * np.diff(bd_))])
                fo = lambda b: np.interp(b, bo_, co_); fd = lambda b: np.interp(b, bd_, cd_)
            else:
                Ao, Ad = mk[kind](bo_, yo_).antiderivative(), mk[kind](bd_, yd_).antiderivative(); fo = lambda b: Ao(bo_[-1]) - Ao(b); fd = lambda b: Ad(b) - Ad(0.0)
            return root(lambda b: -s_inf + fd(b) + fo(b))
        dense_o = np.abs(1 / (KB * bo) - TC0) <= 45; dense_d = np.concatenate([[True, True], np.abs(1 / (KB * bd) - TC0)[1:] <= 45])
        half_o = dense_o | (np.arange(len(bo)) % 2 == 0) | (np.arange(len(bo)) == len(bo) - 1); half_d = dense_d | (np.arange(len(bd0)) % 2 == 0)
        qt = {k: tc_with(k) for k in ("trapz", "pchip", "cubic", "akima")}; qh = {k: tc_with(k, half_o, half_d) for k in ("trapz", "pchip", "cubic", "akima")}
        print("  quadrature check, T_c(inf) [K]  all points: " + "  ".join(f"{k} {v:.2f}" for k, v in qt.items()) + "  | half of the coarse points: " + "  ".join(f"{k} {v:.2f}" for k, v in qh.items()), flush=True)
        for b_ in (b_lo, 0.5 * (b_lo + b_hi), b_hi):
            print(f"  T {1/(KB*b_):8.2f} K: s_inf {s_inf:.5f}  I_d {float(I_d(b_)):.5f}  I_o {float(I_o(b_)):.5f}  dg_inf {float(dg_inf(b_)):+.5f}", flush=True)
        print(f"  c = {c*1e3:.3f} meV/atom; e_d - c at the hottest replica {yd[0]*1e3:.2f} meV, e_o - c at the hottest ordered {yo[0]*1e3:.2f} meV", flush=True)
    fin = dict(tag=tag, Tc_inf=Tc_inf, Tc_N=Tc_N, Tc_inf_trapezoid=Tc_trap, dropped=[float(t) for t in T[~ok]],
               contamination=[(float(T[r]), br[r], float(1 - frac[r])) for r in np.where(win & (frac < 0.999))[0]])
    return fin, dict(m=m, v=v, c=c, ok=ok, frac=frac, dg_N=dg_N, dg_inf=dg_inf, co=co, cd=cd)

res, st = analyse(e_all, "all")
anchor = os.environ.get("ANCHOR") or os.environ.get("ANCHOR_T")
if anchor:                                                                         # T_c(inf) from outside: a full-range analysis (ANCHOR=json) or a finite-N T_c (ANCHOR_T, ANCHOR_N)
    m = st["m"]; ok = st["ok"]
    eo_ = PchipInterpolator(1 / (KB * T[io][ok[io]])[::-1], m[io][ok[io]][::-1]); ed_ = PchipInterpolator(1 / (KB * T[id_][ok[id_]])[::-1], m[id_][ok[id_]][::-1])
    bb = np.linspace(max(1 / (KB * T[io][ok[io]].max()), 1 / (KB * T[id_][ok[id_]].max())), min(1 / (KB * T[io][ok[io]].min()), 1 / (KB * T[id_][ok[id_]].min())), 4001)
    dd = PchipInterpolator(bb, ed_(bb) - eo_(bb)).antiderivative()
    if os.environ.get("ANCHOR_T"):                                                 # T_c(N_ref) -> T_c(inf): N_ref (g_d - g_o) = 0 at T_ref with the finite-N terms
        Tr, Nr = float(os.environ["ANCHOR_T"]), int(os.environ["ANCHOR_N"]); br_ = 1 / (KB * Tr)
        s_r = (lgamma(Nr + 1) - lgamma(round(x * Nr) + 1) - lgamma(Nr - round(x * Nr) + 1)) / Nr; corr = (s_inf - s_r) + log(q) / Nr
        Tc_a = float(1 / (KB * brentq(lambda b: dd(b) - dd(br_) - corr, br_ - 0.2, br_ + 0.2))); print(f"anchor: T_c(N={Nr}) = {Tr:.2f} K -> T_c(inf) = {Tc_a:.2f} K", flush=True)
    else:
        Tc_a = json.load(open(anchor))["Tc_inf"]
    bc = 1 / (KB * Tc_a)
    st["dg_inf"] = lambda b: dd(b) - dd(bc); st["dg_N"] = lambda b: dd(b) - dd(bc) + (s_inf - s_N) + log(q) / N
    b_lo, b_hi = bb[0], bb[-1]; gg = np.linspace(b_lo, b_hi, 2001); fb = st["dg_N"](gg); s_ = np.where(np.diff(np.sign(fb)) != 0)[0]
    res["Tc_N"] = float(1 / (KB * brentq(st["dg_N"], gg[s_[0]], gg[s_[0] + 1]))) if len(s_) else np.nan
    res["Tc_inf"] = Tc_a; res["Tc_inf_from"] = anchor; res["Tc_inf_trapezoid"] = np.nan; print(f"T_c(inf) from the anchor: {Tc_a:.2f} K", flush=True)
blocks = [analyse(bk, f"block {i}")[0] for i, bk in enumerate(np.array_split(e_all, NB))] if not anchor and NB > 1 else []
if blocks:
    tb = np.array([b_["Tc_inf"] for b_ in blocks]); res["Tc_inf_blocks"] = tb.tolist(); res["Tc_inf_err"] = float(np.nanstd(tb) / np.sqrt(NB))

# finite-N C_v on a fine grid.  N_CURVE: the same branch data, drawn for another (larger) N: per-atom branch properties are size independent
# away from the transition; only the phase weights (N (g_d - g_o)) and the latent-heat term carry N.
N_RUN = N; N = int(os.environ.get("N_CURVE", N_RUN))
if N != N_RUN:
    s_N = (lgamma(N + 1) - lgamma(round(x * N) + 1) - lgamma(N - round(x * N) + 1)) / N; dgi = st["dg_inf"]
    st["dg_N"] = lambda b: dgi(b) + (s_inf - s_N) + log(q) / N
    bq = np.linspace(1 / (KB * (res["Tc_inf"] + 5)), 1 / (KB * (res["Tc_inf"] - 5)), 2001); fb = st["dg_N"](bq); s_ = np.where(np.diff(np.sign(fb)) != 0)[0]
    res["Tc_N"] = float(1 / (KB * brentq(st["dg_N"], bq[s_[0]], bq[s_[0] + 1]))); res["N_run"] = N_RUN; print(f"curve drawn for N = {N} from the N = {N_RUN} branch data", flush=True)
m, v, ok = st["m"], st["v"], st["ok"]
W_SLOPE = float(os.environ.get("CV_SLOPE_W", "0")); Tc_ref = res["Tc_inf"]; CV_MODE = os.environ.get("CV_MODE", "derivative"); CV_LAM = (float(os.environ["CV_LAM"]) if os.environ.get("CV_LAM") else None)   # CV_MODE derivative (default) | variance                # env CV_SLOPE_W (K): ordered-branch C_v from de/dT in [T_c - W, T_c]
def branch_interp(idx):
    k = idx[ok[idx]]; t = T[k]; kv = k[v[k] > 0]                                  # frozen replicas (no accepted swap at the lowest T) have zero variance
    from scipy.interpolate import make_smoothing_spline                             # branch C_v: smooth in T (per-replica variances carry sampling noise); GCV picks the smoothing
    kv = kv[(T[kv] < 5000) & (st["frac"][kv] >= 0.99)]                              # replicas that partly changed phase: their variance is not the branch's
    vv = v.copy()
    if W_SLOPE > 0 and br[idx[0]] == "ordered":                                    # just below T_c the ordered replicas drift within the run (weak first-order
        near = kv[(T[kv] > Tc_ref - W_SLOPE) & (T[kv] <= Tc_ref + 1)]                #  transition), so their fluctuation C_v is inflated: use c = de/dT there instead
        fitk = k[(T[k] > Tc_ref - W_SLOPE - 15) & (T[k] <= Tc_ref + 1)]
        if len(near) and len(fitk) >= 4:
            cf = np.polyfit(T[fitk], m[fitk], 2); dedT = np.polyval(np.polyder(cf), T[near])
            vv[near] = np.maximum(dedT, 1e-9) * KB * T[near] ** 2 / N_RUN          # v = c (k T)^2 / N with c = (de/dT) / k_B
    if CV_MODE == "derivative":                                                    # branch C_v = de/dT of the SMOOTHED mean energy (the per-replica variances of a
        kk = k[(st["frac"][k] >= 0.99) & (T[k] < 5000)]; tt = T[kk]; o_ = np.argsort(tt); tt, mm = tt[o_], m[kk][o_]   #  10^6-atom cell are not converged: halves differ 2-3x)
        es = make_smoothing_spline(tt, mm, lam=CV_LAM); c_ = lambda x: np.maximum(es.derivative()(x), 1e-9) / KB    # c in k_B per atom
        tt_chk = np.linspace(tt.min(), tt.max(), 400); cc = c_(tt_chk); mono = bool(np.all(np.diff(cc) > -1e-6 * np.abs(cc[1:]).max()))
        print(f"  {br[idx[0]]} branch C_v from de/dT: {len(kk)} replicas, {tt.min():.0f}-{tt.max():.0f} K, lam={'GCV' if CV_LAM is None else CV_LAM}, monotone {mono}", flush=True)
        class _LV:                                                                 # same interface as the log-variance spline: exp(lv(T)) = v = c (k T)^2 / N_RUN
            def __call__(self, x): x = np.asarray(x, float); return np.log(c_(x) * (KB * x) ** 2 / N_RUN)
        return PchipInterpolator(t, m[k], extrapolate=True), _LV(), t.min(), t.max()
    lv = make_smoothing_spline(T[kv], np.log(vv[kv]))
    return PchipInterpolator(t, m[k], extrapolate=True), lv, t.min(), t.max()
eo, lvo, to_lo, to_hi = branch_interp(io); ed, lvd, td_lo, td_hi = branch_interp(id_)
TcN = res["Tc_N"] if np.isfinite(res["Tc_N"]) else res["Tc_inf"]
de_c = float(ed(TcN) - eo(TcN)); w = KB * TcN ** 2 / (N * de_c)                    # width of the finite-N peak (K)
Tg = np.unique(np.concatenate([np.arange(max(to_lo, 300.0), min(td_hi, 3000.0), 0.05), TcN + w * np.linspace(-30, 30, 6001)]))
Tg = Tg[(Tg >= to_lo) & (Tg <= td_hi)]
def cv_curve(Tg):
    b = 1 / (KB * Tg); dgN = np.array([st["dg_N"](t) if (t >= 1 / (KB * min(to_hi, td_hi)) and t <= 1 / (KB * max(to_lo, td_lo))) else np.nan for t in b])
    dgN = np.where(np.isnan(dgN), np.where(Tg < TcN, 1.0, -1.0), dgN)                   # outside the overlap: pure ordered below / disordered above
    x_ = np.clip(N * dgN, -700, 700); p_o = 1 / (1 + np.exp(-x_)); p_d = 1 - p_o          # p_o = 1/(1+exp(N(g_o-g_d))) = 1/(1+exp(-N dg)), dg = g_d - g_o
    To_, Td_ = np.clip(Tg, to_lo, to_hi), np.clip(Tg, td_lo, td_hi)                     # no extrapolation outside a branch (its weight is 0 there)
    Vo, Vd = np.exp(lvo(To_)), np.exp(lvd(Td_)); dE = ed(Td_) - eo(To_)
    return ((p_o * Vo + p_d * Vd) * N_RUN + p_o * p_d * dE ** 2 * N) / (KB * Tg) ** 2, p_o     # branch fluctuations are intensive (N_RUN var of e), latent term scales with N
cv, p_o = cv_curve(Tg)
res.update(N=N, x=B["x"], q=q, s_conf_N=s_N, s_conf_inf=s_inf, latent_meV=de_c * 1e3, peak_width_K=w, Tc_peak=float(Tg[np.argmax(cv)]), cv_max=float(cv.max()),
           T=Tg.tolist(), cv=cv.tolist(), p_ordered=p_o.tolist(),
           replicas=[dict(T=float(T[r]), branch=br[r], e=float(m[r]) if ok[r] else None, cv=float(N_RUN * v[r] / (KB * T[r]) ** 2) if ok[r] else None) for r in range(len(T))], cv_mode=CV_MODE,
           burn=BURN, samples=int(S))
json.dump(res, open(OUT + ".json", "w"))
print(f"T_c(inf) {res['Tc_inf']:.2f} K" + (f" +- {res['Tc_inf_err']:.2f} (blocks {np.round(res['Tc_inf_blocks'], 2).tolist()})" if 'Tc_inf_err' in res else "")
      + f", trapezoid {res.get('Tc_inf_trapezoid', np.nan):.2f}; T_c(N={N}) {res['Tc_N']:.2f} K, C_v peak {res['Tc_peak']:.2f} K, C_v max {res['cv_max']:.1f}, "
      f"latent heat {de_c*1e3:.2f} meV/atom, peak width {w:.3g} K", flush=True)
print("dropped replicas:", res["dropped"], "; contaminated (T, branch, fraction of samples in the other phase):", [(round(a, 1), b_, round(f, 3)) for a, b_, f in res["contamination"]], flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for idx, col, lab in ((io, "C0", "ordered start"), (id_, "C3", "random start")):
    k = idx[ok[idx]]; ax[0].plot(T[k], m[k] * 1e3, "o", ms=3, color=col, label=lab)
    ax[1].semilogy(T[k], N_RUN * v[k] / (KB * T[k]) ** 2, "o", ms=3, color=col, label=lab)
tw = np.linspace(lo_w - 5, hi_w + 5, 400); ax[0].set_xlim(lo_w - 30, hi_w + 30)
yl = (m[np.isfinite(m) & (T > lo_w - 30) & (T < hi_w + 30)] * 1e3); ax[0].set_ylim(yl.min() - 1, yl.max() + 1)
ax[0].axvline(res["Tc_inf"], color="k", ls="--", lw=1); ax[0].set_xlabel("T (K)"); ax[0].set_ylabel("e (meV/atom)"); ax[0].legend(); ax[0].set_title("branch energies (window)")
ax[1].semilogy(Tg, cv, "k-", lw=1.2, label=f"two-phase, N={N}"); ax[1].set_xlim(res["Tc_inf"] - 60, res["Tc_inf"] + 60)
pt = os.environ.get("PT_CACHE")
if pt and os.path.exists(pt):
    P = json.load(open(pt)); ax[1].semilogy(P["T"], P["cv"], "-", color="C2", lw=1.5, label=f"PT + MBAR (peak {P['T'][int(np.argmax(P['cv']))]:.1f} K)"); res["pt_peak"] = float(P["T"][int(np.argmax(P["cv"]))])
ax[1].set_xlabel("T (K)"); ax[1].set_ylabel("C_v (k_B/atom)"); ax[1].legend(fontsize=9); ax[1].set_title(f"T_c(inf) {res['Tc_inf']:.2f}, peak {res['Tc_peak']:.2f} K")
bb = np.linspace(1 / (KB * (hi_w)), 1 / (KB * (lo_w)), 300); ax[2].plot(1 / (KB * bb), [st["dg_inf"](b) * 1e3 for b in bb], "k-"); ax[2].axhline(0, color="0.5")
ax[2].set_xlabel("T (K)"); ax[2].set_ylabel("(g_d - g_o) x 1e3  (per atom, units of k_B T)"); ax[2].set_title("free-energy difference")
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=110); json.dump(res, open(OUT + ".json", "w")); print("wrote", OUT + ".json", OUT + ".png")
# tracked copy of the small result (figure input): results/ptni_two_phase/<fw>_<x>_<M dir>.json
_mdir = os.path.basename(os.path.dirname(os.path.abspath(OUT)))
_res_json = f"{RESULTS}/ptni_two_phase/{B.get('fw', 'icet')}_{B['x']}_{_mdir}.json"
os.makedirs(os.path.dirname(_res_json), exist_ok=True); json.dump(res, open(_res_json, "w")); print("wrote", _res_json)
