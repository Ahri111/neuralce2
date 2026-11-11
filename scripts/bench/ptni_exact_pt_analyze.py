"""Reference analysis of the PtNi parallel-tempering runs (slurm/ptni_exact_pt.sh) with the evidence written out.

Estimator.  For a temperature ladder the reduced potential is u_k(x) = beta_k E(x), so MBAR depends on the samples only through E.
All post-burn-in samples of all temperatures are put on one fine energy grid (bin width ~ 1e-5 .. 3e-4 eV of TOTAL energy, the bin
energy is the mean of the samples in it -> binning error ~ (beta d)^2 / 24 < 1e-5 relative), each sample weighted by 1 / g_k
(g_k = statistical inefficiency of its temperature; N_k^eff = N_k / g_k), and the MBAR equations are solved by Newton iteration:
    f_i = -ln sum_b M_b exp(-beta_i E_b) / sum_k N_k^eff exp(f_k - beta_k E_b),      M_b = sum_k H_k(b) / g_k.
No thinning: every sample is used.  Absolute free energy: Z(beta) / Z(0) from the same reweighting evaluated at beta = 0 exactly, and
Z(0) = C(N, N_Pt).  Errors: delete-one-run jackknife of the pooled estimate over the independent runs (+ per-run values listed).

Checks (each one is a number in the report, not a statement):
  C1 model md5                                   C2 end points: lowest T vs L1_0 energy, 1e8 K vs the random-mixing energy
  C3 float32 drift of the carried energy         C4 exchange: acceptance of every neighbour pair (from the simultaneous energies)
  C5 burn-in (equilibration detection, per run = the latest of its temperatures)  and g_k, N_eff per temperature
  C6 four time blocks vs their own standard error      C7 first half vs second half        C8 ordered start vs random start
  C9 independent runs agree within their own error bars (chi^2 per temperature)
  C10 order parameter from the configuration snapshots: L1_0 at low T (also validates the atom-order mapping), the 6 (or fewer)
      L1_0 variants visited, number of order <-> disorder switches seen in every temperature slot
  C11 (48 atoms) every quantity against the exact enumeration, and the per-temperature mean / variance of every run against it.
Inputs: PT runs $RUNS/ptni_exact/pt/N<N>/* (--runs overrides), ladder cells $RESULTS/ptni_exact/ladder (ptni_exact_ladder.py),
model $MODELS/nce/nipt_v3_nomax_sel/model.pkl, exact enumeration $RESULTS/ptni_exact/exact_N<N>.{json,npz} (ptni_exact_enum.py --merge).
Output: <out>/analysis_N<N>.json and report_N<N>.md (--out, default $RESULTS/ptni_exact/N<N>[_interim]).  CPU only (JAX_PLATFORMS=cpu).
usage:  python scripts/bench/ptni_exact_pt_analyze.py --N 256 [--interim] [--runs DIR ...] [--out DIR]
        python scripts/bench/ptni_exact_pt_analyze.py --selftest"""
import os, sys, glob, json, math, argparse, hashlib, time, pickle
import numpy as np
from scipy.special import logsumexp
from scipy.fft import rfft, irfft, next_fast_len
from neuralce.paths import MODELS, RUNS, RESULTS
KB = 8.617333262e-5; E_L10, E_RAND = -5.99437, -5.93830; A_FCC = 3.924
MODEL = f"{MODELS}/nce/nipt_v3_nomax_sel/model.pkl"; MD5_REF = "f3d82e3094a2f0c4c0315b86711ee77e"
BASE = f"{RUNS}/ptni_exact/pt"; LAD = f"{RESULTS}/ptni_exact/ladder"
NBINS = 250000


def log(*a): print(*a, flush=True)


# ------------------------------------------------------------------------------------------------------------------ time series
def stat_ineff(x, c=6.0):
    """g = 1 + 2 sum_t rho_t with Sokal's automatic window (smallest M with M >= c g(M)); FFT autocorrelation."""
    x = np.asarray(x, np.float64); n = len(x); x = x - x.mean(); v = float(x @ x)
    if n < 8 or v <= 0: return 1.0
    nf = next_fast_len(2 * n); f = rfft(x, nf); acf = irfft(f * np.conj(f), nf)[:n]; acf /= acf[0]
    g = 2.0 * np.cumsum(acf) - 1.0; m = np.arange(n) < c * g; w = int(np.argmin(m)) if not m.all() else n - 1
    return float(max(g[w], 1.0))


def detect_t0(x, nblk=2048):
    """first equilibrated sample (Chodera: maximise N_eff of the remainder) on the series averaged into nblk blocks."""
    from pymbar import timeseries
    L = max(len(x) // nblk, 1); b = x[:(len(x) // L) * L].reshape(-1, L).mean(1)
    if b.std() == 0: return 0
    try: t0, g, ne = timeseries.detect_equilibration(b, fast=True, nskip=8)
    except Exception: return 0
    return int(t0) * L


def block_se(x, nb=32):
    L = len(x) // nb
    if L < 2: return float("nan")
    return float(x[:nb * L].reshape(nb, L).mean(1).std(ddof=1) / np.sqrt(nb))


# ------------------------------------------------------------------------------------------------------------------ histogram MBAR
class HistMBAR:
    """M (B,) effective counts per energy bin, Eb (B,) bin energies (total eV), beta (K,), Neff (K,)."""
    def __init__(self, M, Eb, beta, Neff, Emean=None, tol=1e-11):
        keep = M > 0; self.M = M[keep]; self.Esh = float(np.average(Eb[keep], weights=M[keep])); self.E = Eb[keep] - self.Esh
        self.beta = np.asarray(beta, float); self.N = np.asarray(Neff, float); self.lnM = np.log(self.M); self.lnN = np.log(self.N)
        self.A = -self.beta[:, None] * self.E[None, :]                                  # (K, B)
        K = len(beta); f = np.zeros(K)
        if Emean is not None:                                                         # d f / d beta = <E>: trapezoid start
            em = np.asarray(Emean) - self.Esh; f[1:] = np.cumsum(np.diff(self.beta) * 0.5 * (em[1:] + em[:-1]))
        for _ in range(10): f = self._sc(f)
        self.f, self.resid, self.n_newton = self._newton(f, tol)
        self.logden = logsumexp(self.lnN[:, None] + self.f[:, None] + self.A, axis=0)

    def _sc(self, f):
        ld = logsumexp(self.lnN[:, None] + f[:, None] + self.A, axis=0); fn = -logsumexp(self.lnM[None, :] + self.A - ld[None, :], axis=1)
        return fn - fn[0]

    def _phi(self, f):
        return float(self.M @ logsumexp(self.lnN[:, None] + f[:, None] + self.A, axis=0) - self.N @ f)

    def _newton(self, f, tol, maxit=100):
        res = np.inf
        for it in range(maxit):
            ld = logsumexp(self.lnN[:, None] + f[:, None] + self.A, axis=0); lW = f[:, None] + self.A - ld[None, :]
            P = np.exp(lW + self.lnM[None, :]); s = P.sum(1); res = float(np.abs(s - 1).max())
            if res < tol: return f, res, it
            g = self.N * (s - 1.0); H = np.diag(self.N * s) - (self.N[:, None] * self.N[None, :]) * ((P / self.M[None, :]) @ P.T)
            try: d = np.zeros_like(f); d[1:] = -np.linalg.solve(H[1:, 1:], g[1:])
            except np.linalg.LinAlgError: f = self._sc(f); continue
            p0 = self._phi(f); t = 1.0
            while t > 1e-6 and not (self._phi(f + t * d) <= p0 + 1e-4 * t * float(g @ d)): t *= 0.5
            f = f + t * d if t > 1e-6 else self._sc(f)
        return f, res, maxit

    def at(self, T, lnW, N):
        """<E> (eV/atom), C_V (k_B/atom), F (eV/atom, absolute), S (k_B/atom) at the temperatures T."""
        T = np.atleast_1d(np.asarray(T, float)); out = [np.empty(len(T)) for _ in range(4)]
        base = self.lnM - self.logden; f0 = -logsumexp(base)                             # beta = 0
        for s in range(0, len(T), 64):
            b = 1.0 / (KB * T[s:s + 64]); lw = base[None, :] - b[:, None] * self.E[None, :]; lq = logsumexp(lw, axis=1); w = np.exp(lw - lq[:, None])
            m1 = w @ self.E; var = w @ self.E ** 2 - m1 ** 2
            lnZ = lnW - ((-lq) - f0) - b * self.Esh; Em = m1 + self.Esh
            out[0][s:s + 64] = Em / N; out[1][s:s + 64] = var * b ** 2 / N; out[2][s:s + 64] = -lnZ / b / N; out[3][s:s + 64] = (lnZ + b * Em) / N
        return dict(E=out[0], Cv=out[1], F=out[2], S=out[3])


# ------------------------------------------------------------------------------------------------------------------ runs
def load_run(d, interim):
    fin, ck = f"{d}/energies.npz", f"{d}/checkpoint.npz"; ys = pickle.load(open(MODEL, "rb"))["ys"]
    if os.path.exists(fin):
        z = np.load(fin); E = np.asarray(z["traj"], np.float64); beta = np.asarray(z["betas"], np.float64)
        drift = np.asarray(z["drift_eV"] if "drift_eV" in z.files else z["drift"] * ys, float); frac = 1.0; snaps = snap_chunks = None
        if os.path.exists(f"{d}/snapshots.npz"): s = np.load(f"{d}/snapshots.npz"); snaps, snap_chunks = s["sigma"], s["chunk"]
        summ = json.load(open(f"{d}/mcmc_summary.json")) if os.path.exists(f"{d}/mcmc_summary.json") else {}; exch = summ.get("exchange_acc")
    elif interim and os.path.exists(ck):
        z = np.load(ck); E = np.asarray(z["traj"], np.float64); beta = np.asarray(z["cfg_betas"], np.float64); drift = np.asarray(z["drift_log"], float) * ys
        frac = int(z["chunk"]) / (int(z["cfg_n_steps"]) // int(z["cfg_chunk"])); snaps, snap_chunks = z["snaps"], z["snap_chunks"]
        exch = int(z["n_ex_acc"]) / max(int(z["n_ex_try"]), 1)
    else: return None
    assert np.all(np.diff(beta) < 0), "ladder is not ascending in temperature"
    nm = os.path.basename(d.rstrip("/"))
    return dict(dir=d, name=nm, E=E, beta=beta, T=1.0 / (KB * beta), drift=drift, frac=frac, snaps=snaps, snap_chunks=snap_chunks, exch=exch,
                init=("L10" if nm.startswith("L10") else "random"))


def reduce_run(r, N, lo, hi):
    """burn-in, per-temperature statistics, weighted energy histograms.  Frees the raw series."""
    E = r["E"]; K, S = E.shape; t0s = np.array([detect_t0(E[k]) for k in range(K)]); raw = float(np.percentile(t0s, 90) / S)
    t0 = int(min(max(np.percentile(t0s, 90), 0.10 * S), 0.5 * S)); P = E[:, t0:]; Sp = P.shape[1]      # discard max(10 %, 90th percentile of the detected points)
    d = (hi - lo) / NBINS; M = np.zeros(NBINS); cnt = np.zeros(NBINS); sumE = np.zeros(NBINS)
    mean, var, g, se, seb = (np.zeros(K) for _ in range(5)); q4, hf = np.zeros((K, 4)), np.zeros((K, 2))
    for k in range(K):
        e = P[k]; mean[k] = e.mean(); var[k] = e.var(); g[k] = stat_ineff(e); se[k] = np.sqrt(var[k] * g[k] / Sp); seb[k] = block_se(e)
        q4[k] = [e[j * Sp // 4:(j + 1) * Sp // 4].mean() for j in range(4)]; hf[k] = [e[:Sp // 2].mean(), e[Sp // 2:].mean()]
        et = e * N; ib = np.floor((et - lo) / d).astype(np.int64); assert ib.min() >= 0 and ib.max() < NBINS, "energy outside the histogram range"
        c = np.bincount(ib, minlength=NBINS); cnt += c; M += c / g[k]; sumE += np.bincount(ib, weights=et, minlength=NBINS)
    # C4: neighbour-pair acceptance from the simultaneous energies
    b = r["beta"]; acc = np.array([np.mean(np.minimum(1.0, np.exp(np.minimum((b[i] - b[i + 1]) * (P[i] - P[i + 1]) * N, 0.0)))) for i in range(K - 1)])
    out = dict(name=r["name"], init=r["init"], frac=r["frac"], S=S, t0=t0, t0_raw_frac=raw, t0_median_frac=float(np.median(t0s) / S), t0_max_frac=float(t0s.max() / S), Sp=Sp, sweeps_used=Sp / 4.0, sweeps_total=S / 4.0,
               mean=mean, var=var, g=g, se=se, se_block=seb, q4=q4, half=hf, Neff=Sp / g, pair_acc=acc, drift_max=float(np.max(r["drift"])),
               exch=r["exch"], M=M, cnt=cnt, sumE=sumE, beta=b, T=r["T"])
    r["E"] = None
    return out


def order_parameter(r, N):
    """L1_0 order parameter of every snapshot: eta_a = mean_i s_i (-1)^(2 x_a / a), s = +1 (Pt) / -1 (Ni); a = x, y, z."""
    from ase.io import read
    if r["snaps"] is None or len(r["snaps"]) == 0: return None
    pos = read(f"{LAD}/init_random_N{N}.cif").get_positions(); ph = (-1.0) ** np.round(2 * pos / A_FCC).astype(int)          # (N, 3)
    l10 = read(f"{LAD}/init_L10_N{N}.cif").numbers; s0 = np.where(l10 == 78, 1.0, -1.0); eta0 = np.abs(s0 @ ph) / N
    assert eta0.max() > 0.999, f"order-parameter definition fails on the ideal L1_0 cell: {eta0}"
    sg = np.asarray(r["snaps"], np.float32) * 2 - 1; eta = np.einsum("trn,na->tra", sg, ph.astype(np.float32)) / N              # (n_snap, R, 3)
    emax = np.abs(eta).max(-1); var_id = np.abs(eta).argmax(-1) * 2 + (np.take_along_axis(eta, np.abs(eta).argmax(-1)[..., None], -1)[..., 0] > 0)
    t = len(emax) // 5; emax, var_id = emax[t:], var_id[t:]                                                                  # drop the first fifth
    K = emax.shape[1]; sw = np.zeros(K, int); occ = np.zeros((K, 6))
    for k in range(K):
        st = 0; n = 0                                                                 # hysteresis: ordered above 0.6, disordered below 0.3
        for v in emax[:, k]:
            ns = 1 if v > 0.6 else (-1 if v < 0.3 else st)
            if st != 0 and ns != st: n += 1
            st = ns
        sw[k] = n; o = emax[:, k] > 0.6
        if o.sum(): occ[k] = np.bincount(var_id[o, k], minlength=6) / o.sum()
    return dict(eta_mean=emax.mean(0), frac_ordered=(emax > 0.6).mean(0), switches=sw, variant_occ=occ, n_snap=int(len(emax)))


def combine(red, idx, lo, hi, N):
    M = sum(red[i]["M"] for i in idx); cnt = sum(red[i]["cnt"] for i in idx); sE = sum(red[i]["sumE"] for i in idx)
    Eb = np.where(cnt > 0, sE / np.maximum(cnt, 1), 0.0); Neff = sum(red[i]["Neff"] for i in idx)
    Em = np.average([red[i]["mean"] for i in idx], axis=0) * N
    return HistMBAR(M, Eb, red[idx[0]]["beta"], Neff, Emean=Em)


def target_grid():
    return np.unique(np.round(np.concatenate([np.arange(300, 700, 5.0), np.arange(700, 1000, 0.1), np.arange(1000, 3000.01, 5.0),
                                              [3500, 4000, 5000, 7000, 1e4, 2e4, 5e4, 1e5, 1e6, 1e8]]), 4))


def analyse(N, dirs, out, interim):
    os.makedirs(out, exist_ok=True); nPt = N // 2; lnW = math.lgamma(N + 1) - 2 * math.lgamma(nPt + 1); lo, hi = N * -5.9965, N * -5.9000
    md5 = hashlib.md5(open(MODEL, "rb").read()).hexdigest(); red, ops = [], []
    for d_ in dirs:
        r = load_run(d_, interim)
        if r is None: log(f"{d_}: no data yet, skipped"); continue
        t = time.time(); ops.append(order_parameter(r, N)); r["snaps"] = None; red.append(reduce_run(r, N, lo, hi)); x = red[-1]; del r
        log(f"{x['name']}: {x['sweeps_total']:.0f} sweeps ({100*x['frac']:.0f}% of the run), burn-in {x['t0']/4:.0f} sweeps (detected {100*x['t0_raw_frac']:.1f}%), "
            f"g median {np.median(x['g']):.1f} max {x['g'].max():.1f} samples, N_eff min {x['Neff'].min():.0f}, pair acceptance min {x['pair_acc'].min():.3f}, "
            f"drift {x['drift_max']:.1e}, {time.time()-t:.0f}s")
    assert red, "no runs with data"
    T = red[0]["T"]; K = len(T); n = len(red); assert all(np.allclose(x["beta"], red[0]["beta"]) for x in red), "different ladders"
    Tq = target_grid(); allr = list(range(n)); t = time.time()
    pooled = combine(red, allr, lo, hi, N); cur = pooled.at(Tq, lnW, N); log(f"pooled MBAR: {len(pooled.M)} occupied bins, Newton {pooled.n_newton} steps, residual {pooled.resid:.1e}, {time.time()-t:.0f}s")
    per = [combine(red, [i], lo, hi, N).at(Tq, lnW, N) for i in allr]
    win = (Tq >= 700) & (Tq < 1000)
    def peak(c): k = int(np.argmax(c["Cv"][win])); return float(Tq[win][k]), float(c["Cv"][win][k])
    res = {q: cur[q] for q in ("E", "Cv", "F", "S")}; pk = peak(cur); jk = {}
    if n >= 3:
        loo = [combine(red, [j for j in allr if j != i], lo, hi, N).at(Tq, lnW, N) for i in allr]
        for q in ("E", "Cv", "F", "S"):
            a = np.array([c[q] for c in loo]); jk[q] = np.sqrt((n - 1) / n * ((a - a.mean(0)) ** 2).sum(0))
        pks = np.array([peak(c) for c in loo]); jk["peak_T"], jk["peak_Cv"] = (float(np.sqrt((n - 1) / n * ((pks[:, j] - pks[:, j].mean()) ** 2).sum())) for j in (0, 1))
    # ---- checks
    chk = dict(C1=dict(md5=md5, ok=md5 == MD5_REF))
    m_all = np.array([x["mean"] for x in red]); se_all = np.array([x["se"] for x in red])
    chk["C2"] = dict(lowT=float(m_all[:, 0].mean()), highT=float(m_all[:, -1].mean()), d_low=float(m_all[:, 0].mean() - E_L10), d_high=float(m_all[:, -1].mean() - E_RAND))
    chk["C3"] = dict(drift_max=float(max(x["drift_max"] for x in red)), ok=max(x["drift_max"] for x in red) < 1e-6)
    pa = np.array([x["pair_acc"] for x in red]); chk["C4"] = dict(pair_acc_min=float(pa.min()), at_T=float(T[int(np.argmin(pa.min(0)))]), pair_acc_mean=pa.mean(0).tolist(), ok=bool(pa.min() > 0.1))
    chk["C5"] = dict(burn_frac=[x["t0"] / x["S"] for x in red], detected_frac=[x["t0_raw_frac"] for x in red], g_max=[float(x["g"].max()) for x in red],
                     g_at_T=[float(T[int(np.argmax(x["g"]))]) for x in red], Neff_min=[float(x["Neff"].min()) for x in red],
                     ok=bool(max(x["t0_raw_frac"] for x in red) < 0.5 and min(x["Neff"].min() for x in red) > 100))
    z6 = np.array([(x["q4"] - x["mean"][:, None]) / np.maximum(x["se"][:, None] * np.sqrt(3.0), 1e-15) for x in red])       # block mean - full mean: var = 3 se^2
    z7 = np.array([(x["half"][:, 0] - x["half"][:, 1]) / np.maximum(x["se"] * 2.0, 1e-15) for x in red])                    # half difference: var = 4 se^2
    live = se_all > 1e-12
    chk["C6"] = dict(rms_z=float(np.sqrt(np.mean(z6[live[:, :, None].repeat(4, 2)] ** 2))), max_abs_z=float(np.abs(z6[live[:, :, None].repeat(4, 2)]).max()))
    chk["C7"] = dict(rms_z=float(np.sqrt(np.mean(z7[live] ** 2))), max_abs_z=float(np.abs(z7[live]).max()))
    chk["C6"]["ok"] = chk["C6"]["max_abs_z"] < 5 and chk["C6"]["rms_z"] < 1.5; chk["C7"]["ok"] = chk["C7"]["max_abs_z"] < 5 and chk["C7"]["rms_z"] < 1.5
    io = [i for i in allr if red[i]["init"] == "L10"]; ir = [i for i in allr if red[i]["init"] == "random"]
    if io and ir:
        mo, mr = m_all[io].mean(0), m_all[ir].mean(0); so = np.sqrt((se_all[io] ** 2).sum(0)) / len(io); sr = np.sqrt((se_all[ir] ** 2).sum(0)) / len(ir)
        z8 = (mo - mr) / np.maximum(np.sqrt(so ** 2 + sr ** 2), 1e-15); lv = (so + sr) > 1e-12
        chk["C8"] = dict(z=z8.tolist(), dE_meV=((mo - mr) * 1e3).tolist(), rms_z=float(np.sqrt(np.mean(z8[lv] ** 2))), max_abs_z=float(np.abs(z8[lv]).max()),
                         max_abs_dE_meV=float(np.abs(mo - mr).max() * 1e3)); chk["C8"]["ok"] = chk["C8"]["max_abs_z"] < 4.5
    if n >= 3:
        wgt = 1.0 / np.maximum(se_all, 1e-15) ** 2; mbar = (wgt * m_all).sum(0) / wgt.sum(0); chi2 = (wgt * (m_all - mbar) ** 2).sum(0) / (n - 1); lv = live.all(0)
        chk["C9"] = dict(chi2_per_dof=chi2.tolist(), mean=float(chi2[lv].mean()), max=float(chi2[lv].max()), at_T=float(T[lv][int(np.argmax(chi2[lv]))]), ok=bool(chi2[lv].mean() < 1.6 and chi2[lv].max() < 6))
    if any(o is not None for o in ops):
        oo = [o for o in ops if o is not None]; sw = np.sum([o["switches"] for o in oo], 0); em = np.mean([o["eta_mean"] for o in oo], 0)
        occ = np.mean([o["variant_occ"] for o in oo], 0); kl = int(np.argmin(T))
        chk["C10"] = dict(eta_lowT=float(em[kl]), eta_highT=float(em[-1]), switches_total_per_T=sw.tolist(), eta_mean=em.tolist(),
                          frac_ordered=np.mean([o["frac_ordered"] for o in oo], 0).tolist(), variant_occ_lowT=occ[kl].tolist(),
                          variant_occ_per_run_lowT=[o["variant_occ"][kl].tolist() for o in oo], n_snap=[o["n_snap"] for o in oo], ok=bool(em[kl] > 0.98))
    # ---- exact comparison
    ex = None; exf = f"{RESULTS}/ptni_exact/exact_N{N}"
    if os.path.exists(exf + ".json") and jk:
        X = json.load(open(exf + ".json")); Tx = np.array(X["T"]); ex = dict(complete=bool(X.get("complete")), rows=[], zmax={}, zrms={})
        ix = [(int(np.argmin(np.abs(Tq - t_))), j) for j, t_ in enumerate(Tx) if 300 <= t_ <= 3000 and np.abs(Tq - t_).min() < 1e-6]
        for q in ("E", "Cv", "F", "S"):
            zz = np.array([(cur[q][i] - X[q][j]) / max(jk[q][i], 1e-15) for i, j in ix]); ex["zmax"][q] = float(np.abs(zz).max()); ex["zrms"][q] = float(np.sqrt(np.mean(zz ** 2)))
            ex.setdefault("absmax", {})[q] = float(max(abs(cur[q][i] - X[q][j]) for i, j in ix))
        for Tw in (300, 400, 500, 600, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 2000, 3000):
            i, j = min(ix, key=lambda p: abs(Tx[p[1]] - Tw)); ex["rows"].append(dict(T=float(Tx[j]), **{q: dict(mbar=float(cur[q][i]), exact=float(X[q][j]), se=float(jk[q][i])) for q in ("E", "Cv", "F", "S")}))
        D = np.load(exf + ".npz"); ec = 0.5 * (D["e_bins"][1:] + D["e_bins"][:-1]); dos = D["dos"]; kk = dos > 0; ec, ldos = ec[kk] * N, np.log(dos[kk])
        zm, zv = [], []
        for k in range(K):                                                            # every ladder temperature of every run vs the exact canonical mean / variance
            lw = ldos - red[0]["beta"][k] * (ec - ec.min()); w = np.exp(lw - logsumexp(lw)); m1 = w @ ec; v1 = w @ ec ** 2 - m1 ** 2
            for x in red:
                if x["se"][k] > 1e-12: zm.append((x["mean"][k] * N - m1) / (x["se"][k] * N)); zv.append((x["var"][k] * N * N - v1) / max(v1, 1e-30))
        ex["ladder_mean_z_rms"] = float(np.sqrt(np.mean(np.square(zm)))); ex["ladder_mean_z_max"] = float(np.max(np.abs(zm))); ex["ladder_n"] = len(zm)
        ex["ladder_var_rel_rms"] = float(np.sqrt(np.mean(np.square(zv)))); ex["ladder_var_rel_max"] = float(np.max(np.abs(zv)))
    rep = dict(N=N, interim=bool(interim), n_runs=n, runs=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items() if k not in ("M", "cnt", "sumE")} for x in red],
               ladder_T=T.tolist(), checks=chk, T=Tq.tolist(), **{q: res[q].tolist() for q in res}, **{q + "_se": jk[q].tolist() for q in ("E", "Cv", "F", "S") if q in jk},
               peak_T=pk[0], peak_Cv=pk[1], peak_T_se=jk.get("peak_T"), peak_Cv_se=jk.get("peak_Cv"), peak_per_run=[peak(c) for c in per],
               mbar=dict(bins=int(len(pooled.M)), newton_steps=int(pooled.n_newton), residual=float(pooled.resid), Neff_total=float(pooled.N.sum())), exact=ex)
    json.dump(rep, open(f"{out}/analysis_N{N}.json", "w"), default=float); write_report(rep, f"{out}/report_N{N}.md"); return rep


def write_report(p, path):
    N, c, T, Tq = p["N"], p["checks"], np.array(p["ladder_T"]), np.array(p["T"]); has = "E_se" in p; L = []
    def se(q, i): return f"{p[q + '_se'][i]:.1e}" if has else "-"
    ok = lambda k: "pass" if c[k].get("ok") else "**FAIL**"
    L += [f"# PtNi 50/50, {N} atoms: PT + MBAR reference" + (" (interim)" if p["interim"] else ""), "",
          f"{p['n_runs']} independent runs, {len(T)} temperatures ({T.min():.0f} K ... {T.max():.0e} K), 4 samples per sweep. Errors: delete-one-run jackknife.", "",
          "## 1. Runs", "", "| run | start | progress | total sweeps | burn-in discarded (sweeps) | equilibration detected at | max g (samples) | min N_eff | min neighbour exchange | carried-energy drift (eV/atom) |", "|" + "---|" * 10]
    for x in p["runs"]:
        L.append(f"| {x['name']} | {x['init']} | {100*x['frac']:.0f}% | {x['sweeps_total']:.0f} | {x['t0']/4:.0f} | {100*x['t0_raw_frac']:.1f}% | {max(x['g']):.0f} | {min(x['Neff']):.0f} | {min(x['pair_acc']):.3f} | {x['drift_max']:.1e} |")
    L += ["", "## 2. Results", "", f"C_V peak temperature **{p['peak_T']:.1f} K**" + (f" +- {p['peak_T_se']:.1f}" if p.get("peak_T_se") is not None else "") + f", C_V maximum **{p['peak_Cv']:.3f}**"
          + (f" +- {p['peak_Cv_se']:.3f}" if p.get("peak_Cv_se") is not None else "") + " k_B/atom. Per-run peaks: " + ", ".join(f"{a:.1f} K / {b:.3f}" for a, b in p["peak_per_run"]), "",
          "| T (K) | <E> (eV/atom) | +- | C_V (k_B/atom) | +- | F (eV/atom) | +- | S (k_B/atom) | +- |", "|" + "---|" * 9]
    for Tw in (300, 400, 500, 600, 700, 750, 800, 820, 840, 850, 860, 880, 900, 950, 1000, 1100, 1200, 1400, 2000, 3000, 1e8):
        i = int(np.argmin(np.abs(Tq - Tw))); L.append(f"| {Tq[i]:.0f} | {p['E'][i]:.6f} | {se('E', i)} | {p['Cv'][i]:.4f} | {se('Cv', i)} | {p['F'][i]:.6f} | {se('F', i)} | {p['S'][i]:.5f} | {se('S', i)} |")
    L += ["", "F is absolute: Z = C(N, N/2) at infinite temperature, connected to beta = 0 by the same reweighting.", "", "## 3. Evidence that these values are right", "",
          "| check | measured | verdict |", "|---|---|---|",
          f"| C1 model file md5 | {c['C1']['md5']} | {ok('C1')} |",
          f"| C2 end-point energies | lowest T {c['C2']['lowT']:.6f} (L1_0 {E_L10}, difference {c['C2']['d_low']*1e3:+.3f} meV), 1e8 K {c['C2']['highT']:.6f} (random {E_RAND}, difference {c['C2']['d_high']*1e3:+.3f} meV) | for reference |",
          f"| C3 float32 drift of the carried energy | max {c['C3']['drift_max']:.1e} eV/atom | {ok('C3')} |",
          f"| C4 neighbour exchange acceptance | min over all pairs {c['C4']['pair_acc_min']:.3f} ({c['C4']['at_T']:.0f} K) | {ok('C4')} |",
          f"| C5 equilibration / autocorrelation | equilibration detected (90th percentile over temperatures) at most at {100*max(c['C5']['detected_frac']):.1f}% of the run, first {100*max(c['C5']['burn_frac']):.0f}% discarded, max g {max(c['C5']['g_max']):.0f} samples ({c['C5']['g_at_T'][int(np.argmax(c['C5']['g_max']))]:.0f} K), min N_eff {min(c['C5']['Neff_min']):.0f} | {ok('C5')} |",
          f"| C6 four time blocks | block mean - full mean in units of its own error: rms {c['C6']['rms_z']:.2f}, max {c['C6']['max_abs_z']:.2f} (expected rms 1) | {ok('C6')} |",
          f"| C7 first half vs second half | rms {c['C7']['rms_z']:.2f}, max {c['C7']['max_abs_z']:.2f} | {ok('C7')} |"]
    if "C8" in c: L.append(f"| C8 ordered start vs random start | <E> difference over all temperatures: rms {c['C8']['rms_z']:.2f} sigma, max {c['C8']['max_abs_z']:.2f} sigma, max {c['C8']['max_abs_dE_meV']:.4f} meV/atom | {ok('C8')} |")
    if "C9" in c: L.append(f"| C9 agreement of independent runs | chi^2 per dof per temperature: mean {c['C9']['mean']:.2f}, max {c['C9']['max']:.2f} ({c['C9']['at_T']:.0f} K) (expected 1) | {ok('C9')} |")
    if "C10" in c:
        sw = np.array(c["C10"]["switches_total_per_T"]); k = int(np.argmax(sw)); occ = c["C10"]["variant_occ_lowT"]
        L.append(f"| C10 order parameter (snapshots) | lowest T eta = {c['C10']['eta_lowT']:.4f}, highest T eta = {c['C10']['eta_highT']:.3f}; order <-> disorder switches in total {int(sw.sum())} (most {int(sw[k])}, at {T[k]:.0f} K); "
                 f"L1_0 variants seen at the lowest T (x-,x+,y-,y+,z-,z+) = " + ", ".join(f"{o:.2f}" for o in occ) + f" | {ok('C10')} |")
    L.append(f"| MBAR convergence | {p['mbar']['bins']} energy bins, {p['mbar']['newton_steps']} Newton steps, residual {p['mbar']['residual']:.1e}, total effective samples {p['mbar']['Neff_total']:.2e} | {'pass' if p['mbar']['residual'] < 1e-8 else '**FAIL**'} |")
    if "C10" in c:
        L += ["", "Order parameter and switches per temperature slot (transition window):", "", "| T (K) | <eta> | fraction ordered | switches (all runs) | exchange acceptance (with the slot above) |", "|---|---|---|---|---|"]
        for k in range(len(T)):
            if 700 <= T[k] <= 1000: L.append(f"| {T[k]:.1f} | {c['C10']['eta_mean'][k]:.3f} | {c['C10']['frac_ordered'][k]:.3f} | {c['C10']['switches_total_per_T'][k]} | {c['C4']['pair_acc_mean'][k] if k < len(T)-1 else float('nan'):.3f} |")
    if p.get("exact"):
        e = p["exact"]; L += ["", "## 4. Comparison with the exact enumeration", "", f"Enumeration completeness: {'complete (sum of weights = C(N, N/2))' if e['complete'] else '**incomplete**'}.", "",
                              "| T (K) | <E> MBAR | exact | diff/sigma | C_V MBAR | exact | diff/sigma | F MBAR | exact | diff/sigma |", "|" + "---|" * 10]
        for r_ in e["rows"]:
            z = lambda q: (r_[q]["mbar"] - r_[q]["exact"]) / max(r_[q]["se"], 1e-15)
            L.append(f"| {r_['T']:.0f} | {r_['E']['mbar']:.6f} | {r_['E']['exact']:.6f} | {z('E'):+.2f} | {r_['Cv']['mbar']:.5f} | {r_['Cv']['exact']:.5f} | {z('Cv'):+.2f} | {r_['F']['mbar']:.6f} | {r_['F']['exact']:.6f} | {z('F'):+.2f} |")
        L += ["", f"Whole 300-3000 K grid: max |diff|/sigma  <E> {e['zmax']['E']:.2f}, C_V {e['zmax']['Cv']:.2f}, F {e['zmax']['F']:.2f}, S {e['zmax']['S']:.2f}; "
              f"max absolute difference <E> {e['absmax']['E']:.1e} eV/atom, C_V {e['absmax']['Cv']:.1e} k_B/atom, F {e['absmax']['F']:.1e} eV/atom.",
              f"Mean energy of each of the {e['ladder_n']} (ladder temperature x run) slots against the exact distribution: diff/sigma rms {e['ladder_mean_z_rms']:.2f}, max {e['ladder_mean_z_max']:.2f} (rms 1 if the errors are right); "
              f"relative variance difference rms {e['ladder_var_rel_rms']:.1e}, max {e['ladder_var_rel_max']:.1e}."]
    L += ["", "## Limitations", "", "- The engine does not record replica round trips in temperature. Instead the verdict rests on the order <-> disorder switches per temperature slot (C10), independence of the starting condition (C8) and agreement of independent runs (C9).",
          "- With few runs the jackknife error has a large uncertainty of its own (about 1/sqrt(2(n-1)) relative for n runs)."]
    if p["interim"]: L.append("- Interim: the values are replaced by those of the finished runs.")
    open(path, "w").write("\n".join(L) + "\n"); log(f"wrote {path}")


# ------------------------------------------------------------------------------------------------------------------ self test
def selftest():
    """two-level toy (M units, gap eps): exact Z = (1 + e^{-b eps})^M.  i.i.d. samples at 12 temperatures + 1e8 K, 8 'runs'."""
    rng = np.random.default_rng(7); Mu, eps = 200, 0.05; T = np.concatenate([np.geomspace(300, 3000, 12), [1e8]]); b = 1 / (KB * T); lnW = Mu * math.log(2.0)
    lo, hi = -1.0, Mu * eps + 1.0; d = (hi - lo) / NBINS; red = []
    for r in range(8):
        M = np.zeros(NBINS); cnt = np.zeros(NBINS); sE = np.zeros(NBINS); mean = np.zeros(len(T))
        for k in range(len(T)):
            e = rng.binomial(Mu, 1 / (1 + np.exp(b[k] * eps)), 200000) * eps; ib = np.floor((e - lo) / d).astype(int); c = np.bincount(ib, minlength=NBINS)
            cnt += c; M += c; sE += np.bincount(ib, weights=e, minlength=NBINS); mean[k] = e.mean()
        red.append(dict(M=M, cnt=cnt, sumE=sE, Neff=np.full(len(T), 200000.0), mean=mean, beta=b))
    Tt = np.array([300, 450, 700, 1000, 1500, 2500, 3000.0]); bt = 1 / (KB * Tt); p = 1 / (1 + np.exp(bt * eps))
    exact = dict(E=Mu * eps * p, Cv=Mu * (bt * eps) ** 2 * p * (1 - p), F=-Mu * np.log1p(np.exp(-bt * eps)) / bt)
    full = combine(red, list(range(8)), lo, hi, 1).at(Tt, lnW, 1); loo = [combine(red, [j for j in range(8) if j != i], lo, hi, 1).at(Tt, lnW, 1) for i in range(8)]; worst = 0.0
    for q in ("E", "Cv", "F"):
        a = np.array([c[q] for c in loo]); s = np.sqrt(7 / 8 * ((a - a.mean(0)) ** 2).sum(0)); z = (full[q] - exact[q]) / s; worst = max(worst, float(np.abs(z).max()))
        log(f"selftest {q}: max |z| {np.abs(z).max():.2f}, max relative error {np.abs(full[q] / exact[q] - 1).max():.1e}")
    rho = 0.99; x = np.zeros(2_000_000); nz = rng.normal(size=len(x))
    for i in range(1, len(x)): x[i] = rho * x[i - 1] + nz[i]
    g = stat_ineff(x); gt = (1 + rho) / (1 - rho); log(f"selftest statistical inefficiency: {g:.1f} (exact {gt:.1f})")
    assert worst < 5.0 and abs(g / gt - 1) < 0.15, "SELFTEST FAILED"; log("SELFTEST_PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int); ap.add_argument("--runs", nargs="*", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--interim", action="store_true"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest(); sys.exit(0)
    dirs = a.runs or sorted(d for d in glob.glob(f"{BASE}/N{a.N}/*") if "smoke" not in d)
    log(f"N={a.N}: {len(dirs)} run directories")
    analyse(a.N, dirs, a.out or f"{RESULTS}/ptni_exact/N{a.N}" + ("_interim" if a.interim else ""), a.interim)
