"""Gates for the template-only (implicit) neural-CE Monte Carlo path (neuralce.nce.gpu_layer0_implicit +
neuralce.nce.local_update_implicit), i.e. the production ``--framework nce`` kernel.

C1  values (float64 and float32): implicit == tiled tables on PtNi 32/256 (depth 1, 'h0' cache): energy, local dE
    on the same swaps (indices mapped through the canonical permutation), dE vs full recompute, cache after a chain
    of committed moves vs full recompute.  Tolerances: |E_tiled - E_implicit| < 1e-9 eV/atom; standardised caches /
    local dE within 1e-9 (float64) or 3e-5 (float32); dE vs E_after - E_before and committed cache vs full within
    10x that.  (values_case also handles spin models; only the PtNi model is shipped.)
C2  GPU speed / resident memory vs N (PtNi, 32 replicas, the production step) up to 10M atoms; the tiled table is
    run alongside for M <= TILED_MAX.
C3  central entry ``python -m neuralce.mc.run --framework nce --pt`` on PtNi 256 (and ``--supercell 2 2 2``):
    E_final[0] == tiled energy of the written final cif (< 1e-5 eV/atom).

Inputs:  $MODELS/nce/nipt_v3_nomax_sel/model.pkl (the PtNi cells are built with ase.build.bulk).
Outputs: $RUNS/gates/nipt_gate_implicit<GATE_TAG>[_sc2]/ (C3 MC output), $RUNS/gates/ptni_gate_M4.cif (C2).
Run:     python gates/gate_nce_implicit.py        (GPU for C2; C1/C3 also run with JAX_PLATFORMS=cpu)
Env:     SECTION=values|speed|central (comma list, default all), MS (C2 sizes, default 8,16,32,64,100,136),
         TILED_MAX (largest M also run tiled in C2, default 8), GATE_TAG (suffix of the C3 output directory).
"""
import os, sys, time, json, subprocess, resource, numpy as np
from neuralce.paths import REPO, MODELS, RUNS
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk
from neuralce.nce.mc_model import load_model, setup
def log(*a): print(*a, flush=True)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
def gpu_mem():
    st = jax.devices()[0].memory_stats() or {}; return st.get('bytes_in_use', 0) / 1e9, st.get('peak_bytes_in_use', 0) / 1e9
SEC = os.environ.get("SECTION", "values,speed,central").split(",")
rng = np.random.RandomState(0)
def ptni(M):
    a = bulk("Ni", "fcc", a=3.924, cubic=True).repeat((M, M, M)); Z = np.full(len(a), 78, np.int64); Z[rng.choice(len(a), len(a) // 2, replace=False)] = 28; a.numbers = Z; return a
NIPT = f"{MODELS}/nce/nipt_v3_nomax_sel/model.pkl"
GATE_DIR = f"{RUNS}/gates"; os.makedirs(GATE_DIR, exist_ok=True)

def values_case(tag, mdl, atoms, spins=None, n_moves=40):
    for dtype, tol in (("float64", 1e-9), ("float32", 3e-5)):
        dt = None if dtype == "float64" else dtype
        mt = setup(mdl, atoms, dtype=dt, table="tiled"); mi = setup(mdl, atoms, dtype=dt, table="implicit")
        perm, inv = mi['perm'], mi['inv_perm']
        st = mt['sig0']; si = mi['sig0']; assert np.array_equal(np.asarray(si), np.asarray(st)[perm]), "sig mismatch through perm"
        sp_t = None if spins is None else jnp.asarray(spins, jnp.float64 if dt is None else jnp.float32); sp_i = None if spins is None else sp_t[perm]
        Et = mt['energy'](st, None if sp_t is None else np.asarray(sp_t)); Ei = mi['energy'](si, None if sp_i is None else np.asarray(sp_i))
        ct = mt['full_cache'](st, sp_t); ci = mi['full_cache'](si, sp_i)
        e_t = float(mt['F']['energy_of'](ct, mt['P'])); e_i = float(mi['F']['energy_of'](ci, mi['P']))
        log(f"  [C1:{tag}:{dtype}] E tiled {Et:.9f} implicit {Ei:.9f} (diff {abs(Et-Ei):.2e} eV/atom); {dtype} caches: {abs(e_t-e_i):.2e} std | cache mode '{mi['cache_mode']}' balls {mi['stat']['ball_w']}")
        assert abs(Et - Ei) < 1e-9 and abs(e_t - e_i) < tol
        N = len(atoms); act = np.asarray(mt['active']); vals = None if spins is None else mi['spin_info']['values']
        errs = {"swap": [], "spin": []}; errs_full = []
        for t in range(n_moves):
            if spins is not None and t % 2 == 0:
                i = int(rng.choice(act)); cur = float(sp_t[i]); new = float(rng.choice([v for v in vals if v != cur]))
                ft = jnp.array([[i, int(st[i])], [i, int(st[i])]]); sfl = jnp.array([new, new], sp_t.dtype); kind = "spin"
            else:
                i, j = rng.choice(act, 2, replace=False); i, j = int(i), int(j)
                ft = jnp.array([[i, int(st[j])], [j, int(st[i])]]); sfl = None if spins is None else jnp.array([sp_t[j], sp_t[i]], sp_t.dtype); kind = "swap"
            fi = ft.at[:, 0].set(jnp.asarray(inv)[ft[:, 0]])                                   # canonical indices
            dEt, pt_, _ = mt['local_delta'](ct, st, ft, sp_t, sfl); dEi, pi_, _ = mi['local_delta'](ci, si, fi, sp_i, sfl)
            errs[kind].append(abs(float(dEt) - float(dEi)))
            # full recompute check on the implicit side
            si1 = si.at[fi[:, 0]].set(fi[:, 1].astype(si.dtype)); sp_i1 = None if sp_i is None else sp_i.at[fi[:, 0]].set(sfl)
            E1 = float(mi['F']['energy_of'](mi['full_cache'](si1, sp_i1), mi['P'])); errs_full.append(abs(float(dEi) - (E1 - e_i)))
            # commit on both (chain)
            ct = mt['commit'](ct, pt_, jnp.asarray(True)); ci = mi['commit'](ci, pi_, jnp.asarray(True))
            st = st.at[ft[:, 0]].set(ft[:, 1].astype(st.dtype)); si = si1; e_i = float(mi['F']['energy_of'](ci, mi['P']))
            if spins is not None: sp_t = sp_t.at[ft[:, 0]].set(sfl); sp_i = sp_i1
        Efull_i = float(mi['F']['energy_of'](mi['full_cache'](si, sp_i), mi['P'])); Et_chain = float(mt['F']['energy_of'](ct, mt['P']))
        log(f"  [C1:{tag}:{dtype}] |dE_implicit - dE_tiled|: swaps max {max(errs['swap']):.2e}" + (f", spin moves max {max(errs['spin']):.2e}" if errs['spin'] else "")
            + f" | |dE - (E_after - E_before)| max {max(errs_full):.2e} | cache after {n_moves} committed moves: implicit vs full {abs(e_i-Efull_i):.2e}, vs tiled chain {abs(e_i-Et_chain):.2e} (std units)")
        assert max(errs['swap']) < tol and (not errs['spin'] or max(errs['spin']) < tol) and max(errs_full) < tol * 10 and abs(e_i - Efull_i) < tol * 10

if "values" in SEC:
    mdl = load_model(NIPT)
    values_case("PtNi32", mdl, ptni(2)); values_case("PtNi256", mdl, ptni(4))
    log("[C1] VALUES PASS")

if "speed" in SEC:
    from neuralce.nce.mcmc import _prepare, _chunk_runner
    from ase.io import write
    R, CH = 32, 2000; TM = int(os.environ.get("TILED_MAX", "8"))
    beta = jnp.full((R,), 1 / (8.617e-5 * 800), jnp.float32)
    cif4 = f"{GATE_DIR}/ptni_gate_M4.cif"; write(cif4, ptni(4))
    def bench(M, table):
        # the 256-atom cif tiled in memory (--supercell path, species re-shuffled): ase's cif reader is minutes at 16k atoms
        assert M % 4 == 0; k = M // 4; a = None
        t0 = time.time(); os.environ["CE_CNN_TABLE"] = table; mc = _prepare(NIPT, cif4, supercell=(k, k, k), dtype="float32"); tb = time.time() - t0
        a = mc['atoms']
        run = _chunk_runner(mc, CH); sig = jnp.stack([mc['sig0']] * R)
        t0 = time.time(); cache = jax.lax.map(mc['full_cache'], sig); E = jax.vmap(lambda c: mc['F']['energy_of'](c, mc['P']))(cache); E.block_until_ready(); tE = time.time() - t0
        key = jax.random.PRNGKey(0); sig, _, cache, E, nacc, _ = run(sig, None, cache, E, beta, jax.random.split(key, R)); E.block_until_ready()
        t0 = time.time()
        for k in range(3): sig, _, cache, E, nacc, _ = run(sig, None, cache, E, beta, jax.random.split(jax.random.PRNGKey(k + 1), R)); E.block_until_ready()
        dt = (time.time() - t0) / 3; used, peak = gpu_mem()
        cache = None                                                     # release the running caches before the check rebuild
        Ef = jax.vmap(lambda c: mc['F']['energy_of'](c, mc['P']))(jax.lax.map(mc['full_cache'], sig)); drift = float(jnp.abs(Ef - E).max())
        log(f"  [C2:{table:8s}] N={len(a):9d}: setup {tb:6.1f}s  full cache x{R} {tE:6.1f}s | {dt/CH*1e3:6.3f} ms/step ({R*CH/dt:8.0f} steps/s, R={R}) acc {float(jnp.mean(nacc))/CH:.3f} drift {drift:.1e} | GPU in use {used:.2f} GB (peak {peak:.2f}) host RSS {rss():.1f} GB")
        del mc, run, sig, cache, E; import gc; gc.collect()
    for M in [int(x) for x in os.environ.get("MS", "8,16,32,64,100,136").split(",")]:
        if M <= TM: bench(M, "tiled")
        bench(M, "implicit")

if "central" in SEC:
    from ase.io import write, read
    out = f"{GATE_DIR}/nipt_gate_implicit{os.environ.get('GATE_TAG', '')}"; os.makedirs(out, exist_ok=True)
    cif = f"{out}/init.cif"; write(cif, ptni(4)); os.environ.pop("CE_CNN_TABLE", None)
    cmd = [sys.executable, "-m", "neuralce.mc.run", "--framework", "nce", "--ckpt", NIPT, "--init_cif", cif, "--pt", "--T_min", "300", "--T_max", "1400",
           "--n_replicas", "8", "--n_steps", "4000", "--chunk", "500", "--exchange_every", "100", "--log_every", "2000", "--out_dir", out, "--energy_mode", "local"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, NCE_REPO=REPO)); assert r.returncode == 0, r.stderr[-3000:]
    s = json.load(open(f"{out}/mcmc_summary.json")); log("\n".join(l for l in r.stdout.splitlines() if "[ce_cnn" in l)[-1200:])
    fin = read(f"{out}/final_T300_r0.cif"); mdl = load_model(NIPT); mt = setup(mdl, fin, dtype=None, table="tiled"); e_cif = mt['energy'](mt['sig0'])
    log(f"  [C3] central PT (implicit): steps/s {s['steps_per_s']:.0f} acc {s['acc_mean']:.3f} exch {s['exchange_acc']} drift {s['drift_max']:.1e} | E_final[0] {s['E_final'][0]:.6f} vs tiled E(final cif) {e_cif:.6f} diff {abs(s['E_final'][0]-e_cif):.2e}")
    assert abs(s['E_final'][0] - e_cif) < 1e-5
    cmd2 = cmd[:cmd.index("--out_dir") + 1] + [out + "_sc2"] + cmd[cmd.index("--out_dir") + 2:] + ["--supercell", "2", "2", "2"]
    r = subprocess.run(cmd2, capture_output=True, text=True, env=dict(os.environ, NCE_REPO=REPO)); assert r.returncode == 0, r.stderr[-3000:]
    s2 = json.load(open(f"{out}_sc2/mcmc_summary.json")); log(f"  [C3] --supercell 2 2 2: N={s2['n_atoms']} steps/s {s2['steps_per_s']:.0f} acc {s2['acc_mean']:.3f} exch {s2['exchange_acc']} drift {s2['drift_max']:.1e}")
    log("[C3] CENTRAL PASS")
log("[SUMMARY] NCE IMPLICIT GATES DONE: " + ",".join(SEC))
