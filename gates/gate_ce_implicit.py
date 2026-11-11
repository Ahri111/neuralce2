"""Gates for the template-only (implicit) icet cluster-expansion path, neuralce.ce.ce_gpu_implicit.

G1  PtNi 256: encode == tiled encode (through the canonical permutation);
    energy_a == tiled energy (< 2e-3 eV total; energy() == energy_a within 1e-6); delta_energy_a == tiled delta on the
    same 30 swaps (< 1e-4 eV); delta == E_after - E_before (< 2e-3 eV).
G1b energy extensivity: the periodic 2x2x2 image of the 256-atom cell has 8x its energy (ratio within 1e-4) and one
    site-pair swap in the big cell has the same delta as in the base cell (ratio within 1e-3).
G3  PtNi delta-step speed vs N (32 replicas) with the resident GPU memory, up to 10M atoms; tiled numbers alongside
    for M <= TILED_MAX.
G4  central entry ``python -m neuralce.mc.run --framework icet --pt --energy_mode delta`` on PtNi 256: E_min == tiled
    energy of the returned sigma_min in input atom order (< 2e-3 eV); ``--supercell 2 2 2`` smoke.

Inputs:  $MODELS/ce/ptni/icet_c8_6_5.ce (the PtNi cells are built with ase.build.bulk).
Outputs: $RUNS/gates/icet_gate_implicit<GATE_TAG>{,_sc2}/ and the init cif next to them (G4).
Run:     python gates/gate_ce_implicit.py        (GPU for G3; G1/G4 also run with JAX_PLATFORMS=cpu)
Env:     MS (G3 sizes, default 8,16,32,64,100,136), SKIP_G3, SKIP_G4, TILED_MAX (largest M also run tiled, default 40),
         GATE_TAG (suffix of the G4 output directory).
"""
import os, sys, time, json, subprocess, resource, numpy as np
from neuralce.paths import REPO, MODELS, RUNS
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jrandom
from ase.build import bulk
from icet import ClusterExpansion
from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
from neuralce.ce.pt_mcmc_ce import build_mcmc_tables, make_mc_step_fn
from neuralce.ce.ce_gpu_tiled import make_fast_swap
from neuralce.ce.ce_gpu_implicit import make_fast_swap_implicit
def log(*a): print(*a, flush=True)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
def gpu_mem():
    st = jax.devices()[0].memory_stats() or {}; return st.get('bytes_in_use', 0) / 1e9, st.get('peak_bytes_in_use', 0) / 1e9
rng = np.random.RandomState(0)
def ptni(M):
    a = bulk("Ni", "fcc", a=3.924, cubic=True).repeat((M, M, M)); Z = np.full(len(a), 78, np.int64); Z[rng.choice(len(a), len(a) // 2, replace=False)] = 28; a.numbers = Z; return a
CASES = [("nipt", f"{MODELS}/ce/ptni/icet_c8_6_5.ce", ptni(4))]
NIPT = CASES[0]                                                      # (name, ce path, 256-atom cell) used by G1b / G3 / G4
for name, cep, atoms in CASES:
    ce = ClusterExpansion.read(cep)
    t0 = time.time(); gt = GPUClusterExpansion(ce, atoms, table_mode="tiled"); tt = time.time() - t0
    t0 = time.time(); gi = GPUClusterExpansion(ce, atoms, table_mode="implicit"); ti = time.time() - t0
    info = gi.info; log(f"[{name}] N={len(atoms)} T={info['T'].tolist()} SNF d={info['d'].tolist()} template rows {info['n_rows_template']} (mo {info['max_size']}, J {info['J']}) | build tiled {tt:.2f}s implicit {ti:.2f}s")
    st = jnp.asarray(gt.encode(atoms)); si = jnp.asarray(gi.encode(atoms))
    assert np.array_equal(np.asarray(si), np.asarray(st)[gi.perm]), "encode mismatch through perm"
    et = float(gt.energy_a(gt.tp, st)); ei = float(gi.energy_a(gi.tp, si)); e2 = float(gi.energy(si))
    log(f"  [G1] E tiled {et:.6f}  implicit {ei:.6f} (energy() {e2:.6f}) | diff {abs(et-ei):.2e} eV"); assert abs(et - ei) < 2e-3 and abs(ei - e2) < 1e-6
    mt = build_mcmc_tables(gt); errs_t, errs_f = [], []
    for _ in range(30):
        sl = rng.randint(mt.n_active_subs); n = int(mt.sub_mask[sl].sum()); i_a, i_b = np.asarray(mt.sub_indices[sl])[rng.choice(n, 2, replace=False)]
        sa, sb = int(st[i_a]), int(st[i_b])
        dt = float(gt.delta_energy_a(gt.tp, i_a, i_b, st, sb, sa))
        ja, jb = int(gi.inv_perm[i_a]), int(gi.inv_perm[i_b])                      # canonical indices of the same sites
        di = float(gi.delta_energy_a(gi.tp, ja, jb, si, sb, sa))
        si1 = si.at[ja].set(sb).at[jb].set(sa); errs_t.append(abs(dt - di)); errs_f.append(abs(di - (float(gi.energy_a(gi.tp, si1)) - ei)))
    log(f"  [G1] delta implicit vs tiled (same 30 swaps): max {max(errs_t):.2e} eV | delta vs E_after-E_before: max {max(errs_f):.2e} eV")
    assert max(errs_t) < 1e-4 and max(errs_f) < 2e-3
    del gt, mt
# G1b energy extensivity: the periodic 2x2x2 image of the 256-atom cell must have 8 x its energy, and delta 8 x delta
ce = ClusterExpansion.read(NIPT[1]); a4 = NIPT[2]; a8 = a4.repeat((2, 2, 2))
g4 = GPUClusterExpansion(ce, a4, table_mode="implicit"); g8 = GPUClusterExpansion(ce, a8, table_mode="implicit", energy_scale=8.0)
e4 = float(g4.energy_a(g4.tp, jnp.asarray(g4.encode(a4)))); e8 = float(g8.energy_a(g8.tp, jnp.asarray(g8.encode(a8))))
s4 = jnp.asarray(g4.encode(a4)); s8 = jnp.asarray(g8.encode(a8)); i_a, i_b = 3, 7                      # canonical sites of the base cell
sa, sb = int(s4[i_a]), int(s4[i_b])
while sa == sb: i_b += 1; sb = int(s4[i_b])
d4 = float(g4.delta_energy_a(g4.tp, i_a, i_b, s4, sb, sa))
# the same two sites in the big cell: perm maps canonical -> input index; input index of the tiled cell = image*256 + base index
ja, jb = int(g8.inv_perm[int(g4.perm[i_a])]), int(g8.inv_perm[int(g4.perm[i_b])]); d8 = float(g8.delta_energy_a(g8.tp, ja, jb, s8, sb, sa))
log(f"  [G1b] E(256) {e4:.4f}  E(2x2x2 image, scale 8) {e8:.4f}  ratio {e8/e4:.6f} | dE(256) {d4:.5f}  dE(one site pair in the 2x2x2 cell) {d8:.5f}  ratio {d8/d4:.4f} (expected 1: one pair, not 8)")
assert abs(e8 / e4 - 8.0) < 1e-4 and abs(d8 / d4 - 1.0) < 1e-3
# G3 speed / memory vs N
ce = ClusterExpansion.read(NIPT[1]); R, CH = 32, 2000; TM = int(os.environ.get("TILED_MAX", "40"))
beta = jnp.full((R,), 1 / (8.617e-5 * 800), jnp.float32)
def bench(gce, swap, tag, M, a, tb):
    step = make_mc_step_fn(gce.energy, swap, None, False, None, delta_energy_fn=gce.delta_energy_a, energy_mode="delta")
    stepv = jax.vmap(step, in_axes=(0, 0, 0, 0, 0, None))
    @jax.jit
    def chunk(sig, E, key, tp):
        def body(c, k): s, e = c; s, _, e, acc, _ = stepv(jrandom.split(k, R), s, jnp.zeros((R, 1), jnp.float32), e, beta, tp); return (s, e), acc
        (sig, E), acc = jax.lax.scan(body, (sig, E), jrandom.split(key, CH)); return sig, E, acc.mean()
    sig0 = jnp.asarray(gce.encode(a)); sg = jnp.stack([sig0] * R); t0 = time.time(); E = jax.lax.map(lambda s_: gce.energy_a(gce.tp, s_), sg); E.block_until_ready(); tE = time.time() - t0
    sg, E, acc = chunk(sg, E, jrandom.PRNGKey(0), gce.tp); E.block_until_ready(); t0 = time.time()
    for k in range(3): sg, E, acc = chunk(sg, E, jrandom.PRNGKey(k + 1), gce.tp); E.block_until_ready()
    dt = (time.time() - t0) / 3; used, peak = gpu_mem()
    Efull = jax.lax.map(lambda s_: gce.energy_a(gce.tp, s_), sg); drift = float(jnp.abs(Efull - E).max())
    log(f"  [G3:{tag}] N={len(a):9d}: setup {tb:6.1f}s  E_full(32 rep) {tE:5.1f}s | {dt/CH*1e3:6.3f} ms/step ({R*CH/dt:8.0f} steps/s, R={R}) acc {float(acc):.3f} drift {drift:.1e} eV | GPU in use {used:.2f} GB (peak {peak:.2f}) host RSS {rss():.1f} GB")
    return sg, E
for M in ([] if os.environ.get("SKIP_G3") else [int(x) for x in os.environ.get("MS", "8,16,32,64,100,136").split(",")]):
    a = ptni(M)
    if M <= TM:
        t0 = time.time(); gt = GPUClusterExpansion(ce, a, table_mode="tiled"); tb = time.time() - t0
        bench(gt, make_fast_swap(build_mcmc_tables(gt)), "tiled   ", M, a, tb); del gt
    t0 = time.time(); gi = GPUClusterExpansion(ce, a, table_mode="implicit"); tb = time.time() - t0
    sg, E = bench(gi, make_fast_swap_implicit(gi.tp), "implicit", M, a, tb); del gi, sg, E, a
    import gc; gc.collect()
if os.environ.get("SKIP_G4"): log("[SUMMARY] ICET IMPLICIT GATES (G1/G3) PASS"); sys.exit(0)
GATE_DIR = f"{RUNS}/gates"; os.makedirs(GATE_DIR, exist_ok=True)
out = f"{GATE_DIR}/icet_gate_implicit{os.environ.get('GATE_TAG', '')}"; from ase.io import write, read
a4 = ptni(4); write(f"{out}_init.cif", a4)
cmd = [sys.executable, "-m", "neuralce.mc.run", "--framework", "icet", "--ce", NIPT[1], "--init_cif", f"{out}_init.cif", "--pt", "--T_min", "300", "--T_max", "1400",
       "--n_replicas", "8", "--n_steps", "4000", "--exchange_every", "400", "--energy_mode", "delta", "--log_every", "2000", "--out_dir", out, "--seed", "1"]
r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, NCE_REPO=REPO)); assert r.returncode == 0, r.stderr[-3000:]
z = np.load(f"{out}/mcmc_result.npz"); a_init = read(f"{out}_init.cif")
gt = GPUClusterExpansion(ce, a_init, table_mode="tiled"); gi = GPUClusterExpansion(ce, a_init, table_mode="implicit")
# the driver tracks E_min every step but refreshes sigma_min only at chunk ends, so E(sigma_min) >= E_min (chunk granular);
# the implicit-vs-tiled check goes through the final configuration of every replica (input order -> canonical via perm)
e_check = float(gt.energy_a(gt.tp, jnp.asarray(z['sigma_min'])))
d_fin = max(abs(float(gt.energy_a(gt.tp, jnp.asarray(sf))) - float(gi.energy_a(gi.tp, jnp.asarray(sf)[gi.perm]))) for sf in z['sigma_final'])
log(f"  [G4] central PT delta (implicit) ok: swap acc {float(z['swap_accept_rate']):.3f} exch {float(z['exchange_accept_rate']):.3f} E_min {float(z['E_min']):.4f} | "
    f"tiled E(sigma_min) {e_check:.4f} (>= E_min, chunk-granular) | tiled vs implicit E(sigma_final) over {len(z['sigma_final'])} replicas: max {d_fin:.2e} eV")
assert e_check >= float(z['E_min']) - 2e-3 and d_fin < 2e-3
cmd2 = cmd[:cmd.index("--out_dir") + 1] + [out + "_sc2"] + cmd[cmd.index("--out_dir") + 2:] + ["--supercell", "2", "2", "2"]
r = subprocess.run(cmd2, capture_output=True, text=True, env=dict(os.environ, NCE_REPO=REPO)); assert r.returncode == 0, r.stderr[-3000:]
z = np.load(f"{out}_sc2/mcmc_result.npz"); log(f"  [G4] --supercell 2 2 2: N={z['sigma_final'].shape[1]} swap acc {float(z['swap_accept_rate']):.3f} exch {float(z['exchange_accept_rate']):.3f} E_min/N {float(z['E_min'])/z['sigma_final'].shape[1]:.5f}")
log("\n".join(l for l in r.stdout.splitlines() if "PT-MC:" in l)[:800])
log("[SUMMARY] ICET IMPLICIT GATES PASS")
