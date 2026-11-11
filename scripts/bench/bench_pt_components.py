"""Where does the icet PT production step lose ~2x vs the pure step loop?  Same 1M-atom cell, 32 replicas, same GPU;
variants of the production scan body (copied from neuralce.ce.pt_mcmc_ce.micro) with one ingredient removed each:
  bench      : pure step loop (gate G3 style)                  full: production micro as is
  no_spins   : spins carry removed (zero (R,N) float32 array)   no_Etraj: no per-step energy output
  no_Emin    : no running-min tracking                          no_exch : no exchange cond / key permutation
Prints ms/step and steps/s (3 chunks of 2000 after warm-up).

Env knobs: M = repeats of the conventional fcc cell (default 64 -> 4 M^3 atoms).
Input: $MODELS/ce/ptni/icet_c8_6_5.ce.   Run on a GPU node:  python scripts/bench/bench_pt_components.py"""
import os, time, numpy as np
from neuralce.paths import MODELS
import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
from jax import random as jrandom
from ase.build import bulk
from icet import ClusterExpansion
from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
from neuralce.ce.pt_mcmc_ce import make_mc_step_fn
from neuralce.ce.ce_gpu_implicit import make_fast_swap_implicit
def log(*a): print(*a, flush=True)
M = int(os.environ.get("M", "64")); R, CH = 32, 2000
rng = np.random.RandomState(0); a = bulk("Ni", "fcc", a=3.924, cubic=True).repeat((M, M, M)); Z = np.full(len(a), 78); Z[rng.choice(len(a), len(a) // 2, replace=False)] = 28; a.numbers = Z
ce = ClusterExpansion.read(f"{MODELS}/ce/ptni/icet_c8_6_5.ce"); gce = GPUClusterExpansion(ce, a, table_mode="implicit", energy_scale=len(a) / 256)
N = len(a); log(f"N={N} R={R}"); tp = gce.tp
step = make_mc_step_fn(gce.energy, make_fast_swap_implicit(tp), None, False, None, delta_energy_fn=gce.delta_energy_a, energy_mode="delta")
stepv = jax.vmap(step, in_axes=(0, 0, 0, 0, 0, None))
betas = jnp.asarray(np.geomspace(1 / (8.617e-5 * 300), 1 / (8.617e-5 * 1400), R), jnp.float32)
sig0 = jnp.asarray(gce.encode(a)); sg = jnp.stack([sig0] * R); E0 = jax.lax.map(lambda s_: gce.energy_a(tp, s_), sg)

def make_chunk(variant):
    spins = variant != "no_spins"; etraj = variant != "no_Etraj"; emin = variant != "no_Emin"; exch = variant != "no_exch"
    def micro(state, key, tp_):
        sigma_b, spins_b, E_b, E_min, step_c, rep_of_slot, beta_r = state
        k_step, k_ex = jrandom.split(key)
        if exch:
            slot_of_rep = jnp.zeros(R, jnp.int32).at[rep_of_slot].set(jnp.arange(R, dtype=jnp.int32)); keys = jrandom.split(k_step, R)[slot_of_rep]
        else:
            keys = jrandom.split(k_step, R)
        sigma_b, spins_b2, E_b, acc, _ = stepv(keys, sigma_b, spins_b, E_b, beta_r, tp_)
        if spins: spins_b = spins_b2
        if exch:
            do_ex = ((step_c + 1) % 400) == 0
            def ex(k, E_b, ros):
                kp, ka = jrandom.split(k); parity = jrandom.randint(kp, (), 0, 2); i_idx = jnp.arange(0, R - 1, 2) + parity; j_idx = i_idx + 1; valid = j_idx < R
                i_s = jnp.where(valid, i_idx, 0); j_s = jnp.where(valid, j_idx, 0); E_slot = E_b[ros]
                la = jnp.where(valid, (betas[j_s] - betas[i_s]) * (E_slot[j_s] - E_slot[i_s]), 0.0); sw = valid & (jnp.log(jrandom.uniform(ka, i_idx.shape)) < la)
                perm = jnp.arange(R).at[i_s].set(jnp.where(sw, j_s, i_s)).at[j_s].set(jnp.where(sw, i_s, j_s)); ros = ros[perm]
                sor = jnp.zeros(R, jnp.int32).at[ros].set(jnp.arange(R, dtype=jnp.int32)); return ros, betas[sor]
            rep_of_slot, beta_r = jax.lax.cond(do_ex, lambda: ex(k_ex, E_b, rep_of_slot), lambda: (rep_of_slot, beta_r))
        if emin: E_min = jnp.minimum(E_min, jnp.min(E_b))
        out = E_b[rep_of_slot] if etraj else jnp.float32(0)
        return (sigma_b, spins_b, E_b, E_min, step_c + 1, rep_of_slot, beta_r), out
    @jax.jit
    def chunk(state, key, tp_):
        return jax.lax.scan(lambda st, k: micro(st, k, tp_), state, jrandom.split(key, CH))
    return chunk

@jax.jit
def bench_chunk(sig, E, key, tp_):
    def body(c, k): s, e = c; s, _, e, acc, _ = stepv(jrandom.split(k, R), s, jnp.zeros((R, 1), jnp.float32), e, betas, tp_); return (s, e), acc
    (sig, E), acc = jax.lax.scan(body, (sig, E), jrandom.split(key, CH)); return sig, E
def timeit(run, state):
    state = run(state, jrandom.PRNGKey(0)); jax.block_until_ready(state); t0 = time.time()
    for k in range(3): state = run(state, jrandom.PRNGKey(k + 1)); jax.block_until_ready(state)
    return (time.time() - t0) / 3
dt = timeit(lambda st, k: bench_chunk(st[0], st[1], k, tp), (sg, E0)); log(f"bench     : {dt/CH*1e3:7.3f} ms/step  {R*CH/dt:9.0f} steps/s")
for variant in ("full", "no_spins", "no_Etraj", "no_Emin", "no_exch"):
    chunk = make_chunk(variant)
    spins_b = jnp.zeros((R, 1), jnp.float32) if variant == "no_spins" else jnp.zeros((R, N), jnp.float32)
    st = (sg, spins_b, E0, jnp.min(E0), jnp.int32(0), jnp.arange(R, dtype=jnp.int32), betas)
    dt = timeit(lambda s_, k: chunk(s_, k, tp)[0], st); log(f"{variant:10s}: {dt/CH*1e3:7.3f} ms/step  {R*CH/dt:9.0f} steps/s")
log("BENCH_DONE")
