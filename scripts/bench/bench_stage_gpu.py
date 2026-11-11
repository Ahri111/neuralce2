"""Per-stage cost of ONE Monte Carlo swap step on the GPU, steady state (jit compiled, inside lax.scan, R replicas vmapped):
   stage 1  proposal            (draw two sites of a sublattice)
   stage 2  energy evaluation   (local dE of the swap: icet-GPU delta kernel / cluster-conv local update)
   stage 3  accept + update     (Metropolis test, write sigma [+ commit the conv caches])
Timed as nested scans  A: proposal only,  B: proposal + dE,  C: full step;  stage times = A, B-A, C-B  (ms per step of all
R replicas, and microseconds per single-replica step).  PtNi, R = 32, sizes via env SIZES (K = repeats of the 256-atom cell).

Env knobs: FW = icet | nce (aliases conv, ce_cnn), R replicas (32), CH steps per timed chunk (2000), SIZES "2:4:8:16".
Inputs: $MODELS/structures/ptni/ref_L10_NiPt.cif, $MODELS/ce/ptni/icet_c8_6_5.ce, $MODELS/nce/nipt_v3_nomax_sel/model.pkl.
Output: <out>/gpu_<fw>_<gpu>[_R<R>].json  (--out, default $RESULTS/bench_stage; the nce rows keep the historical file tag
"conv", i.e. gpu_conv_<gpu>.json, so that they match the stored benchmark files).
Run (GPU node):  python scripts/bench/bench_stage_gpu.py   (see slurm/bench_stage_gpu.sh)"""
import os, time, json, argparse, numpy as np
from neuralce.paths import MODELS, RESULTS
import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
from jax import random as jrandom
from ase.io import read
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{RESULTS}/bench_stage", help="output directory"); args = ap.parse_args()
FW = os.environ.get("FW", "icet"); FW = {"nce": "conv", "ce_cnn": "conv"}.get(FW, FW); assert FW in ("icet", "conv"), FW
R = int(os.environ.get("R", "32")); CH = int(os.environ.get("CH", "2000")); SIZES = [int(k) for k in os.environ.get("SIZES", "2:4:8:16").replace(",", ":").split(":")]
OUT = args.out; os.makedirs(OUT, exist_ok=True); KB = 8.617333e-5
gpu = jax.devices()[0].device_kind if hasattr(jax.devices()[0], "device_kind") else str(jax.devices()[0]); gpu = gpu.replace(" ", "_")
def log(*a): print(*a, flush=True)
base = read(f"{MODELS}/structures/ptni/ref_L10_NiPt.cif"); betas = jnp.asarray(1.0 / (KB * np.linspace(700, 1100, R)), jnp.float32)


def timeit(fn, *args):
    out = fn(*args, jrandom.PRNGKey(0)); jax.block_until_ready(out); t0 = time.perf_counter()
    for k in range(3): out = fn(*args, jrandom.PRNGKey(k + 1)); jax.block_until_ready(out)
    return (time.perf_counter() - t0) / 3 / CH                      # seconds per step (all R replicas)


res = []
for K in SIZES:
    a = base.repeat((K, K, K)); N = len(a); rng = np.random.RandomState(0); Z = a.numbers.copy(); rng.shuffle(Z); a.numbers = Z     # disordered 50/50
    if FW == "icet":
        from icet import ClusterExpansion
        from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
        from neuralce.ce.ce_gpu_implicit import make_fast_swap_implicit, delta_energy_a
        ce = ClusterExpansion.read(f"{MODELS}/ce/ptni/icet_c8_6_5.ce"); gce = GPUClusterExpansion(ce, a, table_mode="implicit", energy_scale=N / 256)
        tp = gce.tp; propose = make_fast_swap_implicit(tp); sig0 = jnp.asarray(gce.encode(a)); sg = jnp.stack([sig0] * R); E0 = jnp.zeros(R)

        def one(key, sigma, E, beta, tp_, stage):
            kp, ka = jrandom.split(key); i_a, i_b, valid = propose(kp, sigma)
            if stage == 1: return sigma, E + (i_a + i_b).astype(E.dtype) * 1e-30
            dE = delta_energy_a(tp_, i_a, i_b, sigma, sigma[i_b], sigma[i_a])
            if stage == 2: return sigma, E + dE * 1e-30
            acc = valid & (jnp.log(jrandom.uniform(ka)) < -beta * dE)
            sa, sb = sigma[i_a], sigma[i_b]; sigma = sigma.at[i_a].set(jnp.where(acc, sb, sa)).at[i_b].set(jnp.where(acc, sa, sb))
            return sigma, E + jnp.where(acc, dE, 0.0)

        def make(stage):
            v = jax.vmap(lambda k, s, e, b, tp_: one(k, s, e, b, tp_, stage), in_axes=(0, 0, 0, 0, None))
            @jax.jit
            def run(sig, E, tp_, key):
                def body(c, k): s, e = c; s, e = v(jrandom.split(k, R), s, e, betas, tp_); return (s, e), None
                (sig, E), _ = jax.lax.scan(body, (sig, E), jrandom.split(key, CH)); return sig, E
            return lambda key_state_sig, E, key: run(key_state_sig, E, tp, key)
        tA, tB, tC = (timeit(make(s), sg, E0) for s in (1, 2, 3))
        # direct timing of the energy stage and of the accept + update stage (fixed site pairs), because stage differences cannot resolve a stage this short
        ia0 = jnp.arange(R, dtype=jnp.int32) * 7 % N; ib0 = (ia0 + 1) % N

        def one_direct(key, sigma, E, beta, tp_, i_a, i_b, stage):
            if stage == 2:
                dE = delta_energy_a(tp_, i_a, i_b, sigma, sigma[i_b], sigma[i_a]); return sigma, E + dE * 1e-30
            acc = jnp.log(jrandom.uniform(key)) < -beta * 0.02                      # Metropolis test with a fixed dE, then the two writes
            sa, sb = sigma[i_a], sigma[i_b]; sigma = sigma.at[i_a].set(jnp.where(acc, sb, sa)).at[i_b].set(jnp.where(acc, sa, sb)); return sigma, E + jnp.where(acc, 0.02, 0.0)

        def make_direct(stage):
            v = jax.vmap(lambda k, s_, e, b, tp_, i_a, i_b: one_direct(k, s_, e, b, tp_, i_a, i_b, stage), in_axes=(0, 0, 0, 0, None, 0, 0))
            @jax.jit
            def run(sig, E, tp_, key):
                def body(c, k): s_, e = c; s_, e = v(jrandom.split(k, R), s_, e, betas, tp_, ia0, ib0); return (s_, e), None
                (sig, E), _ = jax.lax.scan(body, (sig, E), jrandom.split(key, CH)); return sig, E
            return lambda sig, E, key: run(sig, E, tp, key)
        tE_direct, tU_direct = (timeit(make_direct(s), sg, E0) for s in (2, 3))
    else:
        from neuralce.nce.mc_model import load_model, setup
        mdl = load_model(f"{MODELS}/nce/nipt_v3_nomax_sel/model.pkl"); mc = setup(mdl, a, seed_idx=0, dtype="float32"); propose = mc["propose"]
        F = mc["F"]; tab = mc["tab"]; P = mc["P"]; scale = float(mc["ys"]) * mc["n_atoms"]          # same calls as neuralce.nce.mcmc._make_step
        sig0 = mc["sig0"]; sg = jnp.stack([sig0] * R); c1 = jax.jit(lambda s_, t_: F["full_cache"](t_, P, s_))(sig0, tab); cache0 = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], R, axis=0), c1); del c1   # all temperatures start from the same configuration: build the cache once (the vmapped build needs R x the transient memory)

        def one(key, sigma, cache, beta, tab_, stage):
            kp, ka = jrandom.split(key); i, j = propose(kp); flips = jnp.array([[i, sigma[j]], [j, sigma[i]]])
            if stage == 1: return sigma, cache, (flips.sum()).astype(jnp.float32) * 1e-30
            dE, patch, _ = F["local_delta"](tab_, P, cache, sigma, flips); dE_eV = dE * scale
            if stage == 2: return sigma, cache, dE_eV
            acc = (dE_eV <= 0) | (jrandom.uniform(ka) < jnp.exp(-beta * dE_eV))
            cache = F["commit"](cache, patch, acc); sigma = sigma.at[flips[:, 0]].set(jnp.where(acc, flips[:, 1].astype(sigma.dtype), sigma[flips[:, 0]]))
            return sigma, cache, dE_eV

        def make(stage):
            v = jax.vmap(lambda k, s_, c, b, t_: one(k, s_, c, b, t_, stage), in_axes=(0, 0, 0, 0, None))
            @jax.jit
            def run(sig, cache, tab_, key):
                def body(c, k): s_, ca, acc_ = c; s_, ca, d = v(jrandom.split(k, R), s_, ca, betas, tab_); return (s_, ca, acc_ + d), None
                (sig, cache, tot), _ = jax.lax.scan(body, (sig, cache, jnp.zeros(R, jnp.float32)), jrandom.split(key, CH)); return sig, cache, tot
            return lambda sig, cache, key: run(sig, cache, tab, key)
        tA, tB, tC = (timeit(make(s), sg, cache0) for s in (1, 2, 3))
    mem = None
    try: mem = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2 ** 20
    except Exception: pass
    direct = dict(energy=tE_direct / R * 1e6, accept=tU_direct / R * 1e6) if FW == "icet" else None
    r = dict(fw=FW, gpu=gpu, n_atoms=N, R=R, us_per_replica_step_direct=direct, ms_step_propose=tA * 1e3, ms_step_energy=(tB - tA) * 1e3, ms_step_accept=(tC - tB) * 1e3, ms_step_total=tC * 1e3,
             us_per_replica_step=dict(propose=tA / R * 1e6, energy=(tB - tA) / R * 1e6, accept=(tC - tB) / R * 1e6, total=tC / R * 1e6), steps_per_s=R / tC, peak_MB=mem)
    res.append(r); log(json.dumps(r)); json.dump(res, open(f"{OUT}/gpu_{FW}_{gpu}" + ("" if R == 32 else f"_R{R}") + ".json", "w"), indent=1)      # R = 32 is the Fig. 4 reference file; other temperature counts get their own file
log("STAGE_BENCH_DONE")
