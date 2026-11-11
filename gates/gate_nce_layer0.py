"""Gates + benchmark for the general (icet-table) GPU layer 0 and the receptive-field local update on the LEGACY
table path (neuralce.nce.gpu_layer0.build_table = icet full orbit list of the supercell; local_update.build_balls /
make_tab / make_local_fns_v2).  The production MC uses the tiled / implicit tables, whose equality with this path is
gated by gate_nce_implicit C1; this gate checks the layer-0 arithmetic itself against icet.

G1  jax cv(sig) == icet cs.get_cluster_vector (and the numpy reference)           max |diff| < 1e-12
G2  jax h0(sig) == numpy h0, mean over active sites == cv, == icet local features   max |diff| < 1e-10
    (the icet local calculator is skipped on self-interacting cells, where it is wrong)
G3  local_delta dE == E_full(new) - E_full(old) for random single flips and same-sublattice swaps at depth 0/1/2
    (mean and mean+max aggregation), and the committed cache == a full recompute          max |diff| < 1e-9
BENCH steps/s: full recompute vs local update, single replica and vmapped over B_REP replicas.

Cases: random Ni/Pt on the ideal fcc lattice (a = 3.924 A, ase.build.bulk), 32 atoms (2x2x2 conventional cells,
self-interacting at these cutoffs) and 256 atoms (4x4x4); the cluster space (primitive, cutoffs, chemical symbols) and
the conv shell edges are those of the shipped PtNi model.  BIG=1 adds the 864-atom 6x6x6 cell.
Inputs: $MODELS/nce/nipt_v3_nomax_sel/model.pkl (cluster-space definition only).
Run:    python gates/gate_nce_layer0.py        (CPU or GPU)
Env:    B_REP (vmapped replicas, default 64), N_STEPS (bench steps, default 200), BIG (default 1), AGG (bench model
        aggregation, mean|mean+max, default mean), GATE_BOTH (0 = only the AGG model at depth 1/2, default 1).

Note (port): make_local_fns_v2.local_delta returns the UNFLIPPED sigma as its third value (callers apply the flips
themselves, as the MC step does).  The original gate used that value as the flipped state for the full recompute,
so E_full(new) == E_full(old) and G3 reported |dE| itself as the error (4.42 at depth 0).  The flipped state is now
built from `flips` explicitly; the tolerances are unchanged.
"""
import os, time, numpy as np
from neuralce.paths import MODELS
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk
from icet import ClusterSpace
from neuralce.nce.mc_model import load_model
from neuralce.nce.gpu_layer0 import build_table, cv_np, h0_np, make_jax_fns
from neuralce.nce.general import local_cluster_features, shell_partners, pack, init_params
from neuralce.nce.local_update import build_balls, balls_full_indexing, make_tab, make_local_fns_v2 as make_local_fns

B_REP = int(os.environ.get("B_REP", "64")); N_STEPS = int(os.environ.get("N_STEPS", "200")); BIG = int(os.environ.get("BIG", "1"))
NIPT = f"{MODELS}/nce/nipt_v3_nomax_sel/model.pkl"
def log(*a): print(*a, flush=True)


def ptni(M, rng):
    """random Ni0.5Pt0.5 on an M x M x M conventional fcc cell (a = 3.924 A), 4 M^3 atoms."""
    a = bulk("Ni", "fcc", a=3.924, cubic=True).repeat((M, M, M)); Z = np.full(len(a), 78, np.int64)
    Z[rng.choice(len(a), len(a) // 2, replace=False)] = 28; a.numbers = Z; return a


def cases():
    # cluster space and conv shells of the shipped PtNi model (primitive fcc Ni, icet cutoffs, [["Ni", "Pt"]])
    mdl = load_model(NIPT)
    cs = ClusterSpace(mdl['primitive'], list(mdl['cutoffs']), mdl['chemical_symbols']); edges = [float(e) for e in mdl['shell_edges']]
    rng = np.random.RandomState(0)
    yield "PtNi-32", cs, ptni(2, rng), edges                         # 2x2x2 conventional cells: self-interacting at these cutoffs
    yield "PtNi-256", cs, ptni(4, rng), edges
    if BIG:   # MCMC-sized cell: this is where a receptive-field update is supposed to pay off
        yield "PtNi-864", cs, ptni(6, rng), edges


worst = {}
for name, cs, atoms, edges in cases():
    log(f"\n==== {name}: N={len(atoms)} K={len(cs)} ====")
    T = build_table(cs, atoms); L0 = make_jax_fns(T); sig = jnp.asarray(T['sig'])
    nA = len(T['active']); log(f"  table: rows {len(T['col'])}  max order {T['idx'].shape[1]}  active {nA}  rows/site max {T['site_rows'].shape[1]}")
    # G1 / G2
    ref = cs.get_cluster_vector(atoms)
    e1 = float(np.abs(np.asarray(L0['cv'](sig)) - ref).max()); e1n = float(np.abs(cv_np(T, T['sig']) - ref).max())
    h_j = np.asarray(L0['h0'](sig)); h_n = h0_np(T, T['sig'])
    e2 = max(float(np.abs(h_j - h_n).max()), float(np.abs(h_j.mean(0) - ref).max()))
    note = ""
    if len(atoms) <= 200:
        h_g = local_cluster_features(cs, atoms, T['active'])         # icet local calculator (wrong on self-interacting cells)
        if np.abs(h_g.mean(0) - ref).max() < 1e-9: e2 = max(e2, float(np.abs(h_j - h_g).max()))
        else: note = "  (icet local-calculator reference invalid here: self-interacting cell, skipped)"
    log(f"  [G1] cv: jax {e1:.2e}  numpy {e1n:.2e}     [G2] h0 (jax==numpy, mean==cv{', ==icet local' if not note else ''}): {e2:.2e}{note}")
    worst[name + " G1"] = e1; worst[name + " G2"] = e2
    assert e1 < 1e-12 and e2 < 1e-10
    # shells (active indexing)
    part = shell_partners(atoms, T['active'], edges)
    _, _, SI, SM = pack([np.zeros((nA, 1))], [part]); SI, SM = SI[0], SM[0]          # (P, nA, W)
    # species alternatives per site (for flips): other allowed species of its sublattice
    Msite = T['M']
    rng = np.random.default_rng(0)
    if len(atoms) < 8:
        continue                                                     # tiny cell: gates only
    n_trials = 20 if len(atoms) <= 200 else 6
    AGG = os.environ.get("AGG", "mean")            # bench model: "mean" (max aggregation dropped) or "mean+max"
    for depth, agg in [(0, AGG), (1, "mean+max"), (1, "mean"), (2, "mean+max"), (2, "mean")]:
        if depth and agg != AGG and os.environ.get("GATE_BOTH", "1") == "0":
            continue
        balls = balls_full_indexing(T, build_balls(T, SI, SM, depth))
        log(f"  depth {depth} [{agg}]: ball sizes " + ", ".join(f"l{l}={b[0].shape[1]}" for l, b in enumerate(balls)))
        C = h_j.shape[1]; mu = h_j.mean(0); sd = h_j.std(0) + 1e-8
        P = init_params(jax.random.PRNGKey(depth), C, SI.shape[0], 16, depth, lin_w=rng.normal(size=C) * 0.1, lin_b=0.0, agg=agg, gate=True)
        if depth:
            P['head']['w'] = jnp.asarray(rng.normal(size=16) * 0.3)
            for L in P['layers']:                                           # make the layers non-trivial
                L['self'] = L['self'] * 3; L['shell'] = L['shell'] * 3; L['gate'] = L['gate'] * 3
                if 'shell_max' in L: L['shell_max'] = L['shell_max'] * 3
        tab = make_tab(T, SI, SM, balls, mu, sd); F = make_local_fns(L0, depth)
        # G3: random flips / swaps
        def rand_flips(sig_np, k):
            act = T['active']; out = []
            for _ in range(k):
                i = int(rng.choice(act)); s = int(sig_np[i]); s_new = int(rng.choice([x for x in range(Msite[i]) if x != s])) if Msite[i] > 1 else s
                out.append((i, s_new))
            return out
        err = 0.0; sig_np = np.array(T['sig'])
        for trial in range(n_trials):
            cache = F['full_cache'](tab, P, jnp.asarray(sig_np)); E0 = float(F['energy_of'](cache, P))
            if trial % 2 == 0:                                          # single flip (+ no-op pad)
                (i, s), = rand_flips(sig_np, 1); flips = np.array([[i, s], [i, s]])
            else:                                                       # swap two sites of the same sublattice
                act = T['active']; i = int(rng.choice(act)); cand = [j for j in act if T['M'][j] == T['M'][i] and sig_np[j] != sig_np[i] and (T['M'][j] > 1)]
                if not cand: continue
                j = int(rng.choice(cand)); flips = np.array([[i, int(sig_np[j])], [j, int(sig_np[i])]])
            sig_j = jnp.asarray(sig_np); fl = jnp.asarray(flips)
            dE, patch, _ = F['local_delta'](tab, P, cache, sig_j, fl); new_cache = F['commit'](cache, patch, True)
            sig_new = sig_j.at[fl[:, 0]].set(fl[:, 1].astype(sig_j.dtype))   # local_delta leaves sigma unflipped; apply the flips here
            E1 = float(F['energy_of'](F['full_cache'](tab, P, sig_new), P))
            err = max(err, abs(float(dE) - (E1 - E0)))
            # cache consistency: the locally updated cache must equal a full recompute
            fc = F['full_cache'](tab, P, sig_new)
            err = max(err, max((float(jnp.abs(a - b).max()) for a, b in zip(jax.tree_util.tree_leaves({k: v for k, v in new_cache.items() if k not in ('slot', 'slot2')}), jax.tree_util.tree_leaves({k: v for k, v in fc.items() if k not in ('slot', 'slot2')}))), default=0.0))
            sig_np = np.array(sig_new)                                    # walk on
        log(f"  [G3] depth {depth} [{agg}]: max|dE_local - dE_full| (and cache) = {err:.2e}"); worst[f"{name} G3 d{depth} {agg}"] = err
        assert err < 1e-9, err
        # BENCH single replica
        sig_j = jnp.asarray(sig_np); cache = F['full_cache'](tab, P, sig_j)
        flips_list = [np.array([[i, s], [i, s]]) for (i, s) in rand_flips(sig_np, N_STEPS)]
        for _ in range(3): F['full_cache'](tab, P, sig_j)['e'].block_until_ready()
        t0 = time.time()
        for f in flips_list: F['full_cache'](tab, P, sig_j.at[f[0, 0]].set(f[0, 1]))['e'].block_until_ready()
        t_full = (time.time() - t0) / N_STEPS
        for _ in range(3): F['local_delta'](tab, P, cache, sig_j, jnp.asarray(flips_list[0]))[0].block_until_ready()
        t0 = time.time()
        for f in flips_list: F['local_delta'](tab, P, cache, sig_j, jnp.asarray(f))[0].block_until_ready()
        t_loc = (time.time() - t0) / N_STEPS
        # BENCH vmapped over B replicas
        vfull = jax.jit(jax.vmap(lambda s: F['full_cache'](tab, P, s)['e'].mean()))
        vloc = jax.jit(jax.vmap(lambda c, s, f: F['local_delta'](tab, P, c, s, f)[0]))
        sigs = jnp.stack([sig_j] * B_REP); caches = jax.vmap(lambda s: F['full_cache'](tab, P, s))(sigs)
        fl = jnp.asarray(np.stack([np.array([[i, s], [i, s]]) for (i, s) in rand_flips(sig_np, B_REP)]))
        for _ in range(3): vfull(sigs).block_until_ready(); vloc(caches, sigs, fl).block_until_ready()
        t0 = time.time()
        for _ in range(N_STEPS // 4): vfull(sigs).block_until_ready()
        tb_full = (time.time() - t0) / (N_STEPS // 4)
        t0 = time.time()
        for _ in range(N_STEPS // 4): vloc(caches, sigs, fl).block_until_ready()
        tb_loc = (time.time() - t0) / (N_STEPS // 4)
        if agg != AGG: continue                                      # per-call bench only for the bench model
        log(f"  [BENCH] depth {depth}: single  full {1/t_full:8.1f} steps/s  local {1/t_loc:8.1f} steps/s  = {t_full/t_loc:5.1f}x   |  "
            f"B={B_REP}: full {B_REP/tb_full:8.0f} steps/s  local {B_REP/tb_loc:8.0f} steps/s  = {tb_full/tb_loc:5.1f}x")
log("\n[SUMMARY] gates: " + "  ".join(f"{k}={v:.1e}" for k, v in worst.items()))
log("[SUMMARY] ALL GATES PASS")
