"""Cluster convolution on ANY icet lattice (not NiPt-specific).

Layer 0  = icet's own site-local cluster vector (any primitive cell, any number of
           sublattices / species), rescaled so the mean over active sites is exactly
           the icet cluster vector.  A linear readout of layer 0 is therefore the
           cluster expansion itself (gated per structure).
Shells   = neighbour shells taken from the ideal geometry (distance bins), one
           weight matrix per shell, masked means -> works for perovskites, mixed
           cell sizes, inert sublattices.
Spin     = optional per-site Heisenberg-pair features (shell x species-pair), whose
           site mean is the global spin feature of spin/cv10_feni_cr_icet.py.

Everything is padded to a fixed number of sites with masks so a whole dataset of
different cell sizes is one batch on the GPU.
"""
from __future__ import annotations
import numpy as np
from ase.neighborlist import neighbor_list


# ── layer 0: icet local cluster vectors ────────────────────────────────────
def orbit_orders(cs):
    return np.array([o['order'] for o in cs.as_list], dtype=np.int64)


def local_cluster_features(cs, atoms, active):
    """(n_active, K) site features; mean over active sites == cs.get_cluster_vector.

    icet's get_local_cluster_vector(i) counts every cluster that contains site i,
    so summing it over sites counts an order-n cluster n times:
        sum_i local_i = order * cv     (verified on STFO 160-atom cells)
    hence h0_i = n_active * local_i / order.  The zerolet is set to 1.
    """
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator
    ce = ClusterExpansion(cs, np.ones(len(cs)))          # ones: prune() must keep all
    calc = ClusterExpansionCalculator(atoms, ce)
    occ = list(atoms.numbers)
    L = np.array([calc.cpp_calc.get_local_cluster_vector(occ, int(i)) for i in active])
    order = np.maximum(orbit_orders(cs), 1)
    h0 = L * (len(active) / order)[None, :]
    h0[:, 0] = 1.0
    return h0


def gate_layer0(cs, atoms, active, h0, tol=1e-9):
    cv = cs.get_cluster_vector(atoms)
    err = np.abs(h0.mean(0) - cv).max()
    assert err < tol, f"layer-0 gate failed: max|mean_i h0 - cv| = {err:.3e}"
    return err


def enlarge_if_self_interacting(cs, atoms, active_of, spins=None, max_rep=6, tol=1e-9):
    """icet's local calculator is wrong when a cluster can wrap onto itself
    (tiny cells: gate error 0.17 on a 4-atom FeNiCr cell).  Repeat the cell by the
    smallest factors that make the layer-0 gate pass; every site of the repeated
    cell is a periodic copy, so site features / the site-mean readout are unchanged
    physics.  spins are tiled in ase's repeat order (block per repetition).
    The criterion is EMPIRICAL (the gate itself), because icet's
    is_supercell_self_interacting flags 32- and 108-atom fcc cells whose local
    vectors are in fact exact (gate 1e-16); repeating those would inflate the site
    axis 8x for nothing.  Returns (atoms, spins, repeat, h0) with h0 already gated."""
    cv0 = cs.get_cluster_vector(atoms)
    L = np.linalg.norm(atoms.cell.array, axis=1)
    cands = [(i, j, k) for i in range(1, max_rep + 1) for j in range(1, max_rep + 1) for k in range(1, max_rep + 1)]
    cands.sort(key=lambda r: (r[0] * r[1] * r[2], -min(r[0] * L[0], r[1] * L[1], r[2] * L[2])))
    for r in cands:
        rep = atoms if r == (1, 1, 1) else atoms.repeat(r)
        act = active_of(rep)
        h0 = local_cluster_features(cs, rep, act)
        if np.abs(h0.mean(0) - cv0).max() < tol and np.abs(cs.get_cluster_vector(rep) - cv0).max() < tol:
            return rep, (None if spins is None else np.tile(spins, int(np.prod(r)))), r, h0
    raise RuntimeError("could not remove self-interaction by repetition")


# ── shells from geometry ───────────────────────────────────────────────────
def shell_partners(atoms, active, shell_edges):
    """Per active site, per shell: list of active partner sites (local indices,
    repeats allowed = periodic images, which is what a tiny cell really sees)."""
    loc = -np.ones(len(atoms), dtype=np.int64); loc[active] = np.arange(len(active))
    i, j, d = neighbor_list('ijd', atoms, float(shell_edges[-1]) + 1e-6)
    keep = (loc[i] >= 0) & (loc[j] >= 0) & (d > 1e-6)
    i, j, d = loc[i[keep]], loc[j[keep]], d[keep]
    shell = np.digitize(d, shell_edges[1:-1], right=True)      # (lo, hi] bins
    P = len(shell_edges) - 1
    out = [[[] for _ in range(len(active))] for _ in range(P)]
    for a, b, s in zip(i, j, shell):
        out[s][a].append(int(b))
    return out


def spin_site_features(atoms, active, spins, shell_edges, pairs):
    """Per-site Heisenberg-pair features (P*len(pairs),): 0.5*sum_j s_i s_j per
    (shell, sorted species pair).  Site mean == global spin_features()."""
    sym = np.array(atoms.get_chemical_symbols())
    pidx = {p: k for k, p in enumerate(pairs)}
    P = len(shell_edges) - 1
    part = shell_partners(atoms, active, shell_edges)
    f = np.zeros((len(active), P * len(pairs)))
    for s in range(P):
        for a in range(len(active)):
            ia = active[a]
            for b in part[s][a]:
                k = pidx[tuple(sorted((sym[ia], sym[active[b]])))]
                f[a, s * len(pairs) + k] += 0.5 * spins[ia] * spins[active[b]]
    return f


def spin_node_planes(atoms, active, spins, shell_edges, lo=0):
    """Extra layer-0 spin PLANES per site (same grammar as the kernel-output planes
    in the figure): [ s_i^2 ,  s_i * sum_{j in shell p} s_j  for each shell p >= lo ].
    lo > 0 drops the alignment planes of the innermost shells (the 1NN one pushes the
    Fe-Ni 1NN chemical SRO up, which the MC exposed); s_i^2 is shell-independent and stays.
    All even under global spin flip.  Site mean of plane p+1 = 2x the shell-p
    Heisenberg sum, so nothing new is invented -- it is the spin field itself laid
    on the lattice next to the chemistry planes."""
    part = shell_partners(atoms, active, shell_edges)
    P = len(shell_edges) - 1
    f = np.zeros((len(active), 1 + P - lo))
    s = np.asarray(spins, float)
    for a in range(len(active)):
        f[a, 0] = s[active[a]] ** 2
        for p in range(lo, P):
            f[a, 1 + p - lo] = s[active[a]] * sum(s[active[b]] for b in part[p][a])
    return f


def spin_env_planes(atoms, active, spins, shell_edges, pairs, species):
    """Spin-decorated 3-body planes (v3, user's B8): the (shell p, species pair ab)
    Heisenberg plane of site i multiplied by the 1NN composition of site i,
        f_{i,(p,ab,c)} = [0.5 s_i sum_{j in p(i), pair ab} s_j] * n_c(1NN of i) / n_1NN(i)
    = sum over triplets (i, j, k) of s_i s_j theta_c(sigma_k) with k a 1NN of i.
    Gives the linear readout a J that depends on the chemical environment."""
    sym = np.array(atoms.get_chemical_symbols())
    base = spin_site_features(atoms, active, spins, shell_edges, pairs)      # (n, P*len(pairs))
    part = shell_partners(atoms, active, shell_edges)
    comp = np.zeros((len(active), len(species)))
    for a in range(len(active)):
        nb = part[0][a]
        for b in nb:
            comp[a, species.index(sym[active[b]])] += 1.0
        if nb: comp[a] /= len(nb)
    return (base[:, :, None] * comp[:, None, :]).reshape(len(active), -1)


# ── packing a dataset ──────────────────────────────────────────────────────
def pack(feats, partners, s_max=None, w_max=None):
    """feats: list of (n_i, C); partners: list of per-structure [P][n_i][list].
    -> H (N,S,C), site_mask (N,S), S_idx (N,P,S,W) int32, S_mask (N,P,S,W)"""
    N = len(feats); P = len(partners[0]); C = feats[0].shape[1]
    S = s_max or max(f.shape[0] for f in feats)
    W = w_max or max(len(l) for pp in partners for sh in pp for l in sh)
    H = np.zeros((N, S, C)); M = np.zeros((N, S))
    SI = np.zeros((N, P, S, W), np.int32); SM = np.zeros((N, P, S, W))
    for n, (f, pp) in enumerate(zip(feats, partners)):
        k = f.shape[0]; H[n, :k] = f; M[n, :k] = 1
        for p in range(P):
            for s in range(k):
                l = pp[p][s][:W]
                SI[n, p, s, :len(l)] = l; SM[n, p, s, :len(l)] = 1
    return H, M, SI, SM


# ── network ────────────────────────────────────────────────────────────────
def init_params(key, C, n_shells, width, depth, lin_w=None, lin_b=0.0, agg="mean+max", gate=True):
    """v2: the model is  E = <h0, lin> (site mean)  +  <h_L, w> (site mean)  + b.

    The first term is the cluster expansion itself (a linear readout of layer 0,
    initialised from the lasso solution on the training fold, `lin_w`/`lin_b`), the
    second is the cluster-convolution correction with its head initialised at ZERO.
    So at initialisation the network IS the fitted CE, and depth can only add.
    v3 (2026-09-04, user's B6/B7): per shell both a MEAN and a MAX aggregate get their
    own matrix (`shell`, `shell_max`), and the tanh output is multiplied by a sigmoid
    GATE driven by the shell means (`gate`, `gb`; bias +2 so the gate starts ~0.88) --
    an explicit product between a site's own features and its neighbourhood, i.e.
    spin x chemistry coupling without waiting for tanh to invent it."""
    import jax, jax.numpy as jnp
    dims = [C] + [width] * depth
    layers = []
    for l in range(depth):
        key, *ks = jax.random.split(key, 3 * n_shells + 2)
        sc = np.sqrt(1.0 / (dims[l] * (n_shells + 1)))
        L = dict(self=jax.random.normal(ks[0], (dims[l], dims[l + 1])) * sc,
                 shell=jnp.stack([jax.random.normal(ks[1 + p], (dims[l], dims[l + 1])) * sc
                                  for p in range(n_shells)]),
                 b=jnp.zeros(dims[l + 1]))
        if agg == "mean+max":
            L['shell_max'] = jnp.stack([jax.random.normal(ks[1 + n_shells + p], (dims[l], dims[l + 1])) * sc
                                        for p in range(n_shells)])
        if gate:
            L['gate'] = jnp.stack([jax.random.normal(ks[1 + 2 * n_shells + p], (dims[l], dims[l + 1])) * sc
                                   for p in range(n_shells)])
            L['gb'] = jnp.full((dims[l + 1],), 2.0)
        layers.append(L)
    head = dict(w=jnp.zeros((dims[-1],)) if depth > 0 else jnp.zeros((0,)), b=jnp.asarray(float(lin_b)))
    lin = jnp.asarray(lin_w, dtype=jnp.float64) if lin_w is not None else jnp.zeros((C,))
    return dict(layers=layers, head=head, lin=lin)


def make_forward():
    """E(P, H, M, SI, SM) per structure (intensive).

        z_mean[b,p,s,:] = masked mean over shell p partners of h[b,partner,:]
        z_max [b,p,s,:] = masked max  (v3, optional)
        pre  = h W_self + sum_p z_mean_p W_p (+ sum_p z_max_p W'_p) + b
        gate = sigmoid( sum_p z_mean_p G_p + gb )                      (v3, optional)
        h'   = tanh(pre) * gate * site_mask
        E    = site-mean(H . lin) + site-mean(h_L . w) + b
    depth 0 -> linear readout of layer 0 = cluster expansion (+ spin term)."""
    import jax, jax.numpy as jnp
    gather = jax.vmap(lambda hb, sb: hb[sb])                  # (S,C),(P,S,W) -> (P,S,W,C)

    def layer(L, h, M, SI, SM):
        g = gather(h, SI)                                     # (B,P,S,W,C)
        m = SM[..., None]
        z = (g * m).sum(-2) / jnp.maximum(SM.sum(-1), 1.0)[..., None]
        pre = h @ L['self'] + jnp.einsum('bpsc,pcd->bsd', z, L['shell']) + L['b']
        if 'shell_max' in L:
            zmax = jnp.where(m > 0, g, -1e9).max(-2)
            zmax = jnp.where(SM.sum(-1)[..., None] > 0, zmax, 0.0)
            pre = pre + jnp.einsum('bpsc,pcd->bsd', zmax, L['shell_max'])
        out = jnp.tanh(pre)
        if 'gate' in L:
            out = out * jax.nn.sigmoid(jnp.einsum('bpsc,pcd->bsd', z, L['gate']) + L['gb'])
        return out * M[..., None]

    def fwd(P, H, M, SI, SM):
        e = (H @ P['lin']) * M                                 # cluster expansion term
        if P['layers']:
            h = H
            for L in P['layers']:
                h = layer(L, h, M, SI, SM)
            e = e + (h @ P['head']['w']) * M                  # cluster-conv correction
        return e.sum(-1) / M.sum(-1) + P['head']['b']
    return fwd


_STEP = {}


def make_step(fwd):
    """One jitted Adam step.  Data are ARGUMENTS (never closed over): closing over
    the padded tensors turns them into compile-time constants, which made XLA
    constant-fold a 311040x70 gather at every fit."""
    import jax, jax.numpy as jnp
    if fwd in _STEP:
        return _STEP[fwd]

    def loss(P, H, M, SI, SM, y, wd, w):
        # weight decay on the convolution branch only; the CE term keeps its ridge prior
        # w: per-structure loss weights (mean 1; all ones = plain MSE)
        pen = sum(jnp.sum(a ** 2) for a in jax.tree_util.tree_leaves(P['layers'])) + jnp.sum(P['head']['w'] ** 2)
        return jnp.mean(w * (fwd(P, H, M, SI, SM) - y) ** 2) + wd * pen

    @jax.jit
    def step(P, m, v, s, H, M, SI, SM, y, wd, lr, w):
        _, G = jax.value_and_grad(loss)(P, H, M, SI, SM, y, wd, w)
        m = jax.tree_util.tree_map(lambda a, b: .9 * a + .1 * b, m, G)
        v = jax.tree_util.tree_map(lambda a, b: .999 * a + .001 * b ** 2, v, G)
        b1, b2 = 1 - .9 ** s, 1 - .999 ** s
        P = jax.tree_util.tree_map(
            lambda p, mm, vv: p - lr * (mm / b1) / (jnp.sqrt(vv / b2) + 1e-8), P, m, v)
        return P, m, v
    _STEP[fwd] = step
    return step


_VAL = {}


def make_val_loss(fwd):
    import jax, jax.numpy as jnp
    if fwd not in _VAL:
        _VAL[fwd] = jax.jit(lambda P, H, M, SI, SM, y: jnp.mean((fwd(P, H, M, SI, SM) - y) ** 2))
    return _VAL[fwd]


def train(fwd, P, data_tr, y_tr, wd, epochs=2000, lr=3e-3, val=None, eval_every=25, patience=300, w_tr=None):
    """Full-batch Adam on standardised targets.

    val=(H,M,SI,SM,y): early stopping (v3, user's A2) -- evaluate every `eval_every`
    steps, keep the best parameters, stop after `patience` steps without improvement.
    Returns (params, best_step).  Without val: fixed `epochs`, returns (params, epochs)."""
    import jax, jax.numpy as jnp
    H, M, SI, SM = data_tr
    step = make_step(fwd)
    w_tr = jnp.ones_like(y_tr) if w_tr is None else jnp.asarray(w_tr)
    m = jax.tree_util.tree_map(jnp.zeros_like, P)
    v = jax.tree_util.tree_map(jnp.zeros_like, P)
    if val is None:
        for s in range(1, epochs + 1):
            P, m, v = step(P, m, v, float(s), H, M, SI, SM, y_tr, wd, lr, w_tr)
        return P, epochs
    vloss = make_val_loss(fwd); best, best_P, best_s = np.inf, P, 0
    for s in range(1, epochs + 1):
        P, m, v = step(P, m, v, float(s), H, M, SI, SM, y_tr, wd, lr, w_tr)
        if s % eval_every == 0:
            l = float(vloss(P, *val))
            if l < best - 1e-12:
                best, best_P, best_s = l, P, s
            elif s - best_s >= patience:
                break
    return best_P, best_s
