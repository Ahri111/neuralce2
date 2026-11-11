"""Exact (enumerated) thermodynamics of the final PtNi cluster-convolution model on a small fcc cell.

Every configuration of the cell at fixed composition is visited once up to lattice translations:
  * the N translations of the cell (fcc primitive lattice modulo the supercell) form an abelian group  Z_m x Z_(N/m);
    a configuration is a bit word laid out as N/m rows of m bits (row r, column c  <->  site u^r t^c (site 0)),
    translations are cyclic shifts of rows/columns, and the representative of an orbit is the smallest word;
  * enumeration: row 0 runs over the binary necklaces of length m (words minimal under column rotation), the other rows
    over all combinations of the remaining bits at the right total weight; a cheap necessary condition (row 0 <= every
    rotation of every row) prunes ~ (N-1)/N of the candidates before the exact 48-way test;
  * each representative carries the weight |orbit| = N / |stabiliser|; sum of weights == C(N, N_Pt) (completeness gate).
Energies: the model's own icet instance table (gpu_layer0.build_table) -> site layer 0 -> one gated conv layer -> head,
batched on the GPU, float64 forward; gated against mc['energy'] (the MC path) on random configurations.
Accumulates  sum w e^{-b(E-E0)},  sum w (E-E0) e^{..},  sum w (E-E0)^2 e^{..}  on a temperature grid (float64) and a
density-of-states histogram (1e-6 eV/atom bins), plus E_min / E_max and the ground-state word.

Model: $MODELS/nce/nipt_v3_nomax_sel/model.pkl.  Worker parts and progress files: --out (default $RUNS/ptni_exact/enum/N<N>);
--merge writes exact_N<N>.{npz,json} into that directory and copies them to $RESULTS/ptni_exact/ (read by ptni_exact_pt_analyze.py).
Env: CK_SEC checkpoint interval (600 s), STOP_AFTER_SEC test hook (stop after the next checkpoint).
usage (GPU node, see slurm/ptni_exact_enum.sh):
  python scripts/bench/ptni_exact_enum.py --cell 2 2 3 --out DIR --worker K --n_workers W      # worker K of W (necklaces split round-robin)
  python scripts/bench/ptni_exact_enum.py --cell 2 2 2 --gate                                  # 32-atom cell: brute force vs enumeration
  python scripts/bench/ptni_exact_enum.py --merge DIR                                          # merge the workers' parts
"""
from __future__ import annotations
import os, sys, time, json, argparse, math, shutil
import numpy as np
from neuralce.paths import MODELS, RUNS, RESULTS
MODEL = f"{MODELS}/nce/nipt_v3_nomax_sel/model.pkl"
KB = 8.617333262e-5
A_FCC = 3.924                                                     # model primitive: a/2 = 1.962 A
E_REF_ATOM = -5.99437                                             # L1_0 NiPt, the guide's low-T anchor (eV/atom)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------------------- cell / model
def build_cell(rep):
    from ase.build import bulk
    at = bulk("Ni", "fcc", a=A_FCC, cubic=True).repeat(tuple(rep))
    at.pbc = True
    return at


def model_parts(atoms):
    """table (rows), shell adjacency, expanded params; plus the MC-path energy for the gate."""
    import jax, jax.numpy as jnp
    from icet import ClusterSpace
    from neuralce.nce.mc_model import load_model, expand_params, setup
    from neuralce.nce.gpu_layer0 import build_table
    from neuralce.nce.general import shell_partners
    from neuralce.nce.gpu_layer0_tiled import site_decompose
    mdl = load_model(MODEL)
    site_decompose(mdl['primitive'], atoms, tol=1e-3)            # ideal lattice of the model primitive (raises otherwise)
    cs = ClusterSpace(mdl['primitive'], list(mdl['cutoffs']), mdl['chemical_symbols'])
    T = build_table(cs, atoms)
    N = len(atoms); assert len(T['active']) == N, "every site must be active"
    assert int(T['M'].max()) == 2 and int(T['pidx'].max()) == 0, "binary alloy expected (theta = 2 s - 1)"
    P, mu, sd = expand_params(mdl, 0)
    K = int(T['K']); C = K + 1
    assert C == int(P['lin'].shape[0]), (C, P['lin'].shape)
    # shell adjacency (row-normalised mean over partners, periodic images counted with multiplicity, as pack/make_forward)
    part = shell_partners(atoms, T['active'], list(mdl['shell_edges']))     # [shell][site] -> partner list (images repeated)
    nP = len(part); A = np.zeros((nP, N, N))
    for p in range(nP):
        for i in range(N):
            lst = part[p][i]
            for j in lst: A[p, i, j] += 1.0
            if len(lst): A[p, i] /= len(lst)
    # icet gate: the table's cluster vector (site mean of layer 0) == ClusterSpace.get_cluster_vector on random decorations
    from neuralce.nce.gpu_layer0 import cv_np
    rng = np.random.default_rng(1); err = 0.0
    for _ in range(5):
        sig = np.zeros(N, np.int32); sig[rng.choice(N, N // 2, replace=False)] = 1
        dec = atoms.copy(); dec.numbers = np.where(sig == 1, 78, 28)
        err = max(err, float(np.abs(cv_np(T, sig) - cs.get_cluster_vector(dec)).max()))
    assert err < 1e-9, f"instance table != icet cluster vector in this cell (max {err:.2e})"
    log(f"icet gate: table cluster vector == ClusterSpace.get_cluster_vector (max |d| {err:.1e})")
    # row table -> integer scatter matrix S (R, N*K): S[r, site*K + col] = how often `site` is a member of row r.
    # h0[site, col] = w_col[col] * sum_r prod_r S[r, (site, col)],  w_col = nA / (order * norm)  (order and norm are per column)
    idx, mask, col, order, norm = T['idx'], T['mask'], T['col'], T['order'], T['norm']
    Rn, mo = idx.shape
    rr, jj = np.where(mask)
    S = np.zeros((Rn, N * K), np.int16); np.add.at(S, (rr, idx[rr, jj] * K + col[rr]), 1)
    assert S.max() <= 8
    ord_col = np.zeros(K); ord_col[col] = order; w_col = N / (ord_col * norm)
    mc = setup(mdl, atoms, 0, dtype=None, table="legacy")         # MC path, float64 (gate reference)
    L1 = P['layers'][0]
    parts = dict(N=N, K=K, C=C, idx=jnp.asarray(idx), mask=jnp.asarray(mask), S=jnp.asarray(S, jnp.bfloat16), w_col=jnp.asarray(w_col, jnp.float64),
                 A=jnp.asarray(A, jnp.float64),
                 mu=jnp.asarray(mu, jnp.float64), sd=jnp.asarray(sd, jnp.float64),
                 lin=jnp.asarray(P['lin'], jnp.float64), Wself=jnp.asarray(L1['self'], jnp.float64), Wsh=jnp.asarray(L1['shell'], jnp.float64),
                 b=jnp.asarray(L1['b'], jnp.float64), G=jnp.asarray(L1['gate'], jnp.float64), gb=jnp.asarray(L1['gb'], jnp.float64),
                 hw=jnp.asarray(P['head']['w'], jnp.float64), hb=float(P['head']['b']), ym=float(mdl['ym']), ys=float(mdl['ys']),
                 nnz=int(len(rr)), n_rows=int(Rn), mo=int(mo), depth=len(P['layers']), gate=('gate' in L1), has_max=('shell_max' in L1))
    assert parts['depth'] == 1 and not parts['has_max'], "this kernel implements depth-1 mean-aggregation (+gate) only"
    return mdl, T, mc, parts


def make_energy_fn(parts):
    """sig (B, N) int8 in {0,1} (1 = Pt) -> E (B,) eV/atom, float64 forward."""
    import jax, jax.numpy as jnp
    N, K, C = parts['N'], parts['K'], parts['C']
    idx, mask, S, w_col = parts['idx'], parts['mask'], parts['S'], parts['w_col']

    def energy(sig):
        s = (2 * sig.astype(jnp.int8) - 1)                        # theta: Ni(0) -> -1, Pt(1) -> +1
        g = jnp.where(mask[None], s[:, idx], jnp.int8(1))         # (B, R, mo)
        pr = jnp.prod(g, axis=-1).astype(jnp.bfloat16)            # (B, R) in {-1, +1}: exact in bf16
        # integer counts via a tensor-core matmul: bf16 x bf16 -> f32 accumulation is exact for integers < 2^24
        h_int = jnp.matmul(pr, S, preferred_element_type=jnp.float32)          # (B, N*K)
        h = h_int.astype(jnp.float64).reshape(-1, N, K) * w_col
        h0 = jnp.concatenate([jnp.ones((sig.shape[0], N, 1), jnp.float64), h], axis=-1)      # zerolet
        X = (h0 - parts['mu']) / parts['sd']                       # (B, N, C)
        e_lin = (X @ parts['lin']).mean(-1)
        Z = jnp.einsum('pij,bjc->bpic', parts['A'], X)             # shell means (B, P, N, C)
        pre = X @ parts['Wself'] + jnp.einsum('bpic,pcd->bid', Z, parts['Wsh']) + parts['b']
        out = jnp.tanh(pre)
        if parts['gate']:
            out = out * jax.nn.sigmoid(jnp.einsum('bpic,pcd->bid', Z, parts['G']) + parts['gb'])
        e = e_lin + (out @ parts['hw']).mean(-1) + parts['hb']
        return e * parts['ys'] + parts['ym']
    return jax.jit(energy)


# ---------------------------------------------------------------------------------------------------------------- translations
def translation_perms(atoms):
    """all lattice translations of the fcc primitive lattice modulo the supercell, as site permutations (N of them)."""
    prim = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * A_FCC
    cell = atoms.cell.array; inv = np.linalg.inv(cell); frac = atoms.get_scaled_positions(wrap=True)
    N = len(atoms); perms = {}
    rng = range(-6, 7)
    for n1 in rng:
        for n2 in rng:
            for n3 in rng:
                v = n1 * prim[0] + n2 * prim[1] + n3 * prim[2]
                f = (frac + v @ inv) % 1.0
                d = f[:, None, :] - frac[None, :, :]; d -= np.round(d)
                j = np.argmin(np.abs(d).sum(-1), axis=1)
                assert np.abs(d[np.arange(N), j]).max() < 1e-6, "translation does not map the lattice onto itself"
                key = tuple(j.tolist())
                perms[key] = j
                if len(perms) == N: break
    perms = np.array(list(perms.values()), np.int32)
    assert len(perms) == N, (len(perms), N)
    return perms


def _compose(p, q):                                                # (p o q)(i) = p[q[i]]
    return p[q]


def _order(p):
    N = len(p); e = np.arange(N); q = p.copy(); k = 1
    while not np.array_equal(q, e):
        q = _compose(p, q); k += 1
    return k


def row_col_layout(perms):
    """Layout of the (abelian) translation group as rows x columns: t = an element of maximal order m, columns = powers of t,
    rows = the cosets of <t> (any abelian quotient, cyclic or not).  Row r, column c <-> site (u_r t^c)(0), u_r = coset representative.
    Every translation g maps row r to row rowmap[g, r] with the column shift colshift[g, r]:  g u_r = u_{r'} t^{c'}.
    returns site_of (nr, m), m, nr, rowmap (G, nr), colshift (G, nr)."""
    N = perms.shape[0]; e = np.arange(N)
    orders = np.array([_order(p) for p in perms]); m = int(orders.max()); t = perms[int(np.argmax(orders))]
    key = lambda p: tuple(p.tolist())
    pow_t = [e]
    for _ in range(m - 1): pow_t.append(_compose(t, pow_t[-1]))
    coset_of = {}; reps = []                                       # key(g) -> (row, column shift)
    for g in perms:
        if key(g) in coset_of: continue
        r = len(reps); reps.append(g)
        for c in range(m): coset_of[key(_compose(g, pow_t[c]))] = (r, c)
    nr = len(reps); assert nr * m == N, (nr, m, N)
    site_of = np.zeros((nr, m), np.int32)
    for r in range(nr):
        for c in range(m): site_of[r, c] = _compose(reps[r], pow_t[c])[0]
    assert len(set(site_of.ravel().tolist())) == N, "layout does not cover every site once"
    rowmap = np.zeros((N, nr), np.int32); colshift = np.zeros((N, nr), np.int32)
    for gi, g in enumerate(perms):
        for r in range(nr):
            rowmap[gi, r], colshift[gi, r] = coset_of[key(_compose(g, reps[r]))]
    return site_of, m, nr, rowmap, colshift


def make_canon_fns(m, nr, rowmap, colshift):
    """bit layout: word = sum_{r,c} bit(r,c) << ((nr-1-r)*m + c): row 0 in the TOP m bits, column 0 the lowest bit of a row.
    translate_g(x): row r (rotated left by colshift[g, r]: column k -> k + shift) goes to row rowmap[g, r]."""
    import jax, jax.numpy as jnp
    nb = m * nr; MASK = (1 << nb) - 1; full = (1 << m) - 1
    U = lambda v: jnp.uint64(v)

    def rot_row(w, c):                                             # w: one m-bit row in the low bits
        if c == 0: return w
        return ((w << U(c)) | (w >> U(m - c))) & U(full)

    def translate(x, g):
        y = jnp.zeros_like(x)
        for r in range(nr):
            w = (x >> U((nr - 1 - r) * m)) & U(full)
            y = y | (rot_row(w, int(colshift[g, r])) << U((nr - 1 - int(rowmap[g, r])) * m))
        return y

    def canon_min_and_stab(x):                                     # x: (B,) uint64 -> (min translate, # translates equal to x)
        mn = x; eq = jnp.zeros(x.shape, jnp.int32)
        for g in range(rowmap.shape[0]):
            y = translate(x, g); mn = jnp.minimum(mn, y); eq = eq + (y == x).astype(jnp.int32)
        return mn, eq

    def minrot_row(w):                                             # w: (B,) uint64 one row (m bits) -> min over column rotations
        mn = w
        for c in range(1, m): mn = jnp.minimum(mn, rot_row(w, c))
        return mn
    return dict(translate=translate, canon=canon_min_and_stab, minrot_row=minrot_row, nb=nb, MASK=MASK)


def necklaces(m):
    """12-bit words minimal under rotation, by weight."""
    out = []
    for w in range(1 << m):
        mn = w
        for c in range(1, m):
            y = ((w << c) | (w >> (m - c))) & ((1 << m) - 1)
            if y < mn: mn = y
        if mn == w: out.append(w)
    return np.array(out, np.int64)


def binom_table(n_max, k_max):
    Cb = np.zeros((n_max + 1, k_max + 1), np.int64)
    for n in range(n_max + 1):
        Cb[n, 0] = 1
        for k in range(1, min(n, k_max) + 1): Cb[n, k] = Cb[n - 1, k - 1] + Cb[n - 1, k]
    return Cb


def make_unrank(nbits, k, Cb):
    """index in [0, C(nbits, k)) -> nbits-bit word with k ones (combinatorial number system, lexicographic)."""
    import jax, jax.numpy as jnp
    Cj = jnp.asarray(Cb)

    def unrank(idx):                                               # idx (B,) int64 -> word (B,) uint64
        word = jnp.zeros(idx.shape, jnp.uint64); kk = jnp.full(idx.shape, k, jnp.int32); ii = idx
        for pos in range(nbits - 1, -1, -1):                       # decide the bit at position pos (highest first)
            c = Cj[pos, jnp.clip(kk, 0, k)]                        # C(pos, kk): number of words with bit pos = 0
            take = (kk > 0) & (ii >= c)
            ii = jnp.where(take, ii - c, ii); kk = jnp.where(take, kk - 1, kk)
            word = word | (take.astype(jnp.uint64) << jnp.uint64(pos))
        return word
    return jax.jit(unrank)


# ---------------------------------------------------------------------------------------------------------------- accumulation
def temp_grid():
    T = np.concatenate([np.arange(100.0, 700.0, 5.0), np.arange(700.0, 1000.0, 1.0), np.arange(1000.0, 3000.0, 5.0),
                        [3000, 3500, 4000, 5000, 7000, 1e4, 2e4, 5e4, 1e5, 1e6, 1e8]])
    return T


class Accum:
    def __init__(self, N, T, e_bins):
        import jax.numpy as jnp
        self.N = N; self.T = T; self.beta = jnp.asarray(1.0 / (KB * T), jnp.float64)
        self.E0 = (E_REF_ATOM - 0.02) * N                          # shift (eV, total): no overflow unless a state is 0.02 eV/atom below L1_0
        self.lnZ_sum = np.zeros(len(T)); self.E_sum = np.zeros(len(T)); self.E2_sum = np.zeros(len(T))
        self.e_bins = e_bins; self.dos = np.zeros(len(e_bins) - 1, np.float64)
        self.w_total = 0; self.n_rep = 0; self.E_min = np.inf; self.E_max = -np.inf; self.x_min = 0; self.x_max = 0
        self._fn = self._make()

    def _make(self):
        import jax, jax.numpy as jnp
        beta = self.beta; E0 = self.E0; edges = jnp.asarray(self.e_bins)

        def f(E_atom, w):                                          # E (B,) eV/atom, w (B,) int32 weights
            E = E_atom * self.N; d = E - E0; wf = w.astype(jnp.float64)
            ex = jnp.exp(-beta[:, None] * d[None, :]) * wf[None, :]          # (nT, B)
            s0 = ex.sum(1); s1 = (ex * d[None, :]).sum(1); s2 = (ex * d[None, :] ** 2).sum(1)
            k = jnp.clip(jnp.searchsorted(edges, E_atom, side='right') - 1, 0, len(edges) - 2)
            dos = jnp.zeros(len(edges) - 1, jnp.float64).at[k].add(wf)
            return s0, s1, s2, dos, jnp.argmin(E_atom), jnp.argmax(E_atom)
        return jax.jit(f)

    def add(self, E_atom, w, x):
        s0, s1, s2, dos, imin, imax = self._fn(E_atom, w)
        self.lnZ_sum += np.asarray(s0); self.E_sum += np.asarray(s1); self.E2_sum += np.asarray(s2); self.dos += np.asarray(dos)
        w_np = np.asarray(w); self.w_total += int(w_np.sum()); self.n_rep += int(np.count_nonzero(w_np))     # weight 0 = padding
        E_np = np.asarray(E_atom); imin = int(imin); imax = int(imax)
        if E_np[imin] < self.E_min: self.E_min = float(E_np[imin]); self.x_min = int(np.asarray(x)[imin])
        if E_np[imax] > self.E_max: self.E_max = float(E_np[imax]); self.x_max = int(np.asarray(x)[imax])

    def state(self):
        return dict(S0=self.lnZ_sum, S1=self.E_sum, S2=self.E2_sum, dos=self.dos, w_total=np.int64(self.w_total), n_rep=np.int64(self.n_rep),
                    E_min=self.E_min, E_max=self.E_max, x_min=np.uint64(self.x_min), x_max=np.uint64(self.x_max))

    def load_state(self, z):
        self.lnZ_sum = np.array(z['S0']); self.E_sum = np.array(z['S1']); self.E2_sum = np.array(z['S2']); self.dos = np.array(z['dos'])
        self.w_total = int(z['w_total']); self.n_rep = int(z['n_rep']); self.E_min = float(z['E_min']); self.E_max = float(z['E_max'])
        self.x_min = int(z['x_min']); self.x_max = int(z['x_max'])

    def save(self, path, extra):
        np.savez(path, T=self.T, E0=self.E0, S0=self.lnZ_sum, S1=self.E_sum, S2=self.E2_sum, dos=self.dos, e_bins=self.e_bins,
                 w_total=np.int64(self.w_total), n_rep=np.int64(self.n_rep), E_min=self.E_min, E_max=self.E_max,
                 x_min=np.uint64(self.x_min), x_max=np.uint64(self.x_max), **extra)


def summarize(S0, S1, S2, T, E0, N, w_total, nPt):
    """exact <E>(T) eV/atom, C_V (k_B/atom), F (eV/atom) relative to the random-mixing reference (beta F = -ln Z)."""
    beta = 1.0 / (KB * T); m1 = S1 / S0; m2 = S2 / S0
    E_mean = (E0 + m1) / N; var = m2 - m1 ** 2; cv = var * beta ** 2 / N
    lnZ = np.log(S0) - beta * E0; F = -lnZ / beta / N
    lnW = math.lgamma(N + 1) - math.lgamma(nPt + 1) - math.lgamma(N - nPt + 1)
    S = (lnZ + beta * (E0 + m1)) / N                               # entropy per atom (k_B)
    return dict(T=T.tolist(), E=E_mean.tolist(), Cv=cv.tolist(), F=F.tolist(), lnZ=lnZ.tolist(), S=S.tolist(), S_random=lnW / N)


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=3, type=int, default=[2, 2, 3])
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--gate", action="store_true", help="brute force (every configuration) vs the enumeration; small cells only")
    ap.add_argument("--merge", default=None)
    ap.add_argument("--B1", type=int, default=1 << 21, help="candidates per stage-1 batch")
    ap.add_argument("--B2", type=int, default=1 << 14, help="configurations per energy batch")
    ap.add_argument("--max_necklaces", type=int, default=0, help="debug: only the first k necklaces of this worker")
    ap.add_argument("--n_gate", type=int, default=300)
    args = ap.parse_args()
    if args.merge: return merge(args.merge)
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    atoms = build_cell(args.cell); N = len(atoms); nPt = N // 2
    mdl, T, mc, parts = model_parts(atoms)
    energy = make_energy_fn(parts)
    log(f"cell {args.cell}: N={N}, table rows {parts['n_rows']} (max order {parts['mo']}), nnz {parts['nnz']}, channels {parts['C']}, "
        f"device {jax.devices()[0].platform}")
    # ---- gate A: batched kernel == MC path (float64) on random configurations + the ordered reference
    rng = np.random.default_rng(0)
    sigs = np.zeros((args.n_gate, N), np.int8)
    for b in range(args.n_gate): sigs[b, rng.choice(N, nPt, replace=False)] = 1
    sig_ref = np.asarray(mc['sig0']); sigs[0] = sig_ref                                   # as read: all Ni -> set L1_0 below
    E_k = np.asarray(energy(jnp.asarray(sigs))); E_mc = np.array([mc['energy'](jnp.asarray(s)) for s in sigs])
    dmax = float(np.abs(E_k - E_mc).max())
    log(f"gate A: batched kernel vs MC energy on {args.n_gate} random configurations: max |dE| = {dmax:.2e} eV/atom")
    assert dmax < 1e-9, "energy kernel differs from the MC path"
    # gate A2: the table the production MC actually uses (default = implicit, sites in canonical order: sig_canonical = sig_input[perm])
    from neuralce.nce.mc_model import setup as _setup
    mci = _setup(mdl, atoms, 0, dtype=None)
    if mci.get('implicit'):
        pm = np.asarray(mci['perm']); E_imp = np.array([mci['energy'](jnp.asarray(s_[pm].astype(np.int32))) for s_ in sigs[:100]])
        d2 = float(np.abs(E_imp - E_k[:100]).max())
        log(f"gate A2: batched kernel vs the MC engine's implicit-table energy (production path) on 100 configurations: max |dE| = {d2:.2e} eV/atom")
        assert d2 < 1e-9, "enumeration energy differs from the production MC energy in this cell"
    # L1_0 in this cell: alternate (001) planes -> z fractional coordinate parity
    z = atoms.get_scaled_positions(wrap=True)[:, 2] * args.cell[2] * 2; layer = np.round(z).astype(int) % 2
    sig_L10 = (layer == 1).astype(np.int8); assert sig_L10.sum() == nPt
    E_L10 = float(energy(jnp.asarray(sig_L10[None]))[0])
    log(f"L1_0 in this cell: E = {E_L10:.6f} eV/atom (guide: {E_REF_ATOM})")
    # ---- translations and layout
    perms = translation_perms(atoms); site_of, m, nr, rowmap, colshift = row_col_layout(perms)
    log(f"translations: {len(perms)} = {m} columns (powers of t) x {nr} rows (cosets); row 0 = sites {site_of[0].tolist()}")
    cf = make_canon_fns(m, nr, rowmap, colshift); nb = cf['nb']
    bitpos = np.zeros(N, np.int64)                                 # site -> bit position
    for r in range(nr):
        for c in range(m): bitpos[site_of[r, c]] = (nr - 1 - r) * m + c
    site_at_bit = np.argsort(bitpos)                               # bit p -> site
    bp_j = jnp.asarray(bitpos); sab = jnp.asarray(site_at_bit)

    @jax.jit
    def word_to_sig(x):                                            # (B,) uint64 -> (B, N) int8, sig[site] = bit bitpos[site]
        return ((x[:, None] >> bp_j[None, :].astype(jnp.uint64)) & jnp.uint64(1)).astype(jnp.int8)

    @jax.jit
    def sig_to_word(sig):
        return (sig.astype(jnp.uint64) << bp_j[None, :].astype(jnp.uint64)).sum(1)

    # gate B: rotation formulas == site permutations (random words), and translation invariance of the energy
    xs = jnp.asarray(sig_to_word(jnp.asarray(sigs)))
    mn_fast, stab = cf['canon'](xs)
    sig_np = sigs; mins = []
    for b in range(len(sig_np)):
        best = None
        for p in perms:                                            # translate: site i -> p[i]
            s2 = np.zeros(N, np.int8); s2[p] = sig_np[b]; w = int((s2.astype(np.int64) << bitpos).sum())
            best = w if best is None or w < best else best
        mins.append(best)
    assert np.array_equal(np.asarray(mn_fast).astype(np.int64), np.array(mins)), "rotation canonical form != permutation canonical form"
    E_tr = np.asarray(energy(word_to_sig(mn_fast)))
    assert np.abs(E_tr - E_k).max() < 1e-9, "energy not translation invariant?!"
    log(f"gate B: canonical form by bit rotations == by site permutations on {len(sigs)} words; E(translate) == E: PASS")
    # energy throughput (random words, B2 per batch)
    xr = jnp.asarray(rng.integers(0, 1 << 62, args.B2, dtype=np.int64).astype(np.uint64) & jnp.uint64((1 << nb) - 1))
    energy(word_to_sig(xr)).block_until_ready(); t1 = time.time(); nrep = 20
    for _ in range(nrep): E_b = energy(word_to_sig(xr))
    E_b.block_until_ready(); log(f"energy throughput: {nrep * args.B2 / (time.time() - t1):.3e} configurations/s (B2 = {args.B2})")
    # ---- enumeration
    Tg = temp_grid(); e_bins = np.arange(-6.05, -5.85, 1e-6)
    acc = Accum(N, Tg, e_bins)
    Cb = binom_table(nb, nPt + 1); nfree = nb - m
    neck = necklaces(m); wts = np.array([bin(int(w)).count("1") for w in neck])
    mine = [int(i) for i in range(len(neck)) if i % args.n_workers == args.worker]
    if args.max_necklaces: mine = mine[:args.max_necklaces]
    total_cand = sum(int(Cb[nfree, nPt - wts[i]]) for i in mine if 0 <= nPt - wts[i] <= nfree)
    log(f"worker {args.worker}/{args.n_workers}: {len(mine)} of {len(neck)} necklaces, {total_cand:.3e} candidates")
    unranks = {}; stage1 = {}
    full_row = (1 << m) - 1

    def stage1_fn(k):                                              # one compiled function per weight k (n0 is a traced argument)
        if k in stage1: return stage1[k]
        unrank = unranks.setdefault(k, make_unrank(nfree, k, Cb))

        @jax.jit
        def f(n0, idx):
            low = unrank(idx)                                      # (B,) nfree-bit words
            x = (jnp.uint64(n0) << jnp.uint64(nfree)) | low
            # necessary condition: row 0 (= n0, a necklace) <= min rotation of every other row
            ok = jnp.ones(idx.shape, bool)
            for r in range(1, nr):
                row = (x >> jnp.uint64((nr - 1 - r) * m)) & jnp.uint64(full_row)
                ok = ok & (cf['minrot_row'](row) >= jnp.uint64(n0))
            mn, stab = cf['canon'](x)
            keep = ok & (mn == x)
            return x, keep, (N // stab).astype(jnp.int32)
        stage1[k] = f
        return f

    buf_x, buf_w = [], []; nbuf = 0; t0 = time.time(); n_cand_done = 0; n_batches = 0

    def flush(final=False):
        nonlocal buf_x, buf_w, nbuf
        if nbuf == 0: return
        X = np.concatenate(buf_x); W = np.concatenate(buf_w); buf_x, buf_w, nbuf = [], [], 0
        if final and len(X) % args.B2:                             # pad the tail with weight-0 copies: one batch shape, no recompilation
            pad = args.B2 - len(X) % args.B2
            X = np.concatenate([X, np.full(pad, X[0], X.dtype)]); W = np.concatenate([W, np.zeros(pad, W.dtype)])
        for s in range(0, len(X), args.B2):
            xb = jnp.asarray(X[s:s + args.B2]); wb = jnp.asarray(W[s:s + args.B2])
            if len(X) - s < args.B2 and not final:                 # keep a partial tail for the next flush
                buf_x.append(X[s:]); buf_w.append(W[s:]); nbuf = len(X) - s; break
            acc.add(energy(word_to_sig(xb)), wb, xb)

    out = args.out or f"{RUNS}/ptni_exact/enum/N{N}"; os.makedirs(out, exist_ok=True)
    part = f"{out}/part_{args.worker}_of_{args.n_workers}.npz"; prog = f"{out}/prog_{args.worker}_of_{args.n_workers}.npz"
    if os.path.exists(part) and not args.gate and not args.max_necklaces:
        log(f"already finished: {part}"); return
    pos0, s0, t_prev, cand_prev = 0, 0, 0.0, 0                     # resume point: necklace position in `mine`, candidate offset inside it
    if os.path.exists(prog):
        zp = np.load(prog); assert np.array_equal(zp['mine'], np.array(mine)) and int(zp['B1']) == args.B1, "progress file belongs to other settings"
        acc.load_state(zp); pos0, s0, t_prev, cand_prev = int(zp['pos']), int(zp['s']), float(zp['seconds']), int(zp['n_cand_done'])
        log(f"resumed from {prog}: necklace position {pos0}/{len(mine)}, offset {s0:.3e}, {acc.n_rep:.3e} representatives so far, {t_prev/60:.1f} min")
    t_ck = time.time(); CK_SEC = float(os.environ.get("CK_SEC", "600")); STOP_AFTER = float(os.environ.get("STOP_AFTER_SEC", "0"))

    def checkpoint(pos, s_next):                                   # accumulators reflect exactly the candidates before (pos, s_next)
        flush(final=True)
        np.savez(prog + ".tmp.npz", mine=np.array(mine), B1=args.B1, pos=pos, s=s_next, seconds=t_prev + time.time() - t0, n_cand_done=n_cand_done, **acc.state())
        os.replace(prog + ".tmp.npz", prog)

    n_cand_done = cand_prev
    for pos, ni in enumerate(mine):
        if pos < pos0: continue
        n0 = int(neck[ni]); k = nPt - int(wts[ni])
        if not (0 <= k <= nfree): continue
        n_idx = int(Cb[nfree, k]); f1 = stage1_fn(k)
        for s in range(s0 if pos == pos0 else 0, n_idx, args.B1):
            if time.time() - t_ck > CK_SEC:
                checkpoint(pos, s); t_ck = time.time()
                if STOP_AFTER and time.time() - t0 > STOP_AFTER:    # test hook: stop right after a checkpoint (resume test)
                    log(f"STOP_AFTER_SEC reached after a checkpoint at necklace position {pos}, offset {s:.3e}: exiting"); sys.exit(3)
            idx = jnp.arange(s, min(s + args.B1, n_idx), dtype=jnp.int64)
            x, keep, w = f1(jnp.int64(n0), idx)
            keep_np = np.asarray(keep)
            if keep_np.any():
                buf_x.append(np.asarray(x)[keep_np]); buf_w.append(np.asarray(w)[keep_np]); nbuf += int(keep_np.sum())
            n_cand_done += len(keep_np); n_batches += 1
            if nbuf >= 8 * args.B2: flush()
            if n_batches % 200 == 0:
                dt = time.time() - t0
                log(f"  necklace {pos+1}/{len(mine)} #{ni} ({n0:0{m}b}) k={k}: {n_cand_done:.3e}/{total_cand:.3e} candidates, {acc.n_rep:.3e} representatives "
                    f"(sum w {acc.w_total:.4e}), {(t_prev+dt)/60:.1f} min, {(n_cand_done-cand_prev)/dt:.2e} cand/s, E_min {acc.E_min:.6f}")
    flush(final=True)
    dt = t_prev + time.time() - t0
    log(f"worker done: {acc.n_rep} representatives, sum of weights {acc.w_total}, E_min {acc.E_min:.6f}, E_max {acc.E_max:.6f}, {dt/60:.1f} min")
    acc.save(part + ".tmp.npz", dict(N=N, nPt=nPt, cell=np.array(args.cell), m=m, nr=nr, E_L10=E_L10, gate_dE=dmax, necklaces=np.array(mine), seconds=dt))
    os.replace(part + ".tmp.npz", part)
    if os.path.exists(prog): os.remove(prog)
    # ---- gate C (small cells): brute force over every configuration
    if args.gate:
        total = int(Cb[nb, nPt]); unrank_all = make_unrank(nb, nPt, Cb); acc2 = Accum(N, Tg, e_bins); t1 = time.time()
        for s in range(0, total, args.B2 * 4):
            idx = jnp.arange(s, min(s + args.B2 * 4, total), dtype=jnp.int64); x = unrank_all(idx)
            for q in range(0, len(x), args.B2):
                xb = x[q:q + args.B2]; acc2.add(energy(word_to_sig(xb)), jnp.ones(len(xb), jnp.int32), xb)
        log(f"gate C: brute force {acc2.n_rep} configurations in {(time.time()-t1)/60:.1f} min")
        assert acc2.w_total == total == acc.w_total, (acc2.w_total, total, acc.w_total)
        for a, b_, name in ((acc.lnZ_sum, acc2.lnZ_sum, "S0"), (acc.E_sum, acc2.E_sum, "S1"), (acc.E2_sum, acc2.E2_sum, "S2")):
            rel = np.abs(a - b_) / np.maximum(np.abs(b_), 1e-300); log(f"  {name}: max relative difference {rel.max():.2e}")
            assert rel.max() < 1e-10
        assert np.array_equal(acc.dos, acc2.dos), "DOS differs"
        assert abs(acc.E_min - acc2.E_min) < 1e-12 and abs(acc.E_max - acc2.E_max) < 1e-12
        log("gate C: enumeration == brute force: PASS")
        acc2.save(f"{out}/bruteforce.npz", dict(N=N, nPt=nPt))
    res = summarize(acc.lnZ_sum, acc.E_sum, acc.E2_sum, Tg, acc.E0, N, acc.w_total, nPt)
    res.update(N=N, nPt=nPt, n_rep=acc.n_rep, w_total=acc.w_total, E_min=acc.E_min, E_max=acc.E_max, E_L10=E_L10, worker=args.worker, n_workers=args.n_workers)
    json.dump(res, open(f"{out}/part_{args.worker}_of_{args.n_workers}.json", "w"))
    log(f"wrote {out}")


def merge(d):
    import glob
    fs = sorted(glob.glob(f"{d}/part_*_of_*.npz")); assert fs, "no parts"
    z0 = np.load(fs[0]); nW = int(fs[0].split("_of_")[1].split(".")[0]); N = int(z0['N']); nPt = int(z0['nPt'])
    assert len(fs) == nW, f"{len(fs)} of {nW} parts present"
    S0 = sum(np.load(f)['S0'] for f in fs); S1 = sum(np.load(f)['S1'] for f in fs); S2 = sum(np.load(f)['S2'] for f in fs)
    dos = sum(np.load(f)['dos'] for f in fs); w_total = int(sum(int(np.load(f)['w_total']) for f in fs)); n_rep = int(sum(int(np.load(f)['n_rep']) for f in fs))
    E_min = min(float(np.load(f)['E_min']) for f in fs); E_max = max(float(np.load(f)['E_max']) for f in fs)
    total = math.comb(N, nPt)
    res = summarize(S0, S1, S2, z0['T'], float(z0['E0']), N, w_total, nPt)
    res.update(N=N, nPt=nPt, n_rep=n_rep, w_total=w_total, C_N_k=total, complete=(w_total == total), E_min=E_min, E_max=E_max,
               E_L10=float(z0['E_L10']), n_workers=nW)
    np.savez(f"{d}/exact_N{N}.npz", T=z0['T'], E0=z0['E0'], S0=S0, S1=S1, S2=S2, dos=dos, e_bins=z0['e_bins'], w_total=np.int64(w_total), n_rep=np.int64(n_rep))
    json.dump(res, open(f"{d}/exact_N{N}.json", "w"))
    final = f"{RESULTS}/ptni_exact"; os.makedirs(final, exist_ok=True)                      # small merged result, tracked next to the PT analysis
    for ext in ("npz", "json"): shutil.copy2(f"{d}/exact_N{N}.{ext}", f"{final}/exact_N{N}.{ext}")
    log(f"copied exact_N{N}.npz/json to {final}")
    log(f"merged {nW} parts: sum of weights {w_total} vs C({N},{nPt}) = {total}: {'COMPLETE' if w_total == total else 'INCOMPLETE'}; "
        f"E_min {E_min:.6f} (L1_0 {res['E_L10']:.6f}), E_max {E_max:.6f}")
    T = np.asarray(res['T']); cv = np.asarray(res['Cv']); k = int(np.argmax(cv))
    log(f"C_V peak {T[k]:.0f} K, {cv[k]:.3f} k_B/atom; <E>(300 K) {res['E'][int(np.argmin(np.abs(T-300)))]:.6f}, <E>(1e8 K) {res['E'][-1]:.6f}")
    return res


if __name__ == "__main__":
    main()
