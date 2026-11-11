"""Stacked cluster convolution.

Layer 0 is the fixed CE layer, but kept SITE-RESOLVED: h0[site, a] is the
cluster-match feature of orbit `a` anchored on that site.  Averaging h0 over
sites reproduces the icet cluster vector exactly (gate below), so the linear
readout on layer 0 is still the cluster expansion.

Deeper layers are ordinary additive sparse convolutions that reuse the SAME
offset bank: for every pair orbit p (i.e. every neighbour shell) the layer
averages the previous features over that shell, concatenates the shells, and
mixes them per site.  Weights are shared across the symmetry-equivalent copies
inside an orbit, which is what keeps the space-group equivariance.

    z[s, p, :] = mean_{s' in shell p of s}  h[s', :]
    h'[s, :]   = act( W [ h[s,:] , z[s,0,:] , ... , z[s,P-1,:] ] + b )

Receptive field grows by one shell radius per layer, so depth buys range and
implicit high-order terms with a parameter count that is linear in depth.
"""
from __future__ import annotations
import numpy as np


def build_site_index(bank, m=3):
    """Site-resolved cluster indices.

    convert_bank() expanded every cluster into its 4 coset copies (one per
    sublattice anchor); here they are kept apart so features stay per site.
    -> list of (n_sites, n_copies_per_anchor, order) int32, n_sites = m^3 * 4
    """
    n_sub = bank['n_sub']
    cells = np.array([(i, j, l) for i in range(m) for j in range(m) for l in range(m)])

    def flat(cell, sub):
        c = np.asarray(cell) % m
        return ((c[..., 0] * m + c[..., 1]) * m + c[..., 2]) * n_sub + sub

    out = []
    for k in bank['kernels']:
        cl = k['clusters']
        n_cop = len(cl) // n_sub                       # copies per anchor sublattice
        order = len(cl[0])
        idx = np.empty((len(cells), n_sub, n_cop, order), dtype=np.int32)
        for a in range(n_sub):
            grp = cl[a::n_sub]                         # convert_bank emits coset-major
            off = np.array([[d for (s, d) in c] for c in grp])
            sub = np.array([[s for (s, d) in c] for c in grp])
            pos = cells[:, None, None, :] + off[None]
            idx[:, a] = flat(pos, sub[None])
        out.append(idx.reshape(len(cells) * n_sub, n_cop, order))
    return out


def make_layer0_fn(bank, m=3):
    """-> jitted f(occ (…,m,m,m,4)) = h0 (…, n_sites, K); mean over sites == icet cv."""
    import jax, jax.numpy as jnp
    per = build_site_index(bank, m=m)
    n_sites = per[0].shape[0]
    max_order = max(a.shape[2] for a in per)
    n_inst = sum(a.shape[1] for a in per)
    idx = np.zeros((n_sites, n_inst, max_order), np.int32)
    mask = np.zeros((n_inst, max_order), bool)
    kid = np.zeros(n_inst, np.int32)
    p = 0
    for k, a in enumerate(per):
        cp, o = a.shape[1], a.shape[2]
        idx[:, p:p + cp, :o] = a; mask[p:p + cp, :o] = True; kid[p:p + cp] = k; p += cp
    K = len(bank['kernels'])
    mult = jnp.asarray([kk['mult'] for kk in bank['kernels']], dtype=jnp.float64)
    idx_j, mask_j, kid_j = jnp.asarray(idx), jnp.asarray(mask), jnp.asarray(kid)

    @jax.jit
    def f(occ):
        sig = 2.0 * occ.reshape(occ.shape[:-4] + (-1,)) - 1.0
        g = jnp.take(sig, idx_j, axis=-1)
        g = jnp.where(mask_j, g, 1.0)
        pr = jnp.moveaxis(jnp.prod(g, axis=-1), -1, 0)
        h = jax.ops.segment_sum(pr, kid_j, num_segments=K)
        return jnp.moveaxis(h, 0, -1) / mult
    return f


def build_shells(bank, m=3, a=3.924, max_radius=None, return_radii=False):
    """Neighbour shells for the message-passing layers, taken from the pair orbits.

    Two things have to be repaired before an icet pair orbit can be used as a
    graph neighbourhood:

    1. SYMMETRISE.  A cluster expansion assigns each pair {i,j} to ONE endpoint
       (that is what `mult` counts), so the anchored list gives a site half of
       its neighbours -- 6 of 12 for fcc 1NN -- and that half is not closed
       under the point group.  Correct for layer 0, wrong here: it would make
       the graph directed and destroy the equivariance.  Every (anchor,partner)
       is therefore added in both directions.

    2. DROP FOLDED SHELLS.  If the pair distance exceeds half the box, +d and -d
       land on the same periodic image, so the shell stops being a point-group
       orbit (for the 3x3x3 cell L/2 = 5.886 A, which cuts the 6.204 A shell
       from 24 partners down to 12).  Layer 0 keeps the full cutoff -- that is
       the CE convention and it is gated against icet -- but message passing
       must not run on a folded neighbourhood.

    Returns (n_shells, n_sites, n_partners) int32; shells with fewer partners
    are padded by repeating their own entries, which leaves a mean unchanged.
    With `return_radii`, also returns the pair distance of each kept shell.
    """
    if max_radius is None:
        max_radius = 0.5 * m * a - 1e-9

    per = build_site_index(bank, m=m)
    n_sites = per[0].shape[0]
    shells, radii = [], []
    for k, idx in zip(bank['kernels'], per):
        d = 2.0 * k['radius']                       # icet radius is the circumradius
        if k['order'] != 2 or d > max_radius:
            continue
        s_col = np.repeat(np.arange(n_sites)[:, None], idx.shape[1], axis=1)
        anchor, other = idx[:, :, 0], idx[:, :, 1]
        partner = np.where(anchor == s_col, other, anchor)
        nb = [set() for _ in range(n_sites)]
        for s in range(n_sites):
            for t in partner[s]:
                nb[s].add(int(t)); nb[int(t)].add(s)
        sizes = {len(x) for x in nb}
        assert len(sizes) == 1, f"shell at {d:.3f} A is not uniform: {sorted(sizes)}"
        shells.append(np.array([sorted(x) for x in nb], dtype=np.int32))
        radii.append(round(d, 3))

    if not shells:
        raise ValueError(f"no pair shell within {max_radius:.3f} A")
    w = max(x.shape[1] for x in shells)
    for x, d in zip(shells, radii):
        assert w % x.shape[1] == 0, (
            f"shell at {d} A has {x.shape[1]} partners, which does not divide the "
            f"pad width {w}; repetition would bias its mean -- use a mask instead")
    out = np.stack([np.pad(x, ((0, 0), (0, w - x.shape[1])), mode='wrap')
                    for x in shells])
    return (out, radii) if return_radii else out


# ── the network ────────────────────────────────────────────────────────────
def init_params(key, K, n_shells, width, depth):
    """Weights for `depth` cluster-convolution layers plus a per-site readout.

    Per layer:  W_self (C, C'),  W_shell (n_shells, C, C'),  b (C',)
    One weight matrix per shell -- every partner inside a shell shares it, and
    every site uses the same matrices.  That is the whole of the weight sharing.
    """
    import jax, jax.numpy as jnp
    dims = [K] + [width] * depth
    layers = []
    for l in range(depth):
        key, *ks = jax.random.split(key, n_shells + 2)
        sc = np.sqrt(1.0 / (dims[l] * (n_shells + 1)))
        layers.append(dict(
            self=jax.random.normal(ks[0], (dims[l], dims[l + 1])) * sc,
            shell=jnp.stack([jax.random.normal(ks[1 + p], (dims[l], dims[l + 1])) * sc
                             for p in range(n_shells)]),
            b=jnp.zeros(dims[l + 1])))
    key, k = jax.random.split(key)
    head = dict(w=jax.random.normal(k, (dims[-1],)) * 0.1, b=jnp.zeros(()))
    return dict(layers=layers, head=head)


def make_forward(shells):
    """E(h0) for the stacked cluster convolution.

        z[s,p,:] = mean over shell p of h[partner,:]
        h'[s,:]  = tanh( h[s,:] W_self + sum_p z[s,p,:] W_p + b )
        E        = mean over sites of ( h_L[s,:] . w_out ) + b_out

    depth 0 reduces to a linear readout of the cluster features, i.e. the
    cluster expansion itself -- the natural control for "does depth help".
    """
    import jax, jax.numpy as jnp
    S = jnp.asarray(shells)

    def layer(L, h):
        nb = jnp.take(h, S, axis=-2)                       # (..., P, sites, part, C)
        z = nb.mean(axis=-2)                               # average within each shell
        msg = jnp.einsum('...psc,pcd->...sd', z, L['shell'])
        return jnp.tanh(h @ L['self'] + msg + L['b'])

    def fwd(P, h0):
        h = h0
        for L in P['layers']:
            h = layer(L, h)
        return (h @ P['head']['w']).mean(axis=-1) + P['head']['b']
    return fwd
