"""Receptive-field local update (delta energy) for the general cluster convolution.

Flipping site i changes
    layer 0 : only the rows (cluster instances) containing i  -> h0 changes on ball0(i) =
              member sites of those rows  (gpu_layer0.delta_h0)
    layer l : h^l_j changes iff j in ball_{l-1} or j has a shell partner in ball_{l-1}
              -> ball_l = ball_{l-1} U partners(ball_{l-1})
so E_new - E_old only needs the rows of i and the layers re-evaluated on the balls, with
every other activation taken from a cache.  Same idea as the validated NeuralCE RFLU
(CE_neuralce/local_update.py), written for the icet-table layer 0 and the v3 layer
(mean+max shells, gate, CE skip).

All index tables are passed as ARGUMENTS (pytree `tab`), never closed over, so nothing is
constant-folded at compile time.

Cache per structure/replica:  hs = [h0_std, h1, ..., hL]  (n_active, C_l),  e = site energies
(standardised units).  E(std) = mean(e) + b ;  E [eV/atom] = E(std) * ys + ym.
"""
from __future__ import annotations
import numpy as np


# ── geometry tables ─────────────────────────────────────────────────────────
def build_balls(T, SI, SM, depth, spin=False):
    """Per active site i: ball_l(i) for l = 0..depth (padded, ACTIVE indexing).
    SI (P, nA, W), SM (P, nA, W): shells in active indexing (general.shell_partners -> pack).
    spin=True: the model has spin channels, which change on the site AND its shell partners ->
    ball0(i) = cluster members of i  U  {i}  U  partners(i)."""
    nA = len(T['active']); inv = np.full(T['N'], -1); inv[T['active']] = np.arange(nA)
    partners = [set() for _ in range(nA)]
    for p in range(SI.shape[0]):
        for a in range(nA):
            for w in range(SI.shape[2]):
                if SM[p, a, w] > 0:
                    partners[a].add(int(SI[p, a, w])); partners[int(SI[p, a, w])].add(a)   # symmetric
    balls = []
    b0 = []
    for a in range(nA):
        i = int(T['active'][a]); rows = T['site_rows'][i][T['site_rows_mask'][i]]
        mem = set()
        for r in rows:
            for j in range(T['order'][r]):
                mem.add(int(inv[T['idx'][r, j]]))
        if spin:
            mem.add(a); mem |= partners[a]
        b0.append(sorted(mem))
    balls.append(b0)
    for l in range(depth):
        prev = balls[-1]; nxt = []
        for a in range(nA):
            s = set(prev[a])
            for j in prev[a]: s |= partners[j]
            nxt.append(sorted(s))
        balls.append(nxt)
    out = []
    for b in balls:
        w = max(len(x) for x in b); arr = np.zeros((nA, w), np.int32); m = np.zeros((nA, w), bool)
        for a, x in enumerate(b): arr[a, :len(x)] = x; m[a, :len(x)] = True
        out.append((arr, m))
    return out


def make_tab(T, SI, SM, balls, mu, sd):
    """pytree of arrays for the jitted functions (everything geometry / normalisation), including the layer-0 table
    (L0A) so that it enters every jitted function as an ARGUMENT, never as a closed-over constant."""
    import jax.numpy as jnp
    from .gpu_layer0 import layer0_arrays
    return dict(SI=jnp.asarray(SI), SM=jnp.asarray(SM, jnp.float64),
                ball_idx=[jnp.asarray(b[0]) for b in balls], ball_mask=[jnp.asarray(b[1]) for b in balls],
                mu=jnp.asarray(mu), sd=jnp.asarray(sd), L0A=layer0_arrays(T))


# ── single-structure forward pieces (no batch axis; vmap outside) ───────────
def _layer_on_sites(L, h_prev, sites, smask, SI, SM):
    """h^l on `sites` (B,) from the FULL previous-layer buffer h_prev (nA, C).  v3 layer:
    mean+max per shell, tanh, sigmoid gate on shell means, CE skip handled outside."""
    import jax, jax.numpy as jnp
    si = SI[:, sites]                                    # (P, B, W)
    sm = SM[:, sites]                                    # (P, B, W)
    g = h_prev[si]                                       # (P, B, W, C)
    m = sm[..., None]
    z = (g * m).sum(-2) / jnp.maximum(sm.sum(-1), 1.0)[..., None]          # (P, B, C)
    pre = h_prev[sites] @ L['self'] + jnp.einsum('pbc,pcd->bd', z, L['shell']) + L['b']
    if 'shell_max' in L:
        zmax = jnp.where(m > 0, g, -1e9).max(-2)
        zmax = jnp.where(sm.sum(-1)[..., None] > 0, zmax, 0.0)
        pre = pre + jnp.einsum('pbc,pcd->bd', zmax, L['shell_max'])
    out = jnp.tanh(pre)
    if 'gate' in L:
        out = out * jax.nn.sigmoid(jnp.einsum('pbc,pcd->bd', z, L['gate']) + L['gb'])
    return out * smask[:, None]


def make_local_fns(L0, depth):
    """L0 = gpu_layer0.make_jax_fns(T).  Returns jitted
        full_cache(tab, P, sig)              -> cache (hs list, e)          [full recompute]
        energy_of(cache, P)                  -> E (std units)
        local_delta(tab, P, cache, sig, flips) -> (dE (std), new_cache, new_sig)
    flips: int array (2, 2) of (site, new_species) in FULL-cell site indexing; a no-op flip
    (site with its current species) is allowed as padding."""
    import jax, jax.numpy as jnp
    h0_fn, dh0_fn = L0['h0'], L0['delta_h0']
    BIG = 2 ** 30

    def std0(h0raw, tab):
        return (h0raw - tab['mu']) / tab['sd']

    def site_energy(P, h0s, hL):
        e = h0s @ P['lin']
        if P['layers']:
            e = e + hL @ P['head']['w']
        return e

    @jax.jit
    def full_cache(tab, P, sig):
        h0s = std0(h0_fn(sig), tab); hs = [h0s]; nA = h0s.shape[0]
        allsites = jnp.arange(nA); ones = jnp.ones(nA)
        for L in P['layers']:
            hs.append(_layer_on_sites(L, hs[-1], allsites, ones, tab['SI'], tab['SM']))
        e = site_energy(P, h0s, hs[-1])
        slot = jnp.full((nA,), BIG, jnp.int32)          # persistent site->slot scratches, carried with the cache (all BIG at rest)
        if not P['layers']:
            return dict(hs=[], e=e, slot=slot, slot2=slot)   # depth 0: dE comes straight from the rows; no (N, C) cache to carry
        return dict(hs=hs, e=e, slot=slot, slot2=slot)

    @jax.jit
    def energy_of(cache, P):
        return cache['e'].mean() + P['head']['b']

    def _slot_set(slot, urows):
        """mark site -> FIRST slot in `urows` on the persistent scratch (O(|urows|) scatter, in place)."""
        return slot.at[urows].min(jnp.arange(urows.shape[0], dtype=jnp.int32))

    def _slot_clear(slot, urows):
        return slot.at[urows].set(BIG)

    def _patch_gather(gathered, gidx, smap, uvals):
        sl = smap[gidx]; found = sl < BIG
        return jnp.where(found[..., None], uvals[jnp.where(found, sl, 0)], gathered)

    def _layer_patched(L, cache_prev, sites, smask, smap, uvals, SI, SM):
        """h^l on `sites`, reading the previous layer from the OLD cache with the changed rows patched in."""
        si = SI[:, sites]; sm = SM[:, sites]                          # (P, B, W)
        g = _patch_gather(cache_prev[si], si, smap, uvals)           # (P, B, W, C)
        center = _patch_gather(cache_prev[sites], sites, smap, uvals)
        m = sm[..., None]
        z = (g * m).sum(-2) / jnp.maximum(sm.sum(-1), 1.0)[..., None]
        pre = center @ L['self'] + jnp.einsum('pbc,pcd->bd', z, L['shell']) + L['b']
        if 'shell_max' in L:
            zmax = jnp.where(m > 0, g, -1e9).max(-2)
            zmax = jnp.where(sm.sum(-1)[..., None] > 0, zmax, 0.0)
            pre = pre + jnp.einsum('pbc,pcd->bd', zmax, L['shell_max'])
        out = jnp.tanh(pre)
        if 'gate' in L:
            out = out * jax.nn.sigmoid(jnp.einsum('pbc,pcd->bd', z, L['gate']) + L['gb'])
        return out * smask[:, None]

    @jax.jit
    def local_delta(tab, P, cache, sig, flips):
        """O(ball) step: returns (dE, patch, sig_new).  patch = list of (rows, vals) per layer
        (layer 0 first) plus ('e', rows, vals); NOTHING of size N is written here -- commit()
        applies the patch on acceptance.  Padded ball slots point at the ball's first site and
        carry its value, so duplicate writes are idempotent."""
        hs = cache['hs']; nA = cache['e'].shape[0]; C0 = P['lin'].shape[0]; sig1 = sig
        # ── layer 0: changed sites = ball0(i) U ball0(j); new h0 there = old + row contributions ──
        b0 = jnp.concatenate([tab['ball_idx'][0][flips[f, 0]] for f in range(flips.shape[0])])
        m0 = jnp.concatenate([tab['ball_mask'][0][flips[f, 0]] for f in range(flips.shape[0])])
        first = jnp.argmax(m0); b0 = jnp.where(m0, b0, b0[first])
        smap0 = _slot_set(cache['slot'], b0)                               # site -> first slot in b0 (persistent scratch)
        acc = jnp.zeros((b0.shape[0], C0))
        for f in range(flips.shape[0]):
            i, s_new = flips[f, 0], flips[f, 1]
            flat, valid, value, chan = dh0_fn(sig1, i, s_new)              # flat = site*C0 + chan
            slot = smap0[flat // C0]; hit = valid & (slot < BIG)
            acc = acc.reshape(-1).at[jnp.where(hit, slot * C0 + chan, 0)].add(jnp.where(hit, value / tab['sd'][chan], 0.0)).reshape(b0.shape[0], C0)
            sig1 = sig1.at[i].set(s_new.astype(sig1.dtype))
        if not P['layers']:                                                # depth 0: dE = sum(dh0 . lin) over unique ball0 sites
            de_rows = acc[smap0[b0]] @ P['lin']
            firstocc = smap0[b0] == jnp.arange(b0.shape[0], dtype=jnp.int32)
            e_new = cache['e'][b0] + de_rows
            dE = jnp.sum(jnp.where(firstocc, de_rows, 0.0)) / nA
            return dE, dict(rows=[(b0, e_new)], slot=_slot_clear(smap0, b0), slot2=cache['slot2']), sig1
        h0_new = hs[0][b0] + acc[smap0[b0]]                                # every duplicate slot carries the full value
        patch = [(b0, h0_new)]
        smap, uvals, prev_rows, rows = smap0, h0_new, b0, b0
        smap2 = cache['slot2']                                             # second scratch for the layer maps
        # ── layers on the balls, reading old caches + patch ──
        for l, L in enumerate(P['layers']):
            bl = jnp.concatenate([tab['ball_idx'][l + 1][flips[f, 0]] for f in range(flips.shape[0])])
            ml = jnp.concatenate([tab['ball_mask'][l + 1][flips[f, 0]] for f in range(flips.shape[0])])
            fl = jnp.argmax(ml); bl = jnp.where(ml, bl, bl[fl])
            vals = _layer_patched(L, hs[l], bl, jnp.ones(bl.shape[0]), smap, uvals, tab['SI'], tab['SM'])
            patch.append((bl, vals))
            if l > 0:
                smap2 = _slot_clear(smap2, prev_rows)                      # previous layer's map lived in slot2
            smap2 = _slot_set(smap2, bl); smap = smap2
            uvals, prev_rows, rows = vals, bl, bl
        # ── site energies on the last ball (b0 map still live in slot); dE with duplicates counted once ──
        e_new = _patch_gather(hs[0][rows], rows, smap0, h0_new) @ P['lin'] + uvals @ P['head']['w']
        firstocc = smap[rows] == jnp.arange(rows.shape[0], dtype=jnp.int32)   # first occurrence of each site
        dE = jnp.sum(jnp.where(firstocc, e_new - cache['e'][rows], 0.0)) / nA
        patch.append((rows, e_new))
        return dE, dict(rows=patch, slot=_slot_clear(smap0, b0), slot2=_slot_clear(smap2, rows)), sig1

    @jax.jit
    def commit(cache, patch, accept):
        """apply the patch to the cache on ball rows only (no full-array where); carries the cleared slot scratch."""
        hs = list(cache['hs']); rows_vals = patch['rows']
        for l, (rows, vals) in enumerate(rows_vals[:-1]):                 # empty at depth 0
            hs[l] = hs[l].at[rows].set(jnp.where(accept, vals, hs[l][rows]))
        rows, vals = rows_vals[-1]
        e = cache['e'].at[rows].set(jnp.where(accept, vals, cache['e'][rows]))
        return dict(hs=hs, e=e, slot=patch['slot'], slot2=patch['slot2'])

    return dict(full_cache=full_cache, energy_of=energy_of, local_delta=local_delta, commit=commit)


def balls_full_indexing(T, balls):
    """re-index ball tables so they can be looked up with FULL-cell site ids (inactive rows empty)."""
    out = []
    for arr, m in balls:
        A = np.zeros((T['N'], arr.shape[1]), np.int32); Mk = np.zeros((T['N'], arr.shape[1]), bool)
        A[T['active']] = arr; Mk[T['active']] = m
        out.append((A, Mk))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# v2: pre-projected layer-1 caches (model-invariant).  mean_j(h_j) W_p == mean_j(h_j W_p),
# so the layer-0 features (133 ch) are never gathered per partner: per site we cache
#   U[p] = h0 W_p (D),  V[p] = h0 G_p (D, gate),  c = h0 W_self (D)
# and a partner in shell p contributes U[p] (16 floats) instead of h0 (133 floats).
# h0 itself is not cached: everything linear in h0 is patched from the row contributions.
# The max branch (if a layer has 'shell_max') still needs raw h0 -> cached only then.
# ═════════════════════════════════════════════════════════════════════════════
def make_local_fns_v2(L0, depth, dtype=None, SF=None):
    """dtype=jnp.float32 for production MC: XLA's float64 scatter on GPU is ~30x slower than float32
    (1.6 ms vs 0.05 ms for a 64x400x128 row update), and the cache commit is nothing but scatters.
    Gates run in float64 (dtype=None keeps the caller's dtype); MC re-anchors with a full recompute
    per chunk, so float32 drift (~1e-6 in standardised energy units per 1000 steps) never accumulates.
    SF (spin_features.make_spin_fn): feats(sig, spin, sites) -> spin channels appended after the Kc
    chemistry channels of layer 0; full_cache / local_delta then take the spin state (and the new spin
    values of the flipped sites) as extra arguments."""
    import jax, jax.numpy as jnp
    Kc = int(L0['K']); h0_a, dh0_a = L0['h0_a'], L0['delta_h0_a']       # table arrays come in through tab['L0A'] (jit argument)
    BIG = 2 ** 30
    cast = (lambda x: x) if dtype is None else (lambda x: jnp.asarray(x, dtype) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating) else x)
    castP = (lambda P: P) if dtype is None else (lambda P: jax.tree_util.tree_map(cast, P))

    def _slot_set(slot, urows):
        return slot.at[urows].min(jnp.arange(urows.shape[0], dtype=jnp.int32))

    def _slot_clear(slot, urows):
        return slot.at[urows].set(BIG)

    def _pg(gathered, gidx, smap, uvals):                 # patch-gather via slot map
        sl = smap[gidx]; found = sl < BIG
        return jnp.where(found[..., None], uvals[jnp.where(found, sl, 0)], gathered)

    def _proj(L, h):                                      # (..., C) -> U (..., P, D), V (..., P, D)|None, c (..., D)
        U = jnp.einsum('...c,pcd->...pd', h, L['shell']); c = h @ L['self']
        V = jnp.einsum('...c,pcd->...pd', h, L['gate']) if 'gate' in L else None
        return U, V, c

    def _layer1_from_proj(L, U, V, c, sites, SI, SM, smap=None, dU=None, dV=None, dc=None, h0raw=None, h0new=None):
        """layer with pre-projected input on `sites`; optional patch (smap, dU, dV, dc) for changed rows."""
        si = SI[:, sites]; sm = SM[:, sites]; m = sm[..., None]         # (P, B, W)
        Pn = si.shape[0]
        g = jnp.stack([U[si[p], p] for p in range(Pn)])                  # (P, B, W, D): partner's own-shell projection
        cc = c[sites]
        if smap is not None:
            g = jnp.stack([_pg(g[p], si[p], smap, dU[:, p]) for p in range(Pn)])
            cc = _pg(cc, sites, smap, dc)
        z = (g * m).sum(-2) / jnp.maximum(sm.sum(-1), 1.0)[..., None]  # (P, B, D)
        pre = cc + z.sum(0) + L['b']
        if 'shell_max' in L:                                             # raw branch (only if the model has max)
            graw = h0raw[si]
            if smap is not None: graw = _pg(graw, si, smap, h0new)
            zmax = jnp.where(m > 0, graw, -1e9).max(-2); zmax = jnp.where(sm.sum(-1)[..., None] > 0, zmax, 0.0)
            pre = pre + jnp.einsum('pbc,pcd->bd', zmax, L['shell_max'])
        out = jnp.tanh(pre)
        if 'gate' in L:
            gv = jnp.stack([V[si[p], p] for p in range(Pn)])
            if smap is not None: gv = jnp.stack([_pg(gv[p], si[p], smap, dV[:, p]) for p in range(Pn)])
            zg = (gv * m).sum(-2) / jnp.maximum(sm.sum(-1), 1.0)[..., None]
            out = out * jax.nn.sigmoid(zg.sum(0) + L['gb'])
        return out

    def _layer_raw(L, h_prev, sites, SI, SM, smap=None, uvals=None):     # layers >= 2: raw 16-ch input
        si = SI[:, sites]; sm = SM[:, sites]; m = sm[..., None]
        g = h_prev[si]; center = h_prev[sites]
        if smap is not None: g = _pg(g, si, smap, uvals); center = _pg(center, sites, smap, uvals)
        z = (g * m).sum(-2) / jnp.maximum(sm.sum(-1), 1.0)[..., None]
        pre = center @ L['self'] + jnp.einsum('pbc,pcd->bd', z, L['shell']) + L['b']
        if 'shell_max' in L:
            zmax = jnp.where(m > 0, g, -1e9).max(-2); zmax = jnp.where(sm.sum(-1)[..., None] > 0, zmax, 0.0)
            pre = pre + jnp.einsum('pbc,pcd->bd', zmax, L['shell_max'])
        out = jnp.tanh(pre)
        if 'gate' in L: out = out * jax.nn.sigmoid(jnp.einsum('pbc,pcd->bd', z, L['gate']) + L['gb'])
        return out

    @jax.jit
    def full_cache(tab, P, sig, spin=None):
        P = castP(P); tab = jax.tree_util.tree_map(cast, tab)
        h0raw = h0_a(tab['L0A'], sig)
        if SF is not None:
            h0raw = jnp.concatenate([h0raw, SF(sig, spin, jnp.arange(h0raw.shape[0]))], axis=1)
        h0s = cast((h0raw - tab['mu']) / tab['sd']); nA = h0s.shape[0]; allsites = jnp.arange(nA)
        slot = jnp.full((nA,), BIG, jnp.int32)
        e = h0s @ P['lin']
        if not P['layers']:
            return dict(e=e, slot=slot, slot2=slot)
        L1 = P['layers'][0]; U, V, c = _proj(L1, h0s)
        need_raw = 'shell_max' in L1
        h = _layer1_from_proj(L1, U, V, c, allsites, tab['SI'], tab['SM'], h0raw=h0s if need_raw else None)
        hs = [h]
        for L in P['layers'][1:]:
            hs.append(_layer_raw(L, hs[-1], allsites, tab['SI'], tab['SM']))
        cache = dict(e=e + hs[-1] @ P['head']['w'], U=U, c=c, hL=hs[-1], hmid=hs[:-1], slot=slot, slot2=slot)
        if V is not None: cache['V'] = V
        if need_raw: cache['h0'] = h0s
        return cache

    @jax.jit
    def energy_of(cache, P):
        return cache['e'].mean() + cast(P['head']['b'])

    @jax.jit
    def local_delta(tab, P, cache, sig, flips, spin=None, sflips=None):
        """flips (F, 2): (site, new species) in FULL-cell indexing; sflips (F,): new spin value of each
        flipped site (spin models only; a swap carries the spins with the atoms, a spin move repeats the
        site with its current species)."""
        P = castP(P); tab = jax.tree_util.tree_map(cast, tab)
        nA = cache['e'].shape[0]; C0 = P['lin'].shape[0]; sig1 = sig
        b0 = jnp.concatenate([tab['ball_idx'][0][flips[f, 0]] for f in range(flips.shape[0])])
        m0 = jnp.concatenate([tab['ball_mask'][0][flips[f, 0]] for f in range(flips.shape[0])])
        b0 = jnp.where(m0, b0, b0[jnp.argmax(m0)])
        smap0 = _slot_set(cache['slot'], b0)
        acc = jnp.zeros((b0.shape[0], C0), cache['e'].dtype); ov = ()
        for f in range(flips.shape[0]):
            i, s_new = flips[f, 0], flips[f, 1]
            flat, valid, value, chan = dh0_a(tab['L0A'], sig, i, s_new, ov)   # earlier flips applied as overrides, sig never copied
            slot = smap0[flat // Kc]; hit = valid & (slot < BIG)          # flat = active site * Kc + chemistry channel
            acc = acc.reshape(-1).at[jnp.where(hit, slot * C0 + chan, 0)].add(jnp.where(hit, cast(value) / tab['sd'][chan], 0.0)).reshape(b0.shape[0], C0)
            ov = ov + ((i, s_new),)
        if SF is not None:                                               # spin channels: new - old on every b0 row (site + partners)
            ov_sp = tuple((flips[f, 0], flips[f, 1], sflips[f]) for f in range(flips.shape[0]))
            dsp = cast(SF(sig, spin, b0, ov_sp) - SF(sig, spin, b0)) / tab['sd'][Kc:]
            acc = acc.at[:, Kc:].add(dsp)
        sig1 = sig                                                         # callers that need the flipped state apply the flips themselves
        acc = acc[smap0[b0]]                                             # duplicates carry the full value
        de0 = acc @ P['lin']                                              # change of the CE site energies on b0
        first0 = smap0[b0] == jnp.arange(b0.shape[0], dtype=jnp.int32)
        if not P['layers']:
            e_new = cache['e'][b0] + de0
            return jnp.sum(jnp.where(first0, de0, 0.0)) / nA, dict(e=(b0, e_new), slot=_slot_clear(smap0, b0), slot2=cache['slot2']), sig1
        L1 = P['layers'][0]; dU, dV, dc = _proj(L1, acc)                # linear in h0 -> patch from the row contributions
        U_new = cache['U'][b0] + dU; c_new = cache['c'][b0] + dc
        V_new = (cache['V'][b0] + dV) if dV is not None else None
        h0_new = (cache['h0'][b0] + acc) if 'h0' in cache else None
        # layer 1 on ball1
        b1 = jnp.concatenate([tab['ball_idx'][1][flips[f, 0]] for f in range(flips.shape[0])])
        m1 = jnp.concatenate([tab['ball_mask'][1][flips[f, 0]] for f in range(flips.shape[0])])
        b1 = jnp.where(m1, b1, b1[jnp.argmax(m1)])
        h1 = _layer1_from_proj(L1, cache['U'], cache.get('V'), cache['c'], b1, tab['SI'], tab['SM'],
                               smap=smap0, dU=U_new, dV=V_new, dc=c_new, h0raw=cache.get('h0'), h0new=h0_new)
        patch = dict(U=(b0, U_new), c=(b0, c_new)); 
        if V_new is not None: patch['V'] = (b0, V_new)
        if h0_new is not None: patch['h0'] = (b0, h0_new)
        rows, vals, smap2, mids = b1, h1, cache['slot2'], []
        prev_rows = b1
        for l, L in enumerate(P['layers'][1:]):
            mids.append((rows, vals))                                      # h_{l+1} rows for the raw cache 'hmid'
            smap2 = _slot_set(smap2, rows)
            bl = jnp.concatenate([tab['ball_idx'][l + 2][flips[f, 0]] for f in range(flips.shape[0])])
            ml = jnp.concatenate([tab['ball_mask'][l + 2][flips[f, 0]] for f in range(flips.shape[0])])
            bl = jnp.where(ml, bl, bl[jnp.argmax(ml)])
            vals = _layer_raw(L, cache['hmid'][l], bl, tab['SI'], tab['SM'], smap=smap2, uvals=vals)
            smap2 = _slot_clear(smap2, rows); rows = bl
        # site energies on the final ball: e_new = e_old + d(h0.lin) + (hL_new - hL_old).w
        sl0 = smap0[rows]; de0_rows = jnp.where(sl0 < BIG, de0[jnp.where(sl0 < BIG, sl0, 0)], 0.0)
        e_new = cache['e'][rows] + de0_rows + (vals - cache['hL'][rows]) @ P['head']['w']
        smapL = _slot_set(smap2, rows)
        firstocc = smapL[rows] == jnp.arange(rows.shape[0], dtype=jnp.int32)
        dE = jnp.sum(jnp.where(firstocc, e_new - cache['e'][rows], 0.0)) / nA
        patch.update(hmid=mids, hL=(rows, vals), e=(rows, e_new), slot=_slot_clear(smap0, b0), slot2=_slot_clear(smapL, rows))
        return dE, patch, sig1

    @jax.jit
    def commit(cache, patch, accept):
        out = dict(cache)
        for k in ('U', 'V', 'c', 'h0', 'hL', 'e'):
            if k in patch:
                rows, vals = patch[k]
                out[k] = cache[k].at[rows].set(jnp.where(accept, vals, cache[k][rows]))
        if 'hmid' in patch:
            out['hmid'] = [cache['hmid'][l].at[r].set(jnp.where(accept, v, cache['hmid'][l][r])) for l, (r, v) in enumerate(patch['hmid'])]
        out['slot'] = patch['slot']; out['slot2'] = patch['slot2']
        return out

    # prep_tab: cast the table ONCE outside the jitted step (in-jit casts are no-ops when dtypes already match;
    # a float64 table handed to the float32 functions is otherwise converted, O(N), at EVERY step).
    return dict(full_cache=full_cache, energy_of=energy_of, local_delta=local_delta, commit=commit,
                prep_tab=lambda tab: jax.tree_util.tree_map(cast, tab))


def build_balls_sparse(T, SI, SM, depth, spin=False):
    """Same output as build_balls (list of (idx (nA, w) int32, mask (nA, w) bool) per layer, ACTIVE indexing) built
    with scipy.sparse boolean products: ball0 = sites sharing a row with i [+ i + partners if spin],
    ball_l = ball_{l-1} U partners(ball_{l-1}).  O(nnz) memory, no python sets — usable at 10^6 sites."""
    import scipy.sparse as sp
    nA = len(T['active']); N = T['N']; inv = np.full(N, -1, np.int64); inv[T['active']] = np.arange(nA)
    m = T['mask']; R = T['idx'].shape[0]
    rows = np.repeat(np.arange(R), T['idx'].shape[1]).reshape(R, -1)[m]; sites = inv[T['idx'][m]]
    ok = sites >= 0; A = sp.csr_matrix((np.ones(ok.sum(), bool), (sites[ok], rows[ok])), shape=(nA, R))   # site -> rows
    B0 = (A @ A.T).tocsr(); B0.data[:] = True                                                             # sites sharing a row
    P, W = SI.shape[0], SI.shape[2]
    src = np.repeat(np.arange(nA), P * W); dst = np.transpose(SI, (1, 0, 2)).reshape(-1); msk = np.transpose(SM, (1, 0, 2)).reshape(-1) > 0
    Adj = sp.csr_matrix((np.ones(msk.sum(), bool), (src[msk], dst[msk])), shape=(nA, nA)); Adj = ((Adj + Adj.T) > 0).tocsr()
    I = sp.identity(nA, dtype=bool, format="csr")
    if spin: B0 = ((B0 + I + Adj) > 0).tocsr()
    balls = [B0]
    for _ in range(depth):
        Bp = balls[-1]; balls.append(((Bp + Bp @ Adj) > 0).tocsr())
    out = []
    for B in balls:
        B.sort_indices(); cnt = np.diff(B.indptr); w = int(cnt.max()); arr = np.zeros((nA, w), np.int32); mk = np.zeros((nA, w), bool)
        slot = np.arange(B.indptr[-1]) - np.repeat(B.indptr[:-1], cnt); r = np.repeat(np.arange(nA), cnt)
        arr[r, slot] = B.indices; mk[r, slot] = True; out.append((arr, mk))
    return out
