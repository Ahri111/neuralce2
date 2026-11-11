"""Receptive-field local update of the cluster convolution on the template-only geometry
(gpu_layer0_implicit.build_geometry).  Same mathematics as local_update.make_local_fns_v2, with every
O(N) geometry table replaced by on-the-fly (basis, cell) arithmetic:
    balls        tab['ball_idx'][l][i]    ->  ball_sites(G, l, i)
    shells       tab['SI'][:, sites]      ->  shells_of(G, sites)
    layer-0 rows dh0_a(tab['L0A'], ...)   ->  delta_h0(G, ...)
    full h0      h0_a(tab['L0A'], sig)    ->  h0_full(G, sig)   (lax.scan over cell blocks)
Cache modes (per replica, per site):
    'h0'   : h0 (K+1 [+ n_spin] floats), e, two int32 slots.   layer 1 reads h0 of the shell partners
             directly (mean commutes with the linear map, so this equals the projection form).
    'proj' : U/V/c projections (P*D [+P*D] + D floats), e, hL (+hmid), slots — the v2 layout, cheaper
             than 'h0' when the layer-0 width is large (FeNiCr: 131 channels).
Full-cache layers are evaluated in site blocks (lax.map) so the transient gathers stay O(block), not O(N).
"""
from __future__ import annotations
import numpy as np


def make_local_fns_v3(G, stat, dtype=None, SF=None, cache_mode="h0"):
    import jax, jax.numpy as jnp
    from .gpu_layer0_implicit import h0_full, delta_h0, ball_sites, shells_of, active_mask
    Kc = int(stat['K']) + 1; N = int(stat['N']); nA = int(stat['nA']); depth = int(stat['depth'])
    BIG = 2 ** 30; SB = 32768                                                  # site block for full-cache layers
    cast = (lambda x: x) if dtype is None else (lambda x: jnp.asarray(x, dtype) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating) else x)
    castP = (lambda P: P) if dtype is None else (lambda P: jax.tree_util.tree_map(cast, P))
    nblk = -(-N // SB)

    def blocked(f):
        """f(sites (SB,)) -> (SB, C): evaluate on all N sites in blocks -> (N, C)."""
        sites = jnp.minimum(jnp.arange(nblk * SB, dtype=jnp.int32), N - 1).reshape(nblk, SB)
        return jax.lax.map(f, sites).reshape(nblk * SB, -1)[:N]

    def _slot_set(slot, urows): return slot.at[urows].min(jnp.arange(urows.shape[0], dtype=jnp.int32))
    def _slot_clear(slot, urows): return slot.at[urows].set(BIG)
    def _pg(gathered, gidx, smap, uvals):
        sl = smap[gidx]; found = sl < BIG
        return jnp.where(found[..., None], uvals[jnp.where(found, sl, 0)], gathered)

    def _shells(sites, dt):
        si, sm = shells_of(G, sites); return si, sm.astype(dt)

    # ── layer with raw (previous-layer) inputs on `sites`, optional patch (smap, uvals) of the previous layer ──
    def _layer_raw(L, h_prev, sites, smap=None, uvals=None):
        si, sm = _shells(sites, h_prev.dtype); m = sm[..., None]
        g = h_prev[si]; center = h_prev[sites]
        if smap is not None: g = _pg(g, si, smap, uvals); center = _pg(center, sites, smap, uvals)
        z = (g * m).sum(-2) / jnp.maximum(sm.sum(-1), 1.0)[..., None]                       # (P, B, C)
        pre = center @ L['self'] + jnp.einsum('pbc,pcd->bd', z, L['shell']) + L['b']
        if 'shell_max' in L:
            zmax = jnp.where(m > 0, g, -1e9).max(-2); zmax = jnp.where(sm.sum(-1)[..., None] > 0, zmax, 0.0)
            pre = pre + jnp.einsum('pbc,pcd->bd', zmax, L['shell_max'])
        out = jnp.tanh(pre)
        if 'gate' in L: out = out * jax.nn.sigmoid(jnp.einsum('pbc,pcd->bd', z, L['gate']) + L['gb'])
        return out

    # ── 'proj' mode pieces (v2) ──
    def _proj(L, h):
        U = jnp.einsum('...c,pcd->...pd', h, L['shell']); c = h @ L['self']
        V = jnp.einsum('...c,pcd->...pd', h, L['gate']) if 'gate' in L else None
        return U, V, c

    def _layer1_from_proj(L, U, V, c, sites, smap=None, dU=None, dV=None, dc=None, h0raw=None, h0new=None):
        si, sm = _shells(sites, c.dtype); m = sm[..., None]; Pn = si.shape[0]
        g = jnp.stack([U[si[p], p] for p in range(Pn)]); cc = c[sites]
        if smap is not None:
            g = jnp.stack([_pg(g[p], si[p], smap, dU[:, p]) for p in range(Pn)]); cc = _pg(cc, sites, smap, dc)
        z = (g * m).sum(-2) / jnp.maximum(sm.sum(-1), 1.0)[..., None]
        pre = cc + z.sum(0) + L['b']
        if 'shell_max' in L:
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

    def _h0s(tab, sig, spin):
        h0raw = h0_full(G, sig, stat)
        if SF is not None:
            h0raw = jnp.concatenate([h0raw, blocked(lambda s: SF(sig, spin, s))], axis=1)
        act = active_mask(G, N)
        return jnp.where(act[:, None], cast((h0raw - tab['mu']) / tab['sd']), 0.0)

    @jax.jit
    def full_cache(tab, P, sig, spin=None):
        P = castP(P); tab = jax.tree_util.tree_map(cast, tab)
        h0s = _h0s(tab, sig, spin); slot = jnp.full((N,), BIG, jnp.int32); act = active_mask(G, N)
        e = h0s @ P['lin']
        if not P['layers']:
            return dict(e=e, slot=slot, slot2=slot)
        if cache_mode == "h0":
            hs = [h0s]
            for L in P['layers']:
                prev = hs[-1]; hs.append(blocked(lambda s, L=L, prev=prev: _layer_raw(L, prev, s)))
            return dict(h0=h0s, e=jnp.where(act, e + hs[-1] @ P['head']['w'], 0.0), hmid=hs[1:-1], slot=slot, slot2=slot)
        L1 = P['layers'][0]; U, V, c = _proj(L1, h0s); need_raw = 'shell_max' in L1
        h = blocked(lambda s: _layer1_from_proj(L1, U, V, c, s, h0raw=h0s if need_raw else None)); hs = [h]
        for L in P['layers'][1:]:
            prev = hs[-1]; hs.append(blocked(lambda s, L=L, prev=prev: _layer_raw(L, prev, s)))
        cache = dict(e=jnp.where(act, e + hs[-1] @ P['head']['w'], 0.0), U=U, c=c, hL=hs[-1], hmid=hs[:-1], slot=slot, slot2=slot)
        if V is not None: cache['V'] = V
        if need_raw: cache['h0'] = h0s
        return cache

    @jax.jit
    def energy_of(cache, P):
        return cache['e'].sum() / nA + cast(P['head']['b'])

    def _ball(l, flips):
        b = jnp.concatenate([ball_sites(G, l, flips[f, 0]) for f in range(flips.shape[0])]); return b

    @jax.jit
    def local_delta(tab, P, cache, sig, flips, spin=None, sflips=None):
        P = castP(P); tab = jax.tree_util.tree_map(cast, tab)
        C0 = P['lin'].shape[0]; sig1 = sig
        b0 = _ball(0, flips); smap0 = _slot_set(cache['slot'], b0)
        acc = jnp.zeros((b0.shape[0], C0), cache['e'].dtype); ov = ()
        for f in range(flips.shape[0]):
            i, s_new = flips[f, 0], flips[f, 1]
            flat, valid, value, chan = delta_h0(G, sig, i, s_new, stat, ov)
            slot = smap0[flat // Kc]; hit = valid & (slot < BIG)
            acc = acc.reshape(-1).at[jnp.where(hit, slot * C0 + chan, 0)].add(jnp.where(hit, cast(value) / tab['sd'][chan], 0.0)).reshape(b0.shape[0], C0)
            ov = ov + ((i, s_new),)
        if SF is not None:
            ov_sp = tuple((flips[f, 0], flips[f, 1], sflips[f]) for f in range(flips.shape[0]))
            dsp = cast(SF(sig, spin, b0, ov_sp) - SF(sig, spin, b0)) / tab['sd'][Kc:]
            acc = acc.at[:, Kc:].add(dsp)
        acc = acc[smap0[b0]]; de0 = acc @ P['lin']
        first0 = smap0[b0] == jnp.arange(b0.shape[0], dtype=jnp.int32)
        if not P['layers']:
            e_new = cache['e'][b0] + de0
            return jnp.sum(jnp.where(first0, de0, 0.0)) / nA, dict(e=(b0, e_new), slot=_slot_clear(smap0, b0), slot2=cache['slot2']), sig1
        if cache_mode == "h0":
            h0_new = cache['h0'][b0] + acc; patch = dict(h0=(b0, h0_new))
            # layer l+1 on ball_{l+1}; its input h_l is the cached full buffer patched on the previous ball's rows
            smap_in, uvals_in, prev_full = smap0, h0_new, cache['h0']
            smap2, live, mids = cache['slot2'], None, []
            nL = len(P['layers']); rows = b0; vals = h0_new
            for l, L in enumerate(P['layers']):
                bl = _ball(l + 1, flips)
                v = _layer_raw(L, prev_full, bl, smap=smap_in, uvals=uvals_in)
                if l + 1 < nL:
                    mids.append((bl, v))
                    if live is not None: smap2 = _slot_clear(smap2, live)
                    smap2 = _slot_set(smap2, bl); live = bl
                    smap_in, uvals_in, prev_full = smap2, v, cache['hmid'][l]
                rows, vals = bl, v
            h0_rows = _pg(cache['h0'][rows], rows, smap0, h0_new)
            e_new = h0_rows @ P['lin'] + vals @ P['head']['w']
            if live is not None: smap2 = _slot_clear(smap2, live)
            smapL = _slot_set(smap2, rows)
            firstocc = smapL[rows] == jnp.arange(rows.shape[0], dtype=jnp.int32)
            dE = jnp.sum(jnp.where(firstocc, e_new - cache['e'][rows], 0.0)) / nA
            patch.update(hmid=mids, e=(rows, e_new), slot=_slot_clear(smap0, b0), slot2=_slot_clear(smapL, rows))
            return dE, patch, sig1
        # ── 'proj' mode (v2) ──
        L1 = P['layers'][0]; dU, dV, dc = _proj(L1, acc)
        U_new = cache['U'][b0] + dU; c_new = cache['c'][b0] + dc
        V_new = (cache['V'][b0] + dV) if dV is not None else None
        h0_new = (cache['h0'][b0] + acc) if 'h0' in cache else None
        b1 = _ball(1, flips)
        h1 = _layer1_from_proj(L1, cache['U'], cache.get('V'), cache['c'], b1, smap=smap0, dU=U_new, dV=V_new, dc=c_new, h0raw=cache.get('h0'), h0new=h0_new)
        patch = dict(U=(b0, U_new), c=(b0, c_new))
        if V_new is not None: patch['V'] = (b0, V_new)
        if h0_new is not None: patch['h0'] = (b0, h0_new)
        rows, vals, smap2, mids = b1, h1, cache['slot2'], []
        for l, L in enumerate(P['layers'][1:]):
            mids.append((rows, vals)); smap2 = _slot_set(smap2, rows)
            bl = _ball(l + 2, flips)
            vals = _layer_raw(L, cache['hmid'][l], bl, smap=smap2, uvals=vals)
            smap2 = _slot_clear(smap2, rows); rows = bl
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
                rows, vals = patch[k]; out[k] = cache[k].at[rows].set(jnp.where(accept, vals, cache[k][rows]))
        if 'hmid' in patch:
            out['hmid'] = [cache['hmid'][l].at[r].set(jnp.where(accept, v, cache['hmid'][l][r])) for l, (r, v) in enumerate(patch['hmid'])]
        out['slot'] = patch['slot']; out['slot2'] = patch['slot2']
        return out

    return dict(full_cache=full_cache, energy_of=energy_of, local_delta=local_delta, commit=commit,
                prep_tab=lambda tab: jax.tree_util.tree_map(cast, tab))
