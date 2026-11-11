"""Spin channels of the cluster convolution on the MC path (jax), equal to the numpy training
features in general.py (spin_site_features / spin_node_planes / spin_env_planes) evaluated
on the SAME shell tables (SI, SM from general.shell_partners -> pack).

Per active site i, with s = spin (values -1 / 0 / +1 in the FeNiCr data), for the conv shells p
and the sorted species pairs (a, b):
    site planes  f[i, p*n_pairs + k]      = 0.5 * s_i * sum_{j in shell p(i), pair(i,j)=k} s_j
    node planes  [ s_i^2 , s_i * sum_{j in p} s_j  for each p ]
    env planes   f[i, (p*n_pairs + k)*n_species + c] = site plane * n_c(1NN of i) / n_1NN(i)     (SPIN=3)
The spin state of a site changes these features on the site itself and on every shell partner,
so the local update's ball0 is enlarged by the shell partners when a model carries spin channels
(local_update.build_balls(..., spin=True)).
"""
from __future__ import annotations
import numpy as np


def spin_tables(cs, atoms, pairs, species):
    """pair_tab[a, b] -> index in `pairs` of the sorted species pair (a, b are icet species indices
    of the spin sublattice); comp_of[a] -> column of the species in `species`.  Spin models need one
    active sublattice (FeNiCr fcc); asserted."""
    from ase.data import atomic_numbers, chemical_symbols
    from .gpu_layer0 import species_index
    sig, Msite, sub_of, maps = species_index(cs, atoms)
    active_maps = [m for m in maps if len(m) > 1]
    assert len(active_maps) == 1, f"spin MC path needs exactly one active sublattice, found {len(active_maps)}"
    m = active_maps[0]; sym_of = {k: chemical_symbols[Z] for Z, k in m.items()}; M = len(m)
    pair_tab = np.zeros((M, M), np.int32); comp_of = np.zeros(M, np.int32)
    pairs = [tuple(p) for p in pairs]
    for a in range(M):
        comp_of[a] = species.index(sym_of[a])
        for b in range(M):
            pair_tab[a, b] = pairs.index(tuple(sorted((sym_of[a], sym_of[b]))))
    return pair_tab, comp_of


def make_spin_fn(SI, SM, active, pair_tab, comp_of, n_pairs, n_species, mode, n_spin_shells=None, node_lo=0):
    """jitted feats(sig, spin, sites) -> (B, n_spin) on active-indexed `sites`; sig/spin are FULL-cell arrays.
    mode 1: the Heisenberg pair planes only, over the first n_spin_shells conv shells (icet+spin encoding);
    mode 2: site + node planes; mode 3: + env planes (the FeNiCr benchmark model)."""
    import jax, jax.numpy as jnp
    SI = jnp.asarray(SI); SM = jnp.asarray(SM); active = jnp.asarray(active)
    pair_tab = jnp.asarray(pair_tab); comp_of = jnp.asarray(comp_of)
    assert mode in (1, 2, 3)
    P = int(n_spin_shells or SI.shape[0]); SI = SI[:P]; SM = SM[:P]

    def feats(sig, spin, sites, ov=None):
        """ov = ((site, species, spin_value), ...): pending moves applied on the gathered values (no copy of sig/spin)."""
        si = SI[:, sites]; sm = SM[:, sites].astype(spin.dtype)                  # (P, B, W)
        gi = active[sites]; gj = active[si]
        s_i = spin[gi]; s_j = spin[gj]; sg_i = sig[gi]; sg_j = sig[gj]          # (B,), (P, B, W)
        for site, species, sval in (ov or ()):
            s_i = jnp.where(gi == site, sval.astype(s_i.dtype), s_i); s_j = jnp.where(gj == site, sval.astype(s_j.dtype), s_j)
            sg_i = jnp.where(gi == site, species.astype(sg_i.dtype), sg_i); sg_j = jnp.where(gj == site, species.astype(sg_j.dtype), sg_j)
        k = pair_tab[sg_i[None, :, None], sg_j]                                  # (P, B, W)
        w = sm * s_j                                                             # masked partner spins
        base = 0.5 * s_i[None, :, None] * (jax.nn.one_hot(k, n_pairs, dtype=spin.dtype) * w[..., None]).sum(2)   # (P, B, n_pairs)
        B = sites.shape[0]
        base = jnp.transpose(base, (1, 0, 2)).reshape(B, P * n_pairs)
        if mode == 1:                                                                                          # pair planes only
            return base
        node = jnp.concatenate([(s_i ** 2)[:, None], (s_i[None, :] * w.sum(2)).T[:, node_lo:]], axis=1)       # (B, 1+P-node_lo)
        out = [base, node]
        if mode == 3:
            c = comp_of[sg_j[0]]                                                                                 # (B, W) 1NN species
            comp = (jax.nn.one_hot(c, n_species, dtype=spin.dtype) * sm[0][..., None]).sum(1) / jnp.maximum(sm[0].sum(1), 1.0)[:, None]
            out.append((base[:, :, None] * comp[:, None, :]).reshape(B, -1))
        return jnp.concatenate(out, axis=1)

    n_spin = P * n_pairs + (0 if mode == 1 else 1 + P - node_lo + (P * n_pairs * n_species if mode == 3 else 0))
    return feats, n_spin


def spin_features_np(atoms, active, spins, shell_edges, pairs, species, mode, node_lo=0):
    """numpy training-path reference (general.py), same channel order as make_spin_fn."""
    from .general import spin_site_features, spin_node_planes, spin_env_planes
    f = [spin_site_features(atoms, active, spins, shell_edges, pairs)]
    if mode == 1:
        return f[0]                                                        # training path: SPIN=1 has no node planes
    f.append(spin_node_planes(atoms, active, spins, shell_edges, node_lo))
    if mode == 3:
        f.append(spin_env_planes(atoms, active, spins, shell_edges, pairs, species))
    return np.hstack(f)


def spin_tables_implicit(geo, pairs, species):
    """pair_tab / comp_of from the primitive-level species maps of build_geometry (no icet call on the supercell)."""
    from ase.data import chemical_symbols
    act_sl = sorted(set(int(geo['b_sl'][b]) for b in geo['active_b']))
    assert len(act_sl) == 1, f"spin MC path needs exactly one active sublattice, found {len(act_sl)}"
    m = geo['maps_sl'][act_sl[0]]; sym_of = {k: chemical_symbols[Z] for Z, k in m.items()}; M = len(m)
    pair_tab = np.zeros((M, M), np.int32); comp_of = np.zeros(M, np.int32); pairs = [tuple(p) for p in pairs]
    for a in range(M):
        comp_of[a] = species.index(sym_of[a])
        for b in range(M): pair_tab[a, b] = pairs.index(tuple(sorted((sym_of[a], sym_of[b]))))
    return pair_tab, comp_of


def make_spin_fn_implicit(G, pair_tab, comp_of, n_pairs, n_species, mode, n_spin_shells=None, node_lo=0):
    """feats(sig, spin, sites, ov=None) on the template geometry (canonical FULL-cell site indices)."""
    import jax, jax.numpy as jnp
    from .gpu_layer0_implicit import shells_of
    pair_tab = jnp.asarray(pair_tab); comp_of = jnp.asarray(comp_of); assert mode in (1, 2, 3)

    def feats(sig, spin, sites, ov=None):
        si, smb = shells_of(G, sites)
        if n_spin_shells: si, smb = si[:n_spin_shells], smb[:n_spin_shells]     # SPIN=1: spin lives on the first shells only
        sm = smb.astype(spin.dtype); P = si.shape[0]
        s_i = spin[sites]; s_j = spin[si]; sg_i = sig[sites]; sg_j = sig[si]
        for site, species, sval in (ov or ()):
            s_i = jnp.where(sites == site, sval.astype(s_i.dtype), s_i); s_j = jnp.where(si == site, sval.astype(s_j.dtype), s_j)
            sg_i = jnp.where(sites == site, species.astype(sg_i.dtype), sg_i); sg_j = jnp.where(si == site, species.astype(sg_j.dtype), sg_j)
        k = pair_tab[sg_i[None, :, None], sg_j]; w = sm * s_j
        base = 0.5 * s_i[None, :, None] * (jax.nn.one_hot(k, n_pairs, dtype=spin.dtype) * w[..., None]).sum(2)
        B = sites.shape[0]; base = jnp.transpose(base, (1, 0, 2)).reshape(B, P * n_pairs)
        if mode == 1: return base
        node = jnp.concatenate([(s_i ** 2)[:, None], (s_i[None, :] * w.sum(2)).T[:, node_lo:]], axis=1)
        out = [base, node]
        if mode == 3:
            c = comp_of[sg_j[0]]
            comp = (jax.nn.one_hot(c, n_species, dtype=spin.dtype) * sm[0][..., None]).sum(1) / jnp.maximum(sm[0].sum(1), 1.0)[:, None]
            out.append((base[:, :, None] * comp[:, None, :]).reshape(B, -1))
        return jnp.concatenate(out, axis=1)
    return feats
