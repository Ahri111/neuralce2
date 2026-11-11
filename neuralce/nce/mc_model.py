"""Load a FINAL cluster-convolution model (models/nce/<tag>/model.pkl) and set it up
for MCMC on any supercell: general layer 0 (icet instance table), receptive-field local
update, float32 caches.  Everything the model needs is in the pickle: cluster space
definition, selected channels, standardisation, parameters (one set per seed), and for
spin models (FeNiCr, SPIN>=2) the spin channel layout (pairs, species, allowed values).

    mdl = load_model(path)
    mc  = setup(mdl, atoms, seed=0)          # atoms: ideal-lattice supercell (any size)
    E   = mc['energy'](sig[, spin])          # eV/atom, full recompute (float64 gate path)
    cache = mc['full_cache'](sig[, spin]); dE, patch, sig1 = mc['local_delta'](cache, sig, flips[, spin, sflips]); ...

Channel selection is folded into the parameters: unselected layer-0 channels get zero
weights (lin, W_self, W_p, G_p) and mu=0 / sd=1, so the MC path works on the full
K_chem (+ n_spin) icet channels and still equals the trained model exactly (gate_final).
"""
from __future__ import annotations
import pickle, numpy as np


def load_model(path):
    return pickle.load(open(path, "rb"))


def n_full_channels(mdl):
    return int(mdl['K_chem']) + (int(mdl['n_spin']) if mdl.get('spin', 0) else 0)


def _n_spin_shells(mdl):
    """how many of the conv shells carry spin: SPIN=1 puts the Heisenberg pairs on the first shells only, so its
    spin_edges must be a prefix of the conv shell_edges (then the MC reuses the conv shell tables unchanged)."""
    se, ce = list(mdl['spin_edges']), list(mdl['shell_edges'])
    assert se == ce[:len(se)], f"spin shells {se} are not the first conv shells {ce}"
    return len(se) - 1


def expand_params(mdl, seed_idx=0):
    """params on the full (K_chem [+ n_spin])-channel layer 0: zero rows for channels dropped by SELECT."""
    import jax.numpy as jnp
    # K_chem = len(cs) already counts the zerolet (= the table's K + 1 columns of h0)
    keep = np.asarray(mdl['keep']); Kfull = n_full_channels(mdl)
    P = mdl['params'][seed_idx]
    lin = np.zeros(Kfull); lin[keep] = P['lin']
    layers = []
    for L in P['layers']:
        if L['self'].shape[0] == len(keep):                 # first layer reads layer 0
            Lx = dict(L); Lx['self'] = np.zeros((Kfull, L['self'].shape[1])); Lx['self'][keep] = L['self']
            Lx['shell'] = np.zeros((L['shell'].shape[0], Kfull, L['shell'].shape[2])); Lx['shell'][:, keep] = L['shell']
            if 'gate' in L:
                Lx['gate'] = np.zeros((L['gate'].shape[0], Kfull, L['gate'].shape[2])); Lx['gate'][:, keep] = L['gate']
            if 'shell_max' in L:
                Lx['shell_max'] = np.zeros((L['shell_max'].shape[0], Kfull, L['shell_max'].shape[2])); Lx['shell_max'][:, keep] = L['shell_max']
            layers.append(Lx)
        else:
            layers.append(dict(L))
    mu = np.zeros(Kfull); sd = np.ones(Kfull); mu[keep] = mdl['mu']; sd[keep] = mdl['sd']
    Px = dict(lin=lin, layers=layers, head=dict(P['head']))
    return {k: (jnp.asarray(v) if not isinstance(v, (list, dict)) else v) for k, v in Px.items()}, mu, sd


def setup(mdl, atoms, seed_idx=0, dtype="float32", table=None):
    """Build the MC machinery for `atoms` (ideal-lattice supercell of the model's primitive).
    table: 'implicit' (default; template-only geometry, GPU memory independent of the cell size, sites in canonical
    order -> mc['perm'] / mc['inv_perm']), 'tiled' (full O(N) tables) or 'legacy' (icet supercell orbit list).
    Env CE_CNN_TABLE overrides the default."""
    import jax, jax.numpy as jnp
    from icet import ClusterSpace
    from .gpu_layer0 import build_table, make_jax_fns
    from .gpu_layer0_tiled import build_table_tiled
    from .general import shell_partners, pack
    from .local_update import build_balls, build_balls_sparse, balls_full_indexing, make_tab, make_local_fns_v2
    cs = ClusterSpace(mdl['primitive'], list(mdl['cutoffs']), mdl['chemical_symbols'])
    import os as _os; mode = table or _os.environ.get("CE_CNN_TABLE", "implicit"); legacy = mode == "legacy"
    if mode == "implicit":
        return setup_implicit(mdl, atoms, cs, seed_idx, dtype)
    # default: translation-template table (O(N) int32, no per-supercell icet enumeration; gate_tiled_table.py:
    # cv/h0/balls/dE identical, same MC speed).  CE_CNN_TABLE=legacy -> the original icet full-orbit-list builder.
    T = build_table(cs, atoms) if legacy else build_table_tiled(cs, atoms); L0 = make_jax_fns(T); nA = len(T['active'])
    part = shell_partners(atoms, T['active'], list(mdl['shell_edges']))
    _, _, SI, SM = pack([np.zeros((nA, 1))], [part]); SI, SM = SI[0], SM[0]
    P, mu, sd = expand_params(mdl, seed_idx)
    depth = len(P['layers'])
    spin = int(mdl.get('spin', 0) or 0); SF = None; spin_info = None
    if spin:
        from .spin_features import spin_tables, make_spin_fn
        ns = _n_spin_shells(mdl)                                          # SPIN=1: the first 2 conv shells; SPIN>=2: all of them
        pair_tab, comp_of = spin_tables(cs, atoms, mdl['spin_pairs'], mdl['spin_species'])
        SF, n_spin = make_spin_fn(SI, SM, T['active'], pair_tab, comp_of, len(mdl['spin_pairs']), len(mdl['spin_species']), spin, ns,
                                  int(mdl.get('spin_node_lo', 0) or 0))
        assert n_spin == int(mdl['n_spin']), (n_spin, mdl['n_spin'])
        spin_info = dict(values=np.asarray(mdl['spin_values'], float), pair_tab=pair_tab, comp_of=comp_of, n_spin=n_spin)
    balls = balls_full_indexing(T, (build_balls if legacy else build_balls_sparse)(T, SI, SM, depth, spin=bool(spin)))
    tab = make_tab(T, SI, SM, balls, mu, sd)
    DT = {"float32": jnp.float32, "float64": jnp.float64, None: None}[dtype]
    F = make_local_fns_v2(L0, depth, dtype=DT, SF=SF)
    F64 = make_local_fns_v2(L0, depth, dtype=None, SF=SF)
    tab64 = tab; tab = F['prep_tab'](tab)                  # MC table cast once (float32 path); float64 gate path keeps tab64
    ym, ys = mdl['ym'], mdl['ys']; n_atoms = len(atoms)

    def energy(sig, spin_state=None):                      # eV/atom (float64 path)
        sp = None if spin_state is None else jnp.asarray(spin_state, jnp.float64)
        return float(F64['energy_of'](F64['full_cache'](tab64, P, jnp.asarray(sig), sp), P)) * ys + ym

    return dict(cs=cs, T=T, L0=L0, SI=SI, SM=SM, P=P, tab=tab, tab64=tab64, depth=depth, F=F, F64=F64, ym=ym, ys=ys,
                n_atoms=n_atoms, n_active=nA, sig0=jnp.asarray(T['sig']), active=T['active'],
                spin=spin, spin_info=spin_info, SF=SF,
                energy=energy,
                full_cache=lambda sig, spin_state=None: F['full_cache'](tab, P, jnp.asarray(sig), spin_state),
                local_delta=lambda cache, sig, flips, spin_state=None, sflips=None: F['local_delta'](tab, P, cache, sig, flips, spin_state, sflips),
                commit=F['commit'],
                to_eV_per_atom=lambda E_std: E_std * ys + ym,
                dE_eV_total=lambda dE_std: dE_std * ys * n_atoms)


def train_path_energy(mdl, cs, atoms, spins=None, seed_idx=0):
    """Training-time prediction (eV/atom): icet local features [+ numpy spin planes] -> selected channels
    -> general.make_forward.  The MC path must reproduce this to ~1e-9 (gate_final)."""
    import jax.numpy as jnp
    from .general import local_cluster_features, shell_partners, pack, make_forward
    from .gpu_layer0 import build_table
    act = build_table(cs, atoms)['active']
    h0 = local_cluster_features(cs, atoms, act)
    if mdl.get('spin', 0):
        from .spin_features import spin_features_np
        h0 = np.hstack([h0, spin_features_np(atoms, act, np.asarray(spins, float), list(mdl['spin_edges']), mdl['spin_pairs'],
                                             mdl['spin_species'], int(mdl['spin']), int(mdl.get('spin_node_lo', 0) or 0))])
    h0 = h0[:, mdl['keep']]
    part = shell_partners(atoms, act, list(mdl['shell_edges'])); H, M, SI, SM = pack([h0], [part])
    Ptr = mdl['params'][seed_idx]
    Ptr = {k: (jnp.asarray(v) if not isinstance(v, (list, dict)) else v) for k, v in Ptr.items()}
    Ptr['layers'] = [{kk: jnp.asarray(vv) for kk, vv in L.items()} for L in Ptr['layers']]; Ptr['head'] = {kk: jnp.asarray(vv) for kk, vv in Ptr['head'].items()}
    X = jnp.asarray((H - mdl['mu']) / mdl['sd'] * M[..., None])
    return float(make_forward()(Ptr, X, jnp.asarray(M), jnp.asarray(SI), jnp.asarray(SM))[0]) * mdl['ys'] + mdl['ym']


def gate_final(mdl, atoms_list, y, seed_idx=0, tol=2e-3, spins_list=None):
    """The MC path (icet table layer 0 + expanded params) must reproduce the training-time
    prediction on the training structures: |MAE_mc - MAE_train| small and per-structure
    agreement ~1e-9 in float64.  Returns (mae_mc, max_abs_diff_vs_general_forward)."""
    from icet import ClusterSpace
    cs = ClusterSpace(mdl['primitive'], list(mdl['cutoffs']), mdl['chemical_symbols'])
    preds_mc, preds_gen = [], []
    for k, a in enumerate(atoms_list):
        sp = None if spins_list is None else np.asarray(spins_list[k], float)
        mc = setup(mdl, a, seed_idx, dtype=None)
        sp_mc = sp if (sp is None or not mc.get('implicit')) else sp[mc['perm']]   # MC path: canonical site order; training path: input order
        preds_mc.append(mc['energy'](mc['sig0'], sp_mc))
        preds_gen.append(train_path_energy(mdl, cs, a, sp, seed_idx))
    preds_mc, preds_gen = np.array(preds_mc), np.array(preds_gen)
    return float(np.mean(np.abs(preds_mc - y)) * 1e3), float(np.abs(preds_mc - preds_gen).max())


def _ensemble_fns(F):
    """Wrap the local-update functions of one network so that P (stacked over seeds) / cache / patch carry a leading
    seed axis and the energy is the seed mean.  dE = mean of the per-seed dE (exact for the mean-energy model);
    commit applies the same accept flag to every seed's cache."""
    import jax
    fc, eo, ld, cm = F['full_cache'], F['energy_of'], F['local_delta'], F['commit']

    def full_cache(tab, P, sig, spin=None):
        return jax.vmap(lambda p: fc(tab, p, sig, spin))(P)

    def energy_of(cache, P):
        return jax.vmap(eo)(cache, P).mean()

    def local_delta(tab, P, cache, sig, flips, spin=None, sflips=None):
        dE, patch, sig1 = jax.vmap(lambda p, c: ld(tab, p, c, sig, flips, spin, sflips))(P, cache)
        return dE.mean(), patch, jax.tree_util.tree_map(lambda x: x[0], sig1)

    def commit(cache, patch, accept):
        return jax.vmap(lambda c, p: cm(c, p, accept))(cache, patch)

    return dict(F, full_cache=full_cache, energy_of=energy_of, local_delta=local_delta, commit=commit)


def setup_implicit(mdl, atoms, cs, seed_idx=0, dtype="float32", cache_mode=None):
    """Template-only MC machinery (gpu_layer0_implicit + local_update_implicit).  Same interface as setup();
    sig0 / spin / caches are in CANONICAL site order (mc['perm']: canonical -> input index)."""
    import jax, jax.numpy as jnp
    from .gpu_layer0_implicit import build_geometry, make_propose
    from .local_update_implicit import make_local_fns_v3
    # seed_idx < 0 -> ENSEMBLE: the energy is the mean over all trained seeds (params stacked on a leading axis, every
    # local-update function vmapped over it).  Removes the seed lottery from MC observables at ~n_seeds x the cost.
    ens = seed_idx is not None and int(seed_idx) < 0
    if ens:
        Ps = [expand_params(mdl, s)[0] for s in range(len(mdl['params']))]
        _, mu, sd = expand_params(mdl, 0); P0 = Ps[0]
        P = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *Ps)
    else:
        P, mu, sd = expand_params(mdl, seed_idx); P0 = P
    depth = len(P0['layers'])
    spin = int(mdl.get('spin', 0) or 0)
    geo = build_geometry(cs, atoms, list(mdl['shell_edges']), depth, spin=bool(spin))
    G, stat = geo['G'], geo['stat']; SF = None; spin_info = None
    if spin:
        from .spin_features import spin_tables_implicit, make_spin_fn_implicit
        ns = _n_spin_shells(mdl)
        pair_tab, comp_of = spin_tables_implicit(geo, mdl['spin_pairs'], mdl['spin_species'])
        SF = make_spin_fn_implicit(G, pair_tab, comp_of, len(mdl['spin_pairs']), len(mdl['spin_species']), spin, ns,
                                   int(mdl.get('spin_node_lo', 0) or 0))
        spin_info = dict(values=np.asarray(mdl['spin_values'], float), pair_tab=pair_tab, comp_of=comp_of, n_spin=int(mdl['n_spin']))
    C0 = int(P0['lin'].shape[0])
    if cache_mode is None:                      # per-site bytes: h0 cache C0 floats vs projections (2P+1)*D floats
        if depth == 0: cache_mode = "h0"
        else:
            L1 = P0['layers'][0]; D = int(L1['self'].shape[1]); Pn = int(L1['shell'].shape[0])
            cache_mode = "h0" if C0 <= (Pn * (2 if 'gate' in L1 else 1) + 1) * D else "proj"
    tab = dict(mu=jnp.asarray(mu), sd=jnp.asarray(sd))
    DT = {"float32": jnp.float32, "float64": jnp.float64, None: None}[dtype]
    F = make_local_fns_v3(G, stat, dtype=DT, SF=SF, cache_mode=cache_mode)
    F64 = make_local_fns_v3(G, stat, dtype=None, SF=SF, cache_mode=cache_mode)
    if ens:
        F, F64 = _ensemble_fns(F), _ensemble_fns(F64)
    tab64 = tab; tab = F['prep_tab'](tab)
    ym, ys = mdl['ym'], mdl['ys']; n_atoms = len(atoms); nA = int(stat['nA'])
    propose, site_uniform = make_propose(G, stat)

    def energy(sig, spin_state=None):
        sp = None if spin_state is None else jnp.asarray(spin_state, jnp.float64)
        return float(F64['energy_of'](F64['full_cache'](tab64, P, jnp.asarray(sig), sp), P)) * ys + ym

    return dict(cs=cs, T=None, L0=None, SI=None, SM=None, P=P, tab=tab, tab64=tab64, depth=depth, F=F, F64=F64, ym=ym, ys=ys,
                n_atoms=n_atoms, n_active=nA, sig0=jnp.asarray(geo['sig']), active=np.asarray(geo['active_orig']),
                spin=spin, spin_info=spin_info, SF=SF, energy=energy,
                full_cache=lambda sig, spin_state=None: F['full_cache'](tab, P, jnp.asarray(sig), spin_state),
                local_delta=lambda cache, sig, flips, spin_state=None, sflips=None: F['local_delta'](tab, P, cache, sig, flips, spin_state, sflips),
                commit=F['commit'], to_eV_per_atom=lambda E_std: E_std * ys + ym, dE_eV_total=lambda dE_std: dE_std * ys * n_atoms,
                implicit=True, geo=geo, G=G, stat=stat, perm=geo['perm'], inv_perm=geo['inv_perm'], propose=propose, site_uniform=site_uniform,
                cache_mode=cache_mode)
