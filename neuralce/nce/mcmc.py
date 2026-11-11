"""MCMC for the FINAL cluster-convolution model (framework "ce_cnn" in neuralce/models/mcmc.py).

Energy model: models/nce/<tag>/model.pkl (mc_model.load_model).  Energy updates use
the receptive-field local ΔE (local_update.make_local_fns_v2, float32) inside a jitted
lax.scan; every chunk the carried energy is re-anchored with a full recompute and the drift
is logged.  Moves: canonical swaps of two sites of the same sublattice (different species;
a same-species draw is a no-op step).  Replicas are vmapped.

    run_mc        single temperature, n_replicas independent chains
    run_multiT_mc one chain per temperature in --temps (independent, vmapped)
    run_pt_mc     parallel tempering: geometric ladder T_min..T_max (or T_grid),
                  configuration swaps between neighbouring temperatures every exchange_every
Outputs in out_dir: energies.npz (per-chunk means + subsampled trajectory), final_sigma.npy,
final_<T>.cif, mcmc_summary.json (via the central entry).
"""
from __future__ import annotations
import os, time, json, hashlib, numpy as np
KB = 8.617333262e-5


def _log(*a): print(*a, flush=True)


def _prepare(model_path, init_cif, supercell=(1, 1, 1), seed_idx=0, dtype="float32", n_replicas=1, spin_init="random", p_spin=0.5, seed=0,
             spin_allowed=None):
    import jax, jax.numpy as jnp
    from ase.io import read
    from .mc_model import load_model, setup
    mdl = load_model(model_path)
    atoms = read(init_cif)
    # relaxed cifs (PtNi dataset) -> ideal lattice of the model's primitive (icet mapper; identity for ideal cells)
    from icet import ClusterSpace
    from icet.tools import map_structure_to_reference
    prim = mdl['primitive']
    # icet's map_structure_to_reference is only needed for RELAXED cifs; it is O(N^2) in memory (16k atoms > 64 GB),
    # so an ideal-lattice supercell (every site on the primitive lattice, integer cell matrix) skips it.
    from .gpu_layer0_tiled import site_decompose
    try:
        site_decompose(prim, atoms, tol=1e-3); ideal = True
    except AssertionError:
        ideal = False
    if ideal:
        _log(f"[ce_cnn] init structure is on the ideal lattice of the model primitive ({len(atoms)} atoms); no snapping")
    else:
        try:
            mapped, info = map_structure_to_reference(atoms, prim, inert_species=None, assume_no_cell_relaxation=False)
            if info.get('drmax', 0.0) > 1e-6:
                _log(f"[ce_cnn] init structure snapped onto the ideal lattice (drmax {info.get('drmax'):.3f} A)")
            atoms = mapped
        except Exception as e:                                         # e.g. inert sublattice needs inert_species
            _log(f"[ce_cnn] map_structure_to_reference skipped ({type(e).__name__}: {e}); using the cif as is")
    sc = tuple(int(x) for x in supercell)
    if sc != (1, 1, 1):
        # tile in memory and re-shuffle the species within each sublattice: a random configuration at the cif's
        # composition (no periodic-image start, no giant cif on disk).  CE_CNN_SUPERCELL_PERIODIC=1 keeps the images.
        small = atoms; atoms = small.repeat(sc)
        if os.environ.get("CE_CNN_SUPERCELL_PERIODIC", "0") != "1":
            from .gpu_layer0_tiled import site_decompose as _sd
            basis, _, _, _, _ = _sd(prim, atoms); Z = atoms.numbers.copy(); rng_sc = np.random.default_rng(seed)
            sub = ClusterSpace(prim, list(mdl['cutoffs']), mdl['chemical_symbols']).get_sublattices(prim)
            b2sl = np.array([sub.get_sublattice_index_from_site_index(p) for p in range(len(prim))])
            for sl in np.unique(b2sl):
                m = np.isin(basis, np.where(b2sl == sl)[0]); Z[m] = rng_sc.permutation(Z[m])
            atoms.numbers = Z
            _log(f"[ce_cnn] supercell {sc} of the init cif -> {len(atoms)} atoms, species re-shuffled per sublattice (seed {seed})")
        else:
            _log(f"[ce_cnn] supercell {sc} of the init cif -> {len(atoms)} atoms (periodic images kept)")
    t0 = time.time(); mc = setup(mdl, atoms, seed_idx=seed_idx, dtype=dtype)
    _log(f"[ce_cnn] model {os.path.basename(os.path.dirname(model_path))}: depth {mc['depth']}, "
         f"{mc['n_atoms']} atoms ({mc['n_active']} active), setup {time.time()-t0:.1f}s"
         + (f", E0 = {mc['energy'](mc['sig0']):.6f} eV/atom" if not mc['spin'] else " (spin model: E0 logged with the spin init)"))
    mc['mdl'] = mdl; mc['atoms'] = atoms
    if mc.get('implicit'):
        # template-only geometry: swap / spin-site proposals are (cell, basis) arithmetic (mc['propose'], mc['site_uniform']);
        # sites are in canonical order, mc['perm'] maps canonical -> input index
        st = mc['stat']; _log(f"[ce_cnn] implicit geometry: {st['nB']} basis sites x {st['n_cells']} cells, template rows {st['Rt']}, "
                              f"ball widths {st['ball_w']}, cache mode '{mc['cache_mode']}' (memory = state only)")
        act = np.arange(mc['n_atoms'])[np.asarray(mc['G']['Mb'])[np.arange(mc['n_atoms']) % st['nB']] > 1]   # canonical active sites
    else:
        T = mc['T']; act = T['active']; Msite = T['M']
        # canonical-swap tables: active sites grouped by SUBLATTICE (icet sublattice index) -- NOT by M:
        # STFO's Ti/Fe and O/Xe sublattices both have M=2, grouping by M would swap a B-site with an O-site.
        from .gpu_layer0 import species_index
        _, _, sub_of, _ = species_index(mc['cs'], atoms)
        groups = {}
        for i in act:
            groups.setdefault(int(sub_of[i]), []).append(int(i))
        keys = [k for k in groups if Msite[groups[k][0]] > 1]
        _log(f"[ce_cnn] swap groups (sublattices with >1 species): " + ", ".join(f"#{k}: {len(groups[k])} sites, M={int(Msite[groups[k][0]])}" for k in keys))
        W = max(len(groups[k]) for k in keys)
        gtab = np.zeros((len(keys), W), np.int32); glen = np.zeros(len(keys), np.int32)
        for g, k in enumerate(keys):
            gtab[g, :len(groups[k])] = groups[k]; glen[g] = len(groups[k])
        mc['gtab'] = jnp.asarray(gtab); mc['glen'] = jnp.asarray(glen)
    # spin models (FeNiCr): spin state per site, values from the training data (-1 / 0 / +1); spin moves on the active sites
    mc['p_spin'] = 0.0
    if mc['spin']:
        vals = mc['spin_info']['values']; rng = np.random.RandomState(seed)
        spin0 = np.zeros(len(atoms))
        allow = _spin_allow_table(mc, vals, spin_allowed) if spin_allowed else None
        if spin_init == "random" and allow is not None:              # per species: only the values that species takes in the data
            s0 = np.asarray(mc['sig0'])
            for k in range(allow.shape[0]):
                m = s0 == k; spin0[m] = rng.choice(vals[allow[k]], size=int(m.sum()))
        elif spin_init == "random":
            spin0[act] = rng.choice(vals, size=len(act))
        elif spin_init == "fm":
            spin0[act] = vals.max()
        elif spin_init == "zero":
            spin0[act] = 0.0
        else:
            raise ValueError(f"spin_init must be random|fm|zero, got {spin_init}")
        mc['spin0'] = jnp.asarray(spin0, jnp.float32 if dtype == "float32" else jnp.float64)
        mc['spin_sites'] = None if mc.get('implicit') else jnp.asarray(np.asarray(act, np.int32)); mc['spin_values'] = jnp.asarray(vals, mc['spin0'].dtype)
        mc['p_spin'] = float(p_spin)
        mc['spin_allow'] = None
        if allow is not None:
            s0 = np.asarray(mc['sig0']); vi = np.argmin(np.abs(spin0[:, None] - vals[None, :]), axis=1)
            if not allow[s0, vi].all():
                raise ValueError(f"spin_init '{spin_init}' puts spins outside the allowed values {spin_allowed}")
            mc['spin_allow'] = jnp.asarray(allow)
            _log(f"[ce_cnn] allowed spins per species: {spin_allowed}")
        _log(f"[ce_cnn] spin model: {len(vals)} spin values {vals.tolist()}, init '{spin_init}', spin-move probability {p_spin}, "
             f"E0(spin) = {mc['energy'](mc['sig0'], np.asarray(spin0)):.6f} eV/atom")
    else:
        mc['spin0'] = None
    return mc


def _make_step(mc, beta_total):
    """one swap-Metropolis step; beta_total = 1/(kB T) applied to the TOTAL energy change (eV)."""
    import jax, jax.numpy as jnp
    F = mc['F']; tab = mc['tab']; P = mc['P']; implicit = bool(mc.get('implicit'))
    if not implicit: gtab, glen = mc['gtab'], mc['glen']
    scale = mc['ys'] * mc['n_atoms']                                    # std units -> eV (total)

    has_spin = bool(mc['spin']); p_spin = mc['p_spin']; allow = mc.get('spin_allow')

    def propose_swap(key, sig, spin):
        if implicit:
            i, j = mc['propose'](key)
        else:
            k1, k2, k3 = jax.random.split(key, 3)
            g = jax.random.randint(k1, (), 0, gtab.shape[0])
            i = gtab[g, jax.random.randint(k2, (), 0, glen[g])]
            j = gtab[g, jax.random.randint(k3, (), 0, glen[g])]
        flips = jnp.array([[i, sig[j]], [j, sig[i]]])                   # same species -> no-op step
        sfl = None if spin is None else jnp.array([spin[j], spin[i]])   # the spins travel with the atoms
        return flips, sfl

    def propose_spin(key, sig, spin):
        """spin move: one active site, new value drawn from the other allowed values."""
        k1, k2 = jax.random.split(key)
        vals = mc['spin_values']; nv = vals.shape[0]
        if implicit:
            i = mc['site_uniform'](k1)
        else:
            ss = mc['spin_sites']; i = ss[jax.random.randint(k1, (), 0, ss.shape[0])]
        cur = jnp.argmin(jnp.abs(vals - spin[i]))
        if allow is None:
            new = vals[(cur + 1 + jax.random.randint(k2, (), 0, nv - 1)) % nv]
        else:                                                         # uniform over the species' other allowed values (symmetric)
            row = allow[sig[i]]; w = row & (jnp.arange(nv) != cur); w = jnp.where(w.any(), w, row)
            new = vals[jax.random.choice(k2, nv, p=w.astype(vals.dtype) / w.sum())]
        return jnp.array([[i, sig[i]], [i, sig[i]]]), jnp.array([new, new])

    def step(carry, key):
        sig, spin, cache, E, nacc, beta = carry
        kp, ka, ks = jax.random.split(key, 3)
        if has_spin:
            fs, ss_ = propose_swap(kp, sig, spin); fp, sp_ = propose_spin(kp, sig, spin)
            use_spin = jax.random.uniform(ks) < p_spin
            flips = jnp.where(use_spin, fp, fs); sflips = jnp.where(use_spin, sp_, ss_)
            dE, patch, sig_new = F['local_delta'](tab, P, cache, sig, flips, spin, sflips)
        else:
            flips, _ = propose_swap(kp, sig, None)
            dE, patch, sig_new = F['local_delta'](tab, P, cache, sig, flips)
        dE_eV = dE * scale
        acc = (dE_eV <= 0) | (jax.random.uniform(ka) < jnp.exp(-beta * dE_eV))
        cache = F['commit'](cache, patch, acc)
        sig = sig.at[flips[:, 0]].set(jnp.where(acc, flips[:, 1].astype(sig.dtype), sig[flips[:, 0]]))
        if has_spin:
            spin = spin.at[flips[:, 0]].set(jnp.where(acc, sflips.astype(spin.dtype), spin[flips[:, 0]]))
        E = E + jnp.where(acc, dE, 0.0)
        return (sig, spin, cache, E, nacc + acc, beta), E

    return step


def _chunk_runner(mc, chunk):
    import jax, jax.numpy as jnp
    step = _make_step(mc, None)

    def chunk_fn(sig, spin, cache, E, beta, key):
        (sig, spin, cache, E, nacc, _), Es = jax.lax.scan(step, (sig, spin, cache, E, 0, beta), jax.random.split(key, chunk))
        return sig, spin, cache, E, nacc, Es
    # donate the state buffers (sigma, caches, E): without donation XLA holds input AND output copies of the O(N)
    # state during the chunk (peak = 2x resident; 23 GB at 4M atoms).  Callers never reuse the pre-chunk arrays.
    return jax.jit(jax.vmap(chunk_fn), donate_argnums=(0, 2, 3))


def _init_sig_from_npz(mc, init_npz, temps, R):
    """(R, N) canonical sigma of the per-replica starting structures in init_npz (spin-free models, implicit geometry)."""
    import jax.numpy as jnp
    if mc['spin']:
        raise ValueError("--init_npz: per-replica starting structures are implemented for spin-free models "
                         "(a spin model would also need per-replica spin states)")
    if not mc.get('implicit'):
        raise ValueError("--init_npz needs the implicit geometry (CE_CNN_TABLE=implicit, the default)")
    from neuralce.mc.init_structures import load_init_npz
    nums, _ = load_init_npz(init_npz, mc['cs'].primitive_structure, mc['atoms'], temps)
    if nums.shape[0] != R:
        raise ValueError(f"--init_npz has {nums.shape[0]} structures for {R} replicas")
    return jnp.asarray(sigma_from_numbers(mc, nums)).astype(mc['sig0'].dtype)


def sigma_from_numbers(mc, nums):
    """atomic numbers (R, N) in the atom order of mc['atoms'] -> species-index sigma (R, N) in canonical site order (implicit
    geometry); species index = rank within the sublattice, exactly as gpu_layer0_implicit.build_geometry."""
    geo = mc['geo']; nums = np.atleast_2d(np.asarray(nums)); sig = np.zeros(nums.shape, np.int32)
    for b in range(int(geo['nB'])):
        m = geo['basis'] == b; lut = geo['maps_sl'][int(geo['b_sl'][b])]
        bad = sorted(set(int(z) for z in np.unique(nums[:, m])) - set(int(z) for z in lut))
        if bad:
            raise ValueError(f"basis site {b}: atomic numbers {bad} not in {sorted(int(z) for z in lut)}")
        table = np.zeros(256, np.int32)
        for z, k in lut.items():
            table[int(z)] = int(k)
        sig[:, m] = table[nums[:, m]]
    return sig[:, geo['perm']]


def numbers_from_sigma(mc, sig, atoms=None):
    """species-index sigma (R, N) in the atom order of the MC cell -> atomic numbers (R, N).  Inverse of the mapping
    _write_cifs applies, and the counterpart of sigma_from_numbers; final_sigma.npy is already in that atom order, so
    reading a finished run's structures back needs neither its cifs nor the model that wrote them.
    atoms: the MC cell, needed for the species of any INACTIVE site (defaults to mc['atoms'], which only _prepare sets)."""
    geo = mc.get('geo')
    if geo is None: raise ValueError("numbers_from_sigma needs the implicit geometry")
    base = mc.get('atoms') if atoms is None else atoms
    if base is None: raise ValueError("numbers_from_sigma needs the cell: pass atoms= (mc came from setup(), not _prepare())")
    sig = np.atleast_2d(np.asarray(sig))
    maps = [sorted(m, key=lambda z: m[z]) for m in geo['maps_sl']]          # Z list ordered by species index
    sl_of_site = np.asarray(geo['b_sl'])[np.asarray(geo['basis'])]
    Z = np.tile(np.asarray(base.numbers, np.uint8), (sig.shape[0], 1))
    act = np.asarray(geo['active_orig']); sl_act = sl_of_site[act]
    for s_ in np.unique(sl_act):
        idx = act[sl_act == s_]; lut = np.asarray(maps[int(s_)], np.uint8)
        Z[:, idx] = lut[np.asarray(sig)[:, idx]]
    return Z


def _spin_allow_table(mc, vals, spin_allowed):
    """bool table allow[species_index, value_index] from {symbol: [allowed spin values]}.  Species index = the MC's
    sigma convention (rank of Z within the active sublattice, implicit geometry).  The spin surrogate is only trained on
    the spin states each species actually takes in DFT (FeNiCr: Ni is 0 in 84% of the data and almost never
    antiparallel to Fe); letting every species take every value sends the MC into states the model never saw."""
    from ase.data import atomic_numbers
    geo = mc.get('geo')
    if geo is None: raise ValueError("--spin_allowed needs the implicit geometry")
    act_sl = sorted(set(int(geo['b_sl'][b]) for b in geo['active_b']))
    if len(act_sl) != 1: raise ValueError(f"--spin_allowed supports one active sublattice, found {len(act_sl)}")
    zmap = geo['maps_sl'][act_sl[0]]                                  # Z -> species index
    allow = np.zeros((len(zmap), len(vals)), bool)
    for sym, vs in spin_allowed.items():
        if atomic_numbers[sym] not in zmap: raise ValueError(f"--spin_allowed species {sym} is not on the active sublattice")
        for v in vs:
            j = int(np.argmin(np.abs(np.asarray(vals) - float(v))))
            if abs(vals[j] - float(v)) > 1e-6: raise ValueError(f"spin value {v} for {sym} is not one of the model's {list(vals)}")
            allow[zmap[atomic_numbers[sym]], j] = True
    if not allow.any(1).all(): raise ValueError(f"--spin_allowed must list every species: {spin_allowed}")
    return allow


def _run_fingerprint(mc, sig):
    """Short hash of the model parameters and the starting configuration (which carries the init cif, the supercell
    shuffle seed and init_npz).  Stored in the checkpoint so a restart in the same folder with a different model file,
    a different seed_idx or different starting structures is refused instead of resuming that state."""
    import jax
    h = hashlib.md5()
    for leaf in jax.tree_util.tree_leaves(mc['P']):
        a = np.asarray(leaf)
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes() if a.dtype.kind in "buif" else str(leaf).encode())
    h.update(np.ascontiguousarray(np.asarray(sig), dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


def _run_replicas(mc, betas, n_steps, seed, chunk, log_every, sample_interval, out_dir, tag, exchange_every=0, anchor_every=None,
                  init_npz=None, checkpoint_every=0):
    """generic driver: R replicas with their own beta; PT exchange if exchange_every > 0.
    anchor_every: full recompute of the caches / carried energy every K chunks (float32 drift control).  Default: every
    chunk while chunk >= N (small cells), otherwise about once per sweep (a full recompute is O(N x rows), a chunk is O(chunk)).
    init_npz: per-replica starting structures, replica r <-> betas[r] (default: every replica starts from the init cif).
    checkpoint_every: save the state to out_dir/checkpoint.npz every K chunks, rounded up to a multiple of anchor_every so the
    file is written right after a full recompute (a resumed run is then bit-identical to an uninterrupted one); a restart
    with the same settings resumes from it, other settings are refused; the file is removed when the run finishes."""
    import jax, jax.numpy as jnp
    R = len(betas); betas = jnp.asarray(betas, jnp.float32)
    run_chunk = _chunk_runner(mc, chunk)
    temps = 1.0 / (KB * np.asarray(betas, np.float64))
    if init_npz:
        sig = _init_sig_from_npz(mc, init_npz, temps, R)
        _log(f"[{tag}] per-replica starting structures from {init_npz} (matched by lattice position)")
    else:
        sig = jnp.stack([mc['sig0']] * R)
    spin = None if mc['spin0'] is None else jnp.stack([mc['spin0']] * R)
    if spin is not None and init_npz and mc.get('spin_allow') is not None:   # per-replica structures: redraw spins for their species
        A = np.asarray(mc['spin_allow']); v = np.asarray(mc['spin_values']); sg = np.asarray(sig); rs = np.random.RandomState(seed + 1)
        sp = np.zeros(sg.shape)
        for k in range(A.shape[0]): m = sg == k; sp[m] = rs.choice(v[A[k]], size=int(m.sum()))
        spin = jnp.asarray(sp, mc['spin0'].dtype)
    # full recompute per replica with lax.map (sequential over replicas): vmap would materialise the row products of
    # ALL replicas at once, O(R x rows) memory (32 replicas x 47M rows at 256k atoms = 24 GB -> OOM).
    full = (lambda sg: jax.lax.map(mc['full_cache'], sg)) if spin is None else (lambda sg, sp: jax.lax.map(lambda x: mc['full_cache'](x[0], x[1]), (sg, sp)))
    key = jax.random.PRNGKey(seed)
    n_chunks = max(n_steps // chunk, 1)
    E_chunk, acc_chunk, drift_log, traj = [], [], [], []
    n_ex_try = n_ex_acc = 0
    if anchor_every is None: anchor_every = max(1, int(np.ceil(mc['n_atoms'] / chunk)))
    drift = 0.0
    snap_every = int(os.environ.get("CE_CNN_SNAPSHOT_EVERY", "0")); snaps, snap_chunks, spin_snaps = [], [], []
    ck_every = int(checkpoint_every or 0); ck_path = os.path.join(out_dir, "checkpoint.npz"); c0 = 0
    if ck_every:
        ck_every = int(np.ceil(ck_every / anchor_every)) * anchor_every
        want = dict(n_steps=int(n_steps), chunk=int(chunk), R=int(R), n_atoms=int(mc['n_atoms']), sample_interval=int(sample_interval),
                    seed=int(seed), anchor_every=int(anchor_every), exchange_every=int(exchange_every))
        int_keys = list(want)
        want["model"] = _run_fingerprint(mc, sig)                     # model parameters + starting configuration
        if os.path.exists(ck_path):
            ck = np.load(ck_path)
            have = {k: int(ck["cfg_" + k]) for k in int_keys}
            if "cfg_model" in ck.files: have["model"] = str(ck["cfg_model"])
            else:                                                         # written before the fingerprint existed
                have["model"] = want["model"]                             # skip the comparison; later saves do carry it
                _log(f"[{tag}] warning: {ck_path} predates the model fingerprint - resuming without verifying the model / starting structures")
            same_T = np.allclose(np.asarray(ck["cfg_betas"], np.float64), np.asarray(betas, np.float64), rtol=0, atol=1e-6)
            if have != want or not same_T:
                raise RuntimeError(f"checkpoint {ck_path} belongs to a different run (checkpoint {have} vs this run {want}, "
                                   f"same temperatures: {same_T}); delete it to start fresh")
            c0 = int(ck["chunk"]); sig = jnp.asarray(ck["sig"]); key = jnp.asarray(ck["key"]); drift = float(ck["drift"])
            if spin is not None: spin = jnp.asarray(ck["spin"])
            n_ex_try, n_ex_acc = int(ck["n_ex_try"]), int(ck["n_ex_acc"])
            E_chunk = list(ck["E_chunk"]); acc_chunk = list(ck["acc_chunk"]); drift_log = [float(x) for x in ck["drift_log"]]
            traj = [np.asarray(ck["traj"])] if ck["traj"].shape[1] else []
            snaps = list(ck["snaps"]); snap_chunks = [int(x) for x in ck["snap_chunks"]]; spin_snaps = list(ck["spin_snaps"]) if "spin_snaps" in ck.files else []
            _log(f"[{tag}] resumed from {ck_path}: chunk {c0}/{n_chunks}")
    cache = full(sig) if spin is None else full(sig, spin)
    E = jax.vmap(lambda c: mc['F']['energy_of'](c, mc['P']))(cache)
    _log(f"[{tag}] {R} replicas, {n_chunks} chunks of {chunk} steps; full re-anchor every {anchor_every} chunk(s)"
         + (f"; checkpoint every {ck_every} chunks" if ck_every else ""))
    t0 = time.time()
    inv = mc['inv_perm'] if mc.get('implicit') else None
    for c in range(c0, n_chunks):
        key, kc, kx = jax.random.split(key, 3)
        sig, spin, cache, E, nacc, Es = run_chunk(sig, spin, cache, E, betas, jax.random.split(kc, R))
        if snap_every and (c + 1) % snap_every == 0:                    # configuration snapshots for SRO / structure averages
            sg = np.asarray(sig).astype(np.int8); snaps.append(sg[:, inv] if inv is not None else sg); snap_chunks.append(c + 1)
            if spin is not None:                                          # spin snapshots (same atom order) for magnetisation / spin correlations
                sp = np.asarray(spin).astype(np.float16); spin_snaps.append(sp[:, inv] if inv is not None else sp)
        if (c + 1) % anchor_every == 0 or c == n_chunks - 1:
            # re-anchor: full recompute of the carried energy (float32 drift), rebuild caches.  The old caches are
            # released FIRST (otherwise old + new = 2x the resident state during the rebuild; 10M atoms OOM on 48 GB)
            cache = None
            cache = full(sig) if spin is None else full(sig, spin)
            E_full = jax.vmap(lambda cc: mc['F']['energy_of'](cc, mc['P']))(cache)
            drift = float(jnp.abs(E_full - E).max()); E = E_full
        E_eV = np.asarray(E) * mc['ys'] + mc['ym']
        E_chunk.append(E_eV); acc_chunk.append(np.asarray(nacc) / chunk); drift_log.append(drift)
        traj.append(np.asarray(Es[:, ::sample_interval]) * mc['ys'] + mc['ym'])
        if exchange_every > 0 and R > 1:                                # PT: neighbour swaps of configurations
            for _ in range(max(chunk // exchange_every, 1)):
                kx, ku = jax.random.split(kx)
                start = int(np.asarray(jax.random.randint(kx, (), 0, 2)))
                u = np.asarray(jax.random.uniform(ku, (R,)))
                E_np = np.asarray(E) * mc['ys'] * mc['n_atoms']; b = np.asarray(betas)          # only the R energies leave the device
                order = list(range(R))
                for i in range(start, R - 1, 2):
                    n_ex_try += 1
                    d = (b[i] - b[i + 1]) * (E_np[i] - E_np[i + 1])      # PT exchange: exp[(beta_i - beta_j)(E_i - E_j)] (same as CE/pt_mcmc_ce)
                    if d >= 0 or u[i] < np.exp(d):
                        order[i], order[i + 1] = order[i + 1], order[i]; n_ex_acc += 1
                order = jnp.asarray(order)
                sig = sig[order]; cache = jax.tree_util.tree_map(lambda x: x[order], cache); E = E[order]
                if spin is not None: spin = spin[order]
        if log_every and ((c + 1) % max(log_every // chunk, 1) == 0 or c == n_chunks - 1):
            _log(f"[{tag}] chunk {c+1}/{n_chunks}  E(eV/atom) mean {E_eV.mean():.6f} [{E_eV.min():.6f},{E_eV.max():.6f}]  "
                 f"acc {np.mean(acc_chunk[-1]):.3f}  drift {drift:.1e}  {R*chunk*(c+1-c0)/(time.time()-t0):,.0f} steps/s"
                 + (f"  exch {n_ex_acc}/{n_ex_try}" if exchange_every else ""))
        if ck_every and (c + 1) % ck_every == 0 and c + 1 < n_chunks:   # right after a re-anchor (ck_every is a multiple of anchor_every)
            os.makedirs(out_dir, exist_ok=True)
            traj = [np.concatenate(traj, axis=1)] if traj else []
            np.savez(ck_path + ".tmp.npz", chunk=c + 1, sig=np.asarray(sig), spin=(np.zeros(0) if spin is None else np.asarray(spin)),
                     key=np.asarray(key), drift=drift, n_ex_try=n_ex_try, n_ex_acc=n_ex_acc,
                     E_chunk=np.array(E_chunk), acc_chunk=np.array(acc_chunk), drift_log=np.array(drift_log, np.float64),
                     traj=(traj[0] if traj else np.zeros((R, 0))),
                     snaps=(np.stack(snaps) if snaps else np.zeros((0, R, mc['n_atoms']), np.int8)), snap_chunks=np.array(snap_chunks, np.int64),
                     spin_snaps=(np.stack(spin_snaps) if spin_snaps else np.zeros((0, R, mc['n_atoms']), np.float16)),
                     cfg_betas=np.asarray(betas, np.float64), **{"cfg_" + k: v for k, v in want.items()})
            os.replace(ck_path + ".tmp.npz", ck_path)
    os.makedirs(out_dir, exist_ok=True)
    E_chunk = np.array(E_chunk); acc_chunk = np.array(acc_chunk); traj = np.concatenate(traj, axis=1)
    np.savez(f"{out_dir}/energies.npz", E_chunk=E_chunk, acc_chunk=acc_chunk, drift=np.array(drift_log),
             drift_eV=np.array(drift_log) * mc['ys'],          # the same drift in eV/atom (drift_log is in the model's normalised units)
             traj=traj, betas=np.asarray(betas), temps=1.0 / (KB * np.asarray(betas)))
    sig_np = np.asarray(sig); spin_np = None if spin is None else np.asarray(spin)
    if mc.get('implicit'):                                            # canonical -> input atom order
        sig_np = sig_np[:, mc['inv_perm']]; spin_np = None if spin_np is None else spin_np[:, mc['inv_perm']]
    np.save(f"{out_dir}/final_sigma.npy", sig_np)
    if spin_np is not None: np.save(f"{out_dir}/final_spin.npy", spin_np)
    if snaps: np.savez_compressed(f"{out_dir}/snapshots.npz", sigma=np.stack(snaps), chunk=np.asarray(snap_chunks), chunk_steps=int(chunk), temps=1.0 / (KB * np.asarray(betas)),
                                  **({"spin": np.stack(spin_snaps)} if spin_snaps else {}))
    if mc['n_atoms'] <= int(os.environ.get("CE_CNN_CIF_MAX", "200000")):
        _write_cifs(mc, sig_np, np.asarray(1.0 / (KB * betas)), out_dir, spin_np)
    else:
        _log(f"[{tag}] {mc['n_atoms']} atoms: final structures kept as final_sigma.npy (input atom order; species = "
             f"sorted-by-Z rank within the sublattice), no cifs (CE_CNN_CIF_MAX)")
    if ck_every and os.path.exists(ck_path):
        os.remove(ck_path)                                            # a finished run leaves no checkpoint
    return dict(steps_per_s=float(R * chunk * (n_chunks - c0) / (time.time() - t0)), n_replicas=R, n_steps=int(n_steps), chunk=int(chunk),
                E_final=[float(x) for x in E_chunk[-1]], acc_mean=float(acc_chunk.mean()), drift_max=float(np.max(drift_log)),
                exchange_acc=(n_ex_acc / n_ex_try if n_ex_try else None), model_depth=int(mc['depth']),
                n_atoms=int(mc['n_atoms']))


def _write_cifs(mc, sig_np, temps, out_dir, spin_np=None):
    from ase.io import write
    from ase.data import atomic_numbers, chemical_symbols
    cs = mc['cs']; atoms = mc['atoms']
    if mc.get('implicit'):
        geo = mc['geo']; maps = [sorted(m, key=lambda z: m[z]) for m in geo['maps_sl']]   # Z list ordered by species index
        sl_of_site = geo['b_sl'][geo['basis']]; act = geo['active_orig']
    else:
        T = mc['T']; sub = cs.get_sublattices(atoms)
        maps = [[atomic_numbers[x] for x in sorted(sl.chemical_symbols, key=lambda x: atomic_numbers[x])] for sl in sub]
        sl_of_site = np.array([sub.get_sublattice_index_from_site_index(int(i)) for i in range(len(atoms))]); act = T['active']
    for r in range(sig_np.shape[0]):
        a = atoms.copy(); Z = a.numbers.copy()
        for i in act:
            Z[i] = maps[int(sl_of_site[i])][int(sig_np[r, i])]
        a.numbers = Z
        write(f"{out_dir}/final_T{temps[r]:.0f}_r{r}.cif", a)
        if spin_np is not None:                                  # cif has no spin column: extxyz carries the moments
            a.set_initial_magnetic_moments(spin_np[r]); write(f"{out_dir}/final_T{temps[r]:.0f}_r{r}.extxyz", a, format="extxyz")


def run_mc(model_path, init_cif, n_steps=10000, temperature=300.0, seed=0, log_every=0, out_dir=".",
           n_replicas=1, chunk=1000, sample_interval=100, supercell=(1, 1, 1), seed_idx=0, dtype="float32",
           spin_init="random", p_spin=0.5, anchor_every=None, init_npz=None, checkpoint_every=0, spin_allowed=None, **_):
    mc = _prepare(model_path, init_cif, supercell, seed_idx, dtype, spin_init=spin_init, p_spin=p_spin, seed=seed, spin_allowed=spin_allowed)
    betas = [1.0 / (KB * temperature)] * n_replicas
    res = _run_replicas(mc, betas, n_steps, seed, min(chunk, n_steps), log_every, sample_interval, out_dir, f"ce_cnn T={temperature:.0f}",
                        anchor_every=anchor_every, init_npz=init_npz, checkpoint_every=checkpoint_every)
    res.update(temperature=float(temperature)); return res


def run_multiT_mc(model_path, init_cif, temps, n_steps=10000, seed=0, log_every=0, out_dir=".",
                  chunk=1000, sample_interval=100, supercell=(1, 1, 1), seed_idx=0, dtype="float32",
                  spin_init="random", p_spin=0.5, anchor_every=None, init_npz=None, checkpoint_every=0, spin_allowed=None, **_):
    mc = _prepare(model_path, init_cif, supercell, seed_idx, dtype, spin_init=spin_init, p_spin=p_spin, seed=seed, spin_allowed=spin_allowed)
    betas = [1.0 / (KB * t) for t in temps]
    res = _run_replicas(mc, betas, n_steps, seed, min(chunk, n_steps), log_every, sample_interval, out_dir, "ce_cnn multiT",
                        anchor_every=anchor_every, init_npz=init_npz, checkpoint_every=checkpoint_every)
    res.update(temps=[float(t) for t in temps]); return res


def run_pt_mc(model_path, init_cif, n_steps=10000, T_min=100.0, T_max=2000.0, n_replicas=32, T_grid=None,
              exchange_every=100, seed=0, log_every=0, out_dir=".", chunk=1000, sample_interval=100,
              supercell=(1, 1, 1), seed_idx=0, dtype="float32", spin_init="random", p_spin=0.5, anchor_every=None,
              init_npz=None, checkpoint_every=0, spin_allowed=None, **_):
    mc = _prepare(model_path, init_cif, supercell, seed_idx, dtype, spin_init=spin_init, p_spin=p_spin, seed=seed, spin_allowed=spin_allowed)
    temps = np.asarray(T_grid, float) if T_grid is not None else np.geomspace(T_min, T_max, n_replicas)
    betas = [1.0 / (KB * t) for t in temps]
    res = _run_replicas(mc, betas, n_steps, seed, min(chunk, n_steps), log_every, sample_interval, out_dir, "ce_cnn PT",
                        exchange_every=exchange_every, anchor_every=anchor_every, init_npz=init_npz, checkpoint_every=checkpoint_every)
    res.update(temps=[float(t) for t in temps], exchange_every=int(exchange_every)); return res
