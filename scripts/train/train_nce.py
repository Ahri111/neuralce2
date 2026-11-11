"""Neural cluster expansion (NeuralCE) trainer: nested-CV evaluation and final models for STFO / PtNi / FeNiCr(+spin).

Only the PtNi data set (SYSTEM=nipt, the default) is shipped with this repository.  The STFO and FeNiCr code paths are
kept for reference: they read data/new_bench and data/feni_cr, which are not included, and will fail at the data step.

Three modes (selected by env):
  default          nested 10-fold CV of the cluster convolution: (depth, weight decay) chosen INSIDE each outer
                   training set (inner 3-fold), one run per network seed, linear baselines on the same folds.
                   Writes $RUNS/nce_train/<TAG>/{features.npz, oof_nested_seed*.npy, oof_nested_ensemble.npy, nested.json}.
  FINAL=1          final model for Monte Carlo: (depth, wd) by 3-fold inner CV on all data, long training with a
                   held-out validation split, refit on all data, seeds ordered by validation MAE.
                   Writes $MODELS/nce/<TAG>/model.pkl (the path is printed).
  OOF_FIXED=1      10-fold out-of-fold predictions with the FINAL recipe fixed (no inner CV, no early stopping).
                   Writes $RUNS/nce_train/<TAG>/{oof_fixed.npy, oof_fixed_y.npy}.

Systems:
  SYSTEM=stfo      data/new_bench (389 x 160-atom perovskite, Sr inert) or data/new_bench_aug467 (STFO_SET=aug467),
                   icet cutoffs STFO_CUT, StratifiedKFold(10) on comp (seed 42), conv shells STFO_SHELLS.
                   Primitive/sublattices from models/ce/stfo/icet_q56.ce; structures from data/<set>/mapped.pkl
                   (scripts/train/map_structures.py or fit_ce_stfo.py).
  SYSTEM=nipt      data/nipt/icet_csv_FINAL_dedup.csv (121 labels), relaxed cifs from $NIPT_CIF_DIR (default
                   data/nipt/cifs) snapped onto the ideal a=3.924 fcc lattice, StratifiedKFold on n_ni with
                   single-member anchors pinned to train, icet cutoffs CE_CUT.
  SYSTEM=fenicr    data/feni_cr (516 rows that carry the spin design), icet [6,5], decoration GroupKFold(10) from
                   data/feni_cr/folds_decoration.json, structures from data/feni_cr/mapped.pkl, slices all / >=32 / ==108.
                   SPIN>=1 appends spin site features (Heisenberg pairs, node planes, spin x environment planes).

Env knobs (defaults in parentheses):
  SYSTEM (nipt) | SPIN (1, fenicr only: 0 none, 1 Heisenberg-pair, 2 +spin planes, 3 +spin x env) | SELECT (0; 1 = keep
  only the clusters that survive a full-data ARDR fit) | AGG (mean+max; mean = ablation for MC speed) | GATE (1) |
  WIDTH (16) | LR (3e-3) | EPOCHS (6000, cap; early stopping with patience 300 decides) | SEEDS (0,1,2) |
  LIMIT (0; >0 = smoke subset) | OUT_TAG (derived from SYSTEM/SPIN/AGG/GATE/SELECT/LIMIT)
  STFO_CUT (6,5,4.5,4) | STFO_SHELLS (2.41,3.135,3.705,4.08) | STFO_SET (new_bench | aug467) | STFO_DEDUP (1) |
  STFO_LABELFIX (0; 1 = apply data/new_bench/label_corrections.json [+ label_corrections_al.json for aug467]) |
  EXTRA (none; pickle {atoms, y, comp, ids} of extra labelled 160-atom STFO structures appended to the set) |
  CE_CUT (7,5,4.5; nipt icet cutoffs) | FENICR_SPIN_NSHELL (0; 1|2|3 = shells carrying the spin planes) |
  FENICR_SPIN_NODE_LO (0; SPIN>=2: drop the alignment planes of the first shells)
  FINAL (0) | FINAL_DEPTH (inner-CV choice) | FINAL_WD (inner-CV choice) | FINAL_STEPS (20000) | FINAL_PATIENCE (1500) |
  FINAL_VAL_FRAC (0.1) | FINAL_REFIT_STEPS (0 = 1.1 x best validation step) | WEIGHT_T (0; meV, near-ground-state loss
  weights exp(-dE/WEIGHT_T) per composition) | COMP_W_TARGET ("" e.g. Fe34Cr20Ni46) | COMP_W_SIGMA (0.12) |
  COMP_W_FLOOR (0.1) | LASSO_RULE (min | 1se) | LASSO_ALPHA_MULT (1) | OOF_FIXED (0)

Run from the repository root:  SYSTEM=nipt python scripts/train/train_nce.py
"""
import os, sys, json, time, ast, pickle, numpy as np, pandas as pd
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV, LassoCV, ARDRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from icet import ClusterSpace, ClusterExpansion
from neuralce.paths import DATA, MODELS, RUNS
from neuralce.nce.general import (local_cluster_features, gate_layer0, shell_partners, enlarge_if_self_interacting, spin_node_planes, spin_env_planes,
                                  spin_site_features, pack, init_params, make_forward, train)

SYSTEM = os.environ.get("SYSTEM", "nipt")                # only the PtNi data set is shipped
SPIN = int(os.environ.get("SPIN", "1")) if SYSTEM == "fenicr" else 0   # 0 none, 1 Heisenberg-pair, 2 +spin planes, 3 +spin x env
LIMIT = int(os.environ.get("LIMIT", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "6000"))   # cap only; early stopping (patience 300) decides the actual length
NET_SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2").split(","))
GRID = [(0, 0.0)] + [(d, w) for d in (1, 2) for w in (1e-4, 1e-3)]   # depth 0 = CE fine-tuned by Adam
# (no plain-lasso candidate / CE fallback: the network is the model)
AGG = os.environ.get("AGG", "mean+max")          # shell aggregation: "mean+max" (v3 default) or "mean" (ablation for MC speed)
GATE = int(os.environ.get("GATE", "1"))
VERSION = f"v3-skip+earlystop+ensemble+{AGG}+gate{GATE}"
PATIENCE, EVAL_EVERY = 300, 25
WIDTH, LR, SEED = int(os.environ.get("WIDTH", "16")), float(os.environ.get("LR", "3e-3")), 42
TAG = os.environ.get("OUT_TAG", SYSTEM + {0: "", 1: "_spin", 2: "_spin2", 3: "_spin3"}[SPIN] + "_v3"
                     + ("_nomax" if os.environ.get("AGG", "mean+max") == "mean" else "") + ("_nogate" if os.environ.get("GATE", "1") == "0" else "")
                     + ("_sel" if int(os.environ.get("SELECT", "0")) else "") + ("_smoke" if LIMIT else ""))
OUT = f"{RUNS}/nce_train/{TAG}"; os.makedirs(OUT, exist_ok=True)
t_start = time.time()


def log(*a):
    print(*a, flush=True)


# ── data ───────────────────────────────────────────────────────────────────
if SYSTEM == "stfo":
    CUT = [float(c) for c in os.environ.get("STFO_CUT", "6,5,4.5,4").split(",")]
    SHELL_EDGES = [0.0] + [float(c) for c in os.environ.get("STFO_SHELLS", "2.41,3.135,3.705,4.08").split(",")]   # NeuralCE STFO graph A_4.08_s4 by default
    ce0 = ClusterExpansion.read(f"{MODELS}/ce/stfo/icet_q56.ce")
    cs0 = ce0.get_cluster_space_copy()
    cs = ClusterSpace(cs0.primitive_structure, CUT, cs0.chemical_symbols)
    # STFO_SET=aug467: the 389 benchmark structures + 78 active-learning structures (data/new_bench_aug467, mapped by
    # scripts/train/map_structures.py).  Dedup and label corrections below apply by id, so they carry over.
    STFO_SET = os.environ.get("STFO_SET", "new_bench")
    if STFO_SET == "aug467":
        df = pd.read_csv(f"{DATA}/new_bench_aug467/detailed_info.csv")
        atoms = pickle.load(open(f"{DATA}/new_bench_aug467/mapped.pkl", "rb"))
        log(f"[stfo] data set aug467: {len(df)} structures ({int(df.id.astype(str).str.startswith('AL_').sum())} AL)")
    else:
        df = pd.read_csv(f"{DATA}/new_bench/detailed_info.csv")
        atoms = pickle.load(open(f"{DATA}/new_bench/mapped.pkl", "rb"))
    assert len(atoms) == len(df)
    if int(os.environ.get("STFO_DEDUP", "1")):                 # true duplicates (data/new_bench/dedup_stfo.json): keep one with the group-mean label
        dd = json.load(open(f"{DATA}/new_bench/dedup_stfo.json")); df["y"] = df.total_energy / df.n_atoms
        for grp in dd["groups"]:
            df.loc[df.id.isin(grp), "y"] = df.loc[df.id.isin(grp), "y"].mean()
        keep_rows = ~df.id.isin(dd["drop"]); atoms = [a for a, k in zip(atoms, keep_rows) if k]; df = df[keep_rows].reset_index(drop=True)
        df["total_energy"] = df.y * df.n_atoms
        log(f"[stfo] dedup: dropped {dd['drop']} -> {len(df)} structures (group-mean labels)")
    if int(os.environ.get("STFO_LABELFIX", "0")):              # DFT re-check corrections (stfo_label_corrections.py): lower FM-protocol minimum replaces the label
        fx = dict(json.load(open(f"{DATA}/new_bench/label_corrections.json"))["corrections"]); n_fix = 0
        if STFO_SET == "aug467":                                # the AL structures' own re-check corrections (same rule)
            fx.update(json.load(open(f"{DATA}/new_bench_aug467/label_corrections_al.json"))["corrections"])
        for sid, e_new in fx.items():
            m = df.id.astype(str) == sid
            if m.any(): df.loc[m, "total_energy"] = float(e_new) * df.loc[m, "n_atoms"]; n_fix += int(m.sum())
        log(f"[stfo] label corrections applied to {n_fix} structures (data/new_bench/label_corrections.json)")
    y = (df.total_energy / df.n_atoms).to_numpy(); n_at = df.n_atoms.to_numpy()
    strat = df.comp.astype(int).to_numpy(); ids = df.id.astype(str).tolist()
    spins = [None] * len(df)
    if os.environ.get("EXTRA"):                               # extra labelled structures (e.g. comp-250 motifs) appended to the set
        ex = pickle.load(open(os.environ["EXTRA"], "rb"))
        atoms = list(atoms) + list(ex["atoms"]); y = np.concatenate([y, np.asarray(ex["y"], float)])
        n_at = np.concatenate([n_at, np.full(len(ex["y"]), 160)]); strat = np.concatenate([strat, np.asarray(ex["comp"], int)])
        ids = ids + list(ex["ids"]); spins = spins + [None] * len(ex["y"])
        log(f"[stfo] +{len(ex['y'])} extra structures from {os.environ['EXTRA']}: {list(ex['ids'])}")
    active_of = lambda a: np.where(np.array(a.get_chemical_symbols()) != "Sr")[0]
    fold_of = None
elif SYSTEM == "nipt":
    # PtNi rules: labels = icet_csv_FINAL_dedup.csv (121), StratifiedKFold on n_ni with single-member anchors pinned
    # to train; relaxed CIFs snapped onto the ideal a=3.924 fcc lattice.
    from ase.build import bulk
    from neuralce.nce.features import field_from_atoms, atoms_from_field, CIF_DIR as NIPT_CIF_DIR
    from ase.io import read as ase_read
    CUT = [float(x) for x in os.environ.get("CE_CUT", "7,5,4.5").split(",")]
    SHELL_EDGES = [0.0, 3.3, 4.3, 5.2]                          # fcc a=3.924: 1NN 2.775, 2NN 3.924, 3NN 4.806
    cs = ClusterSpace(bulk("Ni", "fcc", a=3.924), CUT, [["Ni", "Pt"]])
    df = pd.read_csv(f"{DATA}/nipt/icet_csv_FINAL_dedup.csv")
    atoms = [atoms_from_field(field_from_atoms(ase_read(f"{NIPT_CIF_DIR}/{i}.cif"))[0]) for i in df.id.astype(str)]
    y = (df.total_energy / df.n_atoms).to_numpy(); n_at = df.n_atoms.to_numpy()
    strat = df.n_ni.to_numpy(); ids = df.id.astype(str).tolist()
    spins = [None] * len(df)
    active_of = lambda a: np.arange(len(a))
    fold_of = None
else:
    from neuralce.ce.spin_pairs import reembed_to_reference_frame, pbc_perm_raw_to_mapped, spin_features, PAIRS
    from neuralce.ce.primitive_no_idealize import detect_primitive_no_idealize
    from ase.io import read as ase_read
    CUT = [6.0, 5.0]
    SHELL_EDGES = [0.0, 3.075, 3.65, 4.5]                    # conv shells: 1NN, 2NN, 3NN
    # Shells that carry the spin pair (and, for SPIN>=2, node) planes.  Defaults keep the published behaviour:
    # SPIN=1 -> 1NN+2NN (the icet+spin Heisenberg config), SPIN>=2 -> every conv shell.
    # FENICR_SPIN_NSHELL=1|2|3 overrides (1 = 1NN only).  It always names a prefix of SHELL_EDGES, which the MC needs:
    # the spin channels reuse the conv shell tables (mc_model._n_spin_shells asserts it).
    _ns = int(os.environ.get("FENICR_SPIN_NSHELL", "0") or 0)          # 1 = 1NN only, 2 = 1NN+2NN, 3 = all conv shells
    SPIN_EDGES = SHELL_EDGES[:_ns + 1] if _ns else ([0.0, 3.075, 3.65] if SPIN == 1 else SHELL_EDGES)
    NODE_LO = int(os.environ.get("FENICR_SPIN_NODE_LO", "0") or 0)     # SPIN>=2: drop the alignment planes of the first NODE_LO shells
    CIF_DIR = f"{DATA}/feni_cr"
    df = pd.read_csv(f"{CIF_DIR}/detailed_info.csv")
    spec = json.load(open(f"{DATA}/feni_cr/folds_decoration.json"))
    df = df[~df.id.astype(str).isin(set(map(str, spec["dropped_exact_duplicates"])))].reset_index(drop=True)
    atoms_all = pickle.load(open(f"{DATA}/feni_cr/mapped.pkl", "rb"))
    assert len(atoms_all) == len(df)
    z = np.load(f"{DATA}/feni_cr/design_matrix.npz", allow_pickle=True)
    keep_ids = set(str(i) for i in z["ids"])                 # rows that carry the spin design
    sel = np.array([k for k, i in enumerate(df.id.astype(str)) if i in keep_ids])
    df = df.iloc[sel].reset_index(drop=True); atoms = [atoms_all[k] for k in sel]
    y = (df.total_energy / df.n_atoms).to_numpy(); n_at = df.n_atoms.to_numpy()
    ids = df.id.astype(str).tolist(); strat = df.comp.astype(str).to_numpy()
    fold_of = np.array([int(spec["folds"][i]) for i in ids])
    ids32 = [i for i, n in zip(ids, n_at) if n == 32]
    PRIM = detect_primitive_no_idealize(f"{CIF_DIR}/{ids32[0]}.cif", {"Ni": "Fe", "Cr": "Fe"}, 1e-2)
    cs = ClusterSpace(PRIM, CUT, [["Fe", "Ni", "Cr"]])
    # spins: raw cif order -> mapped order (same path as neuralce.ce.spin_pairs)
    spins = []
    for i, a in zip(ids, atoms):
        raw = reembed_to_reference_frame(ase_read(f"{CIF_DIR}/{i}.cif"), PRIM.cell.array)
        s_raw = np.array(ast.literal_eval(df.spins[ids.index(i)]), float)
        perm = pbc_perm_raw_to_mapped(raw, a); s_m = np.empty(len(a)); s_m[perm] = s_raw
        spins.append(s_m)
    active_of = lambda a: np.arange(len(a))

if LIMIT:
    if SYSTEM == "fenicr":                                    # smoke on a mix of cell sizes
        sel = np.concatenate([np.where(m)[0][:max(LIMIT // 3, 1)] for m in (n_at < 32, n_at == 32, n_at == 108)])
    else:
        sel = np.arange(len(y))[:LIMIT]
    atoms = [atoms[k] for k in sel]; y = y[sel]; n_at = n_at[sel]; strat = strat[sel]
    ids = [ids[k] for k in sel]; spins = [spins[k] for k in sel]
    if fold_of is not None: fold_of = fold_of[sel]
N = len(y)
log(f"[{SYSTEM}] {N} structures  K={len(cs)}  cutoffs {CUT}  shells {SHELL_EDGES}  spin={SPIN}  device={jax.devices()[0].platform}")

# ── layer 0 + shells (icet-gated) ──────────────────────────────────────────
feats, partners, gate_err, sp_err = [], [], 0.0, 0.0
t0 = time.time()
n_rep = 0
for k, a0 in enumerate(atoms):
    cv0 = cs.get_cluster_vector(a0)
    a, spins[k], r, h0 = enlarge_if_self_interacting(cs, a0, active_of, spins[k])
    if r != (1, 1, 1): n_rep += 1
    act = active_of(a)
    gate_err = max(gate_err, gate_layer0(cs, a, act, h0), float(np.abs(h0.mean(0) - cv0).max()))
    if SPIN == 1:                                             # Heisenberg-pair (2 shells x 6 pairs), as in icet+spin
        f = spin_site_features(a, act, spins[k], SPIN_EDGES, PAIRS)
        nc = min(f.shape[1], 12)                                       # icet+spin reference covers 2 shells x 6 pairs
        sp_err = max(sp_err, np.abs(f.mean(0)[:nc] - spin_features(a, spins[k])[:nc]).max())
        h0 = np.hstack([h0, f])
    elif SPIN >= 2:                                           # spin field through the SAME kernels as the chemistry:
        f = spin_site_features(a, act, spins[k], SPIN_EDGES, PAIRS)    # pair kernels x species pair (P_spin shells x 6)
        nc = min(f.shape[1], 12)                                                          # first 2 shells == icet+spin
        sp_err = max(sp_err, np.abs(f.mean(0)[:nc] - spin_features(a, spins[k])[:nc]).max())
        h0 = np.hstack([h0, f, spin_node_planes(a, act, spins[k], SPIN_EDGES, NODE_LO)])  # + s_i^2, shell alignment planes
        if SPIN == 3:                                         # spin pair x 1NN composition of the site (3-body, 54 ch)
            h0 = np.hstack([h0, spin_env_planes(a, act, spins[k], SPIN_EDGES, PAIRS, ["Fe", "Ni", "Cr"])])
    feats.append(h0); partners.append(shell_partners(a, act, SHELL_EDGES))
    if (k + 1) % 100 == 0: log(f"  features {k+1}/{N}  ({time.time()-t0:.0f}s)")
log(f"[GATE] layer0: max|mean_i h0 - icet cv| = {gate_err:.2e}  ({n_rep} self-interacting cells repeated)" + (f"   spin: max|mean_i f - global| = {sp_err:.2e}" if SPIN else ""))
assert gate_err < 1e-9 and sp_err < 1e-9
H, M, SI, SM = pack(feats, partners)
K_CHEM = len(cs); C_FULL = H.shape[-1]
SELECT = int(os.environ.get("SELECT", "0"))
if SELECT:
    # cluster selection on full data: fit icet on ALL data (ARDR, raw features, no intercept, threshold 1e5) and keep
    # only the clusters that survive as the kernel bank.  Feature selection uses every label -> report as
    # "cluster selection on full data".  Zerolet and every spin channel are always kept.
    phi_all = (H * M[..., None]).sum(1) / M.sum(1)[:, None]
    ard = ARDRegression(max_iter=300, fit_intercept=False, threshold_lambda=1e5, compute_score=False).fit(phi_all[:, :K_CHEM], y)
    keep = np.abs(ard.coef_) > 1e-8; keep[0] = True
    keep = np.concatenate([keep, np.ones(H.shape[-1] - K_CHEM, bool)])
    log(f"[SELECT] full-data ARDR keeps {int(keep[:K_CHEM].sum())}/{K_CHEM} chemistry clusters (+{H.shape[-1]-K_CHEM} spin channels)")
    np.save(f"{OUT}/selected_channels.npy", np.where(keep)[0]); H = H[..., keep]
C = H.shape[-1]; P = SI.shape[1]
sizes = {p: sorted(set(int(x) for x in SM[:, p].sum(-1)[M > 0])) for p in range(P)}
log(f"packed H {H.shape}  shells {SI.shape}  partners per shell {sizes}  ({time.time()-t0:.0f}s)")
np.savez(f"{OUT}/features.npz", H=H, M=M, SI=SI, SM=SM, y=y, n_at=n_at, ids=np.array(ids))

# ── folds ──────────────────────────────────────────────────────────────────
if fold_of is None:
    cnt = pd.Series(strat).value_counts(); ANCH = np.isin(strat, cnt[cnt < 2].index.to_numpy())
    idx_all = np.arange(N); iev = idx_all[~ANCH]
    OUTER = [(np.concatenate([iev[a], idx_all[ANCH]]), iev[b]) for a, b in
             StratifiedKFold(10 if not LIMIT else 3, shuffle=True, random_state=SEED).split(iev, strat[~ANCH])]
    def inner_split(tr, f):
        return list(StratifiedKFold(3, shuffle=True, random_state=100 + f).split(tr, strat[tr]))
else:
    F = sorted(set(fold_of.tolist()))
    OUTER = [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in F if (fold_of == f).any()]
    def inner_split(tr, f):                                   # group the remaining folds 3-ways
        ftr = fold_of[tr]; rest = sorted(set(ftr.tolist())); out = []
        for g in range(3):
            hold = set(rest[g::3]); m = np.isin(ftr, list(hold))
            if m.any() and (~m).any(): out.append((np.where(~m)[0], np.where(m)[0]))
        return out


def rmse(p, t): return float(np.sqrt(np.mean(((p - t) * 1000) ** 2)))


def mae(p, t): return float(np.mean(np.abs(p - t)) * 1000)


def slices(oof):
    ok = ~np.isnan(oof); d = {"all": rmse(oof[ok], y[ok]), "all_mae": mae(oof[ok], y[ok])}
    if SYSTEM == "fenicr":
        for nm, m in (("ge32", n_at >= 32), ("eq108", n_at == 108)):
            mm = ok & m; d[nm] = rmse(oof[mm], y[mm]) if mm.any() else None; d[nm + "_mae"] = mae(oof[mm], y[mm]) if mm.any() else None
    return d


# ── fitting ────────────────────────────────────────────────────────────────
fwd = make_forward()


def standardise(tr):
    Htr = H[tr]; m = M[tr] > 0
    mu = Htr[m].mean(0); sd = Htr[m].std(0) + 1e-8
    return mu, sd


def fit_predict(tr, te, depth, wd, seed, early=False, epochs=None):
    """-> (prediction on te, steps used).  early=True: te doubles as the early-stopping validation set
    (inner folds only).  depth == "lasso": the plain linear model, no network."""
    mu, sd = standardise(tr); ym, ys = y[tr].mean(), y[tr].std() + 1e-12
    Xtr = jnp.asarray(((H[tr] - mu) / sd) * M[tr][..., None]); Xte = jnp.asarray(((H[te] - mu) / sd) * M[te][..., None])
    phi_tr = ((H[tr] - mu) / sd * M[tr][..., None]).sum(1) / M[tr].sum(1)[:, None]
    phi_te = ((H[te] - mu) / sd * M[te][..., None]).sum(1) / M[te].sum(1)[:, None]
    # CE term = lasso on the training fold (site-mean features)
    rg = LassoCV(alphas=np.logspace(-6, 0, 31), cv=min(5, len(tr)), max_iter=50000).fit(phi_tr, (y[tr] - ym) / ys)
    if depth == "lasso":
        return rg.predict(phi_te) * ys + ym, 0
    P0 = init_params(jax.random.PRNGKey(seed), C, P, WIDTH, depth, lin_w=rg.coef_, lin_b=rg.intercept_, agg=AGG, gate=bool(GATE))
    Dte = (Xte, jnp.asarray(M[te]), jnp.asarray(SI[te]), jnp.asarray(SM[te]))
    Pf, steps = train(fwd, P0, (Xtr, jnp.asarray(M[tr]), jnp.asarray(SI[tr]), jnp.asarray(SM[tr])),
                      jnp.asarray((y[tr] - ym) / ys), wd, epochs=(epochs or EPOCHS), lr=LR,
                      val=(Dte + (jnp.asarray((y[te] - ym) / ys),)) if early else None,
                      eval_every=EVAL_EVERY, patience=PATIENCE)
    return np.asarray(fwd(Pf, *Dte)) * ys + ym, steps


if int(os.environ.get("FINAL", "0")):
    # ── FINAL MODEL for MCMC: (depth, wd) by 3-fold inner CV on ALL data, then train on all data ──
    allidx = np.arange(N); inner = inner_split(allidx, 0); scores = {}
    for (depth, wd) in GRID:
        errs, steps = [], []
        for ia, ib in inner:
            pr, st = fit_predict(allidx[ia], allidx[ib], depth, wd, 0, early=True)
            errs.append(rmse(pr, y[allidx[ib]])); steps.append(st)
        scores[(depth, wd)] = (float(np.mean(errs)), int(np.clip(np.median(steps), 200, EPOCHS)))
        log(f"  [FINAL] depth={depth} wd={wd:.0e}: inner RMSE {scores[(depth, wd)][0]:.3f}  steps {scores[(depth, wd)][1]}")
    best = min(scores, key=lambda k: scores[k][0]); depth, wd = best; steps = scores[best][1]
    if os.environ.get("FINAL_DEPTH"): depth = int(os.environ["FINAL_DEPTH"])
    if os.environ.get("FINAL_WD") is not None: wd = float(os.environ["FINAL_WD"])
    log(f"  [FINAL] chosen depth={depth} wd={wd:.0e} (inner-CV steps {steps})")
    # LONG training: hold out FINAL_VAL_FRAC of the data, train up to FINAL_STEPS with patience FINAL_PATIENCE,
    # keep the best-validation step, then refit on ALL data for that many steps.
    F_STEPS = int(os.environ.get("FINAL_STEPS", "20000")); F_PAT = int(os.environ.get("FINAL_PATIENCE", "1500"))
    F_VAL = float(os.environ.get("FINAL_VAL_FRAC", "0.1"))
    mu, sd = standardise(allidx); ym, ys = y.mean(), y.std() + 1e-12
    X = jnp.asarray(((H - mu) / sd) * M[..., None]); phi_all = ((H - mu) / sd * M[..., None]).sum(1) / M.sum(1)[:, None]
    # optional near-ground-state emphasis: w_i = exp(-(E_i - E_min(comp)) / WEIGHT_T [meV/atom]), normalised to mean 1
    W_T = float(os.environ.get("WEIGHT_T", "0"))
    if W_T > 0:
        dEc = np.array([y[k] - y[strat == strat[k]].min() for k in range(N)]) * 1e3
        w_all = np.exp(-dEc / W_T); w_all = w_all / w_all.mean()
        log(f"  [FINAL] loss weights exp(-dE/{W_T:g} meV): min {w_all.min():.3f} max {w_all.max():.3f}")
    else:
        w_all = np.ones(N)
    # optional composition emphasis: w_i *= floor + (1-floor) * exp(-|x_i - x_target|^2 / 2 sigma^2) (site fractions),
    # for a model aimed at one alloy composition (e.g. COMP_W_TARGET=Fe34Cr20Ni46 COMP_W_SIGMA=0.12 COMP_W_FLOOR=0.1)
    COMP_T = os.environ.get("COMP_W_TARGET", "")
    if COMP_T:
        import re as _re
        _t = dict((e, float(v)) for e, v in _re.findall(r"([A-Z][a-z]?)([\d.]+)", COMP_T)); _el = sorted(_t)
        _tv = np.array([_t[e] for e in _el]) / sum(_t.values())
        _xc = np.array([[np.mean(np.array(atoms[k].get_chemical_symbols())[active_of(atoms[k])] == e) for e in _el] for k in range(N)])
        _sg = float(os.environ.get("COMP_W_SIGMA", "0.12")); _fl = float(os.environ.get("COMP_W_FLOOR", "0.1"))
        _dist = np.linalg.norm(_xc - _tv, axis=1); _wc = _fl + (1 - _fl) * np.exp(-_dist ** 2 / (2 * _sg ** 2))
        w_all = w_all * _wc; w_all = w_all / w_all.mean()
        log(f"  [FINAL] composition weights toward {COMP_T} (sigma {_sg:g}, floor {_fl:g}): n within 0.10 = {(_dist < 0.1).sum()}, "
            f"effective n {w_all.sum()**2 / np.sum(w_all**2):.0f}/{N}")
    rg = LassoCV(alphas=np.logspace(-6, 0, 31), cv=5, max_iter=50000).fit(phi_all, (y - ym) / ys)
    # LASSO_RULE=1se: the largest alpha whose 5-fold CV error is within one standard error of the minimum (the usual
    # sparser-model rule); LASSO_ALPHA_MULT scales the chosen alpha.  Both default to the plain CV minimum.
    L_RULE = os.environ.get("LASSO_RULE", "min"); L_MULT = float(os.environ.get("LASSO_ALPHA_MULT", "1"))
    if L_RULE != "min" or L_MULT != 1.0:
        from sklearn.linear_model import Lasso
        mse = rg.mse_path_.mean(1); se = rg.mse_path_.std(1) / np.sqrt(rg.mse_path_.shape[1]); k = int(np.argmin(mse))
        a_sel = float(rg.alphas_[k])
        if L_RULE == "1se":
            a_sel = float(rg.alphas_[np.where(mse <= mse[k] + se[k])[0]].max())
        a_sel *= L_MULT; a_min = float(rg.alpha_)
        rg = Lasso(alpha=a_sel, max_iter=50000).fit(phi_all, (y - ym) / ys)
        log(f"  [FINAL] lasso alpha: CV minimum {a_min:.3g} -> {L_RULE} x {L_MULT:g} = {a_sel:.3g} ({int((rg.coef_ != 0).sum())} nonzero)")
    from sklearn.model_selection import train_test_split
    if fold_of is None:                                       # anchors (single-member strata) stay in train, like the folds
        _cnt = pd.Series(strat).value_counts(); _anch = np.isin(strat, _cnt[_cnt < 2].index.to_numpy())
        _iev = allidx[~_anch]
        tr_i, va_i = train_test_split(_iev, test_size=F_VAL, random_state=SEED, stratify=strat[_iev])
        tr_i = np.concatenate([tr_i, allidx[_anch]])
    else:                                                     # grouped data (FeNiCr decorations): validation = outer fold 0
        va_i = allidx[fold_of == sorted(set(fold_of.tolist()))[0]]; tr_i = allidx[fold_of != sorted(set(fold_of.tolist()))[0]]
    D = lambda ii: (X[ii], jnp.asarray(M[ii]), jnp.asarray(SI[ii]), jnp.asarray(SM[ii]))
    seeds = list(NET_SEEDS); params = []; val_maes = []; best_steps = []; train_preds = []
    for seed in seeds:
        P0 = init_params(jax.random.PRNGKey(seed), C, P, WIDTH, depth, lin_w=rg.coef_, lin_b=rg.intercept_, agg=AGG, gate=bool(GATE))
        Pv, st = train(fwd, P0, D(tr_i), jnp.asarray((y[tr_i] - ym) / ys), wd, epochs=F_STEPS, lr=LR,
                       val=D(va_i) + (jnp.asarray((y[va_i] - ym) / ys),), eval_every=EVAL_EVERY, patience=F_PAT, w_tr=w_all[tr_i])
        pv = np.asarray(fwd(Pv, *D(va_i))) * ys + ym; ptr = np.asarray(fwd(Pv, *D(tr_i))) * ys + ym
        log(f"  [FINAL] seed {seed}: val phase best step {st}  train MAE {mae(ptr, y[tr_i]):.3f}  val MAE {mae(pv, y[va_i]):.3f} meV/atom")
        n_refit = int(os.environ.get("FINAL_REFIT_STEPS", 0)) or int(max(st, 200) * 1.1)   # FINAL_REFIT_STEPS: fixed-length refit on all data (the conv branch needs far more than the early-stopped ~700 steps to fit)
        Pf, _ = train(fwd, P0, (X, jnp.asarray(M), jnp.asarray(SI), jnp.asarray(SM)), jnp.asarray((y - ym) / ys), wd, epochs=n_refit, lr=LR, w_tr=w_all)
        pred = np.asarray(fwd(Pf, X, jnp.asarray(M), jnp.asarray(SI), jnp.asarray(SM))) * ys + ym; train_preds.append(pred)
        log(f"  [FINAL] seed {seed}: refit on all for {n_refit} steps -> train MAE {mae(pred, y):.3f}  RMSE {rmse(pred, y):.3f} meV/atom")
        params.append(jax.tree_util.tree_map(lambda a: np.asarray(a), Pf)); val_maes.append(float(mae(pv, y[va_i]))); best_steps.append(int(st))
    order = list(np.argsort(val_maes)); params = [params[k] for k in order]; seeds = [seeds[k] for k in order]; train_preds = [train_preds[k] for k in order]
    val_maes = [val_maes[k] for k in order]; best_steps = [best_steps[k] for k in order]
    log(f"  [FINAL] seeds ordered by val MAE (best first = fixed model): {list(zip(seeds, [round(v, 3) for v in val_maes], best_steps))}")
    steps = int(best_steps[0])
    keep = np.load(f"{OUT}/selected_channels.npy") if SELECT else np.arange(C)
    model = dict(version=VERSION, system=SYSTEM, spin=SPIN, cutoffs=CUT, shell_edges=SHELL_EDGES, agg=AGG, gate=bool(GATE),
                 depth=int(depth), wd=float(wd), steps=int(steps), width=WIDTH, seeds=seeds, params=params,
                 mu=np.asarray(mu), sd=np.asarray(sd), ym=float(ym), ys=float(ys), keep=np.asarray(keep), K_chem=int(K_CHEM), C=int(C),
                 primitive=cs.primitive_structure.copy(), chemical_symbols=[list(s) for s in cs.chemical_symbols],
                 lasso_coef=np.asarray(rg.coef_), lasso_b=float(rg.intercept_), val_mae=val_maes, best_steps=best_steps,
                 train_ids=list(ids), train_y=np.asarray(y), n_train=int(N), train_pred=[np.asarray(p) for p in train_preds], strat=np.asarray(strat),
                 weight_T=W_T, lasso_rule=(os.environ.get("LASSO_RULE", "min"), float(os.environ.get("LASSO_ALPHA_MULT", "1"))), comp_w=(dict(target=COMP_T, sigma=float(os.environ.get("COMP_W_SIGMA", "0.12")), floor=float(os.environ.get("COMP_W_FLOOR", "0.1"))) if COMP_T else None),
                 # spin metadata for the MC path (FeNiCr): channel layout after the K_chem chemistry channels is
                 # [site pair-Heisenberg (P*len(pairs))] [+ node planes (1+P)] [+ env planes (P*len(pairs)*len(species))]
                 spin_pairs=(list(PAIRS) if SPIN else None), spin_species=(["Fe", "Ni", "Cr"] if SPIN else None),
                 spin_edges=(list(SPIN_EDGES) if SPIN else None), spin_node_lo=(NODE_LO if SPIN >= 2 else 0),
                 spin_values=[-1.0, 0.0, 1.0] if SPIN else None, n_spin=int(C_FULL - K_CHEM),
                 train_spins=([np.asarray(s) for s in spins] if SPIN else None))
    fdir = f"{MODELS}/nce/{TAG}"; os.makedirs(fdir, exist_ok=True)
    pickle.dump(model, open(f"{fdir}/model.pkl", "wb"))
    log(f"[FINAL] saved {fdir}/model.pkl  (depth {depth}, {len(seeds)} seeds, {int(len(keep))} channels)")
    sys.exit(0)

if int(os.environ.get("OOF_FIXED", "0")):
    # ── 10-fold out-of-fold with a FIXED recipe (the FINAL-mode recipe): depth FINAL_DEPTH, wd FINAL_WD, FINAL_REFIT_STEPS
    #    steps, optional near-GS loss weights WEIGHT_T computed inside each training fold.  No inner CV, no early stop.
    depth = int(os.environ.get("FINAL_DEPTH", "2")); wd = float(os.environ.get("FINAL_WD", "1e-3")); steps = int(os.environ["FINAL_REFIT_STEPS"])
    W_T = float(os.environ.get("WEIGHT_T", "0")); seed = NET_SEEDS[0]; oof = np.full(N, np.nan); t0 = time.time()
    for f, (tr, te) in enumerate(OUTER):
        mu, sd = standardise(tr); ym, ys = y[tr].mean(), y[tr].std() + 1e-12
        Xtr = jnp.asarray(((H[tr] - mu) / sd) * M[tr][..., None]); Xte = jnp.asarray(((H[te] - mu) / sd) * M[te][..., None])
        phi_tr = ((H[tr] - mu) / sd * M[tr][..., None]).sum(1) / M[tr].sum(1)[:, None]
        rg = LassoCV(alphas=np.logspace(-6, 0, 31), cv=5, max_iter=50000).fit(phi_tr, (y[tr] - ym) / ys)
        if W_T > 0:
            dEc = np.array([y[k] - y[tr][strat[tr] == strat[k]].min() for k in tr]) * 1e3; w = np.exp(-dEc / W_T); w = w / w.mean()
        else:
            w = None
        P0 = init_params(jax.random.PRNGKey(seed), C, P, WIDTH, depth, lin_w=rg.coef_, lin_b=rg.intercept_, agg=AGG, gate=bool(GATE))
        Pf, _ = train(fwd, P0, (Xtr, jnp.asarray(M[tr]), jnp.asarray(SI[tr]), jnp.asarray(SM[tr])), jnp.asarray((y[tr] - ym) / ys), wd, epochs=steps, lr=LR, w_tr=w)
        oof[te] = np.asarray(fwd(Pf, Xte, jnp.asarray(M[te]), jnp.asarray(SI[te]), jnp.asarray(SM[te]))) * ys + ym
        log(f"  [OOF_FIXED] fold {f}: outer MAE {mae(oof[te], y[te]):.3f} RMSE {rmse(oof[te], y[te]):.3f}  [{time.time()-t0:.0f}s]")
    np.save(f"{OUT}/oof_fixed.npy", oof); np.save(f"{OUT}/oof_fixed_y.npy", y)
    log(f"[OOF_FIXED] {TAG}: depth {depth} wd {wd:g} steps {steps} weight_T {W_T:g}: {slices(oof)}")
    if SYSTEM == "stfo":
        c = strat == 125; g = c & (y - y[c].min() < 0.003); e = (oof[g] - y[g]) * 1e3
        log(f"[OOF_FIXED] comp-125 near-GS group ({g.sum()}): OOF MAE {np.abs(e).mean():.2f} RMSE {np.sqrt((e**2).mean()):.2f} max {np.abs(e).max():.2f}; comp-125 OOF MAE {mae(oof[c], y[c]):.2f}")
    sys.exit(0)

res = {'version': VERSION, 'system': SYSTEM, 'spin': SPIN, 'select_full_data': SELECT, 'cutoffs': CUT, 'shell_edges': SHELL_EDGES, 'K': int(C),
       'n': int(N), 'grid': [list(g) for g in GRID], 'epochs': EPOCHS, 'net_seeds': list(NET_SEEDS),
       'gate_layer0': float(gate_err), 'gate_spin': float(sp_err)}

log("\n== linear baselines (site-mean features == icet cv [+ spin]) on the outer folds ==")
phi = (H * M[..., None]).sum(1) / M.sum(1)[:, None]
for name, mk in [('ridge', lambda: RidgeCV(alphas=np.logspace(-8, 4, 80))),
                 ('lasso', lambda: LassoCV(alphas=np.logspace(-8, 0, 41), cv=5, max_iter=50000)),
                 ('ardr', lambda: ARDRegression()),
                 ('ardr_raw_1e5', None)]:
    oof = np.full(N, np.nan)
    for tr, te in OUTER:
        if mk is None:
            mo = ARDRegression(max_iter=300, fit_intercept=False, threshold_lambda=1e5, compute_score=False).fit(phi[tr], y[tr])
        else:
            mo = make_pipeline(StandardScaler(), mk()).fit(phi[tr], y[tr])
        oof[te] = mo.predict(phi[te])
    res[f'lin_{name}'] = slices(oof); log(f"  {name:13s} {res[f'lin_{name}']}")

log("\n== cluster conv, nested CV ==")
for seed in NET_SEEDS:
    t0 = time.time(); oof = np.full(N, np.nan); picks = []
    for f, (tr, te) in enumerate(OUTER):
        inner = inner_split(tr, f); scores = {}
        for (depth, wd) in GRID:
            errs, steps = [], []
            for ia, ib in inner:
                pr, st = fit_predict(tr[ia], tr[ib], depth, wd, seed, early=True)
                errs.append(rmse(pr, y[tr[ib]])); steps.append(st)
            scores[(depth, wd)] = (float(np.mean(errs)), int(np.clip(np.median(steps), 200, EPOCHS)))
        best = min(scores, key=lambda k: scores[k][0]); best_r, best_steps = scores[best]
        # outer fit: chosen config, trained for the median early-stopping step of its inner runs (no peeking at te)
        oof[te], _ = fit_predict(tr, te, best[0], best[1], seed, early=False, epochs=best_steps)
        picks.append([best[0], best[1], best_steps, best_r])
        log(f"  seed {seed} fold {f}: picked depth={best[0]} wd={best[1]:.0e} steps={best_steps} (inner {best_r:.3f})  outer {rmse(oof[te], y[te]):.3f}  [{time.time()-t0:.0f}s]")
    res[f'conv_seed{seed}'] = dict(**slices(oof), picks=picks)
    np.save(f"{OUT}/oof_nested_seed{seed}.npy", oof)
    log(f"  seed {seed}: {slices(oof)}  ({time.time()-t0:.0f}s)")
    json.dump(res, open(f"{OUT}/nested.json", "w"), indent=2)

p = [res[f'conv_seed{s}']['all'] for s in NET_SEEDS]
res['conv_mean'] = float(np.mean(p)); res['conv_std'] = float(np.std(p))
ens = np.mean([np.load(f"{OUT}/oof_nested_seed{s}.npy") for s in NET_SEEDS], 0)   # seed ensemble
np.save(f"{OUT}/oof_nested_ensemble.npy", ens); res['conv_ensemble'] = slices(ens)
log(f"[SUMMARY] {TAG}: {len(NET_SEEDS)}-seed ensemble = {slices(ens)}")
json.dump(res, open(f"{OUT}/nested.json", "w"), indent=2)
log(f"\n[SUMMARY] {TAG}: cluster conv nested CV = {np.mean(p):.3f} +- {np.std(p):.3f} meV/atom over seeds {NET_SEEDS}")
if SYSTEM == "fenicr":
    for nm in ("ge32", "eq108"):
        v = [res[f'conv_seed{s}'][nm] for s in NET_SEEDS if res[f'conv_seed{s}'][nm] is not None]
        if v: log(f"[SUMMARY] {TAG}: {nm} = {np.mean(v):.3f} +- {np.std(v):.3f}")
log(f"[SUMMARY] best linear on the same folds = " +
    str(min((res[k]['all'], k) for k in res if k.startswith('lin_'))) + f"   total {time.time()-t_start:.0f}s")
