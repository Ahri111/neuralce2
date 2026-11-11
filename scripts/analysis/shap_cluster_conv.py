"""Exact Shapley values of the cluster channels of the cluster convolution, one system per call (SYSTEM = ptni | stfo | fenicr).
What is explained: the model's energy (meV/atom) of configurations sampled by its own parallel-tempering runs (all temperatures, all compositions run).
Features = the model's input clusters, grouped by physical type (pairs by shell / sublattice, triplets, quadruplets, spin pairs by shell).
Value function v(S): model evaluated with only the groups in S (the other inputs at their training mean) -> exact Shapley values over all 2^n subsets.
For every structure  sum_g phi_g = E(structure) - E(all inputs at the training mean).

Environment knobs:
  SYSTEM   ptni | stfo | fenicr (required).
  FILES    ground-state mode: explain the given structures instead of the Monte Carlo samples; "path=label:path=label:..." (T = 0, output <system>_gs.json).
  OUTNAME  output stem (default SYSTEM);  TAG  another checkpoint of the same system (default: the model of the paper).
  FENICR_MC_DIR  directory of the FeNiCr Monte Carlo runs (default results/fenicr_sro; the F6 configurations are shipped there, F5 are not).
Inputs:  models/nce/<TAG>/model.pkl; Monte Carlo configurations under $NCE_RUNS/ptni_ladder/conv/<x>/M8_long/ (PtNi) and $NCE_RUNS/stfo_mc/<TAG>/c<c>_K1/ (STFO),
         FeNiCr from $FENICR_MC_DIR/<TAG>_dense_<F6|F5>/multiT/; models/ce/stfo/icet_q56.ce (the STFO reference primitive cell);
         the shipped ground states are in results/shap/gs/.
Outputs: results/shap/<OUTNAME>.json.
Run:     SYSTEM=ptni python scripts/analysis/shap_cluster_conv.py   (slurm/shap.sh)"""
import os, re, glob, json, itertools, math, numpy as np
from neuralce.paths import MODELS, RUNS, RESULTS
SYSTEM = os.environ["SYSTEM"]
import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
from ase.io import read
from icet import ClusterSpace
from neuralce.nce.mc_model import load_model
from neuralce.nce.general import local_cluster_features, shell_partners, pack, make_forward
from neuralce.nce.gpu_layer0 import build_table
Tof = lambda f: float(re.search(r"final_T(\d+)_r", f).group(1)); every = lambda L, k: sorted(L, key=Tof)[::k]
FENICR_MC = os.environ.get("FENICR_MC_DIR", f"{RESULTS}/fenicr_sro")
if SYSTEM == "ptni":
    TAG, SEED = "nipt_v3_nomax_sel", 0; files = [(f, x) for x in ("x25", "x50", "x75") for f in every(glob.glob(f"{RUNS}/ptni_ladder/conv/{x}/M8_long/final_T*_r*.cif"), 3)]; SH = [(2.77, "pairs 1NN"), (3.92, "pairs 2NN")]
elif SYSTEM == "stfo":
    TAG, SEED = "stfo_dd_p10_w10_wd5_aug467_al1a_w0", 0; files = [(f, c) for c in ("125", "250", "375", "500", "750") for f in every(glob.glob(f"{RUNS}/stfo_mc/{TAG}/c{c}_K1/final_T*_r*.cif"), 2)]
else:
    TAG, SEED = "fenicr_s2b_mm_c65_d1", 2; files = [(f, c) for c in ("F6", "F5") for f in every(glob.glob(f"{FENICR_MC}/{TAG}_dense_{c}/multiT/final_T*_r*.extxyz"), 6)]; SH = [(2.55, "chem. pairs 1NN"), (3.60, "chem. pairs 2NN")]
if os.environ.get("FILES"):                                                                  # ground-state mode: explain given structures (path=label, colon-separated); T = 0; output <system>_gs.json
    files = [(fl.split("=")[0], fl.split("=")[1]) for fl in os.environ["FILES"].split(":")]; Tof = lambda f: 0.0
OUTNAME = os.environ.get("OUTNAME", SYSTEM); TAG = os.environ.get("TAG", TAG)                   # TAG: explain another checkpoint of the same system (e.g. the final STFO model)
if SYSTEM != "ptni":                                                                          # the STFO / Fe-Ni-Cr models, Monte Carlo runs and ground states are not shipped
    _need = [f"{MODELS}/nce/{TAG}/model.pkl"] + ([f"{MODELS}/ce/stfo/icet_q56.ce"] if SYSTEM == "stfo" else []) + [f for f, _ in files]
    if SYSTEM == "fenicr" and not os.environ.get("FILES"): _need.append(FENICR_MC)
    for _p in _need:
        if not os.path.exists(_p):
            print(f"[skip] shap_cluster_conv.py: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_MODELS / $NCE_RUNS / FENICR_MC_DIR / FILES to run", flush=True); raise SystemExit(0)
mdl = load_model(f"{MODELS}/nce/{TAG}/model.pkl"); cs = ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"]); od = cs.as_list; keep = np.asarray(mdl["keep"]); K = int(mdl.get("K_chem", len(od)))
print("orbit keys:", list(od[1].keys()), flush=True)


def group_of(k):
    if k >= K:
        s = (k - K) // len(mdl["spin_pairs"]); return f"spin pairs {s + 1}NN" if s < 2 else "other spin terms"
    d = od[k]; o = int(d["order"]); dist = 2 * float(d["radius"])
    if o == 0: return None
    if o == 1: return "singlets"
    if SYSTEM == "stfo":
        sub = str(d.get("sublattices", "")); nB = sub.count("A") if sub else -1          # sublattice A = B site (Ti/Fe), B = O site (O/V_O) in the STFO cluster space
        if o == 2:
            if dist < 2.3: return "B-O pair 1.98 A  (Fe/Ti - V_O)"
            if dist < 3.1: return "O-O pair 2.8 A  (V_O-V_O, 90 deg)"
            if dist < 4.1: return "B-B 1NN and O-O 180 deg pairs (3.96 A)"
            return "longer pairs"
        return "triplets" if o == 3 else "quadruplets"
    if o == 2:
        for sd_, nm in SH:
            if abs(dist - sd_) < 0.2: return nm
        return ("chem. " if SYSTEM == "fenicr" else "") + "pairs 3NN+"
    return "triplets" if o == 3 else "quadruplets"


cg = [group_of(int(k)) for k in keep]; GROUPS = list(dict.fromkeys(g for g in cg if g)); n = len(GROUPS); print("groups:", {g: cg.count(g) for g in GROUPS}, flush=True)
subsets = [frozenset(s) for r in range(n + 1) for s in itertools.combinations(GROUPS, r)]; masks = jnp.asarray(np.array([[1.0 if (g is None or g in Sx) else 0.0 for g in cg] for Sx in subsets]))
P = mdl["params"][SEED]; P = {k: (jnp.asarray(v) if not isinstance(v, (list, dict)) else v) for k, v in P.items()}; P["layers"] = [{kk: jnp.asarray(vv) for kk, vv in L.items()} for L in P["layers"]]; P["head"] = {kk: jnp.asarray(vv) for kk, vv in P["head"].items()}
fwd = make_forward(); ys = float(mdl["ys"]); W = {(Sx, g): math.factorial(len(Sx)) * math.factorial(n - len(Sx) - 1) / math.factorial(n) for Sx in subsets for g in GROUPS if g not in Sx}; idx = {Sx: q for q, Sx in enumerate(subsets)}
if SYSTEM == "stfo":
    from icet import ClusterExpansion
    from icet.tools import map_structure_to_reference
    PRIM = ClusterExpansion.read(f"{MODELS}/ce/stfo/icet_q56.ce").get_cluster_space_copy().primitive_structure
rows = []
for f, lab in files:
    at = read(f); spins = None
    if SYSTEM == "stfo": at, _ = map_structure_to_reference(at, PRIM, inert_species=["Sr"], assume_no_cell_relaxation=False)
    if SYSTEM == "fenicr": spins = at.get_initial_magnetic_moments().astype(float)
    act = build_table(cs, at)["active"]; h0 = local_cluster_features(cs, at, act)
    if mdl.get("spin", 0):
        from neuralce.nce.spin_features import spin_features_np
        h0 = np.hstack([h0, spin_features_np(at, act, spins, list(mdl["spin_edges"]), mdl["spin_pairs"], mdl["spin_species"], int(mdl["spin"]), int(mdl.get("spin_node_lo", 0) or 0))])
    H, M, SI, SM = pack([h0[:, keep]], [shell_partners(at, act, list(mdl["shell_edges"]))]); X = jnp.asarray((H - mdl["mu"]) / mdl["sd"] * M[..., None]); Mj, SIj, SMj = jnp.asarray(M), jnp.asarray(SI), jnp.asarray(SM)
    v = np.asarray(jax.lax.map(lambda m: fwd(P, X * m[None, None, :], Mj, SIj, SMj)[0], masks)) * ys * 1e3
    phi = {g: float(sum(W[(Sx, g)] * (v[idx[Sx | {g}]] - v[idx[Sx]]) for Sx in subsets if g not in Sx)) for g in GROUPS}
    rows.append(dict(file=os.path.basename(f), label=lab, T=Tof(f), E_meV=float(v[idx[frozenset(GROUPS)]]), E_base_meV=float(v[idx[frozenset()]]), phi=phi)); print(lab, int(Tof(f)), {g: round(x, 2) for g, x in phi.items()}, flush=True)
os.makedirs(f"{RESULTS}/shap", exist_ok=True); json.dump(dict(system=SYSTEM, model=TAG, seed=SEED, groups=GROUPS, n_channels={g: cg.count(g) for g in GROUPS}, rows=rows), open(f"{RESULTS}/shap/{OUTNAME}.json", "w"), indent=1); print("wrote", f"{RESULTS}/shap/{OUTNAME}.json")
