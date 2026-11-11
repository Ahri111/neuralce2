"""Why the cluster expansions order too early in Fe42.5Cr7.5Ni50 and the neural cluster expansion does not (support for Fig. 5d-f and Supplementary Fig. S4).
Both models share the cluster space (icet [6, 5]) and the 12 species-resolved 1NN / 2NN spin-pair planes, so their energies can be split into the SAME
channel groups (exact Shapley values over the groups, as scripts/analysis/shap_cluster_conv.py; for the linear model this is its own sum of ECI x feature).
  Neural cluster expansion              models/nce/fenicr_s2b_mm_c65_d1     (seed index 2, the model of Fig. 5)
  Orthonormal cluster expansion + spin  models/nce/fenicr_icet_heis_bench   (depth-0 form of the benchmark icet + Heisenberg-pair fit)
Structure sets:
  A  the 13 DFT cells of the --dft directory (32 atoms, Fe14 Ni16 Cr2: 10 random cells spanning alpha_2NN + 3 L1_0 cells), computed after both models were
     fitted -> held out.  Ideal lattice (POSCAR) + spin labels from the final DFT moments.  Label rule = the one of the training set
     (|m| >= 2.0 / 0.4 / 1.4 mu_B for Fe / Ni / Cr -> sign(m), else 0), gated on the reproduced training structure mag_data_0000.
  B  the 11 Monte Carlo configurations of the neural-cluster-expansion run used in Fig. 5e, f (4,000 atoms, 300-1389 K).
  C  the Monte Carlo configurations of the orthonormal CE + spin run at the same 11 temperatures.
Magnetism switched off: set A is also evaluated with every spin label set to 0 (the models' counterpart of the non-magnetic DFT run of the same cells).
Spin state of the Monte Carlo configurations (sets B, C): fraction of non-zero spins and mean s_i s_j over the 1NN and 2NN bonds.

The DFT cells (VASP OUTCAR / OSZICAR / POSCAR, ord_manifest.json, des_manifest.json, gate_mag_data_0000_isif3) are NOT shipped; the processed result
of this script is shipped as results/shap/fenicr_order_energy.json and is what Supplementary Fig. S4 reads.
Arguments / knobs:
  --dft DIR      the DFT cells (default $NCE_RUNS/DFT_fenicr)
  --mc-dir DIR   the Monte Carlo runs <tag>_dense_F6/multiT/*.extxyz of both models (default results/fenicr_sro, where they are shipped)
  SMOKE=1        two DFT cells and one configuration per Monte Carlo set, output *_smoke.json
  REUSE=1        take the rows of sets B and C from an existing output file (their features take 15 min) and only add what is missing
Inputs:  models/nce/<tag>/model.pkl (both models), data/feni_cr/{detailed_info.csv, mag_data_0000.cif}, results/shap/fenicr.json (the Fig. 5e, f rows: gate + file list).
Outputs: results/shap/fenicr_order_energy.json (every structure: alpha_1NN, alpha_2NN, DFT energies where they exist, and per model the energy and the group values).
Run:     python scripts/analysis/fenicr_order_energy.py [--dft DIR] [--mc-dir DIR]
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above."""
import os, re, ast, glob, json, itertools, math, argparse, numpy as np
from neuralce.paths import DATA, MODELS, RUNS, RESULTS
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--dft", default=f"{RUNS}/DFT_fenicr", help="directory of the 13 DFT cells (not shipped)")
ap.add_argument("--mc-dir", default=f"{RESULTS}/fenicr_sro", help="directory holding <tag>_dense_F6/multiT/ of both models")
args = ap.parse_args()
def require(paths, hint):                                                                 # the Fe-Ni-Cr inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{MODELS}/nce/fenicr_s2b_mm_c65_d1/model.pkl", f"{MODELS}/nce/fenicr_icet_heis_bench/model.pkl", f"{DATA}/feni_cr/detailed_info.csv", f"{DATA}/feni_cr/mag_data_0000.cif",
         f"{RESULTS}/shap/fenicr.json", args.dft, args.mc_dir], "$NCE_MODELS/nce, $NCE_DATA/feni_cr, $NCE_RESULTS/shap (--dft, --mc-dir)")
import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list
from icet import ClusterSpace
from neuralce.nce.mc_model import load_model
from neuralce.nce.general import local_cluster_features, shell_partners, pack, make_forward
from neuralce.nce.gpu_layer0 import build_table
from neuralce.nce.spin_features import spin_features_np
from neuralce.ce.spin_pairs import reembed_to_reference_frame

DFT = args.dft; SEL = args.mc_dir; SMOKE = bool(int(os.environ.get("SMOKE", "0"))); OUT = f"{RESULTS}/shap/fenicr_order_energy" + ("_smoke" if SMOKE else "")
MODEL_TAGS = {"Neural cluster expansion": ("fenicr_s2b_mm_c65_d1", 2), "Orthonormal cluster expansion + spin": ("fenicr_icet_heis_bench", 0)}
THR = {"Fe": 2.0, "Ni": 0.4, "Cr": 1.4}; SH = [(2.55, "chem. pairs 1NN"), (3.60, "chem. pairs 2NN")]
fwd = make_forward()


def setup(tag, seed):
    mdl = load_model(f"{MODELS}/nce/{tag}/model.pkl"); cs = ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"]); od = cs.as_list
    keep = np.asarray(mdl["keep"]); K = int(mdl.get("K_chem", len(od)))
    def group_of(k):                                                                       # same grouping as scripts/analysis/shap_cluster_conv.py (SYSTEM = fenicr)
        if k >= K:
            s = (k - K) // len(mdl["spin_pairs"]); return f"spin pairs {s + 1}NN" if s < 2 else "other spin terms"
        d = od[k]; o = int(d["order"]); dist = 2 * float(d["radius"])
        if o == 0: return None
        if o == 1: return "singlets"
        if o == 2:
            for sd_, nm in SH:
                if abs(dist - sd_) < 0.2: return nm
            return "chem. pairs 3NN+"
        return "triplets" if o == 3 else "quadruplets"
    cg = [group_of(int(k)) for k in keep]; G = list(dict.fromkeys(g for g in cg if g)); n = len(G)
    subsets = [frozenset(s) for r in range(n + 1) for s in itertools.combinations(G, r)]; idx = {Sx: q for q, Sx in enumerate(subsets)}
    masks = jnp.asarray(np.array([[1.0 if (g is None or g in Sx) else 0.0 for g in cg] for Sx in subsets]))
    Wt = {(Sx, g): math.factorial(len(Sx)) * math.factorial(n - len(Sx) - 1) / math.factorial(n) for Sx in subsets for g in G if g not in Sx}
    P = mdl["params"][seed]; P = {k: (jnp.asarray(v) if not isinstance(v, (list, dict)) else v) for k, v in P.items()}
    P["layers"] = [{kk: jnp.asarray(vv) for kk, vv in L.items()} for L in P["layers"]]; P["head"] = {kk: jnp.asarray(vv) for kk, vv in P["head"].items()}
    print(tag, "seed index", seed, "| depth", mdl["depth"], "| groups:", {g: cg.count(g) for g in G}, flush=True)
    return dict(mdl=mdl, cs=cs, keep=keep, G=G, subsets=subsets, idx=idx, masks=masks, W=Wt, P=P)


def evaluate(S, at, spins, act, h_chem):
    """energy (meV per atom) and exact group Shapley values of one structure"""
    mdl = S["mdl"]; h0 = h_chem
    if mdl.get("spin", 0):
        h0 = np.hstack([h_chem, spin_features_np(at, act, np.asarray(spins, float), list(mdl["spin_edges"]), mdl["spin_pairs"], mdl["spin_species"], int(mdl["spin"]), int(mdl.get("spin_node_lo", 0) or 0))])
    H, M, SI, SM = pack([h0[:, S["keep"]]], [shell_partners(at, act, list(mdl["shell_edges"]))]); X = jnp.asarray((H - mdl["mu"]) / mdl["sd"] * M[..., None]); Mj, SIj, SMj = jnp.asarray(M), jnp.asarray(SI), jnp.asarray(SM)
    v = np.asarray(jax.lax.map(lambda m: fwd(S["P"], X * m[None, None, :], Mj, SIj, SMj)[0], S["masks"])) * float(mdl["ys"]) * 1e3
    phi = {g: float(sum(S["W"][(Sx, g)] * (v[S["idx"][Sx | {g}]] - v[S["idx"][Sx]]) for Sx in S["subsets"] if g not in Sx)) for g in S["G"]}
    return dict(E_meV=float(v[S["idx"][frozenset(S["G"])]] + 1e3 * float(mdl["ym"])), phi=phi)


def alphas(a):
    """(Fe,Cr)-Ni Warren-Cowley SRO at 1NN (2.55 A) and 2NN (3.60 A), composition-weighted over Fe and Cr as in Su et al. (same as scripts/analysis/fenicr_shap_2nn.py)"""
    i, j, d = neighbor_list("ijd", a, 4.0); s = np.array(a.get_chemical_symbols()); cNi = (s == "Ni").mean(); out = []
    for lo, hi in ((2.2, 3.1), (3.1, 4.0)):
        m = (d > lo) & (d < hi); al = {}
        for X in ("Fe", "Cr"):
            mi = m & (s[i] == X); al[X] = 1.0 - (s[j][mi] == "Ni").mean() / cNi
        cF, cC = (s == "Fe").mean(), (s == "Cr").mean(); out.append(float((cF * al["Fe"] + cC * al["Cr"]) / (cF + cC)))
    return out


def spin_stats(a):
    """fraction of atoms with a non-zero spin label, |mean spin|, and the mean of s_i s_j over the 1NN and the 2NN bonds"""
    sp = a.get_initial_magnetic_moments().astype(float); i, j, d = neighbor_list("ijd", a, 4.0); m1, m2 = (d > 2.2) & (d < 3.1), (d > 3.1) & (d < 4.0)
    return dict(frac_nonzero=float((sp != 0).mean()), abs_mean=float(abs(sp.mean())), ss_1NN=float((sp[i] * sp[j])[m1].mean()), ss_2NN=float((sp[i] * sp[j])[m2].mean()),
                by_species={e: {str(int(v)): float(((sp == v) & (np.array(a.get_chemical_symbols()) == e)).sum() / max((np.array(a.get_chemical_symbols()) == e).sum(), 1)) for v in (-1, 0, 1)} for e in THR})


def dft_dir(d):
    """directory holding the final result: a restarted run <d>_r2 takes precedence when it converged"""
    return d + "_r2" if os.path.exists(f"{d}_r2/OUTCAR") and "reached required accuracy" in open(f"{d}_r2/OUTCAR").read() else d


def dft_energy(d):
    d = dft_dir(d)
    if not os.path.exists(f"{d}/OUTCAR") or not os.path.exists(f"{d}/OSZICAR"): return None
    out = open(f"{d}/OUTCAR").read(); E0 = re.findall(r"E0=\s*([-\d.E+]+)", open(f"{d}/OSZICAR").read())
    if not E0 or ("reached required accuracy" not in out and "Voluntary context switches" not in out): return None
    return float(E0[-1]) / 32 * 1e3


def moments(d):
    L = open(f"{dft_dir(d)}/OUTCAR").read().split("magnetization (x)")[-1].split("\n")[4:]; out = []
    for l in L:
        if l.startswith("---"): break
        out.append(float(l.split()[-1]))
    return np.array(out)


def labels(at, m): return np.array([np.sign(x) if abs(x) >= THR[s] else 0.0 for s, x in zip(at.get_chemical_symbols(), m)])


SS = {name: setup(*ts) for name, ts in MODEL_TAGS.items()}; CS = next(iter(SS.values()))["cs"]; PRIM = next(iter(SS.values()))["mdl"]["primitive"]
def both(at, spins):
    act = build_table(CS, at)["active"]; h = local_cluster_features(CS, at, act); return {name: evaluate(S, at, spins, act, h) for name, S in SS.items()}

# ---- gate: the reproduced training structure through this path == the training path of the dataset -----------------------------------------------
g = f"{DFT}/gate_mag_data_0000_isif3"; at_g = read(f"{g}/POSCAR"); lab_g = labels(at_g, moments(g)); df = pd.read_csv(f"{DATA}/feni_cr/detailed_info.csv")
cif = reembed_to_reference_frame(read(f"{DATA}/feni_cr/mag_data_0000.cif"), PRIM.cell.array); s_cif = np.array(ast.literal_eval(df.spins[df.id == "mag_data_0000"].iloc[0]), float)
fa, fc = at_g.get_scaled_positions() % 1, read(f"{DATA}/feni_cr/mag_data_0000.cif").get_scaled_positions() % 1; Dm = np.abs(fa[:, None, :] - fc[None, :, :]); j = np.minimum(Dm, 1 - Dm).sum(-1).argmin(1)
assert np.array_equal(lab_g, s_cif[j]), "spin-label rule does not reproduce the dataset labels of mag_data_0000"
e_mine, e_cif = both(at_g, lab_g), both(cif, s_cif); mN = SS["Neural cluster expansion"]["mdl"]; k0 = list(mN["train_ids"]).index("mag_data_0000")
for name in MODEL_TAGS:
    print(f"GATE {name}: this path {e_mine[name]['E_meV']:.6f}  dataset path {e_cif[name]['E_meV']:.6f} meV/atom  (diff {abs(e_mine[name]['E_meV'] - e_cif[name]['E_meV']):.2e})", flush=True)
    assert abs(e_mine[name]["E_meV"] - e_cif[name]["E_meV"]) < 1e-6
dtp = abs(e_mine["Neural cluster expansion"]["E_meV"] - 1e3 * float(np.asarray(mN["train_pred"])[MODEL_TAGS["Neural cluster expansion"][1], k0]))
print(f"GATE NeuralCE vs stored training prediction: diff {dtp:.2e} meV/atom | DFT label {1e3 * float(df.total_energy[df.id == 'mag_data_0000'].iloc[0]) / 32:.3f} meV/atom; spin labels 32/32 equal", flush=True)
assert dtp < 1e-3

# ---- set A: the DFT cells ---------------------------------------------------------------------------------------------------------------------------
rows = []
man = {r["k"]: r for r in json.load(open(f"{DFT}/ord_manifest.json"))["runs"] if r["mode"] == "spin"}; des = {r["name"]: r for r in json.load(open(f"{DFT}/des_manifest.json")) if r["mode"] == "spin"}
cells = [(f"ord_{k:02d}", "random", man[k]["alpha1"], man[k]["alpha2"]) for k in sorted(man)] + [(f"des_{n}", "L1_0", des[n]["alpha1"], des[n]["alpha2"]) for n in des]
for name, kind, a1m, a2m in (cells[:2] if SMOKE else cells):
    es, en = dft_energy(f"{DFT}/{name}_spin"), dft_energy(f"{DFT}/{name}_nm")
    if es is None: print("skip (no converged spin-polarised DFT):", name, flush=True); continue
    at = read(f"{DFT}/{name}_spin/POSCAR"); m = moments(f"{DFT}/{name}_spin"); lab = labels(at, m); a1, a2 = alphas(at)
    assert abs(a1 - a1m) < 1e-6 and abs(a2 - a2m) < 1e-6, (name, a1, a1m, a2, a2m)       # the alpha of the manifest is recomputed from the structure
    rows.append(dict(set="A", name=name, kind=kind, n_atoms=len(at), a1=a1, a2=a2, E_dft_spin=es, E_dft_nm=en, spin_labels={s: [int(x) for x, t in zip(lab, at.get_chemical_symbols()) if t == s] for s in THR},
                     models=both(at, lab), models_spin_off={k: v["E_meV"] for k, v in both(at, np.zeros(len(at))).items()}))
    print("A", name, f"a1 {a1:+.3f} a2 {a2:+.3f} DFT {es:.2f}", {k: round(v["E_meV"], 2) for k, v in rows[-1]["models"].items()}, flush=True)

# ---- sets B, C: Monte Carlo configurations of the two runs at the temperatures of Fig. 5e, f ----------------------------------------------------------
Tof = lambda f: float(re.search(r"final_T(\d+)_r", f).group(1)); J = json.load(open(f"{RESULTS}/shap/fenicr.json")); fB = [q["file"] for q in J["rows"] if q["label"] == "F6"]; TB = sorted(Tof(f) for f in fB)
allC = glob.glob(f"{SEL}/fenicr_icet_heis_bench_dense_F6/multiT/final_T*_r*.extxyz"); fC = [min(allC, key=lambda f: abs(Tof(f) - t)) for t in TB]
OLD = {(r["set"], r["name"]): r for r in json.load(open(OUT + ".json"))["rows"]} if (int(os.environ.get("REUSE", "0")) and os.path.exists(OUT + ".json")) else {}
for tag, files in (("B", [f"{SEL}/fenicr_s2b_mm_c65_d1_dense_F6/multiT/{f}" for f in fB]), ("C", fC)):
    for f in (sorted(files, key=Tof)[:1] if SMOKE else sorted(files, key=Tof)):
        at = read(f); sp = at.get_initial_magnetic_moments().astype(float); a1, a2 = alphas(at); key = (tag, os.path.basename(f))
        rows.append(dict(set=tag, name=os.path.basename(f), T=Tof(f), n_atoms=len(at), a1=a1, a2=a2, spin=spin_stats(at), models=OLD[key]["models"] if key in OLD else both(at, sp)))
        print(tag, int(Tof(f)), f"a1 {a1:+.3f} a2 {a2:+.3f}", {k: round(v["E_meV"], 2) for k, v in rows[-1]["models"].items()}, flush=True)
if not SMOKE:                                                                               # set B through this script == the stored Shapley rows of Fig. 5e, f
    old = {q["file"]: q for q in J["rows"] if q["label"] == "F6"}
    dmax = max(abs(r["models"]["Neural cluster expansion"]["phi"][g] - old[r["name"]]["phi"][g]) for r in rows if r["set"] == "B" for g in J["groups"]); print(f"GATE set B vs results/shap/fenicr.json: max |phi diff| {dmax:.2e} meV/atom", flush=True)
    assert dmax < 1e-6
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(models={k: dict(tag=t, seed_index=s, groups=SS[k]["G"]) for k, (t, s) in MODEL_TAGS.items()}, spin_label_thresholds=THR, rows=rows), open(OUT + ".json", "w"), indent=1); print("wrote", OUT + ".json", flush=True)

# ---- numbers ----------------------------------------------------------------------------------------------------------------------------------------
def slope(x, y): return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])
A = [r for r in rows if r["set"] == "A" and r["kind"] == "random"]
if len(A) >= 3:
    a2 = [r["a2"] for r in A]; print(f"\nset A, {len(A)} random cells, slope of E vs alpha_2NN (meV/atom per unit alpha):  DFT spin {slope(a2, [r['E_dft_spin'] for r in A]):+.1f}   DFT non-magnetic {slope(a2, [r['E_dft_nm'] for r in A]):+.1f}")
    for name in MODEL_TAGS:
        e = np.array([r["models"][name]["E_meV"] for r in A]); d = np.array([r["E_dft_spin"] for r in A])
        print(f"   {name:38s} slope {slope(a2, e):+7.1f} | RMSE vs DFT after removing the mean {np.sqrt(np.mean(((e - e.mean()) - (d - d.mean())) ** 2)):.2f} | by group:", {g: round(slope(a2, [r['models'][name]['phi'][g] for r in A]), 1) for g in SS[name]["G"]})
    print("   magnetism off (all spin labels 0):", {name: round(slope(a2, [r["models_spin_off"][name] for r in A]), 1) for name in MODEL_TAGS}, "| DFT non-magnetic", round(slope(a2, [r["E_dft_nm"] for r in A]), 1))
    for r in rows:
        if r["set"] == "A" and r["kind"] != "random":
            d0 = np.mean([q["E_dft_spin"] for q in A]); print(f"   {r['name']:16s} DFT {r['E_dft_spin'] - d0:+7.1f}", {name: round(r["models"][name]["E_meV"] - np.mean([q["models"][name]["E_meV"] for q in A]), 1) for name in MODEL_TAGS})
for tag in ("B", "C"):
    Q = sorted([r for r in rows if r["set"] == tag], key=lambda r: r["T"])
    if len(Q) < 3: continue
    a2 = [r["a2"] for r in Q]; print(f"\nset {tag} ({'neural cluster expansion' if tag == 'B' else 'orthonormal CE + spin'} Monte Carlo), alpha_2NN by T:", {int(r["T"]): round(r["a2"], 2) for r in Q})
    print("   spin state by T (non-zero fraction, <s s> 1NN, <s s> 2NN):", {int(r["T"]): (round(r["spin"]["frac_nonzero"], 2), round(r["spin"]["ss_1NN"], 2), round(r["spin"]["ss_2NN"], 2)) for r in Q})
    print("   spin labels by species at the lowest and the highest T:", {int(r["T"]): {e: {k: round(v, 2) for k, v in d_.items()} for e, d_ in r["spin"]["by_species"].items()} for r in (Q[0], Q[-1])})
    for name in MODEL_TAGS: print(f"   {name:38s} slope of E vs alpha_2NN {slope(a2, [r['models'][name]['E_meV'] for r in Q]):+7.1f} | by group:", {g: round(slope(a2, [r['models'][name]['phi'][g] for r in Q]), 1) for g in SS[name]["G"]})
