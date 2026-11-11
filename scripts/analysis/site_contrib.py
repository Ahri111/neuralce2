"""Site-resolved attribution of the cluster-convolution energy (SYSTEM = ptni | stfo | fenicr), on the ground-state structure.
The model is a sliding window:  E = (1/N) sum_i e_i,  e_i = X_i.lin + h_i.w  with X_i the site-local cluster correlations.  Two site maps:
  e_site[i]          exact energy carried by site i (meV), relative to the mean site
  a_site[i, group]   first-order attribution of channel group g at site i:  sum_{k in g} X_ik dE/dX_ik  (meV per site; gradient x input)
Written to $RESULTS/shap/site_<system>.json (positions, symbols, e_site, a_site, groups) and rendered with ASE + POV-Ray to
figures/site_<system>.png: atoms coloured by the attribution of the leading channel group (white = 0, purple = stabilising).

Inputs : models/nce/<TAG>/model.pkl (ptni: nipt_v3_nomax_sel, stfo: env TAG, default stfo_dd_p10_w10_wd5_aug467_al4c_w0,
         fenicr: fenicr_s2b_mm_c65_d1); structure = models/structures/ptni/ref_L10_NiPt.cif | data/new_bench/A00_500.cif |
         $RESULTS/shap/gs/fenicr_gs_mag_data_0000.extxyz (env FILE overrides); STFO is mapped onto the primitive cell of
         models/ce/stfo/icet_q56.ce.
Env    : SYSTEM (required), TAG, FILE, SLAB=z0:z1 (render one (001) layer, Angstrom), POVRAY (povray executable; default
         the `povray` on PATH).  The attribution is written before rendering, so the json is produced even without POV-Ray.
Outputs: $RESULTS/shap/site_<system>.json, figures/site_<system>.{pov,ini,png}.
Run    : SYSTEM=stfo python scripts/analysis/site_contrib.py
"""
import os, json, shutil, numpy as np
from neuralce.paths import REPO, DATA, MODELS, RESULTS
SYSTEM = os.environ["SYSTEM"]
import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
from ase.io import read
from icet import ClusterSpace
from neuralce.nce.mc_model import load_model
from neuralce.nce.general import local_cluster_features, shell_partners, pack, make_forward
from neuralce.nce.gpu_layer0 import build_table
TAGS = {"ptni": ("nipt_v3_nomax_sel", 0, f"{MODELS}/structures/ptni/ref_L10_NiPt.cif", "pairs 1NN"),
        "stfo": (os.environ.get("TAG", "stfo_dd_p10_w10_wd5_aug467_al4c_w0"), 0, f"{DATA}/new_bench/A00_500.cif", "triplets"),
        "fenicr": ("fenicr_s2b_mm_c65_d1", 2, f"{RESULTS}/shap/gs/fenicr_gs_mag_data_0000.extxyz", "spin pairs 2NN")}
TAG, SEED, FILE, LEAD = TAGS[SYSTEM]; FILE = os.environ.get("FILE", FILE)
if SYSTEM != "ptni":                                                                     # the STFO / Fe-Ni-Cr models and ground-state structures are not shipped
    for _p in [f"{MODELS}/nce/{TAG}/model.pkl", FILE] + ([f"{MODELS}/ce/stfo/icet_q56.ce"] if SYSTEM == "stfo" else []):
        if not os.path.exists(_p):
            print(f"[skip] site_contrib.py: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_MODELS / $NCE_DATA / $NCE_RESULTS (TAG, FILE) to run", flush=True); raise SystemExit(0)
mdl = load_model(f"{MODELS}/nce/{TAG}/model.pkl"); cs = ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"]); od = cs.as_list; keep = np.asarray(mdl["keep"]); K = int(mdl.get("K_chem", len(od)))
SH = {"ptni": [(2.77, "pairs 1NN"), (3.92, "pairs 2NN")], "fenicr": [(2.55, "chem. pairs 1NN"), (3.60, "chem. pairs 2NN")], "stfo": []}[SYSTEM]


def group_of(k):                                                                         # same channel grouping as the Shapley analysis
    if k >= K:
        s = (k - K) // len(mdl["spin_pairs"]); return f"spin pairs {s + 1}NN" if s < 2 else "other spin terms"
    d = od[k]; o = int(d["order"]); dist = 2 * float(d["radius"])
    if o == 0: return None
    if o == 1: return "singlets"
    if SYSTEM == "stfo":
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


cg = [group_of(int(k)) for k in keep]; GROUPS = list(dict.fromkeys(g for g in cg if g)); print("groups:", {g: cg.count(g) for g in GROUPS}, flush=True)
P = mdl["params"][SEED]; P = {k: (jnp.asarray(v) if not isinstance(v, (list, dict)) else v) for k, v in P.items()}; P["layers"] = [{kk: jnp.asarray(vv) for kk, vv in L.items()} for L in P["layers"]]; P["head"] = {kk: jnp.asarray(vv) for kk, vv in P["head"].items()}
fwd = make_forward(); ys = float(mdl["ys"])
at = read(FILE); spins = None
if SYSTEM == "stfo":
    from icet import ClusterExpansion
    from icet.tools import map_structure_to_reference
    PRIM = ClusterExpansion.read(f"{MODELS}/ce/stfo/icet_q56.ce").get_cluster_space_copy().primitive_structure
    at, _ = map_structure_to_reference(at, PRIM, inert_species=["Sr"], assume_no_cell_relaxation=False)
if SYSTEM == "fenicr": spins = at.get_initial_magnetic_moments().astype(float)
act = np.asarray(build_table(cs, at)["active"]); h0 = local_cluster_features(cs, at, act)
if mdl.get("spin", 0):
    from neuralce.nce.spin_features import spin_features_np
    h0 = np.hstack([h0, spin_features_np(at, act, spins, list(mdl["spin_edges"]), mdl["spin_pairs"], mdl["spin_species"], int(mdl["spin"]), int(mdl.get("spin_node_lo", 0) or 0))])
H, M, SI, SM = pack([h0[:, keep]], [shell_partners(at, act, list(mdl["shell_edges"]))]); X = jnp.asarray((H - mdl["mu"]) / mdl["sd"] * M[..., None]); Mj, SIj, SMj = jnp.asarray(M), jnp.asarray(SI), jnp.asarray(SM)
N = int(M.sum()); S = lambda Xv: (fwd(P, Xv, Mj, SIj, SMj)[0] - P["head"]["b"]) * N                 # total energy (model units) without the constant
g = np.asarray(jax.grad(S)(X))[0]; Xn = np.asarray(X)[0]; A = g * Xn * ys * 1e3                       # (n_active, C) gradient x input, meV
a_site = {gname: A[:, [c for c, gg in enumerate(cg) if gg == gname]].sum(1) for gname in GROUPS}
# exact site energies: replay the forward layer by layer
def layer(L, h):
    gth = jax.vmap(lambda hb, sb: hb[sb])(h, SIj); m = SMj[..., None]; z = (gth * m).sum(-2) / jnp.maximum(SMj.sum(-1), 1.0)[..., None]
    pre = h @ L["self"] + jnp.einsum("bpsc,pcd->bsd", z, L["shell"]) + L["b"]
    if "shell_max" in L:
        zmax = jnp.where(m > 0, gth, -1e9).max(-2); zmax = jnp.where(SMj.sum(-1)[..., None] > 0, zmax, 0.0); pre = pre + jnp.einsum("bpsc,pcd->bsd", zmax, L["shell_max"])
    out = jnp.tanh(pre)
    if "gate" in L: out = out * jax.nn.sigmoid(jnp.einsum("bpsc,pcd->bsd", z, L["gate"]) + L["gb"])
    return out * Mj[..., None]
h = X
for L in P["layers"]: h = layer(L, h)
e_site = np.asarray((X @ P["lin"] + (h @ P["head"]["w"] if P["layers"] else 0.0)) * Mj)[0] * ys * 1e3; e_site = e_site - e_site.mean()
E_check = float(fwd(P, X, Mj, SIj, SMj)[0]) * ys * 1e3; print(f"{SYSTEM} {TAG} {os.path.basename(FILE)}: N_active={N}  E={E_check:.2f} meV/site(model)  sum of gradient x input = {A.sum() / N:.2f} meV/site", flush=True)
for gname in GROUPS: v = a_site[gname]; print(f"  {gname:42s} mean {v.mean() / 1:+8.2f}  min {v.min():+8.2f}  max {v.max():+8.2f}  (meV per site, gradient x input)", flush=True)
sym = np.array(at.get_chemical_symbols()); pos = at.get_positions()
out = dict(system=SYSTEM, model=TAG, seed=SEED, file=FILE, groups=GROUPS, lead=LEAD, active=act.tolist(), symbols=sym.tolist(), positions=pos.tolist(), cell=at.cell.array.tolist(),
           e_site=e_site.tolist(), a_site={g_: a_site[g_].tolist() for g_ in GROUPS}, spins=(spins.tolist() if spins is not None else None))
os.makedirs(f"{RESULTS}/shap", exist_ok=True); json.dump(out, open(f"{RESULTS}/shap/site_{SYSTEM}.json", "w")); print("wrote", f"{RESULTS}/shap/site_{SYSTEM}.json", flush=True)

# ---- render: element-coloured core + translucent shell coloured by the attribution of the leading channel --------------------------
from ase import Atoms
from ase.io.pov import write_pov
import matplotlib; matplotlib.use("Agg"); import matplotlib.colors as mc
FIG = f"{REPO}/figures"; os.makedirs(FIG, exist_ok=True)
v = np.zeros(len(at)); v[act] = a_site[LEAD]; vmax = float(np.percentile(np.abs(v[act]), 98)) or 1.0
cmap = mc.LinearSegmentedColormap.from_list("pur", ["#5b2a9e", "#ffffff", "#e07b39"]); norm = mc.Normalize(-vmax, vmax)
ELEM = {"Pt": (0.85, 0.85, 0.90), "Ni": (0.31, 0.82, 0.31), "Fe": (0.88, 0.40, 0.20), "Cr": (0.54, 0.60, 0.78), "Ti": (0.75, 0.76, 0.78), "O": (1.0, 0.05, 0.05), "Xe": (0.15, 0.15, 0.15), "Sr": (0.0, 1.0, 0.0)}
keep_idx = np.arange(len(at))
if SYSTEM == "stfo": keep_idx = np.where(sym != "Sr")[0]                                                  # Sr is inert and hides the B-O framework
if os.environ.get("SLAB"):                                                                                # one (001) layer: SLAB=z0:z1 (Angstrom); the attribution is computed on the full cell
    z0, z1 = map(float, os.environ["SLAB"].split(":")); keep_idx = keep_idx[(pos[keep_idx, 2] >= z0) & (pos[keep_idx, 2] < z1)]; print("slab", z0, z1, "atoms", len(keep_idx), flush=True)
core_r = {"Pt": 0.55, "Ni": 0.45, "Fe": 0.45, "Cr": 0.45, "Ti": 0.40, "O": 0.22, "Xe": 0.22}
P2 = np.vstack([pos[keep_idx], pos[keep_idx]]); S2 = [sym[i] for i in keep_idx] * 2; n = len(keep_idx)
at2 = Atoms(symbols=S2, positions=P2, cell=at.cell, pbc=False)
colors = [list(ELEM[s_]) for s_ in S2[:n]] + [list(cmap(norm(v[i]))[:3]) for i in keep_idx]
radii = [core_r[s_] for s_ in S2[:n]] + [core_r[s_] * 2.0 for s_ in S2[:n]]
trans = [0.0] * n + [0.45] * n
ROT = {"ptni": "0x,0y,0z", "stfo": "-90x,0y,0z", "fenicr": "0x,0y,0z"}[SYSTEM]
if os.environ.get("SLAB"): ROT = "0x,0y,0z"
DIST = 400.0
pov = write_pov(f"{FIG}/site_{SYSTEM}.pov", at2, rotation=ROT, show_unit_cell=0, radii=radii, colors=colors,
                povray_settings=dict(canvas_width=1600, transparent=True, camera_type="perspective", camera_dist=DIST, display=False, bondatoms=[], textures=["ase3"] * (2 * n), transmittances=trans, background="White"))
import re, math, subprocess
pf = f"{FIG}/site_{SYSTEM}.pov"; txt = open(pf).read(); rl = float(re.search(r"right\s+-?([\d.]+)\*x", txt).group(1)); ang = 2 * math.degrees(math.atan(rl / 2 / DIST))
open(pf, "w").write(txt.replace("camera {perspective", f"camera {{perspective angle {ang:.3f}", 1))                      # POV-Ray 3.7 needs the angle stated explicitly
POVRAY = os.environ.get("POVRAY") or shutil.which("povray")
if POVRAY is None: raise SystemExit(f"povray not found (set POVRAY); {pf} written, json done")
subprocess.run([POVRAY, f"site_{SYSTEM}.ini"], cwd=FIG, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("rendered", f"{FIG}/site_{SYSTEM}.png", "lead", LEAD, "colour range +-", round(vmax, 2), "meV", flush=True)
