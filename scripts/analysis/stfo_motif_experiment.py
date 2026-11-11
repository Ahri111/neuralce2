"""STFO: does the cluster convolution carry the local physics reported by N. Kim et al. (2022)?  Computational experiment on the model itself:
build the named local arrangements in the 160-site cell at x = 0.125 (4 Fe + 2 V_O in SrTiO3) and read the model's energy differences.
 1. where does a vacancy sit?      both V_O in Fe-V_O-Fe  /  in Fe-V_O-Ti  /  in Ti-V_O-Ti (far from Fe)
 2. two vacancies on ONE Fe:       90 deg V_O-Fe-V_O  /  180 deg V_O-Fe-V_O
 3. two vacancies on ONE Ti:       90 deg V_O-Ti-V_O  /  180 deg V_O-Ti-V_O
Energies in meV per 160-site cell relative to the first arrangement of each group (the paper's preferred one).  Models: cluster conv (CKPT) and
the orthonormal CE (icet ridge on the AL-based set).

Inputs : data/new_bench/{A00_125.cif, dedup_stfo.json}, data/new_bench_aug467/{detailed_info.csv, mapped.pkl} (AL-based
         training set, icet-mapped), models/ce/stfo/icet_q56.ce (primitive cell), the cluster-conv model (env CKPT, default
         models/nce/stfo_dd_p10_w10_wd5_aug467_al1a_w0/model.pkl).
Outputs: $RESULTS/stfo/<OUT>.json (env OUT, default stfo_motif_experiment) and the table on stdout.
Run    : python scripts/analysis/stfo_motif_experiment.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, json, pickle, itertools, numpy as np, pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list
from icet import ClusterExpansion, ClusterSpace
from icet.tools import map_structure_to_reference
from sklearn.linear_model import RidgeCV
from neuralce.paths import DATA, MODELS, RESULTS
from neuralce.nce.mc_model import load_model, train_path_energy
CK = os.environ.get("CKPT", f"{MODELS}/nce/stfo_dd_p10_w10_wd5_aug467_al1a_w0/model.pkl")
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{DATA}/new_bench/A00_125.cif", f"{DATA}/new_bench/dedup_stfo.json", f"{DATA}/new_bench_aug467/detailed_info.csv", f"{DATA}/new_bench_aug467/mapped.pkl", f"{MODELS}/ce/stfo/icet_q56.ce", CK],
        "$NCE_DATA/new_bench*, $NCE_MODELS (CKPT)")
base = read(f"{DATA}/new_bench/A00_125.cif"); base.set_scaled_positions(np.round(base.get_scaled_positions() * 8) / 8); sym0 = np.array(base.get_chemical_symbols()); pos = base.get_positions(); cell = base.cell.array
Bs = np.where(np.isin(sym0, ["Ti", "Fe"]))[0]; Os = np.where(np.isin(sym0, ["O", "Xe"]))[0]; clean = sym0.copy(); clean[Bs] = "Ti"; clean[Os] = "O"
i, j, D = neighbor_list("ijD", base, 2.2); nbO = {b: [(int(o), D[k] / np.linalg.norm(D[k])) for k, o in enumerate(j) if i[k] == b and o in set(Os)] for b in Bs}; nbB = {int(o): [int(b) for k, b in enumerate(j) if i[k] == o and b in set(Bs)] for o in Os}
assert all(len(v) == 6 for v in nbO.values()) and all(len(v) == 2 for v in nbB.values())
mic = lambda v: ((v @ np.linalg.inv(cell) + 0.5) % 1.0 - 0.5) @ cell; dist = lambda a, b: float(np.linalg.norm(mic(pos[a] - pos[b])))
def make(fe, vac):
    s = clean.copy(); s[list(fe)] = "Fe"; s[list(vac)] = "Xe"; a = base.copy(); a.set_chemical_symbols(list(s)); return a
# Fe layout A: two Fe-Fe nearest-neighbour pairs as far apart as the cell allows;  layout B: four isolated Fe, mutually far
bb = [(int(b1), int(b2)) for b1 in Bs for b2 in Bs if b1 < b2 and abs(dist(b1, b2) - 3.96) < 0.1]
p1 = bb[0]; p2 = max(bb, key=lambda p: min(dist(p[0], q) for q in p1) + min(dist(p[1], q) for q in p1)); feA = [*p1, *p2]
bridge = lambda p: next(o for o, _ in nbO[p[0]] if p[1] in nbB[o]); oFF = [bridge(p1), bridge(p2)]
oFT = [next(o for o, _ in nbO[p[0]] if all(b not in feA or b == p[0] for b in nbB[o])) for p in (p1, p2)]
farO = sorted([int(o) for o in Os if all(b not in feA for b in nbB[o])], key=lambda o: -min(dist(o, f) for f in feA)); oTT = [farO[0], max(farO[1:40], key=lambda o: dist(o, farO[0]))]
feB = [int(Bs[0])]
while len(feB) < 4: feB.append(int(max((b for b in Bs if b not in feB), key=lambda b: min(dist(b, f) for f in feB))))
def two_on(center, fe_set, angle):
    for (o1, d1), (o2, d2) in itertools.combinations(nbO[center], 2):
        c = float(d1 @ d2)
        if (angle == 90 and abs(c) < 0.1 or angle == 180 and c < -0.9) and all(b == center or b not in fe_set for o in (o1, o2) for b in nbB[o]): return [o1, o2]
tiC = int(max((b for b in Bs if b not in feB), key=lambda b: min(dist(b, f) for f in feB)))
G = {"1. vacancy site": [("Fe-V-Fe (x2)", make(feA, oFF)), ("Fe-V-Ti (x2)", make(feA, oFT)), ("Ti-V-Ti (x2)", make(feA, oTT))],
     "2. two V on one Fe": [("90 deg V-Fe-V", make(feB, two_on(feB[0], feB, 90))), ("180 deg V-Fe-V", make(feB, two_on(feB[0], feB, 180)))],
     "3. two V on one Ti": [("90 deg V-Ti-V", make(feB, two_on(tiC, feB, 90))), ("180 deg V-Ti-V", make(feB, two_on(tiC, feB, 180)))]}
# models
df = pd.read_csv(f"{DATA}/new_bench_aug467/detailed_info.csv"); atoms = pickle.load(open(f"{DATA}/new_bench_aug467/mapped.pkl", "rb")); df["y"] = df.total_energy / df.n_atoms
dd = json.load(open(f"{DATA}/new_bench/dedup_stfo.json"))
for grp in dd["groups"]: df.loc[df.id.isin(grp), "y"] = df.loc[df.id.isin(grp), "y"].mean()
keep = ~df.id.isin(dd["drop"]); atoms = [a for a, k in zip(atoms, keep) if k]; df = df[keep].reset_index(drop=True)
cs0 = ClusterExpansion.read(f"{MODELS}/ce/stfo/icet_q56.ce").get_cluster_space_copy(); PRIM = cs0.primitive_structure; cs = ClusterSpace(PRIM, [6.0, 5.0, 4.5, 4.0], cs0.chemical_symbols)
ridge = RidgeCV(alphas=np.logspace(-6, 4, 60)).fit(np.array([cs.get_cluster_vector(a) for a in atoms]), df.y.to_numpy()); mdl = load_model(CK); csc = ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"])
out = {}
for g, items in G.items():
    e = []
    for name, a in items:
        m, _ = map_structure_to_reference(a, PRIM, inert_species=["Sr"], assume_no_cell_relaxation=False); e.append((name, float(train_path_energy(mdl, csc, m)) * 160e3, float(ridge.predict(cs.get_cluster_vector(m)[None])[0]) * 160e3))
    out[g] = [dict(arrangement=n, conv_meV=c - e[0][1], icet_meV=t - e[0][2]) for n, c, t in e]; print(g); [print(f"   {r['arrangement']:16s} conv {r['conv_meV']:+8.1f}   icet {r['icet_meV']:+8.1f}  meV per cell") for r in out[g]]
os.makedirs(f"{RESULTS}/stfo", exist_ok=True)
json.dump(dict(model=CK, groups=out), open(f"{RESULTS}/stfo/{os.environ.get('OUT', 'stfo_motif_experiment')}.json", "w"), indent=1)
