"""Data of Fig. 5a: why the orthonormal CE (icet) collapses in STFO while the cluster convolution, built on the SAME clusters, does not.
A path of 160-site configurations from the DFT ground state A00 (x = 0.5) to the state the icet MC fell into (icet_min.cif), one composition-
conserving swap at a time (Fe<->Ti on the B sites, O<->V_O on the O sites).  For every configuration: the number of V_O-V_O 90-degree pairs
(the "vacancy clustering" coordinate), the icet energy and the cluster-convolution energy, both relative to A00.  The training data
(all x = 0.5 structures) are shown as the range of that coordinate they cover.  DFT points: A00 (0) and, when available, the icet minimum.
Environment knobs:
  ICET_CE  the orthonormal CE (.ce, framework convention x160; default models/ce/stfo/icet_aug467_al2b_c6_5_4.5_4.ce)
  CONV     model tag of the cluster convolution (models/nce/<CONV>/model.pkl; default stfo_dd_p10_w10_wd5_aug467_al1a_w0)
  MIN      cif of the state the icet Monte Carlo fell into (default $NCE_RUNS/stfo_mc/icet_aug467_al2b/c500_K1/icet_min.cif; raw MC output, not shipped)
  DFT_RECHECK  directory of the DFT recheck runs of that state (default $NCE_RUNS/DFT_recheck; optional, not shipped)
  NPATH    number of random swap orders (default 4)
Inputs:  data/new_bench/A00_500.cif, data/new_bench_aug467/{detailed_info.csv, mapped.pkl} (training structures at x = 0.5).
Outputs: results/figure_data/fig7_stfo_extrap.json (read by scripts/figures/fig5_stf_fenicr.py; the shipped file is the one used for the paper).
Run:     python scripts/analysis/stfo_extrapolation.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above."""
import os, re, glob, json, pickle, numpy as np
import pandas as pd
from neuralce.paths import DATA, MODELS, RUNS, RESULTS
from ase.io import read
from ase.neighborlist import neighbor_list
from icet import ClusterExpansion, ClusterSpace
from neuralce.nce.mc_model import load_model, train_path_energy
CE = os.environ.get("ICET_CE", f"{MODELS}/ce/stfo/icet_aug467_al2b_c6_5_4.5_4.ce"); TAG = os.environ.get("CONV", "stfo_dd_p10_w10_wd5_aug467_al1a_w0")
MIN = os.environ.get("MIN", f"{RUNS}/stfo_mc/icet_aug467_al2b/c500_K1/icet_min.cif"); NPATH = int(os.environ.get("NPATH", "4")); OUT = f"{RESULTS}/figure_data/fig7_stfo_extrap"
RECHECK = os.environ.get("DFT_RECHECK", f"{RUNS}/DFT_recheck")
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([CE, f"{MODELS}/nce/{TAG}/model.pkl", f"{DATA}/new_bench/A00_500.cif", f"{DATA}/new_bench_aug467/detailed_info.csv", f"{DATA}/new_bench_aug467/mapped.pkl", MIN],
        "$NCE_DATA / $NCE_MODELS / $NCE_RUNS (ICET_CE, CONV, MIN)")
ce = ClusterExpansion.read(CE); mdl = load_model(f"{MODELS}/nce/{TAG}/model.pkl"); csc = ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"])
a0 = read(f"{DATA}/new_bench/A00_500.cif"); am = read(MIN); assert len(a0) == len(am) and np.allclose(a0.get_scaled_positions(), am.get_scaled_positions(), atol=1e-3), "same cell / atom order expected"
N = len(a0); s0 = np.array(a0.get_chemical_symbols()); sm = np.array(am.get_chemical_symbols())


def vv90(a):
    """vacancy-clustering coordinate: number of V_O-V_O pairs within 4.1 A (90-degree 2.8 A and 180-degree 3.96 A neighbours)"""
    i, j, d = neighbor_list("ijd", a, 4.1); s = np.array(a.get_chemical_symbols()); m = np.isin(s, ["X", "Xe"]); return int((m[i] & m[j]).sum() // 2)


def energies(a):
    return float(ce.predict(a)) / N * 1e3, float(train_path_energy(mdl, csc, a, None, 0)) * 1e3            # meV/site: icet (x160 convention -> total / N), conv


e_ic0, e_cv0 = energies(a0); print(f"A00: icet {e_ic0:.2f} conv {e_cv0:.2f} meV/site; V-V 90deg pairs {vv90(a0)}", flush=True)
# swaps that turn A00 into the icet minimum, per sublattice: sites that must change species, paired (one gains Fe, one loses Fe; one gains V_O, one loses V_O)
def pairs(sub, sp_from, sp_to):
    gain = [k for k in range(N) if s0[k] in sub and s0[k] == sp_from and sm[k] == sp_to]; lose = [k for k in range(N) if s0[k] in sub and s0[k] == sp_to and sm[k] == sp_from]
    assert len(gain) == len(lose), (sp_from, sp_to, len(gain), len(lose)); return list(zip(gain, lose))
swB = pairs(("Ti", "Fe"), "Ti", "Fe"); swO = pairs(("O", "Xe"), "O", "Xe"); print(f"{len(swB)} B-site swaps, {len(swO)} O-site swaps to reach the icet minimum", flush=True)
paths = []
for rep in range(NPATH):
    rng = np.random.default_rng(rep); order = list(swB) + list(swO); rng.shuffle(order); a = a0.copy(); sy = list(s0); rows = [dict(step=0, vv=vv90(a), icet=0.0, conv=0.0)]
    for t, (k1, k2) in enumerate(order, 1):
        sy[k1], sy[k2] = sy[k2], sy[k1]; a.set_chemical_symbols(sy); ei, ec = energies(a); rows.append(dict(step=t, vv=vv90(a), icet=ei - e_ic0, conv=ec - e_cv0))
    paths.append(rows); print(f"path {rep}: end icet {rows[-1]['icet']:+.1f} conv {rows[-1]['conv']:+.1f} meV/site, V-V pairs {rows[0]['vv']} -> {rows[-1]['vv']}", flush=True)
# training data at x = 0.5: vacancy-clustering coordinate and DFT energy relative to A00 (same protocol)
df = pd.read_csv(f"{DATA}/new_bench_aug467/detailed_info.csv"); atoms = pickle.load(open(f"{DATA}/new_bench_aug467/mapped.pkl", "rb"))
tr = [(vv90(at), (row.total_energy / row.n_atoms) * 1e3) for at, (_, row) in zip(atoms, df.iterrows()) if int(row.comp) == 500]
E_A00 = df[df.id.astype(str) == "A00_500"].total_energy.iloc[0] / 160 * 1e3 if (df.id.astype(str) == "A00_500").any() else min(e for _, e in tr)
train = [(v, e - E_A00) for v, e in tr]
dft = {"A00_500": (vv90(a0), 0.0)}
for d in sorted(glob.glob(f"{RECHECK}/ICETMIN_500_r*")):                                     # DFT of the icet minimum, when finished
    osz = f"{d}/OSZICAR"
    if os.path.exists(osz):
        E = [float(x) for x in re.findall(r"E0=\s*([-\d.E+]+)", open(osz).read())]; conv_ok = os.path.exists(f"{d}/OUTCAR") and "reached required accuracy" in open(f"{d}/OUTCAR").read()
        if E and conv_ok: dft[os.path.basename(d)] = (vv90(read(f"{d}/mc_structure.cif")), E[-1] / 160 * 1e3 - E_A00)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(icet_ce=CE, conv=TAG, minimum=MIN, paths=paths, train=train, dft=dft, endpoint=dict(icet=paths[0][-1]["icet"], conv=paths[0][-1]["conv"])), open(OUT + ".json", "w"), indent=1); print("wrote", OUT + ".json", "DFT points:", dft)
