"""STFO active learning on the conv MC: collect the finished DFT relaxations of MC-visited structures ($RUNS/DFT_recheck/<prefix>_*),
compare with the models, and write the EXTRA training file for the next model (scripts/train/train_nce.py, env EXTRA).
  python scripts/analysis/stfo_al_compile.py <out stem> <prefix> [<prefix> ...]      e.g.  results/stfo/stfo_al_round1  MCMIN ALMC1
Label = E0 of the last ionic step / 160 sites (same rule as the benchmark labels), only if VASP reports 'reached required accuracy'.
Model columns: the MC model (env CKPT) and the icet ridge re-fitted on the AL-based set (or a given icet CE, env ICET_CE); all relative
to A00 of the composition.

Inputs : VASP output directories $RUNS/DFT_recheck/<prefix>_<comp>_<...>/{OUTCAR, OSZICAR[.run*], mc_structure.cif} (env DFT_DIR
         overrides the parent; not part of this repository), MC snapshot cifs of the MCMIN round in $RUNS/stfo_mc/<MC model tag>/
         (env MC_DIR), data/stfo_pair_structures/stfo_complex_pair_struct_<nn>.cif for the PAIR_ prefix,
         data/new_bench_aug467/{detailed_info.csv, mapped.pkl, label_corrections_al.json}, data/new_bench/{A00_<comp>.cif,
         dedup_stfo.json, label_corrections.json}, models/ce/stfo/icet_q56.ce, the MC model models/nce/<tag>/model.pkl
         (env CKPT; default tag stfo_dd_p10_w10_wd5_aug467_lab17, which is not shipped -- set CKPT to one of the models/nce tags).
Outputs: <out stem>.csv (per-structure table; the stfo_al_round*.csv files in results/stfo/ were produced this way) and
         <out stem>_extra.pkl (atoms, y, comp, ids of the new labels).
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, re, sys, glob, json, pickle, numpy as np, pandas as pd
from ase.io import read
from icet import ClusterExpansion, ClusterSpace
from icet.tools import map_structure_to_reference
from sklearn.linear_model import RidgeCV
from neuralce.paths import DATA, MODELS, RUNS
from neuralce.nce.mc_model import load_model, train_path_energy
def require(paths, hint):                                                                 # the STFO inputs are not shipped: skip cleanly when they are missing
    for p in paths:
        if not os.path.exists(p): print(f"[skip] {os.path.basename(__file__)}: {p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under {hint} to run", flush=True); raise SystemExit(0)
require([f"{DATA}/new_bench_aug467/detailed_info.csv", f"{DATA}/new_bench_aug467/mapped.pkl", f"{DATA}/new_bench/dedup_stfo.json", f"{MODELS}/ce/stfo/icet_q56.ce",
         os.environ.get("CKPT", f"{MODELS}/nce/stfo_dd_p10_w10_wd5_aug467_lab17/model.pkl")], "$NCE_DATA/new_bench*, $NCE_MODELS (CKPT)")
out = sys.argv[1]; prefixes = sys.argv[2:]; CK = os.environ.get("CKPT", f"{MODELS}/nce/stfo_dd_p10_w10_wd5_aug467_lab17/model.pkl")
MC = os.environ.get("MC_DIR", f"{RUNS}/stfo_mc/stfo_dd_p10_w10_wd5_aug467_lab17"); DFT = os.environ.get("DFT_DIR", f"{RUNS}/DFT_recheck")
SRC = {"MCMIN_125_0": f"{MC}/c125_K1/final_T300_r0.cif", "MCMIN_500_0": f"{MC}/c500_K1/final_T300_r0.cif", "MCMIN_500_1": f"{MC}/c500_K1/final_T319_r1.cif"}
df = pd.read_csv(f"{DATA}/new_bench_aug467/detailed_info.csv"); atoms = pickle.load(open(f"{DATA}/new_bench_aug467/mapped.pkl", "rb")); df["y"] = df.total_energy / df.n_atoms
dd = json.load(open(f"{DATA}/new_bench/dedup_stfo.json"))
for grp in dd["groups"]: df.loc[df.id.isin(grp), "y"] = df.loc[df.id.isin(grp), "y"].mean()
keep = ~df.id.isin(dd["drop"]); atoms = [a for a, k in zip(atoms, keep) if k]; df = df[keep].reset_index(drop=True); fx = dict(json.load(open(f"{DATA}/new_bench/label_corrections.json"))["corrections"]); fx.update(json.load(open(f"{DATA}/new_bench_aug467/label_corrections_al.json"))["corrections"])
for sid, e in fx.items(): df.loc[df.id.astype(str) == sid, "y"] = float(e)
cs0 = ClusterExpansion.read(f"{MODELS}/ce/stfo/icet_q56.ce").get_cluster_space_copy(); PRIM = cs0.primitive_structure; cs = ClusterSpace(PRIM, [6.0, 5.0, 4.5, 4.0], cs0.chemical_symbols)
if os.environ.get("ICET_CE"):                                                                  # a given icet CE (framework convention: parameters x 160 -> per site = predict / 160) instead of the ridge refit
    _ce = ClusterExpansion.read(os.environ["ICET_CE"]); cs = _ce.get_cluster_space_copy()
    ridge = None
else: ridge = RidgeCV(alphas=np.logspace(-6, 4, 60)).fit(np.array([cs.get_cluster_vector(a) for a in atoms]), df.y.to_numpy())
ipred = (lambda m_: float(_ce.predict(m_)) / 160.0) if ridge is None else (lambda m_: float(ridge.predict(cs.get_cluster_vector(m_)[None])[0]))
mdl = load_model(CK); csc = ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"])
ref = {}
for comp in sorted(set(df.comp.astype(int))):
    m, _ = map_structure_to_reference(read(f"{DATA}/new_bench/A00_{comp}.cif"), PRIM, inert_species=["Sr"], assume_no_cell_relaxation=False)
    ref[comp] = dict(dft=float(df.loc[df.id == f"A00_{comp}", "y"].iloc[0]), conv=float(train_path_energy(mdl, csc, m)), icet=ipred(m))
rows, ex, seen_cv = [], dict(atoms=[], y=[], comp=[], ids=[]), {}
for d in sorted(p for pre in prefixes for p in glob.glob(f"{DFT}/{pre}_*") if os.path.isdir(p)):
    name = os.path.basename(d); comp = 125 if name.startswith("PAIR_") else int(name.split("_")[1]); outcar = open(f"{d}/OUTCAR", errors="ignore").read() if os.path.exists(f"{d}/OUTCAR") else ""
    E = [float(m.group(1)) for f in sorted(glob.glob(f"{d}/OSZICAR.run*")) + [f"{d}/OSZICAR"] if os.path.exists(f) for m in re.finditer(r"E0=\s*([-\d.E+]+)", open(f).read())]
    conv_ok = "reached required accuracy" in outcar; src = f"{d}/mc_structure.cif" if os.path.exists(f"{d}/mc_structure.cif") else SRC.get(name)
    if name.startswith("PAIR_"): src = f"{DATA}/stfo_pair_structures/stfo_complex_pair_struct_{int(name.split('_')[1]):02d}.cif"
    mags = re.findall(r"mag=\s*([-\d.]+)", open(f"{d}/OSZICAR").read()) if os.path.exists(f"{d}/OSZICAR") else []
    nfe = int(round(32 * comp / 1000.0)); r = dict(dir=name, comp=comp, ionic_steps=len(E), converged=conv_ok, mag=float(mags[-1]) if mags else np.nan)
    r["FM"] = bool(mags) and abs(float(mags[-1]) - 5.0 * nfe) < 0.6                     # project rule: labels come from the ferromagnetic high-spin state (5 mu_B per Fe)
    if src:
        m, _ = map_structure_to_reference(read(src), PRIM, inert_species=["Sr"], assume_no_cell_relaxation=False)
        r["conv_rel"] = (float(train_path_energy(mdl, csc, m)) - ref[comp]["conv"]) * 1e3; r["icet_rel"] = (ipred(m) - ref[comp]["icet"]) * 1e3
        if E and conv_ok:
            y = E[-1] / 160.0; r["E_dft"] = y; r["dft_rel"] = (y - ref[comp]["dft"]) * 1e3; cvv = tuple(np.round(cs.get_cluster_vector(m), 8)); r["duplicate_of"] = seen_cv.get(cvv, "")
            if r["FM"] and cvv not in seen_cv: ex["atoms"].append(m); ex["y"].append(y); ex["comp"].append(comp); ex["ids"].append(name)      # FM only, one label per distinct configuration
            if r["FM"]: seen_cv.setdefault(cvv, name)
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(out + ".csv", index=False); pickle.dump(ex, open(out + "_extra.pkl", "wb"))
print(T[T.ionic_steps > 0].round(3).to_string(index=False)); d_ = T.dropna(subset=["dft_rel"]) if "dft_rel" in T else T.iloc[:0]; d_ = d_[d_.FM]
if len(d_): print(f"\n{len(d_)} converged and ferromagnetic: MAE vs DFT (relative to A00)  conv {np.abs(d_.conv_rel - d_.dft_rel).mean():.2f}  icet {np.abs(d_.icet_rel - d_.dft_rel).mean():.2f} meV/site;  mean signed error conv {np.mean(d_.conv_rel - d_.dft_rel):+.2f}  icet {np.mean(d_.icet_rel - d_.dft_rel):+.2f}")
print(f"wrote {out}.csv and {out}_extra.pkl ({len(ex['y'])} labelled structures)")
