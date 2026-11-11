"""icet cluster expansions for PtNi on the SAME 121 FINAL_dedup labels / snapped ideal-lattice structures as the
neural cluster expansion, fitted on all data with ARDR (threshold 1e5, the heavy-fit recipe), one per cutoff set.

Inputs : data/nipt/icet_csv_FINAL_dedup.csv, cifs from $NIPT_CIF_DIR (default data/nipt/cifs).
Outputs: <out>/<tag>.ce (tag = icet_c<cutoffs>) and <out>/fits.json with the model energies of the ideal
         L1_2 / L1_0 / DO22 / L1_1 references.  Default <out> = models/ce/ptni (--out overrides).
Env    : CUTS       cutoff sets, e.g. "8,6,5;10,8,6" (default both)
         ECI_SCALE  factor applied to the parameters before writing (default 1 = per-atom energies, the fit itself).
                    The shipped models/ce/ptni/icet_c8_6_5.ce carries ECI_SCALE=256 (recorded as "eci_scale" in
                    fits.json): its parameters give the TOTAL energy of the 256-atom cell that the MC framework
                    (`neuralce.mc.run --framework icet`) uses as its base cell.  Reproduce it with ECI_SCALE=256.

Run:  python scripts/train/fit_ce_ptni.py [--out DIR]
"""
import os, json, argparse, numpy as np, pandas as pd
from ase.build import bulk
from ase.io import read
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from sklearn.linear_model import ARDRegression
from neuralce.paths import DATA, MODELS
from neuralce.nce.features import field_from_atoms, atoms_from_field, CIF_DIR

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{MODELS}/ce/ptni", help="output directory for the .ce files and fits.json")
args = ap.parse_args()
OUT = args.out; os.makedirs(OUT, exist_ok=True)
ECI_SCALE = float(os.environ.get("ECI_SCALE", "1"))
df = pd.read_csv(f"{DATA}/nipt/icet_csv_FINAL_dedup.csv")
atoms = [atoms_from_field(field_from_atoms(read(f"{CIF_DIR}/{i}.cif"))[0]) for i in df.id.astype(str)]
y = (df.total_energy / df.n_atoms).to_numpy()
prim = bulk("Ni", "fcc", a=3.924)
# ideal references in a 4x4x4 cell
conv = bulk("Ni", "fcc", a=3.924, cubic=True).repeat((4, 4, 4)); frac = conv.get_scaled_positions() * 4
code = ((np.round(frac * 2).astype(int) % 2) * np.array([1, 2, 4])).sum(1); sub = np.array([{0: 0, 3: 1, 5: 2, 6: 3}[int(c)] for c in code])
h = np.round(frac.sum(1) * 2).astype(int); zc = np.floor(frac[:, 2] + 1e-6).astype(int)
def mk(mask): a = conv.copy(); a.set_chemical_symbols(np.where(mask, "Pt", "Ni").tolist()); return a
REFS = {"L12_Ni3Pt": mk(sub == 0), "DO22_Ni3Pt": mk(np.where((zc // 2) % 2 == 0, sub == 0, sub == 1)), "L12_Pt3Ni": mk(sub != 0),
        "DO22_Pt3Ni": mk(~np.where((zc // 2) % 2 == 0, sub == 0, sub == 1)), "L10_NiPt": mk(np.isin(sub, [0, 1])), "L11_NiPt": mk((h // 2) % 2 == 0)}
res = {}
for cut in os.environ.get("CUTS", "8,6,5;10,8,6").split(";"):
    cutoffs = [float(c) for c in cut.split(",")]; tag = "icet_c" + "_".join(f"{c:g}" for c in cutoffs)
    cs = ClusterSpace(prim, cutoffs, [["Ni", "Pt"]]); sc = StructureContainer(cs)
    for a, e in zip(atoms, y): sc.add_structure(a, properties={"energy": e})
    X, t = sc.get_fit_data(key="energy")
    ard = ARDRegression(max_iter=300, fit_intercept=False, threshold_lambda=1e5, compute_score=False).fit(X, t)
    ce = ClusterExpansion(cs, ard.coef_)                      # per-atom energies: used for the reference energies below
    ClusterExpansion(cs, ard.coef_ * ECI_SCALE).write(f"{OUT}/{tag}.ce")
    pred = X @ ard.coef_; mae = np.mean(np.abs(pred - t)) * 1e3
    r = {"cutoffs": cutoffs, "n_params": len(cs), "n_nonzero": int((np.abs(ard.coef_) > 1e-8).sum()), "train_mae_meV": float(mae),
         "refs": {k: float(ce.predict(a)) for k, a in REFS.items()}}
    if ECI_SCALE != 1.0: r["eci_scale"] = ECI_SCALE
    res[tag] = r
    print(f"{tag}: {len(cs)} params, {r['n_nonzero']} nonzero, train MAE {mae:.2f} meV/atom | " + "  ".join(f"{k} {v:.5f}" for k, v in r["refs"].items()), flush=True)
json.dump(res, open(f"{OUT}/fits.json", "w"), indent=1); print(f"wrote {OUT}/fits.json"); print("ICET_FIT_DONE")
