"""Map every structure of an STFO data set (csv + cif dir) onto the icet_q56 primitive, exactly as
scripts/train/fit_ce_stfo.py does for data/new_bench, and cache the list (csv row order) as <out>/mapped.pkl for
train_nce.py (STFO_SET=...).

Inputs : --csv detailed_info.csv, --cif_dir with <id>.cif, models/ce/stfo/icet_q56.ce (primitive).
Output : <out>/mapped.pkl.  If data/new_bench/mapped.pkl exists, the shared ids are checked to map identically.
Run    : python scripts/train/map_structures.py --csv data/new_bench_aug467/detailed_info.csv \\
             --cif_dir data/new_bench_aug467 --out data/new_bench_aug467
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import argparse, os, pickle, time
import numpy as np, pandas as pd, ase.io
from joblib import Parallel, delayed
from icet import ClusterExpansion
from icet.tools import map_structure_to_reference
from neuralce.paths import DATA, MODELS

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--cif_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--njobs", type=int, default=8); a = ap.parse_args()
for _p in (a.csv, a.cif_dir, f"{MODELS}/ce/stfo/icet_q56.ce"):                             # the STFO inputs are not shipped: skip cleanly when they are missing
    if not os.path.exists(_p):
        print(f"[skip] {os.path.basename(__file__)}: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under --csv / --cif_dir, $NCE_MODELS/ce/stfo to run", flush=True); raise SystemExit(0)
df = pd.read_csv(a.csv); ids = df.id.astype(str).tolist()
cs0 = ClusterExpansion.read(f"{MODELS}/ce/stfo/icet_q56.ce").get_cluster_space_copy()
PRIM = cs0.primitive_structure

def _map(i):
    m, info = map_structure_to_reference(ase.io.read(f"{a.cif_dir}/{i}.cif"), PRIM, inert_species=["Sr"], assume_no_cell_relaxation=False)
    return m, float(info.get("drmax", 0.0))

t0 = time.time(); res = Parallel(n_jobs=a.njobs)(delayed(_map)(i) for i in ids)
mapped = [r[0] for r in res]; drmax = max(r[1] for r in res)
os.makedirs(a.out, exist_ok=True); pickle.dump(mapped, open(f"{a.out}/mapped.pkl", "wb"))
print(f"[map] {len(ids)} structures in {time.time()-t0:.0f}s, drmax={drmax:.2e} -> {a.out}/mapped.pkl")
# consistency with the original 389-row mapping: identical numbers for the shared ids
ref = f"{DATA}/new_bench/mapped.pkl"
if os.path.exists(ref) and os.path.abspath(ref) != os.path.abspath(f"{a.out}/mapped.pkl"):
    ref_ids = pd.read_csv(f"{DATA}/new_bench/detailed_info.csv").id.astype(str).tolist(); refm = pickle.load(open(ref, "rb"))
    pos = {i: k for k, i in enumerate(ids)}; bad = 0
    for i, m0 in zip(ref_ids, refm):
        m1 = mapped[pos[i]]
        if list(m0.numbers) != list(m1.numbers) or not np.allclose(m0.positions, m1.positions, atol=1e-6): bad += 1
    print(f"[map] shared-id check vs data/new_bench/mapped.pkl: {len(ref_ids)-bad}/{len(ref_ids)} identical")
    assert bad == 0
print("MAP_DONE")
