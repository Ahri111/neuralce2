"""Start file for the two-phase C_v calculation of one PtNi composition (icet or nce): one MC job, two families of replicas
that never exchange (the driver is run with the exchange interval longer than the run):
  ordered branch    - every replica starts from the ideal ordered cell (L1_2 Pt3Ni / L1_0 NiPt / L1_2 Ni3Pt tiled K^3),
                      temperatures from 150 K up to TC0 + W_UP (superheated a little past the transition),
  disordered branch - every replica starts from its own random permutation of the same atoms,
                      temperatures from 1e8 K (random limit) down to TC0 - W_DN (supercooled a little past the transition).
Both branches are dense (DT, offset by DT/2 so no temperature is repeated) around TC0, the transition estimate from PT.
The free energy of each branch follows from integrating its energy over 1/T (ordered: from T -> 0, disordered: from T -> infinity,
where it is the exact configurational entropy); where they cross is T_c, and the finite-N C_v is rebuilt from the two branches
(scripts/mc/ptni_two_phase_analyze.py).  WINDOW_ONLY=1: dense window only (the largest cell; the far-from-T_c parts of the integral
are size independent and come from the 1M run).

Inputs: $MODELS/ce/ptni/icet_c8_6_5.ce (icet) or $MODELS/nce/nipt_v3_nomax_sel/model.pkl (nce), $MODELS/structures/ptni/ref_*.cif.
Writes <out>/Tgrid.txt, <out>/init.npz (--init_npz format, neuralce/mc/init_structures.py) and <out>/branch.json.
Env: FW=icet|conv (nce = conv), W_UP, W_DN, DENSE, DT, DT_COARSE, N_HI, WINDOW_ONLY, CE, MODEL.
Usage: python scripts/mc/ptni_two_phase_init.py <x25|x50|x75> <K = repeats of the 256-atom reference cell> <TC0 K> <out_dir>
"""
import sys, os, json, numpy as np
from ase.io import read
from icet import ClusterExpansion
from neuralce.ce.icet_mc import _rebuild_as_clean_supercell
from neuralce.paths import MODELS

KB = 8.617333262e-5
X, K, TC0, OUT = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
W_UP, W_DN = float(os.environ.get("W_UP", "12")), float(os.environ.get("W_DN", "12"))       # how far each branch is pushed past TC0
DENSE = float(os.environ.get("DENSE", "40"))                                                  # dense region: TC0 -/+ DENSE on the stable side
DT = float(os.environ.get("DT", "1"))                                                         # spacing in the dense region (disordered offset by DT/2)
DT_COARSE = float(os.environ.get("DT_COARSE", "25"))                                          # ordered branch below the dense region
N_HI = int(os.environ.get("N_HI", "40"))                                                      # disordered branch above it: N_HI - 1 points uniform in 1/T
WIN = os.environ.get("WINDOW_ONLY", "0") == "1"
REF = dict(x25="ref_L12_Pt3Ni", x50="ref_L10_NiPt", x75="ref_L12_Ni3Pt")[X]
REF_CIF = f"{MODELS}/structures/ptni/{REF}.cif"
os.makedirs(OUT, exist_ok=True)
tc = float(np.round(TC0))
if WIN:
    T_o = np.arange(tc - 20, tc + W_UP + 1e-9, DT)
    T_d = np.arange(tc - W_DN, tc + 20 + 1e-9, DT) + DT / 2
else:
    T_o = np.concatenate([np.arange(150.0, tc - DENSE, DT_COARSE), np.arange(tc - DENSE, tc + W_UP + 1e-9, DT)])
    b_top = 1.0 / (KB * (tc + DENSE))                                                          # disordered: uniform in 1/T from T -> inf to TC0 + DENSE
    b_hi = b_top * np.linspace(0.0, 1.0, N_HI + 1)[1:-1]
    T_d = np.concatenate([[1e8], 1.0 / (KB * b_hi), np.arange(tc - W_DN, tc + DENSE, DT) + DT / 2])
T = np.concatenate([T_o, T_d]); br = np.array(["ordered"] * len(T_o) + ["disordered"] * len(T_d))
o = np.argsort(T); T, br = T[o], br[o]
assert np.all(np.diff(T) > 0), "repeated temperature"
FW = os.environ.get("FW", "icet")
if FW == "nce": FW = "conv"
if FW == "icet":                                                                              # the MC cell exactly as run_pt_mc builds it
    ce = ClusterExpansion.read(os.environ.get("CE", f"{MODELS}/ce/ptni/icet_c8_6_5.ce")); prim = ce._cluster_space.primitive_structure
    atoms = _rebuild_as_clean_supercell(read(REF_CIF), prim).repeat((K, K, K))
else:                                                                                         # ... and as the nce runner builds it
    from neuralce.nce.mc_model import load_model
    from neuralce.nce.gpu_layer0_tiled import site_decompose
    from icet.tools import map_structure_to_reference
    prim = load_model(os.environ.get("MODEL", f"{MODELS}/nce/nipt_v3_nomax_sel/model.pkl"))["primitive"]; atoms = read(REF_CIF)
    try:
        site_decompose(prim, atoms, tol=1e-3)
    except AssertionError:
        atoms = map_structure_to_reference(atoms, prim, inert_species=None, assume_no_cell_relaxation=False)[0]
    atoms = atoms.repeat((K, K, K))
N = len(atoms)
z0 = atoms.numbers.astype(np.uint8); nums = np.empty((len(T), N), np.uint8)
for r in range(len(T)):
    nums[r] = z0 if br[r] == "ordered" else np.random.default_rng(1000 + r).permutation(z0)
np.savez(f"{OUT}/init.npz", numbers=nums, positions=atoms.get_positions(), cell=atoms.cell.array, temps=T)
np.savetxt(f"{OUT}/Tgrid.txt", T, fmt="%.6f", header=f"two-phase grid {X} K={K} N={N} TC0={TC0} W_UP={W_UP} W_DN={W_DN} window_only={WIN}")
json.dump(dict(fw=FW, x=X, K=K, N=N, TC0=TC0, W_UP=W_UP, W_DN=W_DN, window_only=WIN, ref=REF, temps=T.tolist(), branch=br.tolist(),
               n_Ni=int((z0 == 28).sum())), open(f"{OUT}/branch.json", "w"))
print(f"{X} N={N}: {int((br == 'ordered').sum())} ordered replicas ({T_o.min():.0f}-{T_o.max():.0f} K), {int((br == 'disordered').sum())} disordered "
      f"({T_d.min():.1f} K - {T_d.max():.3g} K); {len(T)} temperatures -> {OUT}", flush=True)
