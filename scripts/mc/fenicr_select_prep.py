"""Fe56Cr21Ni23 initial cell for the FeNiCr model selection: 5x5x5 conventional fcc (a = 3.6, the model primitive's
lattice), 500 atoms = Fe 280 / Ni 115 / Cr 105 (exactly 0.56/0.23/0.21), random, seed 0.  The MC tiles it 2x2x2
(--supercell, species re-shuffled) -> 4000 atoms = the 10x10x10 cell of Su et al. (Acta Mater. 2024).

Env: OUTDIR (default $RUNS/fenicr_select).  Writes OUTDIR/init_Fe56Cr21Ni23_500.cif.
Run: python scripts/mc/fenicr_select_prep.py
"""
import os, numpy as np
from ase.build import bulk
from ase.io import write
from neuralce.paths import RUNS

OUT = os.environ.get("OUTDIR", f"{RUNS}/fenicr_select"); os.makedirs(OUT, exist_ok=True)
a = bulk("Fe", "fcc", a=3.6, cubic=True).repeat((5, 5, 5)); n = len(a); rng = np.random.RandomState(0)
Z = np.array([26] * 280 + [28] * 115 + [24] * 105); assert len(Z) == n; rng.shuffle(Z); a.numbers = Z
write(f"{OUT}/init_Fe56Cr21Ni23_500.cif", a); print("PREP_DONE", n, {s: int((Z == z).sum()) for s, z in (("Fe", 26), ("Ni", 28), ("Cr", 24))})
