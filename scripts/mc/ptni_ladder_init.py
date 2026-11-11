"""Random fcc Ni/Pt starting cells for the PtNi size ladder.

M x M x M conventional cells (4 M^3 atoms, a = 3.924 A) at x_Ni = 0.25 / 0.50 / 0.75, Ni listed first, with the
same RNG recipe as ptni_select_prep.py (M = 4 reproduces its init_x*.cif).

Env:  M       space-separated list of M (default "4 8 16 32 64")
      OUTDIR  output directory (default $RUNS/ptni_ladder/init)
Writes OUTDIR/init_M<M>_x{25,50,75}.cif.   Run: python scripts/mc/ptni_ladder_init.py
"""
import os, numpy as np
from ase.build import bulk
from ase.io import write
from neuralce.paths import RUNS

OUT = os.environ.get("OUTDIR", f"{RUNS}/ptni_ladder/init"); os.makedirs(OUT, exist_ok=True)
for M in [int(m) for m in os.environ.get("M", "4 8 16 32 64").split()]:
    conv = bulk("Ni", "fcc", a=3.924, cubic=True).repeat((M, M, M)); n = len(conv); rng = np.random.RandomState(0)
    for x, nm in ((0.25, "x25"), (0.5, "x50"), (0.75, "x75")):
        syms = np.array(["Pt"] * n); syms[rng.choice(n, int(round(x * n)), replace=False)] = "Ni"
        a = conv.copy(); a.set_chemical_symbols(list(syms)); a = a[np.argsort([0 if s == "Ni" else 1 for s in syms], kind="stable")]
        write(f"{OUT}/init_M{M}_{nm}.cif", a); print(M, nm, n, flush=True)
print("LADDER_INIT_DONE")
