"""GATE: the fixed layer-1 cluster convolution (neuralce.nce.conv on an icet kernel bank) must reproduce icet's
cluster vector and energy on random binary decorations of 3x3x3, 4x4x4 and 2x3x4 supercells of the primitive cell:
worst max|cv_conv - cv_icet| < 1e-10 and worst |E_conv - ce.predict| < 1e-8 eV.  Exit 1 on failure.

Run:  python gates/gate_nce_icet_exact.py [file.ce]
      default: $MODELS/ce/ptni/icet_c8_6_5.ce (any BINARY cluster expansion works; the point function is theta_binary)
"""
import os, sys, numpy as np
from neuralce.paths import MODELS
import jax
jax.config.update("jax_enable_x64", True)   # cluster vectors are exact rationals
import jax.numpy as jnp
from icet import ClusterExpansion
from neuralce.nce.kernels import extract_kernels, supercell_index
from neuralce.nce.conv import cluster_conv, theta_binary

CE_PATH = sys.argv[1] if len(sys.argv) > 1 else f"{MODELS}/ce/ptni/icet_c8_6_5.ce"

ce = ClusterExpansion.read(CE_PATH)
cs = ce._cluster_space
prim = cs.primitive_structure
bank = extract_kernels(ce)
assert len(bank['species']) == 2, f"binary cluster expansion expected, got species {bank['species']}"
print(f"[bank] {len(bank['kernels'])} kernels, n_prim={bank['n_prim']}, "
      f"species={bank['species']}")

rng = np.random.default_rng(0)
worst_cv = worst_e = 0.0
for m in [(3, 3, 3), (4, 4, 4), (2, 3, 4)]:
    sc = prim.repeat(m)
    idx = supercell_index(m, bank['n_prim'])
    for trial in range(3):
        occ = rng.integers(0, 2, len(sc))
        sc.set_chemical_symbols([bank['species'][o] for o in occ])
        cv_ref = np.asarray(cs.get_cluster_vector(sc))
        e_ref = float(ce.predict(sc))                    # intensive (cv is per-site)

        field = jnp.asarray(theta_binary(occ[idx].astype(np.float64)))
        cv_me = np.array([float(cluster_conv(field, k['clusters']).mean()) / k['mult']
                          for k in bank['kernels']])
        e_me = bank['zerolet'] + float(np.dot([k['eci'] for k in bank['kernels']], cv_me))

        d_cv = np.abs(cv_me - cv_ref[1:]).max()
        d_e = abs(e_me - e_ref)
        worst_cv = max(worst_cv, d_cv); worst_e = max(worst_e, d_e)
        print(f"  m={m} trial={trial}  max|dcv|={d_cv:.3e}  |dE|={d_e:.3e} eV "
              f"(E={e_ref:.6f})")

print(f"\n[GATE] worst max|dcv| = {worst_cv:.3e}")
print(f"[GATE] worst |dE|     = {worst_e:.3e} eV")
ok = worst_cv < 1e-10 and worst_e < 1e-8
print("[GATE]", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
