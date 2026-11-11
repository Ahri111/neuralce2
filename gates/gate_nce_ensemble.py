"""Gate for the ensemble (``--model_seed -1``) neural-CE MC mode on a small cell:
(1) the ensemble full energy equals the mean of the per-seed energies (< 1e-6 eV/atom),
(2) one local dE equals the mean of the per-seed local dE (< 1e-6, standardised units),
(3) commit keeps every seed's cache exact: a full recompute after 200 committed swaps matches the committed cache
    energy (< 1e-4, float32).
Exit 1 on failure.

Run:  python gates/gate_nce_ensemble.py [model.pkl] [init.cif]
      defaults: $MODELS/nce/nipt_v3_nomax_sel/model.pkl (3 seeds) and the 256-atom L1_0 NiPt cell
      $MODELS/structures/ptni/ref_L10_NiPt.cif
"""
import os, sys, numpy as np
from neuralce.paths import MODELS
import jax, jax.numpy as jnp
from ase.io import read
from neuralce.nce.mc_model import load_model, setup

ckpt = sys.argv[1] if len(sys.argv) > 1 else f"{MODELS}/nce/nipt_v3_nomax_sel/model.pkl"
cif = sys.argv[2] if len(sys.argv) > 2 else f"{MODELS}/structures/ptni/ref_L10_NiPt.cif"
mdl = load_model(ckpt); atoms = read(cif); S = len(mdl['params']); assert S > 1, f"ensemble gate needs several seeds, model has {S}"
mcs = [setup(mdl, atoms, seed_idx=s, dtype="float32") for s in range(S)]; me = setup(mdl, atoms, seed_idx=-1, dtype="float32")
sig = me['sig0']; N = int(me['n_atoms']); rng = np.random.default_rng(0)
spin = None
if me['spin']:
    vals = me['spin_info']['values']; spin = jnp.asarray(rng.choice(vals, size=N))
E_s = [m['energy'](sig, spin) for m in mcs]; E_e = me['energy'](sig, spin)
print("per-seed E (eV/atom):", [round(float(e), 6) for e in E_s], " ensemble:", round(float(E_e), 6), " mean:", round(float(np.mean(E_s)), 6))
ok1 = abs(E_e - np.mean(E_s)) < 1e-6
# one swap between two sites of different species (canonical order)
sig_np = np.asarray(sig); a = int(np.where(sig_np == sig_np[0])[0][0]); b = int(np.where(sig_np != sig_np[0])[0][0])
flips = jnp.asarray([[a, int(sig_np[b])], [b, int(sig_np[a])]], dtype=jnp.int32)
def one(m):
    c = m['full_cache'](sig, spin) if m['spin'] else m['full_cache'](sig)
    if m['spin']:
        sfl = jnp.asarray([spin[a], spin[b]]); dE, patch, _ = m['local_delta'](c, sig, flips, spin, sfl)
    else:
        dE, patch, _ = m['local_delta'](c, sig, flips)
    return float(dE), c, patch
d_s = [one(m)[0] for m in mcs]; dE_e, c_e, patch_e = one(me)
print("per-seed dE:", [round(x, 8) for x in d_s], " ensemble:", round(dE_e, 8), " mean:", round(float(np.mean(d_s)), 8))
ok2 = abs(dE_e - np.mean(d_s)) < 1e-6
# commit exactness: 200 random accepted swaps, compare the committed-cache energy with a fresh full recompute
c = c_e; s_cur = sig; F = me['F']
for k in range(200):
    i, j = rng.integers(N), rng.integers(N)
    if s_cur[i] == s_cur[j]: continue
    fl = jnp.asarray([[i, int(s_cur[j])], [j, int(s_cur[i])]], dtype=jnp.int32)
    if me['spin']:
        dE, patch, _ = me['local_delta'](c, s_cur, fl, spin, jnp.asarray([spin[i], spin[j]]))
    else:
        dE, patch, _ = me['local_delta'](c, s_cur, fl)
    c = me['commit'](c, patch, jnp.asarray(True)); s_cur = s_cur.at[jnp.asarray([i, j])].set(s_cur[jnp.asarray([j, i])])
E_cache = float(F['energy_of'](c, me['P'])); E_fresh = float(F['energy_of'](me['full_cache'](s_cur, spin) if me['spin'] else me['full_cache'](s_cur), me['P']))
print("after 200 committed swaps: cache E", E_cache, " fresh E", E_fresh, " diff", E_cache - E_fresh)
ok3 = abs(E_cache - E_fresh) < 1e-4
print("GATE_ENSEMBLE", "PASS" if (ok1 and ok2 and ok3) else "FAIL", ok1, ok2, ok3)
sys.exit(0 if (ok1 and ok2 and ok3) else 1)
