"""Gates for the SPIN neural-CE Monte Carlo path (FeNiCr; SPIN=1 Heisenberg-pair or SPIN>=2 models).

S1  MC path == training path on training structures WITH their DFT spins (float64): max|E_mc - E_train| < 1e-8
    eV/atom; the MC-path MAE on those structures is printed.
S2  local dE == full dE for spin moves and for swaps carrying spins on the largest picked structure: float64 within
    1e-11 and float32 within 2e-5 (standardised units); the cache after 40 committed moves == full recompute within
    10x that.
S3  central entry (``python -m neuralce.mc.run --framework nce``): single-T run with spin moves, drift_max < 1e-4,
    final_spin.npy written and the final extxyz carries the same moments; PT smoke with exchanges.

Inputs:  MODEL (default $MODELS/nce/fenicr_s2b_mm_c65_d1/model.pkl), $DATA/feni_cr/{detailed_info.csv,
         folds_decoration.json, mapped.pkl}.
Outputs: $RUNS/gates/<tag>_gate_mc/ and <tag>_gate_pt/ (S3 MC output; the init cif is written into the first).
Run:     python gates/gate_nce_spin_mc.py        (GPU; runs on CPU with JAX_PLATFORMS=cpu, slowly)
Env:     MODEL (model.pkl path), NSTRUCT (S1 structure count, default 12).
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import os, sys, json, subprocess, pickle, numpy as np, pandas as pd
from neuralce.paths import REPO, DATA, MODELS, RUNS
# S3 runs the central entry as a child process on the SAME GPU; with JAX's default 75% preallocation the parent owns
# the memory and the child dies in cuBLAS init ("Failed to initialize BLAS support for GemmCmd").  Both grow on demand.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
MODEL = os.environ.get("MODEL", f"{MODELS}/nce/fenicr_s2b_mm_c65_d1/model.pkl")
for _p in (MODEL, f"{DATA}/feni_cr/detailed_info.csv", f"{DATA}/feni_cr/folds_decoration.json", f"{DATA}/feni_cr/mapped.pkl"):   # the Fe-Ni-Cr inputs are not shipped: skip cleanly
    if not os.path.exists(_p):
        print(f"[skip] {os.path.basename(__file__)}: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_MODELS/nce (MODEL), $NCE_DATA/feni_cr to run", flush=True); sys.exit(0)
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from icet import ClusterSpace
from neuralce.nce.mc_model import load_model, setup, gate_final
from neuralce.nce.general import enlarge_if_self_interacting
PY = sys.executable
def log(*a): print(*a, flush=True)

mdl = load_model(MODEL); assert mdl.get('spin', 0) >= 1, "spin model expected"
tag = os.path.basename(os.path.dirname(MODEL))
cs = ClusterSpace(mdl['primitive'], list(mdl['cutoffs']), mdl['chemical_symbols'])
log(f"==== {tag}: spin={mdl['spin']} depth {mdl['depth']} channels {len(mdl['keep'])}/{mdl['K_chem']}+{mdl['n_spin']}  n_train {mdl['n_train']} ====")

# training structures (mapped, ideal fcc frame) by id, enlarged exactly as at training time; spins from the model
df = pd.read_csv(f"{DATA}/feni_cr/detailed_info.csv")
spec = json.load(open(f"{DATA}/feni_cr/folds_decoration.json"))
df = df[~df.id.astype(str).isin(set(map(str, spec["dropped_exact_duplicates"])))].reset_index(drop=True)
atoms_all = pickle.load(open(f"{DATA}/feni_cr/mapped.pkl", "rb")); by_id = dict(zip(df.id.astype(str), atoms_all))
ids = list(mdl['train_ids']); y = np.asarray(mdl['train_y']); spins_all = mdl['train_spins']
n_at = np.array([len(by_id[i]) for i in ids])
NS = int(os.environ.get("NSTRUCT", "12"))
pick = list(np.linspace(0, len(ids) - 1, NS).astype(int)); pick += [int(np.where(n_at == 108)[0][0]), int(np.where(n_at == 32)[0][0]), int(np.argmin(n_at))]
pick = sorted(set(pick))
A, S = [], []
for k in pick:
    a, _, r, _ = enlarge_if_self_interacting(cs, by_id[ids[k]], lambda x: np.arange(len(x)), None)
    assert len(a) == len(spins_all[k]), (len(a), len(spins_all[k]), r)
    A.append(a); S.append(np.asarray(spins_all[k], float))
mae_mc, dmax = gate_final(mdl, A, y[pick], spins_list=S)
log(f"  [S1] MC path vs training path on {len(pick)} training structures (sizes {sorted(set(len(a) for a in A))}): max|dE| = {dmax:.2e}   MC-path MAE {mae_mc:.3f} meV/atom")
assert dmax < 1e-8, dmax

# S2: local vs full dE on the largest picked structure
k108 = int(np.argmax([len(a) for a in A])); a = A[k108]; sp0 = S[k108]
rng = np.random.RandomState(0)
for dtype, tol in (("float64", 1e-11), ("float32", 2e-5)):
    mc = setup(mdl, a, dtype=(None if dtype == "float64" else dtype)); F = mc['F']; tab = mc['tab']; P = mc['P']
    dt = jnp.float64 if dtype == "float64" else jnp.float32
    sig = mc['sig0']; spin = jnp.asarray(sp0, dt); vals = mc['spin_info']['values']
    cache = F['full_cache'](tab, P, sig, spin); E0 = float(F['energy_of'](cache, P))
    errs = {"spin": [], "swap": []}; N = len(a)
    for t in range(40):
        if t % 2 == 0:                                                     # spin move
            i = rng.randint(N); cur = float(spin[i]); new = float(rng.choice([v for v in vals if v != cur]))
            flips = jnp.array([[i, int(sig[i])], [i, int(sig[i])]]); sfl = jnp.array([new, new], dt); kind = "spin"
        else:                                                              # swap with spins travelling
            i, j = rng.choice(N, 2, replace=False)
            flips = jnp.array([[i, int(sig[j])], [j, int(sig[i])]]); sfl = jnp.array([spin[j], spin[i]], dt); kind = "swap"
        dE, patch, _ = F['local_delta'](tab, P, cache, sig, flips, spin, sfl)      # local_delta does not return a flipped copy
        sig1 = sig.at[flips[:, 0]].set(flips[:, 1].astype(sig.dtype)); spin1 = spin.at[flips[:, 0]].set(sfl)
        E1 = float(F['energy_of'](F['full_cache'](tab, P, sig1, spin1), P))
        errs[kind].append(abs(float(dE) - (E1 - E0)))
        # commit every move (chain), then compare the cache energy with a full recompute at the end
        cache = F['commit'](cache, patch, jnp.asarray(True)); sig = sig.at[flips[:, 0]].set(flips[:, 1].astype(sig.dtype)); spin = spin1; E0 = float(F['energy_of'](cache, P))
    Efull = float(F['energy_of'](F['full_cache'](tab, P, sig, spin), P))
    log(f"  [S2:{dtype}] |dE_local - dE_full|: spin moves max {max(errs['spin']):.2e}, swaps max {max(errs['swap']):.2e}; "
        f"cache after 40 committed moves vs full: {abs(E0 - Efull):.2e} (std units)")
    assert max(errs['spin']) < tol and max(errs['swap']) < tol and abs(E0 - Efull) < tol * 10, (errs, E0 - Efull)

# S3: central entry
from ase.io import write
GATE_DIR = f"{RUNS}/gates"
out = f"{GATE_DIR}/{tag}_gate_mc"; os.makedirs(out, exist_ok=True)
cif = f"{out}/init_{ids[k108]}.cif"; write(cif, a)
cmd = [PY, "-m", "neuralce.mc.run", "--framework", "nce", "--ckpt", MODEL, "--init_cif", cif, "--T", "800", "--n_steps", "4000",
       "--chunk", "500", "--n_replicas", "4", "--log_every", "1000", "--out_dir", out, "--energy_mode", "local", "--spin_move_prob", "0.5", "--spin_init", "random"]
log("  [S3] " + " ".join(os.path.basename(x) if x.startswith("/") else x for x in cmd))
r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, NCE_REPO=REPO))
log("\n".join(l for l in r.stdout.splitlines() if "[ce_cnn" in l)[-1500:]); assert r.returncode == 0, r.stderr[-3000:]
s = json.load(open(f"{out}/mcmc_summary.json")); log(f"  [S3] single-T: steps/s {s['steps_per_s']:.0f}  acc {s['acc_mean']:.3f}  drift_max {s['drift_max']:.1e}")
assert s['drift_max'] < 1e-4 and os.path.exists(f"{out}/final_spin.npy")
from ase.io import read
fx = sorted(f for f in os.listdir(out) if f.endswith(".extxyz")); fin = read(f"{out}/{fx[0]}", format="extxyz")
sp_final = np.load(f"{out}/final_spin.npy"); assert np.allclose(fin.get_initial_magnetic_moments(), sp_final[0])
log(f"  [S3] {fx[0]} carries the spins (values {np.unique(sp_final).tolist()}, counts {np.unique(sp_final[0], return_counts=True)[1].tolist()})")
out2 = f"{GATE_DIR}/{tag}_gate_pt"
cmd = [PY, "-m", "neuralce.mc.run", "--framework", "nce", "--ckpt", MODEL, "--init_cif", cif, "--pt", "--T_min", "300", "--T_max", "2000",
       "--n_replicas", "8", "--n_steps", "4000", "--chunk", "500", "--exchange_every", "100", "--log_every", "2000", "--out_dir", out2,
       "--energy_mode", "local", "--spin_move_prob", "0.5"]
r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, NCE_REPO=REPO)); assert r.returncode == 0, r.stderr[-3000:]
s = json.load(open(f"{out2}/mcmc_summary.json")); log(f"  [S3] PT: steps/s {s['steps_per_s']:.0f}  acc {s['acc_mean']:.3f}  exchange acc {s['exchange_acc']}  drift_max {s['drift_max']:.1e}")
log("  E_final by T: " + ", ".join(f"{t:.0f}K {e:.5f}" for t, e in zip(s['temps'], s['E_final'])))
log("\n[SUMMARY] SPIN MC GATES PASS")
