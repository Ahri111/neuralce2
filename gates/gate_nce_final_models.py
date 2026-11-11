"""Gates for the FINAL neural-CE models ($MODELS/nce/<tag>/model.pkl) before they go into production MC.

F1  MC path == training path on training structures (float64): max|E_mc - E_train| < 1e-8 eV/atom on 20 training
    structures spread over the training set; the MC-path MAE is printed.
F2  short MC through the central entry (``python -m neuralce.mc.run --framework nce --energy_mode local``):
    single-T run (800 K, 4 replicas, 4000 steps); drift_max < 1e-4 (float32 local update re-anchored per chunk).
F3  PT smoke (300-2000 K, 8 replicas, exchanges every 100 steps); must run to completion.

Models (GATE_MODELS, comma list of tags; default = the shipped PtNi model nipt_v3_nomax_sel).
Training structures come from the data the model was fitted on, matched BY ID to model['train_ids']:
    PtNi   $DATA/nipt/cifs/<id>.cif snapped onto the ideal fcc lattice (NIPT_CIF_DIR overrides the directory)
Only the PtNi data set is shipped, so only models with system 'nipt' can be gated here.
Outputs: $RUNS/gates/<tag>_gate_mc/ and <tag>_gate_pt/ (F2/F3 MC output).
Run:     python gates/gate_nce_final_models.py        (CPU or GPU)
Env:     GATE_MODELS, SECTION=values|mc (comma list, default both).
"""
import os, sys, json, subprocess, numpy as np
from neuralce.paths import REPO, MODELS, RUNS
import jax; jax.config.update("jax_enable_x64", True)
from neuralce.nce.mc_model import load_model, gate_final, n_full_channels
PY = sys.executable
def log(*a): print(*a, flush=True)

TAGS = os.environ.get("GATE_MODELS", "nipt_v3_nomax_sel").split(",")
SEC = os.environ.get("SECTION", "values,mc").split(",")
GATE_DIR = f"{RUNS}/gates"; os.makedirs(GATE_DIR, exist_ok=True)


def training_structures(mdl, system):
    """(atoms_by_train_index, spins_by_train_index or None, init cif for the MC smoke or None).
    Only the PtNi training data is shipped: the relaxed cifs are snapped onto the ideal fcc lattice as at training time."""
    ids = [str(i) for i in mdl['train_ids']]
    if system == "nipt":
        from ase.io import read
        from neuralce.nce.features import field_from_atoms, atoms_from_field, CIF_DIR
        return [atoms_from_field(field_from_atoms(read(f"{CIF_DIR}/{i}.cif"))[0]) for i in ids], None, None
    raise SystemExit(f"training structures for system '{system}' are not shipped (only the PtNi data set is)")


for name in TAGS:
    model = f"{MODELS}/nce/{name}/model.pkl"
    if not os.path.exists(model):
        log(f"[{name}] no final model: {model}"); continue
    mdl = load_model(model); system = mdl.get('system') or name.split("_")[0]
    spin = int(mdl.get('spin', 0) or 0)
    log(f"\n==== {name}: system {system} spin {spin} depth {mdl['depth']} wd {mdl['wd']} steps {mdl['steps']} channels {len(mdl['keep'])}/{n_full_channels(mdl)} ====")
    if 'train_y' not in mdl:
        log(f"  [F1] skipped: the shipped {name} model carries no training labels (the DFT data of this system are not public)")
        continue
    atoms_src, spins_src, cif = training_structures(mdl, system)
    y = np.asarray(mdl['train_y']); n_all = len(mdl['train_ids'])
    avail = np.array(sorted(atoms_src.keys())) if isinstance(atoms_src, dict) else np.arange(n_all)
    n = min(20, len(avail)); sel = avail[np.linspace(0, len(avail) - 1, n).astype(int)]
    atoms_sel = [atoms_src(k) if callable(atoms_src) else atoms_src[k] for k in sel]
    spins_sel = None if spins_src is None else [spins_src(k) for k in sel]
    if "values" in SEC:
        mae_mc, dmax = gate_final(mdl, atoms_sel, y[sel], spins_list=spins_sel)
        log(f"  [F1] MC path vs training path on {n} training structures (sizes {sorted(set(len(a) for a in atoms_sel))}): max|dE| = {dmax:.2e}   MC-path MAE {mae_mc:.3f} meV/atom")
        assert dmax < 1e-8, dmax
    if "mc" not in SEC:
        continue
    # F2: central entry, single T, local
    from ase.io import write
    out = f"{GATE_DIR}/{name}_gate_mc"; os.makedirs(out, exist_ok=True)
    if cif is None:                                                        # PtNi: a training structure (largest picked)
        k = int(np.argmax([len(a) for a in atoms_sel])); cif = f"{out}/init_{mdl['train_ids'][sel[k]]}.cif"; write(cif, atoms_sel[k])
    spin_args = ["--spin_move_prob", "0.5", "--spin_init", "random"] if spin else []
    cmd = [PY, "-m", "neuralce.mc.run", "--framework", "nce", "--ckpt", model, "--init_cif", cif,
           "--T", "800", "--n_steps", "4000", "--chunk", "500", "--n_replicas", "4", "--log_every", "1000", "--out_dir", out,
           "--energy_mode", "local"] + spin_args
    log("  [F2] " + " ".join(os.path.basename(x) if x.startswith("/") else x for x in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, NCE_REPO=REPO))
    tail = "\n".join(l for l in r.stdout.splitlines() if "[ce_cnn" in l)[-1500:]
    log(tail); assert r.returncode == 0, r.stderr[-2000:]
    s = json.load(open(f"{out}/mcmc_summary.json")); log(f"  [F2] summary: steps/s {s['steps_per_s']:.0f}  acc {s['acc_mean']:.3f}  drift_max {s['drift_max']:.1e}")
    assert s['drift_max'] < 1e-4
    # F3: PT smoke (8 replicas, exchanges)
    out2 = f"{GATE_DIR}/{name}_gate_pt"
    cmd = [PY, "-m", "neuralce.mc.run", "--framework", "nce", "--ckpt", model, "--init_cif", cif, "--pt",
           "--T_min", "300", "--T_max", "2000", "--n_replicas", "8", "--n_steps", "4000", "--chunk", "500", "--exchange_every", "100",
           "--log_every", "2000", "--out_dir", out2, "--energy_mode", "local"] + spin_args
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, NCE_REPO=REPO)); assert r.returncode == 0, r.stderr[-2000:]
    s = json.load(open(f"{out2}/mcmc_summary.json")); log(f"  [F3] PT: steps/s {s['steps_per_s']:.0f}  acc {s['acc_mean']:.3f}  exchange acc {s['exchange_acc']}  drift_max {s['drift_max']:.1e}")
    log(f"  E_final by T: " + ", ".join(f"{t:.0f}K {e:.5f}" for t, e in zip(s['temps'], s['E_final'])))
log("\n[SUMMARY] FINAL MODEL GATES PASS")
