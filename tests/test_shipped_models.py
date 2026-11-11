"""Every shipped model loads and builds its Monte Carlo machinery (template-only tables) on a small cell of its
own lattice; the Fe-Ni-Cr spin model also runs through the command line on its shipped start cell."""
import glob
import json
import os
import subprocess
import sys
import numpy as np
import pytest

from neuralce.paths import MODELS

TAGS = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(MODELS, "nce", "*", "model.pkl")))


@pytest.mark.parametrize("tag", TAGS)
def test_model_builds_mc_machinery(tag):
    from icet import ClusterSpace
    from neuralce.nce.mc_model import load_model, setup
    mdl = load_model(os.path.join(MODELS, "nce", tag, "model.pkl"))
    for k in ("primitive", "cutoffs", "chemical_symbols", "keep", "params", "mu", "sd", "ym", "ys", "shell_edges"):
        assert k in mdl, k
    ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"])
    atoms = mdl["primitive"].repeat((2, 2, 2))
    mc = setup(mdl, atoms)
    spins = None
    if int(mdl.get("spin", 0) or 0):
        vals = np.asarray(mdl["spin_values"], float)
        spins = np.random.RandomState(0).choice(vals, size=len(atoms))[mc["perm"]]
    e = mc["energy"](mc["sig0"], spins)
    assert np.isfinite(e)


@pytest.mark.parametrize("ce", sorted(glob.glob(os.path.join(MODELS, "ce", "*", "*.ce"))))
def test_ce_file_reads(ce):
    from icet import ClusterExpansion
    c = ClusterExpansion.read(ce)
    assert len(c.parameters) > 0


def test_fenicr_spin_model_cli(tmp_path):
    ckpt = os.path.join(MODELS, "nce", "fenicr_s2b_mm_c65_d1", "model.pkl")
    cif = os.path.join(MODELS, "structures", "fenicr", "init_Fe56Cr21Ni23_500.cif")
    allowed = os.path.join(MODELS, "structures", "fenicr", "spin_allowed_dft.json")
    if not (os.path.exists(ckpt) and os.path.exists(cif)):
        pytest.skip("Fe-Ni-Cr model or start cell not shipped")
    cmd = [sys.executable, "-m", "neuralce.mc.run", "--framework", "nce", "--ckpt", ckpt, "--init_cif", cif,
           "--T", "1300", "--n_steps", "1000", "--chunk", "500", "--n_replicas", "2", "--energy_mode", "local",
           "--spin_move_prob", "0.5", "--spin_init", "random", "--spin_allowed_file", allowed,
           "--out_dir", str(tmp_path / "out")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert json.load(open(tmp_path / "out" / "mcmc_summary.json"))
