"""The command line runs end to end for both energy models (tiny parallel-tempering runs, CPU)."""
import json
import os
import subprocess
import sys
import pytest

from tests.conftest import PTNI_MODEL, PTNI_CE, PTNI_REF


def _run(args, tmp_path):
    env = dict(os.environ, NCE_RUNS=str(tmp_path))
    cmd = [sys.executable, "-m", "neuralce.mc.run"] + args + ["--out_dir", str(tmp_path / "out")]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1500)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    summary = json.load(open(tmp_path / "out" / "mcmc_summary.json"))
    return r.stdout, summary


def test_nce_pt(tmp_path):
    out, summary = _run(["--framework", "nce", "--ckpt", PTNI_MODEL, "--init_cif", PTNI_REF,
                         "--pt", "--T_min", "300", "--T_max", "1400", "--n_replicas", "4",
                         "--n_steps", "2000", "--chunk", "500", "--exchange_every", "500", "--sample_interval", "100",
                         "--energy_mode", "local", "--seed", "0"], tmp_path)
    assert os.path.exists(tmp_path / "out" / "energies.npz")
    assert all(k in summary for k in summary)  # summary is a plain json


def test_icet_pt(tmp_path):
    out, summary = _run(["--framework", "icet", "--ce", PTNI_CE, "--init_cif", PTNI_REF,
                         "--pt", "--T_min", "300", "--T_max", "1400", "--n_replicas", "4",
                         "--n_steps", "2000", "--chunk", "500", "--exchange_every", "500", "--sample_interval", "100",
                         "--energy_mode", "delta", "--seed", "0"], tmp_path)
    assert os.path.exists(tmp_path / "out" / "mcmc_result.npz")


def test_alias_ce_cnn_is_accepted(tmp_path):
    _run(["--framework", "ce_cnn", "--ckpt", PTNI_MODEL, "--init_cif", PTNI_REF,
          "--T", "800", "--n_steps", "500", "--chunk", "250", "--n_replicas", "2", "--energy_mode", "local"], tmp_path)
