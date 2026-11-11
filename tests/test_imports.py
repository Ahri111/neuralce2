"""Every public module imports, and the files the README points at exist."""
import importlib
import os
import pytest

from neuralce.paths import REPO, DATA, MODELS, RESULTS

MODULES = [
    "neuralce", "neuralce.paths",
    "neuralce.ce", "neuralce.ce.classical_ce_gpu", "neuralce.ce.ce_gpu_tiled", "neuralce.ce.ce_gpu_implicit",
    "neuralce.ce.pt_mcmc_ce", "neuralce.ce.primitive_no_idealize", "neuralce.ce.train_ce",
    "neuralce.ce.icet_mc", "neuralce.ce.gpu_mc", "neuralce.ce.acceptance_ratio", "neuralce.ce.spin_pairs",
    "neuralce.nce", "neuralce.nce.conv", "neuralce.nce.kernels", "neuralce.nce.lattice", "neuralce.nce.features",
    "neuralce.nce.deep", "neuralce.nce.general", "neuralce.nce.gpu_layer0", "neuralce.nce.gpu_layer0_tiled",
    "neuralce.nce.gpu_layer0_implicit", "neuralce.nce.local_update", "neuralce.nce.local_update_implicit",
    "neuralce.nce.spin_features", "neuralce.nce.mc_model", "neuralce.nce.mcmc",
    "neuralce.mc", "neuralce.mc.run", "neuralce.mc.init_structures",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_cli_help(capsys):
    from neuralce.mc.run import parse_args
    with pytest.raises(SystemExit) as e:
        parse_args(["--help"])
    assert e.value.code == 0
    assert "--framework" in capsys.readouterr().out


FILES = [
    "data/nipt/icet_csv_FINAL_dedup.csv", "data/nipt/cifs", "data/nipt/REF_ideal_lattice.POSCAR",
    "data/benchmark_folds.json",
    "models/nce/nipt_v3_nomax_sel/model.pkl", "models/ce/ptni/icet_c8_6_5.ce", "models/ce/ptni/fits.json",
    "models/structures/ptni/ref_L10_NiPt.cif", "results/bench_stage/mchammer.json",
]


@pytest.mark.parametrize("rel", FILES)
def test_shipped_file(rel):
    assert os.path.exists(os.path.join(REPO, rel)), rel


def test_ptni_cifs_complete():
    import pandas as pd
    ids = pd.read_csv(os.path.join(DATA, "nipt", "icet_csv_FINAL_dedup.csv")).id.astype(str)
    missing = [i for i in ids if not os.path.exists(os.path.join(DATA, "nipt", "cifs", f"{i}.cif"))]
    assert len(ids) == 121 and not missing, missing[:5]
