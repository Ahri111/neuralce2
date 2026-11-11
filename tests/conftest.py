"""Shared fixtures.  All tests run on the CPU (JAX_PLATFORMS=cpu is set here before JAX is imported)
unless the environment variable NCE_TEST_GPU=1 is set; then the platform is left to JAX."""
import os
import pytest

if os.environ.get("NCE_TEST_GPU", "0") != "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

from neuralce.paths import MODELS, DATA  # noqa: E402

PTNI_MODEL = os.path.join(MODELS, "nce", "nipt_v3_nomax_sel", "model.pkl")
PTNI_CE = os.path.join(MODELS, "ce", "ptni", "icet_c8_6_5.ce")
PTNI_REF = os.path.join(MODELS, "structures", "ptni", "ref_L10_NiPt.cif")


@pytest.fixture(scope="session")
def rng():
    import numpy as np
    return np.random.RandomState(0)


@pytest.fixture(scope="session")
def ptni_atoms_256():
    """random Ni0.5Pt0.5 on a 4x4x4 conventional fcc cell (256 atoms)."""
    import numpy as np
    from ase.build import bulk
    a = bulk("Ni", "fcc", a=3.924, cubic=True).repeat((4, 4, 4))
    r = np.random.RandomState(1)
    Z = np.full(len(a), 78, np.int64)
    Z[r.choice(len(a), len(a) // 2, replace=False)] = 28
    a.numbers = Z
    return a


@pytest.fixture(scope="session")
def ptni_model():
    from neuralce.nce.mc_model import load_model
    assert os.path.exists(PTNI_MODEL), PTNI_MODEL
    return load_model(PTNI_MODEL)
