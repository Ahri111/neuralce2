"""Orthonormal cluster expansion on GPU tables: the template-only (implicit) kernels equal the tiled
O(N) tables for the energy and for swap energy differences, energies are extensive under tiling, and the
energy difference of a swap equals the difference of two full energies (PtNi .ce, 256 atoms)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def ce():
    from icet import ClusterExpansion
    from tests.conftest import PTNI_CE
    return ClusterExpansion.read(PTNI_CE)


@pytest.fixture(scope="module")
def gpu_pair(ce, ptni_atoms_256):
    from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
    gt = GPUClusterExpansion(ce, ptni_atoms_256, table_mode="tiled")
    gi = GPUClusterExpansion(ce, ptni_atoms_256, table_mode="implicit")
    return gt, gi


def test_energy_implicit_equals_tiled(gpu_pair, ptni_atoms_256):
    gt, gi = gpu_pair
    st = jnp.asarray(gt.encode(ptni_atoms_256)); si = jnp.asarray(gi.encode(ptni_atoms_256))
    assert np.array_equal(np.asarray(si), np.asarray(st)[gi.perm])
    et = float(gt.energy_a(gt.tp, st)); ei = float(gi.energy_a(gi.tp, si)); e2 = float(gi.energy(si))
    assert abs(et - ei) < 2e-3, (et, ei)          # float32 tables, total energy of ~1e3 eV
    assert abs(ei - e2) < 1e-6


def test_swap_delta_implicit_equals_tiled_and_full(gpu_pair, ptni_atoms_256, rng):
    from neuralce.ce.pt_mcmc_ce import build_mcmc_tables
    gt, gi = gpu_pair
    st = jnp.asarray(gt.encode(ptni_atoms_256)); si = jnp.asarray(gi.encode(ptni_atoms_256))
    ei = float(gi.energy_a(gi.tp, si))
    mt = build_mcmc_tables(gt)
    errs_t, errs_f = [], []
    for _ in range(30):
        sl = rng.randint(mt.n_active_subs); n = int(mt.sub_mask[sl].sum())
        i_a, i_b = np.asarray(mt.sub_indices[sl])[rng.choice(n, 2, replace=False)]
        sa, sb = int(st[i_a]), int(st[i_b])
        dt = float(gt.delta_energy_a(gt.tp, i_a, i_b, st, sb, sa))
        ja, jb = int(gi.inv_perm[i_a]), int(gi.inv_perm[i_b])
        di = float(gi.delta_energy_a(gi.tp, ja, jb, si, sb, sa))
        si1 = si.at[ja].set(sb).at[jb].set(sa)
        errs_t.append(abs(dt - di)); errs_f.append(abs(di - (float(gi.energy_a(gi.tp, si1)) - ei)))
    assert max(errs_t) < 1e-4, max(errs_t)
    assert max(errs_f) < 2e-3, max(errs_f)


def test_energy_extensive_under_tiling(ce, ptni_atoms_256):
    """The periodic 2x2x2 image has 8x the energy (energy_scale) and one swap keeps its energy difference."""
    from neuralce.ce.classical_ce_gpu import GPUClusterExpansion
    a4 = ptni_atoms_256; a8 = a4.repeat((2, 2, 2))
    g4 = GPUClusterExpansion(ce, a4, table_mode="implicit")
    g8 = GPUClusterExpansion(ce, a8, table_mode="implicit", energy_scale=8.0)
    s4 = jnp.asarray(g4.encode(a4)); s8 = jnp.asarray(g8.encode(a8))
    e4 = float(g4.energy_a(g4.tp, s4)); e8 = float(g8.energy_a(g8.tp, s8))
    i_a, i_b = 3, 7
    sa, sb = int(s4[i_a]), int(s4[i_b])
    while sa == sb:
        i_b += 1; sb = int(s4[i_b])
    d4 = float(g4.delta_energy_a(g4.tp, i_a, i_b, s4, sb, sa))
    ja, jb = int(g8.inv_perm[int(g4.perm[i_a])]), int(g8.inv_perm[int(g4.perm[i_b])])
    d8 = float(g8.delta_energy_a(g8.tp, ja, jb, s8, sb, sa))
    assert abs(e8 / e4 - 8.0) < 1e-4
    assert abs(d8 / d4 - 1.0) < 1e-3
