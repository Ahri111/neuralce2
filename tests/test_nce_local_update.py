"""Template-only (implicit) kernels == full O(N) tables, and the receptive-field local energy update ==
difference of two full energies, for the neural cluster expansion (PtNi model, 256 atoms, float64)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def pair(ptni_model, ptni_atoms_256):
    from neuralce.nce.mc_model import setup
    mt = setup(ptni_model, ptni_atoms_256, dtype=None, table="tiled")
    mi = setup(ptni_model, ptni_atoms_256, dtype=None, table="implicit")
    return mt, mi


def test_energy_implicit_equals_tiled(pair):
    mt, mi = pair
    Et = mt["energy"](mt["sig0"])
    Ei = mi["energy"](mi["sig0"])
    assert abs(Et - Ei) < 1e-9, (Et, Ei)
    assert np.array_equal(np.asarray(mi["sig0"]), np.asarray(mt["sig0"])[mi["perm"]])


def test_local_delta_equals_full_recompute(pair, rng):
    mt, mi = pair
    act = np.asarray(mt["active"]); inv = jnp.asarray(mi["inv_perm"])
    st, si = mt["sig0"], mi["sig0"]
    ct, ci = mt["full_cache"](st), mi["full_cache"](si)
    e_i = float(mi["F"]["energy_of"](ci, mi["P"]))
    worst_pair = worst_full = 0.0
    for _ in range(30):
        i, j = (int(x) for x in rng.choice(act, 2, replace=False))
        if int(st[i]) == int(st[j]):
            continue
        ft = jnp.array([[i, int(st[j])], [j, int(st[i])]])
        fi = ft.at[:, 0].set(inv[ft[:, 0]])
        dEt, pt_, _ = mt["local_delta"](ct, st, ft)
        dEi, pi_, _ = mi["local_delta"](ci, si, fi)
        worst_pair = max(worst_pair, abs(float(dEt) - float(dEi)))
        si1 = si.at[fi[:, 0]].set(fi[:, 1].astype(si.dtype))
        E1 = float(mi["F"]["energy_of"](mi["full_cache"](si1), mi["P"]))
        worst_full = max(worst_full, abs(float(dEi) - (E1 - e_i)))
        ct = mt["commit"](ct, pt_, jnp.asarray(True)); ci = mi["commit"](ci, pi_, jnp.asarray(True))
        st = st.at[ft[:, 0]].set(ft[:, 1].astype(st.dtype)); si = si1; e_i = E1
    assert worst_pair < 1e-9, worst_pair
    assert worst_full < 1e-9, worst_full
    # the cache after a chain of committed moves equals a full rebuild
    E_chain = float(mi["F"]["energy_of"](ci, mi["P"]))
    E_full = float(mi["F"]["energy_of"](mi["full_cache"](si), mi["P"]))
    assert abs(E_chain - E_full) < 1e-9
