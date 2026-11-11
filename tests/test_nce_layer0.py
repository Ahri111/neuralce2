"""Layer 0 of the neural cluster expansion is the cluster expansion: the site mean of the sliding-window
cluster maps equals icet's cluster vector (tolerance 1e-9), and the MC path reproduces the training path."""
import numpy as np
import pytest


def test_site_mean_equals_icet_cluster_vector(ptni_model, ptni_atoms_256):
    from icet import ClusterSpace
    from neuralce.nce.general import local_cluster_features, gate_layer0
    from neuralce.nce.gpu_layer0 import build_table
    mdl = ptni_model
    cs = ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"])
    act = build_table(cs, ptni_atoms_256)["active"]
    h0 = local_cluster_features(cs, ptni_atoms_256, act)
    err = gate_layer0(cs, ptni_atoms_256, act, h0, tol=1e-9)
    assert err < 1e-9


def test_mc_path_equals_training_path(ptni_model, ptni_atoms_256):
    """Full-energy evaluation through the MC machinery (template-only tables) equals the training-time forward."""
    from neuralce.nce.mc_model import gate_final
    mae_mc, max_diff = gate_final(ptni_model, [ptni_atoms_256], y=np.array([0.0]), seed_idx=0)
    assert max_diff < 1e-6, max_diff


def test_stored_init_cif_matches_training_structure():
    """The snapped training structure shipped with the PtNi model is the one obtained from the DFT cif."""
    import os
    from ase.io import read
    from neuralce.paths import MODELS, DATA
    from neuralce.nce.features import field_from_atoms, atoms_from_field
    sid = "Ni12Pt96__Ni12Pt96_run_0010"
    ref = read(os.path.join(MODELS, "nce", "nipt_v3_nomax_sel", f"init_{sid}.cif"))
    raw = read(os.path.join(DATA, "nipt", "cifs", f"{sid}.cif"))
    snapped = atoms_from_field(field_from_atoms(raw)[0])
    assert len(ref) == len(snapped) == 108
    assert sorted(ref.get_chemical_symbols()) == sorted(snapped.get_chemical_symbols())
    # same occupation on the ideal lattice: compare sorted (species, rounded fractional position)
    def key(a):
        f = np.round(a.get_scaled_positions() % 1.0, 3) % 1.0
        return sorted(zip(a.get_chemical_symbols(), map(tuple, f)))
    assert key(ref) == key(snapped)
