"""Graph builder of the CGCNN / ALIGNN baselines (self-contained copy of the training-time ``build_graph_lite``).

One-hot node features from ``species_map`` (atomic number -> column, read from ``nipt_gnn_config.yaml`` next to
this file), a padded neighbour table of the ``max_num_nbr`` nearest neighbours within ``cutoff`` and shell one-hot
edge features (``shell_edges`` or ``n_shells`` equal bins).  Numerics are identical to the original builder; only the
``bessel`` edge basis (never used by the benchmark) is not available here.

Usage::

    from gnn_baselines.graph_lite import build_graph_lite
    atom_fea, nbr_fea, nbr_fea_idx, spin_fea = build_graph_lite(
        dict(crystal=pymatgen_structure, spin_states=np.zeros(n)), cutoff=4.5, n_shells=2, max_num_nbr=42,
        shell_edges=[0.0, 3.3, 4.5], edge_basis="shell_onehot")
"""
import os

import numpy as np
import yaml
from pymatgen.core import DummySpecies

CONFIG_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nipt_gnn_config.yaml")


def load_species_map(path=CONFIG_YAML):
    """``species_map`` of the yaml (atomic number -> one-hot column)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {int(k): v for k, v in cfg["species_map"].items()}


SPECIES_MAP = load_species_map()
N_SPECIES = len(SPECIES_MAP)


def get_specie_number(specie):
    """Atomic number of a pymatgen species; DummySpecies (vacancy) -> 0."""
    if isinstance(specie, DummySpecies):
        return 0
    return specie.number


def build_graph_lite(struct, cutoff, n_shells, max_num_nbr=12, include_sisj=False,
                     shell_edges=None, edge_basis='shell_onehot', n_bessel=8):
    """Build graph from pymatgen Structure (per-crystal)."""
    crystal = struct['crystal']
    n_at = len(crystal)
    spins = struct['spin_states']

    atom_fea = np.zeros((n_at, N_SPECIES), dtype=np.float32)
    for i, site in enumerate(crystal):
        z = get_specie_number(site.specie)
        if z in SPECIES_MAP:
            atom_fea[i, SPECIES_MAP[z]] = 1.0

    all_nbrs = crystal.get_all_neighbors(cutoff, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx = np.zeros((n_at, max_num_nbr), dtype=np.int32)
    nbr_dists   = np.full((n_at, max_num_nbr), cutoff + 1.0, dtype=np.float64)

    for i, nbr in enumerate(all_nbrs):
        n_nbr = min(len(nbr), max_num_nbr)
        for j in range(n_nbr):
            nbr_fea_idx[i, j] = nbr[j][2]
            nbr_dists[i, j]   = nbr[j][1]

    if edge_basis == 'bessel':
        raise NotImplementedError("the Bessel edge basis is not part of this frozen copy (the benchmark uses shell_onehot)")
    else:
        valid_mask = nbr_dists < cutoff
        if shell_edges is not None:
            _edges = np.array(shell_edges)
        else:
            min_dist = nbr_dists[valid_mask].min() if valid_mask.any() else 0.1
            _edges = np.linspace(min_dist * 0.99, cutoff, n_shells + 1)
        shell_idx = np.clip(np.digitize(nbr_dists, _edges) - 1, 0, n_shells - 1)
        shell_oh = np.zeros((n_at, max_num_nbr, n_shells), dtype=np.float32)
        for i in range(n_at):
            for j in range(max_num_nbr):
                if nbr_dists[i, j] < cutoff:
                    shell_oh[i, j, shell_idx[i, j]] = 1.0
        edge_radial = shell_oh

    if include_sisj:
        sisj = np.zeros((n_at, max_num_nbr, 1), dtype=np.float32)
        for i in range(n_at):
            for j in range(max_num_nbr):
                if nbr_dists[i, j] < cutoff:
                    sisj[i, j, 0] = spins[i] * spins[nbr_fea_idx[i, j]]
        nbr_fea = np.concatenate([edge_radial, sisj], axis=-1).astype(np.float32)
    else:
        nbr_fea = edge_radial.astype(np.float32)

    spin_fea = spins[:n_at].reshape(-1, 1).astype(np.float32)
    return atom_fea, nbr_fea, nbr_fea_idx, spin_fea
