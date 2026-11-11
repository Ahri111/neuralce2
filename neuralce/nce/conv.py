"""Cluster convolution: the CE evaluated as a sparse 3-D conv with jnp.roll.

Layer 1 (fixed, multiplicative) reproduces the icet cluster vector exactly:

    h[cell, a] = (1/mult_a) * sum_{c in clusters_a} prod_k theta(sigma[cell + d_k, s_k])
    phi_a      = mean_cell h[cell, a]
    E          = n_atoms * ( zerolet + sum_a eci_a * phi_a )

`jnp.roll` wraps, so periodic boundaries come for free (circular padding).
Deeper layers reuse the same offset bank with learned additive weights.
"""
from __future__ import annotations
import jax, jax.numpy as jnp, numpy as np


def theta_binary(occ):
    """icet binary point function on occupation index {0,1}.

    Verified against `ClusterSpace.get_cluster_vector`: icet maps the FIRST
    species (alphabetically, e.g. Ni) to -1 and the second (Pt) to +1.
    """
    return 2.0 * occ - 1.0


def cluster_conv(field, clusters):
    """field: (m0,m1,m2,n_prim).  Returns (m0,m1,m2) summed over equivalent clusters."""
    acc = jnp.zeros(field.shape[:3], field.dtype)
    for cl in clusters:
        h = jnp.ones(field.shape[:3], field.dtype)
        for sub, d in cl:
            h = h * jnp.roll(field[..., sub], shift=(-d[0], -d[1], -d[2]), axis=(0, 1, 2))
        acc = acc + h
    return acc


def make_cluster_vector_fn(bank):
    ks = bank['kernels']

    def cv(field):
        return jnp.stack([cluster_conv(field, k['clusters']).mean() / k['mult']
                          for k in ks])
    return cv


def make_energy_fn(bank):
    cv_fn = make_cluster_vector_fn(bank)
    eci = jnp.asarray([k['eci'] for k in ks_of(bank)])
    z = bank['zerolet']

    def energy(field):
        """Matches ClusterExpansion.predict: cluster vector is already intensive."""
        return z + jnp.dot(eci, cv_fn(field))
    return energy


def ks_of(bank):
    return bank['kernels']
