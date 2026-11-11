# Frozen copy of the graph-network energy models (CGCNN / ALIGNN ports in JAX/flax) used only by the
# MC-cost benchmark scripts/bench/bench_gnn_mc_step.py.  Not part of the neuralce package; needs the `gnn` extra (flax).
"""
NeuralCE JAX — Neural Cluster Expansion for Alloy Energy Prediction

Cluster Expansion:
    E(σ) = J₀ + Σᵢ Jᵢσᵢ + Σᵢⱼ Jᵢⱼσᵢσⱼ + Σᵢⱼₖ Jᵢⱼₖσᵢσⱼσₖ + ...

Model Hierarchy:
    ──────────────────────────────────────────────────────────────────────
    Heavy (atom_init 92-dim + Gaussian 41-dim edges, pt_mcmc compat):
        ising          — Chemical only, no spin
        spin_v3        — SpinEmbedding + MagExchange, E(σ) ≠ E(-σ)
        spin_v8        — Even-Odd dual readout, E(σ) = E(-σ) ✓
        spin_v8u       — Even-Odd unified readout, E(σ) = E(-σ) ✓

    Lite (one-hot 4-dim node + shell edge, Sr excluded, Optuna tuning):
        ising_lite              — No spin (reference)
        neuralce_evenodd_lite   — Product backbone + EvenOdd spin
        neuralce_sisj_lite      — Product backbone + σᵢσⱼ in edge
        gnn_sisj_lite           — Concat-MLP backbone + σᵢσⱼ in edge
        gnn_evenodd_lite        — Concat-MLP backbone + EvenOdd spin

    Ablation table (Lite):
    ┌─────────────┬──────────────────────┬──────────────────────┐
    │             │ EvenOdd              │ SiSj (σᵢσⱼ edge)    │
    ├─────────────┼──────────────────────┼──────────────────────┤
    │ Product(CE) │ neuralce_evenodd_lite│ neuralce_sisj_lite   │
    │ Concat-MLP  │ gnn_evenodd_lite     │ gnn_sisj_lite        │
    └─────────────┴──────────────────────┴──────────────────────┘
    + ising_lite (no spin baseline)
    ──────────────────────────────────────────────────────────────────────
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional


# =============================================================================
# 1. Pooling Modules
# =============================================================================

class FixedSumPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        B = kwargs['batch_size']
        N = kwargs['n_atoms_per_crystal']
        return jnp.sum(site_energy.reshape(B, N), axis=1, keepdims=True)

class FixedMeanPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        B = kwargs['batch_size']
        N = kwargs['n_atoms_per_crystal']
        return jnp.mean(site_energy.reshape(B, N), axis=1, keepdims=True)

class VariableSumPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        seg = kwargs['segment_ids']
        nc = kwargs['num_crystals']
        return jax.ops.segment_sum(site_energy.squeeze(-1), seg, num_segments=nc)[:, None]

class VariableMeanPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        seg = kwargs['segment_ids']
        nc = kwargs['num_crystals']
        flat = site_energy.squeeze(-1)
        total = jax.ops.segment_sum(flat, seg, num_segments=nc)
        counts = jax.ops.segment_sum(jnp.ones_like(flat), seg, num_segments=nc)
        return (total / jnp.maximum(counts, 1.0))[:, None]


class PaddedSumPool(nn.Module):
    """Sum pool with mask for padded atoms. mask shape: (B*N,) or (B, N)."""
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        B = kwargs['batch_size']
        N = kwargs['n_atoms_per_crystal']
        mask = kwargs['atom_mask']  # (B, N) or (B*N,)
        mask = mask.reshape(B, N)
        energy = site_energy.reshape(B, N)
        return jnp.sum(energy * mask, axis=1, keepdims=True)

class PaddedMeanPool(nn.Module):
    """Mean pool with mask for padded atoms."""
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        B = kwargs['batch_size']
        N = kwargs['n_atoms_per_crystal']
        mask = kwargs['atom_mask']
        mask = mask.reshape(B, N)
        energy = site_energy.reshape(B, N)
        total = jnp.sum(energy * mask, axis=1, keepdims=True)
        counts = jnp.sum(mask, axis=1, keepdims=True)
        return total / jnp.maximum(counts, 1.0)


# =============================================================================
# 2. Core Convolution Layers
# =============================================================================

class CEInteractionLayer(nn.Module):
    """Product-based graph conv (NeuralCE backbone) with Pre-LayerNorm."""
    atom_fea_len: int

    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_normed = nn.LayerNorm()(atom_in_fea)
        atom_nbr_fea = jnp.take(atom_normed, nbr_fea_idx, axis=0)
        atom_center_fea = jnp.tile(atom_normed[:, None, :], (1, M, 1))

        phi_center = nn.Dense(self.atom_fea_len)(atom_center_fea)
        phi_nbr    = nn.Dense(self.atom_fea_len)(atom_nbr_fea)
        phi_edge   = nn.Dense(self.atom_fea_len)(nbr_fea)
        interaction = phi_center * phi_nbr * phi_edge

        gate      = nn.sigmoid(nn.Dense(self.atom_fea_len)(interaction))
        magnitude = nn.softplus(nn.Dense(self.atom_fea_len)(interaction))
        nbr_sumed = jnp.sum(gate * magnitude, axis=1)
        return atom_in_fea + nbr_sumed


class ConcatConvLayer(nn.Module):
    """Concat-MLP graph conv (GNN backbone)."""
    node_dim: int
    edge_dim: int

    @nn.compact
    def __call__(self, x, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        x_normed = nn.LayerNorm()(x)
        x_nbr = jnp.take(x_normed, nbr_fea_idx, axis=0)      # (N, M, node_dim)
        x_center = jnp.tile(x_normed[:, None, :], (1, M, 1))  # (N, M, node_dim)

        concat = jnp.concatenate([x_center, x_nbr, nbr_fea], axis=-1)
        msg = nn.Dense(self.node_dim)(nn.silu(nn.Dense(self.node_dim)(concat)))
        agg = jnp.sum(msg, axis=1)
        return x + agg


class ReadoutMLP(nn.Module):
    """2-layer readout: atom features → site energy scalar.

    The penultimate activation (just before the final Dense(1)) is
    `sow`-stored as 'penult' in the 'intermediates' collection.  Used
    by post-hoc Bayesian Linear Regression in active_learning/blr.py.
    Calling code that ignores intermediates is unaffected.
    """
    h_fea_len: int

    @nn.compact
    def __call__(self, atom_fea):
        h = nn.softplus(nn.Dense(self.h_fea_len)(atom_fea))
        h_penult = nn.softplus(nn.Dense(self.h_fea_len // 2)(h))
        self.sow('intermediates', 'penult', h_penult)
        return nn.Dense(1)(h_penult)


class MACEInteractionLayer(nn.Module):
    """MACE v1 — original CE-flavored design (no LayerNorm, concat-mix).

    Structure:
        A_i = Σ_j (h_i ⊙ h_j ⊙ R(e_ij))   # 2-body, 3-way product (CE core)
        A2  = A_i ⊙ A_i                    # body-order 3 (no LayerNorm)
        out = Dense(concat([A_i, A_i²]))   # one mix Dense
        h_i ← h_i + out

    Known limitation: A² = A·A with unbounded A_i can produce outliers when
    paired with discrete edge basis (shell_onehot).  See MACEv2InteractionLayer
    for a norm-stabilized variant.

    conv_type='mace' selects this layer.
    """
    atom_fea_len: int

    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_normed = nn.LayerNorm()(atom_in_fea)

        h_nbr    = jnp.take(atom_normed, nbr_fea_idx, axis=0)              # (N, M, F)
        h_center = jnp.tile(atom_normed[:, None, :], (1, M, 1))            # (N, M, F)
        radial   = nn.Dense(self.atom_fea_len, name='radial')(nbr_fea)     # (N, M, F)

        # 2-body message: center × nbr × edge_radial
        msg = h_center * h_nbr * radial                                     # (N, M, F)
        A1  = jnp.sum(msg, axis=1)                                          # (N, F)

        # Body-order 3 via raw squared aggregate
        A2 = A1 * A1                                                        # (N, F)

        # Mix orders via single Dense on concat
        out = nn.Dense(self.atom_fea_len, name='mix')(
            jnp.concatenate([A1, A2], axis=-1))                             # (N, F)

        return atom_in_fea + out


class MACEv2InteractionLayer(nn.Module):
    """MACE v2 — norm-stabilized self-product + order-separated body weights.

    Structure (vs v1):
        A_i = Σ_j (h_i ⊙ h_j ⊙ R(e_ij))   # 2-body, 3-way product (same as v1)
        Â_i = LayerNorm(A_i)               # NEW: stabilize before squaring
        B_i = Â_i ⊙ Â_i                    # body-order 3, bounded
        out = Dense_2(A_i) + Dense_3(B_i)  # NEW: order-separated linear combo
        h_i ← h_i + out

    Differences from v1:
      - LayerNorm(A_i) before A·A prevents outlier explosion (v1's known
        failure mode: mace+shell on chgnet had RMSE/MAE ≈ 4).
      - Order-separated Dense_2/Dense_3 (math-equivalent to v1's concat-Dense
        but with separate inits and explicit body-order interpretation).

    conv_type='mace_v2' selects this layer.
    """
    atom_fea_len: int

    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_normed = nn.LayerNorm()(atom_in_fea)

        h_nbr    = jnp.take(atom_normed, nbr_fea_idx, axis=0)              # (N, M, F)
        h_center = jnp.tile(atom_normed[:, None, :], (1, M, 1))            # (N, M, F)
        radial   = nn.Dense(self.atom_fea_len, name='radial')(nbr_fea)     # (N, M, F)

        msg = h_center * h_nbr * radial                                     # (N, M, F)
        A1  = jnp.sum(msg, axis=1)                                          # (N, F)

        # NEW v2: norm-stabilize before squaring → prevents quadratic blow-up
        A1_normed = nn.LayerNorm(name='A_norm')(A1)                         # (N, F)
        A2 = A1_normed * A1_normed                                          # (N, F), bounded

        # NEW v2: order-separated linear combo
        out_2body = nn.Dense(self.atom_fea_len, name='mix_2body')(A1)
        out_3body = nn.Dense(self.atom_fea_len, name='mix_3body')(A2)

        return atom_in_fea + out_2body + out_3body


class MACEv3InteractionLayer(nn.Module):
    """MACE v3 — M-invariant scale + balanced body order + centered A².

    Structure (vs v2):
        A_i  = Σ_j (h_i ⊙ h_j ⊙ R(e_ij)) / √M    # 2-body, /√M (M-invariant)
        Â_i  = LayerNorm(A_i)                     # same as v2
        B_i  = Â_i ⊙ Â_i − 1                      # NEW: centered (mean ≈ 0)
        out  = Dense_2(Â_i) + Dense_3(B_i)        # NEW: Dense_2 sees Â_i, not raw A_i
        h_i  ← h_i + out                          # (small kernel init recommended)

    Differences from v2:
      - **A_i / √M**: variance-stabilizing sum normalization. Without it, A_i std
        grows as √M with neighbor count. For NiPt with 3NN edges (M=54),
        A_i std ≈ 7.4 vs perovskite (M=6) ≈ 2.4 — v2's mix_2body sees ~3x
        larger input than other systems → residual stream blow-up.
      - **mix_2body sees Â_i, not raw A_i**: aligns 2-body input scale with
        3-body's LN-stabilized scale. v2 had unnormalized A_i → 2-body
        dominated 3-body in residual updates at init, M-dependent.
      - **B_i centered** (subtract 1): A_i_normed² ~ χ²(1) per channel, mean ≈ 1.
        v2 propagated a constant +1 drift through the residual stream over
        n_conv layers; centering removes it.

    Supercell invariance: M is per-site constant under PBC supercell change
    (same neighbor count per site irrespective of supercell size), so /√M
    does not break extensivity.

    conv_type='mace_v3' selects this layer.
    """
    atom_fea_len: int

    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_normed = nn.LayerNorm()(atom_in_fea)

        h_nbr    = jnp.take(atom_normed, nbr_fea_idx, axis=0)              # (N, M, F)
        h_center = jnp.tile(atom_normed[:, None, :], (1, M, 1))            # (N, M, F)
        radial   = nn.Dense(self.atom_fea_len, name='radial')(nbr_fea)     # (N, M, F)

        msg = h_center * h_nbr * radial                                     # (N, M, F)
        # /√M: variance-stabilizing sum (M-invariant scale)
        A1  = jnp.sum(msg, axis=1) / jnp.sqrt(jnp.asarray(M, dtype=msg.dtype))

        A1_normed = nn.LayerNorm(name='A_norm')(A1)                         # (N, F), std≈1

        # B_i = Â_i² − 1: centered self-product (mean ≈ 0)
        # χ²(1) per channel has mean=1, var=2 → subtract 1 zero-centers.
        A2_centered = A1_normed * A1_normed - 1.0                           # (N, F)

        # Both mix heads see LN-stabilized features → balanced 2-body vs 3-body
        out_2body = nn.Dense(self.atom_fea_len, name='mix_2body')(A1_normed)
        out_3body = nn.Dense(self.atom_fea_len, name='mix_3body')(A2_centered)

        return atom_in_fea + out_2body + out_3body


class AttentionConvLayer(nn.Module):
    """EquiformerV2-inspired multi-head attention conv (scalar-only).

    Adapted from atomicarchitects/equiformer_v2 SO2EquivariantGraphAttention +
    TransBlockV2 (nets/equiformer_v2/transformer_block.py L40-334, L454+),
    with all equivariant machinery removed (no SO(2) conv, no SH, no S2 act).

    Two pre-norm sub-layers with residual connections:
      1) Pre-LN → multi-head GraphAttention → residual
      2) Pre-LN → 2-layer FFN → residual

    v2 GraphAttention (scalar surrogate, line refs to equiformer_v2 source):
      msg_ij  = cat([h_i, h_j])                              (v2 L255)
      e_ij    = cat([edge, src_emb(h_i), dst_emb(h_j)])      (v2 L242-246)
      m_main, α_feats = Linear1(msg_ij).split                 (SO(2) conv 1 + extras)
      m_main  ← SiLU( m_main ⊙ Linear_edge(e_ij) )           (S2 activation surrogate)
      v_ij    = Linear2(m_main)                              (SO(2) conv 2)
      α_ij    = einsum( LeakyReLU(LN(α_feats)), α_dot )      (v2 L309-312, GATv2)
      α_ij    ← softmax_j(α_ij)                              (v2 L313)
      out_i   = proj( Σ_j α_ij ⊙ v_ij )                      (v2 L319-332)

    Key v2-vs-v1 deltas preserved here:
      • concat src+dst (not sum-merge)
      • atom-type emb concatenated to edge feature
      • 2-stage non-linear message (Linear → act → Linear)
      • LayerNorm on alpha features before activation (use_attn_renorm)
      • alpha_dot init = uniform[-1/√d, 1/√d]  (v2 L187-188)
    """
    atom_fea_len: int
    num_heads: int       = 4
    head_dim: int        = 16        # attn_value_channels per head
    alpha_head_dim: int  = 8         # attn_alpha_channels per head
    hidden_mult: int     = 2         # attn_hidden_channels = hidden_mult * atom_fea_len
    ffn_mult: int        = 2

    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        H, D, A  = self.num_heads, self.head_dim, self.alpha_head_dim
        HD       = H * D
        H_ALPHA  = H * A
        HIDDEN   = self.atom_fea_len * self.hidden_mult

        # ===== Sub-layer 1: Pre-LN + GraphAttention + residual =====
        x = nn.LayerNorm(name='ln1')(atom_in_fea)

        # v2 concat-merge: cat([h_src, h_dst])
        h_src = jnp.take(x, nbr_fea_idx, axis=0)              # (N, M, F)
        h_dst = jnp.tile(x[:, None, :], (1, M, 1))            # (N, M, F)
        msg_in = jnp.concatenate([h_src, h_dst], axis=-1)     # (N, M, 2F)

        # v2 edge w/ atom-type embeddings concatenated
        src_emb = nn.Dense(self.atom_fea_len, use_bias=False, name='src_emb')(h_src)
        dst_emb = nn.Dense(self.atom_fea_len, use_bias=False, name='dst_emb')(h_dst)
        edge_ext = jnp.concatenate([nbr_fea, src_emb, dst_emb], axis=-1)
        edge_w   = nn.Dense(HIDDEN, name='edge_proj')(edge_ext)             # (N, M, HIDDEN)

        # SO(2) conv 1 surrogate: shared linear produces (hidden message | α extras)
        combined = nn.Dense(HIDDEN + H_ALPHA, name='msg_1')(msg_in)
        m_main, alpha_feats = jnp.split(combined, [HIDDEN], axis=-1)

        # S2 activation surrogate: edge-gated SiLU
        m_main = nn.silu(m_main * edge_w)

        # SO(2) conv 2 surrogate: produce per-head value
        value = nn.Dense(HD, name='msg_2')(m_main).reshape(N, M, H, D)

        # v2 attention: LN → LeakyReLU → einsum w/ α_dot
        alpha_feats = alpha_feats.reshape(N, M, H, A)
        alpha_feats = nn.LayerNorm(name='alpha_norm')(alpha_feats)
        alpha_feats = nn.leaky_relu(alpha_feats, negative_slope=0.2)
        a_std = 1.0 / (A ** 0.5)                                            # v2 init scale
        alpha_dot = self.param(
            'alpha_dot',
            lambda k, s: jax.random.uniform(k, s, minval=-a_std, maxval=a_std),
            (H, A))
        alpha = jnp.einsum('nmhk,hk->nmh', alpha_feats, alpha_dot)          # (N, M, H)
        alpha = nn.softmax(alpha, axis=1)                                   # over M

        attn = jnp.sum(value * alpha[..., None], axis=1).reshape(N, HD)
        ga_out = nn.Dense(self.atom_fea_len, name='proj_out')(attn)
        x = atom_in_fea + ga_out                                            # residual

        # ===== Sub-layer 2: Pre-LN + FFN + residual =====
        h = nn.LayerNorm(name='ln2')(x)
        h = nn.silu(nn.Dense(self.atom_fea_len * self.ffn_mult, name='ffn_1')(h))
        ffn_out = nn.Dense(self.atom_fea_len, name='ffn_2')(h)
        return x + ffn_out


class CEAttentionConvLayer(nn.Module):
    """Equiformer attention with CE message: σᵢ ⊙ σⱼ ⊙ R(r).

    Replaces AttentionConvLayer's concat-merge (cat([h_src, h_dst]) → Dense)
    with MACE-style triple product (h_src ⊙ h_dst ⊙ R), preserving
    equiformer's learned per-neighbor alpha attention on top.

    Concept: CE term is the message, attention learns which neighbor to weight.
    """
    atom_fea_len: int
    num_heads: int       = 4
    head_dim: int        = 16
    alpha_head_dim: int  = 8
    hidden_mult: int     = 2
    ffn_mult: int        = 2

    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        H, D, A = self.num_heads, self.head_dim, self.alpha_head_dim
        HD      = H * D
        H_ALPHA = H * A
        HIDDEN  = self.atom_fea_len * self.hidden_mult

        # ===== Sub-layer 1: Pre-LN + CE-message attention + residual =====
        x = nn.LayerNorm(name='ln1')(atom_in_fea)

        h_src = jnp.take(x, nbr_fea_idx, axis=0)              # σⱼ  (N, M, F)
        h_dst = jnp.tile(x[:, None, :], (1, M, 1))            # σᵢ  (N, M, F)
        radial = nn.Dense(self.atom_fea_len, name='radial')(nbr_fea)  # R(r)

        # ── CE term: σᵢ ⊙ σⱼ ⊙ R(r) (MACE-style 2-body message) ──
        msg = h_src * h_dst * radial                          # (N, M, F)

        # Project CE message → (m_main | α_feats)
        combined = nn.Dense(HIDDEN + H_ALPHA, name='msg_proj')(msg)
        m_main, alpha_feats = jnp.split(combined, [HIDDEN], axis=-1)
        m_main = nn.silu(m_main)

        value = nn.Dense(HD, name='value')(m_main).reshape(N, M, H, D)

        # Learned per-neighbor alpha (equiformer)
        alpha_feats = alpha_feats.reshape(N, M, H, A)
        alpha_feats = nn.LayerNorm(name='alpha_norm')(alpha_feats)
        alpha_feats = nn.leaky_relu(alpha_feats, negative_slope=0.2)
        a_std = 1.0 / (A ** 0.5)
        alpha_dot = self.param(
            'alpha_dot',
            lambda k, s: jax.random.uniform(k, s, minval=-a_std, maxval=a_std),
            (H, A))
        alpha = jnp.einsum('nmhk,hk->nmh', alpha_feats, alpha_dot)
        alpha = nn.softmax(alpha, axis=1)                     # softmax over M

        attn = jnp.sum(value * alpha[..., None], axis=1).reshape(N, HD)
        ga_out = nn.Dense(self.atom_fea_len, name='proj_out')(attn)
        x = atom_in_fea + ga_out                              # residual

        # ===== Sub-layer 2: Pre-LN + FFN + residual =====
        h = nn.LayerNorm(name='ln2')(x)
        h = nn.silu(nn.Dense(self.atom_fea_len * self.ffn_mult, name='ffn_1')(h))
        ffn_out = nn.Dense(self.atom_fea_len, name='ffn_2')(h)
        return x + ffn_out


def _pick_conv_layer(conv_type: str):
    """Resolve conv_type to a layer class.
    Only meaningful when the model uses CEInteractionLayer (Ising/SiSj/Heisenberg).

    Options:
      'product'  — CEInteractionLayer (legacy, with gate × magnitude)
      'mace'     — MACE v1 (original, no LayerNorm)
      'mace_v2'  — MACE v2 (LayerNorm + order-separated body weights)
      'mace_v3'  — MACE v3 (M-invariant /√M + balanced body order + centered A²)
    """
    if conv_type == 'mace':
        return MACEInteractionLayer
    if conv_type == 'mace_v2':
        return MACEv2InteractionLayer
    if conv_type == 'mace_v3':
        return MACEv3InteractionLayer
    return CEInteractionLayer  # 'product' (default, legacy)


# =============================================================================
# 3. Spin-specific Layers
# =============================================================================

class SpinEmbedding(nn.Module):
    """Learnable spin embedding: scalar spin → vector (v3)."""
    embed_dim: int = 32

    @nn.compact
    def __call__(self, atom_spins):
        x = nn.Dense(self.embed_dim)(atom_spins)
        x = nn.softplus(x)
        x = nn.Dense(self.embed_dim)(x)
        return x


class MagneticExchangeLayer(nn.Module):
    """Spin-asymmetric magnetic exchange (v3). E(σ) ≠ E(-σ)."""
    h_fea_len: int
    spin_embed_dim: int = 32

    @nn.compact
    def __call__(self, atom_chem_fea, nbr_fea, nbr_fea_idx, atom_spins):
        N, M = nbr_fea_idx.shape
        spin_emb = SpinEmbedding(self.spin_embed_dim)(atom_spins)

        center_chem = jnp.tile(atom_chem_fea[:, None, :], (1, M, 1))
        nbr_chem    = jnp.take(atom_chem_fea, nbr_fea_idx, axis=0)
        center_spin = jnp.tile(spin_emb[:, None, :], (1, M, 1))
        nbr_spin    = jnp.take(spin_emb, nbr_fea_idx, axis=0)

        pair_input = jnp.concatenate([
            center_chem + nbr_chem, jnp.abs(center_chem - nbr_chem),
            center_spin + nbr_spin, jnp.abs(center_spin - nbr_spin),
            nbr_fea], axis=-1)
        h    = nn.softplus(nn.Dense(self.h_fea_len)(pair_input))
        J_ij = nn.Dense(1)(nn.softplus(nn.Dense(self.h_fea_len // 2)(h)))

        spin_i = jnp.tile(atom_spins[:, None, :], (1, M, 1))
        spin_j = jnp.take(atom_spins, nbr_fea_idx, axis=0)
        return jnp.sum(J_ij * spin_i * spin_j, axis=1)


class EvenOddConvLayer(nn.Module):
    """Even-Odd Equivariant Convolution (v8).
    σ→-σ ⟹ odd→-odd, odd_cross invariant, even invariant → E(σ)=E(-σ) ✓
    """
    even_fea_len: int
    odd_fea_len: int
    edge_fea_len: int

    @nn.compact
    def __call__(self, even_node, odd_node, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        even_i = jnp.tile(even_node[:, None, :], (1, M, 1))
        even_j = jnp.take(even_node, nbr_fea_idx, axis=0)
        odd_i  = jnp.tile(odd_node[:, None, :], (1, M, 1))
        odd_j  = jnp.take(odd_node, nbr_fea_idx, axis=0)
        odd_cross = odd_i * odd_j  # even ✓

        # Even channel
        even_input = jnp.concatenate([even_i, even_j, nbr_fea, odd_cross], axis=-1)
        even_h    = nn.softplus(nn.Dense(self.even_fea_len, name='even_msg_1')(even_input))
        even_gate = nn.sigmoid(nn.Dense(self.even_fea_len, name='even_gate')(even_h))
        even_msg  = nn.softplus(nn.Dense(self.even_fea_len, name='even_msg_2')(even_h))
        even_agg  = jnp.sum(even_gate * even_msg, axis=1)

        # Odd channel
        odd_ie = odd_i * nn.Dense(self.odd_fea_len, name='odd_proj_ej')(even_j)
        odd_ei = nn.Dense(self.odd_fea_len, name='odd_proj_ei')(even_i) * odd_j
        odd_linear = nn.Dense(self.odd_fea_len, use_bias=False, name='odd_msg_1')(
            jnp.concatenate([odd_ie, odd_ei], axis=-1))
        odd_value = jnp.tanh(odd_linear)

        even_gate_in = jnp.concatenate([even_i, even_j, nbr_fea, odd_cross], axis=-1)
        odd_gate = nn.sigmoid(nn.Dense(self.odd_fea_len, name='odd_gate')(
            nn.softplus(nn.Dense(self.odd_fea_len, name='odd_gate_h')(even_gate_in))))
        odd_agg = jnp.sum(odd_gate * odd_value, axis=1)

        return even_node + even_agg, odd_node + odd_agg


class EvenOddEdgeUpdateLayer(nn.Module):
    """Edge update preserving even parity (v8)."""
    edge_fea_len: int
    odd_fea_len: int

    @nn.compact
    def __call__(self, even_node, odd_node, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        even_i = jnp.tile(even_node[:, None, :], (1, M, 1))
        even_j = jnp.take(even_node, nbr_fea_idx, axis=0)
        odd_i  = jnp.tile(odd_node[:, None, :], (1, M, 1))
        odd_j  = jnp.take(odd_node, nbr_fea_idx, axis=0)
        odd_cross = odd_i * odd_j  # even ✓
        concat = jnp.concatenate([nbr_fea, even_i, even_j, odd_cross], axis=-1)
        delta  = nn.Dense(self.edge_fea_len)(nn.softplus(nn.Dense(self.edge_fea_len)(concat)))
        return nbr_fea + delta


# =============================================================================
# 4. Heavy Models (atom_init + Gaussian, pt_mcmc compatibility)
# =============================================================================

class NeuralCE_Ising(nn.Module):
    """Base: Chemical only, no spin (heavy)."""
    pooler_cls: Callable
    atom_fea_len: int = 64
    nbr_fea_len: int  = 41
    n_conv: int       = 3
    h_fea_len: int    = 128

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        for _ in range(self.n_conv):
            atom_fea = CEInteractionLayer(self.atom_fea_len)(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class NeuralCE_Spin_v3(nn.Module):
    """v3: Spin Early Fusion (heavy). E(σ) ≠ E(-σ)."""
    pooler_cls: Callable
    atom_fea_len: int  = 64
    nbr_fea_len: int   = 41
    n_conv: int        = 3
    h_fea_len: int     = 128
    spin_embed_dim: int = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        if atom_spins is not None:
            spin_emb = SpinEmbedding(self.spin_embed_dim)(atom_spins)
            atom_fea = nn.Dense(self.atom_fea_len)(
                jnp.concatenate([atom_fea, spin_emb], axis=-1))
        for _ in range(self.n_conv):
            atom_fea = CEInteractionLayer(self.atom_fea_len)(atom_fea, nbr_fea, nbr_fea_idx)
        E_chem = ReadoutMLP(self.h_fea_len)(atom_fea)
        if atom_spins is not None:
            E_mag = MagneticExchangeLayer(self.h_fea_len, self.spin_embed_dim)(
                atom_fea, nbr_fea, nbr_fea_idx, atom_spins)
            alpha = self.param('mag_weight', nn.initializers.constant(0.5), (1,))
            E_chem = E_chem + alpha * E_mag
        return self.pooler_cls()(E_chem, **kwargs)


class NeuralCE_Spin_v8(nn.Module):
    """v8: Even-Odd, Dual Readout (heavy). E(σ) = E(-σ) ✓."""
    pooler_cls: Callable
    atom_fea_len: int = 64
    nbr_fea_len: int  = 41
    n_conv: int       = 3
    h_fea_len: int    = 128
    odd_fea_len: int  = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        even_node = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea   = nn.Dense(self.nbr_fea_len)(nbr_fea)
        odd_node  = (nn.Dense(self.odd_fea_len, use_bias=False)(atom_spins)
                     if atom_spins is not None
                     else jnp.zeros((atom_fea.shape[0], self.odd_fea_len)))
        for _ in range(self.n_conv):
            nbr_fea = EvenOddEdgeUpdateLayer(
                edge_fea_len=self.nbr_fea_len, odd_fea_len=self.odd_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
            even_node, odd_node = EvenOddConvLayer(
                even_fea_len=self.atom_fea_len, odd_fea_len=self.odd_fea_len,
                edge_fea_len=self.nbr_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
        E_chem = ReadoutMLP(self.h_fea_len)(even_node)
        E_mag  = ReadoutMLP(self.h_fea_len // 2)(even_node)
        return self.pooler_cls()(E_chem + E_mag, **kwargs)


class NeuralCE_Spin_v8u(nn.Module):
    """v8u: Even-Odd, Unified Readout (heavy). E(σ) = E(-σ) ✓."""
    pooler_cls: Callable
    atom_fea_len: int = 64
    nbr_fea_len: int  = 41
    n_conv: int       = 3
    h_fea_len: int    = 128
    odd_fea_len: int  = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        even_node = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea   = nn.Dense(self.nbr_fea_len)(nbr_fea)
        odd_node  = (nn.Dense(self.odd_fea_len, use_bias=False)(atom_spins)
                     if atom_spins is not None
                     else jnp.zeros((atom_fea.shape[0], self.odd_fea_len)))
        for _ in range(self.n_conv):
            nbr_fea = EvenOddEdgeUpdateLayer(
                edge_fea_len=self.nbr_fea_len, odd_fea_len=self.odd_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
            even_node, odd_node = EvenOddConvLayer(
                even_fea_len=self.atom_fea_len, odd_fea_len=self.odd_fea_len,
                edge_fea_len=self.nbr_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(even_node), **kwargs)


# =============================================================================
# 5. Lite Models (one-hot 4-dim node + shell edge, Sr excluded)
#
#    5-model ablation: 2×2 (backbone × spin) + ising baseline
#    Node: [Ti, Fe, O, Vo] one-hot (4-dim)
#    Edge: shell one-hot (n_shells-dim, tunable)
#    σᵢσⱼ models: edge_attr = [shell_onehot, σᵢσⱼ] → edge_dim = n_shells + 1
#    EvenOdd models: edge_attr = shell_onehot only → edge_dim = n_shells
#                    spin enters via odd channel (bias-free Dense embedding)
# =============================================================================

class NeuralCE_Ising_Lite(nn.Module):
    """Lite: No spin, Product backbone.

    Optional:
      conv_type='mace' — use MACEInteractionLayer (v2) instead of CEInteractionLayer.
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 4
    n_conv: int       = 2
    h_fea_len: int    = 32
    conv_type: str    = 'product'   # 'product' (default) or 'mace'

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        ConvLayer = _pick_conv_layer(self.conv_type)
        for _ in range(self.n_conv):
            atom_fea = ConvLayer(self.atom_fea_len)(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class NeuralCE_Heisenberg_Lite(nn.Module):
    """Lite: HEGNN — Ising backbone + learned Heisenberg Jᵢⱼ(r)·σᵢσⱼ.
    
    E_total = Σᵢ hᵢ(r) + Σᵢⱼ Jᵢⱼ(r)·σᵢσⱼ
    
    The chemical energy hᵢ is learned by Product-based conv (same as ising_lite).
    The Heisenberg coefficient Jᵢⱼ is learned from edge features via MLP.
    Spin enters ONLY through the explicit σᵢσⱼ product — no spin in conv.
    
    Benefits:
      - Spin contribution is fully decomposable: E_chem vs E_mag
      - Jᵢⱼ can be extracted for physical interpretation
      - Guaranteed E(σ) = E(-σ) since σᵢσⱼ is even under global spin flip
    
    Note: atom_spins is passed as kwarg (same as EvenOdd models).
          Edge features are shell one-hot only (no σᵢσⱼ in edge).
          nbr_fea_len = n_shells (same as ising_lite / evenodd_lite).
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 4     # = n_shells (shell one-hot only)
    n_conv: int       = 2
    h_fea_len: int    = 32
    odd_fea_len: int  = 4     # reused as J_ij hidden dim for consistency
    conv_type: str    = 'product'   # 'product' (default) or 'mace'

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        N, M = nbr_fea_idx.shape

        # ── Chemical energy: Ising backbone (spin-free) ──────────────
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea_h = nn.Dense(self.nbr_fea_len)(nbr_fea)
        ConvLayer = _pick_conv_layer(self.conv_type)
        for _ in range(self.n_conv):
            atom_fea = ConvLayer(self.atom_fea_len)(
                atom_fea, nbr_fea_h, nbr_fea_idx)

        E_chem = ReadoutMLP(self.h_fea_len)(atom_fea)

        # ── Heisenberg term: Jᵢⱼ(r)·σᵢσⱼ ────────────────────────────
        if atom_spins is not None:
            atom_i = jnp.tile(atom_fea[:, None, :], (1, M, 1))      # (N, M, F)
            atom_j = jnp.take(atom_fea, nbr_fea_idx, axis=0)        # (N, M, F)
            edge_context = jnp.concatenate([atom_i, atom_j, nbr_fea_h], axis=-1)

            J_h = nn.softplus(nn.Dense(self.odd_fea_len, name='J_fc1')(edge_context))
            J_ij = nn.Dense(1, name='J_fc2')(J_h)  # (N, M, 1)

            spins_flat = atom_spins.squeeze(-1)
            spin_i = jnp.tile(spins_flat[:, None], (1, M))
            spin_j = jnp.take(spins_flat, nbr_fea_idx, axis=0)
            sisj = (spin_i * spin_j)[..., None]

            nbr_mask = jnp.any(nbr_fea > 0, axis=-1, keepdims=True)

            E_mag_per_atom = jnp.sum(J_ij * sisj * nbr_mask, axis=1)
            E_site = E_chem + E_mag_per_atom
        else:
            E_site = E_chem

        return self.pooler_cls()(E_site, **kwargs)


class NeuralCE_EvenOdd_Lite(nn.Module):
    """Lite: Product backbone + EvenOdd spin. E(σ) = E(-σ) ✓.

    Note: conv_type field is accepted for API uniformity but is no-op here
          (EvenOdd conv structure cannot be swapped to MACE without losing
          spin parity invariance).
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 4     # = n_shells (no σᵢσⱼ in edge)
    n_conv: int       = 2
    h_fea_len: int    = 32
    odd_fea_len: int  = 4
    conv_type: str    = 'product'   # accepted for API uniformity; no-op here

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        even_node = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea   = nn.Dense(self.nbr_fea_len)(nbr_fea)
        odd_node  = (nn.Dense(self.odd_fea_len, use_bias=False)(atom_spins)
                     if atom_spins is not None
                     else jnp.zeros((atom_fea.shape[0], self.odd_fea_len)))
        for _ in range(self.n_conv):
            nbr_fea = EvenOddEdgeUpdateLayer(
                edge_fea_len=self.nbr_fea_len, odd_fea_len=self.odd_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
            even_node, odd_node = EvenOddConvLayer(
                even_fea_len=self.atom_fea_len, odd_fea_len=self.odd_fea_len,
                edge_fea_len=self.nbr_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(even_node), **kwargs)


class NeuralCE_SiSj_Lite(nn.Module):
    """Lite: Product backbone + σᵢσⱼ in edge. No EvenOdd.

    edge_attr includes [shell_onehot, σᵢσⱼ].
    Spin enters as σᵢσⱼ pair product in edge features only.
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 5     # = n_shells + 1 (σᵢσⱼ appended)
    n_conv: int       = 2
    h_fea_len: int    = 32
    conv_type: str    = 'product'   # 'product' (default) or 'mace'

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        # nbr_fea already contains [shell_onehot, σᵢσⱼ]
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea  = nn.Dense(self.nbr_fea_len)(nbr_fea)
        ConvLayer = _pick_conv_layer(self.conv_type)
        for _ in range(self.n_conv):
            atom_fea = ConvLayer(self.atom_fea_len)(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class GNN_SiSj_Lite(nn.Module):
    """Lite: Concat-MLP backbone + σᵢσⱼ in edge. No EvenOdd.

    edge_attr includes [shell_onehot, σᵢσⱼ].
    Note: conv_type field is accepted for API uniformity but is no-op here.
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 5     # = n_shells + 1
    n_conv: int       = 2
    h_fea_len: int    = 32
    conv_type: str    = 'product'   # accepted for API uniformity; no-op here

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea  = nn.Dense(self.nbr_fea_len)(nbr_fea)
        for _ in range(self.n_conv):
            atom_fea = ConcatConvLayer(self.atom_fea_len, self.nbr_fea_len)(
                atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class GNN_EvenOdd_Lite(nn.Module):
    """Lite: Concat-MLP backbone + EvenOdd spin. E(σ) = E(-σ) ✓.

    Note: conv_type field is accepted for API uniformity; no-op here.
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 4
    n_conv: int       = 2
    h_fea_len: int    = 32
    odd_fea_len: int  = 4
    conv_type: str    = 'product'   # accepted for API uniformity; no-op here

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        even_node = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea   = nn.Dense(self.nbr_fea_len)(nbr_fea)
        odd_node  = (nn.Dense(self.odd_fea_len, use_bias=False)(atom_spins)
                     if atom_spins is not None
                     else jnp.zeros((atom_fea.shape[0], self.odd_fea_len)))
        for _ in range(self.n_conv):
            nbr_fea = EvenOddEdgeUpdateLayer(
                edge_fea_len=self.nbr_fea_len, odd_fea_len=self.odd_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
            even_node, odd_node = EvenOddConvLayer(
                even_fea_len=self.atom_fea_len, odd_fea_len=self.odd_fea_len,
                edge_fea_len=self.nbr_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(even_node), **kwargs)


class ALIGNNLiteLayer(nn.Module):
    """ALIGNN-lite: alternating atom & edge updates, no explicit angle features.

    One cycle per layer:
      1. Edge update: e_ij ← e_ij + MLP([e_ij, h_i, h_j])
      2. Atom update: h_i  ← h_i  + Aggregate_j( MLP([h_j, e_ij]) )
    Pre-LayerNorm on atom features before each sub-step, residual after each.

    Designed for binary-alloy CE on 2-species + shell_onehot edge — keeps
    feature dims small and uses simple sum aggregation (vs attention's
    softmax) for chemistry inductive bias.
    """
    atom_fea_len: int
    edge_fea_len: int

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape

        # ===== Edge update: pre-LN atoms → cat [e_ij, h_i, h_j] → MLP → residual
        atom_normed_e = nn.LayerNorm(name='ln_edge_atom')(atom_fea)
        h_center = jnp.tile(atom_normed_e[:, None, :], (1, M, 1))
        h_nbr    = jnp.take(atom_normed_e, nbr_fea_idx, axis=0)
        edge_input = jnp.concatenate([nbr_fea, h_center, h_nbr], axis=-1)
        edge_h     = nn.silu(nn.Dense(self.edge_fea_len, name='edge_mlp1')(edge_input))
        edge_delta = nn.Dense(self.edge_fea_len, name='edge_mlp2')(edge_h)
        nbr_fea_new = nbr_fea + edge_delta

        # ===== Atom update: pre-LN atoms → for each nbr j build msg from h_j+e_ij
        # → sum → MLP → residual
        atom_normed_a = nn.LayerNorm(name='ln_atom')(atom_fea)
        h_nbr2 = jnp.take(atom_normed_a, nbr_fea_idx, axis=0)
        msg_input = jnp.concatenate([h_nbr2, nbr_fea_new], axis=-1)
        msg_h = nn.silu(nn.Dense(self.atom_fea_len, name='atom_mlp1')(msg_input))
        agg = jnp.sum(msg_h, axis=1)                                      # (N, F)
        atom_delta = nn.Dense(self.atom_fea_len, name='atom_mlp2')(agg)
        atom_fea_new = atom_fea + atom_delta

        return atom_fea_new, nbr_fea_new


class NeuralCE_ALIGNN_Lite(nn.Module):
    """Lite: ALIGNN-lite (alternating atom+edge update, no angle features).

    Lightweight design for small binary alloys: atom_fea_len=16, edge_fea_len=16.
    Uses sum aggregation (no attention softmax) but updates both atom AND edge
    features at every layer — different inductive bias than MACE (no triple
    products) and from equiformer (no soft attention weights).
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    edge_fea_len: int = 16
    nbr_fea_len: int  = 3      # raw nbr_fea dim (n_shells); projected to edge_fea_len
    n_conv: int       = 3
    h_fea_len: int    = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len, name='atom_init')(atom_fea)
        nbr_fea  = nn.Dense(self.edge_fea_len, name='edge_init')(nbr_fea)
        for _ in range(self.n_conv):
            atom_fea, nbr_fea = ALIGNNLiteLayer(
                atom_fea_len=self.atom_fea_len,
                edge_fea_len=self.edge_fea_len,
            )(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class NeuralCE_Equiformer_Lite(nn.Module):
    """Lite: EquiformerV2-inspired attention backbone (scalar-only), no spin.

    Structure identical to NeuralCE_Ising_Lite but with AttentionConvLayer
    (v2-style) in place of CEInteractionLayer.  Each AttentionConvLayer is a
    full TransBlockV2 (concat-merge multi-head attention + 2-layer FFN, both
    pre-LN with residual).
    """
    pooler_cls: Callable
    atom_fea_len: int    = 16
    nbr_fea_len: int     = 4
    n_conv: int          = 2
    h_fea_len: int       = 32
    num_heads: int       = 4
    head_dim: int        = 16
    alpha_head_dim: int  = 8
    hidden_mult: int     = 2
    ffn_mult: int        = 2

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        for _ in range(self.n_conv):
            atom_fea = AttentionConvLayer(
                atom_fea_len=self.atom_fea_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                alpha_head_dim=self.alpha_head_dim,
                hidden_mult=self.hidden_mult,
                ffn_mult=self.ffn_mult,
            )(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class NeuralCE_CEEquiformer_Lite(nn.Module):
    """Lite: Equiformer attention with CE message (σᵢ ⊙ σⱼ ⊙ R), no spin.

    Same wrapper as NeuralCE_Equiformer_Lite but uses CEAttentionConvLayer
    (CE triple-product message + softmax attention) in place of AttentionConvLayer
    (concat-merge + edge-gated SiLU).  Ablation pair: equiformer_lite vs
    ce_equiformer_lite isolates the message form.
    """
    pooler_cls: Callable
    atom_fea_len: int    = 16
    nbr_fea_len: int     = 4
    n_conv: int          = 2
    h_fea_len: int       = 32
    num_heads: int       = 4
    head_dim: int        = 16
    alpha_head_dim: int  = 8
    hidden_mult: int     = 2
    ffn_mult: int        = 2

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        for _ in range(self.n_conv):
            atom_fea = CEAttentionConvLayer(
                atom_fea_len=self.atom_fea_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                alpha_head_dim=self.alpha_head_dim,
                hidden_mult=self.hidden_mult,
                ffn_mult=self.ffn_mult,
            )(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


# =============================================================================
# 6. Registry & Factory
# =============================================================================

MODEL_REGISTRY = {
    # Heavy
    'ising':                NeuralCE_Ising,
    'spin_v3':              NeuralCE_Spin_v3,
    'spin_v8':              NeuralCE_Spin_v8,
    'spin_v8u':             NeuralCE_Spin_v8u,
    # Lite
    'ising_lite':           NeuralCE_Ising_Lite,
    'hegnn_lite':           NeuralCE_Heisenberg_Lite,
    'neuralce_evenodd_lite': NeuralCE_EvenOdd_Lite,
    'neuralce_sisj_lite':   NeuralCE_SiSj_Lite,
    'gnn_sisj_lite':        GNN_SiSj_Lite,
    'gnn_evenodd_lite':     GNN_EvenOdd_Lite,
    'equiformer_lite':      NeuralCE_Equiformer_Lite,
    'ce_equiformer_lite':   NeuralCE_CEEquiformer_Lite,
    'alignn_lite':          NeuralCE_ALIGNN_Lite,
}

SPIN_MODELS = {
    'spin_v3', 'spin_v8', 'spin_v8u',
    'hegnn_lite',
    'neuralce_evenodd_lite', 'gnn_evenodd_lite',
}

# Models that use σᵢσⱼ in edge features (not through odd channel)
SISJ_MODELS = {
    'neuralce_sisj_lite', 'gnn_sisj_lite',
}

LITE_MODELS = {
    'ising_lite', 'hegnn_lite',
    'neuralce_evenodd_lite', 'neuralce_sisj_lite',
    'gnn_sisj_lite', 'gnn_evenodd_lite',
    'equiformer_lite', 'ce_equiformer_lite',
    'alignn_lite',
}


def is_spin_model(model_type: str) -> bool:
    """True if model uses EvenOdd spin channel (atom_spins kwarg)."""
    return model_type in SPIN_MODELS

def is_sisj_model(model_type: str) -> bool:
    """True if model uses σᵢσⱼ in edge features."""
    return model_type in SISJ_MODELS

def is_lite_model(model_type: str) -> bool:
    return model_type in LITE_MODELS

def needs_spin(model_type: str) -> bool:
    """True if model needs spin info in any form (EvenOdd or SiSj)."""
    return model_type in SPIN_MODELS or model_type in SISJ_MODELS


def create_neuralce(model_type: str = 'ising', pool_mode: str = 'fixed',
                    readout_type: str = 'sum', **kwargs):
    """Factory function for NeuralCE models.

    Args:
        model_type:    One of MODEL_REGISTRY keys
        pool_mode:     'fixed' | 'padded' | 'variable'
        readout_type:  Legacy positional — pool reduction ('sum'|'mean'). For
                       backward compat, this name is kept; if 'sum'|'mean', it
                       sets pool reduction. If a model-field value (e.g. 'mlp',
                       'bayesian'), it's routed to the model class via kwargs.
        **kwargs:      atom_fea_len, nbr_fea_len, n_conv, h_fea_len,
                       odd_fea_len, spin_embed_dim, conv_type, readout_type, ...
                       Filtered against the model class's fields.
    """
    pooler_map = {
        ('fixed',    'sum'):  FixedSumPool,
        ('fixed',    'mean'): FixedMeanPool,
        ('padded',   'sum'):  PaddedSumPool,
        ('padded',   'mean'): PaddedMeanPool,
        ('variable', 'sum'):  VariableSumPool,
        ('variable', 'mean'): VariableMeanPool,
    }
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown: '{model_type}'. Choose from: {sorted(MODEL_REGISTRY.keys())}")

    # readout_type here is the POOL REDUCTION ('sum' | 'mean'), NOT a model
    # field.  After the Bayesian readout removal there is no model-level
    # readout_type field anymore — this parameter only selects FixedSumPool
    # vs FixedMeanPool / PaddedSumPool vs PaddedMeanPool, etc.
    if readout_type not in ('sum', 'mean'):
        raise ValueError(
            f"readout_type (pool reduction) must be 'sum' or 'mean', "
            f"got '{readout_type}'.  The model-level readout_type field "
            f"was removed in the BLR refactor — use post-hoc BLR via "
            f"neuralce.active_learning.blr instead.")

    pooler    = pooler_map[(pool_mode, readout_type)]
    model_cls = MODEL_REGISTRY[model_type]
    valid_fields = {f.name for f in model_cls.__dataclass_fields__.values()
                    if f.name != 'pooler_cls'}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return model_cls(pooler_cls=pooler, **filtered)


# =============================================================================
# 7. Symmetry Verification
# =============================================================================

def verify_spin_symmetry(model, params, atom_fea, nbr_fea, nbr_fea_idx,
                         atom_spins, **kwargs):
    """Check E(σ) == E(-σ) for a batch."""
    E_up   = model.apply(params, atom_fea, nbr_fea, nbr_fea_idx,
                         atom_spins=atom_spins, **kwargs)
    E_down = model.apply(params, atom_fea, nbr_fea, nbr_fea_idx,
                         atom_spins=-atom_spins, **kwargs)
    max_diff = jnp.max(jnp.abs(E_up - E_down))
    return {'E_up': E_up, 'E_down': E_down,
            'max_diff': float(max_diff), 'is_symmetric': float(max_diff) < 1e-5}


# =============================================================================
# 8. External baselines (CGCNN / ALIGNN / SchNet) — standalone reference ports
# =============================================================================
#
# Adapted from public reference implementations:
#   CGCNN     : txie-93/cgcnn        (cgcnn/model.py, ConvLayer)
#   ALIGNN    : usnistgov/alignn     (alignn/models/alignn.py, EdgeGatedGraphConv)
#   SchNet    : atomistic-machine-learning/schnetpack
#               (src/schnetpack/representation/schnet.py, SchNetInteraction)
#
# Deltas from the original implementations:
#   • atom_fea input is one-hot (species_map → n_species dim), not the 92-dim
#     atom_init.json embedding used by CGCNN.  An nn.Dense projects to F.
#   • edge feature nbr_fea is the project's continuous expansion (shell_onehot
#     or bessel), passed through a small projection to a reduced dim
#     (edge_fea_len, default 8) before use.
#   • No CE-style σᵢ ⊙ σⱼ ⊙ R(r) product anywhere — kept faithful to each
#     original message function (concat-MLP for CGCNN, src/dst/edge linear sum
#     for ALIGNN, continuous-filter for SchNet).
#   • BatchNorm in the originals → LayerNorm (small-batch friendly, no running
#     stats issues in pure-jit forward passes).
#
# These are independent models (not LITE_MODELS / SPIN_MODELS / SISJ_MODELS).
# Forward signature matches other models in this file:
#     __call__(atom_fea, nbr_fea, nbr_fea_idx, **kwargs)
# kwargs: batch_size, n_atoms_per_crystal, atom_mask.
# =============================================================================


class CGCNNConvLayer(nn.Module):
    """CGCNN ConvLayer (txie-93/cgcnn).

        z_ij      = cat(h_i, h_j, e_ij)              # (N, M, 2F + E)
        gated_ij  = LN(Linear(z_ij))                 # (N, M, 2F)
        g, c      = split(gated_ij, 2, axis=-1)
        msg_ij    = sigmoid(g) * softplus(c)         # gate × core
        h_i ← softplus( h_i + LN( Σ_j msg_ij ) )

    No σᵢ ⊙ σⱼ ⊙ R(r) product — concat-MLP message.
    BatchNorm → LayerNorm.
    """
    atom_fea_len: int

    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        h_nbr  = jnp.take(atom_in_fea, nbr_fea_idx, axis=0)             # (N, M, F)
        h_self = jnp.tile(atom_in_fea[:, None, :], (1, M, 1))           # (N, M, F)
        z      = jnp.concatenate([h_self, h_nbr, nbr_fea], axis=-1)     # (N, M, 2F+E)

        gated = nn.Dense(2 * self.atom_fea_len, name='fc_full')(z)      # (N, M, 2F)
        gated = nn.LayerNorm(name='ln_msg')(gated)
        gate, core = jnp.split(gated, 2, axis=-1)
        msg = nn.sigmoid(gate) * nn.softplus(core)                      # (N, M, F)

        nbr_sum = jnp.sum(msg, axis=1)                                  # (N, F)
        nbr_sum = nn.LayerNorm(name='ln_agg')(nbr_sum)
        return nn.softplus(atom_in_fea + nbr_sum)                       # residual


class NeuralCE_CGCNN(nn.Module):
    """Standalone CGCNN baseline (one-hot atom, reduced-dim continuous edge)."""
    pooler_cls: Callable
    atom_fea_len: int  = 32
    edge_fea_len: int  = 8
    n_conv: int        = 3
    h_fea_len: int     = 64

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len, name='atom_embed')(atom_fea)
        nbr_fea  = nn.Dense(self.edge_fea_len, name='edge_embed')(nbr_fea)
        for i in range(self.n_conv):
            atom_fea = CGCNNConvLayer(self.atom_fea_len, name=f'conv_{i}')(
                atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class ALIGNNEdgeGatedConv(nn.Module):
    """ALIGNN EdgeGatedGraphConv (usnistgov/alignn).

        m_ij    = src(h_i) + dst(h_j) + edge(e_ij)
        σ_ij    = sigmoid(m_ij)
        Bh_j    = dst_update(h_j)
        agg_i   = Σ_j (σ_ij ⊙ Bh_j) / (Σ_j σ_ij + ε)        # gated mean
        h_i'    = src_update(h_i) + agg_i
        h_i ← h_i + SiLU(LN(h_i'))                          # residual
        e_ij ← e_ij + SiLU(LN(m_ij))

    BatchNorm → LayerNorm.  No line graph / angle features.
    """
    fea_len: int

    @nn.compact
    def __call__(self, atom_in_fea, edge_in_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        h_nbr  = jnp.take(atom_in_fea, nbr_fea_idx, axis=0)             # (N, M, F)
        h_self = jnp.tile(atom_in_fea[:, None, :], (1, M, 1))           # (N, M, F)

        # Edge gate signal m_ij (split linear instead of cat-linear)
        e_src = nn.Dense(self.fea_len, name='src_gate')(h_self)
        e_dst = nn.Dense(self.fea_len, name='dst_gate')(h_nbr)
        e_e   = nn.Dense(self.fea_len, name='edge_gate')(edge_in_fea)
        m = e_src + e_dst + e_e                                          # (N, M, F)
        sigma = nn.sigmoid(m)

        # Gated mean aggregation
        Bh = nn.Dense(self.fea_len, name='dst_update')(h_nbr)            # (N, M, F)
        num = jnp.sum(sigma * Bh, axis=1)                                # (N, F)
        den = jnp.sum(sigma, axis=1) + 1e-6
        agg = num / den

        x = nn.Dense(self.fea_len, name='src_update')(atom_in_fea) + agg
        x = nn.silu(nn.LayerNorm(name='ln_node')(x))
        y = nn.silu(nn.LayerNorm(name='ln_edge')(m))

        return atom_in_fea + x, edge_in_fea + y                          # residual


class NeuralCE_ALIGNN(nn.Module):
    """Standalone ALIGNN baseline (one-hot atom, reduced-dim continuous edge).

    No line graph / angle features (we don't precompute triplet features).
    Alternating node+edge update via EdgeGatedGraphConv only.
    """
    pooler_cls: Callable
    atom_fea_len: int = 32
    edge_fea_len: int = 8
    n_conv: int       = 3
    h_fea_len: int    = 64

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.silu(nn.LayerNorm(name='atom_ln')(
            nn.Dense(self.atom_fea_len, name='atom_embed')(atom_fea)))
        edge_fea = nn.silu(nn.LayerNorm(name='edge_ln')(
            nn.Dense(self.edge_fea_len, name='edge_embed')(nbr_fea)))
        # Project edge to match atom dim for the gating linears
        edge_fea = nn.Dense(self.atom_fea_len, name='edge_to_atom')(edge_fea)
        for i in range(self.n_conv):
            atom_fea, edge_fea = ALIGNNEdgeGatedConv(
                self.atom_fea_len, name=f'conv_{i}')(atom_fea, edge_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class SchNetInteractionLayer(nn.Module):
    """SchNet continuous-filter interaction (atomistic-machine-learning/schnetpack).

        x      = in2f(h)                                  # (N, n_filters)
        W_ij   = filter_net(e_ij)                         # (N, M, n_filters)
        x_ij   = x[j] * W_ij                              # continuous-filter conv
        v_i    = Σ_j x_ij                                 # (N, n_filters)
        out    = f2out(v_i)                               # (N, F)
        h_i ← h_i + out                                   # residual

    No center-atom multiplication (x_i not used in message), no concat-MLP —
    pure continuous-filter convolution.  shifted_softplus replaced with SiLU.
    """
    atom_fea_len: int
    n_filters: int

    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        x = nn.Dense(self.n_filters, use_bias=False, name='in2f')(atom_in_fea)  # (N, F')
        W = nn.Dense(self.n_filters, name='filter_1')(nbr_fea)
        W = nn.silu(W)
        W = nn.Dense(self.n_filters, name='filter_2')(W)                         # (N, M, F')

        x_j  = jnp.take(x, nbr_fea_idx, axis=0)                                  # (N, M, F')
        x_ij = x_j * W                                                           # CF conv
        v    = jnp.sum(x_ij, axis=1)                                             # (N, F')

        out = nn.silu(nn.Dense(self.atom_fea_len, name='f2out_1')(v))
        out = nn.Dense(self.atom_fea_len, name='f2out_2')(out)
        return atom_in_fea + out                                                 # residual


class NeuralCE_SchNet(nn.Module):
    """Standalone SchNet baseline (one-hot atom, reduced-dim continuous edge)."""
    pooler_cls: Callable
    atom_fea_len: int = 32        # = n_atom_basis
    edge_fea_len: int = 8         # = n_rbf  (reduced)
    n_filters: int    = 32
    n_conv: int       = 3
    h_fea_len: int    = 64

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len, name='atom_embed')(atom_fea)
        nbr_fea  = nn.Dense(self.edge_fea_len, name='edge_embed')(nbr_fea)
        for i in range(self.n_conv):
            atom_fea = SchNetInteractionLayer(
                atom_fea_len=self.atom_fea_len,
                n_filters=self.n_filters,
                name=f'conv_{i}',
            )(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


# Register the three external baselines.  They are independent — not added to
# LITE_MODELS / SPIN_MODELS / SISJ_MODELS sets.
MODEL_REGISTRY['cgcnn']  = NeuralCE_CGCNN
MODEL_REGISTRY['alignn'] = NeuralCE_ALIGNN
MODEL_REGISTRY['schnet'] = NeuralCE_SchNet


# =============================================================================
# 9. Port + Heisenberg-head variants (spin channel = Jᵢⱼ(chem)·σᵢσⱼ ONLY)
# =============================================================================
#
# Benchmark decision (2026-08-19): every spin-aware model uses the SAME spin
# mechanism — the hegnn-style Heisenberg head copied from
# NeuralCE_Heisenberg_Lite. The port backbone stays EXACTLY its spin-blind
# config (no σᵢσⱼ in edges, no spin in nodes); after the conv stack:
#     edge_context = cat(h_i, h_j, edge_fea)      # edge_fea = backbone's edges
#     J_ij         = Dense(1)(softplus(Dense(odd_fea_len)(edge_context)))
#     E_mag,i      = Σ_j J_ij · σᵢσⱼ · nbr_mask   # mask from RAW edge input
#     E_site       = ReadoutMLP(h) + E_mag        # then usual pooling
# J is spin-independent and σᵢσⱼ is even ⇒ E(σ,s) = E(σ,−s) exactly.


class HeisenbergHead(nn.Module):
    """Jᵢⱼ(chemistry)·σᵢσⱼ head — exact copy of the NeuralCE_Heisenberg_Lite
    magnetic term (J_fc1/J_fc2 naming kept). `raw_edge_fea` is the un-embedded
    edge input; a real edge has some component > 0 (shell one-hot), padded /
    absent edges are all-zero → nbr_mask."""
    odd_fea_len: int = 4

    @nn.compact
    def __call__(self, atom_fea, edge_fea, raw_edge_fea, nbr_fea_idx, atom_spins):
        N, M = nbr_fea_idx.shape
        atom_i = jnp.tile(atom_fea[:, None, :], (1, M, 1))          # (N, M, F)
        atom_j = jnp.take(atom_fea, nbr_fea_idx, axis=0)            # (N, M, F)
        edge_context = jnp.concatenate([atom_i, atom_j, edge_fea], axis=-1)

        J_h = nn.softplus(nn.Dense(self.odd_fea_len, name='J_fc1')(edge_context))
        J_ij = nn.Dense(1, name='J_fc2')(J_h)                       # (N, M, 1)

        spins_flat = atom_spins.squeeze(-1)
        spin_i = jnp.tile(spins_flat[:, None], (1, M))
        spin_j = jnp.take(spins_flat, nbr_fea_idx, axis=0)
        sisj = (spin_i * spin_j)[..., None]

        nbr_mask = jnp.any(raw_edge_fea > 0, axis=-1, keepdims=True)
        return jnp.sum(J_ij * sisj * nbr_mask, axis=1)              # (N, 1)


class NeuralCE_CGCNN_Heis(nn.Module):
    """CGCNN blind backbone + Heisenberg head (spin enters ONLY via σᵢσⱼ)."""
    pooler_cls: Callable
    atom_fea_len: int  = 32
    edge_fea_len: int  = 8
    n_conv: int        = 3
    h_fea_len: int     = 64
    odd_fea_len: int   = 4

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        raw_edges = nbr_fea
        atom_fea = nn.Dense(self.atom_fea_len, name='atom_embed')(atom_fea)
        nbr_fea  = nn.Dense(self.edge_fea_len, name='edge_embed')(nbr_fea)
        for i in range(self.n_conv):
            atom_fea = CGCNNConvLayer(self.atom_fea_len, name=f'conv_{i}')(
                atom_fea, nbr_fea, nbr_fea_idx)
        e_site = ReadoutMLP(self.h_fea_len)(atom_fea)
        if atom_spins is not None:
            e_site = e_site + HeisenbergHead(self.odd_fea_len, name='heis')(
                atom_fea, nbr_fea, raw_edges, nbr_fea_idx, atom_spins)
        return self.pooler_cls()(e_site, **kwargs)


class NeuralCE_ALIGNN_Heis(nn.Module):
    """ALIGNN blind backbone + Heisenberg head (head sees the FINAL evolved
    edge features — the last edge representation of the stack)."""
    pooler_cls: Callable
    atom_fea_len: int = 32
    edge_fea_len: int = 8
    n_conv: int       = 3
    h_fea_len: int    = 64
    odd_fea_len: int  = 4

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        raw_edges = nbr_fea
        atom_fea = nn.silu(nn.LayerNorm(name='atom_ln')(
            nn.Dense(self.atom_fea_len, name='atom_embed')(atom_fea)))
        edge_fea = nn.silu(nn.LayerNorm(name='edge_ln')(
            nn.Dense(self.edge_fea_len, name='edge_embed')(nbr_fea)))
        edge_fea = nn.Dense(self.atom_fea_len, name='edge_to_atom')(edge_fea)
        for i in range(self.n_conv):
            atom_fea, edge_fea = ALIGNNEdgeGatedConv(
                self.atom_fea_len, name=f'conv_{i}')(atom_fea, edge_fea, nbr_fea_idx)
        e_site = ReadoutMLP(self.h_fea_len)(atom_fea)
        if atom_spins is not None:
            e_site = e_site + HeisenbergHead(self.odd_fea_len, name='heis')(
                atom_fea, edge_fea, raw_edges, nbr_fea_idx, atom_spins)
        return self.pooler_cls()(e_site, **kwargs)


class NeuralCE_SchNet_Heis(nn.Module):
    """SchNet blind backbone + Heisenberg head."""
    pooler_cls: Callable
    atom_fea_len: int = 32
    edge_fea_len: int = 8
    n_filters: int    = 32
    n_conv: int       = 3
    h_fea_len: int    = 64
    odd_fea_len: int  = 4

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        raw_edges = nbr_fea
        atom_fea = nn.Dense(self.atom_fea_len, name='atom_embed')(atom_fea)
        nbr_fea  = nn.Dense(self.edge_fea_len, name='edge_embed')(nbr_fea)
        for i in range(self.n_conv):
            atom_fea = SchNetInteractionLayer(
                atom_fea_len=self.atom_fea_len, n_filters=self.n_filters,
                name=f'conv_{i}')(atom_fea, nbr_fea, nbr_fea_idx)
        e_site = ReadoutMLP(self.h_fea_len)(atom_fea)
        if atom_spins is not None:
            e_site = e_site + HeisenbergHead(self.odd_fea_len, name='heis')(
                atom_fea, nbr_fea, raw_edges, nbr_fea_idx, atom_spins)
        return self.pooler_cls()(e_site, **kwargs)


MODEL_REGISTRY['cgcnn_heis']  = NeuralCE_CGCNN_Heis
MODEL_REGISTRY['alignn_heis'] = NeuralCE_ALIGNN_Heis
MODEL_REGISTRY['schnet_heis'] = NeuralCE_SchNet_Heis
# They consume the atom_spins kwarg (Heisenberg channel) → spin models.
SPIN_MODELS |= {'cgcnn_heis', 'alignn_heis', 'schnet_heis'}
