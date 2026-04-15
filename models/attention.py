"""
Attention mechanism (soft / deterministic).

Paper section 4.2, equations 4-6 and 13:
  e_ti  = f_att(a_i, h_{t-1})             (Eq. 4)
  α_ti  = softmax(e_ti)                    (Eq. 5)
  ẑ_t   = Σ_i α_ti * a_i                  (Eq. 13 — soft attention expectation)

The attention model f_att is a single-hidden-layer MLP conditioned on the
previous LSTM hidden state h_{t-1}.

TODO (Issue #5): Add hard (stochastic) attention variant using REINFORCE
TODO (Issue #5): Verify MLP hidden dimension matches paper (matches ATTENTION_DIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention over L=196 annotation vectors.

    For each timestep t the module computes:
      1. A scalar energy e_i for each of the L locations.
      2. Normalised weights α_i = softmax(e).
      3. Context vector ẑ = Σ α_i * a_i  (soft weighted sum).

    Returns both ẑ and α so the caller can:
      - Pass ẑ into the LSTM cell.
      - Accumulate α for the doubly-stochastic regularisation (Eq. 14).
      - Visualise attention maps (paper Fig. 2/3).

    TODO (Issue #5): Implement hard attention branch (sample from Multinoulli(α)).
    TODO (Issue #5): Profile whether a deeper MLP improves BLEU without
                     significant training slowdown.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Args:
            encoder_dim:   D = 512  (annotation vector dimension)
            decoder_dim:   n        (LSTM hidden state dimension)
            attention_dim: hidden size of the energy MLP

        TODO (Issue #5): Expose these as config dataclass fields.
        """
        super().__init__()

        # Linear projections (no bias so we control the affine transform)
        # W_a  : annotation vectors  a_i  → attention_dim
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # W_h  : previous hidden state h_{t-1} → attention_dim
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # v    : attention_dim → scalar energy
        self.full_att = nn.Linear(attention_dim, 1)

        # TODO (Issue #5): Consider tanh vs relu activation; paper uses tanh
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # normalise over L locations

    def forward(
        self,
        encoder_out: torch.Tensor,  # (batch, L, D)
        decoder_hidden: torch.Tensor,  # (batch, n)
    ):
        """
        Args:
            encoder_out:    annotation vectors, shape (batch_size, L, encoder_dim)
            decoder_hidden: h_{t-1},            shape (batch_size, decoder_dim)

        Returns:
            z_hat: context vector, shape (batch_size, encoder_dim)
            alpha: attention weights, shape (batch_size, L)

        TODO (Issue #5): Double-check broadcasting when batch_size=1 at inference.
        """
        # Project annotation vectors: (batch, L, attention_dim)
        att1 = self.encoder_att(encoder_out)

        # Project hidden state and broadcast over L: (batch, 1, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)

        # Compute energy scores e_ti (Eq. 4): (batch, L, 1) → (batch, L)
        # TODO (Issue #5): Paper uses tanh; confirm relu is equivalent here
        e = self.full_att(self.relu(att1 + att2)).squeeze(2)

        # Normalise to get α weights (Eq. 5): (batch, L)
        alpha = self.softmax(e)

        # Soft context vector ẑ (Eq. 13): weighted sum over annotation vectors
        # (batch, L, 1) * (batch, L, D) → sum over L → (batch, D)
        z_hat = (alpha.unsqueeze(2) * encoder_out).sum(dim=1)

        return z_hat, alpha
