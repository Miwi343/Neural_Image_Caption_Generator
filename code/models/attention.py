"""Soft attention over the encoder's spatial image features."""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Additive attention that returns a context vector and location weights."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """Create the small MLP used to score each image location."""
        super().__init__()

        # Match the paper setup: annotation projection has a bias, hidden-state
        # projection does not, and the scalar energy projection has a bias.
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.full_att = nn.Linear(attention_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        encoder_out: torch.Tensor,  # (batch, L, D)
        decoder_hidden: torch.Tensor,  # (batch, n)
    ):
        """Score each location, normalize the scores, and take a weighted sum."""
        encoder_projection = self.encoder_att(encoder_out)
        hidden_projection = self.decoder_att(decoder_hidden).unsqueeze(1)

        energies = self.full_att(
            self.tanh(encoder_projection + hidden_projection)
        ).squeeze(2)
        alpha = self.softmax(energies)
        z_hat = (alpha.unsqueeze(2) * encoder_out).sum(dim=1)

        return z_hat, alpha
