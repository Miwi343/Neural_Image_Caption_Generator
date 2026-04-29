"""
Adaptive attention with visual sentinel (Lu et al., CVPR 2017).
"Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning."

Extends the soft attention in attention.py with a visual sentinel: a vector the
decoder can attend to instead of the image, allowing the model to signal when
visual information is not needed (e.g. function words like "a", "the", "of").

Two classes:
    AdaptiveLSTMCell   — exposes the input gate needed to compute the sentinel
    AdaptiveAttention  — extended soft attention over L image + 1 sentinel locations
"""

import torch
import torch.nn as nn


class AdaptiveLSTMCell(nn.Module):
    """
    LSTM cell that manually computes and exposes the input gate.

    PyTorch's nn.LSTMCell does not expose individual gates. The sentinel
    formula s_t = input_gate ⊙ tanh(c_t) requires the input gate explicitly,
    so we implement the standard LSTM equations directly using one linear layer
    that maps [x_t || h_{t-1}] → 4 * hidden_size gate pre-activations.

    The (h_new, c_new) outputs are numerically equivalent to nn.LSTMCell.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size:  dimension of x_t
            hidden_size: dimension of h_t and c_t
        """
        super().__init__()
        self.hidden_size = hidden_size
        # One fused linear: [x_t || h_{t-1}] → [i, f, g, o] pre-activations
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(
        self,
        x: torch.Tensor,   # (batch, input_size)
        state: tuple,       # (h, c) each (batch, hidden_size)
    ):
        """
        Args:
            x:     (batch, input_size)   current input
            state: (h, c) each (batch, hidden_size)

        Returns:
            h_new:  (batch, hidden_size)  updated hidden state
            c_new:  (batch, hidden_size)  updated cell state
            i_gate: (batch, hidden_size)  input gate, used to compute the sentinel
        """
        h, c = state
        # Fused gate computation: single matmul over concatenated input + hidden
        gates = self.linear(torch.cat([x, h], dim=1))   # (batch, 4 * hidden_size)
        i, f, g, o = gates.chunk(4, dim=1)
        i_gate = torch.sigmoid(i)   # input gate  — returned for sentinel
        f_gate = torch.sigmoid(f)   # forget gate
        g_gate = torch.tanh(g)      # cell gate
        o_gate = torch.sigmoid(o)   # output gate
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        return h_new, c_new, i_gate


class AdaptiveAttention(nn.Module):
    """
    Soft attention extended with a visual sentinel (Lu et al., 2017).

    At each decoder step the model attends over L + 1 locations:
      - L  image annotation vectors a_1 … a_L
      - 1  visual sentinel vector   s_t

    The joint softmax produces:
      alpha: (batch, L)  visual attention weights over the image
      beta:  (batch,)    sentinel weight — fraction of attention on the sentinel

    Blended context:
      c̃_t = beta * s_t + Σ_i alpha_i * a_i

    The sentinel weight beta_t reveals which words require visual grounding:
      high beta → relying on language model (function words)
      low  beta → attending to the image    (visual / content words)

    Assumes decoder_dim == encoder_dim (both 512 in this project) so that
    sentinel and annotation vectors share the same dimension and can be
    combined without a projection layer.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Args:
            encoder_dim:   D  — annotation vector dimension (512)
            decoder_dim:   n  — LSTM hidden state dimension  (512)
            attention_dim: hidden size of the energy MLP     (512)
        """
        super().__init__()
        # Image annotation projections — same structure as the base Attention class
        self.encoder_att  = nn.Linear(encoder_dim,   attention_dim)
        self.decoder_att  = nn.Linear(decoder_dim,   attention_dim)
        self.full_att     = nn.Linear(attention_dim, 1)
        # Sentinel projection — separate weights to keep image/sentinel paths independent
        self.sentinel_att = nn.Linear(decoder_dim, attention_dim)
        self.tanh = nn.Tanh()

    def forward(
        self,
        encoder_out:    torch.Tensor,   # (batch, L, encoder_dim)
        decoder_hidden: torch.Tensor,   # (batch, decoder_dim)
        sentinel:       torch.Tensor,   # (batch, decoder_dim)
    ):
        """
        Args:
            encoder_out:    annotation vectors  (batch, L, encoder_dim)
            decoder_hidden: h_t                 (batch, decoder_dim)
            sentinel:       s_t                 (batch, decoder_dim)

        Returns:
            context: (batch, encoder_dim)  blended context c̃_t
            alpha:   (batch, L)            visual attention weights
            beta:    (batch,)              sentinel weight
        """
        # Image energy scores: e_i = v^T tanh(W_a * a_i + W_h * h_t)
        att1  = self.encoder_att(encoder_out)                     # (batch, L, attention_dim)
        att2  = self.decoder_att(decoder_hidden).unsqueeze(1)    # (batch, 1, attention_dim)
        e_img = self.full_att(self.tanh(att1 + att2)).squeeze(2) # (batch, L)

        # Sentinel energy score: ê_s = v^T tanh(W_s * s_t + W_h * h_t)
        att_s  = self.sentinel_att(sentinel)                      # (batch, attention_dim)
        att_h  = self.decoder_att(decoder_hidden)                 # (batch, attention_dim)
        e_sent = self.full_att(self.tanh(att_s + att_h))         # (batch, 1)

        # Extended softmax over L + 1 locations: [image_1 … image_L, sentinel]
        extended       = torch.cat([e_img, e_sent], dim=1)       # (batch, L + 1)
        extended_alpha = torch.softmax(extended, dim=1)           # (batch, L + 1)

        alpha = extended_alpha[:, :-1]   # (batch, L)   visual weights
        beta  = extended_alpha[:, -1]    # (batch,)     sentinel weight

        # Visual context: weighted sum over annotation vectors
        z_hat = (alpha.unsqueeze(2) * encoder_out).sum(dim=1)    # (batch, encoder_dim)

        # Blended context — sentinel has dim decoder_dim == encoder_dim == 512
        context = beta.unsqueeze(1) * sentinel + z_hat            # (batch, encoder_dim)

        return context, alpha, beta
