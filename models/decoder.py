"""
Decoder: LSTM with soft attention.

Paper section 3.1.2, equations 1-3 (LSTM cell) and 7 (deep output layer).

Key design choices that match the paper exactly:
  - LSTMCell (manual loop) rather than nn.LSTM, so attention can be called
    at every step before the cell update.
  - h0 / c0 initialised from the mean annotation vector via two separate MLPs
    (paper section 3.1.2, paragraph starting "The initial memory state…").
  - Beta gating scalar β predicted from h_{t-1} (section 4.2.1).
  - Deep output layer: p(y_t) ∝ exp(L_o(E*y_{t-1} + L_h*h_t + L_z*ẑ_t))
    (Eq. 7).

Ablation extensions (all default to the paper's exact behaviour):
  attention_mode : str
    "soft" — learned additive attention (default, paper Sec. 4.2).
    "none" — replace the dynamic context vector with the static mean-pooled
             annotation (ẑ_t = mean_i a_i).  No attention MLP is called;
             a uniform α is returned as a placeholder for visualization.
  use_beta_gate : bool
    True  — predict a scalar β from h_{t-1} and scale the context vector
            (default, paper Sec. 4.2.1).
    False — force β_t = 1 (identity gate) to test whether beta contributes.

Future work (Issue #6, deferred): Add scheduled sampling by introducing a
teacher-forcing ratio that decays from train.py; the default must remain full
teacher forcing so proposal-faithful runs are unchanged.
Future work (Issue #6, deferred): If hard attention is added, thread the sampled
context/index outputs and REINFORCE-specific losses through the decoder without
breaking the current soft-attention training path.
"""

import torch
import torch.nn as nn

from models.attention import Attention

VALID_ATTENTION_MODES = ("soft", "none")


class Decoder(nn.Module):
    """
    One-step-at-a-time LSTM decoder with configurable attention.

    During training, the full sequence is unrolled with teacher forcing.
    During inference, use greedy or beam search in evaluate.py.
    """

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 512,
        dropout: float = 0.5,
        attention_mode: str = "soft",
        use_beta_gate: bool = True,
    ):
        """
        Args:
            attention_dim:   hidden size of the attention MLP
            embed_dim:       word embedding dimension  (m in the paper)
            decoder_dim:     LSTM hidden size          (n in the paper)
            vocab_size:      size of output vocabulary (K in the paper)
            encoder_dim:     annotation vector dim     (D in the paper, =512)
            dropout:         dropout probability applied before deep output
            attention_mode:  "soft" | "none"  (default: "soft")
            use_beta_gate:   whether to apply the learned β scalar gate
                             (default: True, matches paper Sec. 4.2.1)
        """
        if attention_mode not in VALID_ATTENTION_MODES:
            raise ValueError(
                f"attention_mode must be one of {VALID_ATTENTION_MODES}, "
                f"got '{attention_mode}'."
            )
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.attention_mode = attention_mode
        self.use_beta_gate = use_beta_gate

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        # LSTMCell input: [embedding || context_vector] = embed_dim + encoder_dim.
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # Init MLPs: mean annotation → h0 / c0  (paper "init,c" and "init,h")
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Beta gate: scalar in (0, 1) to weight the context vector (Sec. 4.2.1).
        # Always constructed so checkpoints are compatible regardless of
        # use_beta_gate; the gate is simply bypassed (β forced to 1) when disabled.
        self.f_beta = nn.Linear(decoder_dim, 1)

        # Deep output layer projections (Eq. 7)
        self.L_h = nn.Linear(decoder_dim, embed_dim)
        self.L_z = nn.Linear(encoder_dim, embed_dim)
        self.L_o = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.L_o.weight, -0.1, 0.1)
        nn.init.constant_(self.L_o.bias, 0)

    def _init_hidden(self, encoder_out: torch.Tensor):
        """
        Compute h0 and c0 from mean annotation vector (paper section 3.1.2).

        Args:
            encoder_out: (batch, L, D)

        Returns:
            h: (batch, decoder_dim)
            c: (batch, decoder_dim)
        """
        mean_ann = encoder_out.mean(dim=1)   # (batch, D)
        h = torch.tanh(self.init_h(mean_ann))
        c = torch.tanh(self.init_c(mean_ann))
        return h, c

    def _compute_context(
        self,
        encoder_out: torch.Tensor,
        h: torch.Tensor,
    ):
        """
        Return (z_hat, alpha) according to the current attention_mode.

        attention_mode="soft":
            Learned additive attention (paper Eq. 4-6, 13).
        attention_mode="none":
            Static mean-pooled context ẑ = mean_i a_i; no attention MLP used.
            A uniform α is returned as a placeholder for visualization.

        Args:
            encoder_out: (batch, L, D)
            h:           (batch, decoder_dim)

        Returns:
            z_hat: (batch, D)  context vector
            alpha: (batch, L)  attention weights (may be uniform placeholder)
        """
        batch_size = encoder_out.size(0)
        L = encoder_out.size(1)

        if self.attention_mode == "soft":
            z_hat, alpha = self.attention(encoder_out, h)

        else:  # "none"
            z_hat = encoder_out.mean(dim=1)
            # Return uniform alpha as a blank placeholder for visualization.
            alpha = torch.full(
                (batch_size, L), 1.0 / L, device=encoder_out.device
            )

        return z_hat, alpha

    def decode_step(
        self,
        encoder_out: torch.Tensor,
        prev_words: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ):
        """
        Run one decoder step.

        Args:
            encoder_out: (batch, L, D) annotation vectors
            prev_words:  (batch,) token ids from the previous timestep
            h:           (batch, decoder_dim) previous hidden state
            c:           (batch, decoder_dim) previous cell state

        Returns:
            logits: (batch, vocab_size)
            h:      (batch, decoder_dim) updated hidden state
            c:      (batch, decoder_dim) updated cell state
            alpha:  (batch, L) attention weights (or uniform placeholder)
            beta:   (batch, 1) scalar context gate (or ones if disabled)
        """
        if prev_words.dim() == 2 and prev_words.size(1) == 1:
            prev_words = prev_words.squeeze(1)

        embed = self.embedding(prev_words)          # (batch, embed_dim)
        z_hat, alpha = self._compute_context(encoder_out, h)

        if self.use_beta_gate:
            beta = torch.sigmoid(self.f_beta(h))   # (batch, 1)
            z_hat = beta * z_hat
        else:
            beta = torch.ones(h.size(0), 1, device=h.device)

        lstm_input = torch.cat([embed, z_hat], dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))

        logits = self.L_o(
            self.dropout(embed + self.L_h(h) + self.L_z(z_hat))
        )
        return logits, h, c, alpha, beta

    def forward(
        self,
        encoder_out: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ):
        """
        Teacher-forced forward pass over the full caption sequence.

        Args:
            encoder_out:      annotation vectors (batch, L, encoder_dim)
            captions:         padded token ids   (batch, max_len)
            caption_lengths:  actual lengths     (batch,)

        Returns:
            predictions:  (batch, max_len-1, vocab_size)  logits
            alphas:       (batch, max_len-1, L)           attention weights
                          Used for doubly-stochastic regularisation (Eq. 14).
                          For attention_mode "none", contains the
                          1/L uniform placeholder values.
        """
        batch_size = encoder_out.size(0)
        L = encoder_out.size(1)
        decode_lengths = (caption_lengths - 1).clamp(min=1)
        max_decode_len = int(decode_lengths.max().item())

        h, c = self._init_hidden(encoder_out)

        predictions = torch.zeros(
            batch_size, max_decode_len, self.vocab_size, device=encoder_out.device
        )
        alphas = torch.zeros(
            batch_size, max_decode_len, L, device=encoder_out.device
        )

        for t in range(max_decode_len):
            logits, h, c, alpha, _ = self.decode_step(
                encoder_out=encoder_out,
                prev_words=captions[:, t],
                h=h,
                c=c,
            )
            predictions[:, t, :] = logits
            alphas[:, t, :] = alpha

        # Zero out attention for inactive timesteps so the doubly-stochastic
        # loss only accumulates over steps that actually ran.
        active = (
            torch.arange(max_decode_len, device=encoder_out.device).unsqueeze(0)
            < decode_lengths.unsqueeze(1)
        )
        alphas = alphas * active.unsqueeze(2).to(alphas.dtype)

        return predictions, alphas
