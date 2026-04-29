"""
Decoder with adaptive attention and visual sentinel (Lu et al., CVPR 2017).
"Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning."

Key architectural differences from the base Decoder (decoder.py):
  1. AdaptiveLSTMCell replaces nn.LSTMCell — manually computes gates to expose
     the input gate required for sentinel computation.
  2. AdaptiveAttention replaces Attention — softmax is over L + 1 locations
     (L image regions + 1 sentinel).
  3. LSTM input is the word embedding only (not embedding + visual context).
     The attention context is injected solely through the deep output layer,
     letting the LSTM act as a pure language model. The hidden state h_t then
     encodes "what kind of word comes next linguistically" and is used to query
     image regions via attention — a clean separation of roles.
  4. The original f_beta gate is removed; the sentinel weight beta_t from the
     extended softmax is its conceptual replacement and carries richer meaning.

The decode_step() return signature — (logits, h, c, alpha, beta) — intentionally
matches Decoder.decode_step() so that greedy_decode_from_encoder_out and
beam_search_decode in utils/decoding.py work without modification.

forward() returns a 3-tuple (predictions, alphas, betas):
    alphas: (batch, T, L)  visual attention weights — used for doubly-stochastic reg
    betas:  (batch, T)     sentinel weights — high beta signals a function word
"""

import torch
import torch.nn as nn

from models.adaptive_attention import AdaptiveLSTMCell, AdaptiveAttention


class AdaptiveDecoder(nn.Module):
    """LSTM decoder with visual sentinel and adaptive attention."""

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 512,
        dropout: float = 0.5,
    ):
        """
        Args:
            attention_dim: hidden size of the attention MLP
            embed_dim:     word embedding dimension
            decoder_dim:   LSTM hidden state dimension
            vocab_size:    output vocabulary size
            encoder_dim:   annotation vector dimension (must equal decoder_dim
                           because sentinel and visual context are summed directly)
            dropout:       dropout probability before the deep output layer
        """
        super().__init__()

        self.encoder_dim   = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim     = embed_dim
        self.decoder_dim   = decoder_dim
        self.vocab_size    = vocab_size

        self.attention = AdaptiveAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout   = nn.Dropout(p=dropout)

        # LSTM input is word embedding only — visual context enters at output only.
        # This differs from the base Decoder where input = [embed || z_hat] (1024 dims).
        # Here input = embed (512 dims), so the LSTM weight matrix is smaller.
        self.lstm_cell = AdaptiveLSTMCell(embed_dim, decoder_dim)

        # Hidden state initialisation from mean annotation — same as base Decoder
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Deep output layer (Eq. 7) — same structure as base Decoder
        self.L_h = nn.Linear(decoder_dim, embed_dim)
        self.L_z = nn.Linear(encoder_dim, embed_dim)
        self.L_o = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Uniform initialisation for embedding and output layers."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.L_o.weight, -0.1, 0.1)
        nn.init.constant_(self.L_o.bias, 0)

    def _init_hidden(self, encoder_out: torch.Tensor):
        """
        Compute h0 and c0 from mean annotation vector.

        Args:
            encoder_out: (batch, L, encoder_dim)

        Returns:
            h: (batch, decoder_dim)
            c: (batch, decoder_dim)
        """
        mean_ann = encoder_out.mean(dim=1)           # (batch, encoder_dim)
        h = torch.tanh(self.init_h(mean_ann))
        c = torch.tanh(self.init_c(mean_ann))
        return h, c

    def decode_step(
        self,
        encoder_out: torch.Tensor,   # (batch, L, encoder_dim)
        prev_words:  torch.Tensor,   # (batch,) or (batch, 1) token ids
        h:           torch.Tensor,   # (batch, decoder_dim)
        c:           torch.Tensor,   # (batch, decoder_dim)
    ):
        """
        One adaptive-attention decoder step.

        Order of operations:
          1. Embed previous token.
          2. Run AdaptiveLSTMCell on embedding → h_new, c_new, input_gate.
          3. Compute sentinel: s_t = input_gate ⊙ tanh(c_new).
          4. Run AdaptiveAttention(h_new, s_t) → context, alpha, beta.
          5. Compute logits via deep output layer using context.

        Signature matches Decoder.decode_step() so that greedy and beam-search
        decoding helpers in utils/decoding.py require no modification.

        Returns:
            logits: (batch, vocab_size)
            h_new:  (batch, decoder_dim)    updated hidden state
            c_new:  (batch, decoder_dim)    updated cell state
            alpha:  (batch, L)              visual attention weights
            beta:   (batch,)                sentinel weight
        """
        if prev_words.dim() == 2 and prev_words.size(1) == 1:
            prev_words = prev_words.squeeze(1)

        embed = self.embedding(prev_words)                         # (batch, embed_dim)

        # LSTM step — input is the word embedding only
        h_new, c_new, input_gate = self.lstm_cell(embed, (h, c))

        # Visual sentinel: s_t = input_gate ⊙ tanh(c_new)
        sentinel = input_gate * torch.tanh(c_new)                  # (batch, decoder_dim)

        # Adaptive attention over L image regions + sentinel
        context, alpha, beta = self.attention(encoder_out, h_new, sentinel)
        # context: (batch, encoder_dim)

        # Deep output layer (Eq. 7): L_o(dropout(E*y_{t-1} + L_h*h_t + L_z*c̃_t))
        logits = self.L_o(
            self.dropout(
                embed
                + self.L_h(h_new)
                + self.L_z(context)
            )
        )

        return logits, h_new, c_new, alpha, beta

    def forward(
        self,
        encoder_out:     torch.Tensor,   # (batch, L, encoder_dim)
        captions:        torch.Tensor,   # (batch, max_len) token ids incl. <start>
        caption_lengths: torch.Tensor,   # (batch,)
    ):
        """
        Teacher-forced forward pass over the full caption sequence.

        Args:
            encoder_out:     annotation vectors  (batch, L, encoder_dim)
            captions:        padded token ids     (batch, max_len)
            caption_lengths: actual lengths       (batch,)

        Returns:
            predictions: (batch, T, vocab_size)  logits
            alphas:      (batch, T, L)            visual attention weights
                         Used for doubly-stochastic regularisation (Eq. 14).
            betas:       (batch, T)               sentinel weights
                         High beta at step t indicates the model relied on the
                         language model (sentinel) rather than the image.
        """
        batch_size = encoder_out.size(0)
        L = encoder_out.size(1)
        decode_lengths = (caption_lengths - 1).clamp(min=1)
        max_decode_len = int(decode_lengths.max().item())

        h, c = self._init_hidden(encoder_out)

        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size,
                                  device=encoder_out.device)
        alphas      = torch.zeros(batch_size, max_decode_len, L,
                                  device=encoder_out.device)
        betas       = torch.zeros(batch_size, max_decode_len,
                                  device=encoder_out.device)

        for t in range(max_decode_len):
            logits, h, c, alpha, beta = self.decode_step(
                encoder_out=encoder_out,
                prev_words=captions[:, t],
                h=h,
                c=c,
            )
            predictions[:, t, :] = logits
            alphas[:, t, :]      = alpha
            betas[:, t]          = beta

        # Zero out padding positions so DS regularisation ignores them
        active = (
            torch.arange(max_decode_len, device=encoder_out.device).unsqueeze(0)
            < decode_lengths.unsqueeze(1)
        )
        alphas = alphas * active.unsqueeze(2).to(alphas.dtype)
        betas  = betas  * active.to(betas.dtype)

        return predictions, alphas, betas
