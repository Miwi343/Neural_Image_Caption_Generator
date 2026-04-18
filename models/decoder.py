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

TODO (Issue #6, deferred): Add scheduled sampling by introducing a
teacher-forcing ratio that decays from train.py; the default must remain full
teacher forcing so proposal-faithful runs are unchanged.
TODO (Issue #6, deferred): If hard attention is added, thread the sampled
context/index outputs and REINFORCE-specific losses through the decoder without
breaking the current soft-attention training path.
TODO (Issue #6): Extract a one-step decode helper that returns logits, hidden
state, cell state, and alpha so greedy decoding and beam search stop duplicating
decoder internals across evaluate.py and visualize.py.
"""

import torch
import torch.nn as nn

from models.attention import Attention


class Decoder(nn.Module):
    """
    One-step-at-a-time LSTM decoder with soft attention.

    During training, the full sequence is unrolled with teacher forcing.
    During inference, use greedy or beam search in evaluate.py.

    TODO (Issue #6, deferred): If scheduled sampling is implemented, expose the
    teacher-forcing ratio and decay schedule in train.py while keeping the
    default ratio at 1.0 for baseline proposal runs.
    """

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
            embed_dim:     word embedding dimension  (m in the paper)
            decoder_dim:   LSTM hidden size          (n in the paper)
            vocab_size:    size of output vocabulary (K in the paper)
            encoder_dim:   annotation vector dim     (D in the paper, =512)
            dropout:       dropout probability applied before deep output

        TODO (Issue #6, deferred): If constructor cleanup is assigned, replace
        the long positional arg list with a decoder config object shared by
        training, evaluation, and visualization scripts.
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        # LSTMCell input: [embedding || context_vector] = embed_dim + encoder_dim
        # TODO (Issue #6): Add a short architecture note or test confirming that
        # the explicit LSTMCell input is `[embedding_t, context_t]` and that
        # `h_{t-1}` is already provided implicitly via the recurrent state tuple.
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # Init MLPs: mean annotation → h0 / c0  (paper "init,c" and "init,h")
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Beta gate: scalar ∈ (0,1) to weight the context vector (section 4.2.1)
        # TODO (Issue #6, deferred): Run an ablation with and without beta
        # gating on Flickr8k, then record BLEU and qualitative attention-map
        # differences before deciding whether to keep the gate permanently.
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        # Deep output layer projections (Eq. 7):
        #   L_o  : vocab projection       K×m
        #   L_h  : hidden → embed space   m×n
        #   L_z  : context → embed space  m×D
        # E * y_{t-1} is already embed_dim so we reuse the embedding matrix
        self.L_h = nn.Linear(decoder_dim, embed_dim)
        self.L_z = nn.Linear(encoder_dim, embed_dim)
        self.L_o = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Uniform initialisation for embedding and output layers."""
        # TODO (Issue #6, deferred): Compare the current uniform init to Xavier
        # on a short Flickr8k training run and keep the simpler scheme unless
        # the alternate init improves convergence or BLEU consistently.
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

        TODO (Issue #6): Add a unit test asserting both `h0` and `c0` pass
        through `tanh`, which keeps their values bounded in `[-1, 1]`.
        """
        mean_ann = encoder_out.mean(dim=1)   # (batch, D)
        h = torch.tanh(self.init_h(mean_ann))
        c = torch.tanh(self.init_c(mean_ann))
        return h, c

    def forward(
        self,
        encoder_out: torch.Tensor,   # (batch, L, D)
        captions: torch.Tensor,      # (batch, max_len)  token ids incl. <start>
        caption_lengths: torch.Tensor,  # (batch,)  actual lengths (incl. <start>)
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

        TODO (Issue #6, deferred): If qualitative analysis expands beyond alpha
        maps, return the per-step beta gates as an optional third output without
        breaking current callers that unpack `(predictions, alphas)`.
        TODO (Issue #6, deferred): If scheduled sampling is added, make it opt-in
        and preserve the current always-teacher-forced path as the baseline.
        """
        batch_size = encoder_out.size(0)
        L = encoder_out.size(1)
        # Decode length = caption length minus the final <end> token
        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_len = int(max(decode_lengths))

        h, c = self._init_hidden(encoder_out)

        # Pre-embed all tokens; we index into this at each step
        # TODO (Issue #6, deferred): If embedding dropout is added, apply it
        # once to `embeddings` here and verify train/eval mode toggles it
        # correctly without changing tensor shapes.
        embeddings = self.embedding(captions)  # (batch, max_len, embed_dim)

        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, L, device=encoder_out.device)

        for t in range(max_decode_len):
            # Only process samples whose caption is still active at step t
            batch_size_t = sum([l > t for l in decode_lengths])

            # TODO (Issue #6, deferred): Replace this prefix slice with a
            # boolean active-sequence mask if you need support for unsorted or
            # packed batches; verify the resulting logits still align with targets.
            z_hat, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )

            # Beta gating scalar (section 4.2.1): weight context by sigmoid gate
            beta = torch.sigmoid(self.f_beta(h[:batch_size_t]))  # (batch_t, D)
            z_hat = beta * z_hat

            # LSTM step (Eq. 1-3): input = [word_embed || context]
            lstm_input = torch.cat(
                [embeddings[:batch_size_t, t, :], z_hat], dim=1
            )  # (batch_t, embed_dim + encoder_dim)

            h, c = self.lstm_cell(
                lstm_input,
                (h[:batch_size_t], c[:batch_size_t])
            )

            # Deep output (Eq. 7):
            # p ∝ exp(L_o(E*y_{t-1} + L_h*h_t + L_z*ẑ_t))
            # TODO (Issue #6): Add a smoke test confirming dropout is applied to
            # the deep-output sum before `L_o` during training and is disabled in
            # eval mode; if that test is added, this note can be removed.
            preds = self.L_o(
                self.dropout(
                    embeddings[:batch_size_t, t, :]  # E * y_{t-1}
                    + self.L_h(h)                    # L_h * h_t
                    + self.L_z(z_hat)                # L_z * ẑ_t
                )
            )  # (batch_t, vocab_size)

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas
