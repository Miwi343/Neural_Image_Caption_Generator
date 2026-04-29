"""
VQA yes/no model.

Architecture
------------
This reuses the existing VGG-16 Encoder unchanged — the same 196×512
annotation vectors used for captioning.  On top of that we add:

  1. QuestionEncoder  — GRU over question word embeddings → question vector q
  2. VQAAttention     — soft attention over the 196 image locations,
                        conditioned on q (same Bahdanau style as the caption
                        Attention module)
  3. Binary classifier — fuses attended image vector + question vector → logit

Forward pass:
    encoder_out : (B, 196, 512)   ← from Encoder (reused as-is)
    question    : (B, max_q_len)  ← padded token ids
    q_len       : (B,)            ← actual lengths for GRU packing
    → logit     : (B,)            ← raw score; sigmoid → P(yes)
    → alpha     : (B, 196)        ← attention weights (for visualisation)

Loss: BCEWithLogitsLoss

The doubly-stochastic attention regularisation from the caption model
(Eq. 14) is optionally carried over: encouraging Σ_t α_ti ≈ 1.  Here
there is only one attention step so the penalty is just
λ * mean((1 - Σ_i α_i)^2) which collapses to 0 for any valid softmax.
We keep the hook in case multi-step attention is added later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class QuestionEncoder(nn.Module):
    """
    Encodes a variable-length question into a fixed-size vector.

    Uses a GRU (simpler than LSTM, sufficient for short questions).
    The final hidden state of the last GRU layer is used as the question
    representation.
    """

    def __init__(self, vocab_size: int, embed_dim: int, gru_dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, gru_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].zero_()  # keep pad embedding at zero

    def forward(self, questions: torch.Tensor, q_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            questions: (B, max_q_len)  padded token ids
            q_lens:    (B,)            actual lengths

        Returns:
            q_vec: (B, gru_dim)  last hidden state
        """
        embed = self.dropout(self.embedding(questions))  # (B, L_q, embed_dim)

        # Pack so the GRU ignores padding
        packed = nn.utils.rnn.pack_padded_sequence(
            embed, q_lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)   # h_n: (1, B, gru_dim)
        return h_n.squeeze(0)       # (B, gru_dim)


class VQAAttention(nn.Module):
    """
    Single-step soft attention over image annotation vectors, conditioned on
    the question vector.  Mirrors the caption Attention module exactly.

    e_i   = v^T tanh( W_a * a_i  +  W_q * q )   for i in 1..L
    α     = softmax(e)
    ẑ     = Σ_i α_i * a_i
    """

    def __init__(self, encoder_dim: int, question_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att  = nn.Linear(encoder_dim, attention_dim)
        self.question_att = nn.Linear(question_dim, attention_dim)
        self.full_att     = nn.Linear(attention_dim, 1)

    def forward(
        self,
        encoder_out: torch.Tensor,   # (B, L, encoder_dim)
        question_vec: torch.Tensor,  # (B, question_dim)
    ):
        """
        Returns:
            z_hat: (B, encoder_dim)   attended image vector
            alpha: (B, L)             attention weights
        """
        att1 = self.encoder_att(encoder_out)              # (B, L, att_dim)
        att2 = self.question_att(question_vec).unsqueeze(1)  # (B, 1, att_dim)
        e    = self.full_att(torch.tanh(att1 + att2)).squeeze(2)  # (B, L)
        alpha = F.softmax(e, dim=1)                       # (B, L)
        z_hat = (alpha.unsqueeze(2) * encoder_out).sum(1) # (B, encoder_dim)
        return z_hat, alpha


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class VQAModel(nn.Module):
    """
    Full yes/no VQA model.

    The Encoder is constructed internally and exposed as self.encoder so
    the training script can optionally load pretrained caption weights into it.

    Args:
        vocab_size:        question vocabulary size
        embed_dim:         question word embedding dimension
        gru_dim:           GRU hidden dimension
        attention_dim:     attention MLP hidden dimension
        encoder_dim:       VGG annotation vector dimension (512, fixed)
        dropout:           dropout rate in classifier head
        fine_tune_encoder: whether to unfreeze top VGG layers
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        gru_dim: int = 512,
        attention_dim: int = 512,
        encoder_dim: int = 512,
        dropout: float = 0.5,
        fine_tune_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = Encoder(fine_tune=fine_tune_encoder)

        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            gru_dim=gru_dim,
        )

        self.attention = VQAAttention(
            encoder_dim=encoder_dim,
            question_dim=gru_dim,
            attention_dim=attention_dim,
        )

        # Fusion: concatenate attended image vector + question vector → scalar
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim + gru_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        images: torch.Tensor,     # (B, 3, 224, 224)
        questions: torch.Tensor,  # (B, max_q_len)
        q_lens: torch.Tensor,     # (B,)
    ):
        """
        Returns:
            logit: (B,)    raw score (apply sigmoid for probability)
            alpha: (B, L)  attention weights for visualisation
        """
        encoder_out = self.encoder(images)               # (B, 196, 512)
        q_vec       = self.question_encoder(questions, q_lens)  # (B, gru_dim)
        z_hat, alpha = self.attention(encoder_out, q_vec) # (B, 512), (B, 196)

        fused = torch.cat([z_hat, q_vec], dim=1)         # (B, 1024)
        logit = self.classifier(fused).squeeze(1)         # (B,)
        return logit, alpha
