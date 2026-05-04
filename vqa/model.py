"""
VQA yes/no model — minimal adaptation of the paper's architecture.

Reused from models/ (unchanged):
  - Encoder   : VGG-16 → 196×512 annotation vectors  (paper section 3.1.1)
  - Attention : soft Bahdanau attention               (paper section 4.2, Eq. 4-6, 13)

New (VQA-only):
  - QuestionEncoder : GRU over question tokens → question vector q
  - VQAModel        : replaces the LSTM decoder with question encoder + binary
                      classifier; passes q into the paper's Attention in place
                      of the LSTM hidden state h_{t-1}.

The attention formula is identical to the paper:
    e_i   = v^T tanh( W_a * a_i  +  W_q * q )
    α     = softmax(e)
    ẑ     = Σ_i α_i * a_i
"""

import torch
import torch.nn as nn

from models.encoder import Encoder    # paper section 3.1.1 — unchanged
from models.attention import Attention  # paper section 4.2   — unchanged


class QuestionEncoder(nn.Module):
    """GRU over question word embeddings → fixed-size question vector."""

    def __init__(self, vocab_size: int, embed_dim: int, gru_dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, gru_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].zero_()

    def forward(self, questions: torch.Tensor, q_lens: torch.Tensor) -> torch.Tensor:
        embed = self.dropout(self.embedding(questions))
        packed = nn.utils.rnn.pack_padded_sequence(
            embed, q_lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        return h_n.squeeze(0)  # (B, gru_dim)


class VQAModel(nn.Module):
    """
    Full yes/no VQA model.

    Forward pass:
        encoder_out = Encoder(image)             # (B, 196, 512)  — paper encoder
        q_vec       = QuestionEncoder(question)  # (B, 512)        — new
        ẑ, α        = Attention(encoder_out, q_vec)  # paper attention, q replaces h_{t-1}
        logit       = classifier(cat([ẑ, q_vec]))    # (B,)         — new
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
        self.question_encoder = QuestionEncoder(vocab_size, embed_dim, gru_dim)
        # Reuse the paper's Attention exactly; decoder_dim matches gru_dim
        self.attention = Attention(
            encoder_dim=encoder_dim,
            decoder_dim=gru_dim,
            attention_dim=attention_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim + gru_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, images, questions, q_lens):
        encoder_out = self.encoder(images)                       # (B, 196, 512)
        q_vec       = self.question_encoder(questions, q_lens)   # (B, 512)
        z_hat, alpha = self.attention(encoder_out, q_vec)        # (B, 512), (B, 196)
        logit = self.classifier(torch.cat([z_hat, q_vec], dim=1)).squeeze(1)  # (B,)
        return logit, alpha
