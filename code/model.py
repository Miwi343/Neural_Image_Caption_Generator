"""End-to-end Show, Attend and Tell model wrapper."""

import torch
import torch.nn as nn

from models import Decoder, Encoder


class ShowAttendTell(nn.Module):
    """Ties the frozen VGG encoder and soft-attention decoder together."""

    def __init__(
        self,
        vocab_size: int,
        attention_dim: int = 512,
        embed_dim: int = 512,
        decoder_dim: int = 512,
        encoder_dim: int = 512,
        dropout: float = 0.5,
        fine_tune_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(fine_tune=fine_tune_encoder)
        self.decoder = Decoder(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ):
        encoder_out = self.encoder(images)
        return self.decoder(encoder_out, captions, caption_lengths)
