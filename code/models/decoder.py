"""LSTM decoder with soft attention and the paper's deep output layer."""

import torch
import torch.nn as nn

from models.attention import Attention


class PaperDropout(nn.Module):
    """Non-inverted dropout used by the authors' Theano code."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError("dropout probability must be in [0, 1).")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        if self.training:
            return x * torch.empty_like(x).bernoulli_(keep_prob)
        return x * keep_prob


class Decoder(nn.Module):
    """Decode captions one step at a time with teacher forcing in training."""

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 512,
        dropout: float = 0.5,
    ):
        """Create the decoder layers."""
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = PaperDropout(p=dropout)

        # Attention runs before every LSTMCell update, so the context vector is
        # concatenated with the previous word embedding at each step.
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.f_beta = nn.Linear(decoder_dim, 1)

        # Deep output layer: embed + hidden projection + context projection.
        self.L_h = nn.Linear(decoder_dim, embed_dim)
        self.L_z = nn.Linear(encoder_dim, embed_dim)
        self.L_o = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialise weights like the authors' Theano implementation."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        linear_layers = [
            self.attention.encoder_att,
            self.attention.decoder_att,
            self.attention.full_att,
            self.init_h,
            self.init_c,
            self.f_beta,
            self.L_h,
            self.L_z,
            self.L_o,
        ]
        for layer in linear_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

        nn.init.normal_(self.lstm_cell.weight_ih, mean=0.0, std=0.01)
        for recurrent_block in self.lstm_cell.weight_hh.chunk(4, dim=0):
            nn.init.orthogonal_(recurrent_block)
        nn.init.constant_(self.lstm_cell.bias_ih, 0.0)
        nn.init.constant_(self.lstm_cell.bias_hh, 0.0)

    def _init_hidden(self, encoder_out: torch.Tensor):
        """Initialise h0 and c0 from the mean encoder annotation."""
        mean_ann = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean_ann))
        c = torch.tanh(self.init_c(mean_ann))
        return h, c

    def decode_step(
        self,
        encoder_out: torch.Tensor,
        prev_words: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ):
        """Run one attention + LSTM step."""
        if prev_words.dim() == 2 and prev_words.size(1) == 1:
            prev_words = prev_words.squeeze(1)

        embed = self.embedding(prev_words)
        z_hat, alpha = self.attention(encoder_out, h)

        beta = torch.sigmoid(self.f_beta(h))
        z_hat = beta * z_hat

        lstm_input = torch.cat([embed, z_hat], dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))

        output_input = embed + self.L_h(h) + self.L_z(z_hat)
        logits = self.L_o(self.dropout(output_input))
        return logits, h, c, alpha, beta

    def forward(
        self,
        encoder_out: torch.Tensor,   # (batch, L, D)
        captions: torch.Tensor,      # (batch, max_len)  token ids incl. <start>
        caption_lengths: torch.Tensor,  # (batch,)  actual lengths (incl. <start>)
    ):
        """Teacher-force over the caption and return logits plus attention maps."""
        batch_size = encoder_out.size(0)
        locations = encoder_out.size(1)
        decode_lengths = (caption_lengths - 1).clamp(min=1)
        max_decode_len = int(decode_lengths.max().item())

        h, c = self._init_hidden(encoder_out)

        predictions = torch.zeros(
            batch_size,
            max_decode_len,
            self.vocab_size,
            device=encoder_out.device,
        )
        alphas = torch.zeros(
            batch_size,
            max_decode_len,
            locations,
            device=encoder_out.device,
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

        active_steps = (
            torch.arange(max_decode_len, device=encoder_out.device).unsqueeze(0)
            < decode_lengths.unsqueeze(1)
        )
        alphas = alphas * active_steps.unsqueeze(2).to(alphas.dtype)

        return predictions, alphas
