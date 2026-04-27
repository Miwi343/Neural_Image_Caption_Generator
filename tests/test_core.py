from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models import Attention, Decoder, Encoder
from train import doubly_stochastic_attention_loss
from utils import Vocabulary, compute_bleu, tokenize_caption
from utils.dataset import LengthBucketSampler


def test_tokenizer_and_vocab_strip_punctuation():
    assert tokenize_caption("A Dog, on the BEACH!") == [
        "a",
        "dog",
        "on",
        "the",
        "beach",
    ]

    vocab = Vocabulary(max_size=10)
    vocab.build(["A dog runs.", "A dog jumps!"])
    encoded = vocab.encode("A dog.")

    assert encoded[0] == 1
    assert encoded[-1] == 2
    assert vocab.decode(encoded) == "a dog"


def test_bleu_omits_brevity_penalty():
    scores = compute_bleu(
        hypotheses=[["a"]],
        references=[[["a", "dog"]]],
    )

    assert scores["bleu1"] == pytest.approx(1.0)


def test_attention_shapes_and_normalization():
    attention = Attention(encoder_dim=512, decoder_dim=64, attention_dim=32)
    encoder_out = torch.randn(2, 196, 512)
    hidden = torch.randn(2, 64)

    context, alpha = attention(encoder_out, hidden)

    assert context.shape == (2, 512)
    assert alpha.shape == (2, 196)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(2), atol=1e-5)


def test_decoder_forward_masks_inactive_attention_and_uses_scalar_beta():
    decoder = Decoder(
        attention_dim=32,
        embed_dim=64,
        decoder_dim=64,
        vocab_size=20,
        dropout=0.0,
    )
    encoder_out = torch.randn(2, 196, 512)
    captions = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])
    lengths = torch.tensor([4, 3])

    predictions, alphas = decoder(encoder_out, captions, lengths)
    h, c = decoder._init_hidden(encoder_out)
    _, _, _, alpha, beta = decoder.decode_step(encoder_out, captions[:, 0], h, c)

    assert predictions.shape == (2, 3, 20)
    assert alphas.shape == (2, 3, 196)
    assert torch.allclose(alphas[1, 2], torch.zeros(196), atol=1e-6)
    assert alpha.shape == (2, 196)
    assert beta.shape == (2, 1)
    assert h.max() <= 1.0 and h.min() >= -1.0
    assert c.max() <= 1.0 and c.min() >= -1.0


def test_doubly_stochastic_loss_averages_over_batch_and_locations():
    alphas = torch.zeros(2, 3, 4)
    alphas[:, :, 0] = 1.0 / 3.0

    loss = doubly_stochastic_attention_loss(alphas, weight=1.0)

    # Location 0 sums to 1 → penalty 0; locations 1-3 sum to 0 → penalty 1 each.
    # mean over (batch=2, locations=4): (0+1+1+1)*2 / 8 = 0.75
    assert loss.item() == pytest.approx(0.75)


def test_encoder_reshapes_vgg_feature_map(monkeypatch):
    class DummyFeatureMap(nn.Module):
        def forward(self, images):
            return torch.zeros(images.size(0), 512, 14, 14)

    dummy_vgg = SimpleNamespace(features=nn.Sequential(DummyFeatureMap()))

    import torchvision.models as tv_models

    monkeypatch.setattr(tv_models, "vgg16", lambda weights: dummy_vgg)
    encoder = Encoder(fine_tune=False)

    out = encoder(torch.randn(2, 3, 224, 224))

    assert out.shape == (2, 196, 512)


def test_length_bucket_sampler_uses_caption_lengths_and_full_batches():
    dataset = SimpleNamespace(
        split="train",
        samples=[
            ("a.jpg", "short caption"),
            ("b.jpg", "short caption"),
            ("c.jpg", "a much longer caption"),
        ],
        vocab=Vocabulary(max_size=20),
    )
    dataset.vocab.build([caption for _, caption in dataset.samples])

    sampler = LengthBucketSampler(dataset, batch_size=2, drop_last=True)
    batches = list(iter(sampler))

    assert len(batches) == 1
    assert len(batches[0]) == 2
