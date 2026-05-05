"""
Smoke tests for all ablation configurations.

These tests exercise a single forward pass (and doubly-stochastic loss)
for every combination of ablation flags without touching real data or GPU.
They run in a few seconds and are designed to catch shape errors, NaN
outputs, and broken constructor arguments before any expensive training run.

Each test is parameterized with a fixture name that matches the experiment
names in scripts/run_ablations.py so you can target individual configs:

    pytest tests/test_ablations.py -k baseline_soft_attention -v
    pytest tests/test_ablations.py -k no_attention_mean -v
    pytest tests/test_ablations.py -v   # run all
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models import Decoder, Encoder
from models.decoder import VALID_ATTENTION_MODES
from train import doubly_stochastic_attention_loss


# ---------------------------------------------------------------------------
# Shared tiny dimensions (fast CPU-only forward passes)
# ---------------------------------------------------------------------------

TINY = SimpleNamespace(
    batch=2,
    vocab_size=20,
    embed_dim=32,
    decoder_dim=32,
    attention_dim=16,
    encoder_dim=512,   # real VGG output depth; kept real to catch projection bugs
    dropout=0.0,
)


# ---------------------------------------------------------------------------
# Dummy encoder output factory
# ---------------------------------------------------------------------------

def _fake_encoder_out(batch: int, L: int) -> torch.Tensor:
    """Return a random (batch, L, 512) annotation tensor."""
    return torch.randn(batch, L, TINY.encoder_dim)


def _fake_captions(batch: int, seq_len: int) -> tuple:
    """Return (captions, lengths) with START at position 0."""
    captions = torch.randint(1, TINY.vocab_size, (batch, seq_len))
    captions[:, 0] = 1          # START token
    lengths = torch.full((batch,), seq_len, dtype=torch.long)
    return captions, lengths


# ---------------------------------------------------------------------------
# Decoder factory
# ---------------------------------------------------------------------------

def _make_decoder(attention_mode: str = "soft", use_beta_gate: bool = True) -> Decoder:
    return Decoder(
        attention_dim=TINY.attention_dim,
        embed_dim=TINY.embed_dim,
        decoder_dim=TINY.decoder_dim,
        vocab_size=TINY.vocab_size,
        encoder_dim=TINY.encoder_dim,
        dropout=TINY.dropout,
        attention_mode=attention_mode,
        use_beta_gate=use_beta_gate,
    )


# ---------------------------------------------------------------------------
# Parameterized smoke test
# ---------------------------------------------------------------------------

# Each tuple: (pytest_id, attention_mode, use_beta_gate, L, lambda_weight)
SMOKE_CONFIGS = [
    ("baseline_soft_attention",       "soft",    True,  196, 1.0),
    ("no_attention_mean",             "none",    True,  196, 1.0),
    ("uniform_attention",             "uniform", True,  196, 1.0),
    ("no_doubly_stochastic_lambda_0", "soft",    True,  196, 0.0),
    ("no_beta_gate",                  "soft",    False, 196, 1.0),
    ("feature_grid_7x7",              "soft",    True,   49, 1.0),
]


@pytest.mark.parametrize(
    "name,attention_mode,use_beta_gate,L,lambda_weight",
    SMOKE_CONFIGS,
    ids=[c[0] for c in SMOKE_CONFIGS],
)
def test_smoke_forward(name, attention_mode, use_beta_gate, L, lambda_weight):
    """
    Single forward pass + doubly stochastic loss for each ablation config.

    Checks:
      - No crash / no exception during forward
      - predictions shape is (batch, seq_len-1, vocab_size)
      - alphas shape is (batch, seq_len-1, L)
      - No NaN in predictions or alphas
      - doubly_stochastic_attention_loss runs without error
      - loss value is finite
    """
    batch, seq_len = TINY.batch, 6

    decoder = _make_decoder(attention_mode, use_beta_gate)
    encoder_out = _fake_encoder_out(batch, L)
    captions, lengths = _fake_captions(batch, seq_len)

    predictions, alphas = decoder(encoder_out, captions, lengths)

    assert predictions.shape == (batch, seq_len - 1, TINY.vocab_size), (
        f"predictions shape mismatch: {predictions.shape}"
    )
    assert alphas.shape == (batch, seq_len - 1, L), (
        f"alphas shape mismatch: {alphas.shape}"
    )
    assert not torch.isnan(predictions).any(), "NaN in predictions"
    assert not torch.isnan(alphas).any(), "NaN in alphas"

    ds_loss = doubly_stochastic_attention_loss(alphas, weight=lambda_weight)
    assert torch.isfinite(ds_loss), f"doubly_stochastic loss is not finite: {ds_loss}"

    # When lambda=0, the loss must be exactly 0.0
    if lambda_weight == 0.0:
        assert ds_loss.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Alpha semantics per mode
# ---------------------------------------------------------------------------

def test_none_attention_returns_uniform_placeholder_alpha():
    """
    In 'none' mode alpha is a 1/L placeholder (no attention MLP called).
    The context vector equals the static mean annotation.
    """
    L = 196
    decoder = _make_decoder(attention_mode="none")
    encoder_out = _fake_encoder_out(1, L)
    h = torch.zeros(1, TINY.decoder_dim)
    c = torch.zeros(1, TINY.decoder_dim)
    prev = torch.tensor([1])

    _, _, _, alpha, _ = decoder.decode_step(encoder_out, prev, h, c)

    expected = torch.full((1, L), 1.0 / L)
    assert torch.allclose(alpha, expected, atol=1e-6)


def test_uniform_attention_forces_uniform_alpha():
    """
    In 'uniform' mode alpha is forced to 1/L even though the attention MLP exists.
    """
    L = 196
    decoder = _make_decoder(attention_mode="uniform")
    encoder_out = _fake_encoder_out(1, L)
    h = torch.zeros(1, TINY.decoder_dim)
    c = torch.zeros(1, TINY.decoder_dim)
    prev = torch.tensor([1])

    _, _, _, alpha, _ = decoder.decode_step(encoder_out, prev, h, c)

    expected = torch.full((1, L), 1.0 / L)
    assert torch.allclose(alpha, expected, atol=1e-6)


def test_no_beta_gate_returns_ones_beta():
    """With use_beta_gate=False, beta must be 1 for every step."""
    decoder = _make_decoder(attention_mode="soft", use_beta_gate=False)
    encoder_out = _fake_encoder_out(2, 196)
    h = torch.zeros(2, TINY.decoder_dim)
    c = torch.zeros(2, TINY.decoder_dim)
    prev = torch.tensor([1, 1])

    _, _, _, _, beta = decoder.decode_step(encoder_out, prev, h, c)

    assert beta.shape == (2, 1)
    assert torch.allclose(beta, torch.ones(2, 1), atol=1e-6), (
        f"beta should be all-ones when use_beta_gate=False, got {beta}"
    )


# ---------------------------------------------------------------------------
# Encoder grid size
# ---------------------------------------------------------------------------

def test_encoder_14x14_output_shape(monkeypatch):
    """Default encoder produces (B, 196, 512)."""
    _patch_vgg(monkeypatch, grid=14)
    encoder = Encoder(fine_tune=False, feature_grid_size=14)
    out = encoder(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 196, 512), f"Expected (2,196,512), got {out.shape}"


def test_encoder_7x7_output_shape(monkeypatch):
    """7x7 ablation encoder produces (B, 49, 512)."""
    _patch_vgg(monkeypatch, grid=14)          # VGG still outputs 14×14
    encoder = Encoder(fine_tune=False, feature_grid_size=7)
    out = encoder(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 49, 512), f"Expected (2,49,512), got {out.shape}"


def test_encoder_rejects_invalid_grid_size(monkeypatch):
    _patch_vgg(monkeypatch, grid=14)
    with pytest.raises(ValueError, match="feature_grid_size"):
        Encoder(fine_tune=False, feature_grid_size=8)


def test_decoder_rejects_invalid_attention_mode():
    with pytest.raises(ValueError, match="attention_mode"):
        _make_decoder(attention_mode="bad_mode")


# ---------------------------------------------------------------------------
# End-to-end: 7×7 encoder → decoder forward pass
# ---------------------------------------------------------------------------

def test_feature_grid_7x7_end_to_end(monkeypatch):
    """7×7 encoder output feeds a decoder with L=49 without shape errors."""
    _patch_vgg(monkeypatch, grid=14)
    encoder = Encoder(fine_tune=False, feature_grid_size=7)
    decoder = _make_decoder()

    images = torch.randn(2, 3, 224, 224)
    encoder_out = encoder(images)        # (2, 49, 512)
    assert encoder_out.shape == (2, 49, 512)

    captions, lengths = _fake_captions(2, 6)
    predictions, alphas = decoder(encoder_out, captions, lengths)

    assert predictions.shape == (2, 5, TINY.vocab_size)
    assert alphas.shape == (2, 5, 49)
    assert not torch.isnan(predictions).any()


# ---------------------------------------------------------------------------
# Visualize: dynamic grid inference
# ---------------------------------------------------------------------------

def test_visualize_alpha_reshape_is_grid_size_agnostic():
    """
    visualize_attention must handle both L=196 (grid=14) and L=49 (grid=7)
    without hardcoding 14×14.  We only test the reshape logic here, not
    the full plot rendering.
    """
    import torch.nn.functional as F

    for L, expected_grid in [(196, 14), (49, 7)]:
        alpha_row = torch.full((L,), 1.0 / L)
        grid_size = int(round(L ** 0.5))
        assert grid_size == expected_grid

        alpha_map = alpha_row.view(1, 1, grid_size, grid_size)
        alpha_up = F.interpolate(alpha_map, size=(224, 224),
                                 mode="bilinear", align_corners=False)
        assert alpha_up.shape == (1, 1, 224, 224)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _patch_vgg(monkeypatch, grid: int):
    """Monkeypatch torchvision.models.vgg16 to return a fixed-shape dummy."""
    import torchvision.models as tv_models

    class DummyFeatureMap(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), 512, grid, grid)

    dummy_vgg = SimpleNamespace(features=nn.Sequential(DummyFeatureMap()))
    monkeypatch.setattr(tv_models, "vgg16", lambda weights: dummy_vgg)
