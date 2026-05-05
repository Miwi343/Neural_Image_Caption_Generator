"""Attention visualisation for generated captions.

For each generated word, the 196-dimensional attention weight vector α is:
  1. Reshaped to (14, 14).
  2. Upsampled by factor 16 to (224, 224) via bilinear interpolation.
  3. Smoothed with a Gaussian filter (σ = 8px).
  4. Overlaid as a semi-transparent heatmap on the original image.

Paper section 5.4:
  "We simply upsample the weights by a factor of 2^4 = 16 and apply a
   Gaussian filter."

"""

import glob
import json
import math
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")   # headless default; override to "TkAgg" for interactive
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from config import ATTENTION_DIM, DECODER_DIM, EMBED_DIM, MAX_DECODE_LEN
from models import Encoder, Decoder
from models.adaptive_decoder import AdaptiveDecoder
from utils import (
    Vocabulary,
    greedy_decode,
    greedy_decode_from_encoder_out_adaptive,
    load_flickr8k_captions,
)

# ---------------------------------------------------------------------------
# ImageNet inverse transform for display
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN.tolist(), _IMAGENET_STD.tolist()),
])


def _unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) normalised tensor to a (H, W, 3) uint8 array."""
    img = tensor.permute(1, 2, 0).numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Core visualisation
# ---------------------------------------------------------------------------

def visualize_attention(
    image_tensor: torch.Tensor,                   # (3, 224, 224) normalised
    caption_words: List[str],
    alphas: torch.Tensor,                         # (num_steps, 196)
    betas: Optional[torch.Tensor] = None,         # (num_steps,) sentinel weights, or None
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "",
    dpi: int = 200,
    sigma: float = 8.0,
    overlay_style: str = "paper",
) -> None:
    """
    Plot the word-by-word attention overlay, matching paper Figures 2 & 3.

    Args:
        image_tensor:  (3, 224, 224) — ImageNet-normalised input image
        caption_words: list of generated words (excluding <start>/<end>)
        alphas:        (num_steps, 196) — one row per generated word
        betas:         (num_steps,) sentinel weights from AdaptiveDecoder, or None.
                       When provided, each subplot title shows "word\nβ=0.xx" so
                       you can see which words relied on the sentinel vs. the image.
        save_path:     if provided, save figure to this path (PNG/PDF)
        show:          if True, call plt.show() (requires a display)
        title:         optional figure suptitle
        dpi:           output DPI
        sigma:         Gaussian blur strength after upsampling
        overlay_style: "paper" for grayscale/white highlight or "heatmap"
    """
    if overlay_style not in {"paper", "heatmap"}:
        raise ValueError("overlay_style must be 'paper' or 'heatmap'.")

    img_np = _unnormalize(image_tensor)   # (224, 224, 3)

    n_words = len(caption_words)
    # Grid: original image + one cell per word, wrap at 4 columns
    n_cols = 4
    n_rows = max(1, math.ceil((n_words + 1) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Cell 0: original image
    axes[0].imshow(img_np)
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")

    for t, word in enumerate(caption_words):
        ax = axes[t + 1]

        # Reshape α from (196,) to (14, 14), then upsample by 16 to (224, 224).
        alpha_map = alphas[t].view(1, 1, 14, 14)
        alpha_up = F.interpolate(
            alpha_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        alpha_smooth = gaussian_filter(alpha_up, sigma=sigma)
        max_alpha = float(alpha_smooth.max())
        if max_alpha > 0:
            alpha_smooth = alpha_smooth / max_alpha

        if overlay_style == "paper":
            gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            ax.imshow(
                alpha_smooth,
                alpha=np.clip(alpha_smooth * 0.85, 0.0, 0.85),
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=[0, 224, 224, 0],
            )
        else:
            ax.imshow(img_np)
            ax.imshow(
                alpha_smooth,
                alpha=0.5,
                cmap="jet",
                extent=[0, 224, 224, 0],
            )

        word_title = word if betas is None else f"{word}\nβ={float(betas[t]):.2f}"
        ax.set_title(word_title, fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[n_words + 1:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Saved attention figure → {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Sentence-level β visualization
# ---------------------------------------------------------------------------

def visualize_sentence_beta(
    image_tensor: torch.Tensor,         # (3, 224, 224) normalised
    caption_words: List[str],
    betas: torch.Tensor,                # (num_steps,)
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 150,
    words_per_row: int = 8,
) -> None:
    """
    Poster-friendly two-panel figure: image left, colored sentence right.

    Each word is rendered as a chip colored by its β value via the coolwarm
    colormap (blue = low β, model looked at image; red = high β, model relied
    on the language-model sentinel). The β value is printed below each chip.
    """
    img_np    = _unnormalize(image_tensor)
    betas_arr = np.array([float(b) for b in betas])
    n         = len(caption_words)

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    # Pick the words_per_row (between 5 and the caller's max) that wastes the
    # fewest empty slots on the last row, keeping captions compact.
    best_wpr, best_waste = words_per_row, words_per_row
    for wpr in range(5, words_per_row + 1):
        waste = (-n) % wpr
        if waste < best_waste:
            best_waste, best_wpr = waste, wpr
    words_per_row = best_wpr

    n_text_rows = math.ceil(n / words_per_row)
    fig_h = max(3.5, 1.8 + n_text_rows * 1.4)
    fig_w = max(10, 1.6 * words_per_row + 4)   # scale width with columns

    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h * 1.3),
        gridspec_kw={"width_ratios": [0.65, 2.2]},
    )

    # ── Left: image ──────────────────────────────────────────────────────────
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Input Image", fontsize=14, pad=6)

    # ── Right: word chips ────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, words_per_row)
    ax.set_ylim(-(n_text_rows - 0.3), 1.0)
    ax.axis("off")
    ax.set_title(
        "Generated Caption  —  blue = visual attention  |  red = sentinel (language model)",
        fontsize=12, pad=10,
    )

    for idx, (word, beta) in enumerate(zip(caption_words, betas_arr)):
        col = idx % words_per_row
        row = idx // words_per_row
        x   = col + 0.5
        y   = -row

        bg_color = cmap(norm(beta))
        # Pick text color by perceived luminance of background
        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        text_color = "white" if lum < 0.50 else "black"

        ax.text(
            x, y + 0.15, word,
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_color,
                      edgecolor="none", alpha=0.92),
            zorder=3,
        )
        ax.text(
            x, y - 0.30, f"β={beta:.2f}",
            ha="center", va="center", fontsize=10, color="#555555",
        )

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("β  (sentinel weight)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved sentence figure → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _save_caption_json(
    img_name: str,
    caption_words: List[str],
    betas: torch.Tensor,
    lam: float,
    output_dir: str,
) -> None:
    """Save caption + per-word β values as JSON for cross-lambda comparison."""
    data = {
        "image":   img_name,
        "lam":     lam,
        "caption": caption_words,
        "betas":   [round(float(b), 4) for b in betas],
    }
    path = os.path.join(output_dir, f"{img_name}_caption.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_visualization(
    checkpoint_path: str,
    vocab_path: str,
    image_paths: List[str],
    output_dir: str = "results/attention_examples",
    dpi: int = 200,
    sigma: float = 8.0,
    overlay_style: str = "paper",
    also_sentence: bool = False,
    save_json: bool = False,
    lam: float = 1.0,
):
    """
    Load a checkpoint, generate captions, and save attention figures for each
    image in `image_paths`.

    Args:
        checkpoint_path: path to .pt checkpoint from train.py
        vocab_path:      path to vocab.json
        image_paths:     list of image file paths to caption & visualise
        output_dir:      directory to write PNG files
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.load(vocab_path)
    ckpt  = torch.load(checkpoint_path, map_location=device)

    # Auto-detect decoder class from checkpoint
    model_type   = ckpt.get("model_type", "base")
    is_adaptive  = (model_type == "adaptive")
    DecoderClass = AdaptiveDecoder if is_adaptive else Decoder
    if is_adaptive:
        print("Detected adaptive-attention checkpoint — sentinel weights (β) will be shown.")

    encoder = Encoder(fine_tune=False).to(device)
    decoder = DecoderClass(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
        dropout=0.0,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        print(f"Processing: {img_path}")

        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = _EVAL_TRANSFORM(pil_img)           # (3, 224, 224) normalised
        img_batch  = img_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)

        # Use the adaptive decode path when the model has a sentinel
        betas = None
        if is_adaptive:
            with torch.no_grad():
                encoder_out = encoder(img_batch)
            caption, alphas, betas, token_ids = greedy_decode_from_encoder_out_adaptive(
                decoder, encoder_out, vocab, device, max_len=MAX_DECODE_LEN,
            )
        else:
            caption, alphas, token_ids = greedy_decode(
                encoder, decoder, img_batch, vocab, device, max_len=MAX_DECODE_LEN,
            )

        words = [vocab.idx2word.get(idx, "<unk>") for idx in token_ids]
        print(f"  Caption: {caption}")

        img_name  = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{img_name}_attention.png")
        visualize_attention(
            img_tensor.cpu(),
            words,
            alphas,
            betas=betas,
            save_path=save_path,
            show=False,
            title=caption,
            dpi=dpi,
            sigma=sigma,
            overlay_style=overlay_style,
        )

        if also_sentence and betas is not None:
            sent_path = os.path.join(output_dir, f"{img_name}_sentence.png")
            visualize_sentence_beta(
                img_tensor.cpu(), words, betas,
                save_path=sent_path, show=False, dpi=dpi,
            )

        if save_json and betas is not None:
            _save_caption_json(img_name, words, betas, lam, output_dir)


def expand_image_inputs(inputs: Optional[List[str]]) -> List[str]:
    """Expand explicit image paths, directories, and glob patterns."""
    if not inputs:
        return []

    image_paths: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                image_paths.extend(glob.glob(os.path.join(item, ext)))
        else:
            matches = glob.glob(item)
            image_paths.extend(matches or [item])

    return sorted(dict.fromkeys(image_paths))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise attention maps")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt")
    parser.add_argument("--vocab",       default="data/flickr8k/vocab.json")
    parser.add_argument("--images",      nargs="+",
                        help="Paths to one or more images")
    parser.add_argument("--data_root",   default="data/flickr8k")
    parser.add_argument("--split",       default="test", choices=["val", "test"])
    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--output_dir",  default="results/attention_examples")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--overlay_style", default="paper", choices=["paper", "heatmap"])
    parser.add_argument("--also_sentence", action="store_true",
                        help="Also save a sentence-level β coloring figure per image")
    parser.add_argument("--save_json", action="store_true",
                        help="Save caption + β values as JSON (for cross-lambda comparison)")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Lambda value used for training (stored in JSON metadata)")
    args = parser.parse_args()

    image_paths = expand_image_inputs(args.images)
    if not image_paths:
        image_to_caps = load_flickr8k_captions(args.data_root, args.split)
        image_paths = [
            os.path.join(args.data_root, "images", img_name)
            for img_name in sorted(image_to_caps.keys())[: args.num_examples]
        ]

    run_visualization(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        image_paths=image_paths,
        output_dir=args.output_dir,
        dpi=args.dpi,
        sigma=args.sigma,
        overlay_style=args.overlay_style,
        also_sentence=args.also_sentence,
        save_json=args.save_json,
        lam=args.lam,
    )
