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
from utils import Vocabulary, greedy_decode, load_flickr8k_captions

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
    image_tensor: torch.Tensor,     # (3, 224, 224) normalised
    caption_words: List[str],
    alphas: torch.Tensor,           # (num_steps, 196)
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

        ax.set_title(word, fontsize=10)
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

    encoder = Encoder(fine_tune=False).to(device)
    decoder = Decoder(
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

        caption, alphas, token_ids = greedy_decode(
            encoder,
            decoder,
            img_batch,
            vocab,
            device,
            max_len=MAX_DECODE_LEN,
        )
        words = [vocab.idx2word.get(idx, "<unk>") for idx in token_ids]
        print(f"  Caption: {caption}")

        img_name  = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{img_name}_attention.png")
        visualize_attention(
            img_tensor.cpu(),
            words,
            alphas,
            save_path=save_path,
            show=False,
            title=caption,
            dpi=dpi,
            sigma=sigma,
            overlay_style=overlay_style,
        )


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
    )
