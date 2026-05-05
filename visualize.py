"""Attention visualisation for generated captions.

For each generated word, the L-dimensional attention weight vector α is:
  1. Reshaped to (grid, grid) — where grid = sqrt(L), so 14 or 7.
  2. Upsampled to (224, 224) via bilinear interpolation.
  3. Smoothed with a Gaussian filter (σ = 8px).
  4. Overlaid as a semi-transparent heatmap on the original image.

Paper section 5.4:
  "We simply upsample the weights by a factor of 2^4 = 16 and apply a
   Gaussian filter."  (This refers to the 14×14 default; upsampling to 224
   works identically for 7×7 via bilinear interpolation with a fixed target.)

Ablation notes:
  - none/uniform attention: the returned alpha is uniform (1/L).
    Visualization shows a flat, featureless overlay — this is expected and
    correct behaviour (either no spatial attention was used, or α was forced
    to be uniform).
  - feature_grid_size=7: L=49, grid=7.  The upsample target stays 224×224.
  - All other ablation modes (no_beta_gate, lambda=0) produce normal alpha
    maps and work with this script without any special handling.
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

from config import (
    ATTENTION_DIM,
    ATTENTION_MODE,
    DECODER_DIM,
    EMBED_DIM,
    FEATURE_GRID_SIZE,
    MAX_DECODE_LEN,
    USE_BETA_GATE,
)
from models import Encoder, Decoder
from utils import Vocabulary, greedy_decode, load_flickr8k_captions, resolve_images_dir

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
    alphas: torch.Tensor,           # (num_steps, L) — L=196 or 49
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "",
    dpi: int = 200,
    sigma: float = 8.0,
    overlay_style: str = "paper",
    attention_mode: str = "soft",
) -> None:
    """
    Plot the word-by-word attention overlay, matching paper Figures 2 & 3.

    Args:
        image_tensor:   (3, 224, 224) — ImageNet-normalised input image
        caption_words:  list of generated words (excluding <start>/<end>)
        alphas:         (num_steps, L) — one row per generated word.
                        L=196 for the 14×14 grid, L=49 for 7×7.
        save_path:      if provided, save figure to this path (PNG/PDF)
        show:           if True, call plt.show() (requires a display)
        title:          optional figure suptitle
        dpi:            output DPI
        sigma:          Gaussian blur strength after upsampling
        overlay_style:  "paper" for grayscale/white highlight or "heatmap"
        attention_mode: passed through for the subtitle note on uniform/none maps
    """
    if overlay_style not in {"paper", "heatmap"}:
        raise ValueError("overlay_style must be 'paper' or 'heatmap'.")

    img_np = _unnormalize(image_tensor)   # (224, 224, 3)

    L = alphas.shape[-1]
    grid_size = int(round(L ** 0.5))     # 14 for L=196, 7 for L=49

    n_words = len(caption_words)
    n_cols = 4
    n_rows = max(1, math.ceil((n_words + 1) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Cell 0: original image
    axes[0].imshow(img_np)
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")

    # Uniform / none modes produce flat maps — annotate the figure.
    if attention_mode in {"none", "uniform"}:
        flat_note = f"[{attention_mode} — uniform α shown]"
        fig.text(0.5, 0.01, flat_note, ha="center", fontsize=9, color="gray")

    for t, word in enumerate(caption_words):
        ax = axes[t + 1]

        # Reshape α from (L,) to (grid, grid), then upsample to (224, 224).
        alpha_map = alphas[t].view(1, 1, grid_size, grid_size)
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
    attention_mode: str = ATTENTION_MODE,
    use_beta_gate: bool = USE_BETA_GATE,
    feature_grid_size: int = FEATURE_GRID_SIZE,
):
    """
    Load a checkpoint, generate captions, and save attention figures.

    For attention_mode="none" the saved figures will show uniform/flat overlays
    (the model did not use spatial attention during decoding).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.load(vocab_path)
    ckpt  = torch.load(checkpoint_path, map_location=device)

    encoder = Encoder(
        fine_tune=False,
        feature_grid_size=feature_grid_size,
    ).to(device)

    decoder = Decoder(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
        dropout=0.0,
        attention_mode=attention_mode,
        use_beta_gate=use_beta_gate,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        print(f"Processing: {img_path}")

        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = _EVAL_TRANSFORM(pil_img)
        img_batch  = img_tensor.unsqueeze(0).to(device)

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
            attention_mode=attention_mode,
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
    parser.add_argument("--checkpoint",    default="checkpoints/best.pt")
    parser.add_argument("--vocab",         default="data/flickr8k/vocab.json")
    parser.add_argument("--images",        nargs="+",
                        help="Paths to one or more images")
    parser.add_argument("--data_root",     default="data/flickr8k")
    parser.add_argument("--split",         default="test", choices=["val", "test"])
    parser.add_argument("--num_examples",  type=int, default=6)
    parser.add_argument("--output_dir",    default="results/attention_examples")
    parser.add_argument("--dpi",           type=int, default=200)
    parser.add_argument("--sigma",         type=float, default=8.0)
    parser.add_argument("--overlay_style", default="paper", choices=["paper", "heatmap"])
    # --- Ablation flags (must match the flags used during training) ---
    parser.add_argument(
        "--attention_mode",
        choices=["soft", "uniform", "none"],
        default=ATTENTION_MODE,
        help="Must match the mode used when training the checkpoint.",
    )
    parser.add_argument(
        "--no_beta_gate",
        action="store_true",
        default=False,
        help="Disable beta gate (must match training setting).",
    )
    parser.add_argument(
        "--feature_grid_size",
        type=int,
        choices=[7, 14],
        default=FEATURE_GRID_SIZE,
        help="Feature grid size (must match training setting).",
    )
    args = parser.parse_args()

    image_paths = expand_image_inputs(args.images)
    if not image_paths:
        image_to_caps = load_flickr8k_captions(args.data_root, args.split)
        images_dir = resolve_images_dir(args.data_root)
        image_paths = [
            os.path.join(images_dir, img_name)
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
        attention_mode=args.attention_mode,
        use_beta_gate=not args.no_beta_gate,
        feature_grid_size=args.feature_grid_size,
    )
