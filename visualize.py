"""
Attention visualisation — recreates paper Figures 2 and 3.

For each generated word, the 196-dimensional attention weight vector α is:
  1. Reshaped to (14, 14).
  2. Upsampled by factor 16 to (224, 224) via bilinear interpolation.
  3. Smoothed with a Gaussian filter (σ = 8px).
  4. Overlaid as a semi-transparent heatmap on the original image.

Paper section 5.4:
  "We simply upsample the weights by a factor of 2^4 = 16 and apply a
   Gaussian filter."

TODO (Issue #12): Make figure export report-ready by allowing a higher DPI and
explicit `.png` / `.pdf` save path so collaborators can generate publication
quality artifacts without editing matplotlib internals.
TODO (Issue #12): Recreate the paper Figure 2 layout exactly: original image in
cell 0, one panel per generated word, consistent word titles, and no unused axes
left visible in the saved output.
"""

import math
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless default; override to "TkAgg" for interactive
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from models import Encoder, Decoder
from utils import Vocabulary

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

    TODO (Issue #12, deferred): If hard attention is implemented, accept sampled
    location indices and render them as discrete highlights instead of smooth α
    maps while preserving the current soft-attention API.
    TODO (Issue #12, deferred): If poster/report visuals need it, add an optional
    colorbar legend that can be toggled off to preserve the paper-style layout.
    TODO (Issue #12): Replace the current jet overlay with the paper-style
    white-highlight / grayscale attention rendering used in Figures 2 and 3 so
    saved examples look consistent with the reproduced method.
    """
    img_np = _unnormalize(image_tensor)   # (224, 224, 3)

    n_words = len(caption_words)
    # Grid: original image + one cell per word, wrap at 4 columns
    n_cols = 4
    n_rows = math.ceil((n_words + 1) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Cell 0: original image
    axes[0].imshow(img_np)
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")

    for t, word in enumerate(caption_words):
        ax = axes[t + 1]

        # Reshape α from (196,) to (14, 14)
        alpha_map = alphas[t].numpy().reshape(14, 14)

        # Upsample by 16 to (224, 224) — matches paper section 5.4
        # TODO (Issue #12): Replace `scipy.ndimage.zoom` with
        # `torch.nn.functional.interpolate(..., mode="bilinear")` so the
        # upsampling path matches the intended 14x14 -> 224x224 geometry exactly.
        from scipy.ndimage import zoom
        alpha_up = zoom(alpha_map, 16, order=1)   # bilinear zoom

        # Gaussian blur (σ ~ 8 pixels after upsampling)
        # TODO (Issue #12): Tune `sigma` against a few representative Flickr8k
        # examples until the blur strength visually matches the paper's figures,
        # then promote the chosen value to a named constant.
        alpha_smooth = gaussian_filter(alpha_up, sigma=8)

        # Overlay: show original image with attention heatmap on top
        ax.imshow(img_np)
        ax.imshow(alpha_smooth, alpha=0.5, cmap="jet",
                  extent=[0, 224, 224, 0])  # match image coordinates

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
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
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
    output_dir: str = "visualizations",
    beam_width: int = 1,
):
    """
    Load a checkpoint, generate captions, and save attention figures for each
    image in `image_paths`.

    Args:
        checkpoint_path: path to .pt checkpoint from train.py
        vocab_path:      path to vocab.json
        image_paths:     list of image file paths to caption & visualise
        output_dir:      directory to write PNG files
        beam_width:      1 = greedy (default)

    TODO (Issue #12): Accept either explicit image paths or a directory/glob so
    collaborators can render many qualitative examples without shell scripting.
    TODO (Issue #12, deferred): Add a `--grid` mode that composes several images
    into one multi-example figure for poster/report use, matching the spirit of
    paper Figure 3.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.load(vocab_path)
    ckpt  = torch.load(checkpoint_path, map_location=device)

    encoder = Encoder(fine_tune=False).to(device)
    decoder = Decoder(
        attention_dim=512,
        embed_dim=512,
        decoder_dim=512,
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

        # Encode
        encoder_out = encoder(img_batch)           # (1, 196, 512)
        h, c = decoder._init_hidden(encoder_out)

        word_idx  = torch.tensor([1], dtype=torch.long, device=device)
        words     = []
        alpha_list = []

        # TODO (Issue #12): Replace this duplicated decode loop with the shared
        # greedy-decode helper from evaluation so validation captions and saved
        # attention maps are generated by the exact same logic.
        for _ in range(50):
            embed  = decoder.embedding(word_idx)
            z_hat, alpha = decoder.attention(encoder_out, h)
            beta   = torch.sigmoid(decoder.f_beta(h))
            z_hat  = beta * z_hat

            lstm_input = torch.cat([embed, z_hat], dim=1)
            h, c = decoder.lstm_cell(lstm_input, (h, c))

            logits   = decoder.L_o(embed + decoder.L_h(h) + decoder.L_z(z_hat))
            word_idx = logits.argmax(dim=1)

            alpha_list.append(alpha.squeeze(0).cpu())  # (196,)
            idx = word_idx.item()
            if idx == 2:  # <end>
                break
            words.append(vocab.idx2word.get(idx, "<unk>"))

        alphas = torch.stack(alpha_list, dim=0)  # (steps, 196)
        caption = " ".join(words)
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
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # TODO (Issue #12, deferred): Add a `--show` flag for interactive notebook/
    # Colab use while keeping headless saving as the default CLI behavior.
    parser = argparse.ArgumentParser(description="Visualise attention maps")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt")
    parser.add_argument("--vocab",       default="data/flickr8k/vocab.json")
    parser.add_argument("--images",      nargs="+", required=True,
                        help="Paths to one or more images")
    parser.add_argument("--output_dir",  default="visualizations")
    args = parser.parse_args()

    run_visualization(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        image_paths=args.images,
        output_dir=args.output_dir,
    )
