"""Evaluate a trained soft-attention captioning checkpoint on Flickr8k."""

import argparse
import json
import os

import nltk
import torch
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from tqdm import tqdm

from config import (
    ATTENTION_DIM,
    ATTENTION_MODE,
    DECODER_DIM,
    EMBED_DIM,
    FEATURE_GRID_SIZE,
    MAX_DECODE_LEN,
    USE_BETA_GATE,
)
from models import Decoder, Encoder
from utils import (
    Vocabulary,
    beam_search_decode,
    compute_bleu,
    get_dataloader,
    greedy_decode,
    print_bleu_table,
    tokenize_caption,
)


def _warn_config_mismatch(ckpt: dict, args: argparse.Namespace) -> None:
    """Print a warning if checkpoint ablation flags differ from CLI args."""
    mismatches = []
    for key, cli_val in [
        ("attention_mode", args.attention_mode),
        ("use_beta_gate", not args.no_beta_gate),
        ("feature_grid_size", args.feature_grid_size),
    ]:
        ckpt_val = ckpt.get(key)
        if ckpt_val is not None and ckpt_val != cli_val:
            mismatches.append(f"  {key}: checkpoint={ckpt_val!r}, cli={cli_val!r}")
    if mismatches:
        print("WARNING: checkpoint ablation flags differ from CLI arguments:")
        for m in mismatches:
            print(m)
        print("Using CLI values. Pass matching flags to avoid silent mismatch.")


@torch.no_grad()
def evaluate_test_set(
    checkpoint_path: str,
    data_root: str,
    vocab_path: str,
    split: str = "test",
    beam_width: int = 1,
    length_normalize: bool = False,
    batch_size: int = 1,
    results_out: str = "results/test_bleu.json",
    attention_mode: str = ATTENTION_MODE,
    use_beta_gate: bool = USE_BETA_GATE,
    feature_grid_size: int = FEATURE_GRID_SIZE,
):
    """Load a checkpoint, generate captions, print BLEU-1..4 + METEOR, save JSON."""
    nltk.download("wordnet", quiet=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading vocabulary from {vocab_path}...")
    vocab = Vocabulary.load(vocab_path)

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

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

    dataloader = get_dataloader(
        data_root,
        vocab,
        split,
        batch_size=batch_size,
        use_bucket_sampler=False,
    )

    hypotheses = []
    references = []

    for images, _, _, all_caps in tqdm(dataloader, desc=f"Evaluating [{split}]"):
        for i in range(images.size(0)):
            image = images[i : i + 1].to(device)
            if beam_width == 1:
                caption, _, _ = greedy_decode(
                    encoder, decoder, image, vocab, device, max_len=MAX_DECODE_LEN
                )
            else:
                caption = beam_search_decode(
                    encoder,
                    decoder,
                    image,
                    vocab,
                    device,
                    beam_width=beam_width,
                    max_len=MAX_DECODE_LEN,
                    length_normalize=length_normalize,
                )

            hypotheses.append(caption.split())
            references.append([tokenize_caption(c) for c in all_caps[i]])

    scores = compute_bleu(hypotheses, references)

    meteor = sum(
        nltk_meteor(refs, hyp) for hyp, refs in zip(hypotheses, references)
    ) / max(len(hypotheses), 1)
    scores["meteor"] = meteor

    model_label = (
        f"attention={attention_mode}, beta={use_beta_gate}, "
        f"grid={feature_grid_size}, beam={beam_width}"
    )
    print_bleu_table(model_label, scores, dataset=split.capitalize())
    print(f"  METEOR (sentence-level avg): {meteor * 100:.2f}")

    if results_out:
        os.makedirs(os.path.dirname(results_out) or ".", exist_ok=True)
        with open(results_out, "w") as f:
            json.dump(
                {
                    "checkpoint": checkpoint_path,
                    "split": split,
                    "beam_width": beam_width,
                    "length_normalize": length_normalize,
                    "attention_mode": attention_mode,
                    "use_beta_gate": use_beta_gate,
                    "feature_grid_size": feature_grid_size,
                    "scores": scores,
                },
                f,
                indent=2,
            )
        print(f"Saved results to {results_out}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Show, Attend and Tell (+ ablation variants)"
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data_root", default="data/flickr8k")
    parser.add_argument("--vocab", default="data/flickr8k/vocab.json")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--length_normalize", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--results_out", default="results/test_bleu.json")
    # --- Ablation flags (must match the flags used during training) ---
    parser.add_argument(
        "--attention_mode",
        choices=["soft", "none"],
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

    # Warn if checkpoint flags don't match
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    _warn_config_mismatch(ckpt, args)

    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        vocab_path=args.vocab,
        split=args.split,
        beam_width=args.beam_width,
        length_normalize=args.length_normalize,
        batch_size=args.batch_size,
        results_out=args.results_out,
        attention_mode=args.attention_mode,
        use_beta_gate=not args.no_beta_gate,
        feature_grid_size=args.feature_grid_size,
    )
