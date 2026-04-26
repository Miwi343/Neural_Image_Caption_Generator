"""Evaluate a trained soft-attention captioning checkpoint on Flickr8k."""

import argparse
import json
import os

import torch
from tqdm import tqdm

from config import ATTENTION_DIM, DECODER_DIM, EMBED_DIM, MAX_DECODE_LEN
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


@torch.no_grad()
def evaluate_test_set(
    checkpoint_path: str,
    data_root: str,
    vocab_path: str,
    split: str = "test",
    beam_width: int = 1,
    batch_size: int = 1,
    results_out: str = "results/test_bleu.json",
):
    """Load a checkpoint, generate captions, print BLEU-1..4, and save JSON."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading vocabulary from {vocab_path}...")
    vocab = Vocabulary.load(vocab_path)

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

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
                )

            hypotheses.append(caption.split())
            references.append([tokenize_caption(c) for c in all_caps[i]])

    scores = compute_bleu(hypotheses, references)
    model_label = f"Our Soft-Attention (beam={beam_width})"
    print_bleu_table(model_label, scores, dataset=split.capitalize())

    if results_out:
        os.makedirs(os.path.dirname(results_out) or ".", exist_ok=True)
        with open(results_out, "w") as f:
            json.dump(
                {
                    "checkpoint": checkpoint_path,
                    "split": split,
                    "beam_width": beam_width,
                    "scores": scores,
                },
                f,
                indent=2,
            )
        print(f"Saved BLEU results to {results_out}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Show, Attend and Tell")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data_root", default="data/flickr8k")
    parser.add_argument("--vocab", default="data/flickr8k/vocab.json")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--results_out", default="results/test_bleu.json")
    args = parser.parse_args()

    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        vocab_path=args.vocab,
        split=args.split,
        beam_width=args.beam_width,
        batch_size=args.batch_size,
        results_out=args.results_out,
    )
