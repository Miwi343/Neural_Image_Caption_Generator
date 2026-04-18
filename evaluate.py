"""
Evaluation script: greedy decode, beam search, and full BLEU table.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --split test

TODO (Issue #10, deferred): If report scope expands beyond the proposal metric,
add METEOR alongside BLEU while preserving the current BLEU-first output.
TODO (Issue #11, deferred): If search-quality experiments are assigned, compare
beam search against greedy decoding on the same checkpoint and log both BLEU and
qualitative differences.
TODO (Issue #11, deferred): If final-report automation is needed, add a runner
that evaluates train/val/test in one pass and writes a combined results table.
"""

import argparse
import os

import torch
from tqdm import tqdm

from models import Encoder, Decoder
from utils import Vocabulary, get_dataloader, compute_bleu, print_bleu_table

# ---------------------------------------------------------------------------
# Greedy decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(encoder, decoder, image, vocab, device, max_len=50):
    """
    Greedy (argmax) decode for a single image.

    Args:
        encoder:  Encoder (eval mode)
        decoder:  Decoder (eval mode)
        image:    (1, 3, 224, 224) float tensor on `device`
        vocab:    Vocabulary
        device:   torch.device
        max_len:  maximum caption length

    Returns:
        caption:  str — generated caption
        alphas:   (steps, 196) tensor — attention maps for visualisation

    TODO (Issue #11): Refactor this into the shared greedy-decode helper used by
    validation and visualization so all proposal-scope caption generation follows
    one code path and returns identical token sequences.
    TODO (Issue #11, deferred): If batch evaluation becomes necessary, add a
    batched greedy path and optionally return token IDs alongside the decoded
    string for downstream analysis.
    """
    encoder.eval()
    decoder.eval()

    encoder_out = encoder(image)   # (1, 196, 512)
    h, c = decoder._init_hidden(encoder_out)

    word_idx = torch.tensor([1], dtype=torch.long, device=device)  # <start>
    generated_ids = []
    attention_maps = []

    for _ in range(max_len):
        embed = decoder.embedding(word_idx)          # (1, embed_dim)
        z_hat, alpha = decoder.attention(encoder_out, h)  # alpha: (1, 196)
        beta  = torch.sigmoid(decoder.f_beta(h))
        z_hat = beta * z_hat

        lstm_input = torch.cat([embed, z_hat], dim=1)
        h, c = decoder.lstm_cell(lstm_input, (h, c))

        logits = decoder.L_o(embed + decoder.L_h(h) + decoder.L_z(z_hat))
        word_idx = logits.argmax(dim=1)  # (1,)

        attention_maps.append(alpha.squeeze(0).cpu())  # (196,)
        idx = word_idx.item()

        if idx == 2:  # <end>
            break
        generated_ids.append(idx)

    caption = vocab.decode(generated_ids)
    alphas  = torch.stack(attention_maps, dim=0)  # (steps, 196)
    return caption, alphas


# ---------------------------------------------------------------------------
# Beam search decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search_decode(encoder, decoder, image, vocab, device, beam_width=3, max_len=50):
    """
    Beam search decode for a single image.

    At each step the `beam_width` most probable partial sequences are extended.
    Completed sequences (ending in <end>) are collected and the highest-scoring
    one is returned.

    Args:
        encoder:    Encoder (eval mode)
        decoder:    Decoder (eval mode)
        image:      (1, 3, 224, 224) float tensor
        vocab:      Vocabulary
        device:     torch.device
        beam_width: number of beams  (paper does not use beam search; greedy
                    is the default for BLEU evaluation)
        max_len:    maximum length per beam

    Returns:
        best_caption: str

    TODO (Issue #11, deferred): Replace the Python-loop beam expansion with a
    batched implementation if beam search becomes part of the evaluation story.
    TODO (Issue #11, deferred): Preserve and return attention maps for the best
    beam if qualitative beam-search visualizations are needed.
    TODO (Issue #11, deferred): Add a configurable length-normalization penalty
    before comparing completed beams so short captions are not over-favored.
    """
    encoder.eval()
    decoder.eval()

    encoder_out = encoder(image)  # (1, 196, 512)
    h, c = decoder._init_hidden(encoder_out)

    # Each beam: (score, token_ids, h, c)
    # TODO (Issue #11, deferred): If batched beam search is implemented, expand
    # `encoder_out` to `(beam_width, 196, 512)` once here and keep beam state
    # tensors aligned with that repeated encoder batch.
    beams = [(0.0, [1], h, c)]  # start with <start>
    completed = []

    for _ in range(max_len):
        new_beams = []
        for score, ids, bh, bc in beams:
            last_token = torch.tensor([ids[-1]], dtype=torch.long, device=device)
            embed  = decoder.embedding(last_token)
            z_hat, _ = decoder.attention(encoder_out, bh)
            beta   = torch.sigmoid(decoder.f_beta(bh))
            z_hat  = beta * z_hat

            lstm_input = torch.cat([embed, z_hat], dim=1)
            nh, nc = decoder.lstm_cell(lstm_input, (bh, bc))

            logits = decoder.L_o(embed + decoder.L_h(nh) + decoder.L_z(z_hat))
            log_probs = torch.log_softmax(logits, dim=1)  # (1, vocab_size)

            # Expand top-k
            topk_probs, topk_ids = log_probs.topk(beam_width, dim=1)
            for k in range(beam_width):
                new_score = score + topk_probs[0, k].item()
                new_ids   = ids + [topk_ids[0, k].item()]
                if topk_ids[0, k].item() == 2:  # <end>
                    completed.append((new_score, new_ids))
                else:
                    new_beams.append((new_score, new_ids, nh, nc))

        # Keep top beam_width active beams
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
        if not beams:
            break

    if not completed:
        # If no beam completed, take the best active beam
        # TODO (Issue #11, deferred): Apply the same length-normalization rule
        # here that is used for completed beams so the fallback path is comparable.
        completed = [(b[0], b[1]) for b in beams]

    best_score, best_ids = max(completed, key=lambda x: x[0])
    return vocab.decode(best_ids[1:])  # strip <start>


# ---------------------------------------------------------------------------
# Full test-set evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test_set(
    checkpoint_path: str,
    data_root: str,
    vocab_path: str,
    split: str = "test",
    beam_width: int = 1,
    batch_size: int = 1,
):
    """
    Load a checkpoint, generate captions on the test split, and print the
    BLEU table matching Table 1 of the paper.

    Args:
        checkpoint_path: path to .pt file saved by train.py
        data_root:       path to data/flickr8k/
        vocab_path:      path to vocab.json
        split:           "val" or "test"
        beam_width:      1 = greedy, >1 = beam search
        batch_size:      images per batch (beam search is currently 1 at a time)

    TODO (Issue #10, deferred): If METEOR is added, keep the returned score dict
    backward-compatible for callers that currently expect BLEU keys.
    TODO (Issue #11, deferred): If test-time runtime matters, support batched
    greedy decode while verifying captions match the existing one-image path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading vocabulary from {vocab_path}...")
    vocab = Vocabulary.load(vocab_path)

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    encoder = Encoder(fine_tune=False).to(device)
    decoder = Decoder(
        attention_dim=512,
        embed_dim=512,
        decoder_dim=512,
        vocab_size=len(vocab),
        dropout=0.0,   # no dropout at test time
    ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    dataloader = get_dataloader(
        data_root, vocab, split, batch_size=1, use_bucket_sampler=False
    )

    hypotheses = []
    references = []

    decode_fn = greedy_decode if beam_width == 1 else (
        lambda enc, dec, img, v, d: (beam_search_decode(enc, dec, img, v, d, beam_width), None)
    )

    # TODO (Issue #11, deferred): Replace `@torch.no_grad()` with
    # `@torch.inference_mode()` once you have confirmed none of this path relies
    # on autograd-compatible tensor metadata.
    for images, _, _, all_caps in tqdm(dataloader, desc=f"Evaluating [{split}]"):
        image = images[0:1].to(device)  # process one image at a time
        caption, _ = decode_fn(encoder, decoder, image, vocab, device)
        hypotheses.append(caption.split())

        # All reference captions for this image
        refs = [c.lower().split() for c in all_caps[0]]
        references.append(refs)

    scores = compute_bleu(hypotheses, references)
    model_label = f"Our Soft-Attention (beam={beam_width})"
    print_bleu_table(model_label, scores, dataset=split.capitalize())

    return scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # TODO (Issue #11, deferred): If this becomes the report-generation entry
    # point, add a `--split all` mode and a `--results_out` path for writing a
    # combined table under `results/`.
    parser = argparse.ArgumentParser(description="Evaluate Show, Attend and Tell")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data_root",  default="data/flickr8k")
    parser.add_argument("--vocab",      default="data/flickr8k/vocab.json")
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--beam_width", type=int, default=1)
    args = parser.parse_args()

    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        vocab_path=args.vocab,
        split=args.split,
        beam_width=args.beam_width,
    )
