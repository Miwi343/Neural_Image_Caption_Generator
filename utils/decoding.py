"""Shared caption decoding helpers for evaluation and visualization."""

from typing import List, Tuple

import torch

from utils.dataset import END_IDX, START_IDX


@torch.no_grad()
def greedy_decode_from_encoder_out(
    decoder,
    encoder_out: torch.Tensor,
    vocab,
    device: torch.device,
    max_len: int = 50,
) -> Tuple[str, torch.Tensor, List[int]]:
    """Generate one caption from precomputed encoder annotations."""
    decoder.eval()

    encoder_out = encoder_out.to(device)
    h, c = decoder._init_hidden(encoder_out)

    word_idx = torch.tensor([START_IDX], dtype=torch.long, device=device)
    generated_ids: List[int] = []
    attention_maps = []

    for _ in range(max_len):
        logits, h, c, alpha, _ = decoder.decode_step(encoder_out, word_idx, h, c)
        word_idx = logits.argmax(dim=1)
        idx = int(word_idx.item())

        if idx == END_IDX:
            break

        generated_ids.append(idx)
        attention_maps.append(alpha.squeeze(0).detach().cpu())

    alphas = (
        torch.stack(attention_maps, dim=0)
        if attention_maps
        else torch.empty(0, encoder_out.size(1))
    )
    return vocab.decode(generated_ids), alphas, generated_ids


@torch.no_grad()
def greedy_decode(
    encoder,
    decoder,
    image: torch.Tensor,
    vocab,
    device: torch.device,
    max_len: int = 50,
) -> Tuple[str, torch.Tensor, List[int]]:
    """
    Generate one caption with greedy decoding.

    Returns caption text, attention weights, and generated token ids, all
    excluding special tokens.
    """
    encoder.eval()
    decoder.eval()

    image = image.to(device)
    encoder_out = encoder(image)
    return greedy_decode_from_encoder_out(
        decoder, encoder_out, vocab, device, max_len=max_len
    )


@torch.no_grad()
def beam_search_decode(
    encoder,
    decoder,
    image: torch.Tensor,
    vocab,
    device: torch.device,
    beam_width: int = 3,
    max_len: int = 50,
    length_normalize: bool = False,
) -> str:
    """Generate one caption with beam search."""
    encoder.eval()
    decoder.eval()

    image = image.to(device)
    encoder_out = encoder(image)
    h, c = decoder._init_hidden(encoder_out)

    beams = [(0.0, [START_IDX], h, c)]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for score, ids, bh, bc in beams:
            last_token = torch.tensor([ids[-1]], dtype=torch.long, device=device)
            logits, nh, nc, _, _ = decoder.decode_step(
                encoder_out, last_token, bh, bc
            )
            log_probs = torch.log_softmax(logits, dim=1)
            topk_probs, topk_ids = log_probs.topk(beam_width, dim=1)

            for k in range(beam_width):
                next_id = int(topk_ids[0, k].item())
                new_score = score + float(topk_probs[0, k].item())
                new_ids = ids + [next_id]
                if next_id == END_IDX:
                    completed.append((new_score, new_ids))
                else:
                    new_beams.append((new_score, new_ids, nh, nc))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
        if not beams:
            break

    if not completed:
        completed = [(score, ids) for score, ids, _, _ in beams]

    def rank(item):
        score, ids = item
        if not length_normalize:
            return score
        return score / max(len(ids) - 1, 1)

    _, best_ids = max(completed, key=rank)
    return vocab.decode(best_ids[1:])
