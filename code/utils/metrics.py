"""BLEU helpers for the Flickr8k captioning experiments."""

import math
from collections import Counter
from typing import Dict, List


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(
    hypotheses: List[List[str]],
    references: List[List[List[str]]],
    n: int,
    epsilon: float = 1e-9,
) -> float:
    clipped_total = 0
    count_total = 0

    for hyp, refs in zip(hypotheses, references):
        hyp_counts = _ngram_counts(hyp, n)
        count_total += sum(hyp_counts.values())

        max_ref_counts: Counter = Counter()
        for ref in refs:
            max_ref_counts |= _ngram_counts(ref, n)

        clipped_total += sum(
            min(count, max_ref_counts[ngram])
            for ngram, count in hyp_counts.items()
        )

    if count_total == 0:
        return epsilon
    if clipped_total == 0:
        return epsilon / count_total
    return clipped_total / count_total


def compute_bleu(
    hypotheses: List[List[str]],
    references: List[List[List[str]]],
) -> Dict[str, float]:
    """Compute corpus BLEU-1 through BLEU-4 without brevity penalty."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length.")

    scores = {}
    for n, key in enumerate(["bleu1", "bleu2", "bleu3", "bleu4"], start=1):
        precisions = [
            _modified_precision(hypotheses, references, order)
            for order in range(1, n + 1)
        ]
        scores[key] = math.exp(sum(math.log(p) for p in precisions) / n)

    return scores


def print_bleu_table(
    model_name: str,
    scores: Dict[str, float],
    dataset: str = "Flickr8k",
) -> None:
    """Print BLEU scores beside the paper's Flickr8k baselines."""
    b1 = scores["bleu1"] * 100
    b2 = scores["bleu2"] * 100
    b3 = scores["bleu3"] * 100
    b4 = scores["bleu4"] * 100

    sep = "=" * 70
    print(sep)
    print(f"  BLEU results — {dataset}")
    print(sep)
    print(f"  {'Model':<30}  {'BLEU-1':>6}  {'BLEU-2':>6}  {'BLEU-3':>6}  {'BLEU-4':>6}")
    print("-" * 70)
    baselines = [
        ("Soft-Attention (paper)", 67.0, 44.8, 29.9, 19.5),
        ("Hard-Attention (paper)", 67.0, 45.7, 31.4, 21.3),
    ]
    for row in baselines:
        name, r1, r2, r3, r4 = row
        print(f"  {name:<30}  {r1:>6.1f}  {r2:>6.1f}  {r3:>6.1f}  {r4:>6.1f}")
    print("-" * 70)
    print(f"  {model_name:<30}  {b1:>6.1f}  {b2:>6.1f}  {b3:>6.1f}  {b4:>6.1f}")
    print(sep)
