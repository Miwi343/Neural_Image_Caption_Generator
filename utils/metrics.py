"""
BLEU evaluation utilities.

Computes corpus-level modified n-gram precision without brevity penalty to
match the course project protocol for the paper reproduction.

Target performance on Flickr8k (single model, Table 1):
  Soft-Attention:  BLEU-1=67  BLEU-2=44.8  BLEU-3=29.9  BLEU-4=19.5
  Hard-Attention:  BLEU-1=67  BLEU-2=45.7  BLEU-3=31.4  BLEU-4=21.3

Future work (Issue #10): Add a regression test with frozen hypotheses/references that
checks BLEU-1..4 against expected no-brevity-penalty outputs.
Future work (Issue #10, deferred): If the report scope expands beyond proposal BLEU,
add METEOR via NLTK while keeping the existing BLEU-only API backward-compatible.
"""

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
    """
    Compute corpus BLEU-1 through BLEU-4 without brevity penalty.

    Args:
        hypotheses:  list of N generated captions, each as a list of tokens
                     e.g. [["a", "dog", "on", "a", "beach"], ...]
        references:  list of N reference groups, each group is a list of
                     reference captions (each as a list of tokens)
                     e.g. [[["a", "dog", "..."], ["the", "dog", "..."]], ...]

    Returns:
        dict with keys "bleu1", "bleu2", "bleu3", "bleu4" → float in [0, 1]

    This follows the project spec and the paper's Flickr8k reporting convention:
    clipped n-gram precisions are aggregated at corpus level, smoothed only when
    an n-gram precision is zero, and no brevity penalty is applied.
    """
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
    """
    Print a row matching the format of Table 1 in the paper.

    Example output:
        ╔══════════════════════╦════════╦════════╦════════╦════════╗
        ║ Model                ║ BLEU-1 ║ BLEU-2 ║ BLEU-3 ║ BLEU-4 ║
        ╠══════════════════════╬════════╬════════╬════════╬════════╣
        ║ Soft-Attention       ║  67.0  ║  44.8  ║  29.9  ║  19.5  ║
        ║ Hard-Attention       ║  67.0  ║  45.7  ║  31.4  ║  21.3  ║
        ╠══════════════════════╬════════╬════════╬════════╬════════╣
        ║ Our Soft-Attention   ║  XX.X  ║  XX.X  ║  XX.X  ║  XX.X  ║
        ╚══════════════════════╩════════╩════════╩════════╩════════╝

    Paper targets (Table 1, Flickr8k, single model):
        Soft-Attention: BLEU-4 ≈ 19.5
        Hard-Attention: BLEU-4 ≈ 21.3

    Future work (Issue #10, deferred): If METEOR is added to `compute_bleu`, extend
    this table formatter with a METEOR column without changing the current BLEU
    column order used in existing logs.
    Future work (Issue #10, deferred): If report generation needs it, accept multiple
    model rows so paper baselines and several experiment runs can be printed in
    one comparison table.
    """
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
    # Paper baselines for quick comparison
    # Future work (Issue #10, deferred): Move these baseline rows into a small JSON or
    # constants module if multiple datasets/experiments need different reference
    # tables; keep the current hardcoded values until that becomes necessary.
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
