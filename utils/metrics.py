"""
BLEU evaluation utilities.

Uses NLTK corpus_bleu (smoothing method 1) to match the evaluation protocol
described in the paper (section 5.1).

Target performance on Flickr8k (single model, Table 1):
  Soft-Attention:  BLEU-1=67  BLEU-2=44.8  BLEU-3=29.9  BLEU-4=19.5
  Hard-Attention:  BLEU-1=67  BLEU-2=45.7  BLEU-3=31.4  BLEU-4=21.3

TODO (Issue #10): Verify NLTK corpus_bleu with method1 smoothing matches the
                  original Perl BLEU script used by the paper authors.
TODO (Issue #10): Add METEOR metric using the NLTK meteor_score function.
"""

from typing import Dict, List

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def compute_bleu(
    hypotheses: List[List[str]],
    references: List[List[List[str]]],
) -> Dict[str, float]:
    """
    Compute BLEU-1 through BLEU-4 using NLTK corpus_bleu.

    Args:
        hypotheses:  list of N generated captions, each as a list of tokens
                     e.g. [["a", "dog", "on", "a", "beach"], ...]
        references:  list of N reference groups, each group is a list of
                     reference captions (each as a list of tokens)
                     e.g. [[["a", "dog", "..."], ["the", "dog", "..."]], ...]

    Returns:
        dict with keys "bleu1", "bleu2", "bleu3", "bleu4" → float in [0, 1]

    TODO (Issue #10): Support token strings vs token ids — auto-detect and
                      decode ids via a Vocabulary if needed.
    TODO (Issue #10): Add option to compute per-image BLEU for error analysis.
    """
    # Smoothing method 1: add epsilon to precision counts for n-grams with 0
    # count (avoids log(0) for short sentences)
    smoother = SmoothingFunction().method1

    scores = {}
    for n, key in enumerate(["bleu1", "bleu2", "bleu3", "bleu4"], start=1):
        weights = tuple(1.0 / n for _ in range(n)) + tuple(0.0 for _ in range(4 - n))
        # TODO (Issue #10): Double-check weights format: corpus_bleu expects
        #                   a 4-tuple where weights for higher n are 0.
        scores[key] = corpus_bleu(
            references, hypotheses,
            weights=weights,
            smoothing_function=smoother,
        )

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

    TODO (Issue #10): Include METEOR column.
    TODO (Issue #10): Accept multiple model rows for side-by-side comparison.
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
    # TODO (Issue #10): Load baselines from a JSON config instead of hardcoding
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
