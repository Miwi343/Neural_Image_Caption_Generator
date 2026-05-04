"""
Ablation experiment runner for Show, Attend and Tell.

Defines six experiments. By default this script only *prints* the commands
so you can review and run them manually. Pass --run to actually execute them,
or --smoke to run a 1-batch sanity check for each configuration without full
training.

Usage:
    # Print all commands (dry run)
    python scripts/run_ablations.py

    # Run smoke tests for every config (fast, no GPU needed)
    python scripts/run_ablations.py --smoke

    # Execute all full training runs sequentially (expensive)
    python scripts/run_ablations.py --run

Experiments
-----------
a) baseline_soft_attention       — paper's exact soft attention (default flags)
b) no_attention_mean             — static mean-pooled context, no attention MLP
c) no_doubly_stochastic_lambda_0 — disable doubly stochastic regularisation
d) no_beta_gate                  — force β = 1, remove scalar context gate
e) feature_grid_7x7              — 7×7 feature map (L=49) instead of 14×14 (L=196)

All outputs go to results/ablation_results/<experiment_name>/ so results are isolated.
Each run saves:
    results/ablation_results/<name>/config.json          — training args
    results/ablation_results/<name>/training_log.csv     — per-epoch BLEU + loss
    results/ablation_results/<name>/checkpoints/best.pt  — best checkpoint by BLEU-4
    results/ablation_results/<name>/test_bleu.json       — held-out test metrics
"""

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

BASE_OUT = "results/ablation_results"

EXPERIMENTS = [
    {
        "name": "baseline_soft_attention",
        "description": "Paper's soft attention with all defaults (Xu et al. 2015).",
        "train_flags": [],
        "eval_flags": [],
    },
    {
        "name": "no_attention_mean",
        "description": (
            "Replace dynamic attention context with static mean-pooled image "
            "feature. Tests whether any form of attention (vs. mean pooling) matters."
        ),
        "train_flags": ["--attention_mode", "none"],
        "eval_flags": ["--attention_mode", "none"],
    },
    {
        "name": "no_doubly_stochastic_lambda_0",
        "description": (
            "Set λ=0 to fully disable the doubly stochastic attention "
            "regularisation penalty (Eq. 14)."
        ),
        "train_flags": ["--lambda_weight", "0.0"],
        "eval_flags": [],
    },
    {
        "name": "no_beta_gate",
        "description": (
            "Force β_t = 1 at every step, removing the learned scalar context "
            "gate (paper section 4.2.1)."
        ),
        "train_flags": ["--no_beta_gate"],
        "eval_flags": ["--no_beta_gate"],
    },
    {
        "name": "feature_grid_7x7",
        "description": (
            "Use 7×7 visual feature maps (L=49) instead of the paper's 14×14 "
            "(L=196) via adaptive average pooling."
        ),
        "train_flags": ["--feature_grid_size", "7"],
        "eval_flags": ["--feature_grid_size", "7"],
    },
]


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _out(name: str) -> str:
    return os.path.join(BASE_OUT, name)


def _train_cmd(exp: dict) -> list:
    name = exp["name"]
    out = _out(name)
    return (
        [sys.executable, "train.py"]
        + exp["train_flags"]
        + ["--checkpoint_dir", os.path.join(out, "checkpoints")]
        + ["--results_dir", out]
    )


def _eval_cmd(exp: dict) -> list:
    name = exp["name"]
    out = _out(name)
    return (
        [sys.executable, "evaluate.py"]
        + exp["eval_flags"]
        + ["--checkpoint", os.path.join(out, "checkpoints", "best.pt")]
        + ["--results_out", os.path.join(out, "test_bleu.json")]
    )


def _smoke_cmd(exp: dict) -> list:
    name = exp["name"]
    out = _out(name)
    return (
        [sys.executable, "-m", "pytest",
         "tests/test_ablations.py::test_smoke_forward",
         "-k", name,
         "-v", "--tb=short"]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_plan() -> None:
    """Print every train + evaluate command without running anything."""
    print("=" * 72)
    print("Ablation experiment plan — commands to run manually")
    print("=" * 72)
    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n{'─' * 60}")
        print(f"  [{name}]")
        print(f"  {exp['description']}")
        print()
        train_cmd = " ".join(_train_cmd(exp))
        eval_cmd  = " ".join(_eval_cmd(exp))
        print(f"  # Train")
        print(f"  {train_cmd}")
        print()
        print(f"  # Evaluate on test split")
        print(f"  {eval_cmd}")
        out = _out(name)
        print()
        print(f"  # Outputs → {out}/")
    print(f"\n{'─' * 60}")
    print(
        "\nTo execute all runs:\n"
        "    python scripts/run_ablations.py --run\n"
        "\nTo run smoke tests only:\n"
        "    python scripts/run_ablations.py --smoke"
    )


def _run_cmd(cmd: list, label: str) -> bool:
    print(f"\n>>> {label}")
    print("    " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}): {label}")
        return False
    return True


def _run_all() -> None:
    """Run train + evaluate for every experiment sequentially."""
    failures = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        ok = _run_cmd(_train_cmd(exp), f"Training: {name}")
        if not ok:
            failures.append(f"train:{name}")
            continue
        ok = _run_cmd(_eval_cmd(exp), f"Evaluating: {name}")
        if not ok:
            failures.append(f"eval:{name}")

    if failures:
        print(f"\nFailed steps: {failures}")
        sys.exit(1)
    else:
        print("\nAll experiments completed successfully.")


def _run_smoke() -> None:
    """Run the lightweight smoke tests for every ablation config."""
    print("Running smoke tests for all ablation configurations...")
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_ablations.py",
        "-v", "--tb=short",
    ]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Suggested results table
# ---------------------------------------------------------------------------

RESULTS_TABLE_TEMPLATE = """
Suggested results table
=======================

| Experiment                      | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | Val Loss | Notes                        |
|---------------------------------|--------|--------|--------|--------|--------|----------|------------------------------|
| baseline_soft_attention         |        |        |        |        |        |          | Paper target: 67/44.8/29.9/19.5 |
| no_attention_mean               |        |        |        |        |        |          | Static mean context          |
| no_doubly_stochastic_lambda_0   |        |        |        |        |        |          | λ=0, no Eq.14 penalty        |
| no_beta_gate                    |        |        |        |        |        |          | β forced to 1                |
| feature_grid_7x7                |        |        |        |        |        |          | L=49 instead of 196          |

Fill in from results/ablation_results/<name>/test_bleu.json after each run.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation experiment runner for Show, Attend and Tell."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute all training + evaluation runs sequentially.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the lightweight smoke tests for each ablation config.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print the suggested results table template.",
    )
    args = parser.parse_args()

    if args.smoke:
        _run_smoke()
    elif args.run:
        _run_all()
    elif args.table:
        print(RESULTS_TABLE_TEMPLATE)
    else:
        _print_plan()
