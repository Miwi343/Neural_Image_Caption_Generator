"""
Ablation experiment runner for Show, Attend and Tell.

Trains each requested ablation variant from scratch, then evaluates on
validation and test splits. By default this script prints the plan only.

Usage
-----
    # Print the full plan (no training, no side effects)
    python scripts/run_ablations.py

    # Run all experiments sequentially (Colab / GPU server)
    python scripts/run_ablations.py --run

    # Resume — skip any experiment whose best.pt already exists
    python scripts/run_ablations.py --run --resume

    # Run only specific experiments by name
    python scripts/run_ablations.py --run --only no_beta_gate feature_grid_7x7

    # Smoke-test every config (fast, CPU-ok, no full training)
    python scripts/run_ablations.py --smoke

    # Print a results table from whatever has already been evaluated
    python scripts/run_ablations.py --summary

Outputs (per experiment)
------------------------
    outputs/ablations/<name>/config.json
    outputs/ablations/<name>/training_log.csv
    outputs/ablations/<name>/checkpoints/best.pt
    outputs/ablations/<name>/checkpoints/epoch_NNN.pt
    outputs/ablations/<name>/eval_val.json
    outputs/ablations/<name>/eval_test.json

Experiments
-----------
  baseline_soft_attention         — default soft attention
  no_attention_mean               — static mean context (no attention MLP)
  uniform_attention               — force alpha=1/L at every step
  no_doubly_stochastic_lambda_0   — lambda_weight=0
  no_beta_gate                    — beta forced to 1
  feature_grid_7x7                — 7×7 feature maps (L=49)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

BASE_OUT   = "outputs/ablations"
DATA_ROOT  = "data/flickr8k"          # overridden by --data_root
VOCAB_PATH = "data/flickr8k/vocab.json"  # overridden by --vocab

# Paper Flickr8k soft-attention numbers for comparison
PAPER = {"bleu1": 67.0, "bleu2": 44.8, "bleu3": 29.9, "bleu4": 19.5, "meteor": 18.93}

EXPERIMENTS = [
    {
        "name": "baseline_soft_attention",
        "description": "Baseline: learned soft attention, beta gate enabled, 14x14 features, lambda=default.",
        "group": "baseline",
        "train_flags": [],
        "eval_flags":  [],
    },
    {
        "name": "no_attention_mean",
        "description": "No attention: replace dynamic context with mean_i a_i at every timestep (attention_mode=none).",
        "group": "architectural",
        "train_flags": ["--attention_mode", "none"],
        "eval_flags":  ["--attention_mode", "none"],
    },
    {
        "name": "uniform_attention",
        "description": "Uniform attention: force alpha_{t,i}=1/L for all regions (attention_mode=uniform).",
        "group": "architectural",
        "train_flags": ["--attention_mode", "uniform"],
        "eval_flags":  ["--attention_mode", "uniform"],
    },
    {
        "name": "no_doubly_stochastic_lambda_0",
        "description": "Disable doubly stochastic regularization by setting lambda_weight=0.",
        "group": "regularisation",
        "train_flags": ["--lambda_weight", "0.0"],
        "eval_flags":  ["--lambda_weight", "0.0"],
    },
    {
        "name": "no_beta_gate",
        "description": "Disable beta gate by forcing beta_t=1 at every timestep.",
        "group": "architectural",
        "train_flags": ["--no_beta_gate"],
        "eval_flags":  ["--no_beta_gate"],
    },
    {
        "name": "feature_grid_7x7",
        "description": "Feature resolution ablation: adaptive avg pool to 7x7 feature grid (L=49).",
        "group": "architectural",
        "train_flags": ["--feature_grid_size", "7"],
        "eval_flags":  ["--feature_grid_size", "7"],
    },
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def exp_dir(name: str) -> Path:
    return Path(BASE_OUT) / name

def ckpt_dir(name: str) -> Path:
    return exp_dir(name) / "checkpoints"

def best_ckpt(name: str) -> Path:
    return ckpt_dir(name) / "best.pt"

def eval_out(name: str, split: str) -> Path:
    return exp_dir(name) / f"eval_{split}.json"


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def train_cmd(exp: dict) -> list:
    return (
        [sys.executable, "train.py"]
        + exp["train_flags"]
        + ["--data_root",      DATA_ROOT]
        + ["--vocab",          VOCAB_PATH]
        + ["--checkpoint_dir", str(ckpt_dir(exp["name"]))]
        + ["--results_dir",    str(exp_dir(exp["name"]))]
    )

def eval_cmd(exp: dict, split: str) -> list:
    return (
        [sys.executable, "evaluate.py"]
        + exp["eval_flags"]
        + ["--checkpoint",  str(best_ckpt(exp["name"]))]
        + ["--data_root",   DATA_ROOT]
        + ["--vocab",       VOCAB_PATH]
        + ["--split",       split]
        + ["--results_out", str(eval_out(exp["name"], split))]
        + ["--batch_size", "64"]
    )


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def run(cmd: list, label: str) -> bool:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'─'*60}")
    return subprocess.run(cmd).returncode == 0


def run_experiment(exp: dict, resume: bool) -> dict:
    """Train one experiment then run val+test evaluation. Returns split->scores."""
    name = exp["name"]

    # ── Training ─────────────────────────────────────────────────────────────
    if resume and best_ckpt(name).exists():
        print(f"\n[{name}] best.pt already exists — skipping training (--resume)")
    else:
        ok = run(train_cmd(exp), f"TRAIN  {name}")
        if not ok:
            print(f"[{name}] training FAILED — skipping eval")
            return {}

    if not best_ckpt(name).exists():
        print(f"[{name}] best.pt not found after training — skipping eval")
        return {}

    # ── Eval (val + test) ────────────────────────────────────────────────────
    split_scores = {}
    for split in ("val", "test"):
        out_path = eval_out(name, split)
        if resume and out_path.exists():
            print(f"[{name}] {split} already evaluated — skipping")
            with open(out_path) as f:
                split_scores[split] = json.load(f)["scores"]
            continue

        ok = run(eval_cmd(exp, split), f"EVAL   {name}  [{split}]")
        if ok and out_path.exists():
            with open(out_path) as f:
                split_scores[split] = json.load(f)["scores"]

    return split_scores


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def load_all_results() -> dict:
    """Load whatever eval JSONs already exist on disk."""
    results = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        split_scores = {}
        for split in ("val", "test"):
            p = eval_out(name, split)
            if p.exists():
                with open(p) as f:
                    split_scores[split] = json.load(f)["scores"]
        if split_scores:
            results[name] = split_scores
    return results


def print_summary(results: dict) -> None:
    if not results:
        print("No results found. Run with --run first.")
        return

    metrics = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor"]
    mnames  = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR"]

    def fmt_scores(scores: dict) -> str:
        vals = [scores.get(k, 0) * 100 for k in ["bleu1", "bleu2", "bleu3", "bleu4"]]
        vals.append(scores.get("meteor", 0) * 100)
        return "  ".join(f"{v:>7.2f}" for v in vals)

    col_w = 28
    print("\n" + "=" * 90)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"  {'Experiment':<{col_w}}  {'Split':<6}  " + "  ".join(f"{m:>7}" for m in mnames))
    print("  " + "-" * 86)
    # Paper baseline
    print(f"  {'Paper Soft-Att (Table 1)':<{col_w}}  {'—':<6}  "
          + "  ".join(f"{PAPER[k]:>7.1f}" for k in metrics))
    print("  " + "-" * 86)

    # Group results
    groups = {}
    for exp in EXPERIMENTS:
        g = exp["group"]
        groups.setdefault(g, []).append(exp["name"])

    for group, names in groups.items():
        print(f"\n  [{group}]")
        for name in names:
            if name not in results:
                print(f"  {'  ' + name:<{col_w+2}}  {'(not run yet)':<6}")
                continue
            for split in ("val", "test"):
                scores = results[name].get(split)
                if not scores:
                    continue
                print(f"  {'  ' + name:<{col_w+2}}  {split:<6}  {fmt_scores(scores)}")

    print("\n" + "=" * 90)
    print("  Full per-split breakdowns:")
    for name, split_scores in results.items():
        exp_obj = next(e for e in EXPERIMENTS if e["name"] == name)
        print(f"\n  {name}  [{exp_obj['group']}]")
        for split in ("val", "test"):
            scores = split_scores.get(split)
            if scores:
                print(f"    {split:<6}  {fmt_scores(scores)}")
    print()


# ---------------------------------------------------------------------------
# Plan printer
# ---------------------------------------------------------------------------

def print_plan(only: list) -> None:
    exps = [e for e in EXPERIMENTS if not only or e["name"] in only]
    print("=" * 72)
    print(f"Ablation plan — {len(exps)} experiment(s)")
    print("=" * 72)
    for exp in exps:
        name = exp["name"]
        print(f"\n  [{name}]  group={exp['group']}")
        print(f"  {exp['description']}")
        print(f"\n  Train:  {' '.join(train_cmd(exp))}")
        for split in ("val", "test"):
            print(f"  Eval [{split}]:  {' '.join(eval_cmd(exp, split))}")
        print(f"\n  Outputs → {exp_dir(name)}/")
    print(f"\n{'─'*72}")
    print("Run with --run to execute, --resume to skip completed experiments.")


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def run_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_ablations.py", "-v", "--tb=short"]
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate all Show, Attend and Tell ablation variants."
    )
    parser.add_argument("--run",       action="store_true",
                        help="Execute all training + evaluation runs.")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip experiments whose best.pt / eval JSONs already exist.")
    parser.add_argument("--only",      nargs="+", metavar="NAME",
                        help="Run only the named experiments.")
    parser.add_argument("--smoke",     action="store_true",
                        help="Run smoke tests for each ablation config.")
    parser.add_argument("--summary",   action="store_true",
                        help="Print results table from existing eval JSONs.")
    parser.add_argument("--data_root", default=DATA_ROOT,
                        help=f"Path to flickr8k data root (default: {DATA_ROOT}).")
    parser.add_argument("--vocab",     default=VOCAB_PATH,
                        help=f"Path to vocab.json (default: {VOCAB_PATH}).")
    args = parser.parse_args()

    # Rebind module-level names so all cmd builders pick up the CLI values.
    # No `global` needed — this block is already at module scope.
    DATA_ROOT  = args.data_root  # noqa: F811
    VOCAB_PATH = args.vocab      # noqa: F811

    if args.smoke:
        run_smoke()

    elif args.summary:
        print_summary(load_all_results())

    elif args.run:
        exps = [e for e in EXPERIMENTS if not args.only or e["name"] in args.only]
        print(f"Running {len(exps)} experiment(s)"
              + (" (resume mode)" if args.resume else "") + "\n")

        all_results = {}
        failures = []
        for exp in exps:
            scores = run_experiment(exp, resume=args.resume)
            if scores:
                all_results[exp["name"]] = scores
            else:
                failures.append(exp["name"])

        print_summary(all_results)

        if failures:
            print(f"Failed experiments: {failures}")
            sys.exit(1)
        else:
            print("All experiments completed successfully.")

    else:
        print_plan(args.only or [])
