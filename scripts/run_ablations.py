"""
Ablation experiment runner for Show, Attend and Tell.

Trains every ablation variant from scratch, then evaluates each with the full
beam-width × length-normalisation sweep so results are directly comparable.

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
    results/ablation_results/<name>/config.json
    results/ablation_results/<name>/training_log.csv
    results/ablation_results/<name>/checkpoints/best.pt
    results/ablation_results/<name>/checkpoints/epoch_NNN.pt
    results/ablation_results/<name>/eval_beam<N>_ln<bool>.json   (x5 per model)

Experiments
-----------
Architectural (require different model weights — cannot be applied to baseline):
  no_attention       — mean-pooled static context, no spatial attention MLP
  no_beta_gate       — β forced to 1 (removes learned scalar context gate)
  feature_grid_7x7   — 7×7 feature maps (L=49) instead of 14×14 (L=196)

Regularisation:
  lambda_0           — λ=0, doubly stochastic regularisation fully disabled
  lambda_0_5         — λ=0.5, softer regularisation than paper default (1.0)
  lambda_2_0         — λ=2.0, stronger regularisation than paper default

Encoder fine-tuning schedule:
  finetune_ep3       — unlock encoder after epoch 3 (earlier than default 5)
  finetune_ep10      — unlock encoder after epoch 10 (later than default 5)

Combined:
  lambda_0_5_finetune3 — λ=0.5 + encoder unlock at epoch 3
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

BASE_OUT   = "results/ablation_results"
DATA_ROOT  = "data/flickr8k"          # overridden by --data_root
VOCAB_PATH = "data/flickr8k/vocab.json"  # overridden by --vocab

EVAL_SWEEP = [
    # (tag_suffix,          extra_eval_flags)
    ("beam1_lnFalse",  []),
    ("beam3_lnFalse",  ["--beam_width", "3"]),
    ("beam5_lnFalse",  ["--beam_width", "5"]),
    ("beam3_lnTrue",   ["--beam_width", "3", "--length_normalize"]),
    ("beam5_lnTrue",   ["--beam_width", "5", "--length_normalize"]),
]

# Paper Flickr8k soft-attention numbers for comparison
PAPER = {"bleu1": 67.0, "bleu2": 44.8, "bleu3": 29.9, "bleu4": 19.5, "meteor": 18.93}

EXPERIMENTS = [
    # ── Architectural ────────────────────────────────────────────────────────
    {
        "name": "no_attention",
        "description": "Mean-pooled static context; no spatial attention MLP (attention_mode=none).",
        "group": "architectural",
        "train_flags": ["--attention_mode", "none"],
        "eval_flags":  ["--attention_mode", "none"],
    },
    {
        "name": "no_beta_gate",
        "description": "Force β=1 at every step; removes learned scalar context gate (section 4.2.1).",
        "group": "architectural",
        "train_flags": ["--no_beta_gate"],
        "eval_flags":  ["--no_beta_gate"],
    },
    {
        "name": "feature_grid_7x7",
        "description": "7×7 feature maps (L=49) instead of 14×14 (L=196) via adaptive avg pooling.",
        "group": "architectural",
        "train_flags": ["--feature_grid_size", "7"],
        "eval_flags":  ["--feature_grid_size", "7"],
    },
    # ── Regularisation ───────────────────────────────────────────────────────
    {
        "name": "lambda_0",
        "description": "λ=0: doubly stochastic attention regularisation fully disabled.",
        "group": "regularisation",
        "train_flags": ["--lambda_weight", "0.0"],
        "eval_flags":  [],
    },
    {
        "name": "lambda_0_5",
        "description": "λ=0.5: softer doubly stochastic regularisation (half of paper default).",
        "group": "regularisation",
        "train_flags": ["--lambda_weight", "0.5"],
        "eval_flags":  [],
    },
    {
        "name": "lambda_2_0",
        "description": "λ=2.0: stronger doubly stochastic regularisation (double paper default).",
        "group": "regularisation",
        "train_flags": ["--lambda_weight", "2.0"],
        "eval_flags":  [],
    },
    # ── Encoder fine-tuning schedule ─────────────────────────────────────────
    {
        "name": "finetune_ep3",
        "description": "Unlock encoder fine-tuning after epoch 3 (earlier than default 5).",
        "group": "finetune_schedule",
        "train_flags": ["--finetune_epoch", "3"],
        "eval_flags":  [],
    },
    {
        "name": "finetune_ep10",
        "description": "Unlock encoder fine-tuning after epoch 10 (later than default 5).",
        "group": "finetune_schedule",
        "train_flags": ["--finetune_epoch", "10"],
        "eval_flags":  [],
    },
    # ── Combined ─────────────────────────────────────────────────────────────
    {
        "name": "lambda_0_5_finetune3",
        "description": "λ=0.5 combined with early encoder unlock at epoch 3.",
        "group": "combined",
        "train_flags": ["--lambda_weight", "0.5", "--finetune_epoch", "3"],
        "eval_flags":  [],
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

def eval_out(name: str, sweep_tag: str) -> Path:
    return exp_dir(name) / f"eval_{sweep_tag}.json"


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

def eval_cmd(exp: dict, sweep_tag: str, sweep_extra: list) -> list:
    return (
        [sys.executable, "evaluate.py"]
        + exp["eval_flags"]
        + sweep_extra
        + ["--checkpoint",  str(best_ckpt(exp["name"]))]
        + ["--data_root",   DATA_ROOT]
        + ["--vocab",       VOCAB_PATH]
        + ["--results_out", str(eval_out(exp["name"], sweep_tag))]
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
    """Train one experiment then run the full eval sweep. Returns per-sweep scores."""
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

    # ── Eval sweep ───────────────────────────────────────────────────────────
    sweep_scores = {}
    for sweep_tag, sweep_extra in EVAL_SWEEP:
        out_path = eval_out(name, sweep_tag)
        if resume and out_path.exists():
            print(f"[{name}] {sweep_tag} already evaluated — skipping")
            with open(out_path) as f:
                sweep_scores[sweep_tag] = json.load(f)["scores"]
            continue

        ok = run(eval_cmd(exp, sweep_tag, sweep_extra), f"EVAL   {name}  [{sweep_tag}]")
        if ok and out_path.exists():
            with open(out_path) as f:
                sweep_scores[sweep_tag] = json.load(f)["scores"]

    return sweep_scores


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def load_all_results() -> dict:
    """Load whatever eval JSONs already exist on disk."""
    results = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        sweep_scores = {}
        for sweep_tag, _ in EVAL_SWEEP:
            p = eval_out(name, sweep_tag)
            if p.exists():
                with open(p) as f:
                    sweep_scores[sweep_tag] = json.load(f)["scores"]
        if sweep_scores:
            results[name] = sweep_scores
    return results


def print_summary(results: dict) -> None:
    if not results:
        print("No results found. Run with --run first.")
        return

    metrics = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor"]
    mnames  = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR"]

    # Pick the best sweep config (highest BLEU-4) for each experiment
    def best_sweep(sweep_scores):
        return max(sweep_scores.items(),
                   key=lambda kv: kv[1].get("bleu4", 0))

    col_w = 28
    print("\n" + "=" * 90)
    print("  ABLATION RESULTS SUMMARY  (best eval config per experiment)")
    print("=" * 90)
    print(f"  {'Experiment':<{col_w}}  {'Config':<16}  "
          + "  ".join(f"{m:>7}" for m in mnames))
    print("  " + "-" * 86)
    # Paper baseline
    print(f"  {'Paper Soft-Att (Table 1)':<{col_w}}  {'—':16}  "
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
                print(f"  {'  ' + name:<{col_w+2}}  {'(not run yet)':16}")
                continue
            sweep_tag, scores = best_sweep(results[name])
            meteor = scores.get("meteor", 0)
            vals = [scores.get(k, 0)*100 for k in ["bleu1","bleu2","bleu3","bleu4"]]
            vals.append(meteor*100)
            row = "  ".join(f"{v:>7.2f}" for v in vals)
            print(f"  {'  ' + name:<{col_w+2}}  {sweep_tag:<16}  {row}")

    print("\n" + "=" * 90)
    print("  Full per-sweep breakdowns:")
    for name, sweep_scores in results.items():
        exp_obj = next(e for e in EXPERIMENTS if e["name"] == name)
        print(f"\n  {name}  [{exp_obj['group']}]")
        print(f"    {'Sweep config':<20}  " + "  ".join(f"{m:>7}" for m in mnames))
        print("    " + "-" * 68)
        for sweep_tag, scores in sweep_scores.items():
            vals = [scores.get(k, 0)*100 for k in ["bleu1","bleu2","bleu3","bleu4"]]
            vals.append(scores.get("meteor", 0)*100)
            row = "  ".join(f"{v:>7.2f}" for v in vals)
            print(f"    {sweep_tag:<20}  {row}")
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
        for sweep_tag, sweep_extra in EVAL_SWEEP:
            print(f"  Eval [{sweep_tag}]:  {' '.join(eval_cmd(exp, sweep_tag, sweep_extra))}")
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
