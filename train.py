"""
Training script for Show, Attend and Tell (Xu et al., 2015).

Implements:
  - Cross-entropy loss over predicted word distribution
  - Doubly stochastic regularisation (Eq. 14): λ * Σ_i (1 - Σ_t α_ti)²
  - Gradient clipping (5.0) to stabilise LSTM training
  - Adam optimiser on decoder parameters only (encoder is frozen)
  - Per-epoch checkpointing + early stopping on validation BLEU-4
  - Batch length-bucketing for training speed (paper section 4.3)

Ablation flags (all default to the paper's exact behaviour):
  --attention_mode   soft | uniform | none
  --no_beta_gate     disable the learned β scalar gate (default: enabled)
  --lambda_weight    doubly stochastic regularisation weight (default: 1.0)
  --feature_grid_size  14 | 7  (spatial annotation resolution, default: 14)
  --checkpoint_dir   where to save epoch/best checkpoints
  --results_dir      where to save training_log.csv and config.json

Future work (Issue #7, deferred): If training stability work is assigned, log
exactly when the LR changes so runs remain comparable.
Future work (Issue #8, deferred): If scaling beyond one GPU becomes necessary,
wrap the model with DDP/DataParallel only after single-GPU training is stable.
Future work (Issue #9, deferred): If experiment tracking is needed for the report,
log loss and BLEU curves to TensorBoard or W&B without changing the default
CLI-free training path.
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from config import (
    ATTENTION_DIM,
    ATTENTION_MODE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_ROOT,
    DECODER_DIM,
    DROPOUT,
    EMBED_DIM,
    ENCODER_FINETUNE_EPOCH,
    ENCODER_LR,
    FEATURE_GRID_SIZE,
    GRAD_CLIP,
    LAMBDA,
    MAX_DECODE_LEN,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    RESULTS_DIR,
    USE_BETA_GATE,
    VOCAB_PATH,
    VOCAB_SIZE,
)
from models import Encoder, Decoder
from utils import (
    Vocabulary,
    compute_bleu,
    greedy_decode_from_encoder_out,
    get_dataloader,
    load_flickr8k_captions,
    print_bleu_table,
    tokenize_caption,
)
from utils.dataset import PAD_IDX


def doubly_stochastic_attention_loss(alphas: torch.Tensor, weight: float = LAMBDA):
    """Eq. 14: lambda * mean over (batch, locations) of (1 - sum_t alpha_ti)^2.
    Using .mean() over locations keeps this term on the same scale as CE loss.
    The original .sum(dim=1) over 196 locations inflated it ~196× and drowned
    the CE gradient signal.
    """
    return weight * ((1.0 - alphas.sum(dim=1)) ** 2).mean()


def train_epoch(
    encoder,
    decoder,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
    fine_tune_encoder=False,
    lambda_weight: float = LAMBDA,
):
    """
    Run one training epoch.

    Loss = cross-entropy + λ * doubly stochastic regularisation (Eq. 14).
    When lambda_weight=0, the regularisation term is fully disabled.
    """
    if fine_tune_encoder:
        encoder.train()
    else:
        encoder.eval()
    decoder.train()

    total_loss = 0.0
    n_batches = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, captions, lengths, _ in loop:
        images   = images.to(device)
        captions = captions.to(device)
        lengths  = lengths.to(device)

        if fine_tune_encoder:
            encoder_out = encoder(images)
        else:
            with torch.no_grad():
                encoder_out = encoder(images)

        predictions, alphas = decoder(encoder_out, captions, lengths)

        targets = captions[:, 1:]   # (batch, max_len-1)

        vocab_size = predictions.size(-1)
        loss_ce = criterion(
            predictions.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        loss_ds = doubly_stochastic_attention_loss(alphas, lambda_weight)

        loss = loss_ce + loss_ds

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(encoder, decoder, dataloader, vocab, device):
    """
    Greedy-decode the validation set and compute BLEU scores.

    Returns:
        dict with keys `bleu1`/`bleu2`/`bleu3`/`bleu4`.
    """
    encoder.eval()
    decoder.eval()

    hypotheses = []
    references = []

    for images, captions, lengths, all_caps in tqdm(dataloader, desc="[val]", leave=False):
        images = images.to(device)

        encoder_out = encoder(images)   # (batch, L, 512)
        batch_size = images.size(0)

        for i in range(batch_size):
            ann = encoder_out[i].unsqueeze(0)  # (1, L, 512)
            caption, _, _ = greedy_decode_from_encoder_out(
                decoder, ann, vocab, device, max_len=MAX_DECODE_LEN
            )
            hyp_tokens = caption.split()
            hypotheses.append(hyp_tokens)

            refs_tokens = [tokenize_caption(cap) for cap in all_caps[i]]
            references.append(refs_tokens)

    scores = compute_bleu(hypotheses, references)
    return scores


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Show, Attend and Tell (+ ablation variants)"
    )
    # --- Ablation flags ---
    parser.add_argument(
        "--attention_mode",
        choices=["soft", "uniform", "none"],
        default=ATTENTION_MODE,
        help=(
            "Attention variant: 'soft' (default), 'uniform' (force alpha=1/L), "
            "or 'none' (static mean context, no attention MLP)."
        ),
    )
    parser.add_argument(
        "--no_beta_gate",
        action="store_true",
        default=False,
        help="Disable the learned β scalar gate (force β=1 at every step).",
    )
    parser.add_argument(
        "--lambda_weight",
        type=float,
        default=LAMBDA,
        help=(
            f"Doubly stochastic attention regularisation weight (default: {LAMBDA}). "
            "Set to 0 to fully disable."
        ),
    )
    parser.add_argument(
        "--feature_grid_size",
        type=int,
        choices=[7, 14],
        default=FEATURE_GRID_SIZE,
        help=(
            f"Spatial resolution of encoder annotations: 14 (default, L=196) "
            "or 7 (ablation, L=49)."
        ),
    )
    parser.add_argument(
        "--finetune_epoch",
        type=int,
        default=ENCODER_FINETUNE_EPOCH,
        help=f"Epoch to start encoder fine-tuning (default: {ENCODER_FINETUNE_EPOCH}).",
    )
    # --- Data paths ---
    parser.add_argument(
        "--data_root",
        default=DATA_ROOT,
        help=f"Path to Flickr8k data root (default: {DATA_ROOT}).",
    )
    parser.add_argument(
        "--vocab",
        default=VOCAB_PATH,
        help=f"Path to vocab.json (default: {VOCAB_PATH}).",
    )
    # --- Output dirs ---
    parser.add_argument(
        "--checkpoint_dir",
        default=CHECKPOINT_DIR,
        help=f"Directory for epoch and best checkpoints (default: {CHECKPOINT_DIR}).",
    )
    parser.add_argument(
        "--results_dir",
        default=RESULTS_DIR,
        help=f"Directory for training_log.csv and config.json (default: {RESULTS_DIR}).",
    )
    return parser


def main(args=None):
    """
    Full training loop with checkpointing and early stopping.

    Pass a pre-built Namespace to call programmatically (e.g. from
    scripts/run_ablations.py); omit it to parse sys.argv.
    """
    if args is None:
        args = _build_parser().parse_args()

    use_beta_gate = not args.no_beta_gate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Config: attention_mode={args.attention_mode}, "
        f"use_beta_gate={use_beta_gate}, "
        f"lambda={args.lambda_weight}, "
        f"feature_grid_size={args.feature_grid_size}"
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Save the full config so evaluate.py / visualize.py can recreate the model.
    config_path = os.path.join(args.results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "attention_mode": args.attention_mode,
                "use_beta_gate": use_beta_gate,
                "lambda_weight": args.lambda_weight,
                "feature_grid_size": args.feature_grid_size,
                "finetune_epoch": args.finetune_epoch,
                "checkpoint_dir": args.checkpoint_dir,
                "results_dir": args.results_dir,
            },
            f,
            indent=2,
        )
    print(f"Config saved → {config_path}")

    # Build or load vocabulary
    if os.path.exists(args.vocab):
        print("Loading vocabulary...")
        vocab = Vocabulary.load(args.vocab)
    else:
        print("Building vocabulary...")
        image_to_caps = load_flickr8k_captions(args.data_root, "train")
        all_captions = [cap for caps in image_to_caps.values() for cap in caps]
        vocab = Vocabulary(max_size=VOCAB_SIZE)
        vocab.build(all_captions)
        vocab.save(args.vocab)
        print(f"Vocabulary size: {len(vocab)}")

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = get_dataloader(args.data_root, vocab, "train", BATCH_SIZE,
                                  num_workers=num_workers)
    val_loader   = get_dataloader(args.data_root, vocab, "val",   BATCH_SIZE,
                                  num_workers=num_workers, use_bucket_sampler=False)

    encoder = Encoder(
        fine_tune=False,
        feature_grid_size=args.feature_grid_size,
    ).to(device)

    decoder = Decoder(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
        dropout=DROPOUT,
        attention_mode=args.attention_mode,
        use_beta_gate=use_beta_gate,
    ).to(device)

    optimizer = Adam(decoder.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_bleu4 = 0.0
    best_scores = None
    epochs_no_improve = 0
    fine_tune_encoder = False

    log_path = os.path.join(args.results_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss",
            "val_bleu1", "val_bleu2", "val_bleu3", "val_bleu4",
            "seconds",
        ])

    for epoch in range(1, NUM_EPOCHS + 1):

        if epoch == args.finetune_epoch + 1 and not fine_tune_encoder:
            encoder._enable_fine_tune()
            optimizer.add_param_group({
                "params": [p for p in encoder.parameters() if p.requires_grad],
                "lr": ENCODER_LR,
            })
            fine_tune_encoder = True
            print(f"Epoch {epoch}: encoder fine-tuning enabled (LR={ENCODER_LR})")

        t0 = time.time()
        train_loss = train_epoch(
            encoder, decoder, train_loader, optimizer, criterion, device, epoch,
            fine_tune_encoder=fine_tune_encoder,
            lambda_weight=args.lambda_weight,
        )
        val_scores = validate(encoder, decoder, val_loader, vocab, device)
        val_bleu4 = val_scores["bleu4"]
        elapsed = time.time() - t0
        scheduler.step(val_bleu4)

        print(
            f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"val BLEU-1 {val_scores['bleu1']*100:.2f} | "
            f"val BLEU-2 {val_scores['bleu2']*100:.2f} | "
            f"val BLEU-3 {val_scores['bleu3']*100:.2f} | "
            f"val BLEU-4 {val_bleu4*100:.2f} | {elapsed:.0f}s"
        )
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_scores['bleu1']:.6f}",
                f"{val_scores['bleu2']:.6f}",
                f"{val_scores['bleu3']:.6f}",
                f"{val_scores['bleu4']:.6f}",
                f"{elapsed:.2f}",
            ])

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "decoder": decoder.state_dict(),
            "encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_scores": val_scores,
            "val_bleu4": val_bleu4,
            "vocab_size": len(vocab),
            # Ablation config stored in the checkpoint so evaluate.py can
            # warn if the user passes mismatched flags.
            "attention_mode": args.attention_mode,
            "use_beta_gate": use_beta_gate,
            "feature_grid_size": args.feature_grid_size,
            "lambda_weight": args.lambda_weight,
        }, ckpt_path)

        if val_bleu4 > best_bleu4:
            best_bleu4 = val_bleu4
            best_scores = val_scores
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "decoder": decoder.state_dict(),
                "encoder": encoder.state_dict(),
                "val_scores": val_scores,
                "val_bleu4": val_bleu4,
                "vocab_size": len(vocab),
                "attention_mode": args.attention_mode,
                "use_beta_gate": use_beta_gate,
                "feature_grid_size": args.feature_grid_size,
                "lambda_weight": args.lambda_weight,
            }, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  -> New best BLEU-4: {best_bleu4*100:.2f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"Training complete. Best val BLEU-4: {best_bleu4*100:.2f}")
    if best_scores is not None:
        print_bleu_table("Best validation checkpoint", best_scores, dataset="Validation")


if __name__ == "__main__":
    main()
