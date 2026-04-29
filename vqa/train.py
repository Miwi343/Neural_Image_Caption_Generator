"""
Training script for the VQA yes/no model.

Usage:
    python -m vqa.train

Key features that mirror the caption train.py:
  - Adam optimiser, ReduceLROnPlateau on val accuracy
  - Gradient clipping
  - Early stopping
  - Encoder fine-tuning enabled after ENCODER_FINETUNE_EPOCH warm-up epochs
  - CSV training log → results_vqa/training_log.csv
  - Best checkpoint saved to checkpoints_vqa/best.pt
"""

import csv
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vqa.config import (
    ATTENTION_DIM,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_ROOT,
    DROPOUT,
    ENCODER_DIM,
    ENCODER_FINETUNE_EPOCH,
    ENCODER_LR,
    GRU_DIM,
    GRAD_CLIP,
    LAMBDA,
    LEARNING_RATE,
    MAX_QUESTION_LEN,
    NUM_EPOCHS,
    PATIENCE,
    QUESTION_EMBED_DIM,
    RESULTS_DIR,
    VOCAB_PATH,
    VOCAB_SIZE,
)
from vqa.dataset import (
    QuestionVocabulary,
    build_and_save_vocab,
    get_vqa_dataloader,
)
from vqa.model import VQAModel


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def doubly_stochastic_loss(alphas: torch.Tensor, weight: float = LAMBDA) -> torch.Tensor:
    """
    Eq. 14 adapted for single-step attention.
    With one attention step Σ_t α_ti = α_i which is already a valid softmax,
    so this term is always 0 in the single-step case.  Kept for correctness if
    multi-step attention is added later.
    """
    # alphas: (B, L)  — already sums to 1, so this is 0 by construction.
    return weight * ((1.0 - alphas.sum(dim=1)) ** 2).mean()


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, epoch, fine_tune_encoder=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, questions, q_lens, labels in loader:
        images    = images.to(device)
        questions = questions.to(device)
        q_lens    = q_lens.to(device)
        labels    = labels.to(device)

        logits, alphas = model(images, questions, q_lens)

        bce_loss = criterion(logits, labels)
        ds_loss  = doubly_stochastic_loss(alphas)
        loss     = bce_loss + ds_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            GRAD_CLIP,
        )
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds   = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, questions, q_lens, labels in loader:
        images    = images.to(device)
        questions = questions.to(device)
        q_lens    = q_lens.to(device)
        labels    = labels.to(device)

        logits, _ = model(images, questions, q_lens)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds   = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Vocabulary ---
    if os.path.exists(VOCAB_PATH):
        vocab = QuestionVocabulary.load(VOCAB_PATH)
        print(f"Loaded vocab: {len(vocab)} tokens")
    else:
        print("Building vocabulary from training questions …")
        vocab = build_and_save_vocab(DATA_ROOT, VOCAB_PATH, max_size=VOCAB_SIZE)

    # --- Dataloaders ---
    print("Building dataloaders …")
    train_loader = get_vqa_dataloader(
        DATA_ROOT, vocab, "train",
        batch_size=BATCH_SIZE,
        max_question_len=MAX_QUESTION_LEN,
    )
    val_loader = get_vqa_dataloader(
        DATA_ROOT, vocab, "val",
        batch_size=BATCH_SIZE,
        max_question_len=MAX_QUESTION_LEN,
    )
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # --- Model ---
    model = VQAModel(
        vocab_size=len(vocab),
        embed_dim=QUESTION_EMBED_DIM,
        gru_dim=GRU_DIM,
        attention_dim=ATTENTION_DIM,
        encoder_dim=ENCODER_DIM,
        dropout=DROPOUT,
        fine_tune_encoder=False,
    ).to(device)

    # --- Optimiser ---
    # Only train non-encoder params initially; encoder is frozen.
    decoder_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.")
    ]
    optimizer = torch.optim.Adam(decoder_params, lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    # --- Training loop ---
    best_val_acc = 0.0
    epochs_no_improve = 0
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    for epoch in range(1, NUM_EPOCHS + 1):
        # Enable encoder fine-tuning after warm-up
        if epoch == ENCODER_FINETUNE_EPOCH + 1:
            model.encoder._enable_fine_tune()
            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            optimizer.add_param_group({"params": encoder_params, "lr": ENCODER_LR})
            print(f"Epoch {epoch}: encoder fine-tuning enabled.")

        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            fine_tune_encoder=(epoch > ENCODER_FINETUNE_EPOCH),
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.0f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                 f"{val_loss:.6f}", f"{val_acc:.6f}", f"{current_lr:.2e}"]
            )

        # Checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "vocab_size": len(vocab),
        }
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  → New best val acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs with no improvement.")
                break

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {os.path.join(CHECKPOINT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
