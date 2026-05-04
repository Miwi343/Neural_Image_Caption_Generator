"""
Training script for the VQA yes/no model.

Usage:
    python -m vqa.train
    python -m vqa.train --lr 1e-3 --dropout 0.3 --tag experiment1
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from vqa.dataset import QuestionVocabulary, build_and_save_vocab, get_vqa_dataloader
from vqa.model import VQAModel

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DATA_ROOT            = "data/vqa"
VOCAB_PATH           = "data/vqa/vocab.json"
CHECKPOINT_DIR       = "checkpoints_vqa"
RESULTS_DIR          = "results_vqa"

ENCODER_DIM          = 512
QUESTION_EMBED_DIM   = 512
GRU_DIM              = 512
ATTENTION_DIM        = 512

LEARNING_RATE        = 3e-4
ENCODER_LR           = 1e-4
ENCODER_FINETUNE_EPOCH = 5
DROPOUT              = 0.5
BATCH_SIZE           = 128
NUM_EPOCHS           = 15
PATIENCE             = 5
GRAD_CLIP            = 5.0
MAX_QUESTION_LEN     = 20
VOCAB_SIZE           = 15_000
# Doubly-stochastic regularisation weight (paper Eq. 14)
# Single attention step so penalty is trivially 0, but kept for consistency.
LAMBDA               = 1.0


def _ds_loss(alpha: torch.Tensor) -> torch.Tensor:
    """Doubly-stochastic regularisation (paper Eq. 14), single-step variant."""
    return LAMBDA * ((1.0 - alpha.sum(dim=1)) ** 2).mean()


def train_epoch(model, loader, optimizer, criterion, device, epoch, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for images, questions, q_lens, labels in pbar:
        images, questions, q_lens, labels = (
            images.to(device), questions.to(device), q_lens.to(device), labels.to(device)
        )
        logits, alpha = model(images, questions, q_lens)
        loss = criterion(logits, labels) + _ds_loss(alpha)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
        total      += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, questions, q_lens, labels in tqdm(loader, desc="Val", leave=False):
        images, questions, q_lens, labels = (
            images.to(device), questions.to(device), q_lens.to(device), labels.to(device)
        )
        logits, _ = model(images, questions, q_lens)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct    += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
        total      += images.size(0)
    return (total_loss / total, correct / total) if total else (float("nan"), float("nan"))


def main(cfg: dict | None = None):
    c = cfg or {}
    lr                  = c.get("lr",                   LEARNING_RATE)
    encoder_lr          = c.get("encoder_lr",           ENCODER_LR)
    dropout             = c.get("dropout",              DROPOUT)
    batch_size          = c.get("batch_size",           BATCH_SIZE)
    num_epochs          = c.get("num_epochs",           NUM_EPOCHS)
    patience            = c.get("patience",             PATIENCE)
    grad_clip           = c.get("grad_clip",            GRAD_CLIP)
    finetune_ep         = c.get("encoder_finetune_epoch", ENCODER_FINETUNE_EPOCH)
    tag                 = c.get("tag", "")

    suffix         = f"_{tag}" if tag else ""
    checkpoint_dir = c.get("checkpoint_dir", CHECKPOINT_DIR) + suffix
    results_dir    = c.get("results_dir",    RESULTS_DIR)    + suffix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nRun: {tag or 'vanilla'}")
    print(f"  lr={lr}  encoder_lr={encoder_lr}  dropout={dropout}  batch_size={batch_size}")
    print(f"  Device: {device}\n{'='*60}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

    if os.path.exists(VOCAB_PATH):
        vocab = QuestionVocabulary.load(VOCAB_PATH)
        print(f"Loaded vocab: {len(vocab)} tokens")
    else:
        vocab = build_and_save_vocab(DATA_ROOT, VOCAB_PATH, max_size=VOCAB_SIZE)

    train_loader = get_vqa_dataloader(DATA_ROOT, vocab, "train", batch_size, max_question_len=MAX_QUESTION_LEN)
    val_loader   = get_vqa_dataloader(DATA_ROOT, vocab, "val",   batch_size, max_question_len=MAX_QUESTION_LEN)
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model = VQAModel(
        vocab_size=len(vocab),
        embed_dim=QUESTION_EMBED_DIM,
        gru_dim=GRU_DIM,
        attention_dim=ATTENTION_DIM,
        encoder_dim=ENCODER_DIM,
        dropout=dropout,
        fine_tune_encoder=False,
    ).to(device)

    decoder_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("encoder.")]
    optimizer = torch.optim.Adam(decoder_params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc     = 0.0
    epochs_no_improve = 0
    log_path = os.path.join(results_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    for epoch in range(1, num_epochs + 1):
        if epoch == finetune_ep + 1:
            model.encoder._enable_fine_tune()
            optimizer.add_param_group({"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": encoder_lr})
            print(f"Epoch {epoch}: encoder fine-tuning enabled.")

        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, grad_clip)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{num_epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} | lr {current_lr:.2e} | {elapsed:.0f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{current_lr:.2e}"])

        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_acc": val_acc, "vocab_size": len(vocab), "cfg": c}
        torch.save(ckpt, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(ckpt, os.path.join(checkpoint_dir, "best.pt"))
            print(f"  → New best val acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {patience} epochs with no improvement.")
                break

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",                     type=float, default=None)
    parser.add_argument("--encoder_lr",             type=float, default=None)
    parser.add_argument("--dropout",                type=float, default=None)
    parser.add_argument("--batch_size",             type=int,   default=None)
    parser.add_argument("--num_epochs",             type=int,   default=None)
    parser.add_argument("--patience",               type=int,   default=None)
    parser.add_argument("--encoder_finetune_epoch", type=int,   default=None)
    parser.add_argument("--tag",                    type=str,   default="")
    args = parser.parse_args()
    main({k: v for k, v in vars(args).items() if v is not None})
