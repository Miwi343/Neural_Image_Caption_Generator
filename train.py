"""
Training script for Show, Attend and Tell (Xu et al., 2015).

Implements:
  - Cross-entropy loss over predicted word distribution
  - Doubly stochastic regularisation (Eq. 14): λ * Σ_i (1 - Σ_t α_ti)²
  - Gradient clipping (5.0) to stabilise LSTM training
  - Adam optimiser on decoder parameters only (encoder is frozen)
  - Per-epoch checkpointing + early stopping on validation BLEU-4
  - Batch length-bucketing for training speed (paper section 4.3)

Key hyperparameters (paper / standard community values):
  EMBED_DIM      = 512   (word embedding dimension m)
  DECODER_DIM    = 512   (LSTM hidden size n)
  ATTENTION_DIM  = 512   (attention MLP hidden size)
  DROPOUT        = 0.5
  VOCAB_SIZE     = 10000
  LAMBDA         = 1.0   (doubly stochastic regularisation weight)
  GRAD_CLIP      = 5.0
  BATCH_SIZE     = 64
  NUM_EPOCHS     = 50
  PATIENCE       = 5     (early stop if BLEU-4 doesn't improve)

Future work (Issue #7, deferred): If training stability work is assigned, add
`ReduceLROnPlateau` keyed on validation BLEU-4 and log exactly when the LR
changes so runs remain comparable.
Future work (Issue #8, deferred): If scaling beyond one GPU becomes necessary, wrap the
model with DDP/DataParallel only after single-GPU training is stable and verify
that checkpoint loading + BLEU parity still hold on 1 GPU.
Future work (Issue #9, deferred): If experiment tracking is needed for the report, log
loss and BLEU curves to TensorBoard or W&B without changing the default CLI-free
training path.
"""

import csv
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from config import (
    ATTENTION_DIM,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_ROOT,
    DECODER_DIM,
    DROPOUT,
    EMBED_DIM,
    GRAD_CLIP,
    LAMBDA,
    MAX_DECODE_LEN,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    RESULTS_DIR,
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

# Future work (Issue #7, deferred): If run management gets more complex, replace these
# module-level constants with argparse flags while keeping the current values as
# defaults for a drop-in baseline run.


def doubly_stochastic_attention_loss(alphas: torch.Tensor, weight: float = LAMBDA):
    """Eq. 14: lambda * mean_batch(sum_i(1 - sum_t alpha_ti)^2)."""
    return weight * ((1.0 - alphas.sum(dim=1)) ** 2).sum(dim=1).mean()


def train_epoch(encoder, decoder, dataloader, optimizer, criterion, device, epoch):
    """
    Run one training epoch.

    Loss = cross-entropy + λ * doubly stochastic regularisation (Eq. 14).

    Args:
        encoder:    Encoder (frozen)
        decoder:    Decoder (trained)
        dataloader: training DataLoader
        optimizer:  Adam on decoder params
        criterion:  CrossEntropyLoss(ignore_index=PAD_IDX)
        device:     torch.device
        epoch:      current epoch number (for logging)

    Returns:
        avg_loss: float

    Future work (Issue #7, deferred): If extra training diagnostics are useful, compute
    top-5 next-word accuracy from `predictions` vs `targets` and report it
    alongside loss without affecting optimisation.
    Future work (Issue #9, deferred): If TensorBoard/W&B logging is enabled, emit
    per-step and per-epoch loss here using stable metric names.
    """
    encoder.eval()   # encoder stays frozen — no batch norm updates needed
    decoder.train()

    total_loss = 0.0
    n_batches = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, captions, lengths, _ in loop:
        images   = images.to(device)
        captions = captions.to(device)
        lengths  = lengths.to(device)

        # Encode: (batch, 196, 512)
        with torch.no_grad():
            encoder_out = encoder(images)

        # Decode: predictions (batch, max_len-1, vocab_size)
        #         alphas      (batch, max_len-1, 196)
        predictions, alphas = decoder(encoder_out, captions, lengths)

        # Targets are captions shifted left by one (exclude <start>)
        # Shape: (batch * (max_len-1),)
        # Future work (Issue #7): Add a tiny regression test confirming that
        # `captions[:, 1:]` is the correct next-token target for the decoder
        # outputs produced from inputs that still include `<start>`.
        targets = captions[:, 1:]   # (batch, max_len-1)

        # Flatten for cross-entropy
        vocab_size = predictions.size(-1)
        loss_ce = criterion(
            predictions.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        # Doubly stochastic attention regularisation (Eq. 14).
        loss_ds = doubly_stochastic_attention_loss(alphas, LAMBDA)

        loss = loss_ce + loss_ds

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent LSTM exploding gradients
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
        dict with keys `bleu1`/`bleu2`/`bleu3`/`bleu4`; `bleu4` remains the
        checkpoint-selection metric.
    Future work (Issue #10, deferred): If a beam-search comparison is needed for the
    report, add it as an explicit alternate validation path; do not replace the
    current greedy baseline silently.
    """
    encoder.eval()
    decoder.eval()

    hypotheses = []
    references = []

    for images, captions, lengths, all_caps in tqdm(dataloader, desc="[val]", leave=False):
        images = images.to(device)

        encoder_out = encoder(images)   # (batch, 196, 512)
        batch_size = images.size(0)

        # Greedy decode — one sample at a time for simplicity
        # Future work (Issue #10, deferred): If validation throughput becomes a bottleneck,
        # vectorize greedy decoding across the batch and verify hypotheses remain
        # identical to the current per-image implementation.
        for i in range(batch_size):
            ann = encoder_out[i].unsqueeze(0)  # (1, 196, 512)
            caption, _, _ = greedy_decode_from_encoder_out(
                decoder, ann, vocab, device, max_len=MAX_DECODE_LEN
            )
            hyp_tokens = caption.split()
            hypotheses.append(hyp_tokens)

            # All 5 reference captions for this image — tokenise directly,
            # never route through the vocabulary (that would corrupt any word
            # outside the top-10k with <unk> and break BLEU scoring).
            refs_tokens = [tokenize_caption(cap) for cap in all_caps[i]]
            references.append(refs_tokens)

    scores = compute_bleu(hypotheses, references)
    return scores


def main():
    """
    Full training loop with checkpointing and early stopping.

    Future work (Issue #7, deferred): If CLI usage expands, expose `data_root`,
    `checkpoint_dir`, and key hyperparameters via argparse while preserving the
    current constant defaults.
    Future work (Issue #8, deferred): If resume support is needed, load epoch/model/
    optimizer state from a checkpoint and resume the early-stopping counters
    without changing the non-resume path.
    Future work (Issue #9, deferred): If report generation is scripted, print a final
    BLEU summary table at the end of training using the last validation scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build or load vocabulary
    # Future work (Issue #2, deferred): If dataset setup is split into multiple scripts,
    # move vocab construction into a dedicated command and keep this fallback
    # path for small one-shot training runs.
    if os.path.exists(VOCAB_PATH):
        print("Loading vocabulary...")
        vocab = Vocabulary.load(VOCAB_PATH)
    else:
        print("Building vocabulary...")
        image_to_caps = load_flickr8k_captions(DATA_ROOT, "train")
        all_captions = [cap for caps in image_to_caps.values() for cap in caps]
        vocab = Vocabulary(max_size=VOCAB_SIZE)
        vocab.build(all_captions)
        vocab.save(VOCAB_PATH)
        print(f"Vocabulary size: {len(vocab)}")

    # Dataloaders
    train_loader = get_dataloader(DATA_ROOT, vocab, "train", BATCH_SIZE)
    val_loader   = get_dataloader(DATA_ROOT, vocab, "val",   BATCH_SIZE,
                                  use_bucket_sampler=False)

    # Models
    encoder = Encoder(fine_tune=False).to(device)
    decoder = Decoder(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
        dropout=DROPOUT,
    ).to(device)

    # Optimise only the decoder (encoder is frozen)
    # Future work (Issue #7, deferred): If encoder fine-tuning is enabled later,
    # introduce a second optimizer param group with a lower LR after the chosen
    # warm-up epoch and document the schedule clearly.
    optimizer = Adam(decoder.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_bleu4 = 0.0
    best_scores = None
    epochs_no_improve = 0
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_bleu1", "val_bleu2", "val_bleu3", "val_bleu4", "seconds"])

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(
            encoder, decoder, train_loader, optimizer, criterion, device, epoch
        )
        val_scores = validate(encoder, decoder, val_loader, vocab, device)
        val_bleu4 = val_scores["bleu4"]
        elapsed = time.time() - t0

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

        # Checkpoint every epoch
        # Future work (Issue #9, deferred): If resume-from-best is required, keep saving
        # optimizer state here and mirror it into `best.pt` as well so both
        # checkpoint formats remain usable.
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "decoder": decoder.state_dict(),
            "encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_scores": val_scores,
            "val_bleu4": val_bleu4,
            "vocab_size": len(vocab),
        }, ckpt_path)

        # Track best model
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
            }, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  -> New best BLEU-4: {best_bleu4*100:.2f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs (no improvement for {PATIENCE}).")
                break

    print(f"Training complete. Best val BLEU-4: {best_bleu4*100:.2f}")
    if best_scores is not None:
        print_bleu_table("Best validation checkpoint", best_scores, dataset="Validation")


if __name__ == "__main__":
    main()
