"""
Training script for the adaptive-attention variant (Lu et al., CVPR 2017).

Mirrors train.py exactly except for three targeted changes:
  1. Uses AdaptiveDecoder instead of Decoder.
  2. Unpacks the 3-tuple (predictions, alphas, betas) from decoder.forward().
  3. Saves "model_type": "adaptive" in every checkpoint so evaluate.py and
     visualize.py can auto-detect which decoder class to load.
  4. Logs the mean sentinel weight (beta) per epoch alongside BLEU scores.

All hyperparameters, loss functions, scheduling, and early-stopping logic
are identical to train.py so results are directly comparable.
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
    ENCODER_FINETUNE_EPOCH,
    ENCODER_LR,
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
from models import Encoder
from models.adaptive_decoder import AdaptiveDecoder
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
    """
    Eq. 14: lambda * mean over (batch, locations) of (1 - sum_t alpha_ti)^2.

    alphas here are visual-only weights from AdaptiveAttention: each row sums to
    1 - beta_t rather than 1.  The loss therefore gently penalises high sentinel
    weight (encouraging the model to use the image) while still spreading
    attention across locations — a natural inductive bias for captioning.
    """
    return weight * ((1.0 - alphas.sum(dim=1)) ** 2).mean()


def train_epoch(encoder, decoder, dataloader, optimizer, criterion, device, epoch,
                fine_tune_encoder=False):
    """
    Run one training epoch.

    Loss = cross-entropy + λ * doubly stochastic regularisation (Eq. 14).

    Returns:
        avg_loss:     float
        avg_beta:     float — mean sentinel weight across the epoch (diagnostic)
    """
    if fine_tune_encoder:
        encoder.train()
    else:
        encoder.eval()
    decoder.train()

    total_loss = 0.0
    total_beta = 0.0
    n_batches  = 0

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

        # Adaptive decoder returns a 3-tuple
        predictions, alphas, betas = decoder(encoder_out, captions, lengths)

        targets = captions[:, 1:]   # shift left: exclude <start>
        vocab_size = predictions.size(-1)
        loss_ce = criterion(
            predictions.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        loss_ds = doubly_stochastic_attention_loss(alphas, LAMBDA)
        loss    = loss_ce + loss_ds

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        total_beta += float(betas.mean().item())
        n_batches  += 1
        loop.set_postfix(loss=f"{loss.item():.4f}", beta=f"{betas.mean().item():.3f}")

    return total_loss / max(n_batches, 1), total_beta / max(n_batches, 1)


@torch.no_grad()
def validate(encoder, decoder, dataloader, vocab, device):
    """
    Greedy-decode the validation set and compute BLEU scores.

    Uses greedy_decode_from_encoder_out which calls decoder.decode_step() —
    the same signature as base Decoder, so no changes needed here.
    """
    encoder.eval()
    decoder.eval()

    hypotheses = []
    references = []

    for images, captions, lengths, all_caps in tqdm(dataloader, desc="[val]", leave=False):
        images = images.to(device)
        encoder_out = encoder(images)
        batch_size  = images.size(0)

        for i in range(batch_size):
            ann = encoder_out[i].unsqueeze(0)
            caption, _, _ = greedy_decode_from_encoder_out(
                decoder, ann, vocab, device, max_len=MAX_DECODE_LEN
            )
            hypotheses.append(caption.split())
            refs_tokens = [tokenize_caption(cap) for cap in all_caps[i]]
            references.append(refs_tokens)

    return compute_bleu(hypotheses, references)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if os.path.exists(VOCAB_PATH):
        print("Loading vocabulary...")
        vocab = Vocabulary.load(VOCAB_PATH)
    else:
        print("Building vocabulary...")
        image_to_caps = load_flickr8k_captions(DATA_ROOT, "train")
        all_captions  = [cap for caps in image_to_caps.values() for cap in caps]
        vocab = Vocabulary(max_size=VOCAB_SIZE)
        vocab.build(all_captions)
        vocab.save(VOCAB_PATH)
        print(f"Vocabulary size: {len(vocab)}")

    num_workers  = min(4, os.cpu_count() or 1)
    train_loader = get_dataloader(DATA_ROOT, vocab, "train", BATCH_SIZE,
                                  num_workers=num_workers)
    val_loader   = get_dataloader(DATA_ROOT, vocab, "val",   BATCH_SIZE,
                                  num_workers=num_workers, use_bucket_sampler=False)

    encoder = Encoder(fine_tune=False).to(device)
    decoder = AdaptiveDecoder(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
        dropout=DROPOUT,
    ).to(device)

    optimizer = Adam(decoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_bleu4       = 0.0
    best_scores      = None
    epochs_no_improve = 0
    fine_tune_encoder = False

    log_path = os.path.join(RESULTS_DIR, "training_log_adaptive.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "mean_beta",
            "val_bleu1", "val_bleu2", "val_bleu3", "val_bleu4", "seconds",
        ])

    for epoch in range(1, NUM_EPOCHS + 1):

        if epoch == ENCODER_FINETUNE_EPOCH + 1 and not fine_tune_encoder:
            encoder._enable_fine_tune()
            optimizer.add_param_group({
                "params": [p for p in encoder.parameters() if p.requires_grad],
                "lr": ENCODER_LR,
            })
            fine_tune_encoder = True
            print(f"Epoch {epoch}: encoder fine-tuning enabled (LR={ENCODER_LR})")

        t0 = time.time()
        train_loss, mean_beta = train_epoch(
            encoder, decoder, train_loader, optimizer, criterion, device, epoch,
            fine_tune_encoder=fine_tune_encoder,
        )
        val_scores = validate(encoder, decoder, val_loader, vocab, device)
        val_bleu4  = val_scores["bleu4"]
        elapsed    = time.time() - t0
        scheduler.step(val_bleu4)

        print(
            f"Epoch {epoch:3d} | loss {train_loss:.4f} | mean_beta {mean_beta:.3f} | "
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
                f"{mean_beta:.6f}",
                f"{val_scores['bleu1']:.6f}",
                f"{val_scores['bleu2']:.6f}",
                f"{val_scores['bleu3']:.6f}",
                f"{val_scores['bleu4']:.6f}",
                f"{elapsed:.2f}",
            ])

        # Checkpoint — include model_type so evaluate.py / visualize.py auto-detect
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"adaptive_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch":      epoch,
            "model_type": "adaptive",
            "decoder":    decoder.state_dict(),
            "encoder":    encoder.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_scores": val_scores,
            "val_bleu4":  val_bleu4,
            "vocab_size": len(vocab),
        }, ckpt_path)

        if val_bleu4 > best_bleu4:
            best_bleu4   = val_bleu4
            best_scores  = val_scores
            epochs_no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_type": "adaptive",
                "decoder":    decoder.state_dict(),
                "encoder":    encoder.state_dict(),
                "val_scores": val_scores,
                "val_bleu4":  val_bleu4,
                "vocab_size": len(vocab),
            }, os.path.join(CHECKPOINT_DIR, "adaptive_best.pt"))
            print(f"  -> New best BLEU-4: {best_bleu4*100:.2f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs (no improvement for {PATIENCE}).")
                break

    print(f"Training complete. Best val BLEU-4: {best_bleu4*100:.2f}")
    if best_scores is not None:
        print_bleu_table("Adaptive Attention (best val checkpoint)", best_scores,
                         dataset="Validation")


if __name__ == "__main__":
    main()
