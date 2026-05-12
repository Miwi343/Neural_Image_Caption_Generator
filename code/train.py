"""Train the Show, Attend and Tell captioning model on Flickr8k."""

import csv
import os
import time

import torch
import torch.nn as nn
from torch.optim import RMSprop
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
    """Doubly stochastic attention penalty, averaged over the batch."""
    return weight * ((1.0 - alphas.sum(dim=1)) ** 2).sum(dim=1).mean()


def train_epoch(encoder, decoder, dataloader, optimizer, criterion, device, epoch,
                fine_tune_encoder=False):
    """Run one training epoch and return the average loss."""
    if fine_tune_encoder:
        encoder.train()
    else:
        encoder.eval()
    decoder.train()

    total_loss = 0.0
    batch_count = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, captions, lengths, _ in loop:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        if fine_tune_encoder:
            encoder_out = encoder(images)
        else:
            with torch.no_grad():
                encoder_out = encoder(images)

        predictions, alphas = decoder(encoder_out, captions, lengths)
        targets = captions[:, 1:]

        vocab_size = predictions.size(-1)
        ce_loss = criterion(
            predictions.reshape(-1, vocab_size),
            targets.reshape(-1),
        )
        attention_loss = doubly_stochastic_attention_loss(alphas, LAMBDA)
        loss = ce_loss + attention_loss

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(batch_count, 1)


@torch.no_grad()
def validate(encoder, decoder, dataloader, vocab, device):
    """Greedy-decode the validation set and return BLEU scores."""
    encoder.eval()
    decoder.eval()

    hypotheses = []
    references = []

    for images, _, _, all_caps in tqdm(dataloader, desc="[val]", leave=False):
        images = images.to(device)

        encoder_out = encoder(images)
        batch_size = images.size(0)

        for i in range(batch_size):
            ann = encoder_out[i].unsqueeze(0)
            caption, _, _ = greedy_decode_from_encoder_out(
                decoder, ann, vocab, device, max_len=MAX_DECODE_LEN
            )
            hyp_tokens = caption.split()
            hypotheses.append(hyp_tokens)

            # Keep raw references for BLEU; vocabulary encoding would add <unk>.
            refs_tokens = [tokenize_caption(cap) for cap in all_caps[i]]
            references.append(refs_tokens)

    scores = compute_bleu(hypotheses, references)
    return scores


def main():
    """Build the data, train the model, and save checkpoints."""
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
        all_captions = [cap for caps in image_to_caps.values() for cap in caps]
        vocab = Vocabulary(max_size=VOCAB_SIZE)
        vocab.build(all_captions)
        vocab.save(VOCAB_PATH)
        print(f"Vocabulary size: {len(vocab)}")

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = get_dataloader(
        DATA_ROOT, vocab, "train", BATCH_SIZE, num_workers=num_workers
    )
    val_loader = get_dataloader(
        DATA_ROOT,
        vocab,
        "val",
        BATCH_SIZE,
        num_workers=num_workers,
        use_bucket_sampler=False,
    )

    encoder = Encoder(fine_tune=False).to(device)
    decoder = Decoder(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
        dropout=DROPOUT,
    ).to(device)

    optimizer = RMSprop(decoder.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_bleu4 = 0.0
    best_scores = None
    epochs_no_improve = 0
    fine_tune_encoder = False
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_bleu1",
            "val_bleu2",
            "val_bleu3",
            "val_bleu4",
            "seconds",
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
        train_loss = train_epoch(
            encoder, decoder, train_loader, optimizer, criterion, device, epoch,
            fine_tune_encoder=fine_tune_encoder,
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
    import argparse
    parser = argparse.ArgumentParser(description="Train Show, Attend and Tell")
    parser.add_argument("--lambda_weight", type=float, default=LAMBDA,
                        help=f"Doubly stochastic attention regularisation weight (default: {LAMBDA})")
    parser.add_argument("--finetune_epoch", type=int, default=ENCODER_FINETUNE_EPOCH,
                        help=f"Epoch to start encoder fine-tuning (default: {ENCODER_FINETUNE_EPOCH})")
    args = parser.parse_args()

    LAMBDA = args.lambda_weight
    ENCODER_FINETUNE_EPOCH = args.finetune_epoch

    main()
