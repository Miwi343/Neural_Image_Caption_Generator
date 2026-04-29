"""
Evaluation and inference for the VQA yes/no model.

Usage — evaluate on val split:
    python -m vqa.evaluate --checkpoint checkpoints_vqa/best.pt

Usage — answer a single question about an image:
    python -m vqa.evaluate \\
        --checkpoint checkpoints_vqa/best.pt \\
        --image path/to/image.jpg \\
        --question "Is there a dog in this image?"
"""

import argparse
import os

import torch
from PIL import Image

from vqa.config import (
    ATTENTION_DIM,
    BATCH_SIZE,
    DATA_ROOT,
    DROPOUT,
    ENCODER_DIM,
    GRU_DIM,
    MAX_QUESTION_LEN,
    QUESTION_EMBED_DIM,
    VOCAB_PATH,
)
from vqa.dataset import (
    QuestionVocabulary,
    _EVAL_TRANSFORM,
    get_vqa_dataloader,
)
from vqa.model import VQAModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, vocab_size: int, device: torch.device) -> VQAModel:
    model = VQAModel(
        vocab_size=vocab_size,
        embed_dim=QUESTION_EMBED_DIM,
        gru_dim=GRU_DIM,
        attention_dim=ATTENTION_DIM,
        encoder_dim=ENCODER_DIM,
        dropout=DROPOUT,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val(
    checkpoint_path: str,
    data_root: str = DATA_ROOT,
    vocab_path: str = VOCAB_PATH,
    batch_size: int = BATCH_SIZE,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = QuestionVocabulary.load(vocab_path)
    model = load_model(checkpoint_path, len(vocab), device)

    loader = get_vqa_dataloader(
        data_root, vocab, "val",
        batch_size=batch_size,
        max_question_len=MAX_QUESTION_LEN,
    )

    correct = 0
    total = 0
    yes_correct = 0
    yes_total = 0
    no_correct = 0
    no_total = 0

    for images, questions, q_lens, labels in loader:
        images    = images.to(device)
        questions = questions.to(device)
        q_lens    = q_lens.to(device)
        labels    = labels.to(device)

        logits, _ = model(images, questions, q_lens)
        preds = (torch.sigmoid(logits) >= 0.5).float()

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        yes_mask = labels == 1
        no_mask  = labels == 0
        yes_correct += (preds[yes_mask] == labels[yes_mask]).sum().item()
        yes_total   += yes_mask.sum().item()
        no_correct  += (preds[no_mask] == labels[no_mask]).sum().item()
        no_total    += no_mask.sum().item()

    results = {
        "overall_acc": correct / total,
        "yes_acc":     yes_correct / yes_total if yes_total else 0.0,
        "no_acc":      no_correct  / no_total  if no_total  else 0.0,
        "total":       total,
    }
    print(f"Overall accuracy : {results['overall_acc']:.4f}  ({correct}/{total})")
    print(f"Yes accuracy     : {results['yes_acc']:.4f}  ({yes_correct}/{yes_total})")
    print(f"No  accuracy     : {results['no_acc']:.4f}  ({no_correct}/{no_total})")
    return results


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def answer_question(
    checkpoint_path: str,
    image_path: str,
    question: str,
    vocab_path: str = VOCAB_PATH,
) -> dict:
    """
    Answer a yes/no question about a single image.

    Returns:
        {
          "answer":      "yes" or "no",
          "confidence":  float in [0, 1]  — P(yes),
          "alpha":       (196,) numpy array of attention weights,
        }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = QuestionVocabulary.load(vocab_path)
    model = load_model(checkpoint_path, len(vocab), device)

    image = Image.open(image_path).convert("RGB")
    image_tensor = _EVAL_TRANSFORM(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    ids = vocab.encode(question, max_len=MAX_QUESTION_LEN)
    q_len = len(ids)
    ids += [0] * (MAX_QUESTION_LEN - q_len)
    q_tensor = torch.tensor([ids], dtype=torch.long).to(device)   # (1, max_q_len)
    q_len_t  = torch.tensor([q_len], dtype=torch.long).to(device)  # (1,)

    logit, alpha = model(image_tensor, q_tensor, q_len_t)
    prob = torch.sigmoid(logit).item()
    answer = "yes" if prob >= 0.5 else "no"

    print(f"Q: {question}")
    print(f"A: {answer}  (confidence: {prob:.3f})")

    return {
        "answer":     answer,
        "confidence": prob,
        "alpha":      alpha.squeeze(0).cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate or run VQA yes/no model")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_root",  default=DATA_ROOT)
    parser.add_argument("--vocab_path", default=VOCAB_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    # Single-image mode
    parser.add_argument("--image",    default=None, help="Path to a single image")
    parser.add_argument("--question", default=None, help="Yes/no question to answer")
    args = parser.parse_args()

    if args.image and args.question:
        answer_question(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            question=args.question,
            vocab_path=args.vocab_path,
        )
    else:
        evaluate_val(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            vocab_path=args.vocab_path,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
