"""
VQA v2 yes/no dataset loader.

Only yes/no questions are kept (~38% of VQA v2).
Answer is mapped to 0 (no) or 1 (yes).

Images are the same COCO images already on disk at data/vqa/images/.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

PAD_IDX = 0
UNK_IDX = 1

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_PUNCT_RE = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> List[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()


class QuestionVocabulary:
    def __init__(self, max_size: int = 15_000):
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: "<pad>", UNK_IDX: "<unk>"}

    def build(self, questions: List[str]) -> None:
        counter: Counter = Counter()
        for q in questions:
            counter.update(_tokenize(q))
        for idx, (word, _) in enumerate(counter.most_common(self.max_size), start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, question: str, max_len: int = 20) -> List[int]:
        tokens = _tokenize(question)[:max_len]
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx}, f)

    @classmethod
    def load(cls, path: str) -> "QuestionVocabulary":
        with open(path) as f:
            data = json.load(f)
        v = cls()
        v.word2idx = {w: int(i) for w, i in data["word2idx"].items()}
        v.idx2word = {int(i): w for w, i in data["word2idx"].items()}
        return v


def _load_yes_no_pairs(ann_path: str, q_path: str) -> List[Tuple[int, str, int]]:
    with open(ann_path) as f:
        anns = {a["question_id"]: a for a in json.load(f)["annotations"]}
    with open(q_path) as f:
        questions = {q["question_id"]: q for q in json.load(f)["questions"]}

    pairs = []
    for qid, ann in anns.items():
        if ann["answer_type"] != "yes/no":
            continue
        answer = ann["multiple_choice_answer"].strip().lower()
        if answer not in ("yes", "no"):
            continue
        pairs.append((ann["image_id"], questions[qid]["question"], 1 if answer == "yes" else 0))
    return pairs


def _coco_image_path(images_dir: str, split: str, image_id: int) -> str:
    return os.path.join(images_dir, f"{split}2014", f"COCO_{split}2014_{image_id:012d}.jpg")


class VQAYesNoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        vocab: QuestionVocabulary,
        split: str = "train",
        max_question_len: int = 20,
        max_samples: Optional[int] = None,
    ):
        assert split in ("train", "val")
        self.vocab = vocab
        self.max_q = max_question_len
        self.images_dir = os.path.join(data_root, "images")
        self.transform = _TRAIN_TRANSFORM if split == "train" else _EVAL_TRANSFORM

        coco_split = "train" if split == "train" else "val"
        ann_path = os.path.join(data_root, f"v2_mscoco_{coco_split}2014_annotations.json")
        q_path   = os.path.join(data_root, f"v2_OpenEnded_mscoco_{coco_split}2014_questions.json")
        all_pairs = _load_yes_no_pairs(ann_path, q_path)

        self.pairs = [p for p in all_pairs if os.path.exists(_coco_image_path(self.images_dir, coco_split, p[0]))]
        kept, total = len(self.pairs), len(all_pairs)
        print(f"[VQAYesNoDataset/{split}] {kept:,} / {total:,} pairs kept ({total-kept:,} missing images skipped)")

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_id, question_str, label = self.pairs[idx]
        coco_split = "train" if self.transform is _TRAIN_TRANSFORM else "val"
        img_path = _coco_image_path(self.images_dir, coco_split, image_id)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        image = self.transform(image)

        ids = self.vocab.encode(question_str, max_len=self.max_q)
        q_len = len(ids)
        ids += [PAD_IDX] * (self.max_q - q_len)

        return (
            image,
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(q_len, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def get_vqa_dataloader(
    data_root: str,
    vocab: QuestionVocabulary,
    split: str,
    batch_size: int = 64,
    num_workers: int = 4,
    max_question_len: int = 20,
    max_samples: Optional[int] = None,
) -> DataLoader:
    dataset = VQAYesNoDataset(data_root, vocab, split, max_question_len, max_samples)

    def collate(batch):
        images, questions, q_lens, labels = zip(*batch)
        return torch.stack(images), torch.stack(questions), torch.stack(q_lens), torch.stack(labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def build_and_save_vocab(data_root: str, vocab_path: str, max_size: int = 15_000) -> QuestionVocabulary:
    q_path = os.path.join(data_root, "v2_OpenEnded_mscoco_train2014_questions.json")
    with open(q_path) as f:
        questions = [q["question"] for q in json.load(f)["questions"]]
    vocab = QuestionVocabulary(max_size=max_size)
    vocab.build(questions)
    vocab.save(vocab_path)
    print(f"Built vocabulary: {len(vocab)} tokens → {vocab_path}")
    return vocab
