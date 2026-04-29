"""
VQA v2 yes/no dataset loader.

Data layout after setup (see setup_data() below):
    data/vqa/
        images/train2014/   ← COCO train images
        images/val2014/     ← COCO val images
        v2_mscoco_train2014_annotations.json
        v2_mscoco_val2014_annotations.json
        v2_OpenEnded_mscoco_train2014_questions.json
        v2_OpenEnded_mscoco_val2014_questions.json
        vocab.json          ← built by build_and_save_vocab()

Only yes/no questions are kept (~38% of VQA v2, ~250k train questions).
Answer is mapped to 0 (no) or 1 (yes).

Kaggle download (run once in Colab):
    from vqa.dataset import setup_data
    setup_data("data/vqa")
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Special token indices (same convention as caption Vocabulary)
# ---------------------------------------------------------------------------
PAD_IDX = 0
UNK_IDX = 1


# ---------------------------------------------------------------------------
# Image transforms — identical to caption model so encoder weights transfer
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class QuestionVocabulary:
    """
    Word ↔ index mapping for question tokens.
    Uses the same save/load JSON format as the caption Vocabulary so the two
    are interchangeable in scripts.
    """

    def __init__(self, max_size: int = 15_000):
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: "<pad>", UNK_IDX: "<unk>"}
        self._built = False

    def build(self, questions: List[str]) -> None:
        counter: Counter = Counter()
        for q in questions:
            counter.update(_tokenize(q))
        for idx, (word, _) in enumerate(counter.most_common(self.max_size), start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self._built = True

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
        v._built = True
        return v


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_yes_no_pairs(
    ann_path: str,
    q_path: str,
) -> List[Tuple[int, str, int]]:
    """
    Parse annotation + question JSON files and return yes/no pairs.

    Returns:
        list of (image_id, question_str, label)
        where label is 1 for "yes" and 0 for "no".
    """
    with open(ann_path) as f:
        anns = {a["question_id"]: a for a in json.load(f)["annotations"]}
    with open(q_path) as f:
        questions = {q["question_id"]: q for q in json.load(f)["questions"]}

    pairs = []
    for qid, ann in anns.items():
        if ann["answer_type"] != "yes/no":
            continue
        # Majority vote answer — already stored as a plain string
        answer = ann["multiple_choice_answer"].strip().lower()
        if answer not in ("yes", "no"):
            continue
        label = 1 if answer == "yes" else 0
        image_id = ann["image_id"]
        question_str = questions[qid]["question"]
        pairs.append((image_id, question_str, label))

    return pairs


def _coco_image_path(images_dir: str, split: str, image_id: int) -> str:
    """Build the path to a COCO image given its integer ID."""
    # COCO filenames: COCO_train2014_000000123456.jpg
    filename = f"COCO_{split}2014_{image_id:012d}.jpg"
    return os.path.join(images_dir, f"{split}2014", filename)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VQAYesNoDataset(Dataset):
    """
    VQA v2 yes/no binary classification dataset.

    Each sample is (image_tensor, question_tensor, question_len, label).
      image_tensor:    (3, 224, 224)
      question_tensor: (MAX_QUESTION_LEN,) padded with PAD_IDX
      question_len:    scalar int — actual token count (before padding)
      label:           0 or 1 (no / yes)

    Args:
        data_root:        path to data/vqa/
        vocab:            QuestionVocabulary instance
        split:            "train" or "val"
        max_question_len: truncate questions to this many tokens
        max_samples:      if set, subsample the dataset (useful for quick debug runs)
    """

    def __init__(
        self,
        data_root: str,
        vocab: QuestionVocabulary,
        split: str = "train",
        max_question_len: int = 20,
        max_samples: Optional[int] = None,
    ):
        assert split in ("train", "val"), f"Unknown split: {split!r}"
        self.split = split
        self.vocab = vocab
        self.max_q = max_question_len
        self.images_dir = os.path.join(data_root, "images")
        self.transform = _TRAIN_TRANSFORM if split == "train" else _EVAL_TRANSFORM

        coco_split = "train" if split == "train" else "val"
        ann_path = os.path.join(
            data_root,
            f"v2_mscoco_{coco_split}2014_annotations.json",
        )
        q_path = os.path.join(
            data_root,
            f"v2_OpenEnded_mscoco_{coco_split}2014_questions.json",
        )
        self.pairs = _load_yes_no_pairs(ann_path, q_path)

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_id, question_str, label = self.pairs[idx]

        # Determine which COCO split owns this image_id.
        # Val2014 images are used for both VQA val and test; train2014 for train.
        coco_split = "train" if self.split == "train" else "val"
        img_path = _coco_image_path(self.images_dir, coco_split, image_id)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        ids = self.vocab.encode(question_str, max_len=self.max_q)
        q_len = len(ids)
        # Pad to max_question_len
        ids += [PAD_IDX] * (self.max_q - q_len)
        question_tensor = torch.tensor(ids, dtype=torch.long)
        q_len_tensor = torch.tensor(q_len, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image, question_tensor, q_len_tensor, label_tensor


# ---------------------------------------------------------------------------
# Collate + DataLoader
# ---------------------------------------------------------------------------

def _collate_fn(batch):
    images, questions, q_lens, labels = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(questions),
        torch.stack(q_lens),
        torch.stack(labels),
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
    dataset = VQAYesNoDataset(
        data_root, vocab, split,
        max_question_len=max_question_len,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Vocabulary builder (call once after downloading annotations)
# ---------------------------------------------------------------------------

def build_and_save_vocab(data_root: str, vocab_path: str, max_size: int = 15_000) -> QuestionVocabulary:
    """
    Build a QuestionVocabulary from training questions and save it.
    Only needs to be called once; subsequent runs load from vocab_path.
    """
    q_path = os.path.join(data_root, "v2_OpenEnded_mscoco_train2014_questions.json")
    with open(q_path) as f:
        questions = [q["question"] for q in json.load(f)["questions"]]
    vocab = QuestionVocabulary(max_size=max_size)
    vocab.build(questions)
    vocab.save(vocab_path)
    print(f"Built vocabulary: {len(vocab)} tokens → {vocab_path}")
    return vocab


# ---------------------------------------------------------------------------
# One-time Kaggle data setup (run in Colab before training)
# ---------------------------------------------------------------------------

def setup_data(data_root: str = "data/vqa") -> None:
    """
    Download VQA v2 annotations and COCO images via the Kaggle API.

    Requires:
        - kaggle package installed  (pip install kaggle)
        - ~/.kaggle/kaggle.json with your API credentials
          (or KAGGLE_USERNAME / KAGGLE_KEY env vars set)

    Downloads:
        - VQA v2 annotation JSONs (~80 MB)
        - COCO train2014 images (~13 GB) — only needed for training
        - COCO val2014 images   (~6 GB)

    The annotation JSONs are small enough to download unconditionally.
    Image downloads are gated so you can skip them if you already have the
    images mounted (e.g. from a Kaggle dataset or Google Drive).
    """
    import subprocess

    os.makedirs(data_root, exist_ok=True)
    ann_dir = data_root
    img_dir = os.path.join(data_root, "images")
    os.makedirs(img_dir, exist_ok=True)

    def _kaggle(cmd: str) -> None:
        print(f"[kaggle] {cmd}")
        subprocess.run(f"kaggle {cmd}", shell=True, check=True)

    # --- Annotations + questions ---
    # VQA v2 official annotations are available as a Kaggle dataset.
    # We download just the annotation/question JSON files (no images).
    print("Downloading VQA v2 annotation JSONs …")
    _kaggle(
        f"datasets download -d gcm100/vqa-v2 "
        f"--path {ann_dir} "
        f"--unzip "
        f"-f v2_mscoco_train2014_annotations.json"
    )
    _kaggle(
        f"datasets download -d gcm100/vqa-v2 "
        f"--path {ann_dir} "
        f"--unzip "
        f"-f v2_mscoco_val2014_annotations.json"
    )
    _kaggle(
        f"datasets download -d gcm100/vqa-v2 "
        f"--path {ann_dir} "
        f"--unzip "
        f"-f v2_OpenEnded_mscoco_train2014_questions.json"
    )
    _kaggle(
        f"datasets download -d gcm100/vqa-v2 "
        f"--path {ann_dir} "
        f"--unzip "
        f"-f v2_OpenEnded_mscoco_val2014_questions.json"
    )

    # --- COCO images ---
    # COCO 2014 images are ~19 GB total.  We download from a Kaggle mirror.
    # If you already have them mounted, set the env var SKIP_IMAGE_DOWNLOAD=1.
    if os.environ.get("SKIP_IMAGE_DOWNLOAD", "0") == "1":
        print("SKIP_IMAGE_DOWNLOAD=1 — skipping image download.")
        return

    print("Downloading COCO train2014 images (~13 GB) …")
    _kaggle(
        f"datasets download -d awsaf49/coco-2017-dataset "
        f"--path {img_dir} "
        f"--unzip "
        f"-f train2017.zip"
    )
    # Rename train2017 → train2014 if needed (some Kaggle mirrors use 2017 names)
    train17 = os.path.join(img_dir, "train2017")
    train14 = os.path.join(img_dir, "train2014")
    if os.path.isdir(train17) and not os.path.isdir(train14):
        os.rename(train17, train14)
        print(f"Renamed {train17} → {train14}")

    print("Downloading COCO val2014 images (~6 GB) …")
    _kaggle(
        f"datasets download -d awsaf49/coco-2017-dataset "
        f"--path {img_dir} "
        f"--unzip "
        f"-f val2017.zip"
    )
    val17 = os.path.join(img_dir, "val2017")
    val14 = os.path.join(img_dir, "val2014")
    if os.path.isdir(val17) and not os.path.isdir(val14):
        os.rename(val17, val14)
        print(f"Renamed {val17} → {val14}")

    print(f"Data setup complete. Root: {data_root}")
