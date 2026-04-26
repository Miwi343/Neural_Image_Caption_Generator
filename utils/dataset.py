"""
Dataset utilities for Flickr8k.

Covers:
  - Vocabulary: token ↔ index mapping, build from captions, save/load JSON.
  - Flickr8kDataset: loads images + captions, 6000/1000/1000 split.
  - LengthBucketSampler: groups batches by caption length to cut padding waste
    (paper section 4.3: "we build a dictionary mapping the length of a sentence
    to the corresponding subset of captions").
  - get_dataloader: convenience wrapper.

Special tokens:
  PAD = 0   (padding, ignored in loss)
  START = 1 (<start> prepended to every caption)
  END = 2   (<end> appended to every caption)
  UNK = 3   (words outside the top-10000 vocabulary)

Future work (Issue #8, deferred): If the project switches to `pack_padded_sequence` or
`nn.LSTM`, sort collated batches by descending caption length and add a regression
test that verifies both padding placement and length order.
"""

import csv
import json
import os
import random
import re
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Special token indices — must match model embedding layer
# ---------------------------------------------------------------------------
PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3

SPLIT_FILE_MAP = {
    "train": "Flickr_8k.trainImages.txt",
    "val": "Flickr_8k.devImages.txt",
    "test": "Flickr_8k.testImages.txt",
}

EXPECTED_SPLIT_COUNTS = {
    "train": 6000,
    "val": 1000,
    "test": 1000,
}

_PUNCT_RE = re.compile(r"[^\w\s]")


def tokenize_caption(caption: str) -> List[str]:
    """Lowercase, strip punctuation, and split on whitespace."""
    caption = _PUNCT_RE.sub(" ", caption.lower())
    return caption.split()


def _normalize_image_name(image_id: str) -> str:
    """Drop Flickr8k caption suffixes like ``#0`` and keep the basename."""
    return os.path.basename(image_id.strip().split("#")[0])


def validate_dataset_layout(data_root: str, strict_split_counts: bool = True) -> None:
    """
    Validate the expected Flickr8k/Karpathy directory layout before training.

    Expected tree:
        data/flickr8k/
            images/
            captions.txt
            Flickr_8k.trainImages.txt
            Flickr_8k.devImages.txt
            Flickr_8k.testImages.txt
    """
    expected = [
        os.path.join(data_root, "images"),
        os.path.join(data_root, "captions.txt"),
        *[os.path.join(data_root, name) for name in SPLIT_FILE_MAP.values()],
    ]
    missing = [path for path in expected if not os.path.exists(path)]
    if missing:
        layout = "\n".join(f"  - {path}" for path in expected)
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Flickr8k files are missing.\n"
            f"Expected layout:\n{layout}\n"
            f"Missing:\n{missing_text}\n"
            "Download Flickr8k manually from Kaggle and place it under data/flickr8k/."
        )

    if not strict_split_counts:
        return

    split_sets: Dict[str, set] = {}
    for split, filename in SPLIT_FILE_MAP.items():
        split_path = os.path.join(data_root, filename)
        with open(split_path) as f:
            names = {_normalize_image_name(line) for line in f if line.strip()}
        expected_count = EXPECTED_SPLIT_COUNTS[split]
        if len(names) != expected_count:
            raise ValueError(
                f"{filename} contains {len(names)} unique images; expected "
                f"{expected_count} for the standard Flickr8k split."
            )
        split_sets[split] = names

    all_images = set().union(*split_sets.values())
    if len(all_images) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise ValueError(
            "Train/val/test split files overlap or are incomplete; expected "
            "8000 unique image names across the standard Flickr8k splits."
        )


def _read_split_images(data_root: str, split: str) -> set:
    split_path = os.path.join(data_root, SPLIT_FILE_MAP[split])
    with open(split_path) as f:
        return {_normalize_image_name(line) for line in f if line.strip()}


def _read_caption_rows(caps_path: str):
    with open(caps_path, newline="") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if "\t" in line:
                image_id, caption = line.split("\t", 1)
            else:
                row = next(csv.reader([line]))
                if len(row) < 2:
                    continue
                if row[0].lower() in {"image", "image_name", "filename"}:
                    continue
                image_id, caption = row[0], ",".join(row[1:])

            yield _normalize_image_name(image_id), caption.strip()


def load_flickr8k_captions(
    data_root: str,
    split: str,
    strict_split_counts: bool = True,
) -> Dict[str, List[str]]:
    """Load raw captions for one standard Flickr8k split."""
    assert split in ("train", "val", "test"), f"Unknown split: {split}"
    validate_dataset_layout(data_root, strict_split_counts=strict_split_counts)

    split_images = _read_split_images(data_root, split)
    caps_path = os.path.join(data_root, "captions.txt")
    image_to_caps: Dict[str, List[str]] = {}

    for img_name, caption in _read_caption_rows(caps_path):
        if img_name in split_images:
            image_to_caps.setdefault(img_name, []).append(caption)

    missing_captions = sorted(split_images - set(image_to_caps))
    if missing_captions:
        preview = ", ".join(missing_captions[:5])
        raise ValueError(
            f"{len(missing_captions)} split images have no captions in captions.txt "
            f"(first few: {preview})."
        )

    return image_to_caps


class Vocabulary:
    """
    Word ↔ integer index mapping.

    Build from a list of caption strings, keeping the top `max_size` words
    (excluding specials).  Save/load as JSON so the vocab can be reused
    across runs without rebuilding from scratch.

    Future work (Issue #2, deferred): If vocabulary tuning is assigned, add a
    `min_freq` mode alongside top-K and confirm serialized vocabs still round-trip
    through `save()` / `load()` without changing special-token indices.
    Future work (Issue #2, deferred): If subword tokenization is explored, keep it behind
    a separate code path and report whether it improves Flickr8k BLEU enough to
    justify changing the current simple word-level baseline.
    """

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._built = False

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, captions: List[str]) -> None:
        """
        Tokenise captions (lowercase + punctuation stripping) and keep the top
        `max_size` words by frequency.

        Future work (Issue #2, deferred): If tokenizer work is assigned, centralize the
        tokenization logic so `build()` and `encode()` use the same tokenizer and
        document how punctuation handling changes vocabulary size and BLEU.
        """
        counter: Counter = Counter()
        for cap in captions:
            counter.update(tokenize_caption(cap))

        # Reserve indices 0-3 for specials
        specials = [("<pad>", PAD_IDX), ("<start>", START_IDX),
                    ("<end>", END_IDX), ("<unk>", UNK_IDX)]
        self.word2idx = {w: i for w, i in specials}
        self.idx2word = {i: w for w, i in specials}

        # Add most common words
        for idx, (word, _) in enumerate(counter.most_common(self.max_size), start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self._built = True

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, caption: str) -> List[int]:
        """
        Convert a caption string to a list of token ids.
        Prepends START and appends END.

        Future work (Issue #2, deferred): If punctuation normalization is added, keep it
        identical across train/val/test caption processing and record whether the
        resulting vocab/metric changes are acceptable.
        """
        tokens = tokenize_caption(caption)
        ids = [START_IDX]
        ids += [self.word2idx.get(t, UNK_IDX) for t in tokens]
        ids += [END_IDX]
        return ids

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        """
        Convert token ids back to a human-readable string.

        Future work (Issue #2, deferred): If richer decoding is needed, store optional
        original-token metadata so `<unk>` outputs can be surfaced more usefully
        during qualitative inspection without changing training IDs.
        """
        words = []
        for i in ids:
            if skip_specials and i in (PAD_IDX, START_IDX, END_IDX):
                continue
            words.append(self.idx2word.get(i, "<unk>"))
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save vocab to JSON."""
        # Future work (Issue #2, deferred): Extend the JSON schema with `max_size`,
        # tokenizer name, and build metadata so collaborators can audit how a
        # saved vocab was produced.
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx}, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocab from JSON."""
        # Future work (Issue #2, deferred): Validate that the vocab JSON contains at
        # least `word2idx` and the four special tokens before loading; fail with
        # a clear error if the file was produced by an incompatible schema.
        with open(path) as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = {w: int(i) for w, i in data["word2idx"].items()}
        vocab.idx2word = {int(i): w for w, i in data["word2idx"].items()}
        vocab._built = True
        return vocab


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# ImageNet normalisation used by the pretrained VGG-16 encoder
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


class Flickr8kDataset(Dataset):
    """
    Flickr8k image-caption dataset.

    Expected directory layout:
        data/flickr8k/
            images/              ← JPEG images
            captions.txt         ← Karpathy-style: "image.jpg\tcaption text"
            Flickr_8k.trainImages.txt
            Flickr_8k.devImages.txt
            Flickr_8k.testImages.txt

    Each image has 5 reference captions.  During training each image-caption
    pair is a sample so length buckets are exact. During validation/test each
    image is returned once with all 5 references.

    Future work (Issue #3, deferred): If alternate split support is needed, accept a
    Karpathy-style JSON file as an opt-in path while preserving the official
    Flickr8k split files as the default source of truth.
    Future work (Issue #3): Enforce the proposal split assumptions at dataset creation:
    each split count must match the expected `6000/1000/1000`, and any mismatch
    should raise a dataset-setup error before training begins.
    Future work (Issue #8, deferred): If dataloader CPU time becomes a bottleneck,
    pre-encode captions once in `__init__` and reuse them in `__getitem__`
    instead of calling `vocab.encode()` every sample.
    """

    def __init__(
        self,
        data_root: str,
        vocab: Vocabulary,
        split: str = "train",  # "train" | "val" | "test"
        strict_split_counts: bool = True,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.vocab = vocab
        self.transform = _TRAIN_TRANSFORM if split == "train" else _EVAL_TRANSFORM
        self.image_dir = os.path.join(data_root, "images")

        if self.vocab is None:
            raise ValueError("Flickr8kDataset requires a built Vocabulary.")

        image_to_caps = load_flickr8k_captions(
            data_root,
            split,
            strict_split_counts=strict_split_counts,
        )

        self.image_names: List[str] = sorted(image_to_caps.keys())
        self.image_to_caps = image_to_caps
        self.image_captions: List[List[str]] = [
            image_to_caps[n] for n in self.image_names
        ]

        # Training uses one image-caption pair per sample so length buckets are
        # exact. Validation/test remain image-level to preserve the 5 references.
        self.samples: List[Tuple[str, str]] = []
        if self.split == "train":
            for img_name in self.image_names:
                for caption in self.image_to_caps[img_name]:
                    self.samples.append((img_name, caption))

        # Future work (Issue #8, deferred): If dataloader profiling shows repeated
        # encoding overhead, pre-tokenize captions here and keep both raw and
        # encoded forms so BLEU references remain untouched.

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.samples)
        return len(self.image_names)

    def __getitem__(self, idx: int):
        """
        Returns:
          image:           (3, 224, 224) float tensor
          caption_encoded: (caption_len,) long tensor  (with START/END)
          caption_len:     scalar long tensor
          all_captions:    list of raw strings for BLEU reference (eval only)

        Future work (Issue #8, deferred): If train-time caption inspection is added,
        return the chosen raw caption string alongside the encoded caption so
        debug logging can compare it against the five references for that image.
        """
        if self.split == "train":
            img_name, caption = self.samples[idx]
            captions = self.image_to_caps[img_name]
        else:
            img_name = self.image_names[idx]
            captions = self.image_captions[idx]
            # Eval: use first caption for loss; all captions for BLEU.
            caption = captions[0]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoded = self.vocab.encode(caption)
        caption_tensor = torch.tensor(encoded, dtype=torch.long)
        caption_len = torch.tensor(len(encoded), dtype=torch.long)

        return image, caption_tensor, caption_len, captions


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_fn(batch):
    """
    Pad captions to the max length in the batch.

    Returns:
        images:   (B, 3, 224, 224)
        captions: (B, max_len)    — padded with PAD_IDX
        lengths:  (B,)            — actual lengths
        refs:     list[list[str]] — all reference captions per image

    Future work (Issue #8, deferred): If the decoder is refactored to packed sequences,
    sort `batch` here by descending length and add a test confirming `captions`
    and `lengths` stay aligned after sorting.
    """
    images, captions, lengths, refs = zip(*batch)

    images = torch.stack(images, dim=0)
    lengths = torch.stack(lengths, dim=0)

    max_len = int(lengths.max().item())
    padded = torch.full((len(captions), max_len), PAD_IDX, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, : len(cap)] = cap

    return images, padded, lengths, list(refs)


# ---------------------------------------------------------------------------
# Length-bucket sampler
# ---------------------------------------------------------------------------

class LengthBucketSampler(Sampler):
    """
    Groups samples of the same caption length into the same batch to eliminate
    padding waste — matching paper section 4.3.

    This is a *batch* sampler: __iter__ yields whole lists of indices rather
    than individual indices.  It must therefore be passed to DataLoader as
    `batch_sampler=`, not `sampler=`.  The `get_dataloader` helper below does
    this automatically.

    Paper section 4.3: "we build a dictionary mapping the length of a sentence
    to the corresponding subset of captions. Then, during training we randomly
    sample a length and retrieve a mini-batch of size 64 of that length."

    Future work (Issue #8, deferred): If bucket sparsity becomes a problem, add an
    optional tolerance so captions of length `L±1` can share a bucket; verify
    that padding waste does not erase the throughput gain.
    """

    def __init__(
        self,
        dataset: Flickr8kDataset,
        batch_size: int,
        drop_last: bool = True,
    ):
        if dataset.split != "train":
            raise ValueError("LengthBucketSampler is intended for the train split only.")

        self.batch_size = batch_size
        self.drop_last = drop_last

        # Build length → [indices] mapping.
        from collections import defaultdict
        buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, (_, caption) in enumerate(dataset.samples):
            length = len(dataset.vocab.encode(caption))
            buckets[length].append(idx)
        self.buckets = dict(buckets)

        self._num_batches = 0
        for indices in self.buckets.values():
            if self.drop_last:
                self._num_batches += len(indices) // batch_size
            else:
                self._num_batches += (len(indices) + batch_size - 1) // batch_size

    def __iter__(self):
        # Reshuffle every epoch. DataLoader calls this once per epoch.
        batches: List[List[int]] = []
        for indices in self.buckets.values():
            indices = indices[:]
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return self._num_batches


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str,
    vocab: Vocabulary,
    split: str,
    batch_size: int = 64,
    num_workers: int = 4,
    use_bucket_sampler: bool = True,
    strict_split_counts: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for the requested split.

    When use_bucket_sampler=True (training default), LengthBucketSampler is
    passed as `batch_sampler` so DataLoader receives pre-formed same-length
    batches and does not re-group indices itself.  In that mode batch_size and
    shuffle must not be set on the DataLoader directly (PyTorch raises an error
    if you combine batch_sampler with those arguments).

    Future work (Issue #8, deferred): If loader configurability is needed, expose a
    `pin_memory` flag on `get_dataloader()` while keeping the current GPU-friendly
    default enabled for both train and eval.
    """
    dataset = Flickr8kDataset(
        data_root,
        vocab,
        split,
        strict_split_counts=strict_split_counts,
    )

    if split == "train" and use_bucket_sampler:
        # LengthBucketSampler yields complete batches → use batch_sampler=.
        # batch_size and shuffle are handled inside the sampler.
        batch_sampler = LengthBucketSampler(dataset, batch_size, drop_last=drop_last)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )
