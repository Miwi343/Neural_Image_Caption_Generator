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

TODO (Issue #2): Download Flickr8k images and captions, place under
                 data/flickr8k/images/ and data/flickr8k/captions.txt
TODO (Issue #3): Verify 6000/1000/1000 split matches Hodosh et al. (2013)
                 standard split files (Flickr_8k.trainImages.txt etc.)
TODO (Issue #8): Confirm batch collation pads correctly and lengths are sorted
                 descending (required if switching to nn.LSTM with pack_padded)
"""

import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

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


class Vocabulary:
    """
    Word ↔ integer index mapping.

    Build from a list of caption strings, keeping the top `max_size` words
    (excluding specials).  Save/load as JSON so the vocab can be reused
    across runs without rebuilding from scratch.

    TODO (Issue #2): Expose min_freq threshold as an alternative to top-K.
    TODO (Issue #2): Add subword / BPE tokenisation option for future work.
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
        Tokenise captions (simple whitespace split + lower) and keep the top
        `max_size` words by frequency.

        TODO (Issue #2): Use NLTK word_tokenize for punctuation handling
                         (paper uses basic tokenisation on MS COCO).
        """
        counter: Counter = Counter()
        for cap in captions:
            counter.update(cap.lower().split())

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

        TODO (Issue #2): Handle punctuation stripping for cleaner tokens.
        """
        tokens = caption.lower().split()
        ids = [START_IDX]
        ids += [self.word2idx.get(t, UNK_IDX) for t in tokens]
        ids += [END_IDX]
        return ids

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        """
        Convert token ids back to a human-readable string.

        TODO (Issue #2): Handle UNK gracefully (show original token if stored).
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
        # TODO (Issue #2): Also save max_size and build timestamp for traceability
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx}, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocab from JSON."""
        # TODO (Issue #2): Validate JSON schema before loading
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

    Each image has 5 reference captions.  During training a random caption is
    sampled per epoch (paper section 4.3).  During validation/test all 5 are
    returned as references.

    TODO (Issue #3): Support Karpathy split JSON (karpathy_split.json) as an
                     alternative to the official split text files.
    TODO (Issue #3): Add sanity check that 6000+1000+1000 = 8000 images.
    TODO (Issue #8): Cache tokenised captions to avoid re-encoding each epoch.
    """

    def __init__(
        self,
        data_root: str,
        vocab: Vocabulary,
        split: str = "train",  # "train" | "val" | "test"
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.vocab = vocab
        self.transform = _TRAIN_TRANSFORM if split == "train" else _EVAL_TRANSFORM
        self.image_dir = os.path.join(data_root, "images")

        # TODO (Issue #3): Fall back to Karpathy JSON if split files absent
        split_file_map = {
            "train": "Flickr_8k.trainImages.txt",
            "val":   "Flickr_8k.devImages.txt",
            "test":  "Flickr_8k.testImages.txt",
        }
        split_path = os.path.join(data_root, split_file_map[split])
        with open(split_path) as f:
            split_images = set(line.strip() for line in f)

        # Parse captions.txt: each line is "image_name#N\tcaption"
        # Group by image name
        caps_path = os.path.join(data_root, "captions.txt")
        image_to_caps: Dict[str, List[str]] = {}
        # TODO (Issue #3): Handle alternative delimiter formats (comma-separated)
        with open(caps_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                img_id, caption = parts
                # Strip the "#N" annotation index suffix
                img_name = img_id.split("#")[0]
                if img_name in split_images:
                    image_to_caps.setdefault(img_name, []).append(caption)

        self.image_names: List[str] = sorted(image_to_caps.keys())
        self.image_captions: List[List[str]] = [
            image_to_caps[n] for n in self.image_names
        ]

        # TODO (Issue #8): Pre-tokenise here and store encoded lists for speed

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        """
        Returns:
          image:           (3, 224, 224) float tensor
          caption_encoded: (caption_len,) long tensor  (with START/END)
          caption_len:     scalar long tensor
          all_captions:    list of raw strings for BLEU reference (eval only)

        TODO (Issue #8): For train, also return the raw caption string so
                         the BLEU monitor can compare against all 5 references.
        """
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        captions = self.image_captions[idx]

        # Training: pick a random caption (paper section 4.3)
        if self.split == "train":
            caption = random.choice(captions)
        else:
            # Eval: use first caption for loss; all captions for BLEU
            caption = captions[0]

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

    TODO (Issue #8): Sort by descending length if using pack_padded_sequence.
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

    TODO (Issue #8): Re-shuffle bucket order each epoch (currently fixed after
                     construction; re-create the sampler each epoch or call
                     reshuffle() before each DataLoader iteration).
    TODO (Issue #8): Expose a tolerance parameter so buckets of length L can
                     accept samples of length L±1 to improve GPU utilisation
                     on sparse length buckets.
    """

    def __init__(self, dataset: Flickr8kDataset, batch_size: int):
        self.batch_size = batch_size

        # Build length → [indices] mapping.
        # We read caption lengths from the dataset without loading images by
        # accessing the pre-built image_captions list directly.
        # TODO (Issue #8): Store encoded lengths in dataset.__init__ so this
        #                  loop doesn't need to call vocab.encode each time.
        from collections import defaultdict
        buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, caps in enumerate(dataset.image_captions):
            # Use the first caption to determine the bucket length.
            # All 5 captions for an image have similar length, and we sample
            # randomly during training anyway.
            length = len(dataset.vocab.encode(caps[0]))
            buckets[length].append(idx)

        self.batches: List[List[int]] = []
        for length_key in sorted(buckets.keys()):
            indices = buckets[length_key]
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                self.batches.append(indices[start : start + batch_size])
        random.shuffle(self.batches)

    def __iter__(self):
        # Yield complete batches — DataLoader receives a list of indices and
        # does NOT apply its own batch_size grouping on top.
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


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
) -> DataLoader:
    """
    Build a DataLoader for the requested split.

    When use_bucket_sampler=True (training default), LengthBucketSampler is
    passed as `batch_sampler` so DataLoader receives pre-formed same-length
    batches and does not re-group indices itself.  In that mode batch_size and
    shuffle must not be set on the DataLoader directly (PyTorch raises an error
    if you combine batch_sampler with those arguments).

    TODO (Issue #8): Add pin_memory=True for faster GPU transfers on val/test.
    """
    dataset = Flickr8kDataset(data_root, vocab, split)

    if split == "train" and use_bucket_sampler:
        # LengthBucketSampler yields complete batches → use batch_sampler=.
        # batch_size and shuffle are handled inside the sampler.
        batch_sampler = LengthBucketSampler(dataset, batch_size)
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
