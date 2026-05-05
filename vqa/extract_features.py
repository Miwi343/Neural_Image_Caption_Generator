"""
Pre-extract VGG-16 features for all COCO images used by VQA yes/no training.

Saves sharded feature files to avoid OOM during extraction and merging:
    data/vqa/features_train2014/chunk_0000.pt
    data/vqa/features_train2014/chunk_0001.pt
    ...
    data/vqa/features_val2014/chunk_0000.pt
    ...

Each shard is a dict[int, Tensor(196, 512)].  VQAYesNoDataset reads shards
lazily so no single load ever exceeds ~800 MB.
"""

import os
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.encoder import Encoder

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT   = "data/vqa"
IMAGES_DIR  = os.path.join(DATA_ROOT, "images")
DRIVE_CACHE = Path("/content/drive/MyDrive/vqa_results/data/vqa")

CHUNK_SIZE = 2000  # images per shard (~800 MB each at float32)

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class _ImageDataset(Dataset):
    def __init__(self, image_ids, split):
        self.paths = [
            (iid, os.path.join(IMAGES_DIR, f"{split}2014", f"COCO_{split}2014_{iid:012d}.jpg"))
            for iid in image_ids
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_id, path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return image_id, _TRANSFORM(img)


def extract_split(split: str, device: torch.device, encoder: Encoder, batch_size: int = 64):
    shard_dir = Path(DATA_ROOT) / f"features_{split}2014"
    if shard_dir.exists() and any(shard_dir.glob("chunk_*.pt")):
        print(f"[extract] {shard_dir} already has shards — skipping.")
        return

    img_dir   = Path(IMAGES_DIR) / f"{split}2014"
    image_ids = sorted(int(p.stem.split("_")[-1]) for p in img_dir.glob("*.jpg"))
    print(f"[extract] {split}: {len(image_ids):,} images")

    shard_dir.mkdir(parents=True, exist_ok=True)

    ds     = _ImageDataset(image_ids, split)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2,
                        pin_memory=True, persistent_workers=False)

    chunk: dict = {}
    chunk_idx   = 0

    encoder.eval()
    with torch.no_grad():
        for ids, imgs in tqdm(loader, desc=f"Extracting {split}"):
            feats = encoder(imgs.to(device)).cpu()   # (B, 196, 512) float32
            for iid, feat in zip(ids.tolist(), feats):
                chunk[iid] = feat
            if len(chunk) >= CHUNK_SIZE:
                _save_shard(shard_dir, chunk_idx, chunk, split)
                chunk_idx += 1
                chunk = {}

    if chunk:
        _save_shard(shard_dir, chunk_idx, chunk, split)

    # Back up shards to Drive
    drive_dir = DRIVE_CACHE / shard_dir.name
    drive_dir.mkdir(parents=True, exist_ok=True)
    for shard in sorted(shard_dir.glob("chunk_*.pt")):
        dest = drive_dir / shard.name
        if not dest.exists():
            shutil.copy2(shard, dest)
    print(f"[extract] Backed up shards → {drive_dir}")


def _save_shard(shard_dir: Path, idx: int, chunk: dict, split: str):
    path = shard_dir / f"chunk_{idx:04d}.pt"
    torch.save(chunk, path)
    print(f"[extract] Shard {idx:04d}: {len(chunk):,} features → {path}")


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(fine_tune=False).to(device)
    print(f"[extract] Device: {device}")

    for split in ("train", "val"):
        extract_split(split, device, encoder)

    print("[extract] Done.")


if __name__ == "__main__":
    main()
