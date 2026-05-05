"""
Pre-extract VGG-16 features for all COCO images used by VQA yes/no training.

Saves sharded feature files to avoid OOM during extraction and merging:
    data/vqa/features_train2014/chunk_0000.pt  ...
    data/vqa/features_val2014/chunk_0000.pt    ...

Each shard is backed up to Drive immediately after it is written, so a crash
at any point loses at most one in-progress shard.  Re-running resumes from
the last completed shard (local or Drive).
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

CHUNK_SIZE = 2000  # images per shard


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


def _restore_from_drive(shard_dir: Path, drive_dir: Path):
    """Copy any shards that are on Drive but not yet local."""
    if not drive_dir.exists():
        return
    shard_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(drive_dir.glob("chunk_*.pt")):
        dst = shard_dir / src.name
        if not dst.exists():
            print(f"[extract] Restoring {src.name} from Drive ...")
            shutil.copy2(src, dst)


def _backup_shard(shard_path: Path, drive_dir: Path):
    """Copy one shard to Drive immediately after writing."""
    drive_dir.mkdir(parents=True, exist_ok=True)
    dest = drive_dir / shard_path.name
    shutil.copy2(shard_path, dest)


def _already_done(shard_dir: Path, drive_dir: Path, n_images: int) -> bool:
    """True if enough shards exist (locally or on Drive) to cover all images."""
    expected_shards = (n_images + CHUNK_SIZE - 1) // CHUNK_SIZE
    local  = len(list(shard_dir.glob("chunk_*.pt"))) if shard_dir.exists() else 0
    remote = len(list(drive_dir.glob("chunk_*.pt"))) if drive_dir.exists() else 0
    return max(local, remote) >= expected_shards


def extract_split(split: str, device: torch.device, encoder: Encoder, batch_size: int = 64):
    shard_dir = Path(DATA_ROOT) / f"features_{split}2014"
    drive_dir = DRIVE_CACHE / shard_dir.name

    img_dir   = Path(IMAGES_DIR) / f"{split}2014"
    image_ids = sorted(int(p.stem.split("_")[-1]) for p in img_dir.glob("*.jpg"))
    n_images  = len(image_ids)
    print(f"[extract] {split}: {n_images:,} images")

    if _already_done(shard_dir, drive_dir, n_images):
        # Ensure all Drive shards are local before declaring done
        _restore_from_drive(shard_dir, drive_dir)
        print(f"[extract] {split}: all shards present — skipping.")
        return

    # Restore whatever Drive has so we can resume
    _restore_from_drive(shard_dir, drive_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Find already-completed shards to skip their image IDs
    done_shards = sorted(shard_dir.glob("chunk_*.pt"))
    done_ids: set = set()
    for s in done_shards:
        data = torch.load(s, map_location="cpu", weights_only=True)
        done_ids.update(data.keys())
    if done_ids:
        print(f"[extract] Resuming — {len(done_ids):,} images already done, "
              f"{n_images - len(done_ids):,} remaining")

    remaining_ids = [iid for iid in image_ids if iid not in done_ids]
    if not remaining_ids:
        print(f"[extract] {split}: nothing left to extract.")
        return

    chunk_idx = len(done_shards)
    ds     = _ImageDataset(remaining_ids, split)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2,
                        pin_memory=True, persistent_workers=False)

    chunk: dict = {}
    encoder.eval()
    with torch.no_grad():
        for ids, imgs in tqdm(loader, desc=f"Extracting {split}"):
            feats = encoder(imgs.to(device)).cpu()
            for iid, feat in zip(ids.tolist(), feats):
                chunk[iid] = feat
            if len(chunk) >= CHUNK_SIZE:
                shard_path = _write_shard(shard_dir, chunk_idx, chunk)
                _backup_shard(shard_path, drive_dir)
                chunk_idx += 1
                chunk = {}

    if chunk:
        shard_path = _write_shard(shard_dir, chunk_idx, chunk)
        _backup_shard(shard_path, drive_dir)

    print(f"[extract] {split}: done.")


def _write_shard(shard_dir: Path, idx: int, chunk: dict) -> Path:
    path = shard_dir / f"chunk_{idx:04d}.pt"
    torch.save(chunk, path)
    print(f"[extract] Shard {idx:04d}: {len(chunk):,} features → {path}")
    return path


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(fine_tune=False).to(device)
    print(f"[extract] Device: {device}")

    for split in ("train", "val"):
        extract_split(split, device, encoder)

    print("[extract] Done.")


if __name__ == "__main__":
    main()
