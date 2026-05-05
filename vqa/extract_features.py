"""
Pre-extract VGG-16 features for all COCO images used by VQA yes/no training.

Saves two files:
    data/vqa/features_train2014.pt  ->  dict[image_id (int): tensor (196, 512)]
    data/vqa/features_val2014.pt    ->  dict[image_id (int): tensor (196, 512)]

Back both up to Drive so they survive session resets.
Run once; takes ~30-60 min on GPU but training epochs then take ~2-3 min each.
"""

import os
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
    out_path = Path(DATA_ROOT) / f"features_{split}2014.pt"
    if out_path.exists():
        print(f"[extract] {out_path} already exists — skipping.")
        return

    img_dir = Path(IMAGES_DIR) / f"{split}2014"
    image_ids = sorted(
        int(p.stem.split("_")[-1])
        for p in img_dir.glob("*.jpg")
    )
    print(f"[extract] {split}: {len(image_ids):,} images")

    ds     = _ImageDataset(image_ids, split)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2,
                        pin_memory=True, persistent_workers=False)

    features = {}
    encoder.eval()
    with torch.no_grad():
        for ids, imgs in tqdm(loader, desc=f"Extracting {split}"):
            imgs = imgs.to(device)
            feats = encoder(imgs)          # (B, 196, 512)
            feats = feats.cpu()
            for iid, feat in zip(ids.tolist(), feats):
                features[iid] = feat       # (196, 512) float32

    torch.save(features, out_path)
    print(f"[extract] Saved {len(features):,} features → {out_path}")

    drive_path = DRIVE_CACHE / out_path.name
    DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(out_path, drive_path)
    print(f"[extract] Backed up → {drive_path}")


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(fine_tune=False).to(device)
    print(f"[extract] Device: {device}")

    for split in ("train", "val"):
        extract_split(split, device, encoder)

    print("[extract] Done.")


if __name__ == "__main__":
    main()
