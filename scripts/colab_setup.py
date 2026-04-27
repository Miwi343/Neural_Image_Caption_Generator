"""Colab setup helper for the Flickr8k soft-attention project.

This script is intentionally Colab-friendly:
  - validates the repo and Flickr8k layout
  - can normalize common Flickr8k/Kaggle folder layouts
  - auto-generates train/val/test split files when only images/ and captions.txt
    are present (the typical Kaggle download layout)
  - can download/unzip a Kaggle dataset when kaggle.json is available
  - keeps checkpoints/results inside the repo folder in Drive
"""

import argparse
import csv
import os
import random as _random
import shutil
import subprocess
import zipfile
from pathlib import Path


REQUIRED_FILES = [
    "captions.txt",
    "Flickr_8k.trainImages.txt",
    "Flickr_8k.devImages.txt",
    "Flickr_8k.testImages.txt",
]

SPLIT_FILES = [
    "Flickr_8k.trainImages.txt",
    "Flickr_8k.devImages.txt",
    "Flickr_8k.testImages.txt",
]

# Val and test are fixed at 1000 each; training gets ALL remaining images.
N_VAL  = 1000
N_TEST = 1000


def run(cmd, cwd=None):
    print("+", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def find_first(root: Path, names):
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, use_symlink: bool):
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    if use_symlink:
        os.symlink(src.resolve(), dst)
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _read_image_names_from_captions(captions_file: Path):
    """Return sorted list of unique image basenames from captions.txt."""
    image_names = set()
    with open(captions_file, newline="") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                image_id = line.split("\t", 1)[0]
            else:
                row = next(csv.reader([line]))
                if not row:
                    continue
                if row[0].lower() in {"image", "image_name", "filename"}:
                    continue
                image_id = row[0]
            name = os.path.basename(image_id.strip().split("#")[0])
            if name:
                image_names.add(name)
    return sorted(image_names)


def generate_split_files(data_root: Path) -> bool:
    """
    Generate Flickr_8k.{train,dev,test}Images.txt from captions.txt image names.

    Uses a fixed random seed (42) for reproducibility.  The standard 6000/1000/1000
    split is produced when the dataset has ≥ 8000 unique images; a proportional
    75/12.5/12.5 split is used for smaller datasets.

    Returns True if any files were written.
    """
    captions_path = data_root / "captions.txt"
    if not captions_path.exists():
        raise FileNotFoundError(f"captions.txt not found at {captions_path}")

    all_images = _read_image_names_from_captions(captions_path)
    if not all_images:
        raise ValueError("No image names could be read from captions.txt.")

    print(f"Found {len(all_images)} unique images in captions.txt")

    rng = _random.Random(42)
    shuffled = all_images[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    # Reserve fixed val/test pools; every remaining image goes to training.
    n_val   = min(N_VAL,  max(1, n // 8))
    n_test  = min(N_TEST, max(1, n // 8))
    n_train = n - n_val - n_test

    splits = [
        ("Flickr_8k.trainImages.txt", shuffled[:n_train]),
        ("Flickr_8k.devImages.txt",   shuffled[n_train : n_train + n_val]),
        ("Flickr_8k.testImages.txt",  shuffled[n_train + n_val : n_train + n_val + n_test]),
    ]

    wrote_any = False
    for filename, imgs in splits:
        out_path = data_root / filename
        if not out_path.exists():
            out_path.write_text("\n".join(imgs) + "\n")
            print(f"  Generated {filename} ({len(imgs)} images)")
            wrote_any = True
        else:
            print(f"  {filename} already exists — skipping")
    return wrote_any


def normalize_flickr8k(source_dir: Path, data_root: Path, use_symlink: bool = True):
    """Normalize common Kaggle/official Flickr8k layouts into data/flickr8k.

    Handles two common layouts:
      Layout A (official + split files present):
        source_dir/
          images/          (or Flickr8k_Dataset/ etc.)
          captions.txt
          Flickr_8k.trainImages.txt
          Flickr_8k.devImages.txt
          Flickr_8k.testImages.txt

      Layout B (Kaggle download — no split files):
        source_dir/
          images/
          captions.txt     (CSV with header: image,caption)
    """
    source_dir = source_dir.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    ensure_dir(data_root)

    # ------------------------------------------------------------------
    # 1. Locate and link/copy the images folder
    # ------------------------------------------------------------------
    image_dir = None
    for candidate in source_dir.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() in {
            "images",
            "flicker8k_dataset",
            "flickr8k_dataset",
        }:
            jpg_count = len(list(candidate.glob("*.jpg"))) + len(list(candidate.glob("*.jpeg")))
            if jpg_count > 100:
                image_dir = candidate
                break

    if image_dir is None:
        raise FileNotFoundError(
            f"Could not find the Flickr8k image folder under {source_dir}. "
            "Expected a folder named 'images' (or similar) containing .jpg files."
        )

    copy_or_link(image_dir, data_root / "images", use_symlink=use_symlink)

    # ------------------------------------------------------------------
    # 2. captions.txt — required; raise if missing
    # ------------------------------------------------------------------
    captions_src = find_first(source_dir, ["captions.txt"])
    if captions_src is None:
        raise FileNotFoundError(f"Could not find captions.txt under {source_dir}.")
    copy_or_link(captions_src, data_root / "captions.txt", use_symlink=use_symlink)

    # ------------------------------------------------------------------
    # 3. Split files — link if present, otherwise auto-generate
    # ------------------------------------------------------------------
    for filename in SPLIT_FILES:
        src = find_first(source_dir, [filename])
        if src is not None:
            copy_or_link(src, data_root / filename, use_symlink=use_symlink)

    missing_splits = [f for f in SPLIT_FILES if not (data_root / f).exists()]
    splits_generated = False
    if missing_splits:
        print(
            "Split files not found in source directory — generating from captions.txt:\n  "
            + "\n  ".join(missing_splits)
        )
        generate_split_files(data_root)
        splits_generated = True

    print(f"\nFlickr8k is ready at {data_root}")
    print("Contents:")
    for child in sorted(data_root.iterdir()):
        print(" ", child)
    return splits_generated


def download_from_kaggle(dataset: str, download_dir: Path):
    ensure_dir(download_dir)
    run(["python", "-m", "pip", "install", "-q", "kaggle"])
    run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(download_dir)])

    for zip_path in download_dir.glob("*.zip"):
        print(f"Unzipping {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(download_dir)


def validate(data_root: Path, strict: bool = True):
    from utils import validate_dataset_layout

    validate_dataset_layout(str(data_root), strict_split_counts=strict)
    print("Dataset validation passed.")


def main():
    parser = argparse.ArgumentParser(description="Prepare Flickr8k for Colab runs.")
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--data_root", default="data/flickr8k")
    parser.add_argument("--source_dir", default="")
    parser.add_argument("--download_kaggle", action="store_true")
    parser.add_argument(
        "--kaggle_dataset",
        default="adityajn105/flickr8k",
        help="Kaggle dataset slug to download when --download_kaggle is set.",
    )
    parser.add_argument("--kaggle_download_dir", default="/content/kaggle_flickr8k")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    parser.add_argument(
        "--no_strict",
        action="store_true",
        help="Skip exact split-count validation (useful when dataset size differs from standard 8000).",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    data_root = (repo_dir / args.data_root).resolve()

    if args.download_kaggle:
        download_dir = Path(args.kaggle_download_dir)
        download_from_kaggle(args.kaggle_dataset, download_dir)
        source_dir = download_dir
    elif args.source_dir:
        source_dir = Path(args.source_dir)
    else:
        source_dir = data_root

    splits_generated = normalize_flickr8k(source_dir, data_root, use_symlink=not args.copy)
    # Skip strict count check when we auto-generated splits (training size differs from 6000).
    validate(data_root, strict=not (args.no_strict or splits_generated))


if __name__ == "__main__":
    main()
