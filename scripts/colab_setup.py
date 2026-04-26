"""Colab setup helper for the Flickr8k soft-attention project.

This script is intentionally Colab-friendly:
  - validates the repo and Flickr8k layout
  - can normalize common Flickr8k/Kaggle folder layouts
  - can download/unzip a Kaggle dataset when kaggle.json is available
  - keeps checkpoints/results inside the repo folder in Drive
"""

import argparse
import os
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


def normalize_flickr8k(source_dir: Path, data_root: Path, use_symlink: bool = True):
    """Normalize common Kaggle/official Flickr8k layouts into data/flickr8k."""
    source_dir = source_dir.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    ensure_dir(data_root)

    image_dir = None
    for candidate in source_dir.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() in {
            "images",
            "flicker8k_dataset",
            "flickr8k_dataset",
        }:
            jpg_count = len(list(candidate.glob("*.jpg"))) + len(list(candidate.glob("*.jpeg")))
            if jpg_count > 1000:
                image_dir = candidate
                break

    if image_dir is None:
        raise FileNotFoundError(
            f"Could not find the Flickr8k image folder under {source_dir}. "
            "Expected a folder with thousands of .jpg files."
        )

    copy_or_link(image_dir, data_root / "images", use_symlink=use_symlink)

    for filename in REQUIRED_FILES:
        src = find_first(source_dir, [filename])
        if src is None:
            raise FileNotFoundError(f"Could not find {filename} under {source_dir}.")
        copy_or_link(src, data_root / filename, use_symlink=use_symlink)

    print(f"Flickr8k is ready at {data_root}")
    print("Contents:")
    for child in sorted(data_root.iterdir()):
        print(" ", child)


def download_from_kaggle(dataset: str, download_dir: Path):
    ensure_dir(download_dir)
    run(["python", "-m", "pip", "install", "-q", "kaggle"])
    run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(download_dir)])

    for zip_path in download_dir.glob("*.zip"):
        print(f"Unzipping {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(download_dir)


def validate(data_root: Path):
    from utils import validate_dataset_layout

    validate_dataset_layout(str(data_root))
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

    normalize_flickr8k(source_dir, data_root, use_symlink=not args.copy)
    validate(data_root)


if __name__ == "__main__":
    main()
