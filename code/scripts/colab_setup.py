"""Prepare Flickr8k data for local or Colab runs."""

import argparse
import csv
import os
import random as _random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

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
OFFICIAL_TEXT_ZIP_URL = (
    "https://github.com/jbrownlee/Datasets/releases/download/"
    "Flickr8k/Flickr8k_text.zip"
)
OFFICIAL_IMAGE_ZIP_URL = (
    "https://github.com/jbrownlee/Datasets/releases/download/"
    "Flickr8k/Flickr8k_Dataset.zip"
)

# Standard Flickr8k split sizes.
N_TRAIN = 6000
N_VAL = 1000
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
    """Create train/dev/test split files from captions.txt image names."""
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
    if n >= (N_TRAIN + N_VAL + N_TEST):
        n_train, n_val, n_test = N_TRAIN, N_VAL, N_TEST
        strict_count_compatible = True
    else:
        n_val = max(1, n // 8)
        n_test = max(1, n // 8)
        n_train = n - n_val - n_test
        strict_count_compatible = False

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
    if n > (n_train + n_val + n_test):
        print(f"  Leaving {n - (n_train + n_val + n_test)} extra images unused.")
    if not wrote_any:
        print("  Split files already existed; no files were written.")
    return strict_count_compatible


def download_if_missing(url: str, zip_path: Path) -> None:
    """Download a release asset once, leaving existing archives in place."""
    ensure_dir(zip_path.parent)
    if not zip_path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, zip_path)


def convert_token_file_to_captions(token_file: Path, captions_path: Path) -> None:
    """Convert official Flickr8k.token.txt rows into this repo's captions.txt."""
    if captions_path.exists():
        return

    rows = []
    with open(token_file, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "\t" not in line:
                continue
            image_id, caption = line.split("\t", 1)
            image_name = os.path.basename(image_id.strip().split("#")[0])
            caption = caption.strip()
            if image_name and caption:
                rows.append(f"{image_name}\t{caption}")

    if not rows:
        raise ValueError(f"No caption rows could be read from {token_file}.")

    ensure_dir(captions_path.parent)
    captions_path.write_text("\n".join(rows) + "\n")
    print(f"  Converted {token_file.name} -> {captions_path}")


def download_official_split_files(data_root: Path, download_dir: Path | None = None) -> bool:
    """Download the official Flickr8k text archive and copy split files."""
    download_dir = download_dir or (data_root / "_official_text")
    ensure_dir(download_dir)
    zip_path = download_dir / "Flickr8k_text.zip"
    download_if_missing(OFFICIAL_TEXT_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(download_dir)

    copied = False
    for filename in SPLIT_FILES:
        src = find_first(download_dir, [filename])
        if src is not None and not (data_root / filename).exists():
            shutil.copy2(src, data_root / filename)
            print(f"  Copied official {filename}")
            copied = True
    return copied


def download_public_flickr8k_release(
    data_root: Path,
    download_dir: Path | None = None,
) -> bool:
    """Download the public Flickr8k image/text release and normalize it."""
    download_dir = download_dir or (data_root / "_public_download")
    ensure_dir(download_dir)

    image_zip = download_dir / "Flickr8k_Dataset.zip"
    text_zip = download_dir / "Flickr8k_text.zip"
    download_if_missing(OFFICIAL_IMAGE_ZIP_URL, image_zip)
    download_if_missing(OFFICIAL_TEXT_ZIP_URL, text_zip)

    for zip_path in (image_zip, text_zip):
        print(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(download_dir)

    return normalize_flickr8k(
        download_dir,
        data_root,
        use_symlink=True,
        require_official_splits=True,
    )


def normalize_flickr8k(
    source_dir: Path,
    data_root: Path,
    use_symlink: bool = True,
    require_official_splits: bool = False,
):
    """Normalize common Kaggle and official Flickr8k layouts."""
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
            if jpg_count > 100:
                image_dir = candidate
                break

    if image_dir is None:
        raise FileNotFoundError(
            f"Could not find the Flickr8k image folder under {source_dir}. "
            "Expected a folder named 'images' (or similar) containing .jpg files."
        )

    copy_or_link(image_dir, data_root / "images", use_symlink=use_symlink)

    captions_src = find_first(source_dir, ["captions.txt"])
    if captions_src is None:
        token_src = find_first(source_dir, ["Flickr8k.token.txt"])
        if token_src is None:
            raise FileNotFoundError(
                f"Could not find captions.txt or Flickr8k.token.txt under {source_dir}."
            )
        convert_token_file_to_captions(token_src, data_root / "captions.txt")
    else:
        copy_or_link(captions_src, data_root / "captions.txt", use_symlink=use_symlink)

    for filename in SPLIT_FILES:
        src = find_first(source_dir, [filename])
        if src is not None:
            copy_or_link(src, data_root / filename, use_symlink=use_symlink)

    missing_splits = [f for f in SPLIT_FILES if not (data_root / f).exists()]
    strict_count_compatible = True
    if missing_splits:
        print(
            "Split files not found in source directory — trying official Flickr8k text archive:\n  "
            + "\n  ".join(missing_splits)
        )
        try:
            download_official_split_files(data_root)
        except Exception as exc:
            if require_official_splits:
                raise RuntimeError(
                    "Official Flickr8k split files are required for the paper "
                    "reproduction but could not be downloaded."
                ) from exc
            print(f"Official split download failed ({exc}); generating count-compatible splits.")

        missing_splits = [f for f in SPLIT_FILES if not (data_root / f).exists()]
        if missing_splits:
            if require_official_splits:
                raise FileNotFoundError(
                    "Official Flickr8k split files are required for the paper "
                    "reproduction. Missing:\n  " + "\n  ".join(missing_splits)
                )
            strict_count_compatible = generate_split_files(data_root)

    print(f"\nFlickr8k is ready at {data_root}")
    print("Contents:")
    for child in sorted(data_root.iterdir()):
        print(" ", child)
    return strict_count_compatible


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
    parser.add_argument(
        "--download_public_flickr8k",
        action="store_true",
        help="Download the public Flickr8k release assets into data_root.",
    )
    parser.add_argument(
        "--public_download_dir",
        default="",
        help="Optional cache directory for --download_public_flickr8k.",
    )
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
    parser.add_argument(
        "--require_official_splits",
        action="store_true",
        help="Fail instead of generating random splits when official Flickr8k split files are missing.",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    data_root = (repo_dir / args.data_root).resolve()

    if args.download_public_flickr8k:
        public_download_dir = Path(args.public_download_dir) if args.public_download_dir else None
        strict_count_compatible = download_public_flickr8k_release(
            data_root,
            public_download_dir,
        )
        validate(data_root, strict=(not args.no_strict and strict_count_compatible))
        return

    if args.download_kaggle:
        download_dir = Path(args.kaggle_download_dir)
        download_from_kaggle(args.kaggle_dataset, download_dir)
        source_dir = download_dir
    elif args.source_dir:
        source_dir = Path(args.source_dir)
    else:
        source_dir = data_root

    strict_count_compatible = normalize_flickr8k(
        source_dir,
        data_root,
        use_symlink=not args.copy,
        require_official_splits=args.require_official_splits,
    )
    validate(data_root, strict=(not args.no_strict and strict_count_compatible))


if __name__ == "__main__":
    main()
