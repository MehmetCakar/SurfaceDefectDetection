"""Split the NEU-DET dataset into train/val/test class folders.

Expects the Kaggle NEU-DET layout under data/raw, e.g.:
    data/raw/NEU-DET/train/images/<class_name>/*.jpg
or a flat per-class layout:
    data/raw/<class_name>/*.jpg

Produces:
    data/processed/{train,val,test}/<class_name>/*.jpg  (70/15/15 split)
"""

import argparse
import random
import shutil
from pathlib import Path

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def find_class_dirs(raw_dir: Path) -> dict[str, list[Path]]:
    """Collect image paths per class, tolerating different NEU-DET layouts."""
    images: dict[str, list[Path]] = {c: [] for c in CLASSES}
    for path in raw_dir.rglob("*.jpg"):
        # class name appears either as parent folder or filename prefix
        parent = path.parent.name.lower()
        stem = path.stem.lower()
        for cls in CLASSES:
            if parent == cls or stem.startswith(cls):
                images[cls].append(path)
                break
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = find_class_dirs(args.raw_dir)
    missing = [c for c, files in images.items() if not files]
    if missing:
        raise SystemExit(
            f"No images found for classes {missing} under {args.raw_dir}. "
            "Did you extract the NEU-DET dataset into data/raw?"
        )

    rng = random.Random(args.seed)
    for cls, files in images.items():
        files = sorted(set(files))
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        split_files = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }
        for split, split_list in split_files.items():
            dest = args.out_dir / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in split_list:
                shutil.copy2(src, dest / src.name)
        print(
            f"{cls}: {n} images -> "
            + ", ".join(f"{s}={len(v)}" for s, v in split_files.items())
        )


if __name__ == "__main__":
    main()
