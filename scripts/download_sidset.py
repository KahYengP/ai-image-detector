"""Download a compact high-res real vs synthetic subset (SID_Set-description).

CIFAKE reals are 32x32 CIFAR crops. Ordinary photographs then look "not real"
to a CIFAKE-tuned head. These OpenImages-based stills (and matched synthetics)
are the antidote. Tampered (label 2) is skipped.

Never uses COCO val2017 / DALL-E Advanced.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _save(img: Image.Image, dest: Path, max_side: int = 384) -> None:
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=900)
    parser.add_argument("--out", default="data/raw/sid_set")
    args = parser.parse_args()

    from datasets import load_dataset

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        shutil.rmtree(out)

    # Prefer the compact description dump (~3k) over the full 210k SID_Set.
    ds = load_dataset("saberzl/SID_Set_description")
    mapping = {0: "real", 1: "full_synthetic"}
    counts = {"train": {"real": 0, "full_synthetic": 0}, "validation": {"real": 0, "full_synthetic": 0}}

    split_map = {"train": "train", "validation": "validation"}
    for hf_split, dest_split in split_map.items():
        if hf_split not in ds:
            continue
        for i, row in enumerate(tqdm(ds[hf_split], desc=hf_split)):
            lab = int(row["label"])
            if lab not in mapping:
                continue
            folder = mapping[lab]
            if counts[dest_split][folder] >= args.max_per_class:
                if all(v >= args.max_per_class for v in counts[dest_split].values()):
                    break
                continue
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            n = counts[dest_split][folder]
            _save(img, out / dest_split / folder / f"{folder}_{n:05d}.jpg")
            counts[dest_split][folder] += 1
        print(f"{dest_split}: {counts[dest_split]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
