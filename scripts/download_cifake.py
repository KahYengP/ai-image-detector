"""Download a CIFAKE subset from Hugging Face into the local ImageFolder layout.

HF labels in dragonintelligence/CIFAKE-image-dataset: 0 = FAKE, 1 = REAL.
Our loaders use folders train|test / REAL|FAKE (REAL=authentic, FAKE=AIGC).
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAKE (real vs AI) for training.")
    parser.add_argument("--max-per-class", type=int, default=1000, help="Images per class per split.")
    parser.add_argument("--out", default="data/raw/cifake")
    args = parser.parse_args()

    from datasets import load_dataset

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        shutil.rmtree(out)

    # 0=FAKE (AIGC), 1=REAL (authentic) on this Hub dump.
    hf_to_folder = {0: "FAKE", 1: "REAL"}
    for split, dest_split in (("train", "train"), ("test", "test")):
        ds = load_dataset("dragonintelligence/CIFAKE-image-dataset", split=split)
        counts = {"REAL": 0, "FAKE": 0}
        cap = args.max_per_class
        for row in tqdm(ds, desc=split):
            folder = hf_to_folder[int(row["label"])]
            if counts[folder] >= cap:
                if counts["REAL"] >= cap and counts["FAKE"] >= cap:
                    break
                continue
            dest = out / dest_split / folder
            dest.mkdir(parents=True, exist_ok=True)
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            img.save(dest / f"{folder.lower()}_{counts[folder]:05d}.jpg", quality=95)
            counts[folder] += 1
        print(f"{split}: {counts}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
