"""Tiny CIFAKE-layout dump so the pipeline can run without a 100MB+ download.

Creates JPEGs at deliberately different qualities per class so the Day-1
balance check has something real to report. These images are random fields,
not a training dataset — swap in real CIFAKE / SID_Set / WildFake for results.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _noise_image(rng: random.Random, size: int) -> Image.Image:
    pixels = bytes(rng.randint(0, 255) for _ in range(size * size * 3))
    return Image.frombytes("RGB", (size, size), pixels)


def main() -> None:
    root = ROOT / "data" / "raw" / "cifake"
    rng = random.Random(42)
    # Real photos in the wild are often re-encoded; many diffusion dumps are PNG/high-q JPEG.
    plan = {
        ("train", "REAL"): (24, [70, 75, 80, 85]),
        ("train", "FAKE"): (24, [92, 95, 98, 100]),
        ("test", "REAL"): (8, [70, 75, 80, 85]),
        ("test", "FAKE"): (8, [92, 95, 98, 100]),
    }
    n = 0
    for (split, cls), (count, qualities) in plan.items():
        folder = root / split / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = _noise_image(rng, size=128)
            q = qualities[i % len(qualities)]
            img.save(folder / f"{cls.lower()}_{i:03d}.jpg", format="JPEG", quality=q)
            n += 1
    print(f"Wrote {n} smoke JPEGs under {root}")
    print("Next: python -m data.balance_check")


if __name__ == "__main__":
    main()
