"""JPEG compression vs class-label balance check.

If real photos are systematically more compressed than AI images (or vice versa),
a detector can learn the shortcut "JPEG artifacts = real" instead of AIGC signal.
Run this on the training dump and keep the report in the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import collect_samples  # noqa: E402
from utils import load_config, project_root  # noqa: E402

# Standard JPEG luminance quantization table at quality 50 (ITU T.81 / IJG).
_JPEG_STD_LUM = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]


def estimate_jpeg_quality(path: Path) -> Optional[float]:
    """Invert the IJG quality scaling from the luminance quantization table.

    Returns None for non-JPEG files (PNG etc. are treated as lossless).
    """
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return None
            tables = getattr(im, "quantization", None)
            if not tables:
                return None
            qtable = list(tables[0])
    except OSError:
        return None
    if len(qtable) != 64:
        return None
    if all(v <= 1 for v in qtable):
        return 100.0

    scales = []
    for q, std in zip(qtable, _JPEG_STD_LUM):
        if std <= 0:
            continue
        # Forward IJG rule: q = (std * scale + 50) / 100
        scale = max((q * 100.0 - 50.0) / std, 1.0)
        scales.append(scale)
    if not scales:
        return None
    scales.sort()
    scale = scales[len(scales) // 2]
    # Inverse of: quality < 50 → scale = 5000/quality ; else scale = 200 - 2*quality
    if scale > 100:
        quality = 5000.0 / scale
    else:
        quality = (200.0 - scale) / 2.0
    return float(max(1.0, min(100.0, quality)))


def _bucket(quality: float) -> str:
    if quality >= 90:
        return "90-100"
    if quality >= 70:
        return "70-89"
    if quality >= 50:
        return "50-69"
    return "1-49"


def summarize(samples) -> dict:
    stats = {
        "n": 0,
        "n_jpeg": 0,
        "n_non_jpeg": 0,
        "mean_quality": None,
        "quality_hist": {"90-100": 0, "70-89": 0, "50-69": 0, "1-49": 0},
        "qualities": [],
    }
    for sample in samples:
        if sample.path is None:
            continue
        stats["n"] += 1
        q = estimate_jpeg_quality(sample.path)
        if q is None:
            stats["n_non_jpeg"] += 1
            continue
        stats["n_jpeg"] += 1
        stats["qualities"].append(q)
        stats["quality_hist"][_bucket(q)] += 1
    if stats["qualities"]:
        stats["mean_quality"] = round(sum(stats["qualities"]) / len(stats["qualities"]), 2)
    stats["jpeg_fraction"] = round(stats["n_jpeg"] / stats["n"], 4) if stats["n"] else 0.0
    del stats["qualities"]
    return stats


def interpret(real: dict, ai: dict) -> str:
    """Plain-language warning if JPEG stats look like a spurious cue."""
    if real["n"] == 0 or ai["n"] == 0:
        return "Not enough labeled files to compare classes."
    gaps = []
    jpeg_gap = abs(real["jpeg_fraction"] - ai["jpeg_fraction"])
    if jpeg_gap >= 0.15:
        richer = "real" if real["jpeg_fraction"] > ai["jpeg_fraction"] else "AI"
        gaps.append(
            f"JPEG file fraction differs by {jpeg_gap:.0%} ({richer} class is more often JPEG). "
            "The model could use 'is JPEG' as a shortcut."
        )
    rq, aq = real["mean_quality"], ai["mean_quality"]
    if rq is not None and aq is not None and abs(rq - aq) >= 8:
        gaps.append(
            f"Mean JPEG quality differs by {abs(rq - aq):.1f} points "
            f"(real={rq}, AI={aq}). The model could use compression artifacts as a class cue."
        )
    if not gaps:
        return (
            "No large JPEG-rate or JPEG-quality gap between classes on this dump. "
            "Still apply JPEG augmentation so the model cannot latch onto residual codec cues."
        )
    return " ".join(gaps)


def run(cfg_path: Optional[str] = None, split: str = "train") -> dict:
    cfg = load_config(cfg_path)
    samples = collect_samples(cfg, split)
    real = [s for s in samples if s.label == 0]
    ai = [s for s in samples if s.label == 1]
    report = {
        "dataset": cfg["dataset"]["name"],
        "split": split,
        "real": summarize(real),
        "ai": summarize(ai),
    }
    report["warning"] = interpret(report["real"], report["ai"])
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="JPEG class-balance check for the training dump.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    report = run(args.config, args.split)
    out_dir = project_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "balance_check.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== JPEG class-balance check ===")
    print(f"dataset={report['dataset']}  split={report['split']}")
    for key in ("real", "ai"):
        s = report[key]
        print(
            f"  {key:4s}  n={s['n']:5d}  jpeg={s['n_jpeg']:5d} ({s['jpeg_fraction']:.1%})  "
            f"non-jpeg={s['n_non_jpeg']:5d}  mean_q={s['mean_quality']}"
        )
        print(f"         quality hist: {s['quality_hist']}")
    print(f"\nInterpretation:\n  {report['warning']}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
