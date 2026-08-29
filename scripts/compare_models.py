"""Compare CLIP-only vs frequency-only vs fusion under low-quality conditions.

Real photos can be dark, small, and heavily JPEG'd. AI images can be given
the same look. A useful detector must still separate classes after those
degradations — not treat "looks cheap" as a class cue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentations import ModelTransform
from data.datasets import collect_samples
from evaluate import _auc, make_loader, score_loader
from predict import load_checkpoint, predict_image
from utils import get_device, load_config, project_root, set_seed

# Conditions that mimic poor real captures AND post-processed AI dumps.
QUALITY_CONDITIONS = [
    ("Clean", "clean", None),
    ("JPEG q30 (low quality)", "jpeg", 30),
    ("Resize 0.25x (small pixels)", "resize", 0.25),
    ("Blur sigma=2 (soft capture)", "blur", 2.0),
    ("Darken 50% (poor lighting)", "darken", 0.5),
]

SIGNALS = (
    ("CLIP semantic (ViT-B/32)", "semantic"),
    ("Frequency CNN (FFT)", "frequency"),
    ("Fusion MLP (CLIP + FFT)", "pred"),
)


def robust_score(rows: list[dict], key: str) -> tuple[float | None, float | None, float | None]:
    clean = next((r[key] for r in rows if r["name"] == "clean"), None)
    degraded = [r[key] for r in rows if r["name"] != "clean" and r[key] is not None]
    if clean is None or not degraded:
        return clean, None, None
    robust = float(np.mean(degraded))
    return clean, robust, 0.5 * clean + 0.5 * robust


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--user-dir", default="test-images")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["eval"]["max_samples"] = args.max_samples
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"]["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = project_root() / ckpt
    model, saved_cfg = load_checkpoint(ckpt, device)
    saved_cfg["paths"] = cfg["paths"]
    saved_cfg["dataset"] = cfg["dataset"]
    saved_cfg["eval"] = cfg["eval"]
    saved_cfg["train"]["num_workers"] = cfg["train"].get("num_workers", 0)

    samples = collect_samples(cfg, "test")
    rows = []
    for label, name, severity in QUALITY_CONDITIONS:
        eval_tf = None if name == "clean" else (name, severity)
        loader = make_loader(samples, saved_cfg, device, eval_tf, int(cfg["train"]["batch_size"]))
        rec = score_loader(model, loader, device)
        row = {
            "condition": label,
            "name": name,
            "n": len(rec["y"]),
            "pred": _auc(rec["y"], rec["pred"]),
            "semantic": _auc(rec["y"], rec["semantic"]),
            "frequency": _auc(rec["y"], rec["frequency"]),
        }
        rows.append(row)
        print(
            f"{label:32s}  fusion={row['pred']:.4f}  clip={row['semantic']:.4f}  "
            f"fft={row['frequency']:.4f}  n={row['n']}"
        )

    summary = {}
    for title, key in SIGNALS:
        clean, robust, final = robust_score(rows, key)
        summary[key] = {
            "name": title,
            "auc_clean": clean,
            "auc_quality": robust,
            "final_score": final,
        }
        print(f"{title}: clean={clean:.4f}  quality-robust={robust:.4f}  final={final:.4f}")

    winner_key = max(summary, key=lambda k: summary[k]["final_score"] or -1)
    winner = summary[winner_key]
    print(f"WINNER: {winner['name']}  final={winner['final_score']:.4f}")

    # User pair: real poor-light photo vs AI with similar "food photo" look.
    user_dir = Path(args.user_dir)
    if not user_dir.is_absolute():
        user_dir = project_root() / user_dir
    transform = ModelTransform(saved_cfg, augment=False)
    thresholds = saved_cfg.get("thresholds") or {}
    user_rows = []
    if user_dir.is_dir():
        for path in sorted(user_dir.glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rec = predict_image(model, transform, path, device, thresholds)
            user_rows.append(rec)
            print(
                f"user {path.name:16s}  fusion={rec['pred']:.3f}  "
                f"clip={rec['semantic_score']:.3f}  fft={rec['frequency_score']}"
            )

    out = {
        "n_test": rows[0]["n"] if rows else 0,
        "conditions": rows,
        "models": summary,
        "winner": winner_key,
        "winner_name": winner["name"],
        "user_images": user_rows,
        "note": (
            "AUC is threshold-free. Quality-robust = mean AUC after JPEG q30, "
            "0.25x resize, blur, and 50% darken applied to BOTH real and AI."
        ),
    }
    dest = project_root() / "outputs" / "model_comparison.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
