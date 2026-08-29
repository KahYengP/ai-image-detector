"""Required inference script: image directory → JSON predictions.

For each image the JSON record contains:
  image_path, pred, semantic_score, frequency_score, explanation, tier, suggested_policy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentations import ModelTransform
from models.explain import explain_record
from models.fusion import AIGCDetector
from utils import get_device, list_images, load_config, project_root


def load_checkpoint(path: Path, device: torch.device) -> tuple[AIGCDetector, dict]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    cfg = payload["config"]
    model = AIGCDetector(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_image(
    model: AIGCDetector,
    transform: ModelTransform,
    image_path: Path,
    device: torch.device,
    thresholds: dict,
) -> dict:
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
    tensors = transform(rgb)
    pixel_values = tensors["pixel_values"].unsqueeze(0).to(device)
    image_01 = tensors["image_01"].unsqueeze(0).to(device)
    out = model(pixel_values, image_01)
    pred = float(out["pred"].item())
    semantic = float(out["semantic_score"].item())
    freq_t = out["frequency_score"]
    frequency = None if freq_t is None else float(freq_t.item())
    extra = explain_record(
        pred,
        semantic,
        frequency,
        likely_real_max=float(thresholds.get("likely_real_max", 0.4)),
        likely_ai_min=float(thresholds.get("likely_ai_min", 0.6)),
    )
    return {
        "image_path": str(image_path),
        "pred": round(pred, 6),
        "semantic_score": round(semantic, 6),
        "frequency_score": None if frequency is None else round(frequency, 6),
        "explanation": extra["explanation"],
        "tier": extra["tier"],
        "suggested_policy": extra["suggested_policy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIGC detection on a folder of images.")
    parser.add_argument("--image-dir", required=True, help="Directory of images (recursed).")
    parser.add_argument("--output", default="outputs/predictions.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"]["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = project_root() / ckpt
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train first with python train.py")

    device = get_device(args.device)
    model, saved_cfg = load_checkpoint(ckpt, device)
    transform = ModelTransform(saved_cfg, augment=False)
    thresholds = saved_cfg.get("thresholds") or cfg.get("thresholds") or {}

    images = list_images(args.image_dir)
    if not images:
        raise SystemExit(f"No images found under {args.image_dir}")

    records = []
    for path in tqdm(images, desc="predict"):
        try:
            records.append(predict_image(model, transform, path, device, thresholds))
        except Exception as exc:  # keep going on a single corrupt file
            records.append(
                {
                    "image_path": str(path),
                    "pred": None,
                    "semantic_score": None,
                    "frequency_score": None,
                    "explanation": f"Failed to score image: {exc}",
                    "tier": "low_confidence",
                    "suggested_policy": "Flag for reduced distribution weight and queue a secondary automated detector pass.",
                }
            )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = project_root() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
