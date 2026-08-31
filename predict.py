"""Required inference script: image directory → JSON predictions.

For each image the JSON record contains:
  image_path, pred, semantic_score, frequency_score, explanation, tier, suggested_policy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentations import ModelTransform
from models.artifacts import score_artifacts, sigmoid
from models.explain import explain_record
from models.fusion import AIGCDetector
from utils import IMAGE_EXTS, get_device, list_images, load_config, project_root

_DEFAULT_IMAGE_DIRS = ("test-images", "train images", "images")
_RESULT_FROM_TIER = {
    "likely_real": "real",
    "likely_ai_generated": "AI",
    "filters_or_edited": "filtered_or_edited",
}


def resolve_image_dir(raw: str | None) -> Path:
    """Turn --image-dir into a real folder. Default: `test-images` in the repo."""
    root = project_root()
    if raw:
        folder = Path(raw)
        if not folder.is_absolute():
            folder = root / folder
        if folder.is_dir():
            return folder
        examples = [name for name in _DEFAULT_IMAGE_DIRS if (root / name).is_dir()]
        hint = ""
        if examples:
            hint = f'\nExample: python predict.py --image-dir "{examples[0]}"'
        raise SystemExit(f"Image directory not found: {raw}{hint}")

    for name in _DEFAULT_IMAGE_DIRS:
        candidate = root / name
        if not candidate.is_dir():
            continue
        has_images = any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in candidate.iterdir()
        )
        if has_images:
            return candidate
    raise SystemExit(
        "No image folder given. Pass one with --image-dir, for example:\n"
        "  python predict.py --image-dir test-images"
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[AIGCDetector, dict]:
    path = Path(path)
    size = path.stat().st_size if path.is_file() else 0
    if size < 10 * 1024 * 1024:
        raise FileNotFoundError(
            f"{path} is empty or not a real checkpoint ({size} bytes). "
            "Copy the trained outputs/best.pt (~338 MB) from the original project. "
            "Creating a blank best.pt file will fail with 'Ran out of input'."
        )
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except EOFError as exc:
        raise FileNotFoundError(
            f"{path} is truncated or corrupt ({size} bytes). "
            "Re-copy the full outputs/best.pt (~338 MB). "
            "A GitHub clone, empty file, or incomplete download will not work."
        ) from exc
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
    clip_name: str = "openai/clip-vit-base-patch32",
    image_dir: Path | None = None,
) -> dict:
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
    tensors = transform(rgb)
    pixel_values = tensors["pixel_values"].unsqueeze(0).to(device)
    image_01 = tensors["image_01"].unsqueeze(0).to(device)
    out = model(pixel_values, image_01)
    semantic = float(out["semantic_score"].item())
    freq_t = out["frequency_score"]
    frequency = None if freq_t is None else float(freq_t.item())
    cues = score_artifacts(rgb, path=image_path, clip_name=clip_name, device=device)
    p_model = float(max(1e-6, min(1.0 - 1e-6, float(out["pred"].item()))))
    boost = float(cues.get("logit_boost") or 0.0)
    fired = list(cues.get("fired") or [])
    # CLIP live/product/skin cues must not flip a photographic call to likely_ai.
    hard_tells = {"garbled_text", "provenance", "hands"}
    if p_model < 0.52 and not (hard_tells & set(fired)):
        boost = 0.0
        fired = []
        cues["logit_boost"] = 0.0
        cues["fired"] = []
    z = float(np.log(p_model / (1.0 - p_model))) + boost
    pred = sigmoid(z)
    extra = explain_record(
        pred,
        semantic,
        frequency,
        likely_real_max=float(thresholds.get("likely_real_max", 0.4)),
        likely_ai_min=float(thresholds.get("likely_ai_min", 0.6)),
        fired=cues.get("fired") or [],
    )
    display = image_path.name
    if image_dir is not None:
        try:
            display = str(image_path.resolve().relative_to(Path(image_dir).resolve()))
        except ValueError:
            display = image_path.name
    return {
        "image_path": display.replace("\\", "/"),
        "result": _RESULT_FROM_TIER.get(extra["tier"], extra["tier"]),
        "pred": round(pred, 6),
        "semantic_score": round(semantic, 6),
        "frequency_score": None if frequency is None else round(frequency, 6),
        "explanation": extra["explanation"],
        "tier": extra["tier"],
        "suggested_policy": extra["suggested_policy"],
        "artifact_cues": {
            "fired": cues.get("fired") or [],
            "logit_boost": cues.get("logit_boost", 0.0),
            "provenance_ai": cues.get("provenance_ai", False),
            "c2pa": cues.get("c2pa", False),
            "visual": cues.get("visual") or {},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIGC detection on a folder of images.")
    parser.add_argument(
        "--image-dir",
        default=None,
        help='Directory of images (recursed). Defaults to "test-images".',
    )
    parser.add_argument("--output", default="outputs/predictions.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    image_dir = resolve_image_dir(args.image_dir)
    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images found under {image_dir}")

    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"]["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = project_root() / ckpt
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train first with python train.py")

    device = get_device(args.device)
    print(f"checkpoint={ckpt}")
    print(f"scoring {len(images)} images from {image_dir}")
    model, saved_cfg = load_checkpoint(ckpt, device)
    transform = ModelTransform(saved_cfg, augment=False)
    thresholds = saved_cfg.get("thresholds") or cfg.get("thresholds") or {}
    clip_name = saved_cfg.get("model", {}).get("clip_name") or "openai/clip-vit-base-patch32"

    records = []
    for path in tqdm(images, desc="predict"):
        try:
            records.append(
                predict_image(
                    model,
                    transform,
                    path,
                    device,
                    thresholds,
                    clip_name=clip_name,
                    image_dir=image_dir,
                )
            )
        except Exception as exc:  # keep going on a single corrupt file
            records.append(
                {
                    "image_path": path.name,
                    "result": "filtered_or_edited",
                    "pred": None,
                    "semantic_score": None,
                    "frequency_score": None,
                    "explanation": f"Failed to score image: {exc}",
                    "tier": "filters_or_edited",
                    "suggested_policy": (
                        "Treat as a filtered or edited photograph (including partial AI edits). "
                        "Do not treat as a clean camera original, and do not attach a full AIGC label."
                    ),
                    "artifact_cues": None,
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
