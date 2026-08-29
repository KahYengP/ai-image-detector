"""Operating-point tests: ranking, false negatives, false positives.

Run after training + calibration. Exits non-zero if gates fail.

Gates (val split):
  - Spearman(pred, label) >= 0.55  so pred rises with P(AI)
  - mean(pred | AI) > mean(pred | real)
  - P(likely_real | AI) <= 0.12     false negatives
  - P(likely_ai   | real) <= 0.18   false positives
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentations import ModelTransform
from data.datasets import build_datasets
from evaluate import make_loader, score_loader
from predict import load_checkpoint, predict_image
from scripts.calibrate_thresholds import _ece, _spearman, artifact_boosts, apply_calib, tier_rates
from utils import get_device, list_images, load_config, project_root, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-dir", default="test-images")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"]["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = project_root() / ckpt

    model, saved = load_checkpoint(ckpt, device)
    thresholds = saved.get("thresholds") or {}
    t_real = float(thresholds.get("likely_real_max", 0.38))
    t_ai = float(thresholds.get("likely_ai_min", 0.62))
    bias = float(thresholds.get("logit_bias", 0.0))
    temperature = float(thresholds.get("temperature", 1.0))
    clip_name = saved.get("model", {}).get("clip_name", "openai/clip-vit-base-patch32")

    # Score val with the same temperature/bias the checkpoint applies, then
    # add artifact boosts (CIFAKE skipped inside artifact_boosts).
    model.logit_bias = 0.0
    model.temperature = 1.0
    saved["train"]["num_workers"] = cfg["train"].get("num_workers", 0)
    _, val_ds, test_ds = build_datasets(saved, ModelTransform(saved, True), ModelTransform(saved, False))
    loader = make_loader(val_ds.samples, saved, device, None, int(saved["train"]["batch_size"]))
    rec = score_loader(model, loader, device)
    y = np.array(rec["y"], dtype=np.float64)
    p0 = np.clip(np.array(rec["pred"]), 1e-6, 1.0 - 1e-6)
    logit = np.log(p0 / (1.0 - p0))
    boost = artifact_boosts(val_ds.samples, clip_name, device)
    p = apply_calib(logit, boost, bias, temperature)

    spearman = _spearman(y, p)
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")
    stats = tier_rates(y, p, t_real, t_ai)
    mean_real = float(p[y < 0.5].mean())
    mean_ai = float(p[y >= 0.5].mean())

    gates = {
        "spearman_pred_vs_label": {"value": spearman, "min": 0.55, "ok": spearman >= 0.55},
        "mean_ai_above_mean_real": {"value": mean_ai - mean_real, "min": 0.08, "ok": mean_ai > mean_real + 0.08},
        "fn_ai_called_real": {"value": stats["fn_ai_called_real"], "max": 0.12, "ok": stats["fn_ai_called_real"] <= 0.12},
        "fp_real_called_ai": {"value": stats["fp_real_called_ai"], "max": 0.18, "ok": stats["fp_real_called_ai"] <= 0.18},
        "auc": {"value": auc, "min": 0.90, "ok": auc >= 0.90},
    }

    # Also score the CIFAKE test split with calibrated scores (no CLIP artifacts).
    test_loader = make_loader(test_ds.samples, saved, device, None, int(saved["train"]["batch_size"]))
    test_rec = score_loader(model, test_loader, device)
    y_te = np.array(test_rec["y"], dtype=np.float64)
    p0_te = np.clip(np.array(test_rec["pred"]), 1e-6, 1.0 - 1e-6)
    logit_te = np.log(p0_te / (1.0 - p0_te))
    p_te = apply_calib(logit_te, np.zeros_like(logit_te), bias, temperature)
    test_stats = {
        "n": int(len(y_te)),
        "auc": float(roc_auc_score(y_te, p_te)) if len(set(y_te.tolist())) > 1 else float("nan"),
        "spearman": _spearman(y_te, p_te),
        **tier_rates(y_te, p_te, t_real, t_ai),
    }

    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = project_root() / image_dir
    folder_records = []
    transform = ModelTransform(saved, augment=False)
    model.logit_bias = bias
    model.temperature = temperature
    if image_dir.is_dir():
        for path in list_images(image_dir):
            rec_i = predict_image(model, transform, path, device, thresholds, clip_name=clip_name)
            folder_records.append(
                {
                    "image": path.name,
                    "pred": rec_i["pred"],
                    "tier": rec_i["tier"],
                    "explanation": rec_i["explanation"],
                    "fired": (rec_i.get("artifact_cues") or {}).get("fired") or [],
                }
            )

    report = {
        "thresholds": {"likely_real_max": t_real, "likely_ai_min": t_ai, "logit_bias": bias, "temperature": temperature},
        "val": {
            "n": int(len(y)),
            "auc": auc,
            "ece": _ece(y, p),
            "mean_pred_real": mean_real,
            "mean_pred_ai": mean_ai,
            **stats,
        },
        "cifake_test": test_stats,
        "gates": gates,
        "test_images": folder_records,
        "passed": all(g["ok"] for g in gates.values()),
    }
    dest = project_root() / "outputs" / "test_report.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "val": report["val"], "gates": gates, "cifake_test": test_stats}, indent=2))
    print(f"wrote {dest}")
    if folder_records:
        print("test-images:")
        for row in folder_records:
            print(f"  {row['image']:16s}  pred={row['pred']:.3f}  {row['tier']}")
    if not report["passed"]:
        failed = [k for k, g in gates.items() if not g["ok"]]
        raise SystemExit(f"gates failed: {failed}")


if __name__ == "__main__":
    main()
