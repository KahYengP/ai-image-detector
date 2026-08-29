"""Fit temperature + bias so `pred` tracks P(AI), then set three-way tiers.

Previously we only minimized real FPs, which collapsed scores and labeled
true AI as `likely_real` (false negatives). This pass:

  1. Fits temperature / bias by negative log-likelihood on val logits so
     higher `pred` means higher empirical P(AI) (Spearman is checked).
  2. Chooses `likely_real_max` / `likely_ai_min` under dual limits:
       FN = P(likely_real | AI)  must stay low
       FP = P(likely_ai   | real) must stay low
     The middle band is filters/edited, not a dump for missed AI.

High-res SID_Set val images also get artifact logit boosts so the operating
point matches `predict.py`. CIFAKE 32x32 skips visual CLIP cues.
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentations import ModelTransform
from data.datasets import build_datasets
from evaluate import make_loader, score_loader
from models.artifacts import score_artifacts, sigmoid
from predict import load_checkpoint
from utils import get_device, load_config, project_root, set_seed


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    # Rank correlation without requiring scipy at import time.
    ry = np.argsort(np.argsort(y))
    rp = np.argsort(np.argsort(p))
    ry = ry.astype(np.float64)
    rp = rp.astype(np.float64)
    ry -= ry.mean()
    rp -= rp.mean()
    denom = float(np.sqrt((ry * ry).sum() * (rp * rp).sum()))
    return float((ry * rp).sum() / denom) if denom > 0 else 0.0


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(y)
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if not mask.any():
            continue
        total += (mask.sum() / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def apply_calib(logit: np.ndarray, boost: np.ndarray, bias: float, temperature: float) -> np.ndarray:
    z = (logit + bias) / max(temperature, 1e-3) + boost
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def tier_rates(y: np.ndarray, p: np.ndarray, t_real: float, t_ai: float) -> dict[str, float]:
    ai = y >= 0.5
    real = ~ai
    fn = float((p[ai] < t_real).mean()) if ai.any() else float("nan")
    fp = float((p[real] > t_ai).mean()) if real.any() else float("nan")
    ai_recall = float((p[ai] > t_ai).mean()) if ai.any() else float("nan")
    edited_ai = float(((p[ai] >= t_real) & (p[ai] <= t_ai)).mean()) if ai.any() else float("nan")
    edited_real = float(((p[real] >= t_real) & (p[real] <= t_ai)).mean()) if real.any() else float("nan")
    return {
        "fn_ai_called_real": fn,
        "fp_real_called_ai": fp,
        "ai_recall_hard": ai_recall,
        "ai_in_edited_band": edited_ai,
        "real_in_edited_band": edited_real,
    }


def artifact_boosts(samples, clip_name: str, device: torch.device) -> np.ndarray:
    boosts = np.zeros(len(samples), dtype=np.float64)
    for i, sample in enumerate(tqdm(samples, desc="artifact cues")):
        if sample.path is None or sample.source == "cifake":
            continue
        try:
            with Image.open(sample.path) as im:
                rgb = im.convert("RGB")
            cues = score_artifacts(rgb, path=sample.path, clip_name=clip_name, device=device)
            boosts[i] = float(cues.get("logit_boost") or 0.0)
        except Exception:
            continue
    return boosts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-fn", type=float, default=0.12, help="Max P(likely_real | AI).")
    parser.add_argument("--max-fp", type=float, default=0.15, help="Max P(likely_ai | real).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"]["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = project_root() / ckpt

    model, saved = load_checkpoint(ckpt, device)
    model.logit_bias = 0.0
    model.temperature = 1.0
    saved["train"]["num_workers"] = cfg["train"].get("num_workers", 0)
    _, val_ds, _ = build_datasets(saved, ModelTransform(saved, True), ModelTransform(saved, False))
    loader = make_loader(val_ds.samples, saved, device, None, int(saved["train"]["batch_size"]))
    rec = score_loader(model, loader, device)
    y = np.array(rec["y"], dtype=np.float64)
    sources = np.array([s.source for s in val_ds.samples])
    p0 = np.clip(np.array(rec["pred"]), 1e-6, 1.0 - 1e-6)
    logit = np.log(p0 / (1.0 - p0))
    clip_name = saved.get("model", {}).get("clip_name", "openai/clip-vit-base-patch32")
    boost = artifact_boosts(val_ds.samples, clip_name, device)

    best_tb = None
    best_nll = float("inf")
    for temperature in np.linspace(0.75, 1.6, 18):
        for bias in np.linspace(-0.25, 0.45, 15):
            p = apply_calib(logit, boost, float(bias), float(temperature))
            nll = _nll(y, p)
            # Prefer fits where scores still rank AI above real.
            corr = _spearman(y, p)
            if corr < 0.45:
                continue
            if nll < best_nll:
                best_nll = nll
                best_tb = {"temperature": float(temperature), "logit_bias": float(bias), "nll": nll, "spearman": corr}

    if best_tb is None:
        best_tb = {"temperature": 1.0, "logit_bias": 0.0, "nll": _nll(y, apply_calib(logit, boost, 0.0, 1.0)), "spearman": _spearman(y, apply_calib(logit, boost, 0.0, 1.0))}

    p = apply_calib(logit, boost, best_tb["logit_bias"], best_tb["temperature"])

    def search(max_fn: float, max_fp: float):
        best = None
        for t_real in (0.28, 0.32, 0.36, 0.40, 0.44, 0.48):
            for t_ai in (0.55, 0.58, 0.62, 0.66, 0.70, 0.74):
                if t_ai - t_real < 0.14:
                    continue
                stats = tier_rates(y, p, t_real, t_ai)
                if stats["fn_ai_called_real"] > max_fn or stats["fp_real_called_ai"] > max_fp:
                    continue
                # Slightly prefer catching AI as hard-AI over parking it in edited.
                cost = (
                    1.4 * stats["fn_ai_called_real"]
                    + 1.0 * stats["fp_real_called_ai"]
                    + 0.15 * stats["ai_in_edited_band"]
                )
                row = {
                    "likely_real_max": t_real,
                    "likely_ai_min": t_ai,
                    **stats,
                    "cost": cost,
                }
                if best is None or cost < best["cost"]:
                    best = row
        return best

    chosen_tiers = search(args.max_fn, args.max_fp)
    relaxed = False
    if chosen_tiers is None:
        chosen_tiers = search(0.18, 0.20)
        relaxed = True
    if chosen_tiers is None:
        # Last resort: keep a 0.20-wide edited band around 0.5.
        chosen_tiers = {
            "likely_real_max": 0.40,
            "likely_ai_min": 0.62,
            **tier_rates(y, p, 0.40, 0.62),
            "cost": None,
        }
        relaxed = True

    best = {
        **best_tb,
        **chosen_tiers,
        "ece": _ece(y, p),
        "mean_pred_real": float(p[y < 0.5].mean()),
        "mean_pred_ai": float(p[y >= 0.5].mean()),
        "spearman": _spearman(y, p),
        "relaxed_limits": relaxed,
    }

    by_source = {}
    for src in sorted(set(sources.tolist())):
        mask = sources == src
        stats = tier_rates(y[mask], p[mask], best["likely_real_max"], best["likely_ai_min"])
        by_source[src] = {
            "n": int(mask.sum()),
            "spearman": _spearman(y[mask], p[mask]),
            **stats,
        }

    saved.setdefault("thresholds", {})
    saved["thresholds"]["logit_bias"] = best["logit_bias"]
    saved["thresholds"]["temperature"] = best["temperature"]
    saved["thresholds"]["likely_ai_min"] = best["likely_ai_min"]
    saved["thresholds"]["likely_real_max"] = best["likely_real_max"]

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload["config"] = saved
    payload["calibration"] = best
    torch.save(payload, ckpt)

    out = {
        "chosen": {
            "logit_bias": best["logit_bias"],
            "temperature": best["temperature"],
            "likely_real_max": best["likely_real_max"],
            "likely_ai_min": best["likely_ai_min"],
            "fn_ai_called_real": best["fn_ai_called_real"],
            "fp_real_called_ai": best["fp_real_called_ai"],
            "ai_recall_hard": best["ai_recall_hard"],
            "spearman": best["spearman"],
            "ece": best["ece"],
            "mean_pred_real": best["mean_pred_real"],
            "mean_pred_ai": best["mean_pred_ai"],
            "nll": best["nll"],
            "relaxed_limits": best["relaxed_limits"],
        },
        "targets": {"max_fn": args.max_fn, "max_fp": args.max_fp},
        "n_val": int(len(y)),
        "n_real": int((y < 0.5).sum()),
        "n_ai": int((y >= 0.5).sum()),
        "by_source": by_source,
        "limits": {
            "meaning": (
                "pred is temperature-scaled so it increases with P(AI). "
                "likely_real if pred < likely_real_max; likely_ai_generated if "
                "pred > likely_ai_min; the band between is filters_or_edited "
                "(filters, retouching, or partial AI edits)."
            ),
        },
    }
    dest = project_root() / "outputs" / "calibration.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["chosen"], indent=2))
    print(f"patched {ckpt}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
