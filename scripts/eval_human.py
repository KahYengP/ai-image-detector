"""Hold-out metrics for the 284 human-labeled photos.

Reports binary AUC (AI vs photographic) and three-way tier accuracy:
  real / real-blur          → likely_real
  edited / filtered_edited  → filters_or_edited
  ai-generated              → likely_ai_generated
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentations import ModelTransform
from data.datasets import build_datasets
from evaluate import make_loader, score_loader
from models.explain import assign_tier
from predict import load_checkpoint
from utils import get_device, load_config, project_root, set_seed

_EDITED = {"edited_filtered", "filtered_edited"}
_REAL = {"real", "real_blur", "real_live"}


def _expected_tier(generator: str, label: float) -> str:
    if generator in _EDITED:
        return "filters_or_edited"
    if generator in _REAL or label < 0.5:
        return "likely_real"
    return "likely_ai_generated"


def _summarize(samples, rec, t_real: float, t_ai: float) -> dict:
    y = np.array(rec["y"], dtype=np.float64)
    p = np.array(rec["pred"], dtype=np.float64)
    gens = [s.generator or "" for s in samples]
    auc = float(roc_auc_score(y, p)) if len(set(int(v) for v in y)) > 1 else float("nan")
    acc_05 = float(((p >= 0.5) == (y >= 0.5)).mean()) if len(y) else float("nan")

    rows = []
    correct_3 = 0
    by_gen: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    confusion = defaultdict(int)
    for i, sample in enumerate(samples):
        pred = float(p[i])
        tier, _ = assign_tier(pred, t_real, t_ai)
        expected = _expected_tier(sample.generator or "", float(y[i]))
        ok = tier == expected
        correct_3 += int(ok)
        by_gen[sample.generator or "other"]["n"] += 1
        by_gen[sample.generator or "other"]["correct"] += int(ok)
        confusion[f"{expected}->{tier}"] += 1
        rows.append(
            {
                "path": rec["path"][i],
                "generator": sample.generator,
                "label": int(y[i]),
                "pred": round(pred, 6),
                "tier": tier,
                "expected_tier": expected,
                "correct": ok,
            }
        )

    n = max(len(samples), 1)
    return {
        "n": len(samples),
        "n_real": int((y < 0.5).sum()),
        "n_ai": int((y >= 0.5).sum()),
        "auc": auc,
        "binary_acc_at_0.5": acc_05,
        "mean_pred_real": float(p[y < 0.5].mean()) if (y < 0.5).any() else None,
        "mean_pred_ai": float(p[y >= 0.5].mean()) if (y >= 0.5).any() else None,
        "three_way_acc": correct_3 / n,
        "by_generator": {
            g: {"n": v["n"], "acc": v["correct"] / max(v["n"], 1)} for g, v in sorted(by_gen.items())
        },
        "confusion": dict(confusion),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the human hold-out split.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"]["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = project_root() / ckpt

    model, saved = load_checkpoint(ckpt, device)
    thresholds = saved.get("thresholds") or cfg.get("thresholds") or {}
    t_real = float(thresholds.get("likely_real_max", 0.28))
    t_ai = float(thresholds.get("likely_ai_min", 0.70))

    _, val_ds, test_ds = build_datasets(
        saved, ModelTransform(saved, True), ModelTransform(saved, False)
    )
    ds = val_ds if args.split == "val" else test_ds
    loader = make_loader(ds.samples, saved, device, None, int(saved["train"]["batch_size"]))
    rec = score_loader(model, loader, device)
    summary = _summarize(ds.samples, rec, t_real, t_ai)
    summary["split"] = args.split
    summary["thresholds"] = {"likely_real_max": t_real, "likely_ai_min": t_ai}

    dest = project_root() / "outputs" / f"human_eval_{args.split}.json"
    payload = {k: v for k, v in summary.items() if k != "rows"}
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (project_root() / "outputs" / f"human_eval_{args.split}_rows.json").write_text(
        json.dumps(summary["rows"], indent=2), encoding="utf-8"
    )
    (project_root() / "outputs" / "human_eval.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(
        f"split={args.split} n={summary['n']} auc={summary['auc']:.4f} "
        f"acc@0.5={summary['binary_acc_at_0.5']:.4f} "
        f"three_way={summary['three_way_acc']:.4f}"
    )
    print("mean pred  real={:.3f}  ai={:.3f}".format(
        summary["mean_pred_real"] or float("nan"),
        summary["mean_pred_ai"] or float("nan"),
    ))
    print("by generator", json.dumps(summary["by_generator"], indent=2))
    print("confusion", json.dumps(summary["confusion"], indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
