"""Robustness evaluation, charts, and error analysis.

Primary metric is ROC AUC (threshold-free). Competition-style score:

    Final Score = 0.50 * AUC_clean + 0.50 * AUC_robust

where AUC_robust is the mean AUC across the transformed conditions.
Charts are regenerated on every run into outputs/charts/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentations import EVAL_CONDITIONS, ModelTransform
from data.datasets import AIGCImageDataset, collate_batch, collect_samples, dataset_summary
from models.explain import explain_record
from models.fusion import AIGCDetector
from predict import load_checkpoint
from utils import get_device, load_config, project_root, set_seed

sns.set_theme(style="whitegrid")


def _auc(y: list[float], p: list[float]) -> Optional[float]:
    if len(set(int(v) for v in y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return None


@torch.no_grad()
def score_loader(model: AIGCDetector, loader: DataLoader, device: torch.device) -> dict[str, list]:
    model.eval()
    records: dict[str, list] = {
        "path": [],
        "y": [],
        "pred": [],
        "semantic": [],
        "frequency": [],
        "generator": [],
    }
    for batch in tqdm(loader, leave=False, desc="score"):
        pixel_values = batch["pixel_values"].to(device)
        image_01 = batch["image_01"].to(device)
        out = model(pixel_values, image_01)
        records["path"].extend(batch["path"])
        records["y"].extend(batch["label"].tolist())
        records["pred"].extend(out["pred"].detach().cpu().tolist())
        records["semantic"].extend(out["semantic_score"].detach().cpu().tolist())
        freq = out["frequency_score"]
        if freq is None:
            records["frequency"].extend([None] * len(batch["path"]))
        else:
            records["frequency"].extend(freq.detach().cpu().tolist())
        records["generator"].extend(batch.get("generator") or [""] * len(batch["path"]))
    return records


def make_loader(samples, cfg, device, eval_transform=None, batch_size=16) -> DataLoader:
    tf = ModelTransform(cfg, augment=False, eval_transform=eval_transform)
    ds = AIGCImageDataset(samples, transform=tf)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers") or 0),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )


def plot_robustness_bars(rows: list[dict], out_path: Path) -> None:
    labels = [r["condition"] for r in rows]
    values = [r["auc"] if r["auc"] is not None else 0.0 for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2563eb" if r["condition"] == "Clean" else "#64748b" for r in rows]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("AIGC detection AUC by post-processing condition")
    ax.axhline(0.5, color="#94a3b8", linestyle="--", linewidth=1, label="chance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_roc_overlay(clean: dict, robust_y: list, robust_p: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if _auc(clean["y"], clean["pred"]) is not None:
        fpr, tpr, _ = roc_curve(clean["y"], clean["pred"])
        ax.plot(fpr, tpr, label=f"Clean (AUC={_auc(clean['y'], clean['pred']):.3f})")
    if _auc(robust_y, robust_p) is not None:
        fpr, tpr, _ = roc_curve(robust_y, robust_p)
        ax.plot(fpr, tpr, label=f"Transformed pooled (AUC={_auc(robust_y, robust_p):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: clean vs pooled transformed images")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confidence_hist(preds: list[float], lo: float, hi: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(preds, bins=20, range=(0, 1), color="#2563eb", edgecolor="white")
    ax.axvline(lo, color="#dc2626", linestyle="--", label=f"likely_real < {lo}")
    ax.axvline(hi, color="#16a34a", linestyle="--", label=f"likely_ai > {hi}")
    ax.set_xlabel("Predicted P(AIGC)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence-tier distribution on the clean test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_branch_scatter(records: dict, out_path: Path) -> None:
    sem = records["semantic"]
    freq = records["frequency"]
    if any(v is None for v in freq):
        return
    y = np.array(records["y"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        np.array(sem)[y == 0],
        np.array(freq)[y == 0],
        alpha=0.75,
        label="real",
        c="#2563eb",
    )
    ax.scatter(
        np.array(sem)[y == 1],
        np.array(freq)[y == 1],
        alpha=0.75,
        label="AI-generated",
        c="#dc2626",
    )
    ax.set_xlabel("semantic_score")
    ax.set_ylabel("frequency_score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Per-branch scores on the clean test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_error_analysis(
    records: dict,
    thresholds: dict,
    out_md: Path,
    example_dir: Path,
    n_examples: int = 4,
) -> dict:
    example_dir.mkdir(parents=True, exist_ok=True)
    for old in example_dir.glob("*"):
        if old.is_file():
            old.unlink()

    rows = []
    for i in range(len(records["y"])):
        extra = explain_record(
            records["pred"][i],
            records["semantic"][i],
            records["frequency"][i],
            likely_real_max=float(thresholds.get("likely_real_max", 0.4)),
            likely_ai_min=float(thresholds.get("likely_ai_min", 0.6)),
        )
        rows.append(
            {
                "path": records["path"][i],
                "y": int(records["y"][i]),
                "pred": records["pred"][i],
                "semantic": records["semantic"][i],
                "frequency": records["frequency"][i],
                **extra,
            }
        )

    fps = sorted([r for r in rows if r["y"] == 0 and r["pred"] >= 0.5], key=lambda r: -r["pred"])
    fns = sorted([r for r in rows if r["y"] == 1 and r["pred"] < 0.5], key=lambda r: r["pred"])

    def copy_examples(items: list[dict], tag: str) -> list[dict]:
        saved = []
        for i, rec in enumerate(items[:n_examples]):
            src = Path(rec["path"])
            dest_name = f"{tag}_{i:02d}_{src.name}"
            dest = example_dir / dest_name
            if src.is_file():
                shutil.copy2(src, dest)
            rec = dict(rec)
            rec["copied_to"] = str(dest) if src.is_file() else None
            saved.append(rec)
        return saved

    fp_saved = copy_examples(fps, "fp")
    fn_saved = copy_examples(fns, "fn")

    n = len(rows)
    n_fp, n_fn = len(fps), len(fns)
    lines = [
        "# Error analysis",
        "",
        "Examples below are **pulled from the current test split**, not invented.",
        f"Clean test size: {n}. False positives (real, pred>=0.5): {n_fp}. "
        f"False negatives (AI, pred<0.5): {n_fn}.",
        "",
        "## False positives (authentic images scored as AIGC)",
        "",
    ]
    def _fmt_row(rec: dict) -> None:
        freq_s = "n/a" if rec["frequency"] is None else f"{rec['frequency']:.3f}"
        lines.append(
            f"- `{rec['path']}`  pred={rec['pred']:.3f}  "
            f"semantic={rec['semantic']:.3f}  frequency={freq_s}  "
            f"tier={rec['tier']}"
        )
        lines.append(f"  {rec['explanation']}")

    if not fp_saved:
        lines.append("None at threshold 0.5 on this split.")
    for rec in fp_saved:
        _fmt_row(rec)
    lines += ["", "## False negatives (AI images scored as real)", ""]
    if not fn_saved:
        lines.append("None at threshold 0.5 on this split.")
    for rec in fn_saved:
        _fmt_row(rec)
    lines += [
        "",
        "## Trade-offs",
        "",
        "- The frequency branch is most useful when upsampling artifacts survive, but JPEG q30 "
        "and heavy blur flatten the spectrum and push more mass into `filters_or_edited`.",
        "- The semantic branch is more stable under crop/jitter/resize, and can over-trigger on "
        "unusual real photographs that sit far from CLIP's pretraining distribution.",
        "- The middle band is an automated policy hook for filters / retouching / partial AI "
        "edits, not a claim that the fused score is a perfectly calibrated probability.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return {"false_positives": fp_saved, "false_negatives": fn_saved, "n_fp": n_fp, "n_fn": n_fn}


def robustness_markdown(rows: list[dict], auc_clean: Optional[float], auc_robust: Optional[float], final: Optional[float]) -> str:
    lines = [
        "| Condition | Severity | AUC | N |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        auc = "n/a" if r["auc"] is None else f"{r['auc']:.4f}"
        sev = "-" if r["severity"] is None else str(r["severity"])
        lines.append(f"| {r['condition']} | {sev} | {auc} | {r['n']} |")
    lines += [
        "",
        f"AUC_clean = {auc_clean}",
        f"AUC_robust = {auc_robust}  (mean of transformed conditions)",
        f"Final Score = 0.50 * AUC_clean + 0.50 * AUC_robust = {final}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robustness and write charts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-transforms", action="store_true", help="Only score the clean test set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_samples is not None:
        cfg["eval"]["max_samples"] = args.max_samples
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)

    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"]["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = project_root() / ckpt
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    model, saved_cfg = load_checkpoint(ckpt, device)
    # Keep the trained architecture, but read eval/data paths from the live config.
    saved_cfg["paths"] = cfg["paths"]
    saved_cfg["dataset"] = cfg["dataset"]
    saved_cfg["eval"] = cfg["eval"]
    saved_cfg["train"]["num_workers"] = cfg["train"].get("num_workers", 0)

    test_samples = collect_samples(cfg, "test")
    print(f"test {dataset_summary(AIGCImageDataset(test_samples))}")
    batch_size = int(cfg["train"]["batch_size"])
    thresholds = saved_cfg.get("thresholds") or {}
    lo = float(thresholds.get("likely_real_max", 0.4))
    hi = float(thresholds.get("likely_ai_min", 0.6))

    conditions = EVAL_CONDITIONS if not args.skip_transforms else [EVAL_CONDITIONS[0]]
    rows = []
    clean_records = None
    robust_y: list[float] = []
    robust_p: list[float] = []
    unseen_auc = None

    for label, name, severity in conditions:
        eval_tf = None if name == "clean" else (name, severity)
        loader = make_loader(test_samples, saved_cfg, device, eval_tf, batch_size)
        rec = score_loader(model, loader, device)
        auc = _auc(rec["y"], rec["pred"])
        row = {"condition": label, "name": name, "severity": severity, "auc": auc, "n": len(rec["y"])}
        rows.append(row)
        print(f"  {label:20s}  AUC={auc if auc is not None else float('nan'):.4f}  n={row['n']}")
        if name == "clean":
            clean_records = rec
        else:
            robust_y.extend(rec["y"])
            robust_p.extend(rec["pred"])

    # Optional unseen-generator slice: generators that appear in test but not train.
    train_samples = collect_samples(cfg, "train")
    train_gens = {s.generator for s in train_samples if s.generator}
    test_only = sorted({s.generator for s in test_samples if s.generator} - train_gens - {"", "real"})
    if test_only and clean_records is not None:
        mask = [g in test_only for g in clean_records["generator"]]
        y_u = [y for y, m in zip(clean_records["y"], mask) if m]
        p_u = [p for p, m in zip(clean_records["pred"], mask) if m]
        unseen_auc = _auc(y_u, p_u)
        if unseen_auc is not None:
            rows.append(
                {
                    "condition": "Unseen generator",
                    "name": "unseen",
                    "severity": None,
                    "auc": unseen_auc,
                    "n": len(y_u),
                }
            )
            print(f"  {'Unseen generator':20s}  AUC={unseen_auc:.4f}  n={len(y_u)}  gens={test_only}")

    auc_clean = next((r["auc"] for r in rows if r["name"] == "clean"), None)
    transformed = [r["auc"] for r in rows if r["name"] not in {"clean", "unseen"} and r["auc"] is not None]
    auc_robust = float(np.mean(transformed)) if transformed else None
    final = None
    if auc_clean is not None and auc_robust is not None:
        final = 0.50 * auc_clean + 0.50 * auc_robust

    print("\n=== Final Score ===")
    print(f"AUC_clean  = {auc_clean}")
    print(f"AUC_robust = {auc_robust}")
    print(f"Final Score = 0.50 * AUC_clean + 0.50 * AUC_robust = {final}")

    out_dir = project_root() / "outputs"
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    table_md = robustness_markdown(rows, auc_clean, auc_robust, final)
    (out_dir / "robustness_table.md").write_text(table_md, encoding="utf-8")
    print(table_md)

    plot_robustness_bars(rows, chart_dir / "robustness_bars.png")
    if clean_records is not None:
        plot_roc_overlay(clean_records, robust_y, robust_p, chart_dir / "roc_overlay.png")
        plot_confidence_hist(clean_records["pred"], lo, hi, chart_dir / "confidence_histogram.png")
        plot_branch_scatter(clean_records, chart_dir / "branch_scatter.png")
        err = write_error_analysis(
            clean_records,
            thresholds,
            out_dir / "error_analysis.md",
            out_dir / "error_examples",
        )
    else:
        err = {}

    summary = {
        "auc_clean": auc_clean,
        "auc_robust": auc_robust,
        "final_score": final,
        "unseen_generator_auc": unseen_auc,
        "rows": rows,
        "error_counts": {"n_fp": err.get("n_fp"), "n_fn": err.get("n_fn")},
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"charts -> {chart_dir}")
    print(f"error analysis -> {out_dir / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
