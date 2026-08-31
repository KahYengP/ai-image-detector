"""Training loop for the two-branch AIGC detector.

Loss = BCE(fused) + aux_weight * BCE(semantic) + aux_weight * BCE(frequency)
so the per-branch scores used for explainability are actually trained, not
just frozen by-products of the fusion MLP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentations import ModelTransform
from data.datasets import build_datasets, collate_batch, dataset_summary
from models.fusion import AIGCDetector
from utils import count_parameters, get_device, load_config, project_root, set_seed


_PHOTO_GENS = {"real", "real_blur", "real_live", "edited_filtered", "filtered_edited"}
_EDITED_GENS = {"edited_filtered", "filtered_edited"}
_BLUR_GENS = {"real_blur"}
_LIVE_REAL_GENS = {"real_live"}
_PRODUCT_AI_GENS = {"ai_generated_product"}
_LIVE_AI_GENS = {"ai_generated_live"}


def _sample_weights(labels: torch.Tensor, generators: list[str], real_class_weight: float) -> torch.Tensor:
    """Heavier weight on scarce / easily-confused groups."""
    weights = []
    for y, gen in zip(labels.tolist(), generators):
        g = (gen or "").lower()
        if g in _PRODUCT_AI_GENS:
            weights.append(2.8)
        elif g in _LIVE_AI_GENS:
            weights.append(2.6)
        elif g in _LIVE_REAL_GENS:
            weights.append(2.2)
        elif g in _BLUR_GENS:
            weights.append(3.2)
        elif g in _EDITED_GENS:
            weights.append(2.4)
        elif y < 0.5:
            weights.append(float(real_class_weight))
        else:
            weights.append(1.0)
    return torch.tensor(weights, device=labels.device, dtype=labels.dtype)


def run_epoch(
    model: AIGCDetector,
    loader: DataLoader,
    device: torch.device,
    aux_weight: float,
    real_class_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n = 0
    ys: list[float] = []
    ps: list[float] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if train else "eval"):
            pixel_values = batch["pixel_values"].to(device)
            image_01 = batch["image_01"].to(device)
            labels = batch["label"].to(device)
            targets = batch["target"].to(device) if "target" in batch else labels
            generators = list(batch.get("generator") or [""] * labels.size(0))
            out = model(pixel_values, image_01)
            sample_w = _sample_weights(labels, generators, real_class_weight)
            loss = nn.functional.binary_cross_entropy_with_logits(
                out["logit"], targets, weight=sample_w
            )
            if aux_weight > 0:
                loss = loss + aux_weight * nn.functional.binary_cross_entropy_with_logits(
                    out["semantic_logit"], targets, weight=sample_w
                )
                if out["frequency_logit"] is not None:
                    loss = loss + aux_weight * nn.functional.binary_cross_entropy_with_logits(
                        out["frequency_logit"], targets, weight=sample_w
                    )
            # Edited stays below likely_ai; product/live AI stays out of the edited band.
            if train:
                pred = out["pred"]
                is_edited = torch.tensor(
                    [g in _EDITED_GENS for g in generators], device=device, dtype=pred.dtype
                )
                is_blur = torch.tensor(
                    [g in _BLUR_GENS for g in generators], device=device, dtype=pred.dtype
                )
                is_product_ai = torch.tensor(
                    [g in _PRODUCT_AI_GENS for g in generators], device=device, dtype=pred.dtype
                )
                is_live_ai = torch.tensor(
                    [g in _LIVE_AI_GENS for g in generators], device=device, dtype=pred.dtype
                )
                is_live_real = torch.tensor(
                    [g in _LIVE_REAL_GENS for g in generators], device=device, dtype=pred.dtype
                )
                is_photo = (labels < 0.5).to(pred.dtype)
                cap = torch.full_like(pred, 0.68)
                cap = torch.where(is_edited > 0, torch.full_like(pred, 0.52), cap)
                cap = torch.where(is_blur > 0, torch.full_like(pred, 0.40), cap)
                cap = torch.where(is_live_real > 0, torch.full_like(pred, 0.32), cap)
                over = torch.relu(pred - cap) * is_photo
                under_product = torch.relu(0.76 - pred) * is_product_ai
                under_live = torch.relu(0.72 - pred) * is_live_ai
                loss = loss + 0.85 * (over * over).mean()
                loss = loss + 1.15 * (under_product * under_product).mean()
                loss = loss + 1.05 * (under_live * under_live).mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
            ys.extend(labels.detach().cpu().tolist())
            ps.extend(out["pred"].detach().cpu().tolist())

    metrics = {"loss": total_loss / max(n, 1)}
    try:
        metrics["auc"] = float(roc_auc_score(ys, ps)) if len(set(int(y) for y in ys)) > 1 else float("nan")
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AIGC detector.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--clip-only", action="store_true", help="Disable the frequency branch (baseline).")
    parser.add_argument("--no-aug", action="store_true", help="Disable training-time augmentations.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--output", default=None, help="Checkpoint path (default: paths.checkpoint).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from paths.checkpoint so the 48 human photos fine-tune the existing model.",
    )
    parser.add_argument(
        "--reset-best",
        action="store_true",
        help="When resuming, track best val AUC from this run only (do not keep the old best).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.max_samples is not None:
        cfg["train"]["max_samples"] = args.max_samples
        cfg["eval"]["max_samples"] = args.max_samples
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.clip_only:
        cfg["model"]["use_frequency_branch"] = False
    if args.no_aug:
        cfg["augmentations"]["enabled"] = False

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)
    print(f"device={device}")

    train_tf = ModelTransform(cfg, augment=True)
    eval_tf = ModelTransform(cfg, augment=False)
    train_ds, val_ds, test_ds = build_datasets(cfg, train_tf, eval_tf)
    print(f"train {dataset_summary(train_ds)}")
    print(f"val   {dataset_summary(val_ds)}")
    print(f"test  {dataset_summary(test_ds)}")
    from collections import Counter

    print("train sources", dict(Counter(s.source for s in train_ds.samples)))
    print("val sources  ", dict(Counter(s.source for s in val_ds.samples)))

    workers = int(cfg["train"].get("num_workers") or 0)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_batch,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_batch,
        pin_memory=pin,
    )

    model = AIGCDetector(cfg).to(device)
    total, trainable = count_parameters(model)
    print(f"params total={total:,} trainable={trainable:,} (<2B constraint: {total < 2_000_000_000})")
    if total >= 2_000_000_000:
        raise SystemExit("Model exceeds the 2B parameter limit.")

    ckpt_path = Path(args.output) if args.output else Path(cfg["paths"]["checkpoint"])
    if not ckpt_path.is_absolute():
        ckpt_path = project_root() / ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    if args.resume:
        if not ckpt_path.is_file():
            raise SystemExit(f"--resume set but checkpoint not found: {ckpt_path}")
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state"], strict=False)
        print(
            f"resumed {ckpt_path} epoch={blob.get('epoch')} "
            f"val_auc={blob.get('val_auc')}"
        )
        prev = blob.get("val_auc")
        if prev is not None and prev == prev and not args.reset_best:
            best_auc = float(prev)

    base_lr = float(cfg["train"]["lr"])
    clip_ids = {id(p) for p in model.semantic.vision.parameters()}
    clip_params = [p for p in model.parameters() if p.requires_grad and id(p) in clip_ids]
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in clip_ids]
    param_groups = [{"params": head_params, "lr": base_lr}]
    if clip_params:
        param_groups.append({"params": clip_params, "lr": base_lr * 0.1})
        print(f"param groups: heads={len(head_params)} clip_last_block={len(clip_params)}")
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )
    n_epochs = int(cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(n_epochs, 1), eta_min=base_lr * 0.05
    )
    aux_weight = float(cfg["model"].get("aux_loss_weight", 0.3))
    real_w = float(cfg["train"].get("real_class_weight", 1.0))

    history = []
    for epoch in range(1, n_epochs + 1):
        train_m = run_epoch(model, train_loader, device, aux_weight, real_w, optimizer)
        val_m = run_epoch(model, val_loader, device, aux_weight, real_w, None)
        scheduler.step()
        row = {"epoch": epoch, "train": train_m, "val": val_m, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(
            f"epoch {epoch:02d}  train_loss={train_m['loss']:.4f} train_auc={train_m['auc']:.4f}  "
            f"val_loss={val_m['loss']:.4f} val_auc={val_m['auc']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        val_auc = val_m["auc"]
        if val_auc != val_auc:  # NaN: keep last weights but still save if nothing else
            val_auc = -1.0
        if val_auc >= best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "val_auc": val_m["auc"],
                    "epoch": epoch,
                    "use_frequency_branch": model.use_frequency_branch,
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path} (val_auc={val_m['auc']:.4f})")

    hist_path = ckpt_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done. best_val_auc={best_auc:.4f}  checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
