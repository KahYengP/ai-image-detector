"""Shared helpers: config loading, seeding, device selection, image discovery."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(folder: str | Path, recursive: bool = True) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    raw = Path(cfg["paths"][key])
    return raw if raw.is_absolute() else project_root() / raw


def resolve_checkpoint(cfg: dict[str, Any] | None = None, explicit: str | Path | None = None) -> Path:
    """Find outputs/best.pt even if config still has another machine's absolute path."""
    cfg = cfg or load_config()
    raw = Path(explicit) if explicit else Path(str((cfg.get("paths") or {}).get("checkpoint") or "outputs/best.pt"))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        candidates.append(project_root() / "outputs" / raw.name)
    else:
        candidates.append(project_root() / raw)
    candidates.append(project_root() / "outputs" / "best.pt")
    seen: set[Path] = set()
    unique: list[Path] = []
    too_small: list[tuple[Path, int]] = []
    min_bytes = 10 * 1024 * 1024
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size < min_bytes:
            too_small.append((path, size))
            continue
        return path
    tried = "\n".join(f"  - {p}" for p in unique)
    if too_small:
        detail = "\n".join(f"  - {p} ({n} bytes)" for p, n in too_small)
        raise FileNotFoundError(
            "outputs/best.pt is empty or incomplete (pickle error: Ran out of input). "
            "Do not create a blank best.pt. Copy the real trained file from the original "
            "computer — it is about 338 MB — into this project's outputs folder.\n"
            f"Found unusable file(s):\n{detail}"
        )
    raise FileNotFoundError(
        "Detection model not found. Clone does not include outputs/best.pt. "
        "Copy the trained checkpoint into this project folder.\n"
        f"Looked in:\n{tried}"
    )


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
