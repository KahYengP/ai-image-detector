"""Dataset loaders for CIFAKE, SID_Set, and WildFake.

Label convention used everywhere downstream:
  0 = authentic / non-AIGC
  1 = AI-generated (AIGC)

The official demonstration split (COCO val2017 + DALL·E Advanced) is never
returned by the WildFake loader — those subsets are evaluation-only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

from utils import IMAGE_EXTS, project_root, resolve_path

# Folder-name tokens used to infer labels when walking WildFake-style trees.
_REAL_TOKENS = {
    "real",
    "reals",
    "authentic",
    "nature",
    "natural",
    "coco",
    "ffhq",
    "imagenet",
    "lsun",
    "celeba",
    "celeba-hq",
    "celebahq",
    "afhq",
    "laion",
    "openimages",
    "flickr",
}
_FAKE_TOKENS = {
    "fake",
    "fakes",
    "ai",
    "aigc",
    "generated",
    "synthetic",
    "full_synthetic",
    "gan",
    "gans",
    "diffusion",
    "dalle",
    "dall-e",
    "dall_e",
    "midjourney",
    "stable-diffusion",
    "stablediffusion",
    "sd",
    "imagen",
    "adm",
    "tampered",
}


@dataclass
class Sample:
    path: Optional[Path]
    label: int
    source: str
    # Optional Hugging Face row pointer so we do not materialize the whole set.
    hf_index: Optional[int] = None
    generator: Optional[str] = None
    # Soft training target in [0, 1]. Falls back to `label` when unset.
    target: Optional[float] = None


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _iter_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _normalize_path_key(path: Path | str) -> str:
    return str(path).replace("\\", "/").lower()


def is_forbidden_eval_split(path: Path | str, exclude_tokens: list[str]) -> bool:
    """True if this file belongs to the demonstration-only WildFake subset.

    The brief forbids training on COCO val2017 (non-AIGC) and DALL·E Advanced
    (AIGC). Matching is path-based so it works regardless of how the dump is nested.
    """
    key = _normalize_path_key(path)
    for token in exclude_tokens:
        t = token.replace("\\", "/").lower()
        if t and t in key:
            return True
    # Extra belt-and-suspenders for the two named subsets.
    if "val2017" in key and "coco" in key:
        return True
    if "advanced" in key and ("dalle" in key or "dall-e" in key or "dall_e" in key):
        return True
    return False


def _label_from_folder_name(folder_name: str) -> Optional[int]:
    name = folder_name.lower()
    if name in {"real", "reals", "authentic", "nature", "natural"}:
        return 0
    if name in {"fake", "fakes", "ai", "aigc", "generated", "synthetic", "full_synthetic"}:
        return 1
    return None


# ---------------------------------------------------------------------------
# CIFAKE  (Kaggle: birdy654/cifake-real-and-ai-generated-synthetic-images)
# Layout:  {root}/{train,test}/{REAL,FAKE}/*.jpg
# Native resolution is 32x32 — fine for a pipeline smoke test, weak for FFT.
# ---------------------------------------------------------------------------
def load_cifake(root: Path, split: str) -> list[Sample]:
    split_dir = root / split
    if not split_dir.is_dir():
        # Some dumps nest an extra folder, e.g. cifake/cifake/train/...
        matches = list(root.rglob(split))
        split_dir = next((p for p in matches if p.is_dir()), split_dir)
    samples: list[Sample] = []
    seen: set[Path] = set()
    for class_name, label in (("REAL", 0), ("FAKE", 1), ("real", 0), ("fake", 1)):
        class_dir = split_dir / class_name
        for path in _iter_images(class_dir):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            samples.append(Sample(path=path, label=label, source="cifake"))
    if not samples:
        raise FileNotFoundError(
            f"No CIFAKE images found under {split_dir}. "
            "Expected train/REAL, train/FAKE, test/REAL, test/FAKE. "
            "Download from https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images "
            "or run: python scripts/make_smoke_dataset.py"
        )
    return samples


# ---------------------------------------------------------------------------
# SID_Set  (Hugging Face: saberzl/SID_Set)
# Labels: 0 real, 1 full synthetic, 2 tampered.
# Also supports the Google Drive folder layout:
#   {root}/{train,validation}/{real,full_synthetic,tampered}/
# ---------------------------------------------------------------------------
def load_sid_set_local(root: Path, split: str, include_tampered: bool) -> list[Sample]:
    split_aliases = {
        "train": ["train"],
        "val": ["validation", "val"],
        "test": ["test", "validation", "val"],
    }
    split_dir = None
    for name in split_aliases.get(split, [split]):
        candidate = root / name
        if candidate.is_dir():
            split_dir = candidate
            break
    if split_dir is None:
        raise FileNotFoundError(f"SID_Set split folder not found under {root} (looked for {split})")

    mapping = [("real", 0), ("full_synthetic", 1)]
    if include_tampered:
        mapping.append(("tampered", 1))

    samples: list[Sample] = []
    for folder_name, label in mapping:
        for path in _iter_images(split_dir / folder_name):
            samples.append(Sample(path=path, label=label, source="sid_set", generator=folder_name))
    if not samples:
        raise FileNotFoundError(f"No SID_Set images found under {split_dir}")
    return samples


def load_sid_set_hf(split: str, include_tampered: bool, max_samples: Optional[int]) -> list[Sample]:
    """Stream SID_Set from Hugging Face. Requires `datasets` and network on first run."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` to load SID_Set from Hugging Face.") from exc

    hf_split = {"val": "validation", "test": "validation"}.get(split, split)
    ds = load_dataset("saberzl/SID_Set", split=hf_split)
    samples: list[Sample] = []
    for i in range(len(ds)):
        label = int(ds[i]["label"])
        if label == 2 and not include_tampered:
            continue
        y = 0 if label == 0 else 1
        generator = {0: "real", 1: "full_synthetic", 2: "tampered"}.get(label)
        samples.append(
            Sample(path=None, label=y, source="sid_set", hf_index=i, generator=generator)
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    # Stash the HF object on the list so the Dataset can retrieve pixels later.
    samples._hf_dataset = ds  # type: ignore[attr-defined]
    return samples


# ---------------------------------------------------------------------------
# WildFake  (ModelScope: hy2628982280/WildFake)
# Walk a local dump and infer labels from folder names. Skip the forbidden
# demonstration subset so it cannot leak into training.
# ---------------------------------------------------------------------------
def _infer_wildfake_label(path: Path) -> Optional[int]:
    parts = [_normalize_path_key(p) for p in path.parts]
    for part in reversed(parts[:-1]):
        token = part.replace(" ", "_")
        if token in _REAL_TOKENS:
            return 0
        if token in _FAKE_TOKENS:
            return 1
    return None


def _guess_generator(path: Path) -> Optional[str]:
    parts = [_normalize_path_key(p) for p in path.parts]
    for part in reversed(parts[:-1]):
        if part in _FAKE_TOKENS or part in _REAL_TOKENS:
            return part
    return None


# Soft targets match the three inference bands: likely_real / filters_or_edited / likely_ai.
# Edited/filtered photos are NOT AIGC — keep them in the middle band, well below likely_ai.
# Product/live AI must sit high so they are not parked in filters_or_edited.
_HUMAN_SOFT_TARGET = {
    "real": 0.08,
    "real_blur": 0.08,
    "real_live": 0.08,
    "edited_filtered": 0.36,
    "filtered_edited": 0.36,
    "ai_generated": 0.92,
    "ai_generated_live": 0.94,
    "ai_generated_product": 0.94,
}
# Repeat scarce / easily-confused groups so the trainer sees them every epoch.
_HUMAN_MIN_TRAIN = {
    "real_blur": 40,
    "real_live": 55,
    "edited_filtered": 40,
    "filtered_edited": 110,
    "ai_generated_live": 55,
    "ai_generated_product": 90,
}


def _human_label_from_name(stem: str) -> Optional[tuple[int, str]]:
    """Map filename prefixes to (label, generator). More specific prefixes first."""
    name = stem.lower().replace(" ", "-").replace("_", "-")
    if name.startswith("ai-generated-live") or name.startswith("aigenerated-live"):
        return 1, "ai_generated_live"
    if name.startswith("ai-generated-product") or name.startswith("aigenerated-product"):
        return 1, "ai_generated_product"
    if name.startswith("real-live") or name.startswith("reallive"):
        return 0, "real_live"
    if name.startswith("ai-generated") or name.startswith("aigenerated"):
        return 1, "ai_generated"
    if name.startswith("filtered-edited") or name.startswith("filterededited"):
        return 0, "filtered_edited"
    if name.startswith("edited-filtered") or name.startswith("edited"):
        return 0, "edited_filtered"
    if name.startswith("real-blur") or name.startswith("realblur"):
        return 0, "real_blur"
    if name.startswith("real"):
        return 0, "real"
    return None


def load_human_labeled(root: Path, soft_labels: bool = True) -> list[Sample]:
    """Load the user-labeled folder (`real*.jpeg`, `ai-generated*.jpeg`, ...)."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Human-labeled folder not found: {root}. "
            "Point config.yaml paths.human at `train images`."
        )
    samples: list[Sample] = []
    unlabeled = 0
    by_gen: dict[str, int] = {}
    for path in _iter_images(root):
        parsed = _human_label_from_name(path.stem)
        if parsed is None:
            unlabeled += 1
            continue
        label, generator = parsed
        target = _HUMAN_SOFT_TARGET.get(generator, float(label)) if soft_labels else float(label)
        samples.append(
            Sample(
                path=path,
                label=label,
                source="human",
                generator=generator,
                target=target,
            )
        )
        by_gen[generator] = by_gen.get(generator, 0) + 1
    if not samples:
        raise FileNotFoundError(
            f"No labeled images under {root} (unlabeled={unlabeled}). "
            "Name files real*, real-live*, real-blur*, edited-filtered*, filtered_edited*, "
            "ai-generated*, ai-generated_live*, or ai-generated_product*."
        )
    print(
        f"[human] loaded={len(samples)} unlabeled={unlabeled} "
        f"real={sum(1 for s in samples if s.label == 0)} "
        f"ai={sum(1 for s in samples if s.label == 1)} "
        f"by_generator={by_gen}"
    )
    return samples


def load_all_human(cfg: dict[str, Any], soft_labels: bool = True) -> list[Sample]:
    """Merge labeled files from `paths.human` plus optional `paths.human_extra`."""
    roots: list[Path] = []
    try:
        roots.append(resolve_path(cfg, "human"))
    except (KeyError, TypeError):
        pass
    extra = (cfg.get("paths") or {}).get("human_extra")
    if extra:
        items = extra if isinstance(extra, list) else [extra]
        for raw in items:
            p = Path(raw)
            if not p.is_absolute():
                p = project_root() / p
            roots.append(p)
    seen: set[Path] = set()
    out: list[Sample] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            part = load_human_labeled(root, soft_labels=soft_labels)
        except FileNotFoundError:
            continue
        for sample in part:
            key = sample.path.resolve() if sample.path else None
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(sample)
    if not out:
        raise FileNotFoundError("No human-labeled images found under paths.human / human_extra.")
    print(f"[human] merged unique={len(out)} folders={len(roots)}")
    return out


# Scarce new types: keep unique files in train so the visual pattern is learned.
_TRAIN_PRIORITY_GENS = {"ai_generated_live", "ai_generated_product", "real_live"}


def _split_human(
    samples: list[Sample],
    split: str,
    val_fraction: float,
    seed: int,
    test_fraction: float = 0.0,
) -> list[Sample]:
    """Stratified train/val/(optional test) split by filename-prefix generator."""
    rng = random.Random(seed)
    by_gen: dict[str, list[Sample]] = {}
    for sample in samples:
        by_gen.setdefault(sample.generator or "other", []).append(sample)
    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []
    for gen, bucket in by_gen.items():
        shuffled = bucket[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        if gen in _TRAIN_PRIORITY_GENS:
            n_test = 0
            n_val = 1 if n >= 12 else 0
        else:
            n_test = max(1, int(round(n * test_fraction))) if test_fraction > 0 and n >= 6 else 0
            remain = n - n_test
            n_val = max(1, int(round(remain * val_fraction))) if remain >= 4 else 0
            if n_test + n_val >= n:
                n_val = max(0, n - n_test - 1)
        test.extend(shuffled[:n_test])
        val.extend(shuffled[n_test : n_test + n_val])
        train.extend(shuffled[n_test + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}.get(split, train)


def _balance_rare_generators(samples: list[Sample], max_factor: int = 8, seed: int = 0) -> list[Sample]:
    """Repeat scarce / easily-confused groups so they are not drowned by other stills."""
    by_gen: dict[str, list[Sample]] = {}
    for sample in samples:
        by_gen.setdefault(sample.generator or "other", []).append(sample)
    if not by_gen:
        return samples
    out: list[Sample] = []
    for gen, bucket in by_gen.items():
        n = max(len(bucket), 1)
        want = int(_HUMAN_MIN_TRAIN.get(gen, n))
        cap = 10 if gen in _TRAIN_PRIORITY_GENS else max_factor
        factor = max(1, int(round(want / n))) if want > n else 1
        factor = min(cap, factor)
        out.extend(bucket * factor)
    rng = random.Random(seed)
    rng.shuffle(out)
    print(f"[human] rare-class balance unique={len(samples)} -> {len(out)}")
    return out


def load_wildfake(root: Path, exclude_tokens: list[str]) -> list[Sample]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"WildFake root not found: {root}. "
            "Download from https://modelscope.cn/datasets/hy2628982280/WildFake/summary "
            "(use the 中/En toggle) and point config.yaml paths.wildfake at the dump."
        )
    samples: list[Sample] = []
    skipped_forbidden = 0
    unlabeled = 0
    for path in _iter_images(root):
        if is_forbidden_eval_split(path, exclude_tokens):
            skipped_forbidden += 1
            continue
        label = _infer_wildfake_label(path)
        if label is None:
            unlabeled += 1
            continue
        samples.append(
            Sample(path=path, label=label, source="wildfake", generator=_guess_generator(path))
        )
    if not samples:
        raise FileNotFoundError(
            f"No labeled WildFake images under {root} "
            f"(skipped_forbidden={skipped_forbidden}, unlabeled={unlabeled})."
        )
    print(
        f"[wildfake] loaded={len(samples)} skipped_forbidden={skipped_forbidden} unlabeled={unlabeled}"
    )
    return samples


def _subsample(samples: list[Sample], max_samples: Optional[int], seed: int) -> list[Sample]:
    if max_samples is None or max_samples >= len(samples):
        return samples
    rng = random.Random(seed)
    return rng.sample(samples, max_samples)


def _cap_per_class(samples: list[Sample], max_per_class: Optional[int], seed: int) -> list[Sample]:
    if max_per_class is None:
        return samples
    rng = random.Random(seed)
    out: list[Sample] = []
    for y in (0, 1):
        bucket = [s for s in samples if s.label == y]
        if len(bucket) > max_per_class:
            bucket = rng.sample(bucket, max_per_class)
        out.extend(bucket)
    rng.shuffle(out)
    return out


def _cifake_train_val(cfg: dict[str, Any]) -> tuple[list[Sample], list[Sample]]:
    """Carve val from CIFAKE train so the official test split stays untouched."""
    train_pool = load_cifake(resolve_path(cfg, "cifake"), "train")
    rng = random.Random(int(cfg.get("seed", 42)))
    real = [s for s in train_pool if s.label == 0]
    ai = [s for s in train_pool if s.label == 1]
    rng.shuffle(real)
    rng.shuffle(ai)
    frac = float(cfg["train"].get("val_fraction", 0.1))
    n_val_real = max(1, int(len(real) * frac)) if real else 0
    n_val_ai = max(1, int(len(ai) * frac)) if ai else 0
    val = real[:n_val_real] + ai[:n_val_ai]
    train = real[n_val_real:] + ai[n_val_ai:]
    rng.shuffle(val)
    rng.shuffle(train)
    return train, val


def collect_samples(cfg: dict[str, Any], split: str) -> list[Sample]:
    """Build the sample list for `split` in {train, val, test} from config."""
    name = cfg["dataset"]["name"]
    seed = int(cfg.get("seed", 42))
    max_samples = cfg["train"].get("max_samples") if split == "train" else cfg["eval"].get("max_samples")
    include_tampered = bool(cfg["dataset"].get("sid_set_include_tampered", False))
    exclude_tokens = list(cfg["dataset"].get("wildfake_exclude") or [])

    samples: list[Sample] = []

    def maybe_cifake() -> list[Sample]:
        root = resolve_path(cfg, "cifake")
        cifake_split = "train" if split == "train" else "test"
        return load_cifake(root, cifake_split)

    def maybe_sid() -> list[Sample]:
        root = resolve_path(cfg, "sid_set")
        if root.is_dir() and any(root.iterdir()):
            return load_sid_set_local(root, split, include_tampered)
        return load_sid_set_hf(split, include_tampered, max_samples)

    def maybe_wildfake() -> list[Sample]:
        # WildFake dumps are usually not pre-split; we split later.
        return load_wildfake(resolve_path(cfg, "wildfake"), exclude_tokens)

    def maybe_sid_local() -> list[Sample]:
        """Combined mode never streams the 210k SID_Set dump from Hugging Face."""
        root = resolve_path(cfg, "sid_set")
        if not (root.is_dir() and any(p.is_dir() or p.is_file() for p in root.iterdir() if p.name != ".gitkeep")):
            raise FileNotFoundError(
                f"No local SID_Set dump at {root}. Run: python scripts/download_sidset.py"
            )
        return load_sid_set_local(root, split, include_tampered)

    soft_human = bool(cfg["dataset"].get("human_soft_labels", True))

    if name == "human":
        human_all = load_all_human(cfg, soft_labels=soft_human)
        val_frac = float(cfg["dataset"].get("human_val_fraction", cfg["train"].get("val_fraction", 0.15)))
        test_frac = float(cfg["dataset"].get("human_test_fraction", 0.15))
        samples = _split_human(human_all, split, val_frac, seed + 7, test_fraction=test_frac)
        if split == "train" and bool(cfg["dataset"].get("human_balance_rare", True)):
            samples = _balance_rare_generators(samples, max_factor=8, seed=seed)
        return _subsample(samples, max_samples, seed)
    if name == "cifake":
        # Official test split is held out. Val is carved from train so we never
        # tune on the test set.
        if split == "test":
            samples = maybe_cifake()
            return _subsample(samples, max_samples, seed)
        train_split, val_split = _cifake_train_val(cfg)
        samples = train_split if split == "train" else val_split
        return _subsample(samples, max_samples, seed)
    elif name == "sid_set":
        samples = maybe_sid()
    elif name == "wildfake":
        samples = maybe_wildfake()
    elif name == "combined":
        # Keep official splits: CIFAKE train/test, SID_Set train/validation.
        # Do not pool-then-reshuffle (that mixed CIFAKE test into train).
        cifake_cap = cfg["dataset"].get("combined_cifake_max_per_class")
        cifake_cap = int(cifake_cap) if cifake_cap is not None else None
        try:
            if split == "test":
                cifake_part = load_cifake(resolve_path(cfg, "cifake"), "test")
            else:
                train_split, val_split = _cifake_train_val(cfg)
                cifake_part = train_split if split == "train" else val_split
            samples.extend(_cap_per_class(cifake_part, cifake_cap, seed + 1))
        except FileNotFoundError as exc:
            print(f"[data] skipping CIFAKE in combined: {exc}")
        try:
            # SID validation is the high-res photo hold-out. Do not reuse it as test.
            if split != "test":
                samples.extend(maybe_sid_local())
        except FileNotFoundError as exc:
            print(f"[data] skipping SID_Set in combined: {exc}")
        try:
            wf = maybe_wildfake()
            rng = random.Random(seed)
            shuffled = wf[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_val = max(1, int(n * float(cfg["train"].get("val_fraction", 0.1))))
            n_test = max(1, int(n * 0.15))
            bucket = {
                "test": shuffled[:n_test],
                "val": shuffled[n_test : n_test + n_val],
                "train": shuffled[n_test + n_val :],
            }[split]
            samples.extend(bucket)
        except FileNotFoundError as exc:
            print(f"[data] skipping WildFake in combined: {exc}")
        try:
            human_all = load_all_human(cfg, soft_labels=soft_human)
            val_frac = float(cfg["dataset"].get("human_val_fraction", cfg["train"].get("val_fraction", 0.1)))
            human_part = _split_human(human_all, split, val_frac, seed + 7, test_fraction=0.0)
            if split == "train":
                if bool(cfg["dataset"].get("human_balance_rare", False)):
                    human_part = _balance_rare_generators(human_part, max_factor=3, seed=seed)
                factor = int(cfg["dataset"].get("human_oversample") or 1)
                if factor > 1 and human_part:
                    print(
                        f"[human] oversample x{factor} unique={len(human_part)} "
                        f"-> {len(human_part) * factor}"
                    )
                    human_part = human_part * factor
            samples.extend(human_part)
        except (FileNotFoundError, KeyError) as exc:
            print(f"[data] skipping human-labeled set: {exc}")
        if not samples:
            raise FileNotFoundError("combined dataset is empty — point config.yaml at at least one dump.")
        return _subsample(samples, max_samples, seed)
    else:
        raise ValueError(f"Unknown dataset.name: {name}")

    # CIFAKE has an official test split. SID_Set has train/validation.
    # WildFake / combined get a deterministic hold-out from the pool.
    needs_manual_split = name in {"wildfake", "combined"} or (
        name == "sid_set" and split == "test"
    )
    if name == "sid_set" and split in {"train", "val"}:
        samples = _subsample(samples, max_samples, seed)
        return samples

    if needs_manual_split or name == "wildfake":
        rng = random.Random(seed)
        shuffled = samples[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, int(n * float(cfg["train"].get("val_fraction", 0.1))))
        n_test = max(1, int(n * 0.15))
        test = shuffled[:n_test]
        val = shuffled[n_test : n_test + n_val]
        train = shuffled[n_test + n_val :]
        bucket = {"train": train, "val": val, "test": test}[split]
        return _subsample(bucket, max_samples, seed)

    return _subsample(samples, max_samples, seed)


class AIGCImageDataset(Dataset):
    """Returns a dict with CLIP-normalized pixels, 0-1 RGB for the FFT branch, and the label."""

    def __init__(
        self,
        samples: list[Sample],
        transform: Optional[Callable] = None,
        hf_dataset: Any = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.hf_dataset = hf_dataset or getattr(samples, "_hf_dataset", None)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, sample: Sample) -> Image.Image:
        if sample.path is not None:
            return _open_rgb(sample.path)
        if self.hf_dataset is None or sample.hf_index is None:
            raise RuntimeError(f"Cannot load sample with no path and no HF pointer: {sample}")
        row = self.hf_dataset[sample.hf_index]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        return image.convert("RGB")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = self._load_image(sample)
        if sample.path is not None:
            try:
                display_path = str(sample.path.resolve().relative_to(project_root()))
            except ValueError:
                display_path = str(sample.path)
        else:
            display_path = f"hf://{sample.source}/{sample.hf_index}"
        item = {
            "image": image,
            "label": sample.label,
            "target": float(sample.target) if sample.target is not None else float(sample.label),
            "path": display_path,
            "source": sample.source,
            "generator": sample.generator or "",
        }
        if self.transform is not None:
            item.update(self.transform(image, generator=sample.generator))
            item.pop("image", None)
        return item


def build_datasets(
    cfg: dict[str, Any],
    train_transform: Optional[Callable],
    eval_transform: Optional[Callable],
) -> tuple[AIGCImageDataset, AIGCImageDataset, AIGCImageDataset]:
    train_samples = collect_samples(cfg, "train")
    try:
        val_samples = collect_samples(cfg, "val")
    except FileNotFoundError:
        val_samples = []
    try:
        test_samples = collect_samples(cfg, "test")
    except FileNotFoundError:
        test_samples = []

    # If a source has no dedicated val split (shouldn't happen after collect_samples),
    # carve one from train.
    if not val_samples and train_samples:
        rng = random.Random(int(cfg.get("seed", 42)))
        shuffled = train_samples[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * float(cfg["train"].get("val_fraction", 0.1))))
        val_samples = shuffled[:n_val]
        train_samples = shuffled[n_val:]

    if not test_samples:
        test_samples = val_samples

    hf_ds = getattr(train_samples, "_hf_dataset", None)
    return (
        AIGCImageDataset(train_samples, transform=train_transform, hf_dataset=hf_ds),
        AIGCImageDataset(val_samples, transform=eval_transform, hf_dataset=hf_ds),
        AIGCImageDataset(test_samples, transform=eval_transform, hf_dataset=hf_ds),
    )


def dataset_summary(ds: AIGCImageDataset) -> str:
    n_real = sum(1 for s in ds.samples if s.label == 0)
    n_ai = sum(1 for s in ds.samples if s.label == 1)
    return f"n={len(ds)} real={n_real} aigc={n_ai}"


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "image_01": torch.stack([b["image_01"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "target": torch.tensor(
            [b.get("target", b["label"]) for b in batch], dtype=torch.float32
        ),
        "path": [b["path"] for b in batch],
        "generator": [b.get("generator", "") for b in batch],
    }
