"""Visual and file-level cues that the fusion MLP never saw.

These do not replace the trained detector. They only *raise* P(AI) when a
known generator tell is present, so a photographic-looking AI image is less
likely to collapse into `likely_real`.

Cues:
  - C2PA / EXIF / XMP generator tags and Content Credentials
  - Visible corner watermarks (CLIP)
  - Hands / extra fingers, garbled text, mismatched accessories,
    background merge, over-smoothed skin (CLIP + a cheap skin-texture check)
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# Software / provenance strings that almost never appear on a camera original.
_GENERATOR_TOKENS = (
    "midjourney",
    "dall-e",
    "dall·e",
    "dalle",
    "stable diffusion",
    "stablediffusion",
    "stable-diffusion",
    "adobe firefly",
    "google imagen",
    "leonardo.ai",
    "leonardo ai",
    "novelai",
    "civitai",
    "automatic1111",
    "comfyui",
    "invokeai",
    "bing image",
    "chatgpt",
    "openai",
    "synthesia",
    "runwayml",
    "pika labs",
    "ideogram",
    "flux.1",
    "flux1",
    "dreamstudio",
    "sdxl",
)
_C2PA_TOKENS = (
    "c2pa",
    "contentcredentials",
    "content credentials",
    "claim generator",
    "stds.adobe.com/c2pa",
    "c2pa.org",
)

_AI_PROMPTS: list[tuple[str, str]] = [
    ("hands", "a picture of a hand with extra fingers or melted deformed fingers"),
    ("garbled_text", "an image with scrambled fake letters and unreadable background text"),
    ("accessories", "a face with mismatched earrings or mismatched glasses"),
    ("background", "a photo where background objects merge, bend, or make no sense"),
    ("plastic_skin", "a portrait with plastic doll skin, poreless and overly perfect lighting"),
    ("ai_render", "an AI generated synthetic image"),
    ("watermark", "a generated image with a small logo watermark in the corner"),
]
_REAL_PROMPTS = (
    "a real photograph taken with a camera",
    "a candid photo of a real object or person",
    "a natural photograph with real lighting and texture",
    "a real product photograph shot in a store or studio",
)

# How hard each CLIP cue can push the fused logit. Positive only.
_CUE_WEIGHTS = {
    "hands": 1.6,
    "garbled_text": 1.8,
    "accessories": 1.4,
    "background": 1.1,
    "plastic_skin": 1.3,
    "ai_render": 1.2,
    "watermark": 1.2,
    "skin_texture": 1.0,
    "provenance": 4.0,
}

_MIN_SIDE_FOR_CLIP = 64
# Per-cue probability after softmax over AI+real prompts; ordinary photos
# spread mass across the real anchors, so 0.18 is already a strong tell.
_CUE_FIRE = 0.18
_OVERALL_AI_FIRE = 0.55


def _xmp_from_bytes(data: bytes) -> str:
    text = data.decode("latin-1", errors="ignore")
    chunks: list[str] = []
    for pattern in (
        r"<\?xpacket[\s\S]{0,300000}\<?xpacket end",
        r"<x:xmpmeta[\s\S]{0,300000}</x:xmpmeta>",
        r"<rdf:RDF[\s\S]{0,300000}</rdf:RDF>",
    ):
        chunks.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return " ".join(chunks).lower()


def _pil_metadata_blob(image: Image.Image) -> str:
    parts: list[str] = []
    info = image.info or {}
    for key, value in info.items():
        parts.append(str(key))
        parts.append(str(value))
    try:
        exif = image.getexif()
        if exif:
            for tag, value in exif.items():
                parts.append(str(tag))
                parts.append(str(value))
    except Exception:
        pass
    return " ".join(parts).lower()


def provenance_cues(path: Optional[Path], image: Image.Image) -> dict[str, Any]:
    """True if C2PA / generator software tags are embedded in EXIF or XMP.

    We never scan compressed JPEG payload bytes: short tokens like 'c2pa'
    appear by chance in entropy-coded data and would false-trigger.
    """
    blob = _pil_metadata_blob(image)
    if path is not None:
        try:
            blob = blob + " " + _xmp_from_bytes(path.read_bytes()[:4_000_000])
        except OSError:
            pass
    hits = [tok for tok in _GENERATOR_TOKENS if tok in blob]
    c2pa = any(tok in blob for tok in _C2PA_TOKENS)
    return {
        "provenance_ai": bool(hits or c2pa),
        "c2pa": c2pa,
        "generator_tags": hits[:8],
    }


def skin_oversmooth_score(image: Image.Image) -> float:
    """High when large skin-colored regions have almost no pore/edge texture."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    if arr.shape[0] < 48 or arr.shape[1] < 48:
        return 0.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (np.abs(r - g) > 15)
    if float(mask.mean()) < 0.08 or float(mask.mean()) > 0.50:
        return 0.0
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(gray)
    energy = np.sqrt(gx * gx + gy * gy)
    skin_e = float(energy[mask].mean())
    # Camera skin typically sits well above ~6–8 on this scale; doll skin is flatter.
    return float(np.clip((7.5 - skin_e) / 7.5, 0.0, 1.0))


@lru_cache(maxsize=1)
def _clip_bundle(clip_name: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_name)
    processor = CLIPProcessor.from_pretrained(clip_name)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, processor


def clip_artifact_scores(
    image: Image.Image,
    clip_name: str,
    device: torch.device,
) -> dict[str, float]:
    """P(cue) for each visual tell vs a real-photograph anchor."""
    if min(image.size) < _MIN_SIDE_FOR_CLIP:
        return {**{name: 0.0 for name, _ in _AI_PROMPTS}, "overall_ai": 0.0}
    model, processor = _clip_bundle(clip_name, str(device))
    texts = [p for _, p in _AI_PROMPTS] + list(_REAL_PROMPTS)
    inputs = processor(text=texts, images=image.convert("RGB"), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0]
        probs = torch.softmax(logits, dim=0)
    out: dict[str, float] = {}
    n_ai = len(_AI_PROMPTS)
    for i, (name, _) in enumerate(_AI_PROMPTS):
        out[name] = float(probs[i].item())
    out["overall_ai"] = float(probs[:n_ai].sum().item())
    return out


def score_artifacts(
    image: Image.Image,
    path: Optional[Path] = None,
    clip_name: str = "openai/clip-vit-base-patch32",
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Return per-cue scores and a non-negative logit boost."""
    device = device or torch.device("cpu")
    prov = provenance_cues(path, image)
    skin = skin_oversmooth_score(image)
    visual = clip_artifact_scores(image, clip_name, device)
    visual["skin_texture"] = skin
    overall_ai = float(visual.pop("overall_ai", 0.0))

    boost = 0.0
    fired: list[str] = []
    if prov["provenance_ai"]:
        boost += _CUE_WEIGHTS["provenance"]
        fired.append("provenance")
    # Only credit specific visual tells when the image also looks more AI than real overall.
    allow_visual = overall_ai >= _OVERALL_AI_FIRE
    for name, prob in visual.items():
        if name == "skin_texture":
            weight = _CUE_WEIGHTS.get(name, 0.0)
            excess = max(0.0, float(prob) - 0.45)
            if excess > 0 and weight > 0:
                boost += weight * excess / 0.55
                fired.append(name)
            continue
        if not allow_visual:
            continue
        if name == "garbled_text" and (overall_ai < 0.72 or float(prob) < 0.30):
            continue
        weight = _CUE_WEIGHTS.get(name, 0.0)
        excess = max(0.0, float(prob) - _CUE_FIRE)
        if excess > 0 and weight > 0:
            boost += weight * min(1.0, excess / 0.25)
            fired.append(name)
    visual["overall_ai"] = round(overall_ai, 4)
    # Keep visual cues from dominating a clean camera photo.
    if not prov["provenance_ai"]:
        boost = min(boost, 2.6)

    return {
        "provenance_ai": prov["provenance_ai"],
        "c2pa": prov["c2pa"],
        "generator_tags": prov["generator_tags"],
        "visual": {k: round(float(v), 4) for k, v in visual.items()},
        "fired": fired,
        "logit_boost": round(float(boost), 4),
    }


def fuse_logit(raw_logit: float, bias: float, temperature: float, artifact_boost: float) -> float:
    temp = max(float(temperature), 1e-3)
    return (float(raw_logit) + float(bias)) / temp + max(0.0, float(artifact_boost))


def sigmoid(z: float) -> float:
    z = float(np.clip(z, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-z)))
