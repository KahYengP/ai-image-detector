"""Rule-based explainability and automated confidence-tiering.

Tiers are policy hooks for an automated pipeline (soft labels, distribution
downranking, a second automated detector pass) — not a human-review queue.

The middle band is `filters_or_edited`: filters, retouching, or partial AI
edits. Those cases are subjective as full AIGC, so they are not called
`likely_ai_generated` and are not called a clean camera original.
"""

from __future__ import annotations

from typing import Any, Optional

HIGH = 0.70
LOW = 0.28

_CUE_LABELS = {
    "hands": "hand/finger errors (extra, floating, or melted digits)",
    "garbled_text": "scrambled or unreadable text",
    "accessories": "mismatched earrings or glasses",
    "background": "background objects that merge or bend unnaturally",
    "plastic_skin": "over-smoothed, poreless skin and overly perfect lighting",
    "skin_texture": "over-smoothed skin texture",
    "ai_render": "CLIP cues of a synthetic render",
    "watermark": "a generator-style watermark or corner logo",
    "provenance": "embedded generator / C2PA Content Credentials metadata",
}


def assign_tier(pred: float, likely_real_max: float = LOW, likely_ai_min: float = HIGH) -> tuple[str, str]:
    """Map a fused score to (tier, suggested_policy)."""
    if pred < likely_real_max:
        return (
            "likely_real",
            "Allow with standard distribution weight.",
        )
    if pred > likely_ai_min:
        return (
            "likely_ai_generated",
            "Attach an AIGC provenance label and reduce organic distribution rank.",
        )
    return (
        "filters_or_edited",
        "Treat as a filtered or edited photograph (including partial AI edits). "
        "Do not treat as a clean camera original, and do not attach a full AIGC label.",
    )


def make_explanation(
    pred: float,
    semantic_score: Optional[float],
    frequency_score: Optional[float],
    likely_real_max: float = LOW,
    likely_ai_min: float = HIGH,
    fired: Optional[list[str]] = None,
) -> str:
    """Short natural-language rationale from branch scores plus artifact cues."""
    fired = fired or []
    cue_bits = [_CUE_LABELS[name] for name in fired if name in _CUE_LABELS]
    cue_clause = (" Visual/metadata cues: " + "; ".join(cue_bits) + ".") if cue_bits else ""

    if semantic_score is None:
        semantic_score = pred
    if frequency_score is None:
        if pred > likely_ai_min:
            band = "AI-generated"
        elif pred < likely_real_max:
            band = "real"
        else:
            band = "filtered or edited"
        return (
            f"Semantic branch only (frequency branch disabled). "
            f"Content cues point to {band} (score={semantic_score:.2f})."
            + cue_clause
        )

    sem_high, sem_low = semantic_score > likely_ai_min, semantic_score < likely_real_max
    freq_high, freq_low = frequency_score > likely_ai_min, frequency_score < likely_real_max

    if pred < likely_real_max:
        base = "The fused score is consistent with a real photograph"
        if semantic_score is not None and frequency_score is not None:
            if semantic_score > likely_ai_min:
                base += ", though the semantic branch is more suspicious"
            elif frequency_score > likely_ai_min:
                base += ", though the frequency branch is more suspicious"
            else:
                base += "; semantic and frequency branches do not show a strong AI signature"
        base += "."
    elif pred > likely_ai_min:
        base = "The fused score indicates AI generation"
        if semantic_score is not None and frequency_score is not None:
            if freq_high and sem_high:
                base += "; both branches agree"
            elif sem_high:
                base += "; driven mainly by semantic cues"
            elif freq_high:
                base += "; driven mainly by frequency artifacts"
        base += "."
    else:
        base = (
            "The fused score sits between a clean camera original and full AIGC; "
            "treat as a filtered or edited photograph (including partial AI edits)."
        )
    return base + cue_clause


def explain_record(
    pred: float,
    semantic_score: Optional[float],
    frequency_score: Optional[float],
    likely_real_max: float = LOW,
    likely_ai_min: float = HIGH,
    fired: Optional[list[str]] = None,
) -> dict[str, Any]:
    tier, policy = assign_tier(pred, likely_real_max, likely_ai_min)
    return {
        "tier": tier,
        "suggested_policy": policy,
        "explanation": make_explanation(
            pred,
            semantic_score,
            frequency_score,
            likely_real_max,
            likely_ai_min,
            fired=fired,
        ),
    }
