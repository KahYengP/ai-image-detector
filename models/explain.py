"""Rule-based explainability and automated confidence-tiering.

Tiers are policy hooks for an automated pipeline (soft labels, distribution
downranking, a second automated pass) — not a human-review queue.
"""

from __future__ import annotations

from typing import Optional

HIGH = 0.6
LOW = 0.4


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
        "low_confidence",
        "Flag for reduced distribution weight and queue a secondary automated detector pass.",
    )


def make_explanation(
    pred: float,
    semantic_score: Optional[float],
    frequency_score: Optional[float],
) -> str:
    """Short natural-language rationale from the two branch scores. No LLM involved."""
    if semantic_score is None:
        semantic_score = pred
    if frequency_score is None:
        band = "AI-generated" if pred > HIGH else ("real" if pred < LOW else "uncertain")
        return (
            f"Semantic branch only (frequency branch disabled). "
            f"Content cues point to {band} (score={semantic_score:.2f})."
        )

    sem_high, sem_low = semantic_score > HIGH, semantic_score < LOW
    freq_high, freq_low = frequency_score > HIGH, frequency_score < LOW

    if freq_high and sem_high:
        return (
            "Both branches agree: semantic content and frequency artifacts "
            "both indicate AI generation."
        )
    if freq_low and sem_low:
        return (
            "Both branches agree: semantic content and the frequency spectrum "
            "are consistent with a real photograph."
        )
    if freq_high and not sem_high:
        return (
            "High-frequency anomaly detected (frequency branch strongly confident); "
            "semantic content appears ambiguous or real."
        )
    if sem_high and not freq_high:
        return (
            "Semantic cues suggest AI generation; the frequency spectrum does not "
            "show a strong upsampling signature."
        )
    if freq_low and not sem_low:
        return (
            "Frequency spectrum looks photographic; semantic branch is less sure. "
            "Treat as borderline."
        )
    if sem_low and not freq_low:
        return (
            "Semantic content looks photographic; frequency branch is less sure. "
            "Treat as borderline."
        )
    return (
        "Both branches are uncertain; the fused score sits in the low-confidence band."
    )


def explain_record(
    pred: float,
    semantic_score: Optional[float],
    frequency_score: Optional[float],
    likely_real_max: float = LOW,
    likely_ai_min: float = HIGH,
) -> dict:
    tier, policy = assign_tier(pred, likely_real_max, likely_ai_min)
    return {
        "tier": tier,
        "suggested_policy": policy,
        "explanation": make_explanation(pred, semantic_score, frequency_score),
    }
