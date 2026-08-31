# Error analysis

Examples below are **pulled from the current test split**, not invented.
Clean test size: 42. False positives (real, pred>=0.5): 6. False negatives (AI, pred<0.5): 2.

## False positives (authentic images scored as AIGC)

- `train images\filtered_edited9.png`  pred=0.787  semantic=0.751  frequency=0.468  tier=likely_ai_generated
  The fused score indicates AI generation; driven mainly by semantic cues.
- `train images\real-blur7.jpeg`  pred=0.761  semantic=0.803  frequency=0.473  tier=likely_ai_generated
  The fused score indicates AI generation; driven mainly by semantic cues.
- `train images\filtered_edited10-.png`  pred=0.661  semantic=0.634  frequency=0.469  tier=filters_or_edited
  The fused score sits between a clean camera original and full AIGC; treat as a filtered or edited photograph (including partial AI edits).
- `train images\edited-filtered8.jpeg`  pred=0.605  semantic=0.676  frequency=0.476  tier=filters_or_edited
  The fused score sits between a clean camera original and full AIGC; treat as a filtered or edited photograph (including partial AI edits).

## False negatives (AI images scored as real)

- `train images\ai-generated63.jpeg`  pred=0.226  semantic=0.356  frequency=0.473  tier=likely_real
  The fused score is consistent with a real photograph; semantic and frequency branches do not show a strong AI signature.
- `train images\ai-generated2.jpeg`  pred=0.404  semantic=0.458  frequency=0.476  tier=filters_or_edited
  The fused score sits between a clean camera original and full AIGC; treat as a filtered or edited photograph (including partial AI edits).

## Trade-offs

- The frequency branch is most useful when upsampling artifacts survive, but JPEG q30 and heavy blur flatten the spectrum and push more mass into `filters_or_edited`.
- The semantic branch is more stable under crop/jitter/resize, and can over-trigger on unusual real photographs that sit far from CLIP's pretraining distribution.
- The middle band is an automated policy hook for filters / retouching / partial AI edits, not a claim that the fused score is a perfectly calibrated probability.
