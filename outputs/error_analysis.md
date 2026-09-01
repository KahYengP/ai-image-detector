# Error analysis

Examples below are **pulled from the current test split**, not invented.
Clean test size: 43. False positives (real, pred>=0.5): 5. False negatives (AI, pred<0.5): 2.

## False positives (authentic images scored as AIGC)

- `train images\filtered_edited7-.png`  pred=0.859  semantic=0.840  frequency=0.651  tier=likely_ai_generated
  The fused score indicates AI generation; driven mainly by semantic cues.
- `train images\real053.jpeg`  pred=0.781  semantic=0.785  frequency=0.763  tier=likely_ai_generated
  The fused score indicates AI generation; both branches agree.
- `train images\edited-filtered4.jpeg`  pred=0.615  semantic=0.722  frequency=0.582  tier=filters_or_edited
  The fused score sits between a clean camera original and full AIGC; treat as a filtered or edited photograph (including partial AI edits).
- `train images\real057.jpeg`  pred=0.593  semantic=0.703  frequency=0.742  tier=filters_or_edited
  The fused score sits between a clean camera original and full AIGC; treat as a filtered or edited photograph (including partial AI edits).

## False negatives (AI images scored as real)

- `train images\ai-generated76.png`  pred=0.365  semantic=0.456  frequency=0.618  tier=filters_or_edited
  The fused score sits between a clean camera original and full AIGC; treat as a filtered or edited photograph (including partial AI edits).
- `train images\ai-generated4.jpeg`  pred=0.473  semantic=0.558  frequency=0.493  tier=filters_or_edited
  The fused score sits between a clean camera original and full AIGC; treat as a filtered or edited photograph (including partial AI edits).

## Trade-offs

- The frequency branch is most useful when upsampling artifacts survive, but JPEG q30 and heavy blur flatten the spectrum and push more mass into `filters_or_edited`.
- The semantic branch is more stable under crop/jitter/resize, and can over-trigger on unusual real photographs that sit far from CLIP's pretraining distribution.
- The middle band is an automated policy hook for filters / retouching / partial AI edits, not a claim that the fused score is a perfectly calibrated probability.
