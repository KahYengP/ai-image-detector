# Error analysis

Examples below are **pulled from the current test split**, not invented.
Clean test size: 16. False positives (real, pred>=0.5): 2. False negatives (AI, pred<0.5): 6.

## False positives (authentic images scored as AIGC)

- `C:\Users\Sharon\Desktop\ai-image-detector\data\raw\cifake\test\REAL\real_004.jpg`  pred=0.503  semantic=0.709  frequency=0.517  tier=low_confidence
  Semantic cues suggest AI generation; the frequency spectrum does not show a strong upsampling signature.
- `C:\Users\Sharon\Desktop\ai-image-detector\data\raw\cifake\test\REAL\real_002.jpg`  pred=0.501  semantic=0.716  frequency=0.516  tier=low_confidence
  Semantic cues suggest AI generation; the frequency spectrum does not show a strong upsampling signature.

## False negatives (AI images scored as real)

- `C:\Users\Sharon\Desktop\ai-image-detector\data\raw\cifake\test\FAKE\fake_003.jpg`  pred=0.495  semantic=0.708  frequency=0.517  tier=low_confidence
  Semantic cues suggest AI generation; the frequency spectrum does not show a strong upsampling signature.
- `C:\Users\Sharon\Desktop\ai-image-detector\data\raw\cifake\test\FAKE\fake_001.jpg`  pred=0.497  semantic=0.709  frequency=0.518  tier=low_confidence
  Semantic cues suggest AI generation; the frequency spectrum does not show a strong upsampling signature.
- `C:\Users\Sharon\Desktop\ai-image-detector\data\raw\cifake\test\FAKE\fake_006.jpg`  pred=0.498  semantic=0.699  frequency=0.517  tier=low_confidence
  Semantic cues suggest AI generation; the frequency spectrum does not show a strong upsampling signature.
- `C:\Users\Sharon\Desktop\ai-image-detector\data\raw\cifake\test\FAKE\fake_000.jpg`  pred=0.498  semantic=0.711  frequency=0.516  tier=low_confidence
  Semantic cues suggest AI generation; the frequency spectrum does not show a strong upsampling signature.

## Trade-offs

- The frequency branch is most useful when upsampling artifacts survive, but JPEG q30 and heavy blur flatten the spectrum and push more mass into `low_confidence`.
- The semantic branch is more stable under crop/jitter/resize, and can over-trigger on unusual real photographs that sit far from CLIP's pretraining distribution.
- The 0.4–0.6 band is an automated policy hook (downrank + secondary pass), not a claim that the fused score is calibrated as a probability.
