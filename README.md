# Robust Detection of AI-Generated Images Under Real-World Transformations

Prototype detector for **image-level AIGC vs authentic**, with robustness to the post-processing that actually happens in the wild (JPEG re-encode, blur, thumbnail resize, sensor noise, filter-app color jitter, profile-crop).

This is not a “max clean accuracy” project. The grading-style metric we optimize and report is:

```text
Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust
```

`AUC_robust` is the mean ROC AUC across the transformation × severity grid in `evaluate.py`.

**Hard constraints we follow**

- Open-source pretrained backbones only (CLIP ViT-B/32). Total parameters ≪ 2B.
- Public/licensed data only: CIFAKE, SID_Set, WildFake.
- **Never train** on the demonstration split: COCO val2017 (non-AIGC) + DALL·E Advanced (AIGC).
- Original two-branch fusion (CLIP semantic + FFT CNN). We do not clone a published detector as-is.

---

## What this repo delivers

| Deliverable | Where |
| --- | --- |
| Inference: image directory → JSON (`image_path`, `pred`, plus explainability fields) | [`predict.py`](predict.py) |
| Robustness table + four auto-generated charts | [`evaluate.py`](evaluate.py) → `outputs/robustness_table.md`, `outputs/charts/` |
| Error analysis from the actual test split | `outputs/error_analysis.md` (regenerated every eval) |
| JPEG class-balance check (run on the training dump) | [`data/balance_check.py`](data/balance_check.py) → `outputs/balance_check.json` |
| Toggleable training augmentations | [`config.yaml`](config.yaml) + [`data/augmentations.py`](data/augmentations.py) |

JSON record per image (required keys plus the explainability standouts):

```json
{
  "image_path": "img_001.jpg",
  "pred": 0.73,
  "semantic_score": 0.55,
  "frequency_score": 0.91,
  "explanation": "High-frequency anomaly detected (frequency branch strongly confident); semantic content appears ambiguous or real.",
  "tier": "likely_ai_generated",
  "suggested_policy": "Attach an AIGC provenance label and reduce organic distribution rank."
}
```

Confidence tiers are **automation hooks**, not a human-review queue:

| `pred` | `tier` | `suggested_policy` |
| --- | --- | --- |
| `< 0.4` | `likely_real` | Allow with standard distribution weight |
| `0.4–0.6` | `low_confidence` | Reduce distribution weight + secondary automated detector pass |
| `> 0.6` | `likely_ai_generated` | AIGC provenance label + downrank |

---

## Architecture

Two branches, concatenated, then a small MLP. No attention fusion — we wanted something that trains reliably under limited compute.

1. **Semantic branch (CLIP ViT-B/32).** Freeze everything except the last transformer block + post-LN. Linear projection 768 → 512. This branch is meant to survive crop, color jitter, and resize because it reasons about content.
2. **Frequency branch.** Grayscale 2D FFT (`torch.fft.fft2` + `fftshift` + log-magnitude) into a 4-layer CNN trained from scratch → 128-d embedding. This branch is meant to catch GAN/diffusion upsampling peaks that often survive blur/JPEG better than RGB textures.
3. **Fusion.** Concatenate 640-d → 2-layer MLP → one sigmoid `pred`. Auxiliary BCE on each branch so `semantic_score` / `frequency_score` are trained, not decorative.

Explanations are **rule-based** from those two scores (agree / frequency-driven / semantic-driven / both uncertain). No language model.

---

## Setup

Python 3.10+ recommended. A GPU is optional; CPU works for the smoke path.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

The first training run downloads `openai/clip-vit-base-patch32` from Hugging Face (~350MB).

### Data

Point `config.yaml` `paths.*` at local dumps, then set `dataset.name` to `cifake`, `sid_set`, `wildfake`, or `combined`.

| Dataset | Source | Notes |
| --- | --- | --- |
| **CIFAKE** (default) | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | Layout `train\|test / REAL\|FAKE`. **32×32** — good for a pipeline test, weak for FFT artifacts (we have to upsample to 224). |
| **SID_Set** | [Hugging Face](https://huggingface.co/datasets/saberzl/SID_Set) or the [Drive folder layout](https://github.com/hzlsaber/SIDA) | Labels 0 real / 1 full synthetic / 2 tampered. Tampered is **excluded** by default (`sid_set_include_tampered: false`). If no local folder exists, the loader streams the HF dataset. |
| **WildFake** | [ModelScope](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) — use the 中/En toggle | Walks a local dump and infers labels from folder names. Paths matching **COCO val2017** or **DALL·E Advanced** are skipped so they cannot leak into training. |

**No dataset yet?** Generate a tiny CIFAKE-layout JPEG dump (random pixels, *not* a real benchmark):

```bash
python scripts/make_smoke_dataset.py
```

### JPEG balance check

Real photos are often social-media JPEGs; many synthetic dumps are PNG or q≈100 JPEG. A detector can learn “JPEG = real” instead of AIGC signal. We estimate JPEG quality from the luminance quantization table and compare classes:

```bash
python -m data.balance_check
```

Interpretation is written to `outputs/balance_check.json` and printed. Even if the gap is small, we still train with JPEG augmentation so residual codec cues are a poor shortcut.

On the bundled smoke dump, REAL files are saved at q=70–85 and FAKE at q=92–100 **on purpose**, so the checker has a gap to report. Real CIFAKE/SID_Set/WildFake numbers will differ — re-run the checker on the dump you actually train on.

---

## Reproduce results

```bash
# 1. Balance check (methodology, not optional)
python -m data.balance_check

# 2. Train (fusion model, augs on). Drop --max-samples on a real dump.
python train.py --epochs 3

# CLIP-only baseline (no frequency branch), no augs:
python train.py --clip-only --no-aug --epochs 3

# 3. Required inference script
python predict.py --image-dir path/to/images --output outputs/predictions.json

# 4. Robustness table + charts + error analysis
python evaluate.py
```

Useful flags:

- `--max-samples N` — smoke / CPU subset
- `--clip-only` — semantic branch only
- `--no-aug` — disable the training-time robustness stack
- `--device cpu` / `--device cuda`
- `evaluate.py --skip-transforms` — clean AUC only (faster)

Every `evaluate.py` run regenerates:

1. `outputs/charts/robustness_bars.png` — AUC per condition (the required visual summary)
2. `outputs/charts/roc_overlay.png` — clean ROC vs pooled-transformed ROC
3. `outputs/charts/confidence_histogram.png` — `pred` histogram with 0.4 / 0.6 tier lines
4. `outputs/charts/branch_scatter.png` — semantic vs frequency, colored by true label
5. `outputs/robustness_table.md`, `outputs/eval_summary.json`, `outputs/error_analysis.md`

Ablations: set any family under `augmentations:` in `config.yaml` to `enabled: false` (e.g. turn off blur) and retrain. When we have to shrink a large photo to 224, we **crop** rather than downsample so high-frequency generator traces are not destroyed (`dataset.prefer_crop: true`).

---

## Training-time augmentations

Each family is independently toggleable. JPEG is applied last to mimic platform re-encode after a user already edited the file.

| Transform | Parameters | Real-world analog |
| --- | --- | --- |
| JPEG compression | quality 90 / 70 / 50 / 30 | Social re-encode, messaging |
| Gaussian blur | σ = 0.5 / 1.0 / 2.0 | Out-of-focus |
| Resize | 0.5× / 0.25× then upscale | Thumbnails |
| Gaussian noise | σ = 0.02 / 0.05 / 0.10 | Low-light sensor noise |
| Color jitter | brightness/contrast/sat ±20% | Filter apps, auto-enhance |
| Center crop | 80% then restore size | Profile-picture framing |

---

## Limitations and what we would improve with more time

- **CIFAKE is 32×32.** Upsampling to 224 invents frequencies; the FFT branch cannot show its real value there. Prefer SID_Set / WildFake (native higher resolution) for any number you would quote to judges.
- **Generator coverage.** A model trained on one synthesizer (e.g. CIFAKE’s SD-1.4 CIFAR clones) will not automatically transfer to DALL·E / Midjourney / in-the-wild LoRAs. `evaluate.py` reports an “unseen generator” AUC when test generators are disjoint from train.
- **Calibration.** `pred` is a trained sigmoid, not a guaranteed probability. That is why we expose a `low_confidence` band instead of forcing a binary call.
- **Compute.** We fine-tune one CLIP block, not the full ViT, and keep fusion as a 2-layer MLP. With more GPU time we would unfreeze more CLIP layers, add Mixup/CutMix carefully (they can destroy spectral cues), and train longer on WildFake.
- **Tampering vs fully synthetic.** SID_Set label 2 is excluded by default; localization is out of scope for this image-level prototype.
- **Shortcut risk.** Always re-run the JPEG balance check on any new dump. If one class is systematically cleaner JPEG, treat high clean AUC with suspicion until JPEG-augmented AUC holds up.

---

## Error analysis

Do not quote invented failure modes. After training, run:

```bash
python evaluate.py
```

Then read [`outputs/error_analysis.md`](outputs/error_analysis.md). That file lists **actual** false positives and false negatives from the current test split (paths, scores, rule-based explanations) and copies example files into `outputs/error_examples/`.

### Smoke-pipeline check (not a benchmark)

The numbers below come from one CPU epoch on `scripts/make_smoke_dataset.py` (random 128x128 JPEGs). They only prove the stack runs. **Do not put them on Devpost as model quality.**

JPEG balance check on that dump (intentional quality gap so the checker has something to report):

| Class | n | JPEG fraction | Mean estimated quality |
| --- | --- | --- | --- |
| real | 22 | 100% | 77.6 |
| AI | 22 | 100% | 96.4 |

Warning raised: *mean JPEG quality differs by 18.8 points — the model could use compression artifacts as a class cue.* Re-run `python -m data.balance_check` on real CIFAKE/SID_Set/WildFake.

Smoke `evaluate.py` grid (n=16, 1 epoch — near chance, as expected on noise):

| Condition | AUC |
| --- | --- |
| Clean | 0.547 |
| JPEG q90 / q70 / q50 / q30 | 0.344 / 0.203 / 0.516 / 0.672 |
| Blur sigma 0.5 / 1.0 / 2.0 | 0.547 / 0.562 / 0.453 |
| Resize 0.5x / 0.25x | 0.547 / 0.562 |
| Noise sigma 0.02 / 0.05 / 0.10 | 0.453 / 0.422 / 0.391 |
| Color jitter +/-20% | 0.547 |
| Crop 80% | 0.719 |

`AUC_clean = 0.547`, `AUC_robust = 0.496`, **Final Score = 0.521**. Charts: `outputs/charts/robustness_bars.png`, `roc_overlay.png`, `confidence_histogram.png`, `branch_scatter.png`.

After 1 epoch, fused scores sat in the `low_confidence` band (~0.50). Clean test (16 images): **2 false positives**, **6 false negatives** at threshold 0.5, all with `pred` ≈ 0.50:

| Type | Example | pred | semantic | frequency | tier |
| --- | --- | --- | --- | --- | --- |
| FP | `test/REAL/real_004.jpg` | 0.503 | 0.709 | 0.517 | low_confidence |
| FP | `test/REAL/real_002.jpg` | 0.501 | 0.716 | 0.516 | low_confidence |
| FN | `test/FAKE/fake_003.jpg` | 0.495 | 0.708 | 0.517 | low_confidence |
| FN | `test/FAKE/fake_001.jpg` | 0.497 | 0.709 | 0.518 | low_confidence |

Explanations were frequency-vs-semantic disagreement (“semantic cues suggest AI; frequency spectrum does not show a strong upsampling signature”), which is what the rule-based text is for. On real photos we expect FPs on unusual authentic images far from CLIP’s prior, and FNs on heavily JPEG’d/blurred AI images where the frequency branch goes quiet.

Typical trade-off we will look for on real data (confirm against the generated file, do not assume):

- **False positives:** unusual real photos; heavy denoise/sharpen that injects synthetic-looking spectra.
- **False negatives:** strongly JPEG’d or blurred AI images.
- **Policy:** the 0.4–0.6 band **downranks and re-checks automatically**; it is not a claim that `pred` is a calibrated probability.

---

## Project layout

```text
config.yaml              # paths, aug toggles, thresholds
train.py
evaluate.py
predict.py
data/
  datasets.py            # CIFAKE / SID_Set / WildFake loaders
  balance_check.py       # JPEG quality vs class
  augmentations.py       # independently toggleable transforms
models/
  semantic_branch.py     # CLIP ViT-B/32
  frequency_branch.py    # FFT + CNN
  fusion.py              # 640-d MLP + AIGCDetector
  explain.py             # tiers + rule-based text
scripts/make_smoke_dataset.py
```

---

## Team contributions

Update this table before the Devpost submission.

| Member | Contributions |
| --- | --- |
| TBD | Data pipeline, JPEG balance check |
| TBD | CLIP semantic branch + training loop |
| TBD | Frequency branch, fusion, explainability JSON |
| TBD | Robustness eval, charts, README / pitch |

---

## Tools, models, libraries, datasets (Devpost checklist)

- **Problem:** robust image-level AIGC detection under redistribution transforms, not lab-only accuracy.
- **Tools:** VS Code / Cursor, Python, optional CUDA.
- **Models / APIs:** Hugging Face `openai/clip-vit-base-patch32` (vision encoder only). No proprietary APIs.
- **Libraries:** PyTorch, torchvision, Hugging Face Transformers, scikit-learn, Pillow, matplotlib/seaborn, pandas not required.
- **Datasets:** CIFAKE, SID_Set, WildFake (public). Demonstration split (COCO val2017 + DALL·E Advanced) is evaluation-only and is filtered out of the WildFake loader.
