# UM AI Image Detector

Local web app and CLI that score stills as **real**, **filtered / edited**, or **AI-generated**. The model is a two-branch detector (CLIP ViT-B/32 semantic + FFT CNN) fused by a small MLP.

This is **not a hosted website**. After a GitHub clone, each computer must run the server locally and have a copy of the trained weights.

---

## Run the website

Python 3.10+ (GPU optional; CPU works).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
python web/server.py
```

Open **http://127.0.0.1:8765**

- Drag in one or more JPG / PNG / WEBP / HEIC files, or click **Load Sample Test Batch**
- Click **Analyze Images**
- Use **Original / Heatmap Scan / Noise Pattern / Split View** on the results page. Heatmap and noise are CSS inspection filters (`contrast` / `hue-rotate` / inverted grayscale), not Grad-CAM. Orange **!** markers are high local-contrast patches; hover them for a short explanation.

Leave the terminal open while you use the app. Stop with `Ctrl+C`.

First load downloads `openai/clip-vit-base-patch32` from Hugging Face (~350 MB) if it is not cached.

---

## Checkpoint (`outputs/best.pt`)

`*.pt` files are **gitignored**. A clone does **not** include the model.

| What | Detail |
| --- | --- |
| Path | `outputs/best.pt` (see `config.yaml` `paths.checkpoint`) |
| Size | about **338 MB** |
| How to share | copy the file via Drive, OneDrive, or USB — not GitHub (over the 100 MB limit) |

Do **not** create an empty `best.pt`. PyTorch will fail with `Ran out of input`.

If Analyze fails with a path like `/Users/.../outputs/best.pt`, that machine still has an old absolute path in `config.yaml`. Set:

```yaml
checkpoint: outputs/best.pt
```

then put the real 338 MB file in this project’s `outputs/` folder.

---

## What the score means

The fused output `pred` is **P(AI)** in `[0, 1]`. Three display classes (also `result` in JSON):

| `pred` | `tier` | `result` |
| --- | --- | --- |
| `< 0.28` | `likely_real` | `real` |
| `0.28 – 0.74` | `filters_or_edited` | `filtered_or_edited` |
| `> 0.74` | `likely_ai_generated` | `AI` |

The website maps those scores into a 3-way probability breakdown, a confidence gauge, visual-signal bars, and rule-based explanations (semantic vs frequency plus CLIP / C2PA artifact cues). `pred` is a trained sigmoid, not a calibrated probability.

---

## CLI inference

```bash
python predict.py --image-dir path/to/images --output outputs/predictions.json
```

Defaults to `test-images` if present, else `train images`. Each JSON record includes `image_path`, `result`, `pred`, `semantic_score`, `frequency_score`, `explanation`, `tier`, and `artifact_cues`.

---

## Training

Current default data is **human-labeled stills** (`dataset.name: human` in `config.yaml`):

- `paths.human` → `train images`
- `paths.human_extra` → `test-images`

Labels come from **filename prefixes** (more specific first): `ai-generated_live`, `ai-generated_product`, `real_live`, `ai-generated`, `filtered_edited`, `edited`, `real_blur`, `real`. Edited / filtered names use a soft target near 0.5 so they land in the middle band.

```bash
# Fine-tune the existing checkpoint
python train.py --resume --reset-best --epochs 12 --lr 5e-5

# From scratch (uses paths.checkpoint as the save path)
python train.py
```

Optional public dumps (CIFAKE, SID_Set, WildFake) can still be wired through `config.yaml` `paths.*` and `dataset.name`. Do not train on the demonstration split (COCO val2017 + DALL·E Advanced); the WildFake loader skips those folders.

Useful flags: `--max-samples N`, `--clip-only`, `--no-aug`, `--device cpu` / `--device cuda`, `--resume`.

JPEG class-balance check (codec shortcut risk):

```bash
python -m data.balance_check
```

Robustness table and charts (after a checkpoint exists):

```bash
python evaluate.py
```

Writes `outputs/robustness_table.md`, `outputs/error_analysis.md`, and `outputs/charts/`.

---

## Architecture

1. **Semantic branch** — CLIP ViT-B/32; last transformer block + post-LN trainable; 768 → 512.
2. **Frequency branch** — grayscale 2D FFT (log-magnitude) into a 4-layer CNN → 128-d.
3. **Fusion** — concatenate 640-d → 2-layer MLP → sigmoid `pred`. Auxiliary BCE on each branch.

Training-time augmentations (JPEG last) mimic redistribution: JPEG, blur, thumbnail resize, sensor noise, color jitter, center crop. Toggle families under `augmentations:` in `config.yaml`.

---

## Project layout

```text
config.yaml              # paths, thresholds, aug toggles
train.py
predict.py
evaluate.py
utils.py
web/
  server.py              # http://127.0.0.1:8765
  index.html
  styles.css
  app.js
  uploads/               # analyzed uploads (gitignored except .gitkeep)
train images/            # labeled stills (filename prefixes)
test-images/             # optional extra labeled stills
outputs/
  best.pt                # trained weights — copy this; not in git
  predictions.json       # last CLI / web run
data/                    # loaders + augmentations
models/                  # CLIP, FFT CNN, fusion, explanations
scripts/                 # eval helpers, smoke dataset, downloads
```

---

## Limitations

- The website only works on the machine running `python web/server.py` (`127.0.0.1`).
- Inspection views (heatmap / noise / split) are visual filters, not model heatmaps. Key-area dots mark high luminance variance, not Grad-CAM.
- A model trained on one generator or live-shopping look will not automatically transfer to every in-the-wild LoRA or app filter.
- Prefer native-resolution photos over 32×32 CIFAKE if you quote frequency-branch numbers.
- Always re-run the JPEG balance check on a new dump; a codec gap can inflate clean AUC.
