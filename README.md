# UM AI Image Detector

Local web app and CLI that score stills as **real**, **filtered / edited**, or **AI-generated**. The model is a two-branch detector (CLIP ViT-B/32 semantic + FFT CNN) fused by a small MLP.

This is **not a hosted website**. After you clone the repo, run the server on your own computer and download the trained weights (GitHub cannot store the 338 MB checkpoint).

---

## Setup (clone → run)

Need **Python 3.10+**. A GPU is optional; CPU works.

### 1. Clone and create a virtual environment

**Windows (PowerShell or Command Prompt)**

```bat
git clone https://github.com/KahYengP/ai-image-detector.git
cd ai-image-detector
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
git clone https://github.com/KahYengP/ai-image-detector.git
cd ai-image-detector
python -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of the terminal line after activate.

### 2. Install Python packages

With `.venv` still active:

```bat
pip install -r requirements.txt
```

The first Analyze also downloads `openai/clip-vit-base-patch32` from Hugging Face (~350 MB) if it is not already cached.

### 3. Download the trained model (`outputs/best.pt`)

The checkpoint is **not** in git. Download it from Google Drive (anyone with the link can view/download):

**[Download best.pt](https://drive.google.com/file/d/1oa6eWCfhXICZx3Kkt0JYn-JHABARJgFD/view?usp=sharing)**

1. Open the link and click **Download**.
2. Confirm the file is about **338 MB** (345,520 KB). If it is 0 KB or a few bytes, the download failed — try again from the Drive page, not a blank file you created yourself.
3. Rename it to `best.pt` if the browser added extra text.
4. Put it here (create the `outputs` folder if needed):

```text
ai-image-detector/outputs/best.pt
```

Do **not** create an empty `best.pt`. PyTorch will fail with `Ran out of input`.

In `config.yaml`, `paths.checkpoint` must stay:

```yaml
checkpoint: outputs/best.pt
```

### 4. Start the website

Keep `.venv` active. Leave this terminal open while you use the app.

**Windows**

```bat
.venv\Scripts\activate
python web\server.py
```

**macOS / Linux**

```bash
source .venv/bin/activate
python web/server.py
```

In a browser open **http://127.0.0.1:8765**

- Drag in one or more JPG / PNG / WEBP / HEIC files, or click **Load Sample Test Batch**
- Click **Analyze Images**. Scores are saved to **`outputs/predictions.json`** (`image_path` and `pred` per image).
- Use **Original / Heatmap Scan / Noise Pattern / Split View** on the results page. Heatmap and noise are CSS inspection filters (`contrast` / `hue-rotate` / inverted grayscale), not Grad-CAM. Orange **!** markers are high local-contrast patches; hover them for a short explanation.

Stop the server with `Ctrl+C`.

If Windows says port 8765 is already in use, close the other server window, or in PowerShell run:

```bat
netstat -ano | findstr :8765
taskkill /PID <pid> /F
```

Then start `python web\server.py` again.

---

## Checkpoint (`outputs/best.pt`)

`*.pt` files are **gitignored** (GitHub file-size limit). Every clone must download the same weights.

| What | Detail |
| --- | --- |
| Path | `outputs/best.pt` |
| Size | about **338 MB** |
| Download | [Google Drive — best.pt](https://drive.google.com/file/d/1oa6eWCfhXICZx3Kkt0JYn-JHABARJgFD/view?usp=sharing) |

If Analyze fails with a path like `/Users/.../outputs/best.pt`, that machine still has an old absolute path in `config.yaml`. Set `checkpoint: outputs/best.pt` and keep the real file in this project’s `outputs/` folder.

---

## What the score means

The fused output `pred` is **P(AI)** in `[0, 1]`. Three display classes (also `result` in JSON):

| `pred` | `tier` | `result` |
| --- | --- | --- |
| `< 0.28` | `likely_real` | `real` |
| `0.28 – 0.74` | `filters_or_edited` | `filtered_or_edited` |
| `> 0.74` | `likely_ai_generated` | `AI` |

The website maps those scores into a 3-way probability breakdown, a confidence gauge, visual-signal bars, and rule-based explanations (semantic vs frequency plus CLIP / C2PA artifact cues). `pred` is a trained sigmoid, not a calibrated probability.

Analyze in the website also writes the same JSON to **`outputs/predictions.json`**.

---

## CLI inference (required JSON output)

The core scoring script is `predict.py`. It takes an **image directory** and writes a **JSON file** with one object per image. Each object includes at least:

- `image_path` — filename / relative path of the image
- `pred` — confidence that the image is AIGC-generated, in `[0, 1]` (higher = more likely AI)

Default output path:

**`outputs/predictions.json`**

**Windows**

```bat
.venv\Scripts\activate
python predict.py --image-dir "train images" --output outputs\predictions.json
```

**macOS / Linux**

```bash
source .venv/bin/activate
python predict.py --image-dir path/to/images --output outputs/predictions.json
```

The full record also has `result`, `semantic_score`, `frequency_score`, `explanation`, `tier`, and `artifact_cues`. Open the file at:

```text
ai-image-detector/outputs/predictions.json
```

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
  best.pt                # trained weights — download from Google Drive (not in git)
  predictions.json       # JSON list of {image_path, pred, ...} from CLI / web Analyze
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
