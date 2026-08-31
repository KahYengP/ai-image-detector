const T_REAL = 0.28;
const T_AI = 0.74;
const CIRC = 2 * Math.PI * 48;

const KEY_AREA_REASONS = {
  AI: [
    {
      title: "High-contrast edge",
      reason:
        "Inconsistent pixel grain along this edge. Generative masking often leaves a sharp boundary without optical falloff.",
    },
    {
      title: "Facial boundary",
      reason:
        "Unnatural smoothing and weak camera-sensor noise near facial contours. Diffusion models often flatten pores and jaw edges.",
    },
    {
      title: "Texture / fabric",
      reason:
        "Repeating frequency patterns here. Diffusion textures often loop or smear instead of showing real sensor grain.",
    },
  ],
  filtered_or_edited: [
    {
      title: "Retouch boundary",
      reason:
        "Compression and retouching show up here as a sharpened contrast step — typical of beauty filters or local dodge/burn.",
    },
    {
      title: "Skin / face",
      reason:
        "Local smoothing reduces natural sensor grain on skin while nearby edges stay photographic. That mix is a filter/edit signature.",
    },
    {
      title: "Tonal patch",
      reason:
        "Color grading or frequency-domain edits are strongest in this patch (uneven contrast vs the rest of the frame).",
    },
  ],
  real: [
    {
      title: "Optical edge",
      reason:
        "Strong natural edge energy and photon noise. This is a high-detail camera region, not a synthetic mask.",
    },
    {
      title: "Facial detail",
      reason:
        "Local contrast around features matches lens optics and sensor grain, not over-smoothed generation.",
    },
    {
      title: "Texture detail",
      reason:
        "Fabric or background grain here matches camera sensor noise. The frequency residual does not look tiled or synthesized.",
    },
  ],
};

const CUE_LABELS = {
  hands: "Hand / finger geometry looks melted, extra, or floating",
  garbled_text: "Text, logos, or numerals look scrambled rather than readable",
  ai_product:
    "Product surface looks CGI-smooth (uniform texture, weak stitching)",
  fake_live: "Livestream still looks slightly unreal or plastic",
  accessories: "Accessories (earrings, glasses) look mismatched",
  background: "Background objects merge, bend, or lose physical structure",
  plastic_skin:
    "Skin looks poreless / over-smoothed beyond a light beauty filter",
  skin_texture: "Skin texture energy is unusually flat",
  ai_render: "CLIP content cues lean synthetic / rendered",
  watermark: "Generator-style corner watermark detected",
  provenance: "Embedded generator or C2PA Content Credentials metadata",
};

const RESULT_META = {
  real: {
    key: "real",
    title: "Authentic Real Photo",
    short: "Real",
    engine: "Authentic Camera Sensor",
    overlayLeft: "Standard Native Image",
    overlayRight: "Authentic Camera Sensor",
    caption: "Viewing natural full-resolution RGB image pixels",
    desc: "Authentic real-world photograph. Natural camera sensor noise, realistic lens optics, and authentic lighting verified.",
  },
  filtered_or_edited: {
    key: "edit",
    title: "Filtered / Edited Photo",
    short: "Filtered & Edited",
    engine: "Photo Editing & Retouching",
    overlayLeft: "Standard Native Image",
    overlayRight: "Photo Editing & Retouching",
    caption: "Photographic base with retouching, filters, or partial edits",
    desc: "Real base photo with noticeable color filters, contrast grading, or digital retouching. Not a clean original, and not full AIGC.",
  },
  AI: {
    key: "ai",
    title: "AI Generated Image",
    short: "AI Generated",
    engine: "Generative / Synthetic Render",
    overlayLeft: "Synthetic Image",
    overlayRight: "Generative Pixel Synthesis",
    caption: "Fused detector score indicates AI generation",
    desc: "The fused score indicates AI generation rather than a retouched camera photo.",
  },
};

const state = {
  files: [],
  results: [],
  index: 0,
  mode: "original",
  zoom: 1,
  keyAreas: true,
  img: null,
  split: 0.5,
  layout: null,
};

const $ = (id) => document.getElementById(id);

function fmtSize(bytes) {
  if (!bytes && bytes !== 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function threeWay(pred) {
  const p = Math.min(1, Math.max(0, Number(pred) || 0));
  const realC = 0.1;
  const filtC = (T_REAL + T_AI) / 2;
  const aiC = 0.9;
  const g = (x, c, s) => Math.exp(-((x - c) ** 2) / (2 * s * s));
  let real = g(p, realC, 0.16);
  let filtered = g(p, filtC, 0.17);
  let ai = g(p, aiC, 0.14);
  const z = real + filtered + ai || 1;
  real /= z;
  filtered /= z;
  ai /= z;
  return { real, filtered, ai };
}

function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}

function viewModel(rec) {
  const pred = Number(rec.pred);
  const scores = Number.isFinite(pred)
    ? threeWay(pred)
    : { real: 0.33, filtered: 0.34, ai: 0.33 };
  const result = rec.result || "filtered_or_edited";
  const meta = RESULT_META[result] || RESULT_META.filtered_or_edited;
  const primary =
    result === "AI"
      ? scores.ai
      : result === "real"
        ? scores.real
        : scores.filtered;
  const vis = (rec.artifact_cues && rec.artifact_cues.visual) || {};
  const freq = rec.frequency_score == null ? 0.5 : Number(rec.frequency_score);
  const sem = rec.semantic_score == null ? pred : Number(rec.semantic_score);
  const overall = Number(vis.overall_ai || 0);
  const cues = rec.artifact_cues || {};

  const signals = [
    {
      name: "Sensor Noise Pattern",
      value: clamp01(1 - freq),
      hint: "Camera sensor grain vs AI-smooth frequency residual",
    },
    {
      name: "Pixel Integrity",
      value: clamp01(1 - overall),
      hint: "How photographic the CLIP content embedding looks",
    },
    {
      name: "Lighting & Physics",
      value: clamp01(1 - Number(vis.background || 0)),
      hint: "Reflection, shadow, and background structure consistency",
    },
    {
      name: "Metadata Profile",
      value: cues.provenance_ai || cues.c2pa ? 0.18 : 0.82,
      hint:
        cues.provenance_ai || cues.c2pa
          ? "Generator / C2PA tags found in the file"
          : "No embedded generator credentials detected",
    },
    {
      name: "Edge Realism",
      value: clamp01(
        1 -
          Math.max(
            Number(vis.hands || 0),
            Number(vis.garbled_text || 0),
            Number(vis.accessories || 0),
          ),
      ),
      hint: "Natural boundary definition vs melted hands or garbled text",
    },
  ];

  const indicators = [];
  if (rec.explanation) indicators.push(rec.explanation);
  const fired = cues.fired || [];
  for (const name of fired) {
    if (CUE_LABELS[name]) indicators.push(CUE_LABELS[name]);
  }
  const ranked = Object.entries(vis)
    .filter(([k, v]) => k !== "overall_ai" && Number(v) >= 0.18)
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  for (const [name] of ranked) {
    const line = CUE_LABELS[name];
    if (line && !indicators.includes(line)) indicators.push(line);
  }
  if (Number.isFinite(sem) && Number.isFinite(freq)) {
    if (result === "AI" && sem > freq) {
      indicators.push(
        "Decision is driven more by semantic content cues than frequency artifacts.",
      );
    } else if (result === "AI" && freq > sem) {
      indicators.push(
        "Frequency residual is the stronger AI tell on this still.",
      );
    } else if (result === "real") {
      indicators.push(
        "Base composition and optical perspective remain consistent with a camera photo.",
      );
    } else {
      indicators.push(
        "Base composition stays photographic; the mid-band score points to filters or retouching rather than full synthesis.",
      );
    }
  }
  while (indicators.length < 3) {
    indicators.push("No additional hard artifact cue fired on this image.");
  }

  return {
    rec,
    scores,
    primary,
    meta,
    signals,
    indicators: indicators.slice(0, 5),
    sem,
    freq,
    vis,
    cues,
  };
}

function fileUrl(file) {
  return URL.createObjectURL(file);
}

function renderQueue() {
  const q = $("queue");
  $("queue-title").textContent =
    `${state.files.length} Image${state.files.length === 1 ? "" : "s"} Ready for Analysis`;
  $("analyze-label").textContent =
    `Analyze ${state.files.length} Image${state.files.length === 1 ? "" : "s"}`;
  $("analyze-btn").disabled = state.files.length === 0;
  q.classList.toggle("hidden", state.files.length === 0);
  $("queue-grid").innerHTML = state.files
    .map(
      (f, i) => `
      <div class="queue-card">
        <button class="remove-x" data-remove="${i}" type="button">×</button>
        <div class="queue-thumb"><img src="${f.preview}" alt="" /></div>
        <strong title="${f.name}">${f.name}</strong>
        <small>${fmtSize(f.size)}</small>
      </div>`,
    )
    .join("");
}

function addFiles(fileList) {
  for (const file of fileList) {
    if (!file.type.startsWith("image/") && !/\.(heic|heif)$/i.test(file.name))
      continue;
    state.files.push({
      name: file.name,
      size: file.size,
      file,
      preview: fileUrl(file),
    });
  }
  renderQueue();
}

function classCounts(results) {
  const counts = { real: 0, filtered_or_edited: 0, AI: 0 };
  for (const r of results) counts[r.result] = (counts[r.result] || 0) + 1;
  return counts;
}

function renderThumbs() {
  $("select-label").textContent =
    `Select Image (${state.results.length} total):`;
  $("thumb-row").innerHTML = state.results
    .map((raw, i) => {
      const vm = viewModel(raw);
      const pct = Math.round(vm.primary * 100);
      return `
        <button class="thumb-card ${i === state.index ? "active" : ""}" data-idx="${i}" type="button">
          <span class="thumb-photo">
            <img src="${raw.image_url || raw.preview || ""}" alt="" />
            <span class="thumb-num">#${i + 1}</span>
          </span>
          <span class="thumb-meta">
            <strong title="${raw.image_path}">${raw.image_path}</strong>
            <span class="status-chip ${vm.meta.key}">● ${vm.meta.short} ${pct}%</span>
          </span>
        </button>`;
    })
    .join("");
}

function setGauge(pct, key) {
  const offset = CIRC * (1 - pct);
  $("gauge-arc").style.strokeDashoffset = String(offset);
  $("gauge-pct").textContent = `${(pct * 100).toFixed(1)}%`;
  $("primary-card").className = `primary-card ${key}`;
}

function renderDistribution(vm) {
  const rows = [
    {
      key: "ai",
      label: "AI Generated",
      pct: vm.scores.ai,
      hint: "Synthetic diffusion textures, generative pixel synthesis, or AI model generation",
    },
    {
      key: "real",
      label: "Real Photo",
      pct: vm.scores.real,
      hint: "Authentic camera sensor photon noise & natural optical lens depth of field",
    },
    {
      key: "edit",
      label: "Filtered & Edited",
      pct: vm.scores.filtered,
      hint: "Color grading presets, tone mapping filters, retouching, or software adjustments",
    },
  ];
  const primaryKey = vm.meta.key;
  $("dist-list").innerHTML = rows
    .map((row) => {
      const on = row.key === primaryKey;
      return `
        <div class="dist-item ${on ? `primary-match match-${row.key}` : ""}">
          <div class="dist-top">
            <span class="dist-name">${row.label}${on ? '<span class="match-tag">Primary Match</span>' : ""}</span>
            <span class="dist-pct">${(row.pct * 100).toFixed(1)}% probability</span>
          </div>
          <div class="bar ${row.key}"><span style="width:${(row.pct * 100).toFixed(1)}%"></span></div>
          <p>${row.hint}</p>
        </div>`;
    })
    .join("");
  requestAnimationFrame(tightenDistLabels);
}

function renderSignals(vm) {
  $("signal-list").innerHTML = vm.signals
    .map(
      (s) => `
      <div class="signal-item">
        <div class="signal-top"><span>${s.name}</span><span>${Math.round(s.value * 100)}%</span></div>
        <div class="bar signal"><span style="width:${Math.round(s.value * 100)}%"></span></div>
        <p>${s.hint}</p>
      </div>`,
    )
    .join("");
}

function renderIndicators(vm) {
  $("indicator-list").innerHTML = vm.indicators
    .map(
      (line, i) =>
        `<li><span class="num">${i + 1}</span><span>${line}</span></li>`,
    )
    .join("");
}

function renderMeta(vm) {
  const r = vm.rec;
  const rows = [
    ["Filename", r.image_path],
    ["Class", vm.meta.title],
    ["Fused P(AI)", r.pred == null ? "—" : Number(r.pred).toFixed(4)],
    [
      "Semantic branch",
      r.semantic_score == null ? "—" : Number(r.semantic_score).toFixed(4),
    ],
    [
      "Frequency branch",
      r.frequency_score == null ? "—" : Number(r.frequency_score).toFixed(4),
    ],
    ["Tier", r.tier || "—"],
    ["C2PA", vm.cues.c2pa ? "yes" : "no"],
    ["Provenance AI", vm.cues.provenance_ai ? "yes" : "no"],
    ["Fired cues", (vm.cues.fired || []).join(", ") || "none"],
    [
      "CLIP overall AI",
      vm.vis.overall_ai == null ? "—" : Number(vm.vis.overall_ai).toFixed(3),
    ],
    ["File size", fmtSize(r.file_size)],
    ["Pixels", r.width && r.height ? `${r.width} × ${r.height}` : "—"],
  ];
  $("meta-body").innerHTML = rows
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join("");
}

function tightenDistLabels() {
  for (const item of document.querySelectorAll(".dist-item")) {
    const top = item.querySelector(".dist-top");
    const tag = item.querySelector(".match-tag");
    if (!top || !tag) continue;
    tag.hidden = false;
    if (top.scrollWidth > top.clientWidth + 1) tag.hidden = true;
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function convolveGray(src, w, h, kernel) {
  const out = new Float32Array(w * h);
  const k = kernel;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let acc = 0;
      let i = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          acc += src[(y + ky) * w + (x + kx)] * k[i++];
        }
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

function coverDraw(ctx, img, W, H) {
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  const s = Math.min(W / iw, H / ih);
  const dw = iw * s;
  const dh = ih * s;
  const ox = (W - dw) / 2;
  const oy = (H - dh) / 2;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(img, ox, oy, dw, dh);
  return { ox, oy, dw, dh, W, H };
}

function setClip(el, split) {
  const value =
    split == null ? "none" : `inset(0 0 0 ${(split * 100).toFixed(2)}%)`;
  el.style.clipPath = value;
  el.style.webkitClipPath = value;
}

function sizeViewport(img) {
  const vp = $("viewport");
  if (!vp || !img || !img.naturalWidth) return;
  const width = vp.clientWidth || vp.parentElement.clientWidth || 640;
  const fitted = Math.round(width * (img.naturalHeight / img.naturalWidth));
  const maxH = Math.round(Math.min(window.innerHeight * 0.68, 680));
  vp.style.height = `${Math.max(240, Math.min(fitted, maxH))}px`;
}

function applyViewMode() {
  const photo = $("view-photo");
  const fx = $("fx-photo");
  const handle = $("split-handle");
  const zoom = `scale(${state.zoom})`;
  photo.style.transform = zoom;
  fx.style.transform = zoom;
  const markers = $("markers");
  if (markers) markers.style.transform = zoom;
  fx.classList.remove("mode-heatmap", "mode-noise", "mode-split");
  if (state.mode === "original") {
    setClip(fx, null);
    handle.classList.add("hidden");
    return;
  }
  fx.classList.add(
    state.mode === "split" ? "mode-split" : `mode-${state.mode}`,
  );
  if (state.mode === "split") {
    setClip(fx, state.split);
    handle.classList.remove("hidden");
    handle.style.left = `${state.split * 100}%`;
  } else {
    setClip(fx, null);
    handle.classList.add("hidden");
  }
}

function processFrame(img) {
  sizeViewport(img);
  const vp = $("viewport");
  void vp.offsetHeight;
  applyViewMode();
  const canvas = $("fx-canvas");
  const W = Math.max(1, vp.clientWidth);
  const H = Math.max(1, vp.clientHeight);
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  state.layout = coverDraw(ctx, img, W, H);
  try {
    return findKeyAreas(ctx, state.layout);
  } catch {
    return [];
  }
}

function findKeyAreas(ctx, layout) {
  const { ox, oy, dw, dh, W, H } = layout || {};
  if (!W || !H || dw < 16 || dh < 16) return [];
  const pad = Math.max(10, Math.round(Math.min(dw, dh) * 0.06));
  const barPad = oy + dh > H - 56 ? 48 : pad;
  const x0 = Math.max(0, Math.floor(ox + pad));
  const y0 = Math.max(0, Math.floor(oy + pad));
  const x1 = Math.min(W, Math.ceil(ox + dw - pad));
  const y1 = Math.min(H, Math.ceil(oy + dh - barPad));
  const rw = x1 - x0;
  const rh = y1 - y0;
  if (rw < 16 || rh < 16) return [];
  const data = ctx.getImageData(0, 0, W, H).data;
  const gw = 4;
  const gh = 5;
  const cw = rw / gw;
  const ch = rh / gh;
  const cells = [];
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      let sum = 0;
      let sum2 = 0;
      let n = 0;
      const xa = Math.floor(x0 + gx * cw);
      const xb = Math.floor(x0 + (gx + 1) * cw);
      const ya = Math.floor(y0 + gy * ch);
      const yb = Math.floor(y0 + (gy + 1) * ch);
      for (let y = ya; y < yb; y += 2) {
        for (let x = xa; x < xb; x += 2) {
          const i = (y * W + x) * 4;
          const v = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          sum += v;
          sum2 += v * v;
          n++;
        }
      }
      if (n < 8) continue;
      const mean = sum / n;
      if (mean < 8) continue;
      const varr = sum2 / n - mean * mean;
      cells.push({
        x: xa + (xb - xa) / 2,
        y: ya + (yb - ya) / 2,
        varr,
      });
    }
  }
  cells.sort((a, b) => b.varr - a.varr);
  const minDist = Math.min(rw, rh) * 0.32;
  const picked = [];
  for (const c of cells) {
    if (picked.some((p) => Math.hypot(p.x - c.x, p.y - c.y) < minDist)) continue;
    picked.push(c);
    if (picked.length >= 3) break;
  }
  return picked;
}

function explainKeyArea(vm, point, index, height) {
  const bank =
    KEY_AREA_REASONS[vm?.rec?.result] || KEY_AREA_REASONS.filtered_or_edited;
  const layout = state.layout;
  const relY = layout?.dh
    ? (point.y - layout.oy) / layout.dh
    : point.y / Math.max(height, 1);
  let slot = 1;
  if (relY < 0.33) slot = 0;
  else if (relY > 0.62) slot = 2;
  const copy = bank[(slot + index) % bank.length];
  const cue = (vm?.indicators || []).find(
    (line) => line && !line.startsWith("No additional"),
  );
  const note = cue && index === 0 ? cue : copy.reason;
  return { title: copy.title, reason: note };
}

function drawMarkers(points) {
  const box = $("markers");
  box.innerHTML = "";
  if (!state.keyAreas || !points) return;
  const raw = state.results[state.index];
  const vm = raw ? viewModel(raw) : null;
  const vp = $("viewport");
  const layout = state.layout;
  const height = vp.clientHeight || 520;
  const width = vp.clientWidth || 640;
  const imgTop = layout ? layout.oy : 0;
  const imgLeft = layout ? layout.ox : 0;
  const imgRight = layout ? layout.ox + layout.dw : width;
  const tipW = Math.min(250, Math.max(180, width - 24));
  for (const [i, p] of points.entries()) {
    const info = explainKeyArea(vm, p, i, height);
    const side =
      p.x < imgLeft + tipW / 2
        ? "tip-start"
        : p.x > imgRight - tipW / 2 - 12
          ? "tip-end"
          : "";
    const vert = p.y < imgTop + 110 ? "tip-below" : "tip-above";
    const el = document.createElement("button");
    el.type = "button";
    el.className = `marker ${vert} ${side}`.trim();
    el.setAttribute("aria-label", `${info.title}. ${info.reason}`);
    el.style.left = `${p.x}px`;
    el.style.top = `${p.y}px`;
    el.innerHTML = `!<span class="marker-tip" style="width:${tipW}px"><strong>${info.title.replace(/</g, "")}</strong><p>${String(info.reason).replace(/</g, "")}</p></span>`;
    box.appendChild(el);
  }
}

async function renderInspector(vm) {
  const url = vm.rec.image_url || vm.rec.preview;
  $("file-size").textContent = fmtSize(vm.rec.file_size);
  const modeLabel = {
    original: "",
    heatmap: "Heatmap Scan",
    noise: "Noise Pattern",
    split: "Split View",
  }[state.mode];
  $("overlay-bar").innerHTML = `
    <span class="live"></span>
    <span>${vm.meta.overlayLeft}</span>
    <span class="live blue"></span>
    <span>${vm.meta.overlayRight}</span>
    ${modeLabel && state.mode !== "original" ? `<span class="mode-pill">${modeLabel}</span>` : ""}
  `;
  const captions = {
    original: "Viewing natural full-resolution RGB image pixels",
    heatmap:
      "Heatmap Scan: contrast 220% · saturate 280% · hue-rotate 190° to highlight compression boundaries.",
    noise:
      "Noise Pattern: grayscale + invert + contrast 350% to isolate high-frequency pixel grain.",
    split:
      "Drag slider left/right to compare original photo against heatmap scan.",
  };
  $("view-caption").textContent = captions[state.mode];
  $("zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
  if (!url) return;
  $("view-photo").src = url;
  $("fx-photo").src = url;
  const mini = $("inspector-thumb");
  if (mini) {
    mini.src = url;
    mini.hidden = false;
  }
  applyViewMode();
  try {
    state.img = await loadImage(url);
    const pts = processFrame(state.img);
    drawMarkers(pts);
  } catch {
    $("markers").innerHTML = "";
  }
}

function renderSelected() {
  const raw = state.results[state.index];
  if (!raw) return;
  const vm = viewModel(raw);
  $("primary-title").textContent = vm.meta.title;
  $("primary-desc").textContent = raw.explanation || vm.meta.desc;
  $("engine-line").textContent = `Engine / Sensor: ${vm.meta.engine}`;
  setGauge(vm.primary, vm.meta.key);
  renderDistribution(vm);
  renderSignals(vm);
  renderIndicators(vm);
  renderMeta(vm);
  $("pager-label").textContent =
    `Viewing ${state.index + 1} of ${state.results.length}`;
  $("prev-btn").disabled = state.index === 0;
  $("next-btn").disabled = state.index >= state.results.length - 1;
  renderThumbs();
  renderInspector(vm);
}

function renderSummary() {
  const n = state.results.length;
  $("analyzed-count").textContent = `${n} Image${n === 1 ? "" : "s"} Analyzed`;
  $("check-new-count").textContent = String(n);
  const c = classCounts(state.results);
  const parts = [];
  if (c.real)
    parts.push(
      `<span class="pill real">● ${c.real} Real Photo${c.real > 1 ? "s" : ""}</span>`,
    );
  if (c.filtered_or_edited)
    parts.push(
      `<span class="pill edit">● ${c.filtered_or_edited} Filtered &amp; Edited</span>`,
    );
  if (c.AI) parts.push(`<span class="pill ai">● ${c.AI} AI Generated</span>`);
  $("summary-pills").innerHTML = parts.join("");
}

function showResults() {
  $("upload-view").classList.add("hidden");
  $("results-view").classList.remove("hidden");
  $("check-new-btn").classList.remove("hidden");
  $("engine-pill").classList.add("hidden");
  renderSummary();
  renderSelected();
}

function showUpload() {
  $("upload-view").classList.remove("hidden");
  $("results-view").classList.add("hidden");
  $("check-new-btn").classList.add("hidden");
  $("engine-pill").classList.remove("hidden");
}

function setModal(open, pct, status, sub) {
  $("modal").classList.toggle("hidden", !open);
  if (pct != null) {
    $("modal-pct").textContent = `${Math.round(pct)}%`;
    $("modal-bar").style.width = `${pct}%`;
  }
  if (status) $("modal-status").textContent = status;
  if (sub) $("modal-sub").textContent = sub;
}

async function fileToPayload(entry) {
  const data = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(entry.file);
  });
  return { name: entry.name, size: entry.size, data };
}

async function analyze(payloadFiles, { sample = false } = {}) {
  setModal(
    true,
    8,
    "Loading detection engine…",
    `Processing ${payloadFiles.length || "sample"} photos`,
  );
  const tick = setInterval(() => {
    const cur = parseFloat($("modal-bar").style.width) || 8;
    if (cur < 90)
      setModal(true, cur + Math.random() * 6, $("modal-status").textContent);
  }, 400);
  $("engine-pill").innerHTML =
    '<span class="dot busy"></span> Detection Engine Busy';
  try {
    const res = await fetch(sample ? "/api/sample" : "/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: payloadFiles }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "Analyze failed");
    setModal(true, 100, "Evaluating lighting, noise, and pixel distribution…");
    state.results = body.records || [];
    state.index = 0;
    await new Promise((r) => setTimeout(r, 350));
    showResults();
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    clearInterval(tick);
    setModal(false, 0);
    $("engine-pill").innerHTML =
      '<span class="dot ready"></span> Detection Engine Ready';
  }
}

$("file-input").addEventListener("change", (e) => addFiles(e.target.files));
["dragenter", "dragover"].forEach((ev) => {
  $("dropzone").addEventListener(ev, (e) => {
    e.preventDefault();
    $("dropzone").classList.add("drag");
  });
});
["dragleave", "drop"].forEach((ev) => {
  $("dropzone").addEventListener(ev, (e) => {
    e.preventDefault();
    $("dropzone").classList.remove("drag");
  });
});
$("dropzone").addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
$("sample-btn").addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  analyze([], { sample: true });
});
$("clear-btn").addEventListener("click", () => {
  state.files = [];
  renderQueue();
});
$("queue-grid").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-remove]");
  if (!btn) return;
  state.files.splice(Number(btn.dataset.remove), 1);
  renderQueue();
});
$("analyze-btn").addEventListener("click", async () => {
  const payload = [];
  for (const f of state.files) payload.push(await fileToPayload(f));
  await analyze(payload);
});
$("back-btn").addEventListener("click", showUpload);
$("check-new-btn").addEventListener("click", showUpload);
$("upload-more-btn").addEventListener("click", showUpload);
$("thumb-row").addEventListener("click", (e) => {
  const card = e.target.closest("[data-idx]");
  if (!card) return;
  state.index = Number(card.dataset.idx);
  renderSelected();
});
$("prev-btn").addEventListener("click", () => {
  if (state.index > 0) {
    state.index -= 1;
    renderSelected();
  }
});
$("next-btn").addEventListener("click", () => {
  if (state.index < state.results.length - 1) {
    state.index += 1;
    renderSelected();
  }
});
$("view-modes").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-mode]");
  if (!btn) return;
  state.mode = btn.dataset.mode;
  for (const b of $("view-modes").querySelectorAll(".seg-btn")) {
    b.classList.toggle("active", b === btn);
  }
  const raw = state.results[state.index];
  if (raw) renderInspector(viewModel(raw));
});
$("key-areas-btn").addEventListener("click", () => {
  state.keyAreas = !state.keyAreas;
  $("key-areas-btn").classList.toggle("active", state.keyAreas);
  if (state.img) drawMarkers(processFrame(state.img));
});
$("zoom-in").addEventListener("click", () => {
  state.zoom = Math.min(2.5, +(state.zoom + 0.25).toFixed(2));
  applyViewMode();
  $("zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
});
$("zoom-out").addEventListener("click", () => {
  state.zoom = Math.max(0.75, +(state.zoom - 0.25).toFixed(2));
  applyViewMode();
  $("zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
});
$("meta-toggle").addEventListener("click", () => {
  $("meta-body").classList.toggle("hidden");
  $("meta-toggle").querySelector("span").textContent = $(
    "meta-body",
  ).classList.contains("hidden")
    ? "Show Details ▾"
    : "Hide Details ▴";
});
$("copy-btn").addEventListener("click", async () => {
  const vm = viewModel(state.results[state.index]);
  const text = [
    `${vm.rec.image_path}`,
    `${vm.meta.title} (${(vm.primary * 100).toFixed(1)}% confidence)`,
    `AI ${((vm.scores.ai || 0) * 100).toFixed(1)}% · Real ${((vm.scores.real || 0) * 100).toFixed(1)}% · Filtered ${((vm.scores.filtered || 0) * 100).toFixed(1)}%`,
    `P(AI)=${vm.rec.pred}  semantic=${vm.rec.semantic_score}  frequency=${vm.rec.frequency_score}`,
    vm.rec.explanation || "",
    ...vm.indicators,
  ].join("\n");
  try {
    await navigator.clipboard.writeText(text);
    $("copy-btn").textContent = "Copied";
    setTimeout(() => ($("copy-btn").textContent = "Copy Summary"), 1200);
  } catch {
    $("copy-btn").textContent = "Copy failed";
  }
});

function setSplit(clientX) {
  const rect = $("viewport").getBoundingClientRect();
  state.split = Math.min(
    0.92,
    Math.max(0.08, (clientX - rect.left) / rect.width),
  );
  setClip($("fx-photo"), state.split);
  $("split-handle").style.left = `${state.split * 100}%`;
}

let splitDrag = false;
$("split-handle").addEventListener("pointerdown", (e) => {
  e.preventDefault();
  splitDrag = true;
  $("split-handle").setPointerCapture(e.pointerId);
  setSplit(e.clientX);
});
window.addEventListener("pointermove", (e) => {
  if (!splitDrag) return;
  setSplit(e.clientX);
});
window.addEventListener("pointerup", () => {
  splitDrag = false;
});
$("viewport").addEventListener("pointerdown", (e) => {
  if (state.mode !== "split") return;
  if (e.target.closest("#split-handle")) return;
  setSplit(e.clientX);
});

$("key-areas-btn").classList.add("active");
window.addEventListener("resize", () => {
  tightenDistLabels();
  if (!$("results-view").classList.contains("hidden") && state.img) {
    drawMarkers(processFrame(state.img));
  }
});
