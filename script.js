// ===== Helpers =====
const $ = (id) => document.getElementById(id);

const el = {
  sourceInput: $("sourceImage"),
  targetInput: $("targetImage"),
  sourcePreview: $("sourcePreview"),
  targetPreview: $("targetPreview"),
  statusDot: $("statusDot"),
  status: $("statusMessage"),
  loadingBar: $("loading"),
  loadingText: $("loadingText"),
  feather: $("featherRange"),
  featherLabel: $("featherValLabel"),
  colorBlend: $("colorBlend"),
  colorBlendLabel: $("colorBlendLabel"),
  swapBtn: $("swapBtn"),
  resetBtn: $("resetBtn"),
  downloadBtn: $("downloadBtn"),
  canvas: $("resultCanvas"),
  resultEmpty: $("resultEmpty"),
};

// ===== UI helpers =====
function setStatus(msg, state = "loading") {
  el.status.textContent = msg;
  el.statusDot.className = "status-dot " + state;
}

function setLoading(on, text = "Memproses…") {
  el.loadingBar.className = "loading-bar" + (on ? " show" : "");
  el.loadingText.textContent = text;
}

function showPreview(container, imgEl) {
  container.innerHTML = "";
  container.appendChild(imgEl);
}

function makeImgFromFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

async function drawScaled(img, maxSide = 1024) {
  const ratio = Math.min(1, maxSide / Math.max(img.width, img.height));
  const w = Math.round(img.width * ratio);
  const h = Math.round(img.height * ratio);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d", { willReadFrequently: true }).drawImage(img, 0, 0, w, h);
  return c;
}

// ===== Slider live labels =====
el.feather.addEventListener("input", () => {
  el.featherLabel.textContent = parseFloat(el.feather.value).toFixed(2);
});
el.colorBlend.addEventListener("input", () => {
  el.colorBlendLabel.textContent = el.colorBlend.value + "%";
});

// ===== Model loading =====
// Pakai CDN jsDelivr — tidak perlu upload model ke repo
const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model";
let modelsReady = false;

async function loadModels() {
  try {
    setLoading(true, "Memuat model AI dari CDN…");
    setStatus("Memuat model AI…", "loading");

    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    setLoading(true, "Memuat landmark model…");
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

    modelsReady = true;
    setStatus("Model siap ✅  Pilih 2 foto lalu klik Tukar Wajah.", "ready");
    maybeEnableSwap();
  } catch (err) {
    console.error("Model load error:", err);
    // Fallback: coba path lokal ./models
    try {
      setLoading(true, "CDN gagal, coba model lokal…");
      await faceapi.nets.tinyFaceDetector.loadFromUri("./models");
      await faceapi.nets.faceLandmark68Net.loadFromUri("./models");
      modelsReady = true;
      setStatus("Model lokal siap ✅  Pilih 2 foto lalu klik Tukar Wajah.", "ready");
      maybeEnableSwap();
    } catch (err2) {
      console.error("Local model error:", err2);
      setStatus("Gagal memuat model ❌  Cek koneksi internet.", "error");
    }
  } finally {
    setLoading(false);
  }
}

// ===== State =====
let sourceImg = null;
let targetImg = null;

function maybeEnableSwap() {
  el.swapBtn.disabled = !(modelsReady && sourceImg && targetImg);
}

// ===== Face detection =====
async function detectFace(canvasEl) {
  const options = new faceapi.TinyFaceDetectorOptions({
    inputSize: 416,
    scoreThreshold: 0.4,
  });
  return faceapi
    .detectSingleFace(canvasEl, options)
    .withFaceLandmarks();
}

// Bounding box dari landmarks dengan padding
function getBBox(landmarks, canvasW, canvasH) {
  const pts = landmarks.positions;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const w = maxX - minX;
  const h = maxY - minY;
  const padX = w * 0.28;
  const padY = h * 0.38;
  return {
    x: Math.max(0, minX - padX),
    y: Math.max(0, minY - padY),
    w: Math.min(canvasW, w + padX * 2),
    h: Math.min(canvasH, h + padY * 2),
  };
}

// ===== Color matching =====
// Ambil rata-rata warna area wajah
function getAvgColor(ctx, x, y, w, h) {
  const data = ctx.getImageData(
    Math.round(x), Math.round(y),
    Math.max(1, Math.round(w)), Math.max(1, Math.round(h))
  ).data;
  let r = 0, g = 0, b = 0, count = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 128) continue;
    r += data[i]; g += data[i + 1]; b += data[i + 2];
    count++;
  }
  if (!count) return { r: 128, g: 100, b: 90 };
  return { r: r / count, g: g / count, b: b / count };
}

// Terapkan color shift supaya warna kulit sumber mendekati target
function colorMatch(srcCtx, tgtAvg, srcAvg, blendRatio, w, h) {
  const imgData = srcCtx.getImageData(0, 0, w, h);
  const d = imgData.data;
  const ratio = blendRatio / 100;
  const dr = (tgtAvg.r - srcAvg.r) * ratio;
  const dg = (tgtAvg.g - srcAvg.g) * ratio;
  const db = (tgtAvg.b - srcAvg.b) * ratio;
  for (let i = 0; i < d.length; i += 4) {
    d[i]   = Math.min(255, Math.max(0, d[i]   + dr));
    d[i+1] = Math.min(255, Math.max(0, d[i+1] + dg));
    d[i+2] = Math.min(255, Math.max(0, d[i+2] + db));
  }
  srcCtx.putImageData(imgData, 0, 0);
}

// ===== Feather mask =====
function applyFeatherMask(ctx, x, y, w, h, feather) {
  ctx.globalCompositeOperation = "destination-in";
  const cx = x + w / 2, cy = y + h / 2;
  const rx = w / 2, ry = h / 2;
  const innerScale = Math.max(0.1, 1 - feather * 1.2);
  const grad = ctx.createRadialGradient(
    cx, cy, Math.min(rx, ry) * innerScale * 0.5,
    cx, cy, Math.max(rx, ry) * 1.05
  );
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(innerScale, "rgba(255,255,255,1)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx * 1.1, ry * 1.1, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = "source-over";
}

// ===== Main swap =====
async function doSwap() {
  if (!modelsReady || !sourceImg || !targetImg) return;

  el.swapBtn.disabled = true;
  el.downloadBtn.style.display = "none";
  el.canvas.style.display = "none";
  el.resultEmpty.style.display = "block";

  try {
    setLoading(true, "Mendeteksi wajah…");
    setStatus("Mendeteksi wajah…", "loading");

    const sourceC = await drawScaled(sourceImg, 1024);
    const targetC = await drawScaled(targetImg, 1024);

    const [srcDet, tgtDet] = await Promise.all([
      detectFace(sourceC),
      detectFace(targetC),
    ]);

    if (!srcDet) {
      setStatus("Wajah di Foto Sumber tidak terdeteksi ❌  Coba foto yang lebih jelas.", "error");
      return;
    }
    if (!tgtDet) {
      setStatus("Wajah di Foto Target tidak terdeteksi ❌  Coba foto yang lebih jelas.", "error");
      return;
    }

    setLoading(true, "Memproses wajah…");

    const feather = parseFloat(el.feather.value);
    const colorBlendRatio = parseInt(el.colorBlend.value, 10);

    // Setup output canvas
    el.canvas.width = targetC.width;
    el.canvas.height = targetC.height;
    const outCtx = el.canvas.getContext("2d", { willReadFrequently: true });
    outCtx.drawImage(targetC, 0, 0);

    // Bbox
    const srcBox = getBBox(srcDet.landmarks, sourceC.width, sourceC.height);
    const tgtBox = getBBox(tgtDet.landmarks, targetC.width, targetC.height);

    // Crop wajah sumber ke canvas terpisah
    const faceW = Math.max(1, Math.round(srcBox.w));
    const faceH = Math.max(1, Math.round(srcBox.h));
    const faceCrop = document.createElement("canvas");
    faceCrop.width = faceW; faceCrop.height = faceH;
    const cropCtx = faceCrop.getContext("2d", { willReadFrequently: true });
    cropCtx.drawImage(sourceC, srcBox.x, srcBox.y, srcBox.w, srcBox.h, 0, 0, faceW, faceH);

    // Color matching
    const tgtCtxTmp = document.createElement("canvas");
    tgtCtxTmp.width = targetC.width; tgtCtxTmp.height = targetC.height;
    const tctx = tgtCtxTmp.getContext("2d", { willReadFrequently: true });
    tctx.drawImage(targetC, 0, 0);

    const srcAvg = getAvgColor(cropCtx, 0, 0, faceW, faceH);
    const tgtAvg = getAvgColor(tctx, tgtBox.x, tgtBox.y, tgtBox.w, tgtBox.h);
    colorMatch(cropCtx, tgtAvg, srcAvg, colorBlendRatio, faceW, faceH);

    // Layer untuk blend
    const layer = document.createElement("canvas");
    layer.width = el.canvas.width; layer.height = el.canvas.height;
    const lctx = layer.getContext("2d", { willReadFrequently: true });

    // Gambar wajah sumber ke posisi target
    lctx.drawImage(
      faceCrop,
      Math.round(tgtBox.x), Math.round(tgtBox.y),
      Math.round(tgtBox.w), Math.round(tgtBox.h)
    );

    // Feather mask
    applyFeatherMask(lctx, tgtBox.x, tgtBox.y, tgtBox.w, tgtBox.h, feather);

    // Blend ke output
    outCtx.drawImage(layer, 0, 0);

    setStatus("Selesai ✅  Klik Download untuk menyimpan.", "ready");
    el.resultEmpty.style.display = "none";
    el.canvas.style.display = "block";
    el.downloadBtn.style.display = "inline-block";

  } catch (err) {
    console.error(err);
    setStatus("Error ❌  " + (err.message || "Coba lagi dengan foto berbeda."), "error");
  } finally {
    setLoading(false);
    maybeEnableSwap();
  }
}

// ===== Events =====
el.sourceInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0]; if (!file) return;
  sourceImg = await makeImgFromFile(file);
  const preview = new Image();
  preview.src = sourceImg.src;
  preview.style.cssText = "max-width:100%;height:auto;display:block;border-radius:10px";
  showPreview(el.sourcePreview, preview);
  setStatus(modelsReady ? "Foto sumber siap ✅" : "Foto sumber dipilih. Menunggu model…",
    modelsReady ? "ready" : "loading");
  maybeEnableSwap();
});

el.targetInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0]; if (!file) return;
  targetImg = await makeImgFromFile(file);
  const preview = new Image();
  preview.src = targetImg.src;
  preview.style.cssText = "max-width:100%;height:auto;display:block;border-radius:10px";
  showPreview(el.targetPreview, preview);
  setStatus(modelsReady ? "Foto target siap ✅" : "Foto target dipilih. Menunggu model…",
    modelsReady ? "ready" : "loading");
  maybeEnableSwap();
});

el.swapBtn.addEventListener("click", doSwap);

el.resetBtn.addEventListener("click", () => {
  sourceImg = null; targetImg = null;
  el.sourceInput.value = ""; el.targetInput.value = "";
  el.sourcePreview.innerHTML = `<div class="preview-empty"><div class="icon">🖼️</div><div>Belum ada foto</div></div>`;
  el.targetPreview.innerHTML = `<div class="preview-empty"><div class="icon">🖼️</div><div>Belum ada foto</div></div>`;
  el.canvas.style.display = "none";
  el.resultEmpty.style.display = "block";
  el.downloadBtn.style.display = "none";
  setStatus(modelsReady ? "Reset ✅  Pilih 2 foto lagi." : "Reset. Menunggu model…",
    modelsReady ? "ready" : "loading");
  maybeEnableSwap();
});

el.downloadBtn.addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = "tukar-wajah-hasil.png";
  a.href = el.canvas.toDataURL("image/png");
  a.click();
});

// Start
loadModels();
