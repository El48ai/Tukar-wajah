// ===== Helpers UI =====
const $ = (id) => document.getElementById(id);

const el = {
  sourceInput: $("sourceImage"),
  targetInput: $("targetImage"),
  sourcePreview: $("sourcePreview"),
  targetPreview: $("targetPreview"),
  status: $("statusMessage"),
  loadingWrap: $("loading"),
  loadingText: $("loadingText"),
  feather: $("featherRange"),
  swapBtn: $("swapBtn"),
  resetBtn: $("resetBtn"),
  downloadBtn: $("downloadBtn"),
  canvas: $("resultCanvas"),
};

function setStatus(msg) {
  el.status.textContent = msg;
}

function setLoading(on, text = "Tunggu sebentar…") {
  el.loadingWrap.style.display = on ? "flex" : "none";
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

// Resize image to limit size (lebih ringan di HP)
async function drawImageToCanvasScaled(img, maxSide = 1024) {
  const ratio = Math.min(1, maxSide / Math.max(img.width, img.height));
  const w = Math.round(img.width * ratio);
  const h = Math.round(img.height * ratio);

  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;

  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);

  return c;
}

// ===== Model Loading (GitHub Pages safe) =====
let modelsReady = false;

async function loadModels() {
  try {
    setLoading(true, "Memuat komponen…");
    setStatus("Memuat komponen…");

    // IMPORTANT: path relatif untuk GitHub Pages
    const MODEL_URL = "./models";

    // Model minimal untuk deteksi + landmark + descriptor
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

    modelsReady = true;
    setStatus("Model siap ✅ Pilih 2 foto lalu klik Tukar Wajah.");
    maybeEnableSwap();
  } catch (err) {
    console.error(err);
    setStatus("Gagal memuat model ❌ Pastikan folder ./models ada dan path benar.");
  } finally {
    setLoading(false);
  }
}

// ===== State =====
let sourceImg = null;
let targetImg = null;

// Enable swap kalau semua siap
function maybeEnableSwap() {
  const ok = modelsReady && sourceImg && targetImg;
  el.swapBtn.disabled = !ok;
}

// ===== Face ops =====
async function detectOneFace(canvasOrImg) {
  const options = new faceapi.TinyFaceDetectorOptions({
    inputSize: 320, // cukup ringan untuk HP
    scoreThreshold: 0.5,
  });

  return faceapi
    .detectSingleFace(canvasOrImg, options)
    .withFaceLandmarks()
    .withFaceDescriptor();
}

function bboxFromLandmarks(landmarks) {
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

  // perbesar sedikit biar area wajah tidak kepotong
  const padX = w * 0.25;
  const padY = h * 0.35;

  return {
    x: Math.max(0, minX - padX),
    y: Math.max(0, minY - padY),
    w: w + padX * 2,
    h: h + padY * 2,
  };
}

// Feather mask: bikin pinggir lembut
function applyFeatherMask(ctx, x, y, w, h, feather) {
  const fx = Math.max(6, w * feather);
  const fy = Math.max(6, h * feather);

  ctx.globalCompositeOperation = "destination-in";
  const grad = ctx.createRadialGradient(
    x + w / 2, y + h / 2, Math.max(w, h) * 0.20,
    x + w / 2, y + h / 2, Math.max(w, h) * 0.55
  );
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(1, "rgba(255,255,255,0)");

  // Mask ellipse-ish
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(x + w / 2, y + h / 2, w / 2 + fx * 0.1, h / 2 + fy * 0.1, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = "source-over";
}

async function doSwap() {
  if (!modelsReady) {
    setStatus("Model belum siap. Tunggu sampai selesai load.");
    return;
  }
  if (!sourceImg || !targetImg) {
    setStatus("Pilih foto sumber dan target dulu.");
    return;
  }

  el.swapBtn.disabled = true;
  el.downloadBtn.style.display = "none";
  el.canvas.style.display = "none";

  try {
    setLoading(true, "Mendeteksi wajah…");
    setStatus("Mendeteksi wajah…");

    // Gambar ke canvas scaled (lebih ringan)
    const sourceC = await drawImageToCanvasScaled(sourceImg, 1024);
    const targetC = await drawImageToCanvasScaled(targetImg, 1024);

    const [srcDet, tgtDet] = await Promise.all([
      detectOneFace(sourceC),
      detectOneFace(targetC),
    ]);

    if (!srcDet) {
      setStatus("Wajah di Foto Sumber tidak terdeteksi ❌ Coba foto yang lebih jelas.");
      return;
    }
    if (!tgtDet) {
      setStatus("Wajah di Foto Target tidak terdeteksi ❌ Coba foto yang lebih jelas.");
      return;
    }

    setLoading(true, "Membuat hasil…");
    setStatus("Memproses hasil…");

    const feather = parseFloat(el.feather.value);

    // Set output canvas = target size
    el.canvas.width = targetC.width;
    el.canvas.height = targetC.height;
    const outCtx = el.canvas.getContext("2d", { willReadFrequently: true });

    // draw target as base
    outCtx.drawImage(targetC, 0, 0);

    // crop wajah sumber
    const srcBox = bboxFromLandmarks(srcDet.landmarks);
    const faceCrop = document.createElement("canvas");
    faceCrop.width = Math.max(1, Math.round(srcBox.w));
    faceCrop.height = Math.max(1, Math.round(srcBox.h));
    const cropCtx = faceCrop.getContext("2d", { willReadFrequently: true });
    cropCtx.drawImage(
      sourceC,
      srcBox.x, srcBox.y, srcBox.w, srcBox.h,
      0, 0, faceCrop.width, faceCrop.height
    );

    // target placement = bbox target
    const tgtBox = bboxFromLandmarks(tgtDet.landmarks);

    // temp layer for blending
    const layer = document.createElement("canvas");
    layer.width = el.canvas.width;
    layer.height = el.canvas.height;
    const lctx = layer.getContext("2d", { willReadFrequently: true });

    // draw resized source face onto target area
    lctx.drawImage(faceCrop, tgtBox.x, tgtBox.y, tgtBox.w, tgtBox.h);

    // feather mask on the layer
    applyFeatherMask(lctx, tgtBox.x, tgtBox.y, tgtBox.w, tgtBox.h, feather);

    // blend onto output
    outCtx.globalAlpha = 1.0;
    outCtx.drawImage(layer, 0, 0);

    setStatus("Selesai ✅");
    el.canvas.style.display = "block";
    el.downloadBtn.style.display = "inline-block";
  } catch (err) {
    console.error(err);
    setStatus("Terjadi error ❌ Buka console untuk detail. (Biasanya path model / file besar)");
  } finally {
    setLoading(false);
    maybeEnableSwap();
  }
}

// ===== Events =====
el.sourceInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  sourceImg = await makeImgFromFile(file);
  showPreview(el.sourcePreview, sourceImg);
  setStatus(modelsReady ? "Foto sumber siap ✅" : "Foto sumber dipilih. Menunggu model…");
  maybeEnableSwap();
});

el.targetInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  targetImg = await makeImgFromFile(file);
  showPreview(el.targetPreview, targetImg);
  setStatus(modelsReady ? "Foto target siap ✅" : "Foto target dipilih. Menunggu model…");
  maybeEnableSwap();
});

el.swapBtn.addEventListener("click", doSwap);

el.resetBtn.addEventListener("click", () => {
  sourceImg = null;
  targetImg = null;
  el.sourceInput.value = "";
  el.targetInput.value = "";
  el.sourcePreview.innerHTML = "";
  el.targetPreview.innerHTML = "";
  el.canvas.style.display = "none";
  el.downloadBtn.style.display = "none";
  setStatus(modelsReady ? "Reset ✅ Pilih 2 foto lagi." : "Reset ✅ Menunggu model…");
  maybeEnableSwap();
});

el.downloadBtn.addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = "hasil-face-swap.png";
  a.href = el.canvas.toDataURL("image/png");
  a.click();
});

// Start
loadModels();
