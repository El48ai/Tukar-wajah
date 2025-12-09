// script.js - logic utama face-swap
// Pastikan index.html memuat face-api.js sebelum file ini (defer)

let sourceImg = null;
let targetImg = null;
let modelsLoaded = false;

// Helper DOM
const sourceInput = () => document.getElementById('sourceImage');
const targetInput = () => document.getElementById('targetImage');
const sourcePreview = () => document.getElementById('sourcePreview');
const targetPreview = () => document.getElementById('targetPreview');
const swapBtn = () => document.getElementById('swapBtn');
const resetBtn = () => document.getElementById('resetBtn');
const downloadBtn = () => document.getElementById('downloadBtn');
const statusEl = () => document.getElementById('statusMessage');
const loadingEl = () => document.getElementById('loading');
const loadingText = () => document.getElementById('loadingText');
const featherRange = () => document.getElementById('featherRange');
const canvasEl = () => document.getElementById('resultCanvas');

// ---------------- models ----------------
async function loadModels() {
  try {
    setStatus('Memuat model AI...', 'info');
    // MODEL URL publik (workable)
    const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models/';
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

    modelsLoaded = true;
    setStatus('Model AI dimuat. Silakan unggah gambar.', 'success');
    checkReadyToSwap();
  } catch (err) {
    console.error('Load models error', err);
    setStatus('Gagal memuat model. Periksa koneksi internet.', 'error');
  }
}

// ---------------- UI helpers ----------------
function setStatus(text, type = 'info') {
  const el = statusEl();
  el.textContent = text;
  el.className = `status ${type}`;
}
function showLoading(on, text = '') {
  const el = loadingEl();
  if (on) {
    el.classList.add('active');
    if (text) loadingText().textContent = text;
  } else {
    el.classList.remove('active');
  }
}

// ---------------- file handling ----------------
function handleFileInput(inputEl, previewEl, which) {
  inputEl.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        // Simpan reference image (HTMLImageElement) untuk face-api
        if (which === 'source') sourceImg = img;
        else targetImg = img;

        previewEl.innerHTML = `
          <img src="${ev.target.result}" alt="preview" />
          <button class="remove-btn" onclick="removeImage('${which}')">×</button>
        `;
        checkReadyToSwap();
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  });
}

window.removeImage = function(type) {
  if (type === 'source') {
    sourceImg = null;
    sourcePreview().innerHTML = '';
    sourceInput().value = '';
  } else {
    targetImg = null;
    targetPreview().innerHTML = '';
    targetInput().value = '';
  }
  checkReadyToSwap();
};

// ---------------- readiness ----------------
function checkReadyToSwap() {
  swapBtn().disabled = !(sourceImg && targetImg && modelsLoaded);
}

// ---------------- main swap logic ----------------
async function swapFaces() {
  try {
    showLoading(true, 'Mendeteksi wajah...');
    setStatus('Mendeteksi wajah...', 'info');

    // opsi deteksi (sensitivitas)
    const options = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.45 });

    // jalankan deteksi parallel
    const [sDet, tDet] = await Promise.all([
      faceapi.detectSingleFace(sourceImg, options).withFaceLandmarks(),
      faceapi.detectSingleFace(targetImg, options).withFaceLandmarks()
    ]);

    if (!sDet) throw new Error('Wajah tidak terdeteksi pada foto sumber.');
    if (!tDet) throw new Error('Wajah tidak terdeteksi pada foto target.');

    setStatus('Memproses penukaran...', 'info');

    // canvas target
    const canvas = canvasEl();
    const ctx = canvas.getContext('2d');

    // set ukuran canvas sesuai target image natural size
    canvas.width = targetImg.naturalWidth;
    canvas.height = targetImg.naturalHeight;

    // draw full target as base
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(targetImg, 0, 0, canvas.width, canvas.height);

    // ambil box & padding adaptif
    const sBox = sDet.detection.box;
    const tBox = tDet.detection.box;

    const paddingFactor = 0.45; // bisa diatur
    const sPadding = Math.round(Math.max(sBox.width, sBox.height) * paddingFactor);

    const tempW = Math.round(sBox.width + sPadding * 2);
    const tempH = Math.round(sBox.height + sPadding * 2);

    // canvas sementara untuk wajah sumber yang sudah dicrop
    const temp = document.createElement('canvas');
    temp.width = tempW; temp.height = tempH;
    const tctx = temp.getContext('2d');

    tctx.drawImage(
      sourceImg,
      sBox.x - sPadding,
      sBox.y - sPadding,
      sBox.width + sPadding * 2,
      sBox.height + sPadding * 2,
      0,0,tempW,tempH
    );

    // buat mask feather di tempMask
    const mask = document.createElement('canvas');
    mask.width = tempW; mask.height = tempH;
    const mctx = mask.getContext('2d');

    // radial gradient putih -> transparent
    const cx = tempW/2, cy = tempH/2;
    const r0 = Math.min(sBox.width, sBox.height) * 0.15;
    // r1 dikontrol oleh range feather
    const featherVal = parseFloat(featherRange().value); // 0..1
    const r1 = Math.max(tempW, tempH) * (0.35 + featherVal * 0.5);

    const grad = mctx.createRadialGradient(cx,cy,r0,cx,cy,r1);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    mctx.fillStyle = grad;
    mctx.fillRect(0,0,mask.width,mask.height);

    // destination-in untuk membatasi area wajah
    tctx.globalCompositeOperation = 'destination-in';
    tctx.drawImage(mask,0,0);
    tctx.globalCompositeOperation = 'source-over';

    // hitung ukuran gambar akhir di target
    // scaling berdasarkan perbandingan box source->target
    const scaleX = tBox.width / sBox.width;
    const scaleY = tBox.height / sBox.height;
    const drawW = Math.round(tempW * scaleX);
    const drawH = Math.round(tempH * scaleY);
    const drawX = Math.round(tBox.x - sPadding * scaleX);
    const drawY = Math.round(tBox.y - sPadding * scaleY);

    // draw wajah yang sudah dimask ke canvas target (blending)
    ctx.save();
    ctx.globalAlpha = 0.96;
    ctx.drawImage(temp, drawX, drawY, drawW, drawH);
    ctx.restore();

    // tampilkan canvas & tombol download
    canvas.style.display = 'block';
    downloadBtn().style.display = 'inline-block';
    setStatus('Selesai — coba unduh atau reset untuk mencoba lagi.', 'success');
  } catch (err) {
    console.error(err);
    setStatus('Error: ' + (err.message || err), 'error');
  } finally {
    showLoading(false);
  }
}

// ---------------- utility: download & reset ----------------
function downloadResult() {
  const canvas = canvasEl();
  const link = document.createElement('a');
  link.href = canvas.toDataURL('image/png');
  link.download = 'face-swap.png';
  link.click();
  setStatus('Gambar diunduh.', 'success');
}

function resetAll() {
  sourceImg = null; targetImg = null;
  sourceInput().value = ''; targetInput().value = '';
  sourcePreview().innerHTML = ''; targetPreview().innerHTML = '';
  canvasEl().style.display = 'none';
  downloadBtn().style.display = 'none';
  setStatus('Reset selesai.', 'info');
  checkReadyToSwap();
}

// ---------------- init ----------------
window.addEventListener('DOMContentLoaded', () => {
  loadModels();
  handleFileInput(sourceInput(), sourcePreview(), 'source');
  handleFileInput(targetInput(), targetPreview(), 'target');

  swapBtn().addEventListener('click', swapFaces);
  resetBtn().addEventListener('click', resetAll);
  downloadBtn().addEventListener('click', downloadResult);
});

// expose removeImage globally (dipakai oleh tombol remove preview)
window.removeImage = function(which){
  if(which==='source') {
    sourceImg=null; sourcePreview().innerHTML=''; sourceInput().value='';
  } else {
    targetImg=null; targetPreview().innerHTML=''; targetInput().value='';
  }
  checkReadyToSwap();
};
