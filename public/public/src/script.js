// src/script.js
const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models/';

const fileSrc = document.getElementById('fileSrc');
const fileDst = document.getElementById('fileDst');
const imgSrc = document.getElementById('imgSrc');
const imgDst = document.getElementById('imgDst');
const btnSwap = document.getElementById('btnSwap');
const btnClear = document.getElementById('btnClear');
const out = document.getElementById('out');

let srcLoaded = false;
let dstLoaded = false;

async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
  console.log('models loaded');
}

fileSrc.onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  imgSrc.src = URL.createObjectURL(f);
  imgSrc.onload = () => { srcLoaded = true; resizeCanvas(); }
};

fileDst.onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  imgDst.src = URL.createObjectURL(f);
  imgDst.onload = () => { dstLoaded = true; resizeCanvas(); }
};

function resizeCanvas() {
  if (!dstLoaded) return;
  out.width = imgDst.naturalWidth;
  out.height = imgDst.naturalHeight;
}

btnClear.onclick = () => {
  imgSrc.src = '';
  imgDst.src = '';
  out.getContext('2d').clearRect(0,0,out.width,out.height);
  srcLoaded = dstLoaded = false;
}

btnSwap.onclick = async () => {
  if (!srcLoaded || !dstLoaded) { alert('Pilih kedua foto terlebih dahulu'); return; }
  btnSwap.disabled = true;
  try {
    const opts = new faceapi.TinyFaceDetectorOptions();
    const resSrc = await faceapi.detectSingleFace(imgSrc, opts).withFaceLandmarks(true);
    const resDst = await faceapi.detectSingleFace(imgDst, opts).withFaceLandmarks(true);
    if (!resSrc || !resDst) { alert('Wajah tidak terdeteksi pada salah satu foto'); btnSwap.disabled=false; return; }

    const srcPts = resSrc.landmarks.positions.map(p => [p.x, p.y]);
    const dstPts = resDst.landmarks.positions.map(p => [p.x, p.y]);

    const canvas = out;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);
    // draw target as base
    ctx.drawImage(imgDst, 0, 0, canvas.width, canvas.height);

    // simple approach: draw src face scaled to dst bounding box
    const srcBox = resSrc.detection.box;
    const dstBox = resDst.detection.box;

    // create temp canvas with cropped source face
    const tmp = document.createElement('canvas');
    tmp.width = srcBox.width; tmp.height = srcBox.height;
    const tctx = tmp.getContext('2d');
    tctx.drawImage(imgSrc, srcBox.x, srcBox.y, srcBox.width, srcBox.height, 0, 0, tmp.width, tmp.height);

    // draw onto dst area
    ctx.save();
    ctx.globalAlpha = 0.95;
    ctx.drawImage(tmp, 0, 0, tmp.width, tmp.height, dstBox.x, dstBox.y, dstBox.width, dstBox.height);
    ctx.restore();

    alert('Selesai (demo). Untuk hasil lebih rapi butuh blending / triangulation.');
  } catch (e) {
    console.error(e);
    alert('Terjadi error: ' + e.message);
  } finally {
    btnSwap.disabled = false;
  }
};

loadModels().catch(e => { console.error('model load error', e); alert('Gagal muat model') });
