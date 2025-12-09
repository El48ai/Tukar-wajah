// script.js
// Pastikan index.html sudah memuat <script src="https://cdn.jsdelivr.net/npm/face-api.js"></script>
// dan memuat file ini dengan defer.

const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models/';

const sourceInput = document.getElementById('sourceImage');
const targetInput = document.getElementById('targetImage');
const previewSource = document.getElementById('previewSource');
const previewTarget = document.getElementById('previewTarget');
const swapBtn = document.getElementById('swapBtn');
const resultCanvas = document.getElementById('resultCanvas');

let srcLoaded = false;
let dstLoaded = false;

async function ensureModels() {
  try {
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
    console.log('Models ready');
  } catch (e) {
    console.error('Failed load models', e);
    alert('Gagal memuat model face-api.js. Cek koneksi internet.');
  }
}

// helper: create image element from file object
function fileToImage(file, imgElement, onLoadCb) {
  imgElement.src = URL.createObjectURL(file);
  imgElement.style.display = 'block';
  imgElement.onload = () => {
    URL.revokeObjectURL(imgElement.src);
    if (onLoadCb) onLoadCb();
  };
}

// when user selects source image
sourceInput.onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  fileToImage(f, previewSource, () => {
    srcLoaded = true;
    resizeCanvasToTarget();
  });
};

// when user selects target image
targetInput.onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  fileToImage(f, previewTarget, () => {
    dstLoaded = true;
    resizeCanvasToTarget();
  });
};

function resizeCanvasToTarget() {
  if (!dstLoaded) return;
  resultCanvas.width = previewTarget.naturalWidth || previewTarget.width;
  resultCanvas.height = previewTarget.naturalHeight || previewTarget.height;
}

// Create an alpha mask from landmark points (polygon)
function createMaskFromLandmarks(width, height, landmarks) {
  const mask = document.createElement('canvas');
  mask.width = width;
  mask.height = height;
  const mctx = mask.getContext('2d');
  mctx.fillStyle = 'black';
  mctx.fillRect(0,0,width,height);
  mctx.save();
  mctx.translate(0,0);
  mctx.beginPath();
  // use jaw + left eyebrow + right eyebrow as polygon approx
  const pts = landmarks.getJawOutline().map(p => [p.x, p.y])
    .concat(landmarks.getLeftEyeBrow().map(p=>[p.x,p.y]))
    .concat(landmarks.getRightEyeBrow().map(p=>[p.x,p.y]));
  if (pts.length>0) {
    mctx.moveTo(pts[0][0], pts[0][1]);
    for (let i=1;i<pts.length;i++) mctx.lineTo(pts[i][0], pts[i][1]);
    mctx.closePath();
    mctx.fillStyle = 'white';
    mctx.fill();
  }
  mctx.restore();
  return mask;
}

// Main swap function (simple crop+scale+mask)
async function doSwapSimple() {
  if (!srcLoaded || !dstLoaded) {
    alert('Pilih kedua foto terlebih dahulu.');
    return;
  }
  swapBtn.disabled = true;
  try {
    const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.4 });

    const [resSrc, resDst] = await Promise.all([
      faceapi.detectSingleFace(previewSource, opts).withFaceLandmarks(true),
      faceapi.detectSingleFace(previewTarget, opts).withFaceLandmarks(true)
    ]);

    if (!resSrc || !resDst) {
      alert('Wajah tidak terdeteksi pada salah satu foto. Coba foto lain atau perbesar wajah pada foto.');
      swapBtn.disabled = false;
      return;
    }

    // canvas setup: result will be base target image
    resizeCanvasToTarget();
    const ctx = resultCanvas.getContext('2d');
    ctx.clearRect(0,0,resultCanvas.width,resultCanvas.height);
    ctx.drawImage(previewTarget, 0, 0, resultCanvas.width, resultCanvas.height);

    // determine bounding boxes
    const boxSrc = resSrc.detection.box;
    const boxDst = resDst.detection.box;

    // crop source face to temp canvas
    const tmp = document.createElement('canvas');
    tmp.width = boxSrc.width;
    tmp.height = boxSrc.height;
    const tctx = tmp.getContext('2d');
    // adjust if source image is smaller than detection box
    tctx.drawImage(previewSource, boxSrc.x, boxSrc.y, boxSrc.width, boxSrc.height, 0, 0, tmp.width, tmp.height);

    // create mask from dst landmarks, scaled to result canvas coordinates
    // landmarks are in image coordinates of previewTarget natural size -> we assume previewTarget drawn 1:1 on canvas
    const dstLandmarks = resDst.landmarks;
    // create mask same size as result canvas
    const mask = document.createElement('canvas');
    mask.width = resultCanvas.width;
    mask.height = resultCanvas.height;
    const mctx = mask.getContext('2d');
    mctx.clearRect(0,0,mask.width,mask.height);
    mctx.fillStyle = 'black';
    mctx.fillRect(0,0,mask.width,mask.height);
    mctx.beginPath();
    const jaw = dstLandmarks.getJawOutline();
    // draw polygon around jaw (smooth shape)
    if (jaw.length>0) {
      mctx.moveTo(jaw[0].x, jaw[0].y);
      for (let i=1;i<jaw.length;i++) mctx.lineTo(jaw[i].x, jaw[i].y);
      // close top using eyebrows
      const leftBrow = dstLandmarks.getLeftEyeBrow();
      const rightBrow = dstLandmarks.getRightEyeBrow();
      if (leftBrow.length>0) {
        for (let i=leftBrow.length-1;i>=0;i--) mctx.lineTo(leftBrow[i].x, leftBrow[i].y);
      }
      if (rightBrow.length>0) {
        for (let i=0;i<rightBrow.length;i++) mctx.lineTo(rightBrow[i].x, rightBrow[i].y);
      }
      mctx.closePath();
      mctx.fillStyle = 'white';
      mctx.fill();
    }

    // now we need to draw tmp (source face) scaled into dst box and apply mask
    // create a temporary canvas same size as result
    const tempPlacement = document.createElement('canvas');
    tempPlacement.width = resultCanvas.width;
    tempPlacement.height = resultCanvas.height;
    const pctx = tempPlacement.getContext('2d');
    // compute scale factors to map tmp to dst box
    const scaleX = boxDst.width / tmp.width;
    const scaleY = boxDst.height / tmp.height;
    // draw tmp scaled to dst box position
    pctx.save();
    pctx.globalAlpha = 0.98;
    pctx.drawImage(tmp, 0, 0, tmp.width, tmp.height, boxDst.x, boxDst.y, tmp.width * scaleX, tmp.height * scaleY);
    pctx.restore();

    // apply mask: use mask as alpha
    // we will composite by copying only pixels where mask is white
    const maskData = mask.getContext('2d').getImageData(0,0,mask.width,mask.height).data;
    const placeData = pctx.getImageData(0,0,tempPlacement.width,tempPlacement.height);
    const resData = ctx.getImageData(0,0,resultCanvas.width,resultCanvas.height);

    // Blend: if mask pixel alpha > 0 then use placeData pixel; simple copy
    for (let i=0;i<maskData.length;i+=4) {
      const alpha = maskData[i]; // white -> 255
      if (alpha > 10) {
        resData.data[i] = placeData.data[i];
        resData.data[i+1] = placeData.data[i+1];
        resData.data[i+2] = placeData.data[i+2];
        resData.data[i+3] = 255;
      }
    }
    ctx.putImageData(resData, 0, 0);

    alert('Selesai: hasil demo. Untuk hasil lebih halus, butuh blending/feathering.');
  } catch (err) {
    console.error(err);
    alert('Terjadi kesalahan: ' + (err.message || err));
  } finally {
    swapBtn.disabled = false;
  }
};

swapBtn.onclick = async () => {
  swapBtn.disabled = true;
  await ensureModels();
  await doSwapSimple();
};
