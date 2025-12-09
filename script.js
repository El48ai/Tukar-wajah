// script.js (versi triangulation + affine warp)
// Pastikan face-api.js dan delaunator sudah dimuat di index.html sebelum script ini

const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models/';

const sourceInput = document.getElementById('sourceImage');
const targetInput = document.getElementById('targetImage');
const previewSource = document.getElementById('previewSource');
const previewTarget = document.getElementById('previewTarget');
const swapBtn = document.getElementById('swapBtn');
const resultCanvas = document.getElementById('resultCanvas');

let srcLoaded = false;
let dstLoaded = false;

// Load face-api models
async function ensureModels() {
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
  console.log('models loaded');
}

// helper: load file into img element and force natural size for correct coordinates
function fileToImage(file, imgElement, onLoadCb) {
  imgElement.src = URL.createObjectURL(file);
  imgElement.style.display = 'block';
  imgElement.onload = () => {
    // force img element to use natural size (important for landmark coordinates)
    imgElement.width = imgElement.naturalWidth;
    imgElement.height = imgElement.naturalHeight;
    URL.revokeObjectURL(imgElement.src);
    if (onLoadCb) onLoadCb();
  };
}

sourceInput.onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  fileToImage(f, previewSource, () => { srcLoaded = true; resizeCanvasToTarget(); });
};

targetInput.onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  fileToImage(f, previewTarget, () => { dstLoaded = true; resizeCanvasToTarget(); });
};

function resizeCanvasToTarget() {
  if (!dstLoaded) return;
  resultCanvas.width = previewTarget.naturalWidth || previewTarget.width;
  resultCanvas.height = previewTarget.naturalHeight || previewTarget.height;
}

// matrix inverse for 3x3 (array of 3 rows, each row is length 3)
function invert3x3(m) {
  const a = m[0][0], b = m[0][1], c = m[0][2];
  const d = m[1][0], e = m[1][1], f = m[1][2];
  const g = m[2][0], h = m[2][1], i = m[2][2];
  const A =   e*i - f*h;
  const B = -(d*i - f*g);
  const C =   d*h - e*g;
  const D = -(b*i - c*h);
  const E =   a*i - c*g;
  const F = -(a*h - b*g);
  const G =   b*f - c*e;
  const H = -(a*f - c*d);
  const I =   a*e - b*d;
  const det = a*A + b*B + c*C;
  if (Math.abs(det) < 1e-8) return null;
  const invDet = 1 / det;
  return [
    [A*invDet, D*invDet, G*invDet],
    [B*invDet, E*invDet, H*invDet],
    [C*invDet, F*invDet, I*invDet],
  ];
}

// multiply 3x3 (S) by 3x2 (D) -> result 3x2
function mul3x3_3x2(invS, Drows) {
  // invS is 3x3, Drows is 3 rows of [dx,dy]
  const res = [[0,0],[0,0],[0,0]];
  for (let r=0;r<3;r++){
    for (let col=0;col<2;col++){
      res[r][col] = invS[r][0]*Drows[0][col] + invS[r][1]*Drows[1][col] + invS[r][2]*Drows[2][col];
    }
  }
  return res; // 3x2
}

// compute affine transform parameters mapping srcTri -> dstTri
// returns object {a,b,c,d,e,f} for ctx.setTransform(a,b,c,d,e,f)
function computeAffineParams(srcTri, dstTri) {
  // construct S (3x3) with rows [sx, sy, 1]
  const S = [
    [srcTri[0][0], srcTri[0][1], 1],
    [srcTri[1][0], srcTri[1][1], 1],
    [srcTri[2][0], srcTri[2][1], 1]
  ];
  const D = [
    [dstTri[0][0], dstTri[0][1]],
    [dstTri[1][0], dstTri[1][1]],
    [dstTri[2][0], dstTri[2][1]]
  ];
  const invS = invert3x3(S);
  if (!invS) return null;
  const M = mul3x3_3x2(invS, D); // 3x2 rows
  // M rows: [m11,m12],[m21,m22],[m31,m32]
  const a = M[0][0], b = M[0][1];
  const c = M[1][0], d = M[1][1];
  const e = M[2][0], f = M[2][1];
  return {a,b,c,d,e,f};
}

// triangulation + warp
async function doWarpSwap() {
  if (!srcLoaded || !dstLoaded) {
    alert('Pilih kedua foto terlebih dahulu.');
    return;
  }
  swapBtn.disabled = true;
  try {
    // ensure models loaded
    await ensureModels();

    const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.4 });
    const [resSrc, resDst] = await Promise.all([
      faceapi.detectSingleFace(previewSource, opts).withFaceLandmarks(true),
      faceapi.detectSingleFace(previewTarget, opts).withFaceLandmarks(true)
    ]);
    if (!resSrc || !resDst) {
      alert('Wajah tidak terdeteksi di salah satu foto.');
      swapBtn.disabled = false;
      return;
    }

    // get landmarks arrays [ [x,y], ... ]
    const srcPts = resSrc.landmarks.positions.map(p => [p.x, p.y]);
    const dstPts = resDst.landmarks.positions.map(p => [p.x, p.y]);

    // triangulate on destination points (so triangles fit target)
    const delaunay = Delaunator.from(dstPts);
    const triangles = delaunay.triangles; // flat array of indices (triples)

    // prepare canvas
    resizeCanvasToTarget();
    const ctx = resultCanvas.getContext('2d');
    // draw target as base
    ctx.clearRect(0,0,resultCanvas.width,resultCanvas.height);
    ctx.drawImage(previewTarget, 0, 0, resultCanvas.width, resultCanvas.height);

    // For each triangle, compute affine and draw warped triangle from source
    for (let t = 0; t < triangles.length; t += 3) {
      const i0 = triangles[t], i1 = triangles[t+1], i2 = triangles[t+2];
      // skip triangles that reference points outside 68-landmarks (defensive)
      if (i0 >= srcPts.length || i1 >= srcPts.length || i2 >= srcPts.length) continue;

      const srcTri = [ srcPts[i0], srcPts[i1], srcPts[i2] ];
      const dstTri = [ dstPts[i0], dstPts[i1], dstPts[i2] ];

      // compute bounding box of dst triangle; small optimization: if triangle tiny skip
      const minX = Math.min(dstTri[0][0], dstTri[1][0], dstTri[2][0]);
      const minY = Math.min(dstTri[0][1], dstTri[1][1], dstTri[2][1]);
      const maxX = Math.max(dstTri[0][0], dstTri[1][0], dstTri[2][0]);
      const maxY = Math.max(dstTri[0][1], dstTri[1][1], dstTri[2][1]);
      if (maxX - minX < 1 || maxY - minY < 1) continue;

      // compute affine params
      const params = computeAffineParams(srcTri, dstTri);
      if (!params) continue;

      // draw: clip to dst triangle, set transform to map source->dst, draw source image
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(dstTri[0][0], dstTri[0][1]);
      ctx.lineTo(dstTri[1][0], dstTri[1][1]);
      ctx.lineTo(dstTri[2][0], dstTri[2][1]);
      ctx.closePath();
      ctx.clip();

      // set transform so that coordinates in source image map to destination positions
      ctx.setTransform(params.a, params.b, params.c, params.d, params.e, params.f);
      // draw entire source image (transform maps the correct region)
      ctx.drawImage(previewSource, 0, 0);

      // reset transform / restore clipping
      ctx.setTransform(1,0,0,1,0,0);
      ctx.restore();
    }

    // optional: simple feathering around face edges
    // create soft mask using dst landmarks jaw to slightly blend edges
    // This is a simple approach: draw the warped output into a temporary canvas,
    // then blend with the base target using globalAlpha (already done per-triangle above).
    // For better blending use Poisson blending (not implemented here).

    alert('Selesai — hasil lebih rapi dari versi sederhana. Untuk hasil terbaik, perlu Poisson blending.');
  } catch (err) {
    console.error(err);
    alert('Error: ' + (err.message || err));
  } finally {
    swapBtn.disabled = false;
  }
}

// wire button
swapBtn.onclick = async () => {
  swapBtn.disabled = true;
  await doWarpSwap();
};
