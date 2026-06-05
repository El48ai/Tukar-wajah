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
  swapBtn: $("swapBtn"),
  resetBtn: $("resetBtn"),
  downloadBtn: $("downloadBtn"),
  canvas: $("resultCanvas"),
  resultEmpty: $("resultEmpty"),
};

// ===== UI =====
function setStatus(msg, state = "loading") {
  el.status.textContent = msg;
  el.statusDot.className = "status-dot " + state;
}

function setLoading(on, text = "Memproses…") {
  el.loadingBar.className = "loading-bar" + (on ? " show" : "");
  el.loadingText.textContent = text;
}

function showPreview(container, src) {
  const img = new Image();
  img.src = src;
  img.style.cssText = "max-width:100%;height:auto;display:block;border-radius:10px";
  container.innerHTML = "";
  container.appendChild(img);
}

// Convert file ke base64 data URL
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ===== State =====
let sourceB64 = null;
let targetB64 = null;

function maybeEnableSwap() {
  el.swapBtn.disabled = !(sourceB64 && targetB64);
}

// ===== Main swap via Replicate API =====
async function doSwap() {
  if (!sourceB64 || !targetB64) return;

  el.swapBtn.disabled = true;
  el.downloadBtn.style.display = "none";
  el.canvas.style.display = "none";
  el.resultEmpty.style.display = "block";

  try {
    setLoading(true, "Mengirim ke AI Replicate…");
    setStatus("Memproses face swap dengan AI…", "loading");

    const res = await fetch("/api/swap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_image: sourceB64,
        target_image: targetB64,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Gagal memproses");
    }

    // Output bisa string URL atau array
    const outputUrl = Array.isArray(data.output) ? data.output[0] : data.output;
    if (!outputUrl) throw new Error("Tidak ada output dari API");

    setLoading(true, "Memuat hasil…");

    // Load hasil ke canvas
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      el.canvas.width = img.width;
      el.canvas.height = img.height;
      const ctx = el.canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      el.resultEmpty.style.display = "none";
      el.canvas.style.display = "block";
      el.downloadBtn.style.display = "inline-block";
      setStatus("Selesai ✅  Klik Download untuk menyimpan.", "ready");
      setLoading(false);
      maybeEnableSwap();
    };
    img.onerror = () => {
      // Kalau canvas gagal (CORS), tampilkan sebagai img biasa
      el.resultEmpty.innerHTML = `<img src="${outputUrl}" style="max-width:100%;border-radius:12px">`;
      el.resultEmpty.style.display = "block";
      // Download langsung dari URL
      el.downloadBtn.onclick = () => {
        const a = document.createElement("a");
        a.href = outputUrl;
        a.download = "tukar-wajah-hasil.png";
        a.target = "_blank";
        a.click();
      };
      el.downloadBtn.style.display = "inline-block";
      setStatus("Selesai ✅  Klik Download untuk menyimpan.", "ready");
      setLoading(false);
      maybeEnableSwap();
    };
    img.src = outputUrl;

  } catch (err) {
    console.error(err);
    setStatus("Error ❌  " + err.message, "error");
    setLoading(false);
    maybeEnableSwap();
  }
}

// ===== Events =====
el.sourceInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0]; if (!file) return;
  sourceB64 = await fileToBase64(file);
  showPreview(el.sourcePreview, sourceB64);
  setStatus("Foto sumber siap ✅", "ready");
  maybeEnableSwap();
});

el.targetInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0]; if (!file) return;
  targetB64 = await fileToBase64(file);
  showPreview(el.targetPreview, targetB64);
  setStatus("Foto target siap ✅", "ready");
  maybeEnableSwap();
});

el.swapBtn.addEventListener("click", doSwap);

el.resetBtn.addEventListener("click", () => {
  sourceB64 = null; targetB64 = null;
  el.sourceInput.value = ""; el.targetInput.value = "";
  el.sourcePreview.innerHTML = `<div class="preview-empty"><div class="icon">🖼️</div><div>Belum ada foto</div></div>`;
  el.targetPreview.innerHTML = `<div class="preview-empty"><div class="icon">🖼️</div><div>Belum ada foto</div></div>`;
  el.canvas.style.display = "none";
  el.resultEmpty.style.display = "block";
  el.resultEmpty.innerHTML = "Hasil face swap akan muncul di sini";
  el.downloadBtn.style.display = "none";
  setStatus("Reset ✅  Pilih 2 foto lagi.", "ready");
  maybeEnableSwap();
});

el.downloadBtn.addEventListener("click", () => {
  if (el.canvas.style.display !== "none") {
    const a = document.createElement("a");
    a.download = "tukar-wajah-hasil.png";
    a.href = el.canvas.toDataURL("image/png");
    a.click();
  }
});

// Ready
setStatus("Siap ✅  Pilih 2 foto lalu klik Tukar Wajah.", "ready");
