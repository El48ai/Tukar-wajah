// ═══════════════════════════════════════════
//  script.js — Tukar Wajah AI
//  Fix: kuota server-side (anti-bypass), validasi Pro lewat server
// ═══════════════════════════════════════════

const $ = (id) => document.getElementById(id);

const el = {
  sourceInput:   $('sourceImage'),
  targetInput:   $('targetImage'),
  sourcePreview: $('sourcePreview'),
  targetPreview: $('targetPreview'),
  statusDot:     $('statusDot'),
  status:        $('statusMessage'),
  loadingBar:    $('loading'),
  loadingText:   $('loadingText'),
  swapBtn:       $('swapBtn'),
  resetBtn:      $('resetBtn'),
  downloadBtn:   $('downloadBtn'),
  canvas:        $('resultCanvas'),
  resultEmpty:   $('resultEmpty'),
  quotaBadge:    $('quotaBadge'),
};

// ═══════════════════════════════════════════
//  KUOTA — server-side tracking
// ═══════════════════════════════════════════
const LICENSE_KEY = 'tw_license';
const FREE_LIMIT  = 5;

// State kuota (diambil dari server saat init)
let quotaState = { canSwap: true, used: 0, remaining: FREE_LIMIT, isPro: false };

const license = {
  get:  ()    => localStorage.getItem(LICENSE_KEY) || '',
  save: (code)=> localStorage.setItem(LICENSE_KEY, code),
  clear:()    => localStorage.removeItem(LICENSE_KEY),
  has:  ()    => !!localStorage.getItem(LICENSE_KEY),
};

// Cek kuota dari server (bukan localStorage)
async function fetchQuota() {
  // Kalau ada lisensi di localStorage, verifikasi ulang ke server
  if (license.has()) {
    try {
      const res  = await fetch('/api/validate-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: license.get(), checkOnly: true }),
      });
      const data = await res.json();
      if (data.valid) {
        quotaState = { canSwap: true, used: 0, remaining: 999, isPro: true };
        renderQuotaBadge();
        return;
      } else {
        // Lisensi tidak valid lagi (expired/habis), hapus
        license.clear();
      }
    } catch (_) {}
  }

  // Cek kuota gratis dari server
  try {
    const res  = await fetch('/api/check-quota', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'check' }),
    });
    const data = await res.json();
    quotaState = { ...data, isPro: false };
  } catch (_) {
    // Fallback ke localStorage kalau server error
    const used = parseInt(localStorage.getItem('tw_free_used') || '0', 10);
    quotaState = {
      canSwap:   used < FREE_LIMIT,
      used,
      remaining: Math.max(0, FREE_LIMIT - used),
      isPro:     false,
    };
  }
  renderQuotaBadge();
}

// Consume kuota di server setelah swap sukses
async function consumeQuota() {
  if (quotaState.isPro) return true;
  try {
    const res  = await fetch('/api/check-quota', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'consume' }),
    });
    const data = await res.json();
    if (data.success) {
      quotaState.used      = data.used;
      quotaState.remaining = data.remaining;
      quotaState.canSwap   = data.remaining > 0;
      renderQuotaBadge();
      // Sync localStorage sebagai fallback
      localStorage.setItem('tw_free_used', data.used);
      return true;
    }
    return false;
  } catch (_) {
    // Fallback
    const used = parseInt(localStorage.getItem('tw_free_used') || '0', 10) + 1;
    localStorage.setItem('tw_free_used', used);
    quotaState.used      = used;
    quotaState.remaining = Math.max(0, FREE_LIMIT - used);
    quotaState.canSwap   = used < FREE_LIMIT;
    renderQuotaBadge();
    return true;
  }
}

function renderQuotaBadge() {
  if (!el.quotaBadge) return;
  if (quotaState.isPro) {
    el.quotaBadge.textContent = '🔓 Pro — Unlimited';
    el.quotaBadge.style.color       = '#10b981';
    el.quotaBadge.style.borderColor = 'rgba(16,185,129,.4)';
  } else {
    const r = quotaState.remaining;
    el.quotaBadge.textContent = r > 0
      ? `✨ ${r} foto gratis tersisa`
      : '🔒 Kuota habis — Upgrade Pro';
    el.quotaBadge.style.color       = r > 0 ? '#a78bfa' : '#ef4444';
    el.quotaBadge.style.borderColor = r > 0
      ? 'rgba(167,139,250,.4)'
      : 'rgba(239,68,68,.4)';
  }
}

// ── Modal paywall ─────────────────────────
window.showPaywall = function () {
  $('paywallModal').style.display = 'flex';
};
window.hidePaywall = function () {
  $('paywallModal').style.display = 'none';
  $('codeInput').value = '';
  $('codeError').textContent = '';
};

window.validateCode = async function (code) {
  code = (code || '').trim().toUpperCase();
  if (!code) { $('codeError').textContent = 'Masukkan kode dulu.'; return; }

  const btn   = $('activateBtn');
  const errEl = $('codeError');
  errEl.textContent = '';
  btn.disabled    = true;
  btn.textContent = 'Memeriksa…';

  try {
    const res  = await fetch('/api/validate-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    });
    const data = await res.json();

    if (data.valid) {
      license.save(code);
      quotaState = { canSwap: true, used: 0, remaining: 999, isPro: true };
      hidePaywall();
      renderQuotaBadge();
      maybeEnableSwap();
      showToast('✅ Kode valid! Akses Pro diaktifkan.');
    } else {
      errEl.textContent = data.error || 'Kode tidak valid.';
    }
  } catch {
    errEl.textContent = 'Gagal terhubung ke server.';
  } finally {
    btn.disabled    = false;
    btn.textContent = '🔑 Aktifkan';
  }
};

// ── Toast ─────────────────────────────────
function showToast(msg) {
  let t = $('_toast');
  if (!t) {
    t = document.createElement('div');
    t.id = '_toast';
    t.style.cssText = [
      'position:fixed;bottom:24px;left:50%;transform:translateX(-50%)',
      'background:#10b981;color:#fff;padding:12px 20px;border-radius:14px',
      'font-size:14px;font-weight:700;z-index:9999',
      'box-shadow:0 4px 20px rgba(16,185,129,.4)',
      'transition:opacity .3s',
    ].join(';');
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.style.opacity = '1';
  clearTimeout(t._t);
  t._t = setTimeout(() => { t.style.opacity = '0'; }, 3000);
}

// ═══════════════════════════════════════════
//  UI HELPERS
// ═══════════════════════════════════════════
function setStatus(msg, state = 'loading') {
  el.status.textContent  = msg;
  el.statusDot.className = 'status-dot ' + state;
}

function setLoading(on, text = 'Memproses…') {
  el.loadingBar.className = 'loading-bar' + (on ? ' show' : '');
  el.loadingText.textContent = text;
}

function showPreview(container, src) {
  const img = new Image();
  img.src   = src;
  img.style.cssText = 'max-width:100%;height:auto;display:block;border-radius:10px';
  container.innerHTML = '';
  container.appendChild(img);
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ═══════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════
let sourceB64 = null;
let targetB64 = null;

function maybeEnableSwap() {
  el.swapBtn.disabled = !(sourceB64 && targetB64);
}

// ═══════════════════════════════════════════
//  MAIN SWAP
// ═══════════════════════════════════════════
async function doSwap() {
  if (!sourceB64 || !targetB64) return;

  // Cek kuota dari state (sudah diambil dari server saat init)
  if (!quotaState.canSwap && !quotaState.isPro) {
    showPaywall();
    return;
  }

  // Verifikasi sekali lagi ke server sebelum proses (anti-bypass)
  if (!quotaState.isPro) {
    try {
      const res  = await fetch('/api/check-quota', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'check' }),
      });
      const data = await res.json();
      if (!data.canSwap) {
        quotaState = { ...data, isPro: false };
        renderQuotaBadge();
        showPaywall();
        return;
      }
    } catch (_) { /* lanjut kalau server error */ }
  }

  el.swapBtn.disabled          = true;
  el.downloadBtn.style.display = 'none';
  el.canvas.style.display      = 'none';
  el.resultEmpty.style.display = 'block';

  try {
    setLoading(true, 'Mengirim ke AI Replicate…');
    setStatus('Memproses face swap dengan AI…', 'loading');

    const res = await fetch('/api/swap', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source_image: sourceB64,
        target_image: targetB64,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Gagal memproses');

    const outputUrl = Array.isArray(data.output) ? data.output[0] : data.output;
    if (!outputUrl) throw new Error('Tidak ada output dari API');

    setLoading(true, 'Memuat hasil…');

    const onSuccess = async () => {
      setStatus('Selesai ✅  Klik Download untuk menyimpan.', 'ready');
      setLoading(false);
      // Consume kuota di server SETELAH sukses
      if (!quotaState.isPro) await consumeQuota();
      maybeEnableSwap();
    };

    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = async () => {
      el.canvas.width  = img.width;
      el.canvas.height = img.height;
      el.canvas.getContext('2d').drawImage(img, 0, 0);
      el.resultEmpty.style.display = 'none';
      el.canvas.style.display      = 'block';
      el.downloadBtn.style.display = 'inline-block';
      await onSuccess();
    };

    img.onerror = async () => {
      el.resultEmpty.innerHTML     = `<img src="${outputUrl}" style="max-width:100%;border-radius:12px">`;
      el.resultEmpty.style.display = 'block';
      el.downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = outputUrl; a.download = 'tukar-wajah-hasil.png';
        a.target = '_blank'; a.click();
      };
      el.downloadBtn.style.display = 'inline-block';
      await onSuccess();
    };

    img.src = outputUrl;

  } catch (err) {
    console.error(err);
    setStatus('Error ❌  ' + err.message, 'error');
    setLoading(false);
    maybeEnableSwap();
  }
}

// ═══════════════════════════════════════════
//  EVENTS
// ═══════════════════════════════════════════
el.sourceInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0]; if (!file) return;
  sourceB64 = await fileToBase64(file);
  showPreview(el.sourcePreview, sourceB64);
  setStatus('Foto sumber siap ✅', 'ready');
  maybeEnableSwap();
});

el.targetInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0]; if (!file) return;
  targetB64 = await fileToBase64(file);
  showPreview(el.targetPreview, targetB64);
  setStatus('Foto target siap ✅', 'ready');
  maybeEnableSwap();
});

el.swapBtn.addEventListener('click', doSwap);

el.resetBtn.addEventListener('click', () => {
  sourceB64 = null; targetB64 = null;
  el.sourceInput.value = ''; el.targetInput.value = '';
  el.sourcePreview.innerHTML = `<div class="preview-empty"><div class="icon">🖼️</div><div>Belum ada foto</div></div>`;
  el.targetPreview.innerHTML = `<div class="preview-empty"><div class="icon">🖼️</div><div>Belum ada foto</div></div>`;
  el.canvas.style.display      = 'none';
  el.resultEmpty.style.display = 'block';
  el.resultEmpty.innerHTML     = 'Hasil face swap akan muncul di sini';
  el.downloadBtn.style.display = 'none';
  setStatus('Reset ✅  Pilih 2 foto lagi.', 'ready');
  maybeEnableSwap();
});

el.downloadBtn.addEventListener('click', () => {
  if (el.canvas.style.display !== 'none') {
    const a = document.createElement('a');
    a.download = 'tukar-wajah-hasil.png';
    a.href     = el.canvas.toDataURL('image/png');
    a.click();
  }
});

// ── Init: ambil kuota dari server ─────────
(async () => {
  setStatus('Siap ✅  Pilih 2 foto lalu klik Tukar Wajah.', 'ready');
  await fetchQuota();
})();
