const { initializeApp, getApps, cert } = require('firebase-admin/app');
const { getFirestore, FieldValue }     = require('firebase-admin/firestore');

if (!getApps().length) {
  initializeApp({
    credential: cert({
      projectId:   process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey:  process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
    }),
  });
}

const db = getFirestore();

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { code, checkOnly, consumeSwap } = req.body;
  if (!code) return res.status(400).json({ valid: false, error: 'Kode wajib diisi' });

  try {
    const docRef = db.collection('license_codes').doc(code.trim().toUpperCase());
    const doc    = await docRef.get();

    if (!doc.exists) {
      return res.status(200).json({ valid: false, error: 'Kode tidak ditemukan' });
    }

    const data = doc.data();

    if (data.expires_at && data.expires_at.toDate() < new Date()) {
      return res.status(200).json({ valid: false, error: 'Kode sudah kadaluarsa' });
    }

    const swapsRemaining = data.swaps_remaining ?? null;

    // Kalau paket swap, cek sisa swap
    if (swapsRemaining !== null && swapsRemaining <= 0) {
      return res.status(200).json({ valid: false, error: 'Swap habis! Hubungi kami untuk top-up.' });
    }

    // consumeSwap: kurangi 1 swap setelah berhasil
    if (consumeSwap && swapsRemaining !== null) {
      await docRef.update({
        swaps_remaining: FieldValue.increment(-1),
        last_used_at:    FieldValue.serverTimestamp(),
      });
      return res.status(200).json({
        valid:           true,
        plan:            data.plan || 'pro',
        swaps_remaining: swapsRemaining - 1,
      });
    }

    // checkOnly: verifikasi tanpa kurangi
    if (!checkOnly) {
      await docRef.update({
        used_count:   FieldValue.increment(1),
        last_used_at: FieldValue.serverTimestamp(),
      });
    }

    return res.status(200).json({
      valid:           true,
      plan:            data.plan || 'pro',
      swaps_remaining: swapsRemaining,
      message:         'Kode valid! Akses Pro diaktifkan.',
    });

  } catch (err) {
    console.error(err);
    return res.status(500).json({ valid: false, error: 'Server error: ' + err.message });
  }
};
