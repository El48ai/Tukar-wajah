const { initializeApp, getApps, cert } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');

// Init Firebase Admin (hanya sekali)
if (!getApps().length) {
  initializeApp({
    credential: cert({
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
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

  const { code } = req.body;
  if (!code) return res.status(400).json({ valid: false, error: 'Kode wajib diisi' });

  try {
    const docRef = db.collection('license_codes').doc(code.trim().toUpperCase());
    const doc = await docRef.get();

    if (!doc.exists) {
      return res.status(200).json({ valid: false, error: 'Kode tidak ditemukan' });
    }

    const data = doc.data();

    // Cek apakah kode sudah expired
    if (data.expires_at && data.expires_at.toDate() < new Date()) {
      return res.status(200).json({ valid: false, error: 'Kode sudah kadaluarsa' });
    }

    // Cek apakah kode sudah dipakai (opsional: max_uses)
    if (data.max_uses && data.used_count >= data.max_uses) {
      return res.status(200).json({ valid: false, error: 'Kode sudah mencapai batas penggunaan' });
    }

    // Update used_count
    await docRef.update({
      used_count: (data.used_count || 0) + 1,
      last_used_at: new Date(),
    });

    return res.status(200).json({
      valid: true,
      plan: data.plan || 'pro',
      message: 'Kode valid! Akses Pro diaktifkan.'
    });

  } catch (err) {
    console.error(err);
    return res.status(500).json({ valid: false, error: 'Server error: ' + err.message });
  }
};
