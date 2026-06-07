// ═══════════════════════════════════════════
//  api/check-quota.js
//  Tracking kuota gratis di server — anti-bypass localStorage
// ═══════════════════════════════════════════
const { initializeApp, getApps, cert } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const crypto = require('crypto');

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
const FREE_LIMIT = 5;

// Buat fingerprint dari IP + User-Agent (tidak menyimpan data pribadi)
function getFingerprint(req) {
  const ip = req.headers['x-forwarded-for']?.split(',')[0]?.trim()
    || req.headers['x-real-ip']
    || req.socket?.remoteAddress
    || 'unknown';
  const ua = req.headers['user-agent'] || 'unknown';
  return crypto.createHash('sha256').update(ip + ua).digest('hex').slice(0, 32);
}

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { action } = req.body; // 'check' atau 'consume'
  const fp = getFingerprint(req);

  try {
    const docRef = db.collection('free_quota').doc(fp);
    const doc    = await docRef.get();
    const used   = doc.exists ? (doc.data().used || 0) : 0;

    if (action === 'check') {
      return res.status(200).json({
        canSwap:   used < FREE_LIMIT,
        used,
        remaining: Math.max(0, FREE_LIMIT - used),
        limit:     FREE_LIMIT,
      });
    }

    if (action === 'consume') {
      if (used >= FREE_LIMIT) {
        return res.status(200).json({ success: false, error: 'Kuota habis' });
      }
      // Atomic increment — anti race condition
      await docRef.set({
        used:        FieldValue.increment(1),
        last_used_at: FieldValue.serverTimestamp(),
      }, { merge: true });
      const newUsed = used + 1;
      return res.status(200).json({
        success:   true,
        used:      newUsed,
        remaining: Math.max(0, FREE_LIMIT - newUsed),
      });
    }

    return res.status(400).json({ error: 'action tidak valid' });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error: ' + err.message });
  }
};
