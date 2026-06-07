# Tukar Wajah AI

Aplikasi web face swap bertenaga AI. 5 foto gratis, upgrade ke Pro Rp15.000/bulan.

🌐 **Live**: https://tukar-wajah.vercel.app

## Struktur Folder

```
tukar-wajah/
├── api/
│   ├── swap.js            # Serverless: face swap via Replicate API
│   ├── validate-code.js   # Serverless: validasi kode lisensi Pro (Firestore)
│   └── check-quota.js     # Serverless: tracking kuota gratis server-side
├── public/
│   ├── index.html         # Halaman utama
│   ├── script.js          # Logic frontend (kuota server-side, UI)
│   └── style.css          # (kosong, CSS inline di index.html)
├── vercel.json            # Konfigurasi routing Vercel
└── package.json
```

## Environment Variables (Vercel)

```
REPLICATE_API_TOKEN=r8_xxxx
FIREBASE_PROJECT_ID=nama-project
FIREBASE_CLIENT_EMAIL=xxx@xxx.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n..."
```

## Firestore Collections

- `license_codes/{CODE}` — kode lisensi Pro
  - `plan`: "pro"
  - `expires_at`: Timestamp (opsional)
  - `max_uses`: number (opsional)
  - `used_count`: number
  - `last_used_at`: Timestamp

- `free_quota/{fingerprint}` — tracking kuota gratis per user
  - `used`: number
  - `last_used_at`: Timestamp

## Sistem Keamanan

- Kuota gratis di-track di server (Firestore) via fingerprint IP+UA
- Validasi lisensi Pro selalu lewat server, bukan cuma localStorage
- `FieldValue.increment` untuk mencegah race condition
