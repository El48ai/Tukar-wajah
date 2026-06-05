export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { source_image, target_image } = req.body;
  if (!source_image || !target_image) {
    return res.status(400).json({ error: 'source_image dan target_image wajib diisi' });
  }

  const REPLICATE_TOKEN = process.env.REPLICATE_API_TOKEN;
  if (!REPLICATE_TOKEN) {
    return res.status(500).json({ error: 'API token tidak ditemukan' });
  }

  try {
    // Pakai model codeplugtech/face-swap - murah ($0.0024/run) dan cepat
    const createRes = await fetch('https://api.replicate.com/v1/models/codeplugtech/face-swap/predictions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${REPLICATE_TOKEN}`,
        'Content-Type': 'application/json',
        'Prefer': 'wait=60',
      },
      body: JSON.stringify({
        input: {
          input_image: target_image,  // foto yang akan diganti wajahnya
          swap_image: source_image,   // foto sumber wajah
        }
      })
    });

    const prediction = await createRes.json();

    if (!createRes.ok) {
      return res.status(createRes.status).json({ error: prediction.detail || 'Replicate error' });
    }

    if (prediction.status === 'succeeded') {
      return res.status(200).json({ output: prediction.output });
    }

    // Poll sampai selesai
    const predId = prediction.id;
    for (let i = 0; i < 30; i++) {
      await new Promise(r => setTimeout(r, 2000));
      const pollRes = await fetch(`https://api.replicate.com/v1/predictions/${predId}`, {
        headers: { 'Authorization': `Bearer ${REPLICATE_TOKEN}` }
      });
      const poll = await pollRes.json();
      if (poll.status === 'succeeded') {
        return res.status(200).json({ output: poll.output });
      }
      if (poll.status === 'failed') {
        return res.status(500).json({ error: poll.error || 'Prediksi gagal' });
      }
    }

    return res.status(504).json({ error: 'Timeout — coba lagi' });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
