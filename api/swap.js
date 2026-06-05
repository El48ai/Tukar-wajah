module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { source_image, target_image } = req.body;
  if (!source_image || !target_image) {
    return res.status(400).json({ error: 'Foto wajib diisi' });
  }

  const TOKEN = process.env.REPLICATE_API_TOKEN;
  if (!TOKEN) return res.status(500).json({ error: 'Token tidak ada' });

  try {
    const createRes = await fetch('https://api.replicate.com/v1/predictions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${TOKEN}`,
        'Content-Type': 'application/json',
        'Prefer': 'wait=60',
      },
      body: JSON.stringify({
        version: 'cc36d3f27e37ee6a83a90d0b7e7d71f43a9d2e1b72a0d1c5e2e1b5e9e9b0b0b',
        input: {
          target_image: target_image,
          source_image: source_image,
        }
      })
    });

    const prediction = await createRes.json();
    if (!createRes.ok) return res.status(400).json({ error: JSON.stringify(prediction) });
    if (prediction.status === 'succeeded') return res.status(200).json({ output: prediction.output });

    const id = prediction.id;
    for (let i = 0; i < 30; i++) {
      await new Promise(r => setTimeout(r, 2000));
      const p = await (await fetch(`https://api.replicate.com/v1/predictions/${id}`, {
        headers: { 'Authorization': `Bearer ${TOKEN}` }
      })).json();
      if (p.status === 'succeeded') return res.status(200).json({ output: p.output });
      if (p.status === 'failed') return res.status(500).json({ error: p.error });
    }
    return res.status(504).json({ error: 'Timeout' });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
