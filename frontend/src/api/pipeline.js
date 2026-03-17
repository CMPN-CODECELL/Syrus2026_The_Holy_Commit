const BASE = 'http://localhost:8000'

export async function uploadImage(file, jewelryType = 'auto') {
  const form = new FormData()
  form.append('image', file)
  form.append('jewelry_type', jewelryType)
  const res = await fetch(`${BASE}/api/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
  return res.json()
}

export async function chatWithAgent(jobId, text) {
  const res = await fetch(`${BASE}/api/chat/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`)
  return res.json()
}

export async function swapMaterial(jobId, component, material) {
  const res = await fetch(`${BASE}/api/swap/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ component, material }),
  })
  if (!res.ok) throw new Error(`Swap failed: ${res.status}`)
  return res.json()
}

export async function getMaterials() {
  const res = await fetch(`${BASE}/api/materials`)
  if (!res.ok) throw new Error(`Failed to fetch materials: ${res.status}`)
  return res.json()
}

export async function getFile(jobId, filename) {
  const res = await fetch(`${BASE}/api/files/${jobId}/${filename}`)
  if (!res.ok) throw new Error(`File not found: ${res.status}`)
  return res.blob()
}
