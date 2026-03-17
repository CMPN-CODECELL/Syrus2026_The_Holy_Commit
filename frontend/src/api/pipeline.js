export async function uploadImage(file, jewelryType = 'auto') {
  const form = new FormData()
  form.append('image', file)
  form.append('jewelry_type', jewelryType)
  const res = await fetch('/api/upload', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
  return res.json()
}

export async function chatWithAgent(jobId, text) {
  const res = await fetch(`/api/chat/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  const payload = await res.json().catch(() => null)
  if (!res.ok) {
    throw new Error(payload?.error || payload?.message || `Chat failed: ${res.status}`)
  }
  return payload
}

export async function swapMaterial(jobId, component, material) {
  const res = await fetch(`/api/swap/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ component, material }),
  })
  if (!res.ok) throw new Error(`Swap failed: ${res.status}`)
  return res.json()
}

export async function getMaterials() {
  const res = await fetch('/api/materials')
  if (!res.ok) throw new Error(`Failed to fetch materials: ${res.status}`)
  return res.json()
}

export async function getFile(jobId, filename) {
  const res = await fetch(`/api/files/${jobId}/${filename}`)
  if (!res.ok) throw new Error(`File not found: ${res.status}`)
  return res.blob()
}
