/**
 * JewelForge v2 — API client for the FastAPI backend.
 */

const BASE_URL = ''  // Vite proxy rewrites /api → http://localhost:8000

/**
 * Upload a jewelry image and kick off the pipeline.
 * @param {File} file
 * @returns {Promise<object>}  { job_id, mesh_url, labels_url, components, price, agent_greeting }
 */
export async function uploadImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`Upload failed (${response.status}): ${detail}`)
  }

  const data = await response.json()

  // Normalise keys to camelCase for the frontend
  return {
    jobId: data.job_id,
    meshUrl: data.mesh_url,
    labelsUrl: data.labels_url,
    components: data.components ?? [],
    price: data.price ?? null,
    agentGreeting: data.agent_greeting ?? '',
  }
}

/**
 * Send a chat message to the agent for an existing job.
 * @param {string} jobId
 * @param {string} text
 * @returns {Promise<object>}  { response_text, actions, price, materials_applied }
 */
export async function sendChatMessage(jobId, text) {
  const response = await fetch(`${BASE_URL}/api/chat/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`Chat failed (${response.status}): ${detail}`)
  }

  return response.json()
}

/**
 * Fetch all PBR material presets from the backend.
 * @returns {Promise<{ metals: object, gemstones: object }>}
 */
export async function getMaterials() {
  const response = await fetch(`${BASE_URL}/api/materials`)
  if (!response.ok) throw new Error(`getMaterials failed: ${response.status}`)
  return response.json()
}

/**
 * Build the URL for a generated file served by the backend.
 * @param {string} jobId
 * @param {string} filename
 * @returns {string}
 */
export function getFileUrl(jobId, filename) {
  return `${BASE_URL}/api/files/${jobId}/${filename}`
}
