export default function ExportButton({ meshUrl, jobId }) {
  if (!meshUrl) return null

  const handleDownload = () => {
    const a = document.createElement('a')
    a.href = meshUrl
    a.download = `jewelforge_${jobId}.glb`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <button
      onClick={handleDownload}
      style={{
        padding: '8px 16px',
        background: 'rgba(0,0,0,0.7)',
        color: '#f0f0f0',
        border: '1px solid #444',
        borderRadius: 8,
        cursor: 'pointer',
        fontSize: 13,
        backdropFilter: 'blur(4px)',
      }}
    >
      Export GLB
    </button>
  )
}
