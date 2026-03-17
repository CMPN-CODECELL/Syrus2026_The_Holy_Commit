import { useRef } from 'react'
import { uploadImage } from '../api/pipeline'

export default function UploadDropzone({ onUploadComplete, isLoading, setIsLoading }) {
  const inputRef = useRef(null)

  const handleFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) return
    setIsLoading(true)
    try {
      const result = await uploadImage(file)
      onUploadComplete(result)
    } catch (err) {
      console.error('Upload failed:', err)
      alert('Upload failed: ' + (err.message || 'Unknown error'))
    } finally {
      setIsLoading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  const handleDragOver = (e) => e.preventDefault()

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={() => !isLoading && inputRef.current?.click()}
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: isLoading ? 'wait' : 'pointer',
        border: '2px dashed #333',
        borderRadius: 12,
        margin: 24,
        width: 'calc(100% - 48px)',
        height: 'calc(100% - 48px)',
        transition: 'border-color 0.2s',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={e => handleFile(e.target.files[0])}
      />
      {isLoading ? (
        <>
          <div style={{ fontSize: 40, marginBottom: 16 }}>⚙️</div>
          <div style={{ color: '#888', fontSize: 15 }}>Processing pipeline...</div>
          <div style={{ color: '#555', fontSize: 12, marginTop: 8 }}>
            Segmenting → Generating 3D → Baking texture
          </div>
        </>
      ) : (
        <>
          <div style={{ fontSize: 48, marginBottom: 16 }}>💎</div>
          <div style={{ color: '#ccc', fontSize: 16, marginBottom: 8 }}>
            Drop a jewelry image here
          </div>
          <div style={{ color: '#666', fontSize: 13 }}>
            or click to browse — PNG, JPG, WEBP
          </div>
        </>
      )}
    </div>
  )
}
