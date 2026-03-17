import React, { useState, useCallback, useRef } from 'react'
import { useAgent } from '../hooks/useAgent.js'

const STAGES = [
  'Removing background…',
  'Segmenting components…',
  'Generating 3D mesh…',
  'Baking texture…',
  'Estimating price…',
  'Done!',
]

export default function UploadDropzone({ onUploadComplete }) {
  const [isDragging, setIsDragging] = useState(false)
  const [thumbnail, setThumbnail] = useState(null)
  const [stageIndex, setStageIndex] = useState(-1)
  const inputRef = useRef(null)
  const { uploadImage, isLoading } = useAgent(null)

  const processFile = useCallback(
    async (file) => {
      if (!file || !file.type.startsWith('image/')) return

      // Show thumbnail immediately
      const objectUrl = URL.createObjectURL(file)
      setThumbnail(objectUrl)

      // Animate through stages
      setStageIndex(0)
      const stageTimer = setInterval(() => {
        setStageIndex((i) => {
          const next = i + 1
          if (next >= STAGES.length - 1) clearInterval(stageTimer)
          return next
        })
      }, 1200)

      const result = await uploadImage(file)
      clearInterval(stageTimer)
      setStageIndex(STAGES.length - 1)

      if (result) {
        onUploadComplete(result)
      }

      setTimeout(() => setStageIndex(-1), 1500)
    },
    [uploadImage, onUploadComplete]
  )

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files?.[0]
      processFile(file)
    },
    [processFile]
  )

  const handleChange = (e) => processFile(e.target.files?.[0])

  return (
    <div
      style={{
        ...styles.zone,
        borderColor: isDragging ? 'var(--primary)' : 'var(--border)',
        background: isDragging ? 'rgba(124,106,247,0.08)' : 'var(--bg-panel)',
      }}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => !isLoading && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        style={{ display: 'none' }}
        onChange={handleChange}
      />

      {thumbnail ? (
        <img src={thumbnail} alt="Preview" style={styles.thumbnail} />
      ) : (
        <div style={styles.icon}>🖼</div>
      )}

      {stageIndex >= 0 ? (
        <div style={styles.stage}>{STAGES[stageIndex]}</div>
      ) : (
        <div style={styles.label}>
          {isLoading ? 'Processing…' : 'Drop image or click to upload'}
        </div>
      )}

      {stageIndex >= 0 && stageIndex < STAGES.length - 1 && (
        <div style={styles.progressBar}>
          <div
            style={{
              ...styles.progressFill,
              width: `${((stageIndex + 1) / STAGES.length) * 100}%`,
            }}
          />
        </div>
      )}
    </div>
  )
}

const styles = {
  zone: {
    border: '2px dashed',
    borderRadius: 10,
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 8,
    cursor: 'pointer',
    transition: 'border-color 0.2s, background 0.2s',
    userSelect: 'none',
    minHeight: 110,
    justifyContent: 'center',
  },
  icon: {
    fontSize: 28,
    opacity: 0.5,
  },
  label: {
    fontSize: 12,
    color: 'var(--text-secondary)',
    textAlign: 'center',
  },
  stage: {
    fontSize: 12,
    color: 'var(--accent)',
    textAlign: 'center',
    fontWeight: 500,
  },
  thumbnail: {
    width: 72,
    height: 72,
    objectFit: 'cover',
    borderRadius: 8,
    border: '2px solid var(--border)',
  },
  progressBar: {
    width: '100%',
    height: 3,
    background: 'rgba(255,255,255,0.1)',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    background: 'var(--primary)',
    borderRadius: 2,
    transition: 'width 0.5s ease',
  },
}
