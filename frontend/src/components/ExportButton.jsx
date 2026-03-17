import React, { useState, useRef } from 'react'

export default function ExportButton({ scene, meshUrl }) {
  const [open, setOpen] = useState(false)
  const [exporting, setExporting] = useState(false)
  const ref = useRef(null)

  const isDisabled = !scene && !meshUrl

  const exportGLB = async () => {
    if (!scene) return
    setExporting(true)
    try {
      const { GLTFExporter } = await import('three/examples/jsm/exporters/GLTFExporter.js')
      const exporter = new GLTFExporter()
      exporter.parse(
        scene,
        (result) => {
          const blob = new Blob([result], { type: 'model/gltf-binary' })
          downloadBlob(blob, 'jewelforge_model.glb')
          setExporting(false)
        },
        (err) => { console.error(err); setExporting(false) },
        { binary: true }
      )
    } catch (e) {
      console.error(e)
      setExporting(false)
    }
  }

  const exportSTL = async () => {
    if (!scene) return
    setExporting(true)
    try {
      const { STLExporter } = await import('three/examples/jsm/exporters/STLExporter.js')
      const exporter = new STLExporter()
      const result = exporter.parse(scene, { binary: true })
      const blob = new Blob([result], { type: 'model/stl' })
      downloadBlob(blob, 'jewelforge_model.stl')
    } catch (e) {
      console.error(e)
    } finally {
      setExporting(false)
    }
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div ref={ref} style={styles.wrapper}>
      <button
        style={{
          ...styles.btn,
          opacity: isDisabled ? 0.4 : 1,
          cursor: isDisabled ? 'not-allowed' : 'pointer',
        }}
        disabled={isDisabled}
        onClick={() => setOpen((v) => !v)}
      >
        {exporting ? 'Exporting…' : 'Export Model ▾'}
      </button>

      {open && !isDisabled && (
        <div style={styles.dropdown}>
          <button style={styles.option} onClick={() => { exportGLB(); setOpen(false) }}>
            Download GLB
          </button>
          <button style={styles.option} onClick={() => { exportSTL(); setOpen(false) }}>
            Download STL
          </button>
        </div>
      )}
    </div>
  )
}

const styles = {
  wrapper: {
    position: 'relative',
  },
  btn: {
    width: '100%',
    padding: '9px 14px',
    background: 'var(--secondary)',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    fontSize: 13,
    fontWeight: 500,
    transition: 'opacity 0.15s',
  },
  dropdown: {
    position: 'absolute',
    bottom: '110%',
    left: 0,
    right: 0,
    background: 'var(--bg-panel)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    overflow: 'hidden',
    zIndex: 100,
  },
  option: {
    display: 'block',
    width: '100%',
    padding: '9px 14px',
    background: 'none',
    border: 'none',
    color: 'var(--text-primary)',
    fontSize: 13,
    textAlign: 'left',
    cursor: 'pointer',
    transition: 'background 0.12s',
  },
}
