import React, { useState, useCallback } from 'react'
import JewelryViewer from './components/JewelryViewer.jsx'
import MaterialPanel from './components/MaterialPanel.jsx'
import AgentChat from './components/AgentChat.jsx'
import PriceDisplay from './components/PriceDisplay.jsx'
import UploadDropzone from './components/UploadDropzone.jsx'
import ExportButton from './components/ExportButton.jsx'

export default function App() {
  const [currentJob, setCurrentJob] = useState(null)
  // currentJob: { jobId, meshUrl, labelsUrl, components, price }

  const [selectedComponent, setSelectedComponent] = useState(null)
  // materialsApplied: { [componentName]: { materialType, materialKey, preset } }
  const [materialsApplied, setMaterialsApplied] = useState({})
  const [priceData, setPriceData] = useState(null)
  const [sceneRef, setSceneRef] = useState(null)

  const handleUploadComplete = useCallback((jobData) => {
    setCurrentJob(jobData)
    setPriceData(jobData.price ?? null)
    setMaterialsApplied({})
    setSelectedComponent(null)
  }, [])

  const handleSelectComponent = useCallback((componentName) => {
    setSelectedComponent(componentName)
  }, [])

  const handleApplyMaterial = useCallback((component, materialType, materialKey, preset) => {
    setMaterialsApplied((prev) => ({
      ...prev,
      [component]: { materialType, materialKey, preset },
    }))
  }, [])

  // Called by AgentChat when the agent returns actions
  const handleAgentAction = useCallback((actions) => {
    for (const action of actions) {
      if (action.tool === 'apply_material' && action.result?.success) {
        const { component, preset } = action.result
        const materialType = action.input.material_type
        const materialKey = action.input.material_key
        handleApplyMaterial(component, materialType, materialKey, preset)
      }
      if (action.tool === 'estimate_price' && action.result?.total) {
        setPriceData(action.result)
      }
    }
  }, [handleApplyMaterial])

  return (
    <div style={styles.root}>
      {/* Left sidebar */}
      <aside style={styles.sidebar}>
        <UploadDropzone onUploadComplete={handleUploadComplete} />
        <MaterialPanel
          selectedComponent={selectedComponent}
          materialsApplied={materialsApplied}
          onApplyMaterial={handleApplyMaterial}
          jobId={currentJob?.jobId}
        />
        <div style={styles.sidebarBottom}>
          <PriceDisplay
            priceData={priceData}
            jobId={currentJob?.jobId}
          />
          <ExportButton scene={sceneRef} meshUrl={currentJob?.meshUrl} />
        </div>
      </aside>

      {/* Center — 3D viewer */}
      <main style={styles.viewer}>
        <JewelryViewer
          meshUrl={currentJob?.meshUrl}
          labelsUrl={currentJob?.labelsUrl}
          materialsApplied={materialsApplied}
          selectedComponent={selectedComponent}
          onSelectComponent={handleSelectComponent}
          onSceneReady={setSceneRef}
        />
      </main>

      {/* Right sidebar — agent chat */}
      <aside style={styles.chat}>
        <AgentChat
          jobId={currentJob?.jobId}
          components={currentJob?.components ?? []}
          materialsApplied={materialsApplied}
          priceData={priceData}
          onAction={handleAgentAction}
        />
      </aside>
    </div>
  )
}

const styles = {
  root: {
    display: 'flex',
    width: '100%',
    height: '100%',
    overflow: 'hidden',
    background: 'var(--bg-dark)',
  },
  sidebar: {
    width: 280,
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    padding: 12,
    background: 'var(--bg-card)',
    borderRight: '1px solid var(--border)',
    overflowY: 'auto',
  },
  sidebarBottom: {
    marginTop: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  viewer: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
  },
  chat: {
    width: 320,
    flexShrink: 0,
    background: 'var(--bg-card)',
    borderLeft: '1px solid var(--border)',
    overflow: 'hidden',
  },
}
