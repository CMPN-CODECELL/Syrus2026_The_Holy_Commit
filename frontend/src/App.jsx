import { useState } from 'react'
import JewelryViewer from './components/JewelryViewer'
import MaterialPanel from './components/MaterialPanel'
import AgentChat from './components/AgentChat'
import PriceDisplay from './components/PriceDisplay'
import UploadDropzone from './components/UploadDropzone'
import ExportButton from './components/ExportButton'

export default function App() {
  const [jobId, setJobId] = useState(null)
  const [meshUrl, setMeshUrl] = useState(null)
  const [labelsUrl, setLabelsUrl] = useState(null)
  const [components, setComponents] = useState({})
  const [price, setPrice] = useState(null)
  const [selectedComponent, setSelectedComponent] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleUploadComplete = (result) => {
    setJobId(result.job_id)
    setMeshUrl(result.mesh_url)
    setLabelsUrl(result.labels_url)
    setComponents(result.components || {})
    setPrice(result.price)
  }

  const handlePriceUpdate = (newPrice) => {
    setPrice(newPrice)
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Left panel: 3D Viewer */}
      <div style={{ flex: '1 1 60%', position: 'relative', background: '#111' }}>
        {!meshUrl ? (
          <UploadDropzone
            onUploadComplete={handleUploadComplete}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        ) : (
          <>
            <JewelryViewer
              meshUrl={meshUrl}
              labelsUrl={labelsUrl}
              selectedComponent={selectedComponent}
              onSelectComponent={setSelectedComponent}
            />
            <div style={{ position: 'absolute', bottom: 16, left: 16 }}>
              <ExportButton meshUrl={meshUrl} jobId={jobId} />
            </div>
          </>
        )}
      </div>

      {/* Right panel: Controls */}
      <div style={{
        flex: '0 0 380px',
        display: 'flex',
        flexDirection: 'column',
        borderLeft: '1px solid #2a2a2a',
        background: '#141414',
      }}>
        {/* Price display */}
        <PriceDisplay price={price} />

        {/* Material swatches */}
        {jobId && (
          <MaterialPanel
            jobId={jobId}
            components={components}
            selectedComponent={selectedComponent}
            onPriceUpdate={handlePriceUpdate}
          />
        )}

        {/* Chat */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <AgentChat
            jobId={jobId}
            onPriceUpdate={handlePriceUpdate}
          />
        </div>
      </div>
    </div>
  )
}
