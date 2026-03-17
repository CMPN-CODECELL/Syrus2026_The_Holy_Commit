import { useState } from 'react'
import { METAL_PRESETS, GEMSTONE_PRESETS } from '../materials/presets'
import { swapMaterial } from '../api/pipeline'

function hexColor(num) {
  return '#' + num.toString(16).padStart(6, '0')
}

function Swatch({ label, color, onClick, selected }) {
  return (
    <button
      title={label}
      onClick={onClick}
      style={{
        width: 36,
        height: 36,
        borderRadius: 6,
        border: selected ? '2px solid #fff' : '2px solid transparent',
        background: hexColor(color),
        cursor: 'pointer',
        transition: 'border 0.15s',
      }}
    />
  )
}

export default function MaterialPanel({
  jobId, components, selectedComponent, onPriceUpdate
}) {
  const [appliedMaterials, setAppliedMaterials] = useState({})

  const handleSwap = async (material) => {
    if (!selectedComponent || !jobId) return
    const result = await swapMaterial(jobId, selectedComponent, material)
    if (result.price) {
      setAppliedMaterials(prev => ({ ...prev, [selectedComponent]: material }))
      onPriceUpdate(result.price)
    }
  }

  const selectedMaterial = selectedComponent ? appliedMaterials[selectedComponent] : null

  return (
    <div style={{ padding: 16, borderBottom: '1px solid #2a2a2a' }}>
      <div style={{ fontSize: 12, color: '#888', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
        {selectedComponent
          ? `Editing: ${selectedComponent}`
          : 'Click a part to select'}
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: '#666', marginBottom: 6 }}>METALS</div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {Object.entries(METAL_PRESETS).map(([key, preset]) => (
            <Swatch
              key={key}
              label={preset.name}
              color={preset.color}
              selected={selectedMaterial === key}
              onClick={() => handleSwap(key)}
            />
          ))}
        </div>
      </div>

      <div>
        <div style={{ fontSize: 11, color: '#666', marginBottom: 6 }}>GEMSTONES</div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {Object.entries(GEMSTONE_PRESETS).map(([key, preset]) => (
            <Swatch
              key={key}
              label={preset.name}
              color={preset.color}
              selected={selectedMaterial === key}
              onClick={() => handleSwap(key)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
