import React, { useState } from 'react'
import { METAL_PRESETS, GEMSTONE_PRESETS } from '../materials/presets.js'

export default function MaterialPanel({
  selectedComponent,
  materialsApplied,
  onApplyMaterial,
}) {
  const [hoveredKey, setHoveredKey] = useState(null)

  const currentMaterial = selectedComponent ? materialsApplied[selectedComponent] : null
  const isDisabled = !selectedComponent

  const handleSelect = (materialType, key) => {
    if (isDisabled) return
    const presets = materialType === 'metal' ? METAL_PRESETS : GEMSTONE_PRESETS
    onApplyMaterial(selectedComponent, materialType, key, presets[key])
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>Materials</div>

      {isDisabled && (
        <div style={styles.hint}>Select a component in the viewer to apply a material</div>
      )}

      {!isDisabled && (
        <div style={styles.selectedInfo}>
          Editing: <strong style={{ color: 'var(--accent)' }}>{selectedComponent}</strong>
          {currentMaterial && (
            <span style={styles.currentMat}> → {currentMaterial.preset?.name}</span>
          )}
        </div>
      )}

      <Section
        title="Metals"
        presets={METAL_PRESETS}
        materialType="metal"
        isDisabled={isDisabled}
        currentKey={currentMaterial?.materialType === 'metal' ? currentMaterial.materialKey : null}
        hoveredKey={hoveredKey}
        onHover={setHoveredKey}
        onSelect={handleSelect}
      />

      <Section
        title="Gemstones"
        presets={GEMSTONE_PRESETS}
        materialType="gemstone"
        isDisabled={isDisabled}
        currentKey={currentMaterial?.materialType === 'gemstone' ? currentMaterial.materialKey : null}
        hoveredKey={hoveredKey}
        onHover={setHoveredKey}
        onSelect={handleSelect}
      />
    </div>
  )
}

function Section({ title, presets, materialType, isDisabled, currentKey, hoveredKey, onHover, onSelect }) {
  return (
    <div style={styles.section}>
      <div style={styles.sectionTitle}>{title}</div>
      <div style={styles.swatchGrid}>
        {Object.entries(presets).map(([key, preset]) => {
          const hex = '#' + (preset.color >>> 0).toString(16).padStart(6, '0')
          const isActive = currentKey === key
          const isHovered = hoveredKey === key
          return (
            <div
              key={key}
              title={preset.name}
              style={{
                ...styles.swatch,
                background: hex,
                border: isActive
                  ? '2px solid var(--accent)'
                  : isHovered
                  ? '2px solid var(--primary)'
                  : '2px solid transparent',
                opacity: isDisabled ? 0.4 : 1,
                cursor: isDisabled ? 'not-allowed' : 'pointer',
                transform: isHovered && !isDisabled ? 'scale(1.15)' : 'scale(1)',
              }}
              onMouseEnter={() => onHover(key)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onSelect(materialType, key)}
            />
          )
        })}
      </div>
      {hoveredKey && presets[hoveredKey] && (
        <div style={styles.tooltip}>{presets[hoveredKey].name}</div>
      )}
    </div>
  )
}

const styles = {
  panel: {
    background: 'var(--bg-panel)',
    borderRadius: 10,
    padding: 14,
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
    border: '1px solid var(--border)',
  },
  header: {
    fontSize: 13,
    fontWeight: 600,
    color: 'var(--text-primary)',
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
  },
  hint: {
    fontSize: 12,
    color: 'var(--text-secondary)',
    fontStyle: 'italic',
  },
  selectedInfo: {
    fontSize: 12,
    color: 'var(--text-secondary)',
  },
  currentMat: {
    color: 'var(--silver)',
    fontSize: 11,
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: 500,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
  },
  swatchGrid: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
  },
  swatch: {
    width: 28,
    height: 28,
    borderRadius: '50%',
    transition: 'transform 0.12s ease, border-color 0.12s ease',
    boxShadow: '0 2px 6px rgba(0,0,0,0.4)',
    flexShrink: 0,
  },
  tooltip: {
    fontSize: 11,
    color: 'var(--accent)',
    marginTop: 2,
  },
}
