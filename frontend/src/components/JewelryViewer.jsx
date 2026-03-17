import React, { useRef, useEffect, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import {
  OrbitControls,
  Environment,
  ContactShadows,
  useGLTF,
} from '@react-three/drei'
import * as THREE from 'three'
import { ACESFilmicToneMapping } from 'three'
import { useSegmentedMesh } from '../hooks/useSegmentedMesh.js'
import { useMaterialSwap } from '../hooks/useMaterialSwap.js'
import { METAL_PRESETS, GEMSTONE_PRESETS } from '../materials/presets.js'

function JewelryModel({ meshUrl, labelsUrl, materialsApplied, selectedComponent, onSelectComponent }) {
  const { scene, faceLabels, components, isLoading } = useSegmentedMesh(meshUrl, labelsUrl)
  const { swapMaterial } = useMaterialSwap()
  const meshRef = useRef()

  // Apply material swaps whenever materialsApplied changes
  useEffect(() => {
    if (!scene) return
    scene.traverse((obj) => {
      if (obj.isMesh) {
        for (const [component, { materialType, preset }] of Object.entries(materialsApplied)) {
          swapMaterial(obj, component, preset, faceLabels)
        }
      }
    })
  }, [scene, materialsApplied, faceLabels, swapMaterial])

  // Highlight selected component
  useEffect(() => {
    if (!scene) return
    scene.traverse((obj) => {
      if (obj.isMesh && obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material]
        mats.forEach((mat) => {
          if (mat.emissive) {
            mat.emissive.set(selectedComponent ? 0x221144 : 0x000000)
          }
        })
      }
    })
  }, [scene, selectedComponent])

  if (isLoading || !scene) return null

  const handleClick = (e) => {
    e.stopPropagation()
    if (!faceLabels || !e.face) return
    const faceIndex = e.faceIndex
    const componentName = faceLabels[String(faceIndex)]
    if (componentName && onSelectComponent) {
      onSelectComponent(componentName)
    }
  }

  return (
    <primitive
      ref={meshRef}
      object={scene}
      onClick={handleClick}
      dispose={null}
    />
  )
}

export default function JewelryViewer({
  meshUrl,
  labelsUrl,
  materialsApplied,
  selectedComponent,
  onSelectComponent,
  onSceneReady,
}) {
  return (
    <div style={{ width: '100%', height: '100%', background: '#0d0d1a' }}>
      {!meshUrl ? (
        <div style={styles.placeholder}>
          <div style={styles.placeholderIcon}>💎</div>
          <div style={styles.placeholderText}>Upload a jewelry image to begin</div>
          <div style={styles.placeholderSub}>Your 3D model will appear here</div>
        </div>
      ) : (
        <Canvas
          camera={{ position: [0, 0, 3], fov: 45 }}
          gl={{ toneMapping: ACESFilmicToneMapping, toneMappingExposure: 1.2 }}
          shadows
          onCreated={({ scene }) => onSceneReady && onSceneReady(scene)}
        >
          <ambientLight intensity={0.4} />
          <spotLight
            position={[5, 8, 5]}
            angle={0.3}
            penumbra={0.5}
            intensity={1.5}
            castShadow
          />
          <spotLight
            position={[-5, 4, -5]}
            angle={0.4}
            penumbra={0.6}
            intensity={0.8}
            color="#ffeedd"
          />

          <Suspense fallback={null}>
            <Environment preset="studio" />
            <JewelryModel
              meshUrl={meshUrl}
              labelsUrl={labelsUrl}
              materialsApplied={materialsApplied}
              selectedComponent={selectedComponent}
              onSelectComponent={onSelectComponent}
            />
            <ContactShadows
              position={[0, -1.2, 0]}
              opacity={0.6}
              scale={4}
              blur={2}
              far={2}
            />
          </Suspense>

          <OrbitControls
            enableDamping
            dampingFactor={0.08}
            minDistance={0.5}
            maxDistance={8}
          />
        </Canvas>
      )}
    </div>
  )
}

const styles = {
  placeholder: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    gap: 12,
    color: 'var(--text-secondary)',
    userSelect: 'none',
  },
  placeholderIcon: {
    fontSize: 64,
    opacity: 0.4,
  },
  placeholderText: {
    fontSize: 18,
    fontWeight: 500,
    color: 'var(--text-secondary)',
  },
  placeholderSub: {
    fontSize: 13,
    opacity: 0.6,
  },
}
