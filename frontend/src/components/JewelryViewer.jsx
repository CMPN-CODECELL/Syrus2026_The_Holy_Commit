import { useRef, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, useGLTF, Environment, Html } from '@react-three/drei'
import { useSegmentedMesh } from '../hooks/useSegmentedMesh'

function JewelModel({ meshUrl, labelsUrl, selectedComponent, onSelectComponent }) {
  const { scene, labels } = useSegmentedMesh(meshUrl, labelsUrl)

  const handleClick = (e) => {
    e.stopPropagation()
    if (!labels || !e.face) return
    const faceIndex = e.face.materialIndex ?? Math.floor(e.faceIndex)
    const faceToComp = labels.face_to_component
    const comp = faceToComp?.[String(faceIndex)] || faceToComp?.[faceIndex]
    if (comp) onSelectComponent(comp)
  }

  if (!scene) return null

  return (
    <primitive
      object={scene}
      onClick={handleClick}
      scale={1.5}
    />
  )
}

export default function JewelryViewer({
  meshUrl, labelsUrl, selectedComponent, onSelectComponent
}) {
  return (
    <Canvas
      camera={{ position: [0, 0, 3], fov: 50 }}
      style={{ width: '100%', height: '100%' }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={1.2} castShadow />
      <directionalLight position={[-5, -5, -5]} intensity={0.4} />
      <pointLight position={[0, 3, 0]} intensity={0.8} />

      <Suspense fallback={
        <Html center>
          <div style={{ color: '#aaa', fontSize: 14 }}>Loading 3D model...</div>
        </Html>
      }>
        {meshUrl && (
          <JewelModel
            meshUrl={meshUrl}
            labelsUrl={labelsUrl}
            selectedComponent={selectedComponent}
            onSelectComponent={onSelectComponent}
          />
        )}
        <Environment preset="studio" />
      </Suspense>

      <OrbitControls
        enablePan={false}
        minDistance={1}
        maxDistance={10}
        autoRotate
        autoRotateSpeed={0.5}
      />
    </Canvas>
  )
}
