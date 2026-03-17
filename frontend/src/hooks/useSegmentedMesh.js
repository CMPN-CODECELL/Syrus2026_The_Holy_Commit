import { useState, useEffect } from 'react'
import { useGLTF } from '@react-three/drei'

export function useSegmentedMesh(meshUrl, labelsUrl) {
  const [labels, setLabels] = useState(null)
  const fullMeshUrl = meshUrl || null
  const { scene } = fullMeshUrl ? useGLTF(fullMeshUrl) : { scene: null }

  useEffect(() => {
    if (!labelsUrl) return
    fetch(labelsUrl)
      .then(r => r.json())
      .then(setLabels)
      .catch(console.error)
  }, [labelsUrl])

  return { scene, labels }
}
