import { useState, useEffect } from 'react'
import { useGLTF } from '@react-three/drei'

/**
 * Loads a GLB mesh and its face-to-component labels JSON.
 *
 * @param {string|null} meshUrl    URL to the .glb file
 * @param {string|null} labelsUrl  URL to the _labels.json file
 * @returns {{ scene, faceLabels, labelNames, components, isLoading }}
 */
export function useSegmentedMesh(meshUrl, labelsUrl) {
  const [faceLabels, setFaceLabels] = useState(null)  // { [faceIndex]: componentName }
  const [labelNames, setLabelNames] = useState([])    // unique component names
  const [isLoading, setIsLoading] = useState(false)

  // useGLTF caches by URL and throws a Promise (Suspense) — call unconditionally
  // but only consume the result when meshUrl is set
  const gltf = meshUrl ? useGLTF(meshUrl) : null

  useEffect(() => {
    if (!labelsUrl) {
      setFaceLabels(null)
      setLabelNames([])
      return
    }

    setIsLoading(true)
    fetch(labelsUrl)
      .then((r) => r.json())
      .then((json) => {
        setFaceLabels(json)
        const unique = [...new Set(Object.values(json))]
        setLabelNames(unique)
        setIsLoading(false)
      })
      .catch(() => {
        setFaceLabels(null)
        setIsLoading(false)
      })
  }, [labelsUrl])

  return {
    scene: gltf?.scene ?? null,
    faceLabels,
    labelNames,
    components: labelNames,
    isLoading: isLoading || (meshUrl && !gltf),
  }
}
