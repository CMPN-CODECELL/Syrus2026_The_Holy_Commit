import { useCallback, useRef } from 'react'
import * as THREE from 'three'

/**
 * Custom hook for swapping PBR material properties on a Three.js mesh.
 */
export function useMaterialSwap() {
  // Cache created materials by component label to avoid unnecessary allocations
  const materialCache = useRef({})

  /**
   * Apply a material preset to all faces of a mesh that belong to componentLabel.
   * If faceLabels is null, applies the material to the entire mesh.
   *
   * @param {THREE.Mesh} mesh
   * @param {string}     componentLabel
   * @param {object}     preset         - JewelForge PBR preset object
   * @param {object|null} faceLabels    - { [faceIndex]: componentName } or null
   */
  const swapMaterial = useCallback((mesh, componentLabel, preset, faceLabels) => {
    if (!mesh || !mesh.isMesh || !preset) return

    const key = `${componentLabel}__${preset.name ?? JSON.stringify(preset)}`

    if (!materialCache.current[key]) {
      const mat = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(preset.color ?? 0xffffff),
        metalness: preset.metalness ?? 0,
        roughness: preset.roughness ?? 0.5,
        envMapIntensity: preset.envMapIntensity ?? 1,
      })

      // Gemstone-specific properties
      if (preset.transmission != null) {
        mat.transmission = preset.transmission
        mat.transparent = true
      }
      if (preset.ior != null) mat.ior = preset.ior
      if (preset.thickness != null) mat.thickness = preset.thickness

      materialCache.current[key] = mat
    }

    const newMat = materialCache.current[key]

    if (!faceLabels) {
      // No face labels — apply to whole mesh
      mesh.material = newMat
      mesh.material.needsUpdate = true
      return
    }

    // Build a face→group map and assign per-group materials
    // Find all face indices belonging to this component
    const geometry = mesh.geometry
    if (!geometry) return

    const faceCount = geometry.index
      ? geometry.index.count / 3
      : geometry.attributes.position.count / 3

    // Build groups: runs of consecutive faces with the same material index
    // We collect face indices for the target component and patch material index
    // This is a simplified approach: create a multi-material mesh
    const groups = geometry.groups
    if (groups && groups.length > 0) {
      // Already grouped — just swap the relevant material slot
      const currentMaterials = Array.isArray(mesh.material)
        ? mesh.material
        : [mesh.material]

      const updatedMaterials = [...currentMaterials]
      groups.forEach((group, gi) => {
        const midFace = group.start / 3 + Math.floor(group.count / 6)
        if (faceLabels[String(midFace)] === componentLabel) {
          updatedMaterials[gi] = newMat
        }
      })
      mesh.material = updatedMaterials
    } else {
      // No groups — apply to whole mesh if component name matches
      mesh.material = newMat
    }

    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    mats.forEach((m) => { m.needsUpdate = true })
  }, [])

  return { swapMaterial, materialCache: materialCache.current }
}
