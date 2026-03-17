import { useState, useCallback } from 'react'
import { swapMaterial } from '../api/pipeline'

export function useMaterialSwap(jobId, onPriceUpdate) {
  const [appliedMaterials, setAppliedMaterials] = useState({})
  const [isSwapping, setIsSwapping] = useState(false)

  const swap = useCallback(async (component, material) => {
    if (!jobId || !component || !material) return null
    setIsSwapping(true)
    try {
      const result = await swapMaterial(jobId, component, material)
      if (!result.error) {
        setAppliedMaterials(prev => ({ ...prev, [component]: material }))
        if (result.price && onPriceUpdate) onPriceUpdate(result.price)
      }
      return result
    } finally {
      setIsSwapping(false)
    }
  }, [jobId, onPriceUpdate])

  return { swap, appliedMaterials, isSwapping }
}
