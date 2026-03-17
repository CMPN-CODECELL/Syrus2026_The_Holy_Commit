import { useState, useCallback } from 'react'
import { uploadImage as apiUpload, sendChatMessage } from '../api/pipeline.js'

/**
 * Custom hook for agent communication.
 *
 * @param {string|null} jobId  Current job ID (null before first upload)
 * @returns {{ sendMessage, uploadImage, isLoading, error }}
 */
export function useAgent(jobId) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  /**
   * Send a chat message to the agent.
   * @param {string} text
   * @param {object} currentConfig  Context about current state (materials, price, etc.)
   * @returns {object|null}  Parsed response or null on failure
   */
  const sendMessage = useCallback(
    async (text, currentConfig = {}) => {
      if (!jobId) return null
      setIsLoading(true)
      setError(null)
      try {
        const payload = { text, config: currentConfig }
        const result = await sendChatMessage(jobId, payload.text)
        return result
      } catch (err) {
        setError(err.message)
        return null
      } finally {
        setIsLoading(false)
      }
    },
    [jobId]
  )

  /**
   * Upload an image file and kick off the pipeline.
   * @param {File} file
   * @returns {object|null}  Job data or null on failure
   */
  const uploadImage = useCallback(async (file) => {
    setIsLoading(true)
    setError(null)
    try {
      const result = await apiUpload(file)
      return result
    } catch (err) {
      setError(err.message)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  return { sendMessage, uploadImage, isLoading, error }
}
