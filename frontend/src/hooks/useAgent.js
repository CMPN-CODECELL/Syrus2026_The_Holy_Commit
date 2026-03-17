import { useState, useCallback } from 'react'
import { chatWithAgent } from '../api/pipeline'

export function useAgent(jobId, onPriceUpdate) {
  const [messages, setMessages] = useState([])
  const [isTyping, setIsTyping] = useState(false)

  const sendMessage = useCallback(async (text) => {
    if (!jobId) return

    setMessages(prev => [...prev, { role: 'user', text }])
    setIsTyping(true)

    try {
      const result = await chatWithAgent(jobId, text)
      const responseText = result.response_text || 'Done.'
      setMessages(prev => [...prev, { role: 'assistant', text: responseText }])
      if (result.price && onPriceUpdate) onPriceUpdate(result.price)
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: err.message || 'Sorry, something went wrong. Please try again.',
      }])
    } finally {
      setIsTyping(false)
    }
  }, [jobId, onPriceUpdate])

  return { messages, sendMessage, isTyping }
}
