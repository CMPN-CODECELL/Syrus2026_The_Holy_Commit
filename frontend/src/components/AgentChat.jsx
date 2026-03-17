import { useState, useRef, useEffect } from 'react'
import { useAgent } from '../hooks/useAgent'

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 8,
    }}>
      <div style={{
        maxWidth: '80%',
        padding: '8px 12px',
        borderRadius: isUser ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
        background: isUser ? '#2563eb' : '#1e1e1e',
        color: '#f0f0f0',
        fontSize: 13,
        lineHeight: 1.5,
        border: isUser ? 'none' : '1px solid #2a2a2a',
      }}>
        {msg.text}
      </div>
    </div>
  )
}

export default function AgentChat({ jobId, onPriceUpdate }) {
  const [inputText, setInputText] = useState('')
  const { messages, sendMessage, isTyping } = useAgent(jobId, onPriceUpdate)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  const handleSend = () => {
    if (!inputText.trim() || !jobId) return
    sendMessage(inputText.trim())
    setInputText('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{
        padding: '10px 16px',
        fontSize: 12,
        color: '#888',
        textTransform: 'uppercase',
        letterSpacing: 1,
        borderBottom: '1px solid #2a2a2a',
      }}>
        AI Assistant
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
        {!jobId && (
          <div style={{ color: '#555', fontSize: 13, textAlign: 'center', marginTop: 24 }}>
            Upload a jewelry image to start chatting
          </div>
        )}
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        {isTyping && (
          <div style={{ color: '#666', fontSize: 13, fontStyle: 'italic' }}>
            Thinking...
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div style={{ padding: '12px 16px', borderTop: '1px solid #2a2a2a', display: 'flex', gap: 8 }}>
        <textarea
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={jobId ? 'e.g. "Make it premium under $1000"' : 'Upload an image first'}
          disabled={!jobId}
          rows={2}
          style={{
            flex: 1,
            background: '#1e1e1e',
            border: '1px solid #333',
            borderRadius: 8,
            color: '#f0f0f0',
            padding: '8px 12px',
            fontSize: 13,
            resize: 'none',
            outline: 'none',
          }}
        />
        <button
          onClick={handleSend}
          disabled={!jobId || !inputText.trim() || isTyping}
          style={{
            padding: '8px 16px',
            background: '#2563eb',
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            cursor: 'pointer',
            fontSize: 13,
            alignSelf: 'flex-end',
            opacity: (!jobId || !inputText.trim() || isTyping) ? 0.4 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  )
}
