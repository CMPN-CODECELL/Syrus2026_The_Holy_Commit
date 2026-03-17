import React, { useState, useRef, useEffect } from 'react'
import { useAgent } from '../hooks/useAgent.js'

const GREETING = {
  role: 'assistant',
  text: 'Hi! I\'m your JewelForge AI assistant. Upload a jewelry image to get started — I\'ll generate a 3D model, identify the components, and help you customize materials and pricing.',
}

export default function AgentChat({ jobId, components, materialsApplied, priceData, onAction }) {
  const [messages, setMessages] = useState([GREETING])
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const { sendMessage, isLoading } = useAgent(jobId)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || isLoading) return
    setInput('')

    setMessages((prev) => [...prev, { role: 'user', text }])

    const currentConfig = {
      components,
      materialsApplied,
      priceTotal: priceData?.total,
    }

    const result = await sendMessage(text, currentConfig)

    if (result) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: result.response_text || '...' },
      ])
      if (result.actions?.length && onAction) {
        onAction(result.actions)
      }
    } else {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: 'Sorry, something went wrong. Please try again.' },
      ])
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>AI Assistant</div>

      <div style={styles.messages}>
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              ...styles.bubble,
              ...(msg.role === 'user' ? styles.userBubble : styles.assistantBubble),
            }}
          >
            {msg.text}
          </div>
        ))}

        {isLoading && (
          <div style={{ ...styles.bubble, ...styles.assistantBubble }}>
            <TypingIndicator />
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div style={styles.inputBar}>
        <textarea
          style={styles.textarea}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={jobId ? 'Ask about materials, price, or design…' : 'Upload an image first…'}
          disabled={!jobId || isLoading}
          rows={2}
        />
        <button
          style={{
            ...styles.sendBtn,
            opacity: (!jobId || isLoading || !input.trim()) ? 0.5 : 1,
          }}
          onClick={handleSend}
          disabled={!jobId || isLoading || !input.trim()}
        >
          ↑
        </button>
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <span style={styles.typing}>
      <span style={styles.dot} />
      <span style={{ ...styles.dot, animationDelay: '0.2s' }} />
      <span style={{ ...styles.dot, animationDelay: '0.4s' }} />
    </span>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
  },
  header: {
    padding: '12px 16px',
    fontSize: 13,
    fontWeight: 600,
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
    color: 'var(--text-secondary)',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '12px 12px 4px',
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  bubble: {
    maxWidth: '88%',
    padding: '8px 12px',
    borderRadius: 12,
    fontSize: 13,
    lineHeight: 1.5,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  userBubble: {
    alignSelf: 'flex-end',
    background: 'var(--primary)',
    color: '#fff',
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    alignSelf: 'flex-start',
    background: 'var(--bg-panel)',
    color: 'var(--text-primary)',
    borderBottomLeftRadius: 4,
    border: '1px solid var(--border)',
  },
  inputBar: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: 8,
    padding: '8px 12px',
    borderTop: '1px solid var(--border)',
    flexShrink: 0,
  },
  textarea: {
    flex: 1,
    background: 'var(--bg-panel)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    color: 'var(--text-primary)',
    padding: '8px 10px',
    fontSize: 13,
    resize: 'none',
    outline: 'none',
    lineHeight: 1.4,
  },
  sendBtn: {
    width: 36,
    height: 36,
    borderRadius: '50%',
    background: 'var(--primary)',
    color: '#fff',
    border: 'none',
    fontSize: 18,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    transition: 'opacity 0.15s',
  },
  typing: {
    display: 'flex',
    gap: 4,
    alignItems: 'center',
    height: 18,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: 'var(--text-secondary)',
    animation: 'pulse 1s infinite ease-in-out',
    display: 'inline-block',
  },
}

// Inject keyframes once
if (typeof document !== 'undefined' && !document.getElementById('jf-keyframes')) {
  const style = document.createElement('style')
  style.id = 'jf-keyframes'
  style.textContent = `
    @keyframes pulse {
      0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
      40% { transform: scale(1); opacity: 1; }
    }
  `
  document.head.appendChild(style)
}
