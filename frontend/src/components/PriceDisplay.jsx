import React, { useState, useEffect, useRef } from 'react'

export default function PriceDisplay({ priceData, jobId, onRequestAlternatives }) {
  const [expanded, setExpanded] = useState(false)
  const [displayTotal, setDisplayTotal] = useState(0)
  const animRef = useRef(null)
  const prevTotalRef = useRef(0)

  // Animate price change
  useEffect(() => {
    const target = priceData?.total ?? 0
    const start = prevTotalRef.current
    if (start === target) return

    const duration = 600
    const startTime = performance.now()
    prevTotalRef.current = target

    const step = (now) => {
      const elapsed = now - startTime
      const t = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      setDisplayTotal(Math.round(start + (target - start) * eased))
      if (t < 1) {
        animRef.current = requestAnimationFrame(step)
      }
    }
    animRef.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(animRef.current)
  }, [priceData?.total])

  if (!priceData) {
    return (
      <div style={styles.card}>
        <div style={styles.empty}>Upload an image to see pricing</div>
      </div>
    )
  }

  const fmt = (n) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n)

  return (
    <div style={styles.card}>
      <div style={styles.totalRow}>
        <span style={styles.totalLabel}>Estimated Price</span>
        <span style={styles.totalValue}>{fmt(displayTotal)}</span>
      </div>

      <button style={styles.expandBtn} onClick={() => setExpanded((v) => !v)}>
        {expanded ? 'Hide breakdown ▲' : 'Show breakdown ▼'}
      </button>

      {expanded && (
        <div style={styles.breakdown}>
          {(priceData.breakdown ?? []).map((item, i) => (
            <div key={i} style={styles.line}>
              <span style={styles.lineLabel}>{item.label}</span>
              <span style={styles.lineCost}>{fmt(item.cost)}</span>
            </div>
          ))}
        </div>
      )}

      {onRequestAlternatives && (
        <button style={styles.altBtn} onClick={onRequestAlternatives}>
          See budget alternatives
        </button>
      )}
    </div>
  )
}

const styles = {
  card: {
    background: 'var(--bg-panel)',
    borderRadius: 10,
    padding: 14,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    border: '1px solid var(--border)',
  },
  empty: {
    fontSize: 12,
    color: 'var(--text-secondary)',
    fontStyle: 'italic',
    textAlign: 'center',
    padding: '8px 0',
  },
  totalRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'baseline',
  },
  totalLabel: {
    fontSize: 12,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  totalValue: {
    fontSize: 22,
    fontWeight: 700,
    color: 'var(--gold)',
    fontVariantNumeric: 'tabular-nums',
  },
  expandBtn: {
    background: 'none',
    border: 'none',
    color: 'var(--primary)',
    fontSize: 11,
    cursor: 'pointer',
    textAlign: 'left',
    padding: 0,
  },
  breakdown: {
    display: 'flex',
    flexDirection: 'column',
    gap: 5,
    borderTop: '1px solid var(--border)',
    paddingTop: 8,
  },
  line: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 11,
    color: 'var(--text-secondary)',
  },
  lineLabel: {
    flex: 1,
    marginRight: 8,
  },
  lineCost: {
    fontVariantNumeric: 'tabular-nums',
    color: 'var(--text-primary)',
  },
  altBtn: {
    background: 'transparent',
    border: '1px solid var(--primary)',
    borderRadius: 6,
    color: 'var(--primary)',
    fontSize: 11,
    padding: '5px 10px',
    cursor: 'pointer',
    transition: 'background 0.15s',
  },
}
