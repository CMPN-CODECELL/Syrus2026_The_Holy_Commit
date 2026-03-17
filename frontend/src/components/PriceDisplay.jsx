function PriceLine({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
      <span style={{ color: '#888', fontSize: 12 }}>{label}</span>
      <span style={{ color: '#f0f0f0', fontSize: 12 }}>${value?.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
    </div>
  )
}

export default function PriceDisplay({ price }) {
  if (!price) {
    return (
      <div style={{ padding: 16, borderBottom: '1px solid #2a2a2a' }}>
        <div style={{ fontSize: 12, color: '#555', textAlign: 'center' }}>
          Price estimate will appear here
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: 16, borderBottom: '1px solid #2a2a2a' }}>
      <div style={{ fontSize: 12, color: '#888', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>
        Price Estimate
      </div>

      {/* Metal line */}
      {price.metal && (
        <PriceLine
          label={`Metal (${price.metal.type?.replace(/_/g, ' ')} ${price.metal.grams}g)`}
          value={price.metal.cost}
        />
      )}

      {/* Gemstone lines */}
      {price.gemstones && Object.entries(price.gemstones).map(([comp, gem]) => (
        <PriceLine
          key={comp}
          label={`${comp.replace(/_/g, ' ')} (${gem.type?.replace(/_/g, ' ')} ${gem.carats}ct)`}
          value={gem.cost}
        />
      ))}

      {/* Labor */}
      {price.labor_estimate != null && (
        <PriceLine label="Labor estimate" value={price.labor_estimate} />
      )}

      {/* Total */}
      <div style={{
        marginTop: 10,
        paddingTop: 10,
        borderTop: '1px solid #2a2a2a',
        display: 'flex',
        justifyContent: 'space-between',
      }}>
        <span style={{ color: '#f0f0f0', fontWeight: 600, fontSize: 15 }}>Total</span>
        <span style={{ color: '#fbbf24', fontWeight: 700, fontSize: 18 }}>
          ${price.total?.toLocaleString('en-US', { minimumFractionDigits: 2 })}
        </span>
      </div>
    </div>
  )
}
