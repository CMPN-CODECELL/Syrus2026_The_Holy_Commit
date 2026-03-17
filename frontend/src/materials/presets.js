// Auto-generated from backend/materials/presets.py
// Do not edit manually — run scripts/sync_presets_to_frontend.py

export const METAL_PRESETS = {
  yellow_gold:  { name: "18K Yellow Gold", color: 0xFFD700, metalness: 1.0, roughness: 0.15, envMapIntensity: 1.5, price_per_gram: 62.50 },
  white_gold:   { name: "18K White Gold",  color: 0xE8E8E8, metalness: 1.0, roughness: 0.12, envMapIntensity: 1.8, price_per_gram: 65.00 },
  rose_gold:    { name: "18K Rose Gold",   color: 0xE8B4B8, metalness: 1.0, roughness: 0.18, envMapIntensity: 1.4, price_per_gram: 60.00 },
  platinum:     { name: "Platinum 950",    color: 0xE5E4E2, metalness: 1.0, roughness: 0.08, envMapIntensity: 2.0, price_per_gram: 32.00 },
  silver:       { name: "Sterling Silver", color: 0xC0C0C0, metalness: 1.0, roughness: 0.22, envMapIntensity: 1.3, price_per_gram: 0.85  },
  copper:       { name: "Copper",          color: 0xB87333, metalness: 1.0, roughness: 0.35, envMapIntensity: 1.0, price_per_gram: 0.01  },
}

export const GEMSTONE_PRESETS = {
  diamond:        { name: "Diamond",        color: 0xFFFFFF, metalness: 0.0, roughness: 0.00, transmission: 0.95, ior: 2.42, thickness: 0.5, envMapIntensity: 3.0, price_per_carat: 5000.00 },
  ruby:           { name: "Ruby",           color: 0xE0115F, metalness: 0.0, roughness: 0.05, transmission: 0.70, ior: 1.77, thickness: 0.5, envMapIntensity: 2.5, price_per_carat: 3000.00 },
  sapphire:       { name: "Sapphire",       color: 0x0F52BA, metalness: 0.0, roughness: 0.05, transmission: 0.70, ior: 1.77, thickness: 0.5, envMapIntensity: 2.5, price_per_carat: 2500.00 },
  emerald:        { name: "Emerald",        color: 0x50C878, metalness: 0.0, roughness: 0.10, transmission: 0.60, ior: 1.58, thickness: 0.5, envMapIntensity: 2.0, price_per_carat: 4000.00 },
  amethyst:       { name: "Amethyst",       color: 0x9966CC, metalness: 0.0, roughness: 0.08, transmission: 0.75, ior: 1.55, thickness: 0.5, envMapIntensity: 2.0, price_per_carat: 20.00  },
  topaz:          { name: "Topaz",          color: 0xFFC87C, metalness: 0.0, roughness: 0.05, transmission: 0.80, ior: 1.63, thickness: 0.5, envMapIntensity: 2.2, price_per_carat: 25.00  },
  cubic_zirconia: { name: "Cubic Zirconia", color: 0xFAFAFA, metalness: 0.0, roughness: 0.02, transmission: 0.90, ior: 2.15, thickness: 0.5, envMapIntensity: 2.8, price_per_carat: 2.00   },
}

export const DEFAULT_WEIGHTS = {
  ring:      { metal_grams: 5.0,  center_stone_carats: 1.0, accent_stone_carats: 0.15 },
  necklace:  { metal_grams: 15.0, center_stone_carats: 1.0, accent_stone_carats: 0.10 },
  earring:   { metal_grams: 3.0,  center_stone_carats: 0.5, accent_stone_carats: 0.05 },
  bracelet:  { metal_grams: 20.0, center_stone_carats: 0.0, accent_stone_carats: 0.10 },
}
