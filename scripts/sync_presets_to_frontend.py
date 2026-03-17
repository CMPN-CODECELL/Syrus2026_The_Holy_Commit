#!/usr/bin/env python3
"""
Sync material presets from backend/materials/presets.py to
frontend/src/materials/presets.js

Run from the project root: python scripts/sync_presets_to_frontend.py
"""
import os
import sys

# Add backend to path so we can import presets
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from materials.presets import METAL_PRESETS, GEMSTONE_PRESETS, DEFAULT_WEIGHTS


def py_to_js_value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, int):
        return hex(v) if v > 0xFFFF else str(v)
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def dict_to_js_object(d, indent=2):
    lines = ['{']
    items = list(d.items())
    for i, (k, v) in enumerate(items):
        comma = ',' if i < len(items) - 1 else ''
        lines.append(f"{' ' * indent}{k}: {py_to_js_value(v)}{comma}")
    lines.append('}')
    return ' '.join(lines)


def generate_js():
    lines = [
        '// Auto-generated from backend/materials/presets.py',
        '// Do not edit manually — run scripts/sync_presets_to_frontend.py',
        '',
        'export const METAL_PRESETS = {',
    ]
    for key, preset in METAL_PRESETS.items():
        inner = ', '.join(
            f'{k}: {py_to_js_value(v)}' for k, v in preset.items()
        )
        lines.append(f'  {key}: {{ {inner} }},')
    lines += ['}', '', 'export const GEMSTONE_PRESETS = {']
    for key, preset in GEMSTONE_PRESETS.items():
        inner = ', '.join(
            f'{k}: {py_to_js_value(v)}' for k, v in preset.items()
        )
        lines.append(f'  {key}: {{ {inner} }},')
    lines += ['}', '', 'export const DEFAULT_WEIGHTS = {']
    for key, weights in DEFAULT_WEIGHTS.items():
        inner = ', '.join(
            f'{k}: {py_to_js_value(v)}' for k, v in weights.items()
        )
        lines.append(f'  {key}: {{ {inner} }},')
    lines += ['}', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    out_dir = os.path.join(
        os.path.dirname(__file__), '..', 'frontend', 'src', 'materials'
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'presets.js')
    content = generate_js()
    with open(out_path, 'w') as f:
        f.write(content)
    print(f'Written: {out_path}')
    print(f'  Metals: {len(METAL_PRESETS)}, Gemstones: {len(GEMSTONE_PRESETS)}')
