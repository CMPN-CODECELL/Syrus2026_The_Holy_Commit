/**
 * viewer.js — Three.js GLB viewer.
 *
 * Loads a GLB from the backend and displays the raw mesh exactly as
 * TripoSG outputs it — no material override. Labels JSON is fetched
 * only to log component info; geometry is never split on load.
 */

import * as THREE from 'three';
import { OrbitControls }   from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader }      from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const BACKEND = window.location.origin;

// ─── DOM refs ────────────────────────────────────────────────────────────────
const dropZone          = document.getElementById('drop-zone');
const fileInput         = document.getElementById('file-input');
const previewOrig       = document.getElementById('preview-original');
const previewSeg        = document.getElementById('preview-seg');
const previewSegMasks   = document.getElementById('preview-seg-masks');
const segPlaceholder    = document.getElementById('seg-placeholder');
const btnGenerate       = document.getElementById('btn-generate');
const statusEl          = document.getElementById('status');
const loadingOverlay    = document.getElementById('loading-overlay');
const loadingText       = document.getElementById('loading-text');
const viewerPlaceholder = document.getElementById('viewer-placeholder');
const canvas            = document.getElementById('viewer-canvas');

// ─── State ───────────────────────────────────────────────────────────────────
let selectedFile = null;
let sceneGroup   = null;

// ─── Default materials ────────────────────────────────────────────────────────
const DEFAULT_BAND_MAT = () => new THREE.MeshPhysicalMaterial({
  color: new THREE.Color(0xFFD700), metalness: 1.0, roughness: 0.15,
  reflectivity: 1.0, clearcoat: 0.3, clearcoatRoughness: 0.1,
});

const DEFAULT_GEM_MAT = () => new THREE.MeshPhysicalMaterial({
  color: new THREE.Color(0xEEF4FF), metalness: 0.0, roughness: 0.02,
  transmission: 0.9, thickness: 1.5, ior: 2.42,
  reflectivity: 1.0, clearcoat: 1.0, clearcoatRoughness: 0.0,
});

// ─── Three.js scene setup ────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace    = THREE.SRGBColorSpace;
renderer.toneMapping         = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.shadowMap.enabled   = true;
renderer.useLegacyLights     = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0e11);

const pmremGenerator = new THREE.PMREMGenerator(renderer);
scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
pmremGenerator.dispose();

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0, 0.8, 2.5);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance   = 0.2;
controls.maxDistance   = 10;
controls.target.set(0, 0, 0);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
scene.add(new THREE.HemisphereLight(0xb0c8ff, 0x5a3c10, 0.8));

const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
keyLight.position.set(3, 5, 3);
keyLight.castShadow = true;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x8090ff, 0.8);
fillLight.position.set(-3, 2, -3);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffeecc, 1.0);
rimLight.position.set(0, -2, -4);
scene.add(rimLight);

// Resize
function resizeRenderer() {
  const wrap = canvas.parentElement;
  renderer.setSize(wrap.clientWidth, wrap.clientHeight, false);
  camera.aspect = wrap.clientWidth / wrap.clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resizeRenderer);
resizeRenderer();

// Render loop
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

// ─── File upload handling ─────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) setFile(f);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  previewOrig.src = URL.createObjectURL(file);
  previewOrig.style.display = 'block';
  btnGenerate.disabled = false;
  setStatus('Image loaded — click "Generate 3D Model".', 'info');
}

// ─── Generate button ──────────────────────────────────────────────────────────
btnGenerate.addEventListener('click', async () => {
  if (!selectedFile) return;

  btnGenerate.disabled = true;
  setStatus('Uploading and running pipeline… (may take 1–3 min)', 'info');
  loadingOverlay.classList.add('visible');
  loadingText.textContent = 'Running segmentation + 3D generation…';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch(`${BACKEND}/generate`, { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Server error');
    }

    const data = await resp.json();
    loadingText.textContent = 'Loading 3D model into viewer…';

    // Show segmentation images — prefer DINO boxes over the RMBG overlay when available
    const boxesUrl = data.seg_boxes
      ? `${BACKEND}${data.seg_boxes}?t=${Date.now()}`
      : `${BACKEND}${data.segmentation}?t=${Date.now()}`;
    previewSeg.src = boxesUrl;
    previewSeg.style.display     = 'block';
    segPlaceholder.style.display = 'none';

    // Show SAM2 masks image if DINO+SAM2 path ran
    if (data.seg_masks && previewSegMasks) {
      previewSegMasks.src          = `${BACKEND}${data.seg_masks}?t=${Date.now()}`;
      previewSegMasks.style.display = 'block';
    }

    // Load GLB + labels (labels are logged to console only, not used for splitting)
    const meshUrl   = `${BACKEND}${data.mesh}?t=${Date.now()}`;
    const labelsUrl = data.labels ? `${BACKEND}${data.labels}?t=${Date.now()}` : null;
    await loadGLB(meshUrl, labelsUrl);

    setStatus('Done! Drag to rotate, scroll to zoom.', 'ok');
  } catch (err) {
    setStatus(`Error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    loadingOverlay.classList.remove('visible');
    btnGenerate.disabled = false;
  }
});

// ─── GLB loading ─────────────────────────────────────────────────────────────
async function loadGLB(meshUrl, labelsUrl) {
  if (sceneGroup) {
    scene.remove(sceneGroup);
    sceneGroup = null;
  }

  const gltf = await new Promise((resolve, reject) => {
    new GLTFLoader().load(meshUrl, resolve, undefined, reject);
  });

  let sourceGeo  = null;
  let defaultMat = null;
  gltf.scene.traverse((child) => {
    if (child.isMesh && sourceGeo === null) {
      child.geometry.computeVertexNormals();
      sourceGeo  = child.geometry;
      defaultMat = child.material;
    }
  });

  if (!sourceGeo) throw new Error('No mesh found in GLB');

  // Fetch labels for console info only — not used for geometry splitting or materials
  if (labelsUrl) {
    try {
      const resp = await fetch(labelsUrl);
      if (resp.ok) {
        const labelsData = await resp.json();
        console.info('[VIEWER] label_counts:', labelsData.label_counts,
                     '| components:', labelsData.components);
      }
    } catch (e) {
      console.warn('[VIEWER] Could not load labels JSON:', e);
    }
  }

  // Render the raw TripoSG mesh with its own material (no overrides)
  const group = new THREE.Group();
  const mesh  = new THREE.Mesh(sourceGeo, defaultMat);
  mesh.castShadow    = true;
  mesh.receiveShadow = true;
  group.add(mesh);

  // Centre + scale
  const box    = new THREE.Box3().setFromObject(group);
  const centre = box.getCenter(new THREE.Vector3());
  const size   = box.getSize(new THREE.Vector3());
  const scale  = 1.6 / Math.max(size.x, size.y, size.z);
  group.position.sub(centre.multiplyScalar(scale));
  group.scale.setScalar(scale);

  scene.add(group);
  sceneGroup = group;

  controls.target.set(0, 0, 0);
  camera.position.set(0, 0.5, 2.2);
  controls.update();

  viewerPlaceholder.style.display = 'none';
}

// ─── Geometry splitter ────────────────────────────────────────────────────────
function splitGeometryByFaceLabels(geometry, faceLabels, numComponents) {
  try {
    const indexAttr    = geometry.index;
    const glbFaceCount = indexAttr
      ? Math.floor(indexAttr.count / 3)
      : Math.floor(geometry.attributes.position.count / 3);

    const nFaces = Math.min(glbFaceCount, faceLabels.length);

    // Count faces per component
    const faceCounts = new Int32Array(numComponents);
    for (let fi = 0; fi < nFaces; fi++) {
      faceCounts[Math.max(0, Math.min(faceLabels[fi], numComponents - 1))]++;
    }

    // Pre-allocate index arrays
    const indexArrays = Array.from(
      { length: numComponents },
      (_, i) => new Uint32Array(faceCounts[i] * 3)
    );
    const offsets = new Int32Array(numComponents);

    // Fill
    for (let fi = 0; fi < nFaces; fi++) {
      const label = Math.max(0, Math.min(faceLabels[fi], numComponents - 1));
      const off   = offsets[label];
      const base  = fi * 3;
      indexArrays[label][off]     = indexAttr ? indexAttr.getX(base)     : base;
      indexArrays[label][off + 1] = indexAttr ? indexAttr.getX(base + 1) : base + 1;
      indexArrays[label][off + 2] = indexAttr ? indexAttr.getX(base + 2) : base + 2;
      offsets[label] += 3;
    }

    return indexArrays.map((idxArr) => {
      if (idxArr.length === 0) return null;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', geometry.attributes.position);
      if (geometry.attributes.normal) geo.setAttribute('normal', geometry.attributes.normal);
      if (geometry.attributes.color)  geo.setAttribute('color',  geometry.attributes.color);
      if (geometry.attributes.uv)     geo.setAttribute('uv',     geometry.attributes.uv);
      geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
      return geo;
    });
  } catch (err) {
    console.error('[SPLIT] ERROR:', err);
    return Array(numComponents).fill(null);
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function setStatus(msg, type) {
  statusEl.textContent = msg;
  statusEl.className   = type;
}
