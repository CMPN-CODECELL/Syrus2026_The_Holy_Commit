"""
JewelForge Streamlit App — 2D jewelry image → segmentation → 3D model.

Run from the repository root:
    python -m streamlit run streamlit_app.py

Requirements:
    pip install -r requirements.txt
    pip install -r TripoSG/requirements.txt

Hardware:
    A CUDA-capable GPU is required for TripoSG 3D generation and RMBG
    background removal.  Grounding DINO and SAM2 run on CPU when no GPU is
    available so segmentation still works, but 3D generation will be very slow
    (or fail) without a GPU.
"""

import logging
import os
import sys
import tempfile

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — make TripoSG sub-packages importable
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
TRIPOSG_DIR = os.path.join(REPO_ROOT, "TripoSG")
SCRIPTS_DIR = os.path.join(TRIPOSG_DIR, "scripts")
for _p in (TRIPOSG_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Serialise CUDA kernel launches (avoids illegal-memory-access on some GPUs during debugging)
# Only set if not already defined to allow the caller to override.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
# Reduce VRAM fragmentation on ≤ 8 GB GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Cached model loading (loaded once per Streamlit session / process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models — this may take several minutes on first run…")
def _load_models():
    """
    Download (if needed) and load all heavy models:
    - RMBG-1.4 (background removal, used by TripoSG pre-processing)
    - TripoSGPipeline (3D generation)
    - Grounding DINO (component detection)
    - SAM 2 (mask segmentation)

    Models are cached so they are loaded only once per process.
    """
    import torch
    from huggingface_hub import snapshot_download
    from briarmbg import BriaRMBG
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from utils.segment_ring import load_dino, load_sam2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TripoSG requires float16 on GPU to fit within ≤ 8 GB VRAM.
    dtype  = torch.float16 if device == "cuda" else torch.float32

    triposg_weights = os.path.join(TRIPOSG_DIR, "pretrained_weights", "TripoSG")
    rmbg_weights    = os.path.join(TRIPOSG_DIR, "pretrained_weights", "RMBG-1.4")

    if not os.path.isfile(os.path.join(triposg_weights, "model_index.json")):
        st.info("Downloading TripoSG weights (~2 GB) from HuggingFace…")
        snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights)

    if not os.path.isfile(os.path.join(rmbg_weights, "config.json")):
        st.info("Downloading RMBG-1.4 weights from HuggingFace…")
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights)

    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights).to(device)
    rmbg_net.eval()

    pipe = TripoSGPipeline.from_pretrained(triposg_weights).to(device, dtype)

    # DINO + SAM2 are pre-loaded on CPU; run_pipeline moves them to GPU only
    # during the segmentation step, then moves them back to free VRAM for TripoSG.
    dino_processor, dino_model = load_dino("cpu")
    sam2_model = load_sam2("cpu")

    return {
        "pipe":           pipe,
        "rmbg_net":       rmbg_net,
        "dino_processor": dino_processor,
        "dino_model":     dino_model,
        "sam2_model":     sam2_model,
        "device":         device,
    }


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="JewelForge — 2D → 3D",
    page_icon="💎",
    layout="wide",
)

st.title("💎 JewelForge — Image → Segmentation → 3D")
st.caption(
    "Upload a jewelry image. The app will detect components with "
    "Grounding DINO + SAM 2, then generate a 3D mesh with TripoSG."
)

# Two-column layout: controls on the left, results on the right
col_left, col_right = st.columns([1, 2])

with col_left:
    uploaded_file = st.file_uploader(
        "Upload jewelry image (JPG / PNG / WebP)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    st.subheader("Generation settings")
    seed         = st.slider("Seed",             min_value=0,   max_value=9999, value=42)
    num_steps    = st.slider("Inference steps",  min_value=10,  max_value=100,  value=50,
                             help="More steps → higher quality, slower generation")
    guidance     = st.slider("Guidance scale",   min_value=1.0, max_value=20.0, value=7.0,
                             step=0.5,
                             help="Higher values follow the image more strictly")

    run_btn = st.button(
        "🚀 Generate 3D Model",
        disabled=(uploaded_file is None),
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_btn and uploaded_file is not None:
    # Lazy-import here so the module-level code never imports torch directly,
    # allowing the page to render even when torch is not yet installed.

    # --- Load (or reuse cached) models -------------------------------------
    with st.spinner("Preparing models…"):
        try:
            models = _load_models()
        except Exception as exc:
            st.error(f"❌ Failed to load models: {exc}")
            logging.exception("Model loading failed")
            st.stop()

    # --- Save the uploaded bytes to a temporary file -----------------------
    suffix     = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    tmp_input  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_input.write(uploaded_file.getvalue())
    tmp_input.close()
    image_path = tmp_input.name

    output_dir = tempfile.mkdtemp(prefix="jewelforge_")

    # --- Run pipeline ------------------------------------------------------
    with col_right:
        with st.spinner(
            "Running segmentation + 3D generation — this can take 1–3 min on GPU…"
        ):
            try:
                from pipeline.run_pipeline import run_pipeline
                result = run_pipeline(
                    image_path=image_path,
                    output_dir=output_dir,
                    pipe=models["pipe"],
                    rmbg_net=models["rmbg_net"],
                    seed=seed,
                    num_steps=num_steps,
                    guidance_scale=guidance,
                    dino_processor=models.get("dino_processor"),
                    dino_model=models.get("dino_model"),
                    sam2_model=models.get("sam2_model"),
                )
            except Exception as exc:
                st.error(f"❌ Pipeline failed: {exc}")
                logging.exception("Pipeline error")
                st.stop()
            finally:
                # Clean up temporary input file
                try:
                    os.unlink(image_path)
                except OSError:
                    pass

        st.success("✅ Generation complete!")

        # --- Show segmentation images ------------------------------------
        st.subheader("Segmentation")

        seg_boxes = result.get("seg_boxes") or ""
        seg_masks = result.get("seg_masks") or result.get("segmentation") or ""

        if seg_boxes and os.path.isfile(seg_boxes):
            st.image(seg_boxes, caption="Grounding DINO — bounding box detections",
                     use_container_width=True)
        if seg_masks and os.path.isfile(seg_masks):
            st.image(seg_masks, caption="SAM 2 — segmentation masks",
                     use_container_width=True)

        if not (seg_boxes and os.path.isfile(seg_boxes)) and \
           not (seg_masks and os.path.isfile(seg_masks)):
            st.info("No segmentation images were produced.")

        # --- 3D model download button ------------------------------------
        st.subheader("3D Model")
        mesh_path = result.get("mesh") or ""
        if mesh_path and os.path.isfile(mesh_path):
            with open(mesh_path, "rb") as mf:
                st.download_button(
                    label="⬇️ Download 3D Model (.glb)",
                    data=mf.read(),
                    file_name="jewelry_3d.glb",
                    mime="model/gltf-binary",
                    use_container_width=True,
                )
            st.info(
                "Open the downloaded `.glb` file in "
                "[model-viewer.dev](https://modelviewer.dev/editor/) or Blender "
                "to view and customise the 3D model."
            )
        else:
            st.warning("3D mesh was not generated.")

        # --- Component labels (optional) ---------------------------------
        labels_path = result.get("labels") or ""
        if labels_path and os.path.isfile(labels_path):
            import json
            with open(labels_path) as lf:
                labels_data = json.load(lf)
            st.subheader("Component label summary")
            st.json({
                "components":    labels_data.get("components", []),
                "face_count":    labels_data.get("face_count", 0),
                "label_counts":  labels_data.get("label_counts", {}),
            })
