AI-Driven 2D to 3D Jewelry Generation with Real-Time Customization
Track: Agentic AI [Rezinix AI]

Link to the slides:
Link to the video:

JewelForge converts a single jewelry image into a customizable 3D model.

Users can upload a jewelry image (ring, pendant, etc.), automatically generate a 3D model, and customize materials such as metals and gemstones in an interactive viewer. The final design can be exported for rendering or manufacturing.

What It Does

Converts a 2D jewelry image into a 3D mesh

Automatically detects components like metal body and gemstones

Allows real-time customization of materials

Provides an interactive 3D viewer

Enables export of the final model

Pipeline
Input Image
     ↓
Background Removal
     ↓
Component Detection
     ↓
3D Model Generation
     ↓
Component Mapping onto Mesh
     ↓
Interactive Customization
     ↓
Export (GLB / STL)


How It Works:

Upload an image of a jewelry design.

1. The system processes the image and generates a 3D model.

2. Individual parts of the jewelry are identified (such as the metal body and gemstones).

3. Users can select components and change materials.

4. The customized model can be exported.


Streamlit UI

This repo now includes a simple Streamlit app that:

1. Accepts a 2D jewelry image upload.
2. Segments components (Grounding DINO + SAM2).
3. Runs TripoSG 2D to 3D generation and returns a GLB.
4. Shows segmented output and 3D rendered output in one interface.

Files:

- streamlit_app.py
- requirements.txt

Run:

1. Install dependencies:

     pip install -r requirements.txt

2. Start the app:

     streamlit run streamlit_app.py

3. In the sidebar, set TripoSG command with placeholders:

     python TripoSG/main.py --image {input} --output {output}

Notes:

- The command must write a GLB file to the {output} path.
- If TripoSG command is not set or fails, the app falls back to output.glb in repo root if present.
- SAM2 is optional in this UI. If SAM2 is unavailable, the app uses bounding-box masks as a fallback so you can still run end-to-end.
