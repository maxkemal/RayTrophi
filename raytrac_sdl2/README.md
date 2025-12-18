# RayTrophi - Advanced OptiX Ray Tracing Engine

A high-performance modular ray tracing engine built with NVIDIA OptiX 7, SDL2, and ImGui. Features real-time path tracing, volumetric clouds, and a comprehensive scene editor.

## 🚀 Key Features

### 🎨 Rendering Core
- **Hybrid Rendering:** CPU (Embree/Parallel BVH) and GPU (OptiX 7) support.
- **Materials:** Principled BSDF with support for Albedo, Metallic, Roughness, Normal, Transmission, and Emission maps.
- **Volumetrics:** 
  - Advanced parametric cloud system with dual layers.
  - 3D Noise-based volumetric fog and clouds.
  - Accurate light absorption and scattering.
- **Lighting:** Area, Spot, Point, and Directional lights with soft shadows.
- **Denoiser:** Integrated OIDN (Open Image Denoise) for clean previews.

### 🛠️ Scene Editor (New!)
- **Manipulators:** Maya/Blender style Gizmos for Translate, Rotate, Scale.
- **Undo/Redo System:** 
  - Robust history stack for **Object Deletion**, **Duplication**, and **Transformations**.
  - Smart memory management (limits separate for heavy geometry vs. lightweight transforms).
  - Shortcuts: `Ctrl+Z` (Undo), `Ctrl+Y` (Redo).
- **Object Management:**
  - Shift+Drag to Duplicate objects safely.
  - Delete objects with `Del` or `X`.
  - Hierarchy and Outliner panels.

## 🎮 Controls

### Camera
- **Orbit:** Middle Mouse Drag
- **Pan:** Shift + Middle Mouse Drag
- **Zoom:** Mouse Wheel or Ctrl + Middle Mouse Drag
- **Move:** Arrow Keys & PageUp/PageDown

### Scene Manipulation
- **Select:** Left Click
- **Gizmo Modes:**
  - `G` or `W`: Translate
  - `R`: Rotate
  - `S`: Scale
- **Actions:**
  - `Shift + Drag`: Duplicate Object
  - `Del` or `X`: Delete Object
  - `Ctrl + Z`: Undo
  - `Ctrl + Y`: Redo
  - `F12`: Trigger Final Render

## 📦 Recent Updates (v1.5)
- **Full Undo/Redo Implementation:** Added Command Pattern architecture to handle scene state reversal.
- **Smart History:** Differentiates between 'Heavy' operations (Geometry changes) and 'Light' operations (Transforms) to optimize RAM usage.
- **Crash Fixes:** Solved empty scene crashes by properly handling BVH/OptiX cleanup states.
- **OptiX Sync:** Fixed issues where GPU geometry wasn't updating correctly after deletion or undo.
- **UI Logic:** Added 'Edit' menu and cleaned up verbose logging for better performance.

## 🔧 Build Instructions
1. Open `raytrac_sdl2.sln` in Visual Studio 2022.
2. Ensure NVIDIA OptiX 7.x SDK and CUDA Toolkit are installed.
3. Build in `Release` mode for optimal performance.
4. Run `raytracing_render_code.exe`.

## 📜 License
This project is for educational and portfolio purposes.
