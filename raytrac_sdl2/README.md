# RayTrophi - Advanced OptiX & Hybrid Ray Tracing Engine

RayTrophi is a high-performance, modular ray tracing engine built with **NVIDIA OptiX 7**, **SDL2**, **ImGui**, and **OpenVDB (NanoVDB)**. It bridges the gap between real-time preview and offline path tracing, offering advanced features like volumetric rendering, node-based terrain generation, and a complete animation timeline.

![RayTrophi](RayTrophi_image.png)

## 🚀 Key Features

### 🔥 Volumetric Rendering (VDB)
- **OpenVDB / NanoVDB Support:** Import standard `.vdb` files and sequences.
- **Sequence Playback:** Real-time playback of volumetric animations (explosions, smoke, fire).
- **GPU Path Tracing:** Fully accelerated volumetric rendering on NVIDIA GPUs.
- **Blackbody Emission:** Physically accurate temperature-based emission (Fire/Explosion).
- **Hybrid Support:** Falls back to CPU rendering if GPU is unavailable.

### 🎬 Animation System
- **Timeline & Keyframing:** Animate Objects, Lights, Cameras, and World properties.
- **Graph Editor:** Node-based animation control.
- **Animation Render Mode:** Batch render image sequences with `render_Animation` loop.
- **Skinned Mesh Support:** Basic character animation (CPU skinning -> GPU upload).

### 🌍 Terrain & Environment
- **Terrain Node System (V2):** 
  - Graph-based terrain generation (Perlin, Erosion, Hydraulic, Thermal).
  - **AutoSplat:** Slope/Height based texturing.
  - **Splat Maps:** Export generated masks to PNG.
- **Water System:**
  - **FFT Ocean:** Real-time deep ocean simulation.
  - **Gerstner Waves:** Shoreline/Lake wave simulation.
  - **River Editor:** Bezier-spline based river placement tools.
- **Atmosphere:**
  - Nishita Sky Model (Spectral Day/Night Cycle).
  - Volumetric Fog & God Rays.
  - Dual-Lobe Cumulus Clouds.

### 🖌️ Scene Editor & Tools
- **Scatter Brush:** Paint foliage/instances directly onto terrain surfaces.
- **Terrain Brush:** Sculpt and paint terrain height/features in real-time.
- **Gizmos:** Blender-style 3D manipulators (Translate, Rotate, Scale).
- **Undo/Redo:** Robust command history for all scene operations.
- **Asset Management:** GLTF/GLB import support with materials.

### 🎨 Rendering Core
- **Hybrid Engine:** 
  - **GPU:** OptiX 7 (RTX Accelerated) Path Tracing.
  - **CPU:** Intel Embree / Parallel BVH Fallback.
- **Materials:** Principled BSDF (Disney), Glass, Metal, Emission, Volumetric.
- **Denoiser:** OIDN (Open Image Denoise) integration.

## 🎮 Controls

### Viewport Navigation
- **Orbit:** Middle Mouse Drag
- **Pan:** Shift + Middle Mouse Drag
- **Zoom:** Mouse Wheel or Ctrl + Middle Mouse Drag
- **Focus:** `F` (Focus on selected object)

### Tools & Edit
- **Select:** Left Click
- **Gizmo Modes:** `G` (Grab), `R` (Rotate), `S` (Scale)
- **Duplicate:** `Shift + Drag`
- **Delete:** `Del` or `X`
- **Undo/Redo:** `Ctrl+Z` / `Ctrl+Y`
- **Play Animation:** `Space`

### Rendering
- **Final Render:** `F12`
- **Animation Render:** (Via Render Panel)

## 🔧 Build Instructions
1. **Requirements:**
   - Visual Studio 2022
   - NVIDIA Driver (Latest)
   - CUDA Toolkit 11.x or 12.x
   - OptiX 7.x SDK
2. **Setup:**
   - Open `raytrac_sdl2.sln`
   - Ensure `vcpkg` dependencies are installed (SDL2, ImGui, Assimp, OIDN, OpenVDB/NanoVDB).
3. **Build:**
   - Select `Release` configuration.
   - Build Solution (`Ctrl+Shift+B`).
4. **Run:**
   - Launch `raytracing_render_code.exe` (or from VS Debugger).

## 📜 License
Developed by **Kemal DEMİRTAŞ**.
This project is for educational and portfolio purposes.
