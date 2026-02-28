# RayTrophi - Quick Start & Reviewer Guide

Welcome to **RayTrophi**, a high-performance, open-source ray tracing engine and 3D studio. Thank you for taking the time to review this project.

This guide provides a quick overview of how to navigate the portable executable build and explore the engine's core capabilities.

## 🚀 1. Initial Setup & Navigation

### Starting the Engine
- Extract the `.zip` file into a dedicated folder.
- Run `raytracing_render_code.exe` (or `RayTrophi.exe`).
- **Wait briefly:** The engine will detect your hardware (CPU, NVIDIA GPU, Vulkan API) and initialize the corresponding rendering backends via our "Graceful Fallback" system.

### Basic Camera Navigation (Viewport)
- **Rotate:** Middle Mouse Button (Drag)
- **Pan:** Shift + Middle Mouse Button (Drag)
- **Zoom:** Mouse Wheel (or Ctrl + Middle Mouse Drag)
- **Move (FPS Style):** Use Arrow Keys or PageUp/PageDown

## 📦 2. Loading a Scene & Hardware Acceleration

### Importing Models
1. Go to **File > Import Model...** (or `Ctrl + I`)
2. Select a `.gltf` or `.obj` file from your local machine.
3. The engine uses a fully optimized multi-threaded loader.

### Switching Rendering Backends
RayTrophi is completely hardware-agnostic. You can switch rendering modes seamlessly in real-time:
1. In the **Properties Panel** (right side), expand the **Render Settings** section.
2. Look for the **Engine** dropdown:
   - **OptiX (GPU):** Fastest. Selected automatically if an NVIDIA RTX/GTX card is detected.
   - **Vulkan RT (GPU):** Experimental. Selected if you want cross-vendor hardware RT.
   - **CPU (Embree):** High-quality path tracing used as a smart fallback if no GPU is found.

*Note: Switching engines reconstructs the BVH (Bounding Volume Hierarchy) dynamically.*

## ✨ 3. Exploring Key Technical Features

### The Transform Gizmo (Idle Preview)
We implemented a non-blocking `Idle Preview` system to keep the UI perfectly smooth during complex BVH updates:
1. Select an object by clicking it in the Viewport.
2. Press `G` (Move), `R` (Rotate), or `S` (Scale).
3. **The Magic:** Drag the gizmo handle, but **do not release the mouse**. Pause your cursor for `0.3s`.
4. The engine will render a fast, temporary preview of the ray-traced result without committing the BVH rebuild!

### World & Atmospherics (Nishita Sky)
Explore our physically-based global volumetric cloud scattering system:
1. Open the **World** tab from the top menu (`View > World Tab`).
2. Adjust the **Sun Elevation** to see real-time Day/Night cycles with procedural stars/moon.
3. In the **Clouds** section, enable clouds and tweak the **Coverage**, **Density**, and **Silver Intensity** (Anisotropy). The clouds are fully ray-marched!

### Advanced Selection & Undo System
- **Box Selection:** Right-click and drag over multiple items to select them.
- **Mixed Selection:** You can select a combination of 3D objects and Light Sources simultaneously.
- **Undo/Redo:** We wrote a custom Command-Pattern Undo/Redo Engine. Press `Ctrl + Z` to instantly revert any transform, deletion, or light mutation in `O(1)` time complexity.

## 📚 4. Where to Find More Information

- **In-App Help:** Click **Help > Quick Guide & Shortcuts** (or press `F12` for Render Window, `F1` for Web Docs) inside the application.
- **Comprehensive Documentation:** Visit our Web Manual included in the GitHub repository or the `docs/` folder.
- **Source Code Architecture:** All backend scripts, BVH integrations, and procedural systems are open source at our [GitHub Repository](https://github.com/maxkemal/RayTrophi).

Thank you for reviewing RayTrophi!
