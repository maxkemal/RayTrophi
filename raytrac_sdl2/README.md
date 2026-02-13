# RayTrophi Studio - The Open-Source Cinematic World Building Engine

**RayTrophi** is a high-performance, modular world-building and rendering engine powered by **NVIDIA OptiX 7 (RTX)**. Designed to bridge the gap between creative freedom and high-end cinematic rendering, it offers a comprehensive suite of tools for terrain generation, fluid simulations, procedural hair, and advanced animation.

![RayTrophi Studio Display](RayTrophi_image.png)

---

## 🌟 Core Pillars

### 🦁 Procedural Hair & Fur System (NEW)
*   **Integrated Grooming:** Advanced grooming tools with support for guide strands and interpolated children.
*   **Interactive Hair Painting:** Real-time brush system for Adding, Removing, Combing, Cutting, and Clumping hair.
*   **Root UV Mapping:** Dynamically sample textures from underlying emitters (scalp/skin) for seamless integration.
*   **Physics & Styles:** Built-in support for Clumping, Waves, Frizz, and Gravity with a deep preset library.

### 🌊 Environmental Simulation
*   **Node-Based Terrain (V2):** Generative noise nodes (Perlin/Worley) coupled with **Hydraulic Erosion** for realistic geology.
*   **Advanced Water System:** 
    *   **FFT Ocean:** High-fidelity deep water simulation with interactive whitecaps.
    *   **Spline Rivers:** Create flowing water bodies with custom flow-maps, turbulence, and depth control.
*   **Foliage Painting:** Instant displacement of millions of instanced trees, rocks, and grass with optimized GPU culling.

### 🌪️ Volumetrics & Gas Dynamics
*   **GPU Gas Solver:** Real-time fluid simulation (Smoke, Fire) with interactive **Force Fields** (Vortex, Turbulence, Point).
*   **VDB Support:** High-speed import and rendering of industry-standard `.vdb` and `.nvdb` (NanoVDB) sequences.
*   **Cinematic Shading:** Multi-Scattering, Dual-Lobe phase functions, and Blackbody radiation for hyper-realistic fire.

### 🎬 Animation & Logic
*   **AnimGraph:** State-machine-based logic for complex character movement (Idle to Run blending).
*   **Timeline Editor:** Integrated keyframe animation for objects, lights, cameras, and even atmospheric properties.
*   **Skinned Mesh:** High-performance GPU skinning for animated characters and creatures.

---

## 🎨 Rendering Excellence

*   **Wavefront Path Tracer:** Optimized for NVIDIA RTX hardware using OptiX 7.x.
*   **Hybrid Fallback:** Seamlessly switch to CPU rendering (Intel Embree) for systems without RTX.
*   **Materials:** Full **Disney Principled BSDF** implementation, plus specialized shaders for Glass, Thin-Film, and Hair.
*   **Post-Processing:** Real-time Tonemapping (AGX, ACES, Filmic), Gamma control, and **Intel OIDN** denoising.

---

## 🛠️ Getting Started

### Prerequisites
*   **OS:** Windows 10/11
*   **GPU:** NVIDIA RTX recommended (OptiX 7+ support)
*   **Tools:** Visual Studio 2022, CUDA Toolkit 12.x

### Build Instructions
1.  Clone the repository: `git clone https://github.com/maxkemal/RayTrophi.git`
2.  Open `raytrac_sdl2.sln` in Visual Studio 2022.
3.  Install dependencies via `vcpkg` (Assimp, SDL2, OpenVDB, NanoVDB, ImGui, Intel OIDN).
4.  Set configuration to **Release** and build solution.

---

## 🎮 Navigation & Shortcuts

| Action | Shortcut |
| :--- | :--- |
| **Orbit Camera** | Middle Mouse Drag |
| **Focus Selection** | `F` |
| **Gizmo Move/Rot/Scale** | `G` / `R` / `S` |
| **Interactive Paint** | `P` (Toggle in Hair/Terrain panels) |
| **Final Render (FHD)** | `F12` |
| **Undo / Redo** | `Ctrl+Z` / `Ctrl+Y` |

---

## 📜 Roadmap & License
Developed and Maintained by **Kemal DEMİRTAŞ (maxkemal)**.

RayTrophi is open-source and intended for artists, developers, and researchers. 
*   **License:** MIT Core. (Check individual 3rd party licenses for OptiX, OIDN, etc.)
*   **Goal:** To become the go-to open-source studio for cinematic ray-traced content.

---
*Inspired by the beauty of light and the complexity of nature.*
