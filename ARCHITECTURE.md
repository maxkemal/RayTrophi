# 📊 RayTrophi Studio: Technical Specifications & Architecture Report
**Report Date:** May 23, 2026
**Status:** Active Development / Multi-Backend GPU Stylize Phase
**Author:** Kemal Demirtaş (maxkemal)

---

## 🛠️ Project Overview
RayTrophi Studio is a high-performance, modular **Path Tracing** engine designed for professionals. It features a dual-backend architecture supporting **NVIDIA OptiX (CUDA)** and **Vulkan Ray Tracing**, integrating advanced terrain erosion physics, skeletal animation systems, and a professional-grade node-based editor.

---

## 📉 1. Codebase Statistics (Verified)
*Measured from the repository on May 23, 2026. Counts cover `raytrac_sdl2/source`; project counts exclude vendored single-file libraries (`simdjson`, `stb`, `json.hpp`, `tinyexr`).*

| Layer | Files | Lines of Code | Description |
| :--- | :---: | :---: | :--- |
| **Project Code + Shaders** | 327 | 207,957 | Engine-owned C++, CUDA, Vulkan GLSL, and OptiX shader sources. |
| **GPU Kernels + Shader Files** | 52 | 21,125 | CUDA kernels, OptiX PTX sources, Vulkan RT shaders, and compute shaders. |
| **Full Source Tree** | 333 | 467,165 | Project code plus embedded single-file libraries in `raytrac_sdl2/source`. |
| **Vendored External Trees** | External | Not counted here | Large third-party trees such as vcpkg, Ozz, and NanoVDB are intentionally excluded from the verified project LoC. |

### 📂 Structural Breakdown (Source Focus)
*   **UI & Editor (53.6k LoC):** Modern shell, timeline widgets, hierarchy, material, terrain, mesh-paint, and property editors.
*   **Backend Layer (19.9k LoC):** Vulkan RT, OptiX, backend abstraction, and viewport backend integration.
*   **Physics & Nodes (19.0k LoC):** Hydraulic/Thermal erosion, terrain nodes, VDB processing, gas, fluid, and ocean systems.
*   **Render Core (16.3k LoC):** Embree BVH integration, renderer orchestration, OptiX wrapper, light sampling, and acceleration managers.

---

## 🕹️ 2. User Interaction & Complexity
*Quantifying the control-surface available to the end-user.*

### ⌨️ UI Control Points
*   **Numerical Inputs (Sliders/Drag):** 512 (Fine-tuning lights, materials, terrain, physics, and stylize profiles).
*   **Action Buttons:** 218 (Command execution, tools, file ops, bake/apply workflows).
*   **Selectors (Checkbox/Combo/Menu/List):** 514 (Feature toggles, algorithm switches, mode and preset selection).
*   **Text/Value Inputs:** 57.
*   **Color Controls:** 30.
*   **Tree/Collapsing Headers:** 110.
*   **Total Interaction Elements:** **1,278+ Active Control Points.**

### 🧩 Dynamic Graph Systems
The engine provides **61+ unique node types** for visual programming:
*   **Terrain Graph (36+ nodes):** Advanced erosion workflows and mask generation.
*   **Animation Graph (14+ nodes):** State machines, IK blend spaces.
*   **Material Graph (11+ nodes):** Procedural PBR shader construction.

---

## 🏗️ 3. Technology Stack
Built upon industry-standard frameworks and hardware-accelerated APIs:

*   **[NVIDIA OptiX](https://developer.nvidia.com/optix):** Hardware-accelerated Ray Tracing.
*   **[Vulkan SDK](https://www.vulkan.org/):** Modern cross-platform graphics & compute.
*   **[Intel Embree](https://github.com/embree/embree):** High-performance CPU ray-tracing kernels.
*   **[Assimp](https://github.com/assimp/assimp):** Robust 3D model import (40+ formats).
*   **[NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb):** GPU-friendly sparse volume representation.
*   **[Intel OIDN](https://www.openimagedenoise.org/):** Professional-grade AI denoiser. Integrated with dual support:
    *   **CPU Mode:** High-performance denoising using Intel TBB and ISPC kernels.
    *   **CUDA Mode:** GPU-accelerated denoising for real-time high-quality preview on NVIDIA hardware.
*   **[ImGui](https://github.com/ocornut/imgui):** Bloat-free immediate mode graphical user interface.

---

## 🎮 Render Devices & Workload Distribution
To prevent developer and AI agent confusion, the rendering backend strategy is defined as follows:
*   **Vulkan RT (Default Render Device):** Our primary and default path tracing / render device on the GPU.
*   **Vulkan Raster (Preview & Edit Mode):** Core editing, sculpting, and painting operations run in raster mode by default.
*   **OptiX (Secondary Render Device):** Positioned as the secondary/alternative path tracing engine on the GPU for now.
*   **Intel Embree (CPU Core):** Forms the core engine structure for CPU-based ray tracing and scene BVH management.

---

## 🚀 Why This Matters
For developers and employers, this report demonstrates:
1.  **Scalability:** Experience managing a large multi-backend graphics codebase with a 200k+ line engine-owned source/shader surface.
2.  **Full-Stack Graphics:** Proficiency in both low-level GPU kernels and high-level UI/UX logic.
3.  **Complex Engineering:** Deep integration of physics, math, and professional software architecture.

---
**Note:** Generated via automated codebase analysis and architectural scan.
