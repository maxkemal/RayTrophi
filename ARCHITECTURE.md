# 📊 RayTrophi Studio: Technical Specifications & Architecture Report
**Report Date:** October 11, 2026
**Status:** Active Development / Multi-Backend GPU Stylize Phase
**Author:** Kemal Demirtaş (maxkemal)

---

## 🛠️ Project Overview
RayTrophi Studio is a high-performance, modular **Path Tracing** engine designed for professionals. It features a dual-backend architecture supporting **NVIDIA OptiX (CUDA)** and **Vulkan Ray Tracing**, integrating advanced terrain erosion physics, skeletal animation systems, and a professional-grade node-based editor.

---

## 📉 1. Codebase Statistics (Verified)
*Measured from the repository on October 11, 2026. Third-party vendor libraries (`simdjson`, `nlohmann/json`, `miniz`, `stb`, `tinyexr`, `imgui`, `JoltPhysics`, `ozz-animation`, `nanovdb`) and runtime data/assets (JSON configs, templates, presets, 3D meshes) are explicitly separated from first-party code.*

### 🛠️ First-Party Code (Engine & Ecosystem)

| Layer / Component | File Types | Files | Lines of Code | Net LoC (Non-blank) | Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **C++ Core Engine & UI** | `.h`, `.hpp`, `.inl`, `.cpp` | 1,180 | 541,459 | 492,365 | Engine architecture, scene graphs, Vulkan/OptiX pipelines, UI & editors |
| **CUDA Kernels** | `.cu`, `.cuh` | 31 | 18,135 | 15,648 | GPU hardware-accelerated physics, MPM, and simulation kernels |
| **Shaders (Vulkan / RT / Compute)** | `.comp`, `.glsl`, `.vert`, `.frag`, `.rchit`, etc. | 364 | 43,216 | 39,858 | Path tracing shaders, compute simulation passes, post-processing |
| **RayTrophiAgent** | Python (`.py`) | 17 | 1,699 | 1,474 | Autonomous AI agent core, IPC tooling, and multi-provider orchestration |
| **Tooling, IPC & Test Scripts** | `scripts/` (`.py`, `.ps1`) | 364 | 57,027 | 51,665 | External named-pipe IPC test probes, verification harnesses, build scripts |
| **Root Utilities** | `.hpp`, `.py` | 3 | 202 | 185 | Root helpers and IPC verification scripts |
| **TOTAL (First-Party Code)** | — | **1,959** | **661,738** | **601,195** | **Engine-owned first-party implementation** |

### 📦 Third-Party & Vendor Libraries (Excluded from Project LoC)

| Library | Location | Files | Lines of Code | Description |
| :--- | :--- | :---: | :---: | :--- |
| **simdjson** | `source/include/simdjson.h`, `src/simdjson.cpp` | 2 | 248,846 | High-throughput JSON parser |
| **nlohmann/json** | `source/include/json.hpp` | 2 | 49,533 | Modern C++ JSON single-header library |
| **miniz / miniz_zip** | `external/miniz*` | 10 | 9,021 | Deflate / Zip archive compression library |
| **stb_image & write** | `source/include/stb_image*` | 2 | 9,598 | Single-header image loading & writing |
| **tinyexr** | `source/include/tinyexr.h` | 1 | 9,405 | Single-header OpenEXR loading/saving |
| **ImGuizmo & UI Extras** | `source/include/ImGuizmo*` | 2 | 3,486 | 3D transform gizmo implementation |
| **PNanoVDB** | `source/shaders/PNanoVDB.*` | 3 | 7,236 | NVIDIA NanoVDB shader integration |
| **IconsMaterialDesign** | `source/include/IconsMaterialDesignIcons.h` | 1 | 7,461 | Generated icon codepoint mapping |
| **ImGui & ImSequencer** | `external/imgui/` | 18 | 67,153 | Immediate mode graphical user interface |
| **JoltPhysics** | `external/JoltPhysics/` | 444 | 103,636 | Rigid body & collision physics library |
| **ozz-animation** | `external/ozz-animation/` | 523 | 732,125 | Skeletal animation and blending runtime |
| **nanovdb** | `libs/nanovdb/` | 191 | 63,869 | Sparse volumetric data structures |
| **ufbx & cgltf** | `external/ufbx/`, `external/cgltf/` | 4 | 46,556 | FBX and glTF 2.0 import parsers |
| **TOTAL (Third-Party)** | — | **1,203** | **1,358,325** | *Third-party dependency footprint* |

### 📂 Structural Breakdown (First-Party Engine Source)
*   **UI & Editor Systems (~109.2k LoC):** Modern docking shell, node editors, timeline widgets, viewport overlay, mesh-paint, and inspector panels.
*   **Physics, Simulation & Particles (~100.0k LoC):** Hydraulic/thermal LEM erosion, granular MPM solver, fluid dynamics, gas/smoke, ocean water systems, and VDB volumes.
*   **API, IPC & Scripting (~71.6k LoC):** Canonical Python scripting layer, external named-pipe IPC server, and dynamic node serialization.
*   **Backend & Hardware Rendering (~46.2k LoC):** Vulkan RT pipeline, rasterizer preview, OptiX backend, memory heaps, and AI denoising (OIDN) integration.
*   **Shaders & Compute Passes (~43.3k LoC):** SPIR-V / GLSL path tracer, compute shaders for fluid/particle simulations, and post-processing kernels.
*   **Render Core & Scene Representation (~34.2k LoC):** Flat TriangleMesh DNA SoA geometry, Embree BVH acceleration, light sampling, and camera orchestration.
*   **Device Abstraction & Architecture (~19.3k LoC):** Device limits, queue management, descriptor binding, and fence synchronization.

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
*   **[cgltf](https://github.com/jkuhlmann/cgltf):** glTF 2.0 / GLB parsing, used by both the importer and the exporter.
*   **[ufbx](https://github.com/ufbx/ufbx):** FBX import, including skinning and baked transform animation.
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
1.  **Scalability:** Experience managing a large multi-backend graphics codebase with a 600k+ line engine-owned source/shader surface.
2.  **Full-Stack Graphics:** Proficiency in both low-level GPU kernels and high-level UI/UX logic.
3.  **Complex Engineering:** Deep integration of physics, math, and professional software architecture.

---
**Note:** Generated via automated codebase analysis and architectural scan.
