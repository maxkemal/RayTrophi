# RayTrophi Studio

<div align="center">

![Version](https://img.shields.io/badge/status-active%20development-orange.svg)
![C++](https://img.shields.io/badge/C++-20-00599C.svg?logo=c%2B%2B)
![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D6.svg?logo=windows)
![Backends](https://img.shields.io/badge/render-CPU%20%7C%20OptiX%20%7C%20Vulkan%20RT-76B900.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**An open-source 3D content-creation suite built around a hybrid CPU/GPU path tracer.**

Model, sculpt, paint, groom, simulate, light, animate, and render — in one application.

[![RayTrophi Showcase](https://img.youtube.com/vi/-xRiPhc-p6k/maxresdefault.jpg)](https://www.youtube.com/watch?v=-xRiPhc-p6k)
**[▶️ Watch the showcase on YouTube](https://www.youtube.com/watch?v=-xRiPhc-p6k)**

[What it is](#-what-it-is) • [Workspaces](#-workspaces) • [Rendering](#-rendering--backends) • [Simulation](#-physics--simulation-suite) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Gallery](#-gallery)

</div>

---

## 📖 What it is

**RayTrophi Studio** began as a path-tracing renderer and has grown into a full **digital-content-creation (DCC) application**. It is a single desktop program where you can build a scene from scratch — polygon modeling and sculpting, texture painting, hair grooming, terrain and vegetation, fluid/gas/whitewater simulation, ocean and rivers — and render it with a physically-based path tracer that runs on three interchangeable backends (CPU, NVIDIA OptiX, and Vulkan Ray Tracing).

It is not a render farm plugin or a library. It is an interactive editor with a modern docked UI, an animation timeline, undo/redo across every tool, project save/load, and a non-destructive art-direction (Stylize) layer on top of the converged image.

### Design goals

- **One application, full pipeline.** Geometry authoring, look-dev, FX, animation, and final-frame rendering live in the same scene, the same `.rtp`/`.rts` project, the same undo stack.
- **Three render backends, one feature set.** Switch between CPU (Embree), OptiX, and Vulkan RT without changing the scene. The Vulkan path is the recommended interactive backend; OptiX and CPU remain first-class.
- **Physically-based, but art-directable.** Principled BSDF + spectral hair + volumetrics + DCC-grade fluids, with a Stylize layer that can repaint the result into oil/ink/toon looks without touching the underlying physics.
- **Honest about its state.** This is an active solo project. Where a subsystem is experimental or in progress, it says so.

> **Status:** active development. There is no versioned release yet; the `main` branch is the current build.

---

## 📊 Project at a glance
<!-- STATS_START -->
| Metric | Value |
| :--- | :--- |
| **Project code / shader lines** | ~259,000 |
| **Project code / shader files** | 360+ |
| **GPU kernel & shader files** | 56 (CUDA, OptiX PTX, Vulkan GLSL/RT, compute) |
| **UI control points** | 1,278+ |
| **Render backends** | CPU (Embree) · NVIDIA OptiX · Vulkan RT |
| **Node systems** | Terrain (66), Animation (14+), Material (11+) |
| **Last verified** | 2026-07-22 — full build, local IPC and remote TLS/token/capability checks |
<!-- STATS_END -->

Counts cover `RayTrophiStudio/source` and exclude vendored single-file libraries (`simdjson`, `stb`, `json.hpp`, `tinyexr`).

Full technical report: **[ARCHITECTURE.md](ARCHITECTURE.md)** · Türkçe: **[README_TR.md](README_TR.md)**

---

## 🧭 Workspaces

RayTrophi Studio is organized into task-focused workspaces that all operate on the same live scene:

| Workspace | What you do there |
|-----------|-------------------|
| **Layout / Scene** | Import assets, place and transform objects/lights/cameras, build hierarchy, box-select, gizmo-edit |
| **Modeling** | Polygon editing (extrude, inset, bevel, loop cut, weld, merge, UV unwrap), modifier stack |
| **Sculpt** | Brush-based surface sculpting on meshes and terrain (PBVH-accelerated) |
| **Paint** | Layered PBR texture painting directly on the mesh (multi-channel, blend modes) |
| **Terrain** | Sculpt + non-destructive terrain/biome graph, erosion, hydrology, snow/glaciers, named fields, heightmap I/O |
| **Foliage / Scatter** | Biome-aware node layers, Asset Library + scene sources, rule-based and hand-painted GPU instancing |
| **Hair** | Groom, comb, cut/grow, simulate, and render hair & fur |
| **Simulation** | Liquid (APIC/FLIP), gas/smoke/fire, whitewater, rigid + soft bodies & cloth (Jolt), mesh/primitive colliders, force fields, emitters |
| **Animation** | Multi-track timeline, per-channel keyframing, skeletal animation, animation graph |
| **Render / Look-dev** | Pick a backend, set sampling/quality, denoise, tonemap, and Stylize the result |

---

## 🎛️ Rendering & backends

A single physically-based path tracer feeds three acceleration backends. The scene, materials, and lights are identical across all three — you choose the backend that fits the moment (CPU for headless/no-GPU, OptiX for NVIDIA curve hardware, Vulkan RT for fast interactive look-dev).

### Materials & shading
- **Principled BSDF** (Disney-style uber-shader): albedo, roughness, metallic, specular, clearcoat, sheen, anisotropy, transmission/IOR
- **Lambertian, Metal, Dielectric** classic models
- **Subsurface scattering (SSS)**
- **Interior Volume** (resin / glass-marble interiors): Beer-Lambert depth absorption over an opaque base or inside see-through glass (all three backends), plus a procedural interior system (Vulkan RT): deterministic DDA-traversed inclusions — sharp dirt specks, air bubbles, and colored glass shards (round chips or elongated faceted crystals that flash as the object turns) — and styled dust clouds (nebula, two-color billow, wispy streaks, domain-warped paint-swirl "ink in water"), all with object/world anchoring, curated interior presets, and no extra scene rays. Shard colors carry into photon caustics for stained-glass patterns. Technical note: [docs/INTERIOR_VOLUME.md](docs/INTERIOR_VOLUME.md)
- **Spectral / melanin-based hair BSDF**
- **Volumetric rendering** with NanoVDB sparse volumes and procedural noise density
- Full texture support (albedo, roughness, metallic, normal, emission, transmission, opacity) with sRGB/linear handling

### Lighting & sky
- Point, directional, spot, and **mesh-based area lights**; emissive materials
- **HDR/EXR environment maps** (equirectangular)
- **Nishita physical sky** with day/night cycle, procedural stars & moon (phases, horizon magnification, atmospheric dimming), sun glow, and automatic sun↔directional-light sync
- **Global volumetric clouds** (Henyey-Greenstein scattering, adaptive ray marching, coverage/density/altitude/wind controls, soft horizon fade) — works over HDRI, solid color, or Nishita sky
- Soft shadows with multiple importance sampling (MIS)

### Photon caustics & volumetric light shafts
- **Progressive photon-mapped caustics** (Vulkan RT): a light-side photon pass reuses the same RT pipeline and BSDFs as the camera path, splatting refracted (LS⁺D) energy into a world-anchored hash grid that accumulates across frames — glass focus spots, ring caustics, and colored-glass patterns that plain path tracing practically never converges
- **Spectral dispersion carries into caustics for free**: the same stochastic hero-wavelength transport used by camera rays rides along with photons, producing rainbow fringes in both the surface pattern and the shafts
- **Volumetric light shafts without a volume object**: photons also deposit energy along their flight segments into a second, coarser world grid; a bounded camera march turns it into visible in-scatter. A glass object and a light are enough — no fog or participating-media setup. Includes scatter-strength (virtual dust density), optional 3D-turbulence density modulation, and a direct-shaft mode where the light→glass leg glows too (point lights switch to mixture-sampled emission so they stay omnidirectional)
- Camera-independent world-space targeting (photons aim at the live union bounds of transmissive objects, re-evaluated per frame), scale-aware grid sizing, smooth cone-kernel and trilinear density reads
- Honest limits: Vulkan RT only for now (OptiX/CPU ports planned), isotropic phase in the shafts, sharpness bounded by grid resolution

### Sampling & post
- Progressive **accumulative path tracing** with **adaptive sampling** (focuses samples on noisy regions)
- Depth of field, motion blur
- **Intel Open Image Denoise (OIDN)** — CPU and CUDA-accelerated paths, viewport and final
- Tone mapping and post-processing

### Backend comparison

<details>
<summary>⚡ <b>Feature parity: OptiX vs Vulkan RT</b> (expand)</summary>

| Feature | OptiX | Vulkan RT | Notes |
|---------|:-----:|:---------:|-------|
| Principled BSDF | ✅ | ✅ | Full parity |
| Lambertian / Metal / Dielectric | ✅ | ✅ | Full parity |
| Subsurface Scattering (SSS) | ✅ | ✅ | Minor colour tint difference |
| Clearcoat & Anisotropic | ✅ | ✅ | Full parity |
| Volumetric rendering (NanoVDB) | ✅ | ✅ | Persistent leaf-cache accessor; equal or faster than OptiX interactively |
| **Hair system** | ✅ | ✅ | Analytical LSS intersection + LSS-tight AABBs; outperforms OptiX hardware curves here |
| HDR / EXR environment | ✅ | ✅ | Full parity |
| Nishita sky & day/night | ✅ | ✅ | Full parity |
| Volumetric clouds | ✅ | 🧪 | Minor scattering differences |
| Water / Ocean (FFT) | ✅ | 🧪 | Wave reflection differences |
| Skeletal animation (GPU skinning) | ✅ | ✅ | Vulkan compute shader |
| Depth of field / motion blur | ✅ | ✅ | Full parity |
| Soft shadows (MIS) / area lights | ✅ | ✅ | Full parity |
| Tone mapping & post-FX | ✅ | ✅ | GPU compute tonemap on Vulkan, fused into the trace command buffer |
| OIDN denoising | ✅ | ✅ | OptiX has the tighter CUDA-interop path |
| Adaptive / progressive render | ✅ | ✅ | Vulkan converges faster (lower per-frame overhead) |
| Stylize layer | ✅ | ✅ | CPU / Vulkan / OptiX produce matched output |
| Photon caustics + volumetric light shafts | ❌ | ✅ | Vulkan-only for now; photon pass shares the camera RT pipeline |

> **Legend:** ✅ full support &nbsp;|&nbsp; 🧪 supported, minor output differences possible

</details>

<details>
<summary>📈 <b>Interactive benchmarks (measured)</b> (expand)</summary>

Same scene, same settings, same hardware, camera in motion. These are interactive-viewport frame rates, not final-frame numbers. On static scenes adaptive sampling pushes both backends well past 500 fps as pixels converge.

| Scene | Vulkan RT | OptiX | Ratio |
|---|:---:|:---:|:---:|
| Mesh-heavy + Nishita atmosphere | 600 fps | 50 fps | 12.0× |
| Hair-heavy (cubic B-spline strands, LSS intersection) | 300 fps | 70 fps | 4.3× |
| Volume / VDB cloud (Fast preset) | 300 fps | 200 fps | 1.5× |
| Volume / VDB cloud (Balanced preset) | comparable | comparable | ≈1.0× |
| Volume / VDB cloud (Exact preset, camera moving) | 16 fps | 23 fps | 0.7× |

**Why Vulkan leads interactively:** async fence-based ping-pong frame pipeline (no per-frame `vkQueueWaitIdle`), GPU compute tonemap into small RGBA8 staging, analytical Linear-Swept-Sphere hair intersection, persistent NanoVDB read-accessor across march steps, and a lean kernel without per-pixel accumulation atomics in the hot path.

**Where OptiX still wins:** the Exact volume preset during camera motion (hardware CUDA NanoVDB texture path), CUDA-native zero-copy OIDN interop, and final stills that specifically need NVIDIA's curve hardware primitives.

</details>

---

## 🌀 Physics & simulation suite

A multi-threaded grid- and particle-based FX suite with CUDA and CPU backends, integrated directly into the path-traced render pipeline. Multiple simulation domains, emitters, colliders, rigid bodies, and force fields coexist in one workspace and are saved with the project.

### Liquid — APIC / FLIP solver
- Hybrid **APIC/FLIP** solver with adjustable blending, preserving angular momentum and minimizing numerical dissipation
- **MAC staggered grid** with PCG + MIC(0) preconditioned pressure solve (CPU) and a Jacobi-PCG / multigrid (MGPCG) pressure solve on the GPU
- **Variational (cut-cell) solid coupling** (Batty/Bridson): fractional MAC-face weights give sub-grid-accurate collisions against analytic primitives, and moving colliders impart real momentum/splash through the pressure solve
- **Ghost-fluid 2nd-order free surface** (Gibou/Enright): sub-cell level set removes the voxel "staircase" on the liquid surface
- Keyframe-animated colliders are re-posed per sub-step so the fluid tracks moving geometry
- Adaptive resolution, open/closed boundary modes, dynamic particle reseeding to prevent leaks, fluid material presets (Water, Oil, Custom)

### Gas, smoke & fire
- Multi-threaded dense-grid solver for temperature, soot, and fuel density
- Combustion dynamics (ignition, heat release, flame dissipation) with procedural FBM curl-noise turbulence
- Sparse-VDB active-voxel Poisson solve for efficient large domains

### Whitewater (Ihmsen et al. 2012)
Secondary **spray** (airborne), **foam** (surface), and **bubbles** (submerged) generated from trapped-air and wave-crest potentials, and advected through the solver with full collider response:
- **Dynamic PBR material routing** — transmissive droplets for spray, scattering rough-white PBR for foam, silvery semi-transmissive bubbles — with a *Custom Material Override* to bind any scene material
- **Underwater bubble TIR correction** to reduce total-internal-reflection dark-circle artifacts
- **Newton-Raphson wave snapping** projects surface foam onto the smoothed level-set water mesh, eliminating floating foam on wavy water
- Deterministic hash-based size variation and smooth dissolve near end-of-life
- Adjustable icosphere subdivision (0–3) for close-up detail

### Bodies — Jolt Physics: rigid, soft & cloth + two-way fluid coupling
- **Jolt Physics**-backed body solver on the shared simulation timeline: tag any scene object **Static, Dynamic, or Kinematic**, with **box / sphere / capsule / oriented-box** primitives fitted to its bounds, or a **mesh collider** that uses the object's real geometry — an exact triangle mesh for static bodies and a convex hull for moving ones, so a collider driven by an SDF/mesh source collides against the true shape instead of an OBB
- **Soft bodies & cloth.** Tag a mesh as a deformable **soft body** or **cloth** (Jolt soft-body solver): per-body stiffness/compliance, pressure (closed-volume inflation), damping, iterations, vertex collision thickness, and **vertex pinning** (hold rest vertices fixed to hang cloth from corners/edges); the deformed mesh is written straight back for rendering
- Per-body **mass or auto-mass-from-density, linear & angular damping, friction, restitution, gravity scale, initial linear/angular velocity, sleep, and per-axis translation/rotation locks**
- **Force fields drive every body kind** — rigid (force at the COM), soft & cloth (per-vertex velocity push, pinned vertices excluded)
- **Two-way fluid coupling.** A body is voxelized into the liquid/gas grid as a moving solid through the variational cut-cell path, so it pushes and splashes the fluid; in return, **buoyancy and linear/angular drag** sampled from the fluid level set act back on the body — float, sink, and bob driven by the same field the renderer reads
- **Kinematic** bodies are keyframe-driven (animated colliders that stir the fluid); **Dynamic** bodies are owned by the solver, so the timeline never fights the simulated pose
- **Selective re-bake.** Editing or moving a body only drops the (expensive) fluid bake when that body actually couples to a fluid domain — an unrelated static prop re-simulates on its own (cheap) while the liquid cache is preserved

### Surfaces, caching & serialization
- **Yu-Turk anisotropic surface reconstruction** with Laplacian smoothing for the render-time liquid mesh; surface resolution decoupled from the sim grid
- **SimCache disk baking** — bake heavy liquid/foam/gas frames to binary `.simcache` files next to the project and scrub the timeline in real time without re-simulating
- Full serialization of simulation state, domain settings, custom materials, timeline caches, and presets into `.rtp` / `.rts`

> The GPU MGPCG pressure path is live; the GPU port of variational solids + ghost-fluid (Stage 2) is in progress, as are surface tension, implicit viscosity, and narrow-band/sparse performance work for full DCC parity.

---

## 🛠️ Procedural & authoring tools

### 🏔️ Terrain, biome & hydrology graph

- Real-time sculpting brushes (raise, lower, smooth, flatten, stamp) plus 16-bit heightmap import/export for World Machine / Gaea workflows
- **Terrain Nodes V2** — a serialized, non-destructive graph with 66 registered nodes, live property editing, previews, grouping, reusable setups, and Apply-to-scene output
- **Generation & shaping** — heightmap/hardness inputs, procedural noise, fault, mesa, shear, terrace, smooth, normalize, resample, remap, blend, math, clamp, overlay, and screen operations
- **Erosion & geology** — hydraulic, thermal, fluvial, and wind erosion; sediment deposition, alluvial fans, deltas, wetness, soil depth, lithology, and strata synthesis
- **Terrain analysis & named fields** — slope, height, curvature, flow, exposure, watershed, concavity, convexity, valley, and wetness data can be published once and reused by downstream surface, biome, and foliage nodes
- **Biome Composer** — Temperate Mixed, Lush Valleys, Alpine Tundra, Arid Highlands, and Boreal Mountains presets generate normalized Forest / Grass / Rock / Alpine masks and a packed biome splat map
- **Hydrology** — watershed analysis, river networks and hydraulics, river-bed carving, spline output, lake-basin detection, lake surfaces, channels, shoreline, and foam masks
- **Climate & snow** — climate, snowfall, settling, melt/freeze, relative snow line, glacier flow, and snow/surface composition nodes
- **Fast setup actions** — build organized biome-field and biome-foliage graph branches automatically, while keeping every generated node editable

### 🌿 Node-based biome foliage & scatter

- **Foliage Layer → Foliage Set / Biome → Foliage Output** keeps every vegetation class as an independent rule, adds non-destructive set-level density/seed controls, and can scatter all connected layers when the terrain graph is applied
- Placement rules include target count, seed, minimum spacing, slope and elevation ranges, plus named density, exclusion, and scale fields with adjustable thresholds/influence
- Each layer accepts weighted multiple sources from **Recommended**, **Asset Library**, or **Scene Objects**; search, biome/type filtering, duplicate protection, thumbnails, hover previews, and compact per-source editing are shared by the Terrain UI and node properties
- The Asset Library scans model metadata first and loads geometry lazily into a shared cache. User vegetation and rock assets added under `RayTrophiStudio/assets` automatically join the same catalog and recommendation workflow
- Library assets use portable relative references, report missing files, are planted from their mesh base, and can target a real-world height with per-source variation calculated from the source bounding box
- Vegetation defaults to **World Y-Up**; **Follow Slope** and normal influence remain available per source for grass, rocks, or intentionally terrain-aligned assets
- The Terrain UI and foliage nodes are synchronized views of the same `InstanceGroup` data: add, remove, weight, scale/height, orientation, and layer-rule edits propagate both ways without deleted sources being restored by stale graph data
- Existing scatter workflows remain available: GPU-instanced grass/trees/rocks, collision-aware procedural placement, hand-painted detail, and global dynamic wind (strength, direction, gust)

### 💇 Hair & fur
- GPU simulated and rendered; analytical LSS intersection on Vulkan
- Grooming brushes: comb, cut/grow, smooth
- Physics: strands collide with meshes and respond to gravity/forces
- Melanin-based hair BSDF

### 🌊 Ocean & 🏞️ rivers
- **FFT ocean** with foam generation, caustics, and depth-based underwater volumetrics
- **Spline/bezier rivers** with auto-carving into terrain, flow mapping, and flow-driven object drift

### 🗿 Modeling, sculpt & paint
- **Edit Mesh mode** — extrude, inset, bevel, loop cut, delete/merge/weld/split, flip normal, smart re-triangulation, UV auto-unwrap/smart packer
- **Sculpt mode** — Grab, Draw, Inflate, Layer, Clay, Clay Strips, Pinch, Smooth, Flatten, Scrape, Crease; Shift→Smooth, Ctrl→invert; X/Y/Z mirror; PBVH pruning for dense meshes; modifier-stack subdivision; shared mesh/terrain sculpt path
- **Mesh Paint** — layered PBR painting (Base Color, Normal, Roughness, Metallic, Emission, Mask, Transmission, Opacity); paint/erase/soften/stamp/fill/clone/spray brushes; per-layer stack with Normal/Add/Multiply/Screen/Overlay blend modes; height-to-normal baking; dirty-region GPU updates; serialized into the project as PNG blobs
- Full **undo/redo** across all edit modes with optional step grouping; mesh edits propagate to CPU/GPU buffers and export as GLB with modifiers applied

### 🎨 Stylize — non-destructive art direction
A post-convergence layer that reads the path-traced result + AOV buffers and restyles the image without changing scene geometry, materials, or lights. Domain-masked compositing keeps sky, material, outline, and world effects separate.
- **Sky layer** — view-ray-locked stylized gradients, cloud banks, and sun (Painterly Clouds, Cartoon Cel, Sunset Bands, Ink Wash, Clear Gradient)
- **Painterly material layer** — surface-locked stroke fields (no screen-space swimming), palette influence, edge respect, pigment thickness, and a Wet Oil model (Body/Load/Pickup/Deposit/Buildup)
- **Outline layer** — depth/normal/material-discontinuity edges with Ink, Oil, Pencil, Dry Brush, and Pressure line types
- **Profiles** — Painterly Oil, Gouache, Ink + Wash, Graphic Toon, Clay/Maquette, Dreamy Sunset
- **Backend parity** — CPU, Vulkan compute (`stylize.comp`), and OptiX CUDA (`StylizeKernel.cu`) produce matched output; re-applies without resetting accumulation

### 🖥️ Viewport shading
- Vulkan raster **Solid + Matcap** mode for fast sculpt/paint feedback (drop matcaps in `raytrac_sdl2/assets/matcaps/`)
- Ray-traced interactive preview on any backend, with idle-preview during gizmo manipulation

---

## 🎞️ Animation & UI

- **Multi-track timeline & Graph Editor** with group hierarchy (Objects / Lights / Cameras / World), independent Bezier curve channel animation (Position, Rotation, Scale, Light settings, Camera parameters, and PBR Material properties), resizable splitter panel, collapsible group headers with bulk visibility toggles, and shortcut-driven keyframe editing (drag handles, delete/fit shortcuts).
- **Skeletal animation** with quaternion interpolation and GPU compute skinning; **animation graph** (14+ nodes) for state machines and IK blend spaces
- **Batch / sequence rendering** — export animation to image sequences (with material keyframes), cancellable mid-render, simulation-driven per frame
- Modern **ImGui** docked dark UI, render quality presets (Low/Medium/High/Ultra), dynamic resolution scaling, scene hierarchy, material editor, performance metrics
- **Selection** — box select (right-drag), mixed light+object selection, Ctrl+click add/remove, select all/none, multi-object transform
- **Undo/redo** for transforms, deletion, duplication, lights — Ctrl+Z / Ctrl+Y

### 📦 Asset browser & library
- Metadata-driven discovery for `model`, `anim_clip`, `vdb`, and `vdb_sequence`
- Built-in project `assets` root plus user-added local libraries
- Asset cards with preview/thumbnail cache, favorites, tags, saved collections and smart folders
- Drag-and-drop placement with viewport ghost preview and auto-selection
- Project-scoped UI persistence for layout, library list, and filters

---

## Python automation & secure IPC

RayTrophi includes an embedded Python 3.11 automation layer (`rt`, API version `0.5.0`) and a
transport-independent JSON IPC protocol. Both surfaces drive the same live scene through the
main-thread command queue and preserve undo/redo semantics.

- Scene/object transforms, primitives, duplication and import
- Materials, mesh attributes, lights, camera, world, post-processing and timeline/keyframes
- Final-frame and sequence render jobs with status/cancel
- Node graphs and parameters, modifiers, scatter and physics
- Fluid/gas, terrain/river, hair, layered PBR paint and deterministic sculpt automation
- Embedded Python console/workspace, live API reflection, addons and event callbacks
- Local same-user Windows Named Pipe plus optional private TLS 1.2/1.3 control plane
- Persistent hashed token vault, capabilities, expiry/revoke/rotation, IPv4 CIDR and canonical
  workspace/export-root policies
- Connection/session registry, bounded audit log, rate limits and **View → Remote IPC Control**
  management panel

The TLS listener is intended for loopback, trusted LAN or VPN use—not direct public-Internet
exposure. Internet integrations belong behind a separate OIDC/mTLS gateway. See
[IPC security and performance](docs/IPC_SECURITY_PERFORMANCE.md),
[gateway boundary](docs/REMOTE_IPC_GATEWAY.md), and the
[API & scripting roadmap](docs/API_SCRIPTING_ROADMAP.md).

Regression clients live in `scripts/`: `rt_api_smoke_test.py`, `ipc_test_client.py`,
`remote_ipc_client.py`, and `remote_ipc_security_test.py`.

---

## 🚦 Quick Start

### Prerequisites

**Required**
- **Visual Studio 2022** (MSVC v143) — recommended build system
- Windows 10/11 (x64)
- CMake 3.20+ (optional; VS2022 preferred)

**Optional (GPU rendering)**
- NVIDIA GPU (SM 5.0+): GTX 9xx/10xx/16xx or RTX series
- CUDA Toolkit 12.0+, OptiX 7.x/8.x SDK
- Vulkan SDK 1.3+ (for the Vulkan RT path)

| GPU Series | Architecture | Mode | Performance |
|------------|--------------|------|-------------|
| RTX 40xx | Ada Lovelace | Hardware RT | ⚡ Fastest |
| RTX 30xx | Ampere | Hardware RT | ⚡ Very fast |
| RTX 20xx | Turing | Hardware RT | ⚡ Fast |
| GTX 16xx | Turing | Compute | 🔶 Good |
| GTX 10xx | Pascal | Compute | 🔶 Moderate |
| GTX 9xx | Maxwell | Compute | 🔶 Slower |

### Environment variables

The project resolves dependencies via system environment variables. Set these to your local install paths before building:

| Variable | Description | Example |
|----------|-------------|---------|
| `SDL2_ROOT` | SDL2 root | `E:\...\SDL2-2.30.4` |
| `OPTIX_ROOT` | OptiX SDK | `C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0` |
| `EMBREE_ROOT` | Embree root | `E:\...\embree-4.4.0.x64.windows` |
| `OIDN_ROOT` | Intel OIDN root | `E:\...\oidn-2.3.0.x64.windows` |
| `ASSIMP_ROOT` | Assimp root | `E:\...\Assimp` |
| `CUDA_PATH` | CUDA Toolkit | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x` (usually auto-set) |
| `VULKAN_SDK` | Vulkan SDK | `C:\VulkanSDK\1.3.xxx.0` |

Managed dependencies: SDL2, Embree 4.x, Assimp 5.x, ImGui, OpenMP, stb_image, TinyEXR, Intel OIDN, NanoVDB, and CUDA/OptiX (optional).

### Build

**Visual Studio 2022 (recommended)**
```bash
git clone https://github.com/maxkemal/RayTrophi.git
cd RayTrophi
# Open RayTrophiStudio.sln in Visual Studio 2022
# Set Release | x64, then Build > Build Solution (Ctrl+Shift+B)
# Output: x64/Release/RayTrophiStudio.exe
```
All dependencies (DLLs, PTX, shaders, resources) are copied to the output directory automatically.

**CMake**
```bash
cmake -S RayTrophiStudio -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j 12
# Output: build/bin/RELEASE/RayTrophiStudio.exe
```
CMake keeps its executable, PTX, Vulkan shaders, and runtime DLLs isolated under `build/bin/<CONFIG>` so it never overwrites the VS2022 `x64` output.

### Run
Launch the executable; the docked UI appears. Use **File → Load Scene** to import a model (GLTF recommended; 40+ formats via Assimp).

---

## 🏗️ Architecture

```
RayTrophi/
└── RayTrophiStudio/
    └── source/
        ├── src/
        │   ├── Core/        # Entry point (Main.cpp), project management
        │   ├── Render/      # Renderer, OptiX wrapper, Embree/Parallel BVH, camera, textures
        │   ├── Backend/      # Vulkan RT, OptiX, viewport backends, scene texture manager
        │   ├── Scene/        # Objects, lights, materials, instancing, mesh, BSDFs
        │   ├── Physics/      # Fluid (APIC/FLIP), gas, whitewater, terrain, ocean, river, sim world
        │   ├── Device/       # CUDA kernels (.cu/.cuh), OptiX device code, Vulkan compute
        │   ├── Hair/         # Hair system, strands, skinning, hair BSDF
        │   ├── Paint/        # Mesh & terrain paint adapters, layer stack
        │   ├── Stylize/      # Stylize CPU/CUDA kernels and state
        │   ├── Animation/    # Animation controller, nodes, Ozz runtime
        │   ├── Api/          # Python rt facade, JSON IPC, security, sessions, audit and panel
        │   ├── Viewport/     # Viewport scene sync
        │   ├── Math/         # Vec/Matrix/Quaternion
        │   ├── UI/           # ImGui panels, timeline, gizmos, editors
        │   └── Utils/        # Loaders, serialization, helpers
        └── include/          # Headers (Backend, Core, Fluid, Hair, NodeSystem, Paint, Stylize, Viewport, Utils)
```

**Render backends**
- **EmbreeBVH** (`Render/EmbreeBVH.cpp`) — Intel CPU kernels
- **ParallelBVHNode** (`Render/ParallelBVHNode.cpp`) — custom SAH BVH, OpenMP-parallel build
- **OptixWrapper** (`Render/OptixWrapper.cpp`, `Device/*.cu`) — CUDA/OptiX, SBT + texture-object caching
- **VulkanBackend** (`Backend/VulkanBackend.cpp`) — `VK_KHR_ray_tracing_pipeline`, TLAS/BLAS refit, compute skinning, async ping-pong frame pipeline, GPU tonemap

**Node systems** (`include/NodeSystem/`) — graph core shared by the Terrain, Animation, and Material editors.

---

## 🎨 Gallery

[![RayTrophi Showcase](https://img.youtube.com/vi/-xRiPhc-p6k/maxresdefault.jpg)](https://www.youtube.com/watch?v=-xRiPhc-p6k)
**[▶️ Watch the full demo reel](https://www.youtube.com/watch?v=-xRiPhc-p6k)**

<div align="center">

<img src="render_samples/1.png" width="800" alt="Complex architectural scene"><br>
<i>Complex architectural scene — 3.3M triangles, Embree BVH</i>

<img src="render_samples/indoor2.png" width="800" alt="Interior design"><br>
<i>Interior with volumetric lighting and subsurface scattering</i>

<img src="render_samples/output1.png" width="800" alt="OptiX GPU rendering"><br>
<i>GPU path tracing with OptiX</i>

<img src="render_samples/stylesed_winter_dragon1.png" width="800" alt="Stylized dragon"><br>
<i>Stylized render via the non-destructive Stylize layer</i>

<img src="render_samples/RayTrophi_cpu1.png" width="800" alt="CPU rendering"><br>
<i>Pure CPU path tracing with progressive refinement</i>

<img src="render_samples/yelken.png" width="800" alt="Outdoor scene"><br>
<i>Outdoor environment with the Nishita physical sky</i>

</div>

---

## 🗺️ Roadmap

**Recently shipped**
- ✅ Vulkan RT backend (interactive primary) with GPU skinning, async ping-pong pipeline, analytical LSS hair
- ✅ Physics & particle simulation suite (APIC/FLIP liquid, gas/fire, whitewater)
- ✅ Rigid, soft-body & cloth dynamics (Jolt Physics) — primitive/mesh colliders, vertex pinning, force-field coupling, two-way fluid coupling (solid voxelization + buoyancy/drag), selective fluid re-bake
- ✅ GPU MGPCG fluid pressure solve (CUDA)
- ✅ Variational cut-cell solid coupling + ghost-fluid 2nd-order free surface (CPU)
- ✅ Multi-material whitewater PBR routing + Newton-Raphson wave snapping
- ✅ SimCache on-disk frame baking + full simulation serialization
- ✅ Stylize layer with CPU / Vulkan / OptiX parity
- ✅ Sculpt mode (mesh + terrain) and layered mesh paint
- ✅ Progressive photon-mapped caustics + spectral dispersion + volumetric light shafts (Vulkan RT)

**Planned / in progress**
- [ ] Caustics on OptiX / CPU; anisotropic phase & real density fields (VDB) for the light shafts
- [ ] GPU port of variational solids + ghost-fluid surface (Stage 2)
- [ ] Fluid surface tension, implicit viscosity, narrow-band/sparse performance
- [ ] Binned SAH / index-based BVH / SBVH spatial splits
- [ ] USD format support
- [ ] Network / distributed rendering
- [ ] Light-path visualization & debugging
- [ ] Linux / macOS support (currently Windows-only: SDL2 + Windows dependencies)

---

## 🐛 Known limitations

- **Windows-only** today (SDL2 + Windows dependencies); Linux/macOS would require porting.
- **OptiX** needs an NVIDIA GPU (SM 5.0+); RTX uses hardware RT cores, GTX uses compute (slower).
- Very large scenes (>10M triangles) can stress memory.
- CMake and VS2022 use **separate output folders** — keep them separate to avoid mixing stale PTX/DLLs.
- Vulkan volumetric clouds and FFT ocean show minor output differences vs OptiX (see parity table).
- GPU fluid pressure is live, but variational solids + ghost-fluid surface are CPU-only for now.

---

## 🤝 Contributing

Contributions are welcome — performance work, new material/FX models, format support, bug fixes, and docs.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request

---

## 📝 License

MIT License — see [LICENSE.txt](LICENSE.txt).

Third-party libraries and SDKs remain under their own licenses. See the **Third-party components** section in [LICENSE.txt](LICENSE.txt) for Jolt Physics, Assimp, Dear ImGui, ozz-animation, Intel OIDN, Embree, OptiX/CUDA, Vulkan, SDL2, JSON libraries, stb, TinyEXR, NanoVDB/OpenVDB, miniz, and related notices.

## 🙏 Acknowledgments

**Embree** (Intel CPU ray tracing) · **OptiX** (NVIDIA GPU ray tracing) · **Vulkan** · **Jolt Physics** (rigid-body physics) · **Assimp** (asset import) · **ImGui** (UI) · **SDL2** · **Intel OIDN** (denoising) · **NanoVDB** (sparse volumes) · **Ozz-animation** (skeletal animation) · **stb** · **TinyEXR**

## 👤 Author

**Kemal Demirtaş** — [@maxkemal](https://github.com/maxkemal)

- **Issues:** [GitHub Issues](https://github.com/maxkemal/RayTrophi/issues)
- **Discussions:** [GitHub Discussions](https://github.com/maxkemal/RayTrophi/discussions)

---

<div align="center">

**⭐ Star the repo if RayTrophi Studio is useful to you.**

Made with ❤️ and lots of ☕

</div>
