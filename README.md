# 🌟 RayTrophi - Advanced Real-Time Ray Tracing Engine

<div align="center">

![Version](https://img.shields.io/badge/version-1.0-blue.svg)
![C++](https://img.shields.io/badge/C++-20-00599C.svg?logo=c%2B%2B)
![Platform](https://img.shields.io/badge/platform-Windows-0078D6.svg?logo=windows)
![CUDA](https://img.shields.io/badge/CUDA-12.0-76B900.svg?logo=nvidia)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**A high-performance, production-ready ray tracing renderer with hybrid CPU/GPU rendering**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Performance](#-performance) • [Gallery](#-gallery)

</div>

---

## 📖 Overview

**RayTrophi** is a state-of-the-art physically-based ray tracing engine designed for architectural visualization, product rendering, and real-time graphics. It combines the flexibility of CPU rendering with the raw power of GPU acceleration through NVIDIA OptiX.

### 🎯 Key Highlights

- **Hybrid Rendering**: Seamlessly switch between CPU (Embree/Custom BVH) and GPU (OptiX) acceleration
- **Production-Ready**: Principled BSDF, advanced materials, volumetrics, subsurface scattering
- **High Performance**: Optimized BVH construction (<1s for 3.3M triangles), multi-threaded rendering
- **Real-time Preview**: Interactive UI with ImGui, live parameter adjustments
- **Industry Standard**: AssImp loader supports 40+ 3D formats (GLTF, FBX, OBJ, etc.)

---

## ✨ Features

### 🎨 Rendering Capabilities

- **Materials**
  - ✅ Principled BSDF (Disney-style uber-shader)
  - ✅ Lambertian, Metal, Dielectric
  - ✅ Volumetric rendering with noise-based density
  - ✅ Subsurface Scattering (SSS)
  - ✅ Clearcoat, Anisotropic materials
  
- **Lighting**
  - ✅ Point lights, Directional lights
  - ✅ Area lights (mesh-based)
  - ✅ Emissive materials
  - ✅ Environment/HDRI lighting
  - ✅ Soft shadows with multiple importance sampling

- **Advanced Features**
  - ✅ Depth of Field (DOF)
  - ✅ Motion Blur
  - ✅ Intel Open Image Denoise (OIDN) integration
  - ✅ Tone mapping & post-processing
  - ✅ Bone-based skeletal animation

### 🚀 Performance & Optimization

- **Multi-BVH Support**
  - Embree BVH (Intel, production-grade)
  - Custom ParallelBVH (SAH-based, OpenMP parallelized)
  - OptiX GPU acceleration structure

- **Optimizations**
  - SIMD vector operations
  - Multi-threaded tile-based rendering
  - Progressive refinement
  - Adaptive sampling
  - Memory-optimized triangle representation (146 bytes/triangle)

### 🖥️ User Interface

- Modern ImGui-based interface
- Real-time parameter tweaking
- Scene hierarchy viewer
- Material editor
- Performance metrics (FPS, rays/s, memory usage)
- Multiple render backend selection

---

## 🚦 Quick Start

### Prerequisites

**Required:**
- **Visual Studio 2022** (MSVC v143) - **RECOMMENDED BUILD SYSTEM**
- Windows 10/11 (x64)
- CMake 3.20+ (optional, VS2022 preferred)

**Optional (for GPU rendering):**
- NVIDIA GPU with RTX support
- CUDA Toolkit 12.0+
- OptiX 7.x SDK

### 📦 Dependencies

All dependencies are managed automatically:
- SDL2 (graphics output)
- Embree 4.x (CPU BVH)
- AssImp 5.x (model loading)
- ImGui (UI)
- OpenMP (parallelization)
- stb_image (texture loading)
- Intel OIDN (denoising)
- CUDA/OptiX (GPU rendering - optional)

### 🔨 Build Instructions

#### **Method 1: Visual Studio 2022 (RECOMMENDED)**

```bash
# 1. Clone the repository
git clone https://github.com/maxkemal/RayTrophi.git
cd RayTrophi/raytrac_sdl2

# 2. Open the solution
# Double-click raytrac_sdl2.vcxproj or open in Visual Studio 2022

# 3. Build
# Set configuration to "Release" and platform to "x64"
# Build > Build Solution (Ctrl+Shift+B)

# 4. Run
# The executable will be in: x64/Release/raytracing_render_code.exe
```

**Note**: All dependencies (DLLs, resources) are automatically copied to the output directory by the build system.

#### **Method 2: CMake (Known Issues - See below)**

```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

⚠️ **CMake Known Issue**: CPU rendering with SDL has a screen update bug. Use VS2022 .vcxproj build for stable CPU rendering.

### ▶️ Running

```bash
cd x64/Release
raytracing_render_code.exe
```

The UI will appear. Use File > Load Scene to import models (GLTF recommended).

---

## 🏗️ Architecture

### Project Structure

```
RayTrophi/
├── raytrac_sdl2/                  # Main project
│   ├── source/
│   │   ├── cpp_file/              # Implementation files
│   │   │   ├── Renderer.cpp       # Main rendering loop
│   │   │   ├── EmbreeBVH.cpp     # Embree BVH wrapper
│   │   │   ├── ParallelBVHNode.cpp # Custom SAH BVH
│   │   │   ├── OptixWrapper.cpp   # OptiX GPU backend
│   │   │   ├── AssimpLoader.cpp   # Model/texture loader
│   │   │   ├── PrincipledBSDF.cpp # Disney BSDF
│   │   │   └── ...
│   │   ├── header/                # Header files
│   │   │   ├── Ray.h, Vec3.h     # Math primitives
│   │   │   ├── Material.h         # Material base
│   │   │   ├── Triangle.h         # Optimized triangle
│   │   │   ├── Camera.h           # Camera & DOF
│   │   │   └── ...
│   │   ├── imgui/                 # ImGui library
│   │   └── res/                   # Resources (icons, etc.)
│   ├── raytrac_sdl2.vcxproj      # Visual Studio project
│   ├── CMakeLists.txt             # CMake build (has issues)
│   └── raygen.ptx                 # OptiX shader
└── README.md                      # This file
```

### Core Components

1. **Renderer** (`Renderer.cpp`)
   - Tile-based multi-threaded rendering
   - Progressive refinement
   - Denoising integration

2. **BVH Systems**
   - **EmbreeBVH**: Industry-standard, optimized for speed
   - **ParallelBVHNode**: Custom SAH-based, OpenMP parallel build
   - **OptiX BVH**: GPU-accelerated structure

3. **Material System** (`Material.h`, `PrincipledBSDF.cpp`)
   - Modular property-based materials
   - Texture support (albedo, roughness, metallic, normal, emission)
   - sRGB/Linear color space handling

4. **OptixWrapper** (`OptixWrapper.cpp`)
   - CUDA/OptiX backend
   - SBT (Shader Binding Table) management
   - Texture object caching

5. **AssimpLoader** (`AssimpLoader.cpp`)
   - Supports 40+ formats
   - Embedded texture extraction
   - Material conversion to Principled BSDF

---

## ⚡ Performance

### BVH Construction (3.3M Triangles)

| BVH Type       | Build Time | Quality | Use Case              |
|----------------|------------|---------|------------------------|
| Embree         | **872 ms** | High    | Production rendering   |
| ParallelBVH    | ~2000 ms   | High    | Custom research/debug  |
| OptiX (GPU)    | ~150 ms    | Very High | Real-time GPU        |

### Rendering Speed

- **CPU (Embree)**: ~1-5 million rays/s (16 threads)
- **GPU (OptiX RTX 3080)**: ~100-500 million rays/s
- **Memory**: 146 bytes/triangle (optimized layout)

### Optimizations Applied

- ✅ Direct Embree buffer writes (no intermediate vectors)
- ✅ Vector pre-allocation with `reserve()`
- ✅ Two-pass BVH construction (count → allocate → build)
- ✅ Embree build quality tuning (MEDIUM for speed)
- ✅ Material ID lookup via MaterialManager (no shared_ptr in Triangle)

---

## 🎨 Gallery

*(Add your rendered images here)*

---

## 🛠️ Building from Source - Detailed Guide

### Dependencies Setup

**Automatic (recommended):**
The Visual Studio project manages dependencies via vcpkg or manual paths.

**Manual:**
1. Download SDL2, Embree, AssImp from official sources
2. Update include/library paths in project properties

### Build Configurations

- **Debug**: Full symbols, slower (~10x)
- **Release**: Optimized, production use
- **RelWithDebInfo**: Optimized + symbols (profiling)

### CMake vs Visual Studio

| Feature                  | VS2022 .vcxproj | CMake         |
|--------------------------|-----------------|---------------|
| CPU Rendering (SDL)      | ✅ Working      | ⚠️ Has bugs   |
| GPU Rendering (OptiX)    | ✅ Working      | ✅ Working     |
| Dependency Management    | ✅ Excellent    | ⚠️ Manual     |
| Build Speed              | Fast            | Slower        |
| **Recommendation**       | **USE THIS**    | Experimental  |

**Why VS2022?**
- All dependencies are pre-configured
- Resource files (icons, PTX) auto-copied
- No SDL refresh bugs in CPU rendering
- Better debugging experience

---

## 📚 Usage Examples

### Basic Rendering

```cpp
#include "Renderer.h"
#include "SceneData.h"

int main() {
    Renderer renderer(1920, 1080, 8, 128);
    SceneData scene;
    OptixWrapper optix;
    
    // Load scene
    renderer.create_scene(scene, &optix, "path/to/model.gltf");
    
    // Render
    SDL_Surface* surface = /* ... */;
    renderer.render_image(surface, scene, /* ... */);
    
    return 0;
}
```

### Switching BVH Backend

```cpp
// Use Embree (fastest)
renderer.rebuildBVH(scene, true);  // use_embree = true

// Use custom ParallelBVH
renderer.rebuildBVH(scene, false); // use_embree = false
```

### Material Creation

```cpp
auto mat = std::make_shared<PrincipledBSDF>();
mat->albedoProperty.constant_value = Vec3(0.8, 0.1, 0.1); // Red
mat->roughnessProperty.constant_value = Vec3(0.3, 0.3, 0.3);
mat->metallicProperty.constant_value = Vec3(1.0, 1.0, 1.0); // Metallic
```

---

## 🐛 Known Issues & Limitations

### Build System
- ⚠️ **CMake build has SDL screen update bug in CPU rendering** → Use VS2022 instead
- DLL dependencies must be in same folder as .exe

### Rendering
- OptiX requires NVIDIA RTX GPU
- Very large scenes (>10M triangles) may cause memory issues
- Denoising requires Intel CPU (OIDN) or may be slow

### Platform
- Currently Windows-only (SDL2, DirectX dependencies)
- Linux/macOS support would require porting

---

## 🗺️ Roadmap

- [ ] Binned SAH for faster BVH construction
- [ ] Index-based BVH (remove vector copying)
- [ ] SBVH (Spatial BVH splits)
- [ ] Linux/macOS support
- [ ] Vulkan backend (alternative to OptiX)
- [ ] Network rendering (distributed ray tracing)
- [ ] USD format support
- [ ] Light path visualization/debugging

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Performance optimizations
- New material models
- Additional 3D format support
- Bug fixes
- Documentation improvements

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](source/LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Embree** - Intel's high-performance ray tracing kernels
- **OptiX** - NVIDIA's GPU ray tracing engine
- **AssImp** - Open Asset Import Library
- **ImGui** - Dear ImGui for user interface
- **SDL2** - Simple DirectMedia Layer
- **Intel OIDN** - Open Image Denoise
- **stb** - Sean Barrett's public domain libraries

---

## 👤 Author

**Kemal** - [@maxkemal](https://github.com/maxkemal)

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/maxkemal/RayTrophi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/maxkemal/RayTrophi/discussions)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ and lots of ☕

</div>
