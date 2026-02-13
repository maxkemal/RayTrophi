# 🦊 RayTrophi Hair & Fur System

A physically-based, high-performance hair and fur rendering system for RayTrophi Studio. It features a complete production pipeline from interactive grooming to cinematic Marschner BSDF rendering on both CPU and GPU (OptiX).

## 📁 File Structure

```
raytrac_sdl2/source/
├── include/Hair/
│   ├── HairStrand.h      # Data structures for hair strands
│   ├── HairSystem.h      # Core management, generation, and baking
│   ├── HairUI.h          # Modern icon-based vertical UI panel
│   └── HairBSDF.h        # Marschner BSDF model (CPU)
├── src/Hair/
│   ├── HairSystem.cpp    # CPU Grooming, Refitting, and I/O
│   └── HairGPUManager.cpp # GPU Accelerated grooming & deformation
└── src/Device/
    ├── hair_bsdf.cuh     # GPU BSDF (CUDA/OptiX)
    └── hitgroup_kernels.cu # OptiX Hair Shading & Root UV logic
```

## ✨ Features

### 🖌️ Interactive Grooming
- **Real-time Brushes:** Add, Remove, Comb, Cut, Length, Density, Clump, Puff, Wave, Frizz, and Smooth brushes.
- **Mirror Support:** Symmetry painting across X, Y, and Z axes.
- **Brush Falloff:** Quadratic and linear falloff modes for organic styling.
- **GPU Acceleration:** Optional GPU-accelerated grooming for low-latency feedback on millions of strands.

### 🦁 Strand Generation & Stylization
- **Guide + Children:** Professional workflow using master guide strands with dense interpolated children.
- **Root UV Mapping:** Inherit albedo, roughness, and other parameters from the emitter mesh's textures.
- **Advanced Physics:** Parametric control over Gravity, Clumpiness (Positive/Negative), Wave Frequency/Amplitude, and Frizz.

### 🎨 Cinematic Shading (Marschner BSDF)
- **Three-Lobe Model:** High-fidelity R, TT, and TRT lobes for realistic specular and internal scattering.
- **Color Modes:** Melanin (Pigmentation), Direct RGB, and Absorption modes.
- **Root-To-Tip Variation:** Gradual transition of material properties along the fiber length.

## 🚀 Style Library (Presets)
RayTrophi includes a curated selection of presets to jumpstart your project:
- **Human Hair:** Blonde, Brown, Black, and Red hair with fine-tuned melanin values.
- **Environment:** Luxury Carpet, Wild Grass, Dense Moss.
- **Creature:** Velvet Fur, Long Wolf Fur.

## 🔧 Workflow: Integration

### 1. Attaching Hair to a Mesh
```cpp
Hair::HairSystem hairSystem;
// Generate on a selected mesh using presets or custom params
hairSystem.generateOnMesh(emitterTriangles, params, "main_groom");
```

### 2. Updating for Animation
```cpp
// Refit strands to match underlying mesh deformation (Skinning/Rigid)
hairSystem.updateTransforms(emitterMesh);
// Sync to GPU (OptiX)
renderer.uploadHairToGPU();
```

## 🔗 Optimization Tips
1. **Guide Ratio:** Use a lower number of guides (e.g., 5,000) with higher child counts (e.g., 10-20) for efficiency.
2. **GPU Refit:** Enable GPU refitting in the Grooming panel when dealing with high-density character animations.
3. **LOD:** Reduce `pointsPerStrand` for background objects to save memory.

---
*Part of RayTrophi Studio by Kemal Demirtaş*
