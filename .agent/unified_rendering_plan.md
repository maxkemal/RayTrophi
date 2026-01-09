# Unified Rendering System - Implementation Plan

## Status: PHASE 1 COMPLETE ✅

The CPU `ray_color` function has been updated to use unified rendering functions
that match the GPU implementation exactly.

## Overview

This document describes the unified rendering system that ensures CPU/GPU render parity by sharing common BRDF, material, and light calculations.

## Completed Work

### ✅ Created Unified Header Files (include/)

1. **unified_types.h** - Platform-agnostic core types
   - `Vec3f`, `Vec2f` - Platform-independent vectors (work on CPU and GPU)
   - `UnifiedMaterial` - Common material structure (96 bytes, 16-byte aligned)
   - `UnifiedLight` - Common light structure (80 bytes)
   - `UnifiedHitResult` - Common hit record
   - `UnifiedConstants` - Shared rendering constants (MAX_CONTRIBUTION, RR params, etc.)

2. **unified_brdf.h** - Shared BRDF calculations
   - `D_GGX()` - GGX normal distribution
   - `G_Smith()` - Smith geometry term
   - `F_Schlick()` - Fresnel approximation
   - `evaluate_brdf_unified()` - Full Cook-Torrance BRDF
   - `pdf_brdf_unified()` - PDF for importance sampling
   - `power_heuristic()` - MIS weight calculation
   - `importance_sample_ggx()` - GGX importance sampling
   - `russian_roulette_probability()` - RR survival calc
   - `background_factor()` - Background contribution per bounce


3. **unified_light_sampling.h** - Shared light sampling
   - `compute_light_pdf()` - Light sampling PDF
   - `spot_light_falloff()` - Spot light cone falloff
   - `sample_light_direction()` - Sample direction toward light
   - `calculate_light_contribution_unified()` - Direct lighting with MIS
   - `pick_smart_light_unified()` - Importance-based light selection

4. **unified_converters.h** - Type conversion functions
   - `toVec3f()` / `toVec3()` - Vec3 conversions
   - `toUnifiedMaterial()` - PrincipledBSDF -> UnifiedMaterial
   - `toUnifiedLight()` - Light -> UnifiedLight
   - `toGpuMaterial()` - UnifiedMaterial -> GpuMaterial (CUDA only)
   - `toGpuLight()` - UnifiedLight -> LightGPU (CUDA only)

5. **unified_cpu_adapter.h** - CPU renderer integration
   - `CPUTextureSampler` - Texture sampling wrapper
   - `UnifiedCPURenderContext` - Pre-converted scene data
   - `cpu_shadow_test()` - Shadow ray helper
   - `extract_material_params()` - Get textured material values
   - `compute_direct_lighting_unified()` - Direct lighting for CPU

### Source Files (src/)

1. **unified_converters.cpp** - Converter implementations
   - Implements all conversion functions
   - Handles PrincipledBSDF, PointLight, DirectionalLight, AreaLight, SpotLight

## Integration Steps

### Step 1: Add Files to Build System

Add to CMakeLists.txt:
```cmake
# Header files (unified system)
set(UNIFIED_HEADERS
    source/include/unified_types.h
    source/include/unified_brdf.h
    source/include/unified_light_sampling.h
    source/include/unified_converters.h
    source/include/unified_cpu_adapter.h
)

# Source files
set(UNIFIED_SOURCES
    source/src/unified_converters.cpp
)
```

### Step 2: Update CPU Renderer (Renderer.cpp)

Replace `ray_color()` with unified version:

```cpp
#include "unified_cpu_adapter.h"

Vec3 Renderer::ray_color(const Ray& r, const Hittable* bvh,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color, int depth, int sample_index) 
{
    Vec3f color(0.0f);
    Vec3f throughput(1.0f);
    Ray current_ray = r;
    
    for (int bounce = 0; bounce < render_settings.max_bounces; ++bounce) {
        HitRecord rec;
        
        if (!bvh->hit(current_ray, 0.001f, INFINITY, rec)) {
            // Use unified background factor
            float bg_factor = background_factor(bounce);
            color += throughput * toVec3f(background_color) * bg_factor;
            break;
        }
        
        // Normal map, etc...
        apply_normal_map(rec);
        
        // Extract material params (with texture sampling)
        Vec3f albedo, emission;
        float roughness, metallic, opacity, transmission;
        extract_material_params(rec.material, rec.u, rec.v,
            &albedo, &roughness, &metallic, &opacity, &transmission, &emission);
        
        Vec3f N = toVec3f(rec.interpolated_normal);
        Vec3f wo = toVec3f(-current_ray.direction.normalize());
        Vec3f hit_pos = toVec3f(rec.point);
        
        // Direct lighting using unified function
        Vec3f direct_light(0.0f);
        if (!lights.empty()) {
            int light_idx = pick_smart_light_unified(
                unified_lights.data(), 
                unified_lights.size(),
                hit_pos,
                Vec3::random_float()
            );
            
            if (light_idx >= 0) {
                direct_light = compute_direct_lighting_unified(
                    unified_lights[light_idx],
                    bvh, hit_pos, N, wo,
                    albedo, roughness, metallic,
                    Vec3::random_float(), Vec3::random_float()
                );
                
                // Monte Carlo correction for light picking
                direct_light *= static_cast<float>(unified_lights.size());
                
                // Firefly clamp (matching GPU)
                direct_light = clamp_contribution(direct_light, 
                    UnifiedConstants::MAX_CONTRIBUTION);
            }
        }
        
        // Scatter ray...
        // (Use existing scatter logic, just ensuring attenuation uses unified BRDF)
        
        // Russian Roulette (matching GPU)
        if (bounce > UnifiedConstants::RR_START_BOUNCE) {
            float p = russian_roulette_probability(throughput);
            if (Vec3::random_float() > p) break;
            throughput /= p;
            
            // Throughput clamp (matching GPU)
            throughput = clamp_contribution(throughput, 
                UnifiedConstants::MAX_CONTRIBUTION);
        }
        
        // Accumulate
        color += throughput * (emission + direct_light) * opacity;
        
        // Update for next bounce...
    }
    
    // Final clamp (matching GPU)
    color = color.clamp(0.0f, 100.0f);
    
    return toVec3(color);
}
```

### Step 3: Pre-convert Scene Data

At render start, convert lights once:

```cpp
void Renderer::prepare_unified_context(const SceneData& scene) {
    unified_lights.clear();
    for (const auto& light : scene.lights) {
        unified_lights.push_back(toUnifiedLight(light));
    }
}
```

### Step 4: Update GPU Renderer (Optional)

The GPU already uses the correct math. To ensure parity, include the unified headers:

```cuda
// In ray_color.cuh or similar
#include "unified_brdf.h"

// Use unified constants
const float MAX_CONTRIBUTION = UnifiedConstants::MAX_CONTRIBUTION;
```

## Key Parity Points

| Feature | CPU Before | GPU | CPU After (Unified) |
|---------|------------|-----|---------------------|
| Firefly Clamp | None | 10.0f | 10.0f |
| Background Falloff | pow(0.5, bounce) | 1/(1+bounce*0.5) | Same as GPU |
| Russian Roulette | bounce>2, p<0.98 | bounce>2, p∈[0.05,0.95] | Same as GPU |
| MIS Weight | power_heuristic | power_heuristic | Identical |
| BRDF | PrincipledBSDF::scatter | evaluate_brdf | Identical math |
| Light PDF | Per-light-type | Per-light-type | Identical |

## Testing

1. Render same scene on CPU and GPU
2. Compare output images pixel-wise
3. Statistical difference should be < 1% after convergence
4. Visual difference should be imperceptible

## Notes

- Texture sampling still uses CPU-side code (not unified yet)
- Material conversion happens at scene load time for efficiency
- The unified system is backward-compatible - existing code still works
- CUDA headers are only included when __CUDACC__ is defined
