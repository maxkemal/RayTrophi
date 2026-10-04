#pragma once

#include "Vec3.h"

#include <cstdint>

struct InstanceGroup;
struct InstanceTransform;

namespace RayTrophiSim::Fluid {

inline constexpr uint32_t kFluidRenderProxyMaxChildren = 32;
inline constexpr float kFluidRenderProxyMaxSizeVariation = 0.75f;
inline constexpr uint64_t kFluidRenderProxyDefaultVisualSphereBudget = 8000000ull;
inline constexpr uint64_t kFluidRenderProxyMinVisualSphereBudget = 1000000ull;
inline constexpr uint64_t kFluidRenderProxyMaxVisualSphereBudget = 32000000ull;
inline constexpr uint64_t kFluidRenderProxyRtBytesPerSphereEstimate = 112ull;
inline constexpr float kFluidRenderProxyFillSupportVoxels = 0.72f;
inline constexpr uint32_t kFluidRenderProxyNeighborCandidates = 10;
inline constexpr float kFluidRenderProxyNeighborSupportScale = 2.5f;

struct FluidRenderProxyLayout {
    uint32_t children_per_parent = 1;
    float child_radius_scale = 1.0f;
    float spread_radius_scale = 0.0f;
};

// Particle refinement is deliberately render-only. The children retain the
// parent's represented volume while reducing the isolated-sphere silhouette
// caused by drawing one large sphere for every APIC carrier.
FluidRenderProxyLayout resolveFluidRenderProxyLayout(
    bool virtual_grains_enabled,
    uint32_t requested_children = 8);

uint32_t limitFluidRenderProxyChildren(uint32_t requested_children,
                                       uint64_t parent_count,
                                       uint64_t max_visual_spheres);

uint64_t fluidRenderProxyCarrierCapacity(uint64_t primary_capacity,
                                         uint64_t secondary_capacity);

// Stable Material Object Info -> Random value for a logical render sphere.
// Position is deliberately excluded so a moving grain does not change colour.
float fluidRenderProxyObjectRandom(int group_id,
                                   uint32_t parent_index,
                                   uint32_t child_index);

// Resolve one RayFusion triangle-sphere child from the same parent metadata
// used by Vulkan RT and the raster impostor path. Returns false for bad indices.
bool resolveFluidRenderProxyTransform(const InstanceGroup& group,
                                      uint32_t parent_index,
                                      uint32_t child_index,
                                      InstanceTransform& transform);

// CPU mirror of the shader's deterministic offset pattern. Vulkan RT uses it
// while RayFusion evaluates the same pattern from gl_InstanceIndex.
Vec3 fluidRenderProxyOffset(uint32_t parent_index,
                            uint32_t child_index,
                            uint32_t children_per_parent,
                            float spread_radius);

// Pull a child toward a distance- and direction-weighted average of nearby
// carriers. The caller supplies a small stable candidate window; candidates
// outside support_radius are ignored. With no usable neighbor this falls back
// to the deterministic free-space pattern above.
Vec3 fluidRenderProxyNeighborOffset(uint32_t parent_index,
                                    uint32_t child_index,
                                    uint32_t children_per_parent,
                                    const Vec3& parent_position,
                                    const Vec3* candidates,
                                    uint32_t candidate_count,
                                    float support_radius,
                                    float fallback_spread_radius);

// Deterministic per-child radius multiplier, including the 1/cbrt(N) base.
// Opposite pairs are normalized so the sum of child volumes equals the parent
// volume even when size variation is non-zero.
float fluidRenderProxyChildRadiusScale(uint32_t parent_index,
                                       uint32_t child_index,
                                       uint32_t children_per_parent,
                                       float size_variation);

} // namespace RayTrophiSim::Fluid
