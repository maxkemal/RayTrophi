#pragma once

#include "../Vec3.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim {
struct SurfaceMeshTriangle;
namespace Fluid {

// Flat GPU ABI: vec3/uint + vec3/uint, 32 bytes. Median split bounds depth;
// leaves reference reordered flat triangles, never scene Triangle facades.
struct alignas(16) MatterGrainBvhNode {
    float low[3]{};
    uint32_t first = 0;
    float high[3]{};
    uint32_t second = 0;
};
static_assert(sizeof(MatterGrainBvhNode) == 32);
static_assert(offsetof(MatterGrainBvhNode, first) == 12);
static_assert(offsetof(MatterGrainBvhNode, second) == 28);

struct MatterGrainColliderBvh {
    std::vector<MatterGrainBvhNode> nodes;
    std::vector<Vec3> vertices;
    std::vector<uint32_t> source_faces;
    std::vector<uint32_t> surface_patches;
    // Vertex velocities in the reordered triangle order (empty = static).
    std::vector<Vec3> velocities;
};

// Transactional, validates finite coordinates, topology/cardinality and area.
// `velocities` (3 per triangle, optional): node bounds then cover each
// triangle's sweep over the last `sweep_seconds` (end - v * t .. end).
bool buildMatterGrainColliderBvh(const std::vector<SurfaceMeshTriangle>& triangles,
                                MatterGrainColliderBvh& result, std::string& error,
                                const std::vector<Vec3>* velocities = nullptr,
                                float sweep_seconds = 0.0f);
uint64_t matterGrainColliderFingerprint(const std::vector<SurfaceMeshTriangle>& triangles,
                                        const std::vector<Vec3>* velocities = nullptr);

} // namespace Fluid
} // namespace RayTrophiSim
