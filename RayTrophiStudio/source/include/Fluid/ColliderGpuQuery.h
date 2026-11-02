#pragma once

#include "../SimulationCompute.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace RayTrophiSim {
struct SurfaceMeshTriangle;
namespace Fluid {

// A lattice of query points q = M * (idx + base + 0.5), M a 3x4 affine map into
// the mesh's own space (sim_collider_bvh_query.comp says why one matrix).
struct ColliderQueryLattice {
    float m[3][4]{};
    int dims[3]{};
    int base[3]{};
    bool signed_distance = false;
};

// GPU distance queries against collider meshes (docs/dev/COLLIDER_TEMAS_YUZEYI.md,
// P2 cook and P3 per-frame shell). The flat BVH is built on the CPU once per mesh
// revision and kept on the device; each query is one dispatch and one download.
class ColliderGpuQuery {
public:
    // `key` + `revision` identify the triangle set; a new pair rebuilds and
    // re-uploads its BVH. `out` receives one distance per lattice point, in
    // mesh-space units.
    bool query(SimulationComputeContext& compute, const void* key, uint64_t revision,
               const std::vector<SurfaceMeshTriangle>& triangles,
               const ColliderQueryLattice& lattice, std::vector<float>& out,
               std::string& error);
    void release(SimulationComputeContext* compute);
    double lastMilliseconds() const { return last_ms_; }

private:
    struct Mesh {
        ComputeBufferHandle nodes;
        ComputeBufferHandle vertices;
        uint64_t revision = 0;
        std::size_t triangles = 0;
        uint64_t last_use = 0;
    };
    bool ensureMesh(SimulationComputeContext& compute, const void* key, uint64_t revision,
                    const std::vector<SurfaceMeshTriangle>& triangles, Mesh*& mesh,
                    std::string& error);
    std::unordered_map<const void*, Mesh> meshes_;
    ComputeBufferHandle output_;
    std::size_t output_bytes_ = 0;
    uint64_t use_counter_ = 0;
    double last_ms_ = 0.0;
};

} // namespace Fluid
} // namespace RayTrophiSim
