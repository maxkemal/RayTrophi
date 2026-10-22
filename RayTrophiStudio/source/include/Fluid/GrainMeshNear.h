#pragma once

// Near-camera grain meshes (K0 of docs/dev/KUM_TANE_GEOMETRISI_VE_MINERAL_KARISIMI.md).
//
// A Matter domain normally draws every splat particle through the procedural
// sphere cloud (one AABB BLAS). With near meshes enabled, the particles within
// a camera distance are drawn instead as triangle instances of a small library
// of exemplar grains (e.g. the shards of a Voronoi-fractured sphere), and the
// rest stay in the sphere cloud. Both pools come from the SAME particle walk in
// ParticleRenderBridge, so a particle is drawn exactly once.
//
// Render-only: the solver never sees the exemplar shape. The point of this
// first stage is a measurement - does hardware triangle instancing cost less
// than the procedural intersection shader, and where does the instance count
// start to dominate? - so every count the split produces is published.

#include "Vec2.h"
#include "Vec3.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class Hittable;
class Triangle;

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
}

namespace RayTrophiSim::Fluid {

class FluidParticles;

bool isGrainMeshLibraryShape(const std::string& shape);
uint32_t primarySplatVisualChildren(const SimulationGridDomainDesc& domain);
float splatCarrierDiameter(const SimulationGridDomainDesc& domain,
    const FluidParticles& particles, std::size_t index, float parcel_diameter);

// One recentered, unit-max-extent triangle of a scene node.
struct NormalizedSceneFace {
    Vec3 p[3];
    Vec3 n[3];
    Vec2 uv[3];
    uint16_t material = 0;
};

// Reads a scene node from the canonical flat TriangleMesh SoA (through its
// index buffer), recenters it on its bounds and scales its largest extent to 1.
// Returns false when the node has no usable triangles.
bool gatherNormalizedSceneFaces(const std::vector<std::shared_ptr<Hittable>>& objects,
                                const std::string& node_name,
                                std::vector<NormalizedSceneFace>& out);

struct GrainExemplar {
    std::string node;
    std::vector<NormalizedSceneFace> faces;
    // Uniform scale applied on top of the particle diameter so the exemplar
    // encloses the same VOLUME as the sphere it replaces. A Voronoi shard
    // normalized to unit extent holds far less volume than the unit-diameter
    // sphere; without this the bed reads thin and see-through - plausible,
    // and never reported as a bug.
    float volume_scale = 1.0f;
    // False when the mesh is open or degenerate: the divergence-theorem volume
    // is meaningless there, so no correction is applied and the count says so.
    bool volume_measured = false;
};

// Clamp of volume_scale; outside it the exemplar is not a grain-like solid
// (a sliver or an open surface), and exemplars_without_volume says so.
// Distance and budget have NO ceiling here: the panel's drag range is the
// only limit, the device's TLAS maxInstanceCount the only hard one.
inline constexpr float kGrainMeshMinVolumeScale = 0.5f;
inline constexpr float kGrainMeshMaxVolumeScale = 2.0f;

// Builds scene, seeded rounded-rock, or convex Voronoi render exemplars.
// Scene mode fails (and names the
// node) when any listed source is missing, so a script cannot measure a
// library that silently lost half its shapes.
bool buildGrainExemplars(const std::vector<std::shared_ptr<Hittable>>& objects,
                         const std::vector<std::string>& nodes,
                         std::vector<GrainExemplar>& out, std::string& error,
                         const std::string& shape = "scene");

// Fingerprint only selected flat source data, never transient render groups.
// Material/UV/mesh edits refresh the cached library without resetting physics.
uint64_t grainMeshSceneSignature(const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::vector<std::string>& nodes);

// Stable per-particle choices, keyed by the solver's particle_id so reseeding
// or compaction never swaps a grain's shape or orientation.
uint32_t grainExemplarFor(uint64_t particle_id, uint32_t exemplar_count);
Vec3 grainEulerDegreesFor(uint64_t particle_id);

// A negative material preserves each exemplar face material.
// Builds one Triangle soup per (material, exemplar). Transient instance
// sources use Triangle soups on purpose: ScatterSource::flat_meshes reuse BLASes
// owned by world objects, and these exemplars are not world objects.
void buildGrainExemplarTriangles(const GrainExemplar& exemplar, int material,
                                 const std::string& node_name,
                                 std::vector<std::shared_ptr<Triangle>>& out);

struct GrainMeshNearStats {
    bool enabled = false;
    // "off" | "active" | the reason the split is not running.
    std::string status = "off";
    uint32_t exemplars = 0;
    uint32_t exemplars_without_volume = 0;
    uint64_t exemplar_triangles = 0;   // one copy of the library
    float mean_volume_scale = 1.0f;
    uint32_t mesh_sources = 0;         // exemplars x material slots
    uint64_t near_instances = 0;       // particles drawn as meshes
    uint64_t near_over_budget = 0;     // inside the distance, past the budget
    uint64_t sphere_instances = 0;     // particles left in the sphere cloud
    uint64_t mesh_slots = 0;           // TLAS slots incl. zero-scale pool slack
    float camera_distance = 0.0f;      // threshold actually used
};

void publishGrainMeshNearStats(const std::string& domain, const GrainMeshNearStats& stats);
bool readGrainMeshNearStats(const std::string& domain, GrainMeshNearStats& out);
void clearGrainMeshNearStats(const std::string& domain);

} // namespace RayTrophiSim::Fluid
