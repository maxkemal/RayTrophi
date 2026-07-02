/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          ParticleRenderBridge.cpp
* =========================================================================
* Discrete-particle render bridge.
*
* The volumetric side of a particle system (gas/fluid grid domains) is mirrored
* into live NanoVDB volumes by SceneData::syncSimulationRenderVolumes(). This
* file is its sibling for the *discrete* particles (Spark / Granular / Fluid SoA
* points): each visible system's alive particles are mirrored into a transient
* InstanceManager group — one instance per particle, sharing a single built-in
* primitive BLAS. Every RT backend (OptiX, Vulkan offline, Vulkan viewport)
* already iterates InstanceManager::getGroups(), so this lights particles up in
* all three real render paths with no backend edits.
*
* Design notes:
*   - The primitive mesh (sphere/cube/tetra/quad) is generated ONCE per system
*     and stored in the group's ScatterSource. The backends cache the BLAS by
*     the triangle-vector pointer (Vulkan) / node-name+material (OptiX), so a
*     stable vector + stable node name means only the TLAS is refit per step.
*   - Particle motion changes the TLAS every frame; a path tracer resets
*     accumulation during animation anyway, so this is inherent, not wasteful.
*   - The sim layer stays render-agnostic: per-particle source selection (for
*     the weighted SceneMeshes mode) is derived here from a stable slot hash,
*     not stored in the SoA.
*/

#include "scene_data.h"
#include "InstanceManager.h"
#include "InstanceGroup.h"
#include "Triangle.h"
#include "HittableInstance.h"
#include "EmbreeBVH.h"
#include "ParallelBVHNode.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "PBRMaterialSnapshot.h"
#include "Fluid/FluidFoam.h"
#include "globals.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

// ── Build signature: rebuild the primitive source + material only when the
// system's render config actually changes (not every frame). Keyed by group id.
struct SourceState {
    uint64_t signature = 0;       // primitive/material config hash
    uint64_t content_hash = 1;    // alive particle transforms hash (1 = "never synced")
    uint64_t bucket_hash = 1;     // over-life age-bucket (source_index) hash
};
std::unordered_map<int, SourceState> g_source_state;

uint64_t hashCombine(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

uint64_t quantize(float v) {
    return static_cast<uint64_t>(static_cast<int64_t>(std::lround(v * 1000.0f)));
}

Vec3 lerpColor(const Vec3& a, const Vec3& b, float t) {
    return Vec3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}

uint64_t renderSignature(const SceneData::ParticleRenderSettings& r) {
    uint64_t h = 1469598103934665603ull;
    h = hashCombine(h, static_cast<uint64_t>(r.shape));
    h = hashCombine(h, static_cast<uint64_t>(r.sphere_subdivisions));
    h = hashCombine(h, static_cast<uint64_t>(std::max(1, std::min(r.color_buckets, 64))));
    h = hashCombine(h, r.emissive ? 1ull : 0ull);
    h = hashCombine(h, r.over_life_color ? 1ull : 0ull);
    h = hashCombine(h, r.inherit_color_from_emitter ? 1ull : 0ull);
    h = hashCombine(h, quantize(r.base_color.x));
    h = hashCombine(h, quantize(r.base_color.y));
    h = hashCombine(h, quantize(r.base_color.z));
    h = hashCombine(h, quantize(r.color_end.x));
    h = hashCombine(h, quantize(r.color_end.y));
    h = hashCombine(h, quantize(r.color_end.z));
    h = hashCombine(h, quantize(r.emission_strength));
    h = hashCombine(h, quantize(r.roughness));
    if (r.shape == SceneData::ParticleRenderShape::SceneMeshes) {
        for (const auto& ms : r.mesh_sources) {
            for (char c : ms.node_name) h = hashCombine(h, static_cast<uint64_t>(c));
            h = hashCombine(h, 0x5bd1e995ull);  // separator
            h = hashCombine(h, quantize(ms.weight));
        }
    }
    return h;
}

// ── Preset material for one color bucket. Built/updated by stable name so a
// gradient edit refreshes the existing material instead of leaking a new one.
uint16_t ensureParticleMaterial(const std::string& name, const Vec3& color,
                                const SceneData::ParticleRenderSettings& r,
                                float emission_scale = 1.0f) {
    auto& mgr = MaterialManager::getInstance();

    const uint16_t existing_id = mgr.getMaterialID(name);
    std::shared_ptr<PrincipledBSDF> pbsdf;
    if (existing_id != MaterialManager::INVALID_MATERIAL_ID) {
        pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(mgr.getMaterialShared(existing_id));
    }
    const bool fresh = !pbsdf;
    if (fresh) {
        pbsdf = std::make_shared<PrincipledBSDF>();
    }

    const float rough = std::max(0.02f, r.roughness);
    pbsdf->albedoProperty = MaterialProperty(color, 1.0f);
    pbsdf->roughnessProperty = MaterialProperty(Vec3(rough), rough);
    pbsdf->metallicProperty = MaterialProperty(Vec3(0.0f), 0.0f);
    pbsdf->specularProperty = MaterialProperty(Vec3(0.5f), 0.5f);
    pbsdf->opacityProperty.alpha = 1.0f;
    pbsdf->setRoughness(rough);
    if (r.emissive) {
        pbsdf->emissionProperty =
            MaterialProperty(color, std::max(0.0f, r.emission_strength) * std::max(0.0f, emission_scale));
    } else {
        pbsdf->emissionProperty = MaterialProperty(Vec3(0.0f), 0.0f);
    }

    // Build the GPU material snapshot locally (avoids touching the global sync).
    if (!pbsdf->gpuMaterial) {
        pbsdf->gpuMaterial = std::make_shared<GpuMaterial>();
    }
    applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(*pbsdf), *pbsdf->gpuMaterial);

    if (fresh) {
        pbsdf->materialName = name;
        const uint16_t new_id = mgr.getOrCreateMaterialID(name, pbsdf);
        // The MaterialManager just grew, so any cached "last uploaded material
        // count" the backends keep is now stale. Without this flag, a UI add-
        // material flow that follows a particle preset path can stay sub-count
        // and never trigger the full updateBackendMaterials fallback — only the
        // first material reaches the GPU and subsequent ones are silently
        // skipped. Setting g_materials_dirty forces Main.cpp's per-frame sync
        // (Main.cpp:1983) to flush the full table on the next tick.
        ::g_materials_dirty = true;
        return new_id;
    }
    return existing_id;
}

// ── Built-in primitive geometry (centered, unit diameter, radius 0.5). ──────
std::shared_ptr<Triangle> makeTri(const Vec3& a, const Vec3& b, const Vec3& c,
                                  const Vec3& na, const Vec3& nb, const Vec3& nc,
                                  uint16_t matId, const std::string& node) {
    auto t = std::make_shared<Triangle>(a, b, c, na, nb, nc,
                                        Vec2(0.0f, 0.0f), Vec2(0.0f, 0.0f), Vec2(0.0f, 0.0f),
                                        matId);
    t->setNodeName(node);
    return t;
}

std::shared_ptr<Triangle> makeFlatTri(const Vec3& a, const Vec3& b, const Vec3& c,
                                      uint16_t matId, const std::string& node) {
    Vec3 e1 = b - a, e2 = c - a;
    Vec3 n = Vec3::cross(e1, e2);
    const float len = n.length();
    n = (len > 1e-8f) ? n * (1.0f / len) : Vec3(0.0f, 1.0f, 0.0f);
    return makeTri(a, b, c, n, n, n, matId, node);
}

void buildIcosphere(int subdivisions, uint16_t matId, const std::string& node,
                    std::vector<std::shared_ptr<Triangle>>& out) {
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    std::vector<Vec3> verts = {
        {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
        { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
        { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}
    };
    std::vector<std::array<int, 3>> faces = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
    };

    const int levels = std::max(0, std::min(subdivisions, 3));
    for (int s = 0; s < levels; ++s) {
        std::vector<std::array<int, 3>> next;
        next.reserve(faces.size() * 4);
        std::unordered_map<uint64_t, int> midpoint;
        auto getMid = [&](int i, int j) -> int {
            uint64_t key = (static_cast<uint64_t>(std::min(i, j)) << 32) | static_cast<uint32_t>(std::max(i, j));
            auto it = midpoint.find(key);
            if (it != midpoint.end()) return it->second;
            Vec3 m = (verts[i] + verts[j]) * 0.5f;
            verts.push_back(m);
            int idx = static_cast<int>(verts.size()) - 1;
            midpoint[key] = idx;
            return idx;
        };
        for (const auto& f : faces) {
            int a = getMid(f[0], f[1]);
            int b = getMid(f[1], f[2]);
            int c = getMid(f[2], f[0]);
            next.push_back({f[0], a, c});
            next.push_back({f[1], b, a});
            next.push_back({f[2], c, b});
            next.push_back({a, b, c});
        }
        faces.swap(next);
    }

    // Project to radius 0.5; vertex normal = radial direction.
    auto norm = [](const Vec3& v) {
        const float l = v.length();
        return (l > 1e-8f) ? v * (1.0f / l) : Vec3(0.0f, 1.0f, 0.0f);
    };
    out.reserve(out.size() + faces.size());
    for (const auto& f : faces) {
        Vec3 n0 = norm(verts[f[0]]), n1 = norm(verts[f[1]]), n2 = norm(verts[f[2]]);
        out.push_back(makeTri(n0 * 0.5f, n1 * 0.5f, n2 * 0.5f, n0, n1, n2, matId, node));
    }
}

void buildCube(uint16_t matId, const std::string& node,
               std::vector<std::shared_ptr<Triangle>>& out) {
    const float h = 0.5f;
    const Vec3 p[8] = {
        {-h,-h,-h}, { h,-h,-h}, { h, h,-h}, {-h, h,-h},
        {-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h}
    };
    const int quads[6][4] = {
        {0,1,2,3}, // -Z
        {5,4,7,6}, // +Z
        {4,0,3,7}, // -X
        {1,5,6,2}, // +X
        {4,5,1,0}, // -Y
        {3,2,6,7}  // +Y
    };
    for (auto& q : quads) {
        out.push_back(makeFlatTri(p[q[0]], p[q[1]], p[q[2]], matId, node));
        out.push_back(makeFlatTri(p[q[0]], p[q[2]], p[q[3]], matId, node));
    }
}

void buildTetra(uint16_t matId, const std::string& node,
                std::vector<std::shared_ptr<Triangle>>& out) {
    const float r = 0.5f;
    const Vec3 a( r,  r,  r);
    const Vec3 b( r, -r, -r);
    const Vec3 c(-r,  r, -r);
    const Vec3 d(-r, -r,  r);
    out.push_back(makeFlatTri(a, c, b, matId, node));
    out.push_back(makeFlatTri(a, b, d, matId, node));
    out.push_back(makeFlatTri(a, d, c, matId, node));
    out.push_back(makeFlatTri(b, c, d, matId, node));
}

void buildQuad(uint16_t matId, const std::string& node,
               std::vector<std::shared_ptr<Triangle>>& out) {
    const float h = 0.5f;
    const Vec3 a(-h, -h, 0.0f), b(h, -h, 0.0f), c(h, h, 0.0f), d(-h, h, 0.0f);
    const Vec3 n(0.0f, 0.0f, 1.0f);
    out.push_back(makeTri(a, b, c, n, n, n, matId, node));
    out.push_back(makeTri(a, c, d, n, n, n, matId, node));
}

void buildPrimitive(SceneData::ParticleRenderShape shape, int subdiv, uint16_t matId,
                    const std::string& node, std::vector<std::shared_ptr<Triangle>>& out) {
    out.clear();
    switch (shape) {
        case SceneData::ParticleRenderShape::Cube:  buildCube(matId, node, out); break;
        case SceneData::ParticleRenderShape::Tetra: buildTetra(matId, node, out); break;
        case SceneData::ParticleRenderShape::Quad:  buildQuad(matId, node, out); break;
        case SceneData::ParticleRenderShape::Sphere:
        default:
            buildIcosphere(subdiv, matId, node, out);
            break;
    }
}

// Collect a scene node's triangles, recenter them on their bounding-box center and
// normalize to unit max-extent so an instance scale == particle size behaves like
// the built-in primitives. The mesh keeps its own material (debris look). Returns
// centered copies tagged with a unique node name (stable BLAS cache key).
void gatherSceneMeshSource(const std::vector<std::shared_ptr<Hittable>>& objects,
                           const std::string& node_name,
                           std::vector<std::shared_ptr<Triangle>>& out) {
    out.clear();
    if (node_name.empty()) return;

    std::vector<Triangle*> src;
    Vec3 mn(1e30f, 1e30f, 1e30f), mx(-1e30f, -1e30f, -1e30f);
    for (const auto& obj : objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (!tri || tri->getNodeName() != node_name) continue;
        src.push_back(tri.get());
        for (int v = 0; v < 3; ++v) {
            const Vec3 p = tri->getOriginalVertexPosition(v);
            mn = Vec3::min(mn, p);
            mx = Vec3::max(mx, p);
        }
    }
    if (src.empty()) return;

    const Vec3 center = (mn + mx) * 0.5f;
    const Vec3 ext = mx - mn;
    const float inv = 1.0f / std::max({ ext.x, ext.y, ext.z, 1e-4f });
    const std::string out_node = "[PSysSceneGeo] " + node_name;

    out.reserve(src.size());
    for (Triangle* t : src) {
        const Vec3 p0 = (t->getOriginalVertexPosition(0) - center) * inv;
        const Vec3 p1 = (t->getOriginalVertexPosition(1) - center) * inv;
        const Vec3 p2 = (t->getOriginalVertexPosition(2) - center) * inv;
        const Vec3 n0 = t->getOriginalVertexNormal(0);
        const Vec3 n1 = t->getOriginalVertexNormal(1);
        const Vec3 n2 = t->getOriginalVertexNormal(2);
        const auto uv = t->getUVCoordinates();
        uint16_t mat = t->getMaterialID();
        if (mat == MaterialManager::INVALID_MATERIAL_ID) mat = 0;
        auto nt = std::make_shared<Triangle>(p0, p1, p2, n0, n1, n2,
                                             std::get<0>(uv), std::get<1>(uv), std::get<2>(uv), mat);
        nt->setNodeName(out_node);
        out.push_back(nt);
    }
}

constexpr std::size_t kMaxParticleInstancesTotal = 300000;

// Fluid particle instancing caps. Without a cap a million-particle seed
// produces a million TLAS instances; the OptiX/Vulkan accel build then runs out
// of memory or hits a build limit, which surfaces as a CUDA illegal-memory
// access that poisons the whole context (every later op — even the density
// splat — then fails). Two-tier cap:
//   * the per-domain render budget = min(UI fluid_max_particles, HARD ceiling),
//     so the cap tracks the value the user actually set in the panel instead
//     of a magic number divorced from the UI;
//   * the HARD ceiling is a hardware-safety backstop — one TLAS instance per
//     particle is inherently heavy (instance records + per-frame refit), so
//     beyond this the right tool is the SurfaceSDF render mode (a single
//     volume, independent of particle count) rather than a sphere per drop.
// Growth is CHUNKED so the expensive structural rebuild (BLAS/SBT) only fires
// at chunk boundaries instead of every frame during fill-up — that per-frame
// rebuild was racing the in-flight render and was a crash vector.
constexpr std::size_t kFluidInstanceHardCeiling = 1000000;
constexpr std::size_t kFluidPoolChunk = 16384;

inline std::size_t fluidRenderBudget(std::size_t ui_max_particles) {
    const std::size_t budget = (ui_max_particles == 0) ? kFluidInstanceHardCeiling
                                                        : ui_max_particles;
    return std::min(budget, kFluidInstanceHardCeiling);
}

inline std::size_t fluidPoolCapacityFor(std::size_t live_count, std::size_t budget) {
    const std::size_t capped = std::min(live_count, budget);
    if (capped == 0) return 0;
    // Geometric (doubling) growth instead of fixed kFluidPoolChunk steps. The
    // pool capacity only ever grows (callers clamp with `if (want > cap)`), and
    // every growth event triggers a structural backend rebuild — the spammy
    // "Rebuilding Geometry" path on both backends during a filling sim. Fixed
    // chunks crossed O(N / chunk) boundaries (≈31 rebuilds for 500k); doubling
    // makes it O(log) (≈5) and amortises to O(1) per particle, at the cost of up
    // to ~2x transient slack in the (degenerate, scale-0) instance pool.
    std::size_t cap = kFluidPoolChunk;
    while (cap < capped) cap <<= 1;
    return std::min(cap, budget);
}

} // namespace

void SceneData::syncParticleRenderInstances(bool enable_rt_geometry) {
    // Mutating InstanceManager / building geometry while a backend tears down or
    // rebuilds its accel structures poisons the CUDA context — the same hazard
    // syncSimulationRenderVolumes() guards against. Skip; resume next frame.
    if (g_optix_rebuild_in_progress.load() || g_viewport_rebuild_in_progress.load()) {
        return;
    }

    auto& im = InstanceManager::getInstance();

    // Caller can suppress the instanced geometry (e.g. Debug display mode shows the
    // lightweight ImGui overlay instead). Empty the groups but keep them + their
    // materials alive so re-enabling is a cheap rebuild, not a full recreate.
    if (!enable_rt_geometry) {
        bool emptied = false;
        for (auto& system : particle_systems) {
            if (system.render_instance_group_id < 0) continue;
            if (InstanceGroup* g = im.getGroup(system.render_instance_group_id)) {
                if (!g->instances.empty()) {
                    g->instances.clear();
                    g->gpu_dirty = true;
                    g_source_state[system.render_instance_group_id].content_hash = 1;
                    emptied = true;
                }
            }
        }
        if (emptied) {
            g_geometry_dirty = true;
            g_vulkan_rebuild_pending = true;
            g_optix_rebuild_pending = true;
        }
        return;
    }

    // A structural change (group/primitive/material built, or the instance-pool
    // size grew) needs a full rebuild so the backends create the scatter-instance
    // slots. Pure motion (pool size unchanged) takes the cheap TLAS-refit path:
    // updateInstanceTransforms / syncInstanceTransforms re-read group->instances
    // by index and refit the TLAS only — no BLAS rebuild, no scene walk. The
    // stable pool (dead slots collapsed to scale 0) is what keeps those index
    // bindings valid frame-to-frame; shrinking the vector would leave ghosts.
    bool structural_change = false;
    bool motion_change = false;
    std::size_t total_instances = 0;

    for (auto& system : particle_systems) {
        const bool wants = system.visible && system.enabled && system.runtime &&
                           system.render.render_in_raytrace;

        if (!wants) {
            destroyParticleRenderGroup(system);
            continue;
        }

        // Ensure the transient group exists.
        InstanceGroup* group = (system.render_instance_group_id >= 0)
            ? im.getGroup(system.render_instance_group_id) : nullptr;
        if (!group) {
            const std::string gname = "[Particles] " + system.name + " #" + std::to_string(system.id);
            system.render_instance_group_id = im.createGroup(gname, "", {});
            group = im.getGroup(system.render_instance_group_id);
            if (!group) continue;
            group->transient = true;
            group->sources.clear();
            g_source_state.erase(system.render_instance_group_id);
        }
        group->transient = true;

        // (Re)build the sources on config change. Two modes, both feeding the
        // per-source scatter material path (no shader change):
        //   * SceneMeshes: one ScatterSource per weighted scene-mesh node (debris);
        //     each keeps its own material. Particles pick a source by weighted CDF.
        //   * Built-in: one ScatterSource per color bucket (same primitive mesh, a
        //     material sampled along base_color -> color_end); particles pick a
        //     bucket by stable hash -> per-particle color variety (spark look).
        // Structural (BLAS/material set), so a full rebuild bakes per-instance data.
        SourceState& st = g_source_state[system.render_instance_group_id];
        uint64_t sig = renderSignature(system.render);
        // Editing the first emitter's start/end colors must invalidate the bucket
        // materials when the inherit toggle is on, otherwise the bridge keeps
        // serving the cached colors and the appearance edit looks ignored.
        if (system.render.inherit_color_from_emitter &&
            system.runtime && !system.runtime->emitters().empty()) {
            const auto& em = system.runtime->emitters().front();
            sig = hashCombine(sig, quantize(em.start_color.x));
            sig = hashCombine(sig, quantize(em.start_color.y));
            sig = hashCombine(sig, quantize(em.start_color.z));
            sig = hashCombine(sig, quantize(em.end_color.x));
            sig = hashCombine(sig, quantize(em.end_color.y));
            sig = hashCombine(sig, quantize(em.end_color.z));
        }
        const bool want_scene_meshes =
            system.render.shape == SceneData::ParticleRenderShape::SceneMeshes &&
            !system.render.mesh_sources.empty();
        if (group->sources.empty() || st.signature != sig) {
            group->sources.clear();

            if (want_scene_meshes) {
                for (const auto& ms : system.render.mesh_sources) {
                    ScatterSource src;
                    gatherSceneMeshSource(world.objects, ms.node_name, src.triangles);
                    if (src.triangles.empty()) continue;  // unresolved node — skip
                    src.name = "[PSysSceneGeo] " + ms.node_name;
                    src.weight = std::max(0.0f, ms.weight);
                    src.computeCenter();
                    group->sources.push_back(std::move(src));
                }
            }

            // Built-in primitive buckets (also the fallback when no scene mesh
            // resolved). base_color -> color_end gradient sampled into N materials.
            if (group->sources.empty()) {
                const int build_buckets = std::max(1, std::min(system.render.color_buckets, 64));
                group->sources.reserve(static_cast<std::size_t>(build_buckets));
                const SceneData::ParticleRenderShape prim_shape =
                    (system.render.shape == SceneData::ParticleRenderShape::SceneMeshes)
                        ? SceneData::ParticleRenderShape::Sphere
                        : system.render.shape;
                // Over-life: bucket index maps to AGE, so dim the emission toward
                // the end of life (sparks fade out). Variety mode keeps full glow.
                const bool over_life = system.render.over_life_color;
                // Single source of truth: when inherit is on, the gradient mirrors
                // the first emitter's Solid-billboard start/end colors so the user
                // does not have to maintain two color pairs (appearance panel +
                // RT panel). Toggle off to author RT-only colors.
                Vec3 grad_start = system.render.base_color;
                Vec3 grad_end   = system.render.color_end;
                if (system.render.inherit_color_from_emitter &&
                    system.runtime && !system.runtime->emitters().empty()) {
                    const auto& em = system.runtime->emitters().front();
                    grad_start = em.start_color;
                    grad_end   = em.end_color;
                }
                for (int b = 0; b < build_buckets; ++b) {
                    const float t = (build_buckets > 1) ? static_cast<float>(b) / (build_buckets - 1) : 0.0f;
                    const Vec3 col = lerpColor(grad_start, grad_end, t);
                    const float emis_scale = over_life ? (1.0f - 0.85f * t) : 1.0f;
                    const std::string mat_name = "[PSysMat] #" + std::to_string(system.id) + " b" + std::to_string(b);
                    const std::string geo_node = "[PSysGeo] #" + std::to_string(system.id) + " b" + std::to_string(b);
                    const uint16_t mat = ensureParticleMaterial(mat_name, col, system.render, emis_scale);

                    ScatterSource src;
                    src.name = geo_node;
                    src.weight = 1.0f;
                    buildPrimitive(prim_shape, system.render.sphere_subdivisions,
                                   mat, geo_node, src.triangles);
                    for (auto& tri : src.triangles) {
                        if (tri) tri->setMaterialID(mat);
                    }
                    src.computeCenter();
                    group->sources.push_back(std::move(src));
                }
            }
            st.signature = sig;
            structural_change = true;
        }

        // Refresh the stable instance pool from the alive SoA. The pool is kept at
        // the SoA capacity (which only ever grows), so per-frame the instance
        // COUNT is constant and the backends' scatter bindings stay index-valid.
        // Dead slots collapse to scale 0 (degenerate -> no GPU intersection). A
        // cheap content hash lets a settled simulation skip all GPU work so the
        // path tracer converges instead of refitting the TLAS for nothing.
        const auto& buf = system.runtime->buffers();
        const std::size_t cap = buf.alive.size();
        const float mult = std::max(1e-4f, system.render.size_multiplier);

        std::vector<InstanceTransform>& inst = group->instances;
        const bool pool_grew = cap != inst.size();  // capacity only grows
        if (pool_grew) {
            inst.resize(cap);
            structural_change = true;  // backend must (re)create the scatter slots
        }

        // Weighted source-pick CDF from the actually-built sources (uniform for
        // built-in color buckets, weighted for scene-mesh debris). Indices line up
        // with group->sources so the backend's per-instance bindings are correct.
        const std::size_t nsrc = group->sources.size();
        std::vector<float> src_cdf(nsrc, 0.0f);
        float weight_sum = 0.0f;
        for (std::size_t k = 0; k < nsrc; ++k) {
            weight_sum += std::max(0.0f, group->sources[k].weight);
            src_cdf[k] = weight_sum;
        }
        // Spin/orientation is invisible on a sphere and would only churn the refit
        // hash, so only 3D-asymmetric shapes get rotation.
        const bool apply_rotation =
            system.render.shape != SceneData::ParticleRenderShape::Sphere;
        // Over-life color is DISABLED: making each particle's material follow its
        // age needs a per-frame material change, which the cheap TLAS refit can't
        // do, so it forced a full rebuild — and emitter spawn/die churn made that
        // happen EVERY frame (continuous rebuild). The real fix is a per-instance
        // color buffer the refit re-uploads + the closesthit reads (shader path).
        // Until then over-life falls back to the cheap stable color VARIETY.
        const bool over_life = false;

        uint64_t content = 1469598103934665603ull;     // transforms (drives cheap refit)
        uint64_t bucket_sig = 1469598103934665603ull;   // over-life age buckets (drives rebuild)
        std::size_t alive_drawn = 0;
        for (std::size_t i = 0; i < cap; ++i) {
            InstanceTransform& tr = inst[i];
            tr.rotation = Vec3(0.0f, 0.0f, 0.0f);

            // Stable per-slot hash drives the source pick (color bucket / weighted
            // debris mesh) + a random base orientation. STABLE is essential: the
            // cheap TLAS refit only updates transforms, so a particle's
            // material/source_index must not change frame-to-frame (it is only
            // re-read on a structural rebuild). A slot hash is deterministic, so the
            // bridge always reproduces the same source_index the backend baked in.
            const uint64_t slot_hash = hashCombine(static_cast<uint64_t>(i) + 1469u,
                                                   static_cast<uint64_t>(system.id) + 7919u);
            int src_index = 0;
            if (nsrc > 1 && weight_sum > 0.0f) {
                const float u = static_cast<float>(slot_hash % 1000000ull) / 1000000.0f * weight_sum;
                while (src_index < static_cast<int>(nsrc) - 1 && u > src_cdf[src_index]) {
                    ++src_index;
                }
            }
            tr.source_index = src_index;

            float sz = (i < buf.size.size()) ? buf.size[i] : 0.05f;
            sz *= mult;
            const bool alive = (buf.alive[i] != 0u) && sz > 1e-5f &&
                               total_instances < kMaxParticleInstancesTotal;
            if (alive) {
                const float px = buf.position_x[i], py = buf.position_y[i], pz = buf.position_z[i];
                tr.position = Vec3(px, py, pz);
                tr.scale = Vec3(sz, sz, sz);
                if (over_life) {
                    // Bucket = age fraction -> color/emission walk the gradient as
                    // the particle ages (matches the Solid billboard fade).
                    const float age = (i < buf.age_seconds.size()) ? buf.age_seconds[i] : 0.0f;
                    const float life = (i < buf.lifetime_seconds.size()) ? buf.lifetime_seconds[i] : 1.0f;
                    const float frac = (life > 1e-5f) ? std::min(0.9999f, std::max(0.0f, age / life)) : 0.0f;
                    src_index = std::min(static_cast<int>(nsrc) - 1,
                                         static_cast<int>(frac * static_cast<float>(nsrc)));
                    tr.source_index = src_index;
                    // Bucket changes drive a (rare) full rebuild; transform motion
                    // between crossings stays on the cheap refit path.
                    bucket_sig = hashCombine(bucket_sig, static_cast<uint64_t>(i));
                    bucket_sig = hashCombine(bucket_sig, static_cast<uint64_t>(src_index));
                }
                if (apply_rotation) {
                    const float spin = (i < buf.rotation.size()) ? buf.rotation[i] : 0.0f;
                    const float spin_deg = spin * (180.0f / 3.14159265f);
                    const float base_x = static_cast<float>(slot_hash & 0xFFFFu) / 65535.0f * 360.0f;
                    const float base_z = static_cast<float>((slot_hash >> 20) & 0xFFFFu) / 65535.0f * 360.0f;
                    tr.rotation = Vec3(base_x, spin_deg, base_z);
                    content = hashCombine(content, quantize(spin_deg));
                }
                ++alive_drawn;
                ++total_instances;
                content = hashCombine(content, quantize(px));
                content = hashCombine(content, quantize(py));
                content = hashCombine(content, quantize(pz));
                content = hashCombine(content, quantize(sz));
            } else {
                // Degenerate (zero-scale) -> culled by the TLAS without shifting
                // any index. Keeps the cheap-refit bindings stable.
                tr.position = Vec3(0.0f, 0.0f, 0.0f);
                tr.scale = Vec3(0.0f, 0.0f, 0.0f);
            }
        }
        content = hashCombine(content, static_cast<uint64_t>(alive_drawn));

        // A bucket crossing (over-life) changes per-instance materials, which only a
        // full rebuild can apply. Pure transform motion stays on the cheap refit —
        // so over-life only pays the rebuild when a particle actually steps to the
        // next age color, not every frame.
        const bool buckets_changed = over_life && (bucket_sig != st.bucket_hash);
        if (structural_change || buckets_changed) {
            st.content_hash = content;
            st.bucket_hash = bucket_sig;
            group->gpu_dirty = true;
            structural_change = true;
        } else if (content != st.content_hash) {
            st.content_hash = content;
            group->gpu_dirty = true;
            motion_change = true;
        }
    }

    // Structural changes go through the full rebuild (builds scatter slots / BLAS);
    // pure motion takes the cheap deferred TLAS refit. Both flags also self-trigger
    // a redraw in the main loop, so this bridge does NOT touch the timeline driver's
    // simulation_render_updated gating — particle render is decoupled from the
    // timeline bake/scrub machinery and just mirrors the live SoA every frame.
    if (structural_change) {
        g_geometry_dirty = true;
        g_vulkan_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        // Solid/Matcap viewport: structural fluid changes must wake the raster
        // backend too, else new/grown particle geometry only appears after a
        // manual Rendered-mode toggle. Consumed only when a raster viewport is
        // active (harmless in Rendered mode).
        g_viewport_raster_rebuild_pending = true;
    } else if (motion_change) {
        g_gpu_refit_pending = true;
    }
    // CPU reference render mirrors the same change on its own rebuild path.
    if (structural_change || motion_change) g_particle_cpu_geometry_dirty = true;

    // Fluid objects (legacy fluid_objects vector) and SimulationGridDomain
    // (type=Fluid, fluid_render_mode=Particles — the active path) both ride
    // the same instancing machinery. Both run in the same tick.
    syncFluidParticleRenderInstances(enable_rt_geometry);
    syncDomainFluidParticleInstances(enable_rt_geometry);
    syncFluidFoamRenderInstances(enable_rt_geometry);
}

void SceneData::destroyParticleRenderGroup(ParticleSystemObject& system) {
    if (system.render_instance_group_id < 0) {
        return;
    }
    auto& im = InstanceManager::getInstance();
    if (InstanceGroup* group = im.getGroup(system.render_instance_group_id)) {
        if (!group->instances.empty()) {
            // Force one rebuild so the now-removed instances leave the TLAS.
            g_geometry_dirty = true;
            g_vulkan_rebuild_pending = true;
            g_optix_rebuild_pending = true;
        }
    }
    im.deleteGroup(system.render_instance_group_id);
    g_source_state.erase(system.render_instance_group_id);
    system.render_instance_group_id = -1;
}

void SceneData::releaseParticleRenderInstances() {
    for (auto& system : particle_systems) {
        destroyParticleRenderGroup(system);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fluid (APIC liquid) particle render bridge
// ─────────────────────────────────────────────────────────────────────────────
// FluidObject particles share the same instance-group machinery, but they have
// no per-particle alive flag / size / age — the SoA is dense and uniform — so
// the loop is a stripped-down version of the ParticleSystem path. Single
// material, single sphere source, fixed radius derived from voxel_size.
//
// Pool capacity grows monotonically (mirroring the ParticleSystem contract):
// reseed can shrink particle count, but shrinking the InstanceTransform vector
// would invalidate the backends' scatter-instance index bindings (cheap-refit
// only updates transforms by index, never rebinds materials). So we keep the
// pool at peak-seen size and collapse surplus slots to scale 0.

namespace {

struct FluidSourceState {
    uint64_t signature = 0;
    uint64_t content_hash = 1;
    std::size_t drawn_count = 0;
};
std::unordered_map<int, FluidSourceState> g_fluid_source_state;

uint64_t fluidRenderSignature(const RayTrophiSim::Fluid::FluidObject& obj) {
    uint64_t h = 1469598103934665603ull;
    h = hashCombine(h, quantize(obj.particle_render_color.x));
    h = hashCombine(h, quantize(obj.particle_render_color.y));
    h = hashCombine(h, quantize(obj.particle_render_color.z));
    h = hashCombine(h, quantize(obj.particle_render_radius_factor));
    h = hashCombine(h, quantize(obj.particle_render_emission));
    h = hashCombine(h, obj.particle_render_emissive ? 1ull : 0ull);
    h = hashCombine(h, static_cast<uint64_t>(
        std::max(0, std::min(obj.particle_render_subdivisions, 3))));
    return h;
}

// Core material builder. Takes plain fields so both the legacy FluidObject
// loop and the SimulationGridDomain loop can reuse it without a shared struct.
uint16_t ensureFluidParticleMaterialFields(const std::string& name,
                                           const Vec3& col,
                                           bool emissive,
                                           float emission_strength) {
    auto& mgr = MaterialManager::getInstance();
    const uint16_t existing_id = mgr.getMaterialID(name);
    std::shared_ptr<PrincipledBSDF> pbsdf;
    if (existing_id != MaterialManager::INVALID_MATERIAL_ID) {
        pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(mgr.getMaterialShared(existing_id));
    }
    const bool fresh = !pbsdf;
    if (fresh) pbsdf = std::make_shared<PrincipledBSDF>();

    const float rough = 0.25f;
    pbsdf->albedoProperty = MaterialProperty(col, 1.0f);
    pbsdf->roughnessProperty = MaterialProperty(Vec3(rough), rough);
    pbsdf->metallicProperty = MaterialProperty(Vec3(0.0f), 0.0f);
    pbsdf->specularProperty = MaterialProperty(Vec3(0.5f), 0.5f);
    pbsdf->opacityProperty.alpha = 1.0f;
    pbsdf->setRoughness(rough);
    if (emissive) {
        pbsdf->emissionProperty =
            MaterialProperty(col, std::max(0.0f, emission_strength));
    } else {
        pbsdf->emissionProperty = MaterialProperty(Vec3(0.0f), 0.0f);
    }
    if (!pbsdf->gpuMaterial) pbsdf->gpuMaterial = std::make_shared<GpuMaterial>();
    applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(*pbsdf), *pbsdf->gpuMaterial);

    if (fresh) {
        pbsdf->materialName = name;
        const uint16_t new_id = mgr.getOrCreateMaterialID(name, pbsdf);
        ::g_materials_dirty = true;
        return new_id;
    }
    return existing_id;
}

// Backwards-compatible wrapper for the existing FluidObject loop.
uint16_t ensureFluidParticleMaterial(const std::string& name,
                                     const RayTrophiSim::Fluid::FluidObject& obj) {
    return ensureFluidParticleMaterialFields(
        name, obj.particle_render_color,
        obj.particle_render_emissive, obj.particle_render_emission);
}

// White scattering foam material — diffuse, high albedo, opaque. Foam is air
// bubbles in water, so it reads white and very rough. A distinct material
// instance keeps the renderer treating whitewater separately from the
// refractive liquid surface (the "separate material" the foam layer needs).
uint16_t ensureFoamParticleMaterial(RayTrophiSim::Fluid::FoamType type, const std::string& name) {
    auto& mgr = MaterialManager::getInstance();
    std::string typed_name = name;
    if (type == RayTrophiSim::Fluid::FoamType::Spray) typed_name += "_Spray";
    else if (type == RayTrophiSim::Fluid::FoamType::Bubble) typed_name += "_Bubble";
    else typed_name += "_Foam";

    const uint16_t existing_id = mgr.getMaterialID(typed_name);
    std::shared_ptr<PrincipledBSDF> pbsdf;
    if (existing_id != MaterialManager::INVALID_MATERIAL_ID)
        pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(mgr.getMaterialShared(existing_id));
    const bool fresh = !pbsdf;
    if (fresh) pbsdf = std::make_shared<PrincipledBSDF>();

    if (type == RayTrophiSim::Fluid::FoamType::Spray) {
        // Spray: Water droplets (fully transmissive, refractive, smooth, sparkling)
        const Vec3 water_color(1.0f, 1.0f, 1.0f);
        const float rough = 0.02f;
        pbsdf->albedoProperty = MaterialProperty(water_color, 1.0f);
        pbsdf->roughnessProperty = MaterialProperty(Vec3(rough), rough);
        pbsdf->metallicProperty = MaterialProperty(Vec3(0.0f), 0.0f);
        pbsdf->specularProperty = MaterialProperty(Vec3(1.0f), 1.0f);
        pbsdf->opacityProperty.alpha = 1.0f;
        pbsdf->setRoughness(rough);
        pbsdf->setTransmission(1.0f, 1.333f); // 100% transmissive, water refractive index (1.333)
        pbsdf->emissionProperty = MaterialProperty(Vec3(0.0f), 0.0f);
    } else if (type == RayTrophiSim::Fluid::FoamType::Bubble) {
        // Bubble: thin-shell dielectric film (the new model). Light Fresnel-reflects
        // off the shell (bright silver rim) or passes straight through — no roughness,
        // no emission, no transmission TIR. The bubble BSDF branch handles everything;
        // the old IOR-1.1 / 0.12-emission hacks are gone (they masked the wrong model).
        const Vec3 bubble_color(0.95f, 0.98f, 1.0f);
        pbsdf->albedoProperty = MaterialProperty(bubble_color, 1.0f);
        pbsdf->roughnessProperty = MaterialProperty(Vec3(0.0f), 0.0f);
        pbsdf->metallicProperty = MaterialProperty(Vec3(0.0f), 0.0f);
        pbsdf->specularProperty = MaterialProperty(Vec3(1.0f), 1.0f);
        pbsdf->opacityProperty.alpha = 1.0f;
        pbsdf->setRoughness(0.0f);
        pbsdf->setTransmission(0.0f, 1.33f);
        pbsdf->emissionProperty = MaterialProperty(Vec3(0.0f), 0.0f);
        pbsdf->setIsBubble(true);     // thin-shell bubble model
        pbsdf->setBubbleIor(1.33f);    // air/water rim Fresnel
        pbsdf->setBubbleFilm(0.0f);     // clean air bubble by default (user dials soap shimmer)
    } else {
        // Foam: Scattering surface foam (diffuse, rough white)
        const Vec3 white(0.93f, 0.95f, 0.98f);
        const float rough = 0.85f;
        pbsdf->albedoProperty = MaterialProperty(white, 1.0f);
        pbsdf->roughnessProperty = MaterialProperty(Vec3(rough), rough);
        pbsdf->metallicProperty = MaterialProperty(Vec3(0.0f), 0.0f);
        pbsdf->specularProperty = MaterialProperty(Vec3(0.2f), 0.2f);
        pbsdf->opacityProperty.alpha = 1.0f;
        pbsdf->setRoughness(rough);
        pbsdf->setTransmission(0.0f, 1.5f);
        pbsdf->emissionProperty = MaterialProperty(Vec3(0.0f), 0.0f);
    }

    if (!pbsdf->gpuMaterial) pbsdf->gpuMaterial = std::make_shared<GpuMaterial>();
    applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(*pbsdf), *pbsdf->gpuMaterial);

    if (fresh) {
        pbsdf->materialName = typed_name;
        const uint16_t new_id = mgr.getOrCreateMaterialID(typed_name, pbsdf);
        ::g_materials_dirty = true;
        return new_id;
    }
    return existing_id;
}

// Resolve the foam material: a user-picked scene material id when valid,
// otherwise the built-in physical presets.
uint16_t resolveFoamMaterial(int foam_material_id, RayTrophiSim::Fluid::FoamType type, const std::string& default_name) {
    if (foam_material_id >= 0) {
        auto& mm = MaterialManager::getInstance();
        if (static_cast<size_t>(foam_material_id) < mm.getMaterialCount() &&
            static_cast<uint16_t>(foam_material_id) != MaterialManager::INVALID_MATERIAL_ID) {
            return static_cast<uint16_t>(foam_material_id);
        }
    }
    return ensureFoamParticleMaterial(type, default_name);
}

} // namespace

void SceneData::syncFluidParticleRenderInstances(bool enable_rt_geometry) {
    if (g_optix_rebuild_in_progress.load() || g_viewport_rebuild_in_progress.load()) {
        return;
    }
    auto& im = InstanceManager::getInstance();

    if (!enable_rt_geometry) {
        bool emptied = false;
        for (auto& obj : fluid_objects) {
            if (obj.render_instance_group_id < 0) continue;
            if (InstanceGroup* g = im.getGroup(obj.render_instance_group_id)) {
                if (!g->instances.empty()) {
                    g->instances.clear();
                    g->gpu_dirty = true;
                    g_fluid_source_state[obj.render_instance_group_id].content_hash = 1;
                    emptied = true;
                }
            }
        }
        if (emptied) {
            g_geometry_dirty = true;
            g_vulkan_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
        }
        return;
    }

    bool structural_change = false;
    bool motion_change = false;

    for (auto& obj : fluid_objects) {
        // Particles render_mode OR Solid/Matcap viewport (splat-sphere proxy, since
        // the raster viewport can't draw the SurfaceSDF volume).
        const bool wants = obj.visible && obj.enabled &&
            (obj.render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles ||
             g_solid_viewport_active);
        if (!wants) {
            destroyFluidParticleRenderGroup(obj);
            continue;
        }

        InstanceGroup* group = (obj.render_instance_group_id >= 0)
            ? im.getGroup(obj.render_instance_group_id) : nullptr;
        if (!group) {
            const std::string gname = "[FluidParticles] " + obj.name + " #" + std::to_string(obj.id);
            obj.render_instance_group_id = im.createGroup(gname, "", {});
            group = im.getGroup(obj.render_instance_group_id);
            if (!group) continue;
            group->transient = true;
            group->sources.clear();
            obj.render_pool_capacity = 0;
            g_fluid_source_state.erase(obj.render_instance_group_id);
        }
        group->transient = true;

        FluidSourceState& st = g_fluid_source_state[obj.render_instance_group_id];
        const uint64_t sig = fluidRenderSignature(obj);
        if (group->sources.empty() || st.signature != sig) {
            group->sources.clear();
            const std::string mat_name = "[FluidPMat] #" + std::to_string(obj.id);
            const std::string geo_node = "[FluidPGeo] #" + std::to_string(obj.id);
            const uint16_t mat = ensureFluidParticleMaterial(mat_name, obj);
            ScatterSource src;
            src.name = geo_node;
            src.weight = 1.0f;
            const int subdiv = std::max(0, std::min(obj.particle_render_subdivisions, 3));
            buildIcosphere(subdiv, mat, geo_node, src.triangles);
            for (auto& tri : src.triangles) {
                if (tri) tri->setMaterialID(mat);
            }
            src.computeCenter();
            group->sources.push_back(std::move(src));
            st.signature = sig;
            structural_change = true;
        }

        const std::size_t live_count = obj.particles.size();
        const float voxel = std::max(1e-4f, obj.voxel_size);
        const float radius = std::max(1e-4f,
            voxel * obj.particle_render_radius_factor * obj.particle_render_size_multiplier);
        // Built-in primitives are diameter-1 (centred, unit max extent), so
        // instance scale == diameter (= 2 * radius).
        const float diam = radius * 2.0f;

        // Chunked + budget-capped pool (same crash-avoidance contract as the
        // domain bridge — see fluidRenderBudget / fluidPoolCapacityFor).
        const std::size_t budget = fluidRenderBudget(obj.max_particles);
        const std::size_t draw_count = std::min(live_count, budget);
        const std::size_t want_cap = fluidPoolCapacityFor(live_count, budget);
        if (want_cap > obj.render_pool_capacity) {
            obj.render_pool_capacity = want_cap;
        }
        std::vector<InstanceTransform>& inst = group->instances;
        const bool pool_grew = inst.size() != obj.render_pool_capacity;
        if (pool_grew) {
            inst.resize(obj.render_pool_capacity);
            structural_change = true;
        }

        uint64_t content = 1469598103934665603ull;
        const std::size_t cap_now = inst.size();
        std::size_t drawn = 0;
        for (std::size_t i = 0; i < cap_now; ++i) {
            InstanceTransform& tr = inst[i];
            tr.rotation = Vec3(0.0f, 0.0f, 0.0f);
            tr.source_index = 0;
            if (i < draw_count) {
                const Vec3& p = obj.particles.position[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                    tr.position = Vec3(0.0f, 0.0f, 0.0f);
                    tr.scale = Vec3(0.0f, 0.0f, 0.0f);
                    continue;
                }
                tr.position = p;
                tr.scale = Vec3(diam, diam, diam);
                content = hashCombine(content, quantize(p.x));
                content = hashCombine(content, quantize(p.y));
                content = hashCombine(content, quantize(p.z));
                ++drawn;
            } else {
                tr.position = Vec3(0.0f, 0.0f, 0.0f);
                tr.scale = Vec3(0.0f, 0.0f, 0.0f);
            }
        }
        content = hashCombine(content, quantize(diam));
        content = hashCombine(content, static_cast<uint64_t>(drawn));

        if (structural_change) {
            st.content_hash = content;
            group->gpu_dirty = true;
        } else if (content != st.content_hash) {
            st.content_hash = content;
            group->gpu_dirty = true;
            motion_change = true;
        }
    }

    if (structural_change) {
        g_geometry_dirty = true;
        g_vulkan_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_viewport_raster_rebuild_pending = true;   // refresh Solid/Matcap viewport too
        g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    } else if (motion_change) {
        g_gpu_refit_pending = true;
    }
    if (structural_change || motion_change) g_particle_cpu_geometry_dirty = true;
}

void SceneData::destroyFluidParticleRenderGroup(RayTrophiSim::Fluid::FluidObject& obj) {
    if (obj.render_instance_group_id < 0) {
        return;
    }
    auto& im = InstanceManager::getInstance();
    if (InstanceGroup* group = im.getGroup(obj.render_instance_group_id)) {
        if (!group->instances.empty()) {
            g_geometry_dirty = true;
            g_vulkan_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
        }
    }
    im.deleteGroup(obj.render_instance_group_id);
    g_fluid_source_state.erase(obj.render_instance_group_id);
    obj.render_instance_group_id = -1;
    obj.render_pool_capacity = 0;
}

void SceneData::releaseFluidParticleRenderInstances() {
    for (auto& obj : fluid_objects) {
        destroyFluidParticleRenderGroup(obj);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulationGridDomain (type=Fluid, render_mode=Particles) bridge
// ─────────────────────────────────────────────────────────────────────────────
// Mirrors `state.particles` (FluidParticles SoA, source of truth for the APIC
// liquid solver) into a per-domain transient InstanceGroup. The render config
// (color / radius / emissive) lives on the desc, the alive particles live on
// the state — both are read inside the per-domain loop. Pool grows only;
// reseed-driven shrinks fall back to scale=0 in the surplus slots so the
// backends' scatter-instance bindings stay index-valid.

void SceneData::syncDomainFluidParticleInstances(bool enable_rt_geometry) {
    if (g_optix_rebuild_in_progress.load() || g_viewport_rebuild_in_progress.load()) {
        return;
    }
    auto& im = InstanceManager::getInstance();

    bool structural_change = false;
    bool motion_change = false;

    for (auto& system : particle_systems) {
        if (!system.runtime) continue;
        const auto& states  = system.runtime->gridDomainStates();
        auto& domains       = system.runtime->gridDomains();

        // Shrink parallel-indexed transient state if domains disappeared.
        for (std::size_t d = states.size();
             d < system.domain_particle_render_group_ids.size(); ++d) {
            const int gid = system.domain_particle_render_group_ids[d];
            if (gid >= 0) {
                im.deleteGroup(gid);
                g_fluid_source_state.erase(gid);
                structural_change = true;
            }
        }
        system.domain_particle_render_group_ids.resize(states.size(), -1);
        system.domain_particle_pool_capacities.resize(states.size(), 0);

        for (std::size_t d = 0; d < states.size(); ++d) {
            const auto& state  = states[d];
            const bool is_fluid = state.type == RayTrophiSim::SimulationDomainType::Fluid;

            // Render the splat-sphere proxy when the fluid's render_mode is Particles,
            // OR whenever the active viewport is Solid/Matcap — the raster viewport
            // can't draw the SurfaceSDF NanoVDB volume, so spheres are the live proxy
            // there. In Rendered mode + SurfaceSDF this stays false (the volume renders
            // instead) and the gate below tears the sphere group down.
            const bool wants = enable_rt_geometry &&
                system.visible && system.enabled && state.valid && is_fluid &&
                d < domains.size() &&
                (domains[d].fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles ||
                 g_solid_viewport_active) &&
                !state.particles.empty();

            int& group_id = system.domain_particle_render_group_ids[d];

            if (!wants) {
                if (group_id >= 0) {
                    if (InstanceGroup* g = im.getGroup(group_id)) {
                        if (!g->instances.empty()) structural_change = true;
                    }
                    im.deleteGroup(group_id);
                    g_fluid_source_state.erase(group_id);
                    group_id = -1;
                    system.domain_particle_pool_capacities[d] = 0;
                }
                continue;
            }

            const auto& dconfig = domains[d];  // safe — wants requires d < domains.size()

            InstanceGroup* group = (group_id >= 0) ? im.getGroup(group_id) : nullptr;
            if (!group) {
                const std::string gname = "[FluidDomParticles] " + system.name +
                                          " D" + std::to_string(d);
                group_id = im.createGroup(gname, "", {});
                group = im.getGroup(group_id);
                if (!group) continue;
                group->transient = true;
                group->sources.clear();
                system.domain_particle_pool_capacities[d] = 0;
                g_fluid_source_state.erase(group_id);
            }
            group->transient = true;

            // ── Source rebuild on config change. Single sphere bucket + a
            //    material derived from the domain's particle render config.
            uint64_t sig = 1469598103934665603ull;
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_color.x));
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_color.y));
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_color.z));
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_radius_factor));
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_emission));
            sig = hashCombine(sig, dconfig.fluid_particle_emissive ? 1ull : 0ull);
            sig = hashCombine(sig, static_cast<uint64_t>(
                std::max(0, std::min(dconfig.fluid_particle_subdivisions, 3))));
            // User-assigned materials must invalidate the cached BLAS — different
            // ID = different shader binding on the backends.
            sig = hashCombine(sig, static_cast<uint64_t>(
                static_cast<uint32_t>(dconfig.fluid_particle_material_id)));

            FluidSourceState& st = g_fluid_source_state[group_id];
            if (group->sources.empty() || st.signature != sig) {
                group->sources.clear();
                const std::string mat_name = "[FluidDomMat] " + system.name +
                                              " D" + std::to_string(d);
                const std::string geo_node = "[FluidDomGeo] " + system.name +
                                              " D" + std::to_string(d);
                // User-picked material wins (gives full PBR control — transmittance,
                // IOR, roughness — for proper water authoring). Auto-synthesised
                // material is the fallback when nothing is picked yet.
                uint16_t mat;
                if (dconfig.fluid_particle_material_id >= 0 &&
                    dconfig.fluid_particle_material_id != MaterialManager::INVALID_MATERIAL_ID) {
                    mat = static_cast<uint16_t>(dconfig.fluid_particle_material_id);
                } else {
                    mat = ensureFluidParticleMaterialFields(
                        mat_name, dconfig.fluid_particle_color,
                        dconfig.fluid_particle_emissive, dconfig.fluid_particle_emission);
                }
                ScatterSource src;
                src.name = geo_node;
                src.weight = 1.0f;
                const int subdiv = std::max(0, std::min(dconfig.fluid_particle_subdivisions, 3));
                buildIcosphere(subdiv, mat, geo_node, src.triangles);
                for (auto& tri : src.triangles) if (tri) tri->setMaterialID(mat);
                src.computeCenter();
                group->sources.push_back(std::move(src));
                st.signature = sig;
                structural_change = true;
            }

            // ── Pool grows monotonically, in chunks, and is budget-capped. ───
            // Budget tracks the UI's Max Particles (clamped to the hardware
            // ceiling). draw_count is what we render; cap is the chunk-rounded
            // pool size. Chunked growth means a filling sim doesn't trigger a
            // structural BLAS rebuild every frame.
            const std::size_t live_count = state.particles.size();
            const std::size_t budget = fluidRenderBudget(dconfig.fluid_max_particles);
            const std::size_t draw_count = std::min(live_count, budget);
            const std::size_t want_cap = fluidPoolCapacityFor(live_count, budget);
            std::size_t& cap = system.domain_particle_pool_capacities[d];
            // Grow immediately; shrink once well below (4x hysteresis). Same reason
            // as the foam pool: every surplus slot is a scale-0 instance that still
            // sits in the TLAS, so a pool that only ever grew left a draining/
            // settling fluid tracing against a giant stale instance set.
            if (want_cap > cap) {
                cap = want_cap;
            } else if (cap > kFluidPoolChunk && want_cap * 4 <= cap) {
                cap = want_cap;
            }

            std::vector<InstanceTransform>& inst = group->instances;
            const bool pool_grew = inst.size() != cap;
            if (pool_grew) {
                inst.resize(cap);
                structural_change = true;
            }

            const float voxel  = std::max(1e-4f, state.grid.voxel_size);
            const float radius = std::max(1e-4f,
                voxel * dconfig.fluid_particle_radius_factor *
                dconfig.fluid_particle_size_multiplier);
            const float diam = radius * 2.0f;  // unit-diameter primitives -> scale = diameter

            uint64_t content = 1469598103934665603ull;
            std::size_t drawn = 0;
            for (std::size_t i = 0; i < cap; ++i) {
                InstanceTransform& tr = inst[i];
                tr.rotation = Vec3(0.0f, 0.0f, 0.0f);
                tr.source_index = 0;
                if (i < draw_count) {
                    const Vec3& p = state.particles.position[i];
                    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                        tr.position = Vec3(0.0f, 0.0f, 0.0f);
                        tr.scale = Vec3(0.0f, 0.0f, 0.0f);
                        continue;
                    }
                    tr.position = p;
                    tr.scale = Vec3(diam, diam, diam);
                    content = hashCombine(content, quantize(p.x));
                    content = hashCombine(content, quantize(p.y));
                    content = hashCombine(content, quantize(p.z));
                    ++drawn;
                } else {
                    tr.position = Vec3(0.0f, 0.0f, 0.0f);
                    tr.scale = Vec3(0.0f, 0.0f, 0.0f);
                }
            }
            content = hashCombine(content, quantize(diam));
            content = hashCombine(content, static_cast<uint64_t>(drawn));

            if (structural_change) {
                st.content_hash = content;
                group->gpu_dirty = true;
            } else if (content != st.content_hash) {
                st.content_hash = content;
                group->gpu_dirty = true;
                motion_change = true;
            }
        }
    }

    if (structural_change) {
        g_geometry_dirty = true;
        g_vulkan_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_viewport_raster_rebuild_pending = true;   // refresh Solid/Matcap viewport too
        // Bump the scene geometry generation token — the backends gate their
        // accel/raster rebuild on it (VulkanBackend::buildRasterGeometryImpl
        // skips when m_rasterBuiltGeometryGeneration matches). Without the bump
        // a structural fluid change set the dirty bools but the backend saw an
        // unchanged generation and rebuilt the TLAS against a stale BLAS/source
        // set — a prime suspect for the cross-backend corruption.
        g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    } else if (motion_change) {
        g_gpu_refit_pending = true;
    }
    if (structural_change || motion_change) g_particle_cpu_geometry_dirty = true;
}

void SceneData::releaseDomainFluidParticleInstances() {
    auto& im = InstanceManager::getInstance();
    for (auto& system : particle_systems) {
        for (int gid : system.domain_particle_render_group_ids) {
            if (gid >= 0) {
                im.deleteGroup(gid);
                g_fluid_source_state.erase(gid);
            }
        }
        system.domain_particle_render_group_ids.clear();
        system.domain_particle_pool_capacities.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Whitewater (foam/spray/bubble) bridge — separate white-material spheres
// ─────────────────────────────────────────────────────────────────────────────
// Mirrors `state.foam` into a per-domain transient InstanceGroup with a fixed
// white scattering material. Independent of fluid_render_mode (foam draws over
// any liquid render). Same monotonic-chunked pool + scale-0 surplus contract as
// the liquid particle bridge so cheap TLAS refit stays index-valid.
void SceneData::syncFluidFoamRenderInstances(bool enable_rt_geometry) {
    if (g_optix_rebuild_in_progress.load() || g_viewport_rebuild_in_progress.load()) {
        return;
    }
    auto& im = InstanceManager::getInstance();

    bool structural_change = false;
    bool motion_change = false;

    for (auto& system : particle_systems) {
        if (!system.runtime) continue;
        const auto& states  = system.runtime->gridDomainStates();
        auto& domains       = system.runtime->gridDomains();

        for (std::size_t d = states.size();
             d < system.domain_foam_render_group_ids.size(); ++d) {
            const int gid = system.domain_foam_render_group_ids[d];
            if (gid >= 0) { im.deleteGroup(gid); g_fluid_source_state.erase(gid); structural_change = true; }
        }
        system.domain_foam_render_group_ids.resize(states.size(), -1);
        system.domain_foam_pool_capacities.resize(states.size(), 0);

        for (std::size_t d = 0; d < states.size(); ++d) {
            const auto& state   = states[d];
            const bool is_fluid = state.type == RayTrophiSim::SimulationDomainType::Fluid;
            // NOTE: do NOT gate on `!state.foam.empty()`. Foam is volatile — it is
            // culled by lifetime and re-spawned constantly, so its count oscillates
            // through 0 every few frames. Gating on non-empty here tore the group
            // DOWN on an empty frame and rebuilt it on the next non-empty one, and
            // each create/destroy is a STRUCTURAL change → a full backend AS
            // rebuild. During a sequence render (which rebuilds synchronously,
            // per frame) that thrashed into a render⇄rebuild loop. Keep the group
            // alive for the whole life of a foam-enabled domain; a transiently
            // empty frame just draws 0 instances (cheap motion refit, pool only
            // grows). The group is torn down only when foam is actually disabled.
            const auto& dconfig = domains[d];
            const auto& fparams = dconfig.fluid_foam_params;
            const std::string mat_name = "[FluidDomFoamMat] " + system.name + " D" + std::to_string(d);
            
            // Build the materials for all three FoamTypes.
            uint16_t mats[3];
            mats[0] = resolveFoamMaterial(fparams.spray_material_id, RayTrophiSim::Fluid::FoamType::Spray, mat_name);
            mats[1] = resolveFoamMaterial(fparams.foam_material_id, RayTrophiSim::Fluid::FoamType::Foam, mat_name);
            mats[2] = resolveFoamMaterial(fparams.bubble_material_id, RayTrophiSim::Fluid::FoamType::Bubble, mat_name);

            const bool base_ok  = enable_rt_geometry &&
                system.visible && system.enabled && state.valid && is_fluid &&
                d < domains.size() &&
                domains[d].fluid_foam_params.enabled;
            // Foam renders as instanced spheres EXCEPT in Volume mode, where it is
            // splatted into the fluid surface volume's temperature channel by
            // SceneData::syncSimulationRenderVolumes (no per-particle geometry). The
            // two paths are mutually exclusive — when Volume is selected we tear the
            // sphere group down so the foam isn't drawn twice.
            // Volume foam rides the fluid SURFACE volume's temperature channel, so it
            // only applies when the fluid is in Surface SDF render mode. With any other
            // fluid mode there is no host volume → fall back to spheres instead of
            // silently drawing nothing.
            const bool is_volume_foam =
                base_ok &&
                domains[d].fluid_foam_params.render_mode ==
                    RayTrophiSim::Fluid::FoamRenderMode::Volume &&
                domains[d].fluid_render_mode ==
                    RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
            const bool is_spheres = base_ok && !is_volume_foam;
            const bool wants = is_spheres;

            int& group_id = system.domain_foam_render_group_ids[d];

            if (!wants) {
                if (group_id >= 0) {
                    if (InstanceGroup* g = im.getGroup(group_id))
                        if (!g->instances.empty()) structural_change = true;
                    im.deleteGroup(group_id);
                    g_fluid_source_state.erase(group_id);
                    group_id = -1;
                    system.domain_foam_pool_capacities[d] = 0;
                }
                continue;
            }

            InstanceGroup* group = (group_id >= 0) ? im.getGroup(group_id) : nullptr;
            if (!group) {
                const std::string gname = "[FluidDomFoam] " + system.name + " D" + std::to_string(d);
                group_id = im.createGroup(gname, "", {});
                group = im.getGroup(group_id);
                if (!group) continue;
                group->transient = true;
                group->sources.clear();
                system.domain_foam_pool_capacities[d] = 0;
                g_fluid_source_state.erase(group_id);
            }
            group->transient = true;
            // OptiX renders foam as analytic sphere GAS (one per type) instead of
            // one TLAS instance per particle — see InstanceGroup::point_sphere_mode.
            // Vulkan / CPU ignore the flag and keep the icosphere instance path.
            group->point_sphere_mode = true;
            FluidSourceState& st = g_fluid_source_state[group_id];

            // ── SPHERES: three instanced sources per foam type. ──
            const int subdiv = std::max(0, std::min(fparams.foam_sphere_subdivisions, 3));
            uint64_t sig = hashCombine(1469598103934665603ull, 0xF0A1Bull);
            sig = hashCombine(sig, static_cast<uint64_t>(subdiv));
            sig = hashCombine(sig, static_cast<uint64_t>(mats[0]));
            sig = hashCombine(sig, static_cast<uint64_t>(mats[1]));
            sig = hashCombine(sig, static_cast<uint64_t>(mats[2]));

            if (group->sources.empty() || st.signature != sig) {
                group->sources.clear();
                group->sources.resize(3);
                
                const std::string type_names[3] = { "Spray", "Foam", "Bubble" };
                for (int t = 0; t < 3; ++t) {
                    const std::string geo_node = "[FluidDomFoamGeo_" + type_names[t] + "] " + system.name + " D" + std::to_string(d);
                    ScatterSource src;
                    src.name = geo_node;
                    src.weight = 1.0f;
                    buildIcosphere(subdiv, mats[t], geo_node, src.triangles);
                    for (auto& tri : src.triangles) if (tri) tri->setMaterialID(mats[t]);
                    src.computeCenter();
                    group->sources[t] = std::move(src);
                }
                st.signature = sig;
                structural_change = true;
            }

            const std::size_t live_count = state.foam.size();
            const std::size_t budget     = fluidRenderBudget(fparams.max_foam);
            const std::size_t draw_count = std::min(live_count, budget);
            const std::size_t want_cap   = fluidPoolCapacityFor(live_count, budget);
            std::size_t& cap = system.domain_foam_pool_capacities[d];
            // Grow immediately when foam exceeds the pool; SHRINK once foam has
            // dropped below it (2x hysteresis — one doubling band — so the pool
            // tracks the LIVE foam closely instead of the post-spike PEAK). Foam is
            // volatile — it spikes during a splash then dissipates. A grows-only pool
            // kept the post-spike peak forever, and EVERY surplus slot is a scale-0
            // instance that still lands in the TLAS (the backends don't cull
            // degenerate instances), so per-frame refit+trace cost scaled with the
            // PEAK, not the current foam — the "32k slower than 38k" non-monotonic
            // timing (cost followed pool cap = peak history, not live count) and the
            // "13 frames in it jumps to 15-20s and never recovers" cliff. A tighter
            // 2x band releases the pool one doubling step after the foam drops, at the
            // cost of a slightly more frequent shrink-rebuild (rare vs per-frame refit;
            // grow is still immediate so a re-spike just regrows). Was 4x.
            if (want_cap > cap) {
                cap = want_cap;
            } else if (cap > kFluidPoolChunk && want_cap * 2 <= cap) {
                cap = want_cap;
            }

            std::vector<InstanceTransform>& inst = group->instances;
            if (inst.size() != cap) { inst.resize(cap); structural_change = true; }

            const float voxel  = std::max(1e-4f, state.grid.voxel_size);
            const float radius = std::max(1e-4f, voxel * fparams.render_radius_voxels);
            const float diam   = radius * 2.0f;
            uint64_t content = 1469598103934665603ull;
            content = hashCombine(content, state.version);
            content = hashCombine(content, static_cast<uint64_t>(draw_count));
            content = hashCombine(content, quantize(diam));

            const std::vector<float>& sdf_buf = system.domain_sdf_buffers[d];
            const auto& lsp = dconfig.fluid_level_set_params;
            const int m = std::clamp(lsp.surface_resolution_multiplier, 1, 4);
            const int nx = state.grid.nx * m;
            const int ny = state.grid.ny * m;
            const int nz = state.grid.nz * m;
            const float sdf_voxel = (m > 1) ? (state.grid.voxel_size / static_cast<float>(m)) : state.grid.voxel_size;
            const Vec3 origin = state.grid.origin;
            const bool has_sdf = !sdf_buf.empty() && sdf_buf.size() == static_cast<size_t>(nx) * ny * nz;

            auto sampleSdfDensity = [&](const Vec3& pos) -> float {
                if (!has_sdf) return 0.0f;
                const Vec3 local = (pos - origin) / sdf_voxel - Vec3(0.5f, 0.5f, 0.5f);
                const int i0 = static_cast<int>(std::floor(local.x));
                const int j0 = static_cast<int>(std::floor(local.y));
                const int k0 = static_cast<int>(std::floor(local.z));
                const float fx = local.x - i0;
                const float fy = local.y - j0;
                const float fz = local.z - k0;
                
                auto valAt = [&](int idx_i, int idx_j, int idx_k) -> float {
                    idx_i = std::max(0, std::min(nx - 1, idx_i));
                    idx_j = std::max(0, std::min(ny - 1, idx_j));
                    idx_k = std::max(0, std::min(nz - 1, idx_k));
                    return sdf_buf[static_cast<size_t>(idx_i) + static_cast<size_t>(idx_j) * nx + static_cast<size_t>(idx_k) * nx * ny];
                };

                const float c000 = valAt(i0, j0, k0);
                const float c100 = valAt(i0 + 1, j0, k0);
                const float c010 = valAt(i0, j0 + 1, k0);
                const float c110 = valAt(i0 + 1, j0 + 1, k0);
                const float c001 = valAt(i0, j0, k0 + 1);
                const float c101 = valAt(i0 + 1, j0, k0 + 1);
                const float c011 = valAt(i0, j0 + 1, k0 + 1);
                const float c111 = valAt(i0 + 1, j0 + 1, k0 + 1);

                const float c00 = c000 * (1.0f - fx) + c100 * fx;
                const float c10 = c010 * (1.0f - fx) + c110 * fx;
                const float c01 = c001 * (1.0f - fx) + c101 * fx;
                const float c11 = c011 * (1.0f - fx) + c111 * fx;

                const float c0 = c00 * (1.0f - fy) + c10 * fy;
                const float c1 = c01 * (1.0f - fy) + c11 * fy;

                return c0 * (1.0f - fz) + c1 * fz;
            };

            auto sampleSdfGradient = [&](const Vec3& pos) -> Vec3 {
                const float eps = sdf_voxel * 0.25f;
                const float dx = sampleSdfDensity(pos + Vec3(eps, 0.0f, 0.0f)) - sampleSdfDensity(pos - Vec3(eps, 0.0f, 0.0f));
                const float dy = sampleSdfDensity(pos + Vec3(0.0f, eps, 0.0f)) - sampleSdfDensity(pos - Vec3(0.0f, eps, 0.0f));
                const float dz = sampleSdfDensity(pos + Vec3(0.0f, 0.0f, eps)) - sampleSdfDensity(pos - Vec3(0.0f, 0.0f, eps));
                return Vec3(dx, dy, dz) * (0.5f / eps);
            };

            std::size_t drawn = 0;
            for (std::size_t i = 0; i < draw_count; ++i) {
                InstanceTransform& tr = inst[i];
                tr.rotation = Vec3(0.0f, 0.0f, 0.0f);
                
                const uint8_t foam_type = state.foam.type[i];
                tr.source_index = std::min(2, static_cast<int>(foam_type));
                
                Vec3 p = state.foam.position[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                    tr.position = Vec3(0.0f, 0.0f, 0.0f);
                    tr.scale = Vec3(0.0f, 0.0f, 0.0f);
                    continue;
                }
                
                // Snap Foam (surface foam) particles to the reconstructed liquid surface
                if (has_sdf && foam_type == static_cast<uint8_t>(RayTrophiSim::Fluid::FoamType::Foam)) {
                    const float D = sampleSdfDensity(p);
                    // The liquid surface boundary is at D = 0.5f.
                    // Only perform projection if the particle is near the narrow band (D > 0.01f && D < 0.99f).
                    if (D > 0.01f && D < 0.99f) {
                        const Vec3 grad = sampleSdfGradient(p);
                        const float lenSq = grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
                        if (lenSq > 1e-5f) {
                            Vec3 corr = grad * (-(D - 0.5f) / lenSq);
                            const float max_corr = sdf_voxel * 2.0f; // maximum 2 voxels correction
                            const float corr_len = corr.length();
                            if (corr_len > max_corr) {
                                corr = corr * (max_corr / corr_len);
                            }
                            p = p + corr;
                        }
                    }
                }
                
                tr.position = p;
                
                // Stable per-slot hash for size variation (0.6 to 1.4)
                const uint64_t slot_hash = hashCombine(static_cast<uint64_t>(i) + 1469u,
                                                       static_cast<uint64_t>(system.id) + 7919u);
                const float size_var = 0.6f + 0.8f * (static_cast<float>(slot_hash % 1000ull) / 1000.0f);
                
                // Type specific scale: Spray = 0.5, Foam = 0.8, Bubble = 1.0
                float type_scale = 1.0f;
                if (foam_type == static_cast<uint8_t>(RayTrophiSim::Fluid::FoamType::Spray)) {
                    type_scale = 0.5f;
                } else if (foam_type == static_cast<uint8_t>(RayTrophiSim::Fluid::FoamType::Foam)) {
                    type_scale = 0.8f;
                }
                
                // Dissolving/Popping scale: if lifetime < 0.5s, shrink to zero
                float dissolve_scale = 1.0f;
                const float life = state.foam.lifetime[i];
                if (life < 0.5f && life > 0.0f) {
                    dissolve_scale = life / 0.5f;
                } else if (life <= 0.0f) {
                    dissolve_scale = 0.0f;
                }
                
                const float p_diam = diam * size_var * type_scale * dissolve_scale;
                tr.scale = Vec3(p_diam, p_diam, p_diam);
                ++drawn;
            }
            const std::size_t clear_begin = structural_change
                ? draw_count
                : std::min(draw_count, st.drawn_count);
            const std::size_t clear_end = structural_change
                ? cap
                : std::min(cap, std::max(draw_count, st.drawn_count));
            for (std::size_t i = clear_begin; i < clear_end; ++i) {
                InstanceTransform& tr = inst[i];
                tr.position = Vec3(0.0f, 0.0f, 0.0f);
                tr.rotation = Vec3(0.0f, 0.0f, 0.0f);
                tr.scale = Vec3(0.0f, 0.0f, 0.0f);
                tr.source_index = 0;
            }

            if (structural_change) {
                st.content_hash = content;
                group->gpu_dirty = true;
            } else if (content != st.content_hash || drawn != st.drawn_count) {
                st.content_hash = content;
                group->gpu_dirty = true;
                motion_change = true;
            }
            st.drawn_count = drawn;
        }
    }

    if (structural_change) {
        g_geometry_dirty = true;
        g_vulkan_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_viewport_raster_rebuild_pending = true;   // refresh Solid/Matcap viewport too
        g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    } else if (motion_change) {
        g_gpu_refit_pending = true;
    }
    if (structural_change || motion_change) g_particle_cpu_geometry_dirty = true;
}

void SceneData::releaseDomainFluidFoamInstances() {
    auto& im = InstanceManager::getInstance();
    for (auto& system : particle_systems) {
        for (int gid : system.domain_foam_render_group_ids) {
            if (gid >= 0) { im.deleteGroup(gid); g_fluid_source_state.erase(gid); }
        }
        system.domain_foam_render_group_ids.clear();
        system.domain_foam_pool_capacities.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference render bridge
// ─────────────────────────────────────────────────────────────────────────────
// The GPU backends draw discrete particles by iterating InstanceManager's
// transient groups directly. The CPU BVH, however, is built only from
// world.objects (InstanceManager::rebuildSceneObjects deliberately skips
// transient groups so they don't churn the selection list / get re-centered).
// So the CPU reference renderer never saw particles. This walks every transient
// particle/fluid/foam group and expands its live instances into HittableInstances
// that the caller appends to the CPU-BVH-only snapshot — particles thus reach the
// CPU render WITHOUT entering world.objects (no GPU double-render, no UI churn).
//
// Each primitive ScatterSource already holds origin-centred unit-diameter
// triangles with the bucket material baked in. We build ONE child EmbreeBVH per
// source (cached on source.bvh, reused until the bridge clears sources on a
// config change) and instance it per particle via the same machinery foliage
// uses. Embree natively instances the child scene; the ParallelBVH fallback uses
// HittableInstance's object-space ray transform — both work with the same child.
void SceneData::appendParticleCPUHittables(std::vector<std::shared_ptr<Hittable>>& out) {
    auto& im = InstanceManager::getInstance();
    for (auto& group : im.getGroups()) {
        if (!group.transient) continue;            // particle/fluid/foam bridges only
        if (group.instances.empty() || group.sources.empty()) continue;

        // Lazily build + cache a child BVH per source. The primitive triangles are
        // already centred on the origin, so (unlike the foliage path) we build the
        // BVH directly with no re-centering — instance scale == particle diameter
        // then places the sphere/cube at the particle position.
        for (auto& src : group.sources) {
            if (src.bvh || src.triangles.empty()) continue;
            std::vector<std::shared_ptr<Hittable>> prim;
            prim.reserve(src.triangles.size());
            for (const auto& tri : src.triangles) {
                if (tri) prim.push_back(tri);
            }
            if (prim.empty()) continue;
            auto embree = std::make_shared<EmbreeBVH>();
            embree->build(prim);
            src.bvh = embree;
            src.has_local_bbox = src.bvh->bounding_box(0.0f, 0.0f, src.local_bbox);
            // HittableInstance keeps a triangle list for normal/material fallbacks;
            // the centred primitives double as that list.
            src.centered_triangles_ptr =
                std::make_shared<std::vector<std::shared_ptr<Triangle>>>(src.triangles);
        }

        const int nsrc = static_cast<int>(group.sources.size());
        out.reserve(out.size() + group.instances.size());
        for (const auto& inst : group.instances) {
            // Dead/surplus slots collapse to scale 0 in every bridge — skip them so
            // the CPU BVH only carries live particles.
            if (inst.scale.x <= 1e-5f) continue;
            int si = inst.source_index;
            if (si < 0 || si >= nsrc) si = 0;
            ScatterSource& src = group.sources[static_cast<std::size_t>(si)];
            if (!src.bvh) continue;
            const Matrix4x4 mat = inst.toMatrix();
            if (src.has_local_bbox) {
                out.push_back(std::make_shared<HittableInstance>(
                    src.bvh, src.centered_triangles_ptr, mat, "[ParticleCPU]", src.local_bbox));
            } else {
                out.push_back(std::make_shared<HittableInstance>(
                    src.bvh, src.centered_triangles_ptr, mat, "[ParticleCPU]"));
            }
        }
    }
}

void SceneData::rebuildParticleBVH(bool use_embree) {
    std::vector<std::shared_ptr<Hittable>> parts;
    appendParticleCPUHittables(parts);
    if (parts.empty()) {
        particle_bvh = nullptr;   // composite falls back to the static scene BVH
        return;
    }
    if (use_embree) {
        auto bvh = std::make_shared<EmbreeBVH>();
        bvh->build(parts);
        particle_bvh = bvh;
    } else {
        particle_bvh = std::make_shared<ParallelBVHNode>(parts, 0, parts.size(), 0.0f, 1.0f, 0);
    }
}
