/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApi.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*/
#include "Api/RtApi.h"
#include "RtApiInternal.h"

#include <deque>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include "scene_ui.h"          // UIContext, SceneHistory
#include "SceneCommand.h"      // TransformCommand / TransformState
#include "Backend/IBackend.h"  // backend resetAccumulation
#include "TriangleMesh.h"
#include "Camera.h"            // Faz 5.1a: active-camera get/set
#include "World.h"             // Faz 5.1c: NishitaSkyParams / background (rt.world)
#include "GeometryNodesV2.h"   // rebakeFromOrig / recomputeOrigNormals; GeometryNodeGraphV2 (Faz 3d)
#include "MaterialNodesV2.h"   // MaterialNodeGraphV2 (Faz 3d node-graph construction)
#include "TerrainManager.h"
#include "TerrainNodesV2.h"
#include "NodeSystem/Graph.h"  // GraphBase: registerNode / addLink / markAllDirty
#include "NodeSystem/NodeRegistry.h"  // typeId -> factory (Faz 3d)
#include "KeyframeSystem.h"    // TimelineManager / Keyframe / TransformKeyframe (Faz 3c)
#include "ProjectManager.h"
#include "Api/RtPython.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "PBRMaterialSnapshot.h"
#include "ColorProcessingParams.h"  // Faz 5.1d: rt.post parameters
#include "MeshModifiers.h"          // Faz 5.2b: rt.modifiers
#include "InstanceManager.h"        // Faz 5.2c: rt.scatter
#include "InstanceGroup.h"          // Faz 5.2c: rt.scatter
#include "RigidBodySystem.h"        // Faz 5.3a: rt.physics
#include "Fluid/FluidObject.h"      // Faz 5.3b: rt.fluid
#include "Fluid/FluidSimulationSystem.h" // Faz 5.3b: rt.fluid
#include "Fluid/APICFluidSolver.h"  // Faz 5.3b: rt.fluid

// Defined by Core/Main.cpp and scene_ui_stylize.cpp.
extern SceneUI ui;
extern bool stylize_redisplay;
extern std::atomic<bool> rendering_in_progress;
extern std::atomic<bool> rendering_stopped_cpu;
extern std::atomic<bool> rendering_stopped_gpu;

// Viewport-driven sequence save state machine globals (Main.cpp file scope).
// Accessed through the API facade so Python/CLI automation does not reach Main.cpp
// internals directly.
extern bool        g_seq_save_active;
extern int         g_seq_save_frame;
extern int         g_seq_save_end;
extern std::string g_seq_save_dir;
extern bool        g_seq_save_denoise;
extern bool        g_camera_dirty;   // Core/Main.cpp — arms backend camera-buffer resync
extern bool        g_world_dirty;    // Core/Main.cpp — arms backend world/sky resync
extern bool        g_lights_dirty;   // Core/Main.cpp — arms backend light-buffer resync
extern bool        g_solid_viewport_active;
extern bool        g_geometry_dirty;
extern std::atomic<uint64_t> g_scene_geometry_generation;
extern bool        g_bvh_rebuild_pending;
extern bool        g_optix_rebuild_pending;
extern bool        g_vulkan_rebuild_pending;
extern bool        g_viewport_raster_rebuild_pending;

namespace rtapi {

// g_ctx and g_history have external linkage within rtapi so they can be
// accessed by other workspace components (like RtConsole.cpp)
UIContext* g_ctx = nullptr;
SceneHistory* g_history = nullptr;
RenderJobInfo g_render_job;

namespace {

std::mutex g_queue_mutex;
std::deque<std::function<void(UIContext&)>> g_queue;

// Event callbacks (Faz 3b). Registered/fired on the main thread only, so no
// lock is needed. int keys are handed back to callers for unsubscription.
std::vector<std::pair<int, std::function<void(int)>>> g_frame_change_cbs;
std::vector<std::pair<int, std::function<void()>>>    g_scene_load_cbs;
int  g_next_cb_id = 1;
int  g_last_seen_frame = -1;     // drain-poll baseline; -1 = uninitialised
bool g_scene_loaded_pending = false;  // set by notifySceneLoaded(), drained next frame

TriangleMesh* findFlatMesh(UIContext& ctx, const std::string& name) {
    for (auto& obj : ctx.scene.world.objects) {
        if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (tm->nodeName == name) return tm.get();
        }
    }
    return nullptr;
}

enum class MaterialParamKind {
    BaseColor,
    Roughness,
    Metallic,
    Specular,
    Emission,
    EmissionStrength,
    Transmission,
    Ior,
    Opacity
};

struct MaterialValue {
    float scalar = 0.0f;
    Vec3 color;
};

struct MaterialEdit {
    uint16_t material_id = MaterialManager::INVALID_MATERIAL_ID;
    MaterialValue before;
    MaterialValue after;
};

bool parseMaterialParam(const std::string& name, MaterialParamKind& out, bool& is_color) {
    is_color = false;
    if (name == "base_color") { out = MaterialParamKind::BaseColor; is_color = true; return true; }
    if (name == "roughness") { out = MaterialParamKind::Roughness; return true; }
    if (name == "metallic") { out = MaterialParamKind::Metallic; return true; }
    if (name == "specular") { out = MaterialParamKind::Specular; return true; }
    if (name == "emission") { out = MaterialParamKind::Emission; is_color = true; return true; }
    if (name == "emission_strength") { out = MaterialParamKind::EmissionStrength; return true; }
    if (name == "transmission") { out = MaterialParamKind::Transmission; return true; }
    if (name == "ior") { out = MaterialParamKind::Ior; return true; }
    if (name == "opacity") { out = MaterialParamKind::Opacity; return true; }
    return false;
}

MaterialValue readMaterialValue(const PrincipledBSDF& material, MaterialParamKind kind) {
    MaterialValue value;
    switch (kind) {
        case MaterialParamKind::BaseColor:        value.color = material.albedoProperty.color; break;
        case MaterialParamKind::Roughness:        value.scalar = material.getScalarRoughness(); break;
        case MaterialParamKind::Metallic:         value.scalar = material.getScalarMetallic(); break;
        case MaterialParamKind::Specular:         value.scalar = material.specularProperty.intensity; break;
        case MaterialParamKind::Emission:         value.color = material.emissionProperty.color; break;
        case MaterialParamKind::EmissionStrength: value.scalar = material.emissionProperty.intensity; break;
        case MaterialParamKind::Transmission:     value.scalar = material.transmission; break;
        case MaterialParamKind::Ior:              value.scalar = material.ior; break;
        case MaterialParamKind::Opacity:          value.scalar = material.opacityProperty.alpha; break;
    }
    return value;
}

void writeMaterialValue(PrincipledBSDF& material, MaterialParamKind kind, const MaterialValue& value) {
    switch (kind) {
        case MaterialParamKind::BaseColor:        material.albedoProperty.color = value.color; break;
        case MaterialParamKind::Roughness:        material.roughnessProperty.color = Vec3(value.scalar); break;
        case MaterialParamKind::Metallic:         material.metallicProperty.intensity = value.scalar; break;
        case MaterialParamKind::Specular:         material.specularProperty.intensity = value.scalar; break;
        case MaterialParamKind::Emission:         material.emissionProperty.color = value.color; break;
        case MaterialParamKind::EmissionStrength: material.emissionProperty.intensity = value.scalar; break;
        case MaterialParamKind::Transmission:     material.setTransmission(value.scalar, material.ior); break;
        case MaterialParamKind::Ior:              material.setTransmission(material.transmission, value.scalar); break;
        case MaterialParamKind::Opacity:          material.opacityProperty.alpha = value.scalar; break;
    }
    if (!material.gpuMaterial) material.gpuMaterial = std::make_shared<GpuMaterial>();
    applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(material), *material.gpuMaterial);
}

std::vector<uint16_t> objectMaterialIds(UIContext& ctx, const std::string& object_name) {
    std::vector<uint16_t> ids;
    std::unordered_set<uint16_t> seen;
    for (const auto& object : ctx.scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || mesh->nodeName != object_name || !mesh->geometry) continue;
        const uint16_t* material_ids = mesh->geometry->get_material_ids();
        const size_t vertex_count = mesh->num_vertices();
        if (!material_ids) {
            if (MaterialManager::getInstance().getMaterial(0) && seen.insert(0).second) ids.push_back(0);
            continue;
        }
        for (size_t i = 0; i < vertex_count; ++i) {
            const uint16_t id = material_ids[i];
            if (id != MaterialManager::INVALID_MATERIAL_ID && seen.insert(id).second) ids.push_back(id);
        }
    }
    return ids;
}

class MaterialParamCommand final : public SceneCommand {
public:
    MaterialParamCommand(std::string object_name, std::string param_name,
                         MaterialParamKind kind, std::vector<MaterialEdit> edits)
        : object_name_(std::move(object_name)), param_name_(std::move(param_name)),
          kind_(kind), edits_(std::move(edits)) {}

    void execute(UIContext& ctx) override { apply(ctx, false); }
    void undo(UIContext& ctx) override { apply(ctx, true); }
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override {
        return "Set " + param_name_ + " on " + object_name_;
    }

private:
    void apply(UIContext& ctx, bool before) {
        for (const MaterialEdit& edit : edits_) {
            auto* material = dynamic_cast<PrincipledBSDF*>(
                MaterialManager::getInstance().getMaterial(edit.material_id));
            if (!material) continue;
            writeMaterialValue(*material, kind_, before ? edit.before : edit.after);
            ctx.renderer.updateBackendMaterial(ctx.scene, edit.material_id);
        }
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
        ctx.start_render = true;
        ProjectManager::getInstance().markModified();
    }

    std::string object_name_;
    std::string param_name_;
    MaterialParamKind kind_;
    std::vector<MaterialEdit> edits_;
};

} // namespace

Version version() { return {}; }

void bind(UIContext* ctx, SceneHistory* history) {
    g_ctx = ctx;
    g_history = history;
    g_render_job = {};
}

void unbind() {
    g_ctx = nullptr;
    g_history = nullptr;
    g_render_job = {};
    clearEventCallbacks();
    g_last_seen_frame = -1;
    g_scene_loaded_pending = false;
    std::lock_guard<std::mutex> lock(g_queue_mutex);
    g_queue.clear();
}

bool isBound() { return g_ctx != nullptr; }

void enqueue(std::function<void(UIContext&)> fn) {
    if (!fn) return;
    std::lock_guard<std::mutex> lock(g_queue_mutex);
    g_queue.push_back(std::move(fn));
}

void drainMainThreadQueue() {
    if (!g_ctx) return;
    // Swap out under the lock, run without it: a callback may enqueue again.
    std::deque<std::function<void(UIContext&)>> pending;
    {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        pending.swap(g_queue);
    }
    for (auto& fn : pending) fn(*g_ctx);

    // Terrain graph workers finish off-thread, but splat/river/mesh publication
    // must be finalized from this once-per-frame main-thread hook.
    pollTerrainEvaluations();

    // Event dispatch (Faz 3b). Runs here because drainMainThreadQueue is already
    // the once-per-frame, scene-load-gated, main-thread hook.
    //
    // Dispatch by MOVE-out / MOVE-back, never copy: a callback holds a py::function,
    // and copying one mutates a Python refcount (needs the GIL, which this drain does
    // not necessarily hold). std::move touches no refcounts. Moving the list out also
    // makes dispatch re-entrancy-safe — a callback that subscribes lands in the now-
    // empty member list and is appended back afterwards.
    if (g_scene_loaded_pending) {
        g_scene_loaded_pending = false;
        g_last_seen_frame = ui.timeline.getCurrentFrame();  // rebaseline; a load is not a frame change

        std::vector<std::pair<int, std::function<void()>>> local;
        local.swap(g_scene_load_cbs);
        for (auto& [id, fn] : local) if (fn) fn();
        for (auto& e : local) g_scene_load_cbs.push_back(std::move(e));
    }

    const int frame = ui.timeline.getCurrentFrame();
    if (g_last_seen_frame < 0) {
        g_last_seen_frame = frame;  // first drain establishes the baseline silently
    } else if (frame != g_last_seen_frame) {
        g_last_seen_frame = frame;

        std::vector<std::pair<int, std::function<void(int)>>> local;
        local.swap(g_frame_change_cbs);
        for (auto& [id, fn] : local) if (fn) fn(frame);
        for (auto& e : local) g_frame_change_cbs.push_back(std::move(e));
    }
}

// ---------------------------------------------------------------------------
// Events (Faz 3b).
// ---------------------------------------------------------------------------
int addFrameChangeCallback(std::function<void(int)> fn) {
    if (!fn) return -1;
    const int id = g_next_cb_id++;
    g_frame_change_cbs.emplace_back(id, std::move(fn));
    return id;
}

int addSceneLoadCallback(std::function<void()> fn) {
    if (!fn) return -1;
    const int id = g_next_cb_id++;
    g_scene_load_cbs.emplace_back(id, std::move(fn));
    return id;
}

void removeFrameChangeCallback(int id) {
    g_frame_change_cbs.erase(
        std::remove_if(g_frame_change_cbs.begin(), g_frame_change_cbs.end(),
                       [id](const auto& e) { return e.first == id; }),
        g_frame_change_cbs.end());
}

void removeSceneLoadCallback(int id) {
    g_scene_load_cbs.erase(
        std::remove_if(g_scene_load_cbs.begin(), g_scene_load_cbs.end(),
                       [id](const auto& e) { return e.first == id; }),
        g_scene_load_cbs.end());
}

void clearEventCallbacks() {
    g_frame_change_cbs.clear();
    g_scene_load_cbs.clear();
}

void notifySceneLoaded() {
    g_scene_loaded_pending = true;
}

std::vector<std::string> listObjects() {
    std::vector<std::string> names;
    if (!g_ctx) return names;
    std::unordered_set<std::string> seen;
    for (auto& obj : g_ctx->scene.world.objects) {
        if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (tm->nodeName.empty()) continue;
            if (g_ctx->scene.isEditorPendingDeleteObjectName(tm->nodeName)) continue;
            if (seen.insert(tm->nodeName).second) names.push_back(tm->nodeName);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            const std::string& name = tri->getNodeName();
            if (name.empty()) continue;
            if (g_ctx->scene.isEditorPendingDeleteObjectName(name)) continue;
            if (seen.insert(name).second) names.push_back(name);
        }
    }
    return names;
}

bool objectExists(const std::string& name) {
    if (!g_ctx || name.empty()) return false;
    if (g_ctx->scene.isEditorPendingDeleteObjectName(name)) return false;
    if (findFlatMesh(*g_ctx, name)) return true;
    for (const auto& obj : g_ctx->scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (tri->getNodeName() == name) return true;
        }
    }
    return false;
}

Result getObjectInfo(const std::string& name, ObjectInfo& out) {
    if (!g_ctx) return notBound();
    TriangleMesh* tm = objectExists(name) ? findFlatMesh(*g_ctx, name) : nullptr;
    if (!tm) return Result::fail("object not found: " + name);

    out = {};
    out.name = name;
    out.triangle_count = tm->num_triangles();
    out.vertex_count = tm->num_vertices();
    return Result::success();
}

// ---------------------------------------------------------------------------
// Mesh data (Faz 3a).
// ---------------------------------------------------------------------------
Result getMeshPositions(const std::string& name, MeshBufferView& out) {
    if (!g_ctx) return notBound();
    TriangleMesh* tm = findFlatMesh(*g_ctx, name);
    if (!tm || !tm->geometry) return Result::fail("object not found: " + name);
    const Vec3* p = tm->geometry->get_positions_orig();
    if (!p) return Result::fail("object has no position data: " + name);
    out.data = const_cast<float*>(reinterpret_cast<const float*>(p));
    out.vertex_count = tm->geometry->get_vertex_count();
    out.components = 3;
    return Result::success();
}

Result getMeshNormals(const std::string& name, MeshBufferView& out) {
    if (!g_ctx) return notBound();
    TriangleMesh* tm = findFlatMesh(*g_ctx, name);
    if (!tm || !tm->geometry) return Result::fail("object not found: " + name);
    const Vec3* n = tm->geometry->get_normals_orig();
    if (!n) return Result::fail("object has no normal data: " + name);
    out.data = const_cast<float*>(reinterpret_cast<const float*>(n));
    out.vertex_count = tm->geometry->get_vertex_count();
    out.components = 3;
    return Result::success();
}

Result getMeshUVs(const std::string& name, MeshBufferView& out) {
    if (!g_ctx) return notBound();
    TriangleMesh* tm = findFlatMesh(*g_ctx, name);
    if (!tm || !tm->geometry) return Result::fail("object not found: " + name);
    const Vec2* uv = tm->geometry->get_uvs();
    if (!uv) return Result::fail("object has no UV data: " + name);
    out.data = const_cast<float*>(reinterpret_cast<const float*>(uv));
    out.vertex_count = tm->geometry->get_vertex_count();
    out.components = 2;
    return Result::success();
}

namespace {

Result checkMeshWrite(TriangleMesh*& tm, const std::string& name, size_t vertex_count) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    tm = findFlatMesh(*g_ctx, name);
    if (!tm || !tm->geometry) return Result::fail("object not found: " + name);
    const size_t expected = tm->geometry->get_vertex_count();
    if (vertex_count != expected) {
        return Result::fail("vertex count mismatch for '" + name + "': expected " +
                             std::to_string(expected) + ", got " + std::to_string(vertex_count));
    }
    return Result::success();
}

} // namespace

Result setMeshPositions(const std::string& name, const float* data, size_t vertex_count) {
    if (!data) return Result::fail("positions data is null");
    TriangleMesh* tm = nullptr;
    Result err = checkMeshWrite(tm, name, vertex_count);
    if (!err) return err;

    Vec3* po = tm->geometry->get_attribute_data_mut<Vec3>("P_orig");
    if (!po) return Result::fail("object has no P_orig buffer: " + name);
    std::memcpy(po, data, vertex_count * sizeof(Vec3));
    if (tm->transform) GeometryNodesV2::rebakeFromOrig(*tm);
    scheduleSceneMutationRebuilds(*g_ctx, true);
    return Result::success();
}

Result setMeshNormals(const std::string& name, const float* data, size_t vertex_count) {
    if (!data) return Result::fail("normals data is null");
    TriangleMesh* tm = nullptr;
    Result err = checkMeshWrite(tm, name, vertex_count);
    if (!err) return err;

    Vec3* no = tm->geometry->get_attribute_data_mut<Vec3>("N_orig");
    if (!no) return Result::fail("object has no N_orig buffer: " + name);
    std::memcpy(no, data, vertex_count * sizeof(Vec3));
    if (tm->transform) GeometryNodesV2::rebakeFromOrig(*tm);
    scheduleSceneMutationRebuilds(*g_ctx, true);
    return Result::success();
}

Result setMeshUVs(const std::string& name, const float* data, size_t vertex_count) {
    if (!data) return Result::fail("uvs data is null");
    TriangleMesh* tm = nullptr;
    Result err = checkMeshWrite(tm, name, vertex_count);
    if (!err) return err;

    Vec2* uv = tm->geometry->get_uvs_mut();
    if (!uv) return Result::fail("object has no UV buffer: " + name);
    std::memcpy(uv, data, vertex_count * sizeof(Vec2));
    scheduleSceneMutationRebuilds(*g_ctx, false);
    return Result::success();
}

Result recomputeMeshNormals(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    TriangleMesh* tm = findFlatMesh(*g_ctx, name);
    if (!tm || !tm->geometry) return Result::fail("object not found: " + name);

    GeometryNodesV2::recomputeOrigNormals(*tm->geometry);
    if (tm->transform) GeometryNodesV2::rebakeFromOrig(*tm);
    scheduleSceneMutationRebuilds(*g_ctx, true);
    return Result::success();
}

namespace {

// Flat-only transform lookup: every scene node type (skinned, terrain, water,
// procedural, imported) is a flat SoA TriangleMesh — the API has no facade path.
Transform* findObjectTransform(UIContext& ctx, const std::string& name, Result& err) {
    TriangleMesh* fm = findFlatMesh(ctx, name);
    if (!fm) {
        err = Result::fail("object not found: " + name);
        return nullptr;
    }
    if (!fm->transform) {
        err = Result::fail("object has no transform handle: " + name);
        return nullptr;
    }
    return fm->transform.get();
}

} // namespace

Result getObjectTransform(const std::string& name, Matrix4x4& out) {
    if (!g_ctx) return notBound();
    Result err;
    Transform* t = findObjectTransform(*g_ctx, name, err);
    if (!t) return err;
    out = t->base;
    return Result::success();
}

Result setObjectTransform(const std::string& name, const Matrix4x4& matrix) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");

    Result err;
    Transform* t = findObjectTransform(*g_ctx, name, err);
    if (!t) return err;

    TransformState old_state;
    old_state.matrix = t->base;
    TransformState new_state;
    new_state.matrix = matrix;

    // Same sequence as the gizmo drag-commit path: apply through the command
    // (so backend TLAS/refit bookkeeping runs), then record it for undo.
    auto cmd = std::make_unique<TransformCommand>(name, old_state, new_state);
    cmd->execute(*g_ctx);
    g_history->record(std::move(cmd));
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result deleteObject(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");
    if (!objectExists(name)) return Result::fail("object not found: " + name);

    // Empty facade list: flat nodes are captured inside the command's execute().
    auto cmd = std::make_unique<DeleteObjectCommand>(name, std::vector<std::shared_ptr<Triangle>>{});
    cmd->execute(*g_ctx);
    g_history->record(std::move(cmd));
    return Result::success();
}

Result duplicateObject(const std::string& name, std::string& out_new_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    out_new_name.clear();

    // Locate the flat mesh + its world index, then drive the UI's own duplicate
    // flow through the selection — it owns the deep SoA copy, selection-cache
    // registration and the incremental backend clone; reimplementing that here
    // would rot the moment the UI path evolves.
    std::shared_ptr<TriangleMesh> mesh;
    int index = -1;
    for (size_t i = 0; i < g_ctx->scene.world.objects.size(); ++i) {
        if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(g_ctx->scene.world.objects[i])) {
            if (tm->nodeName == name &&
                !g_ctx->scene.isEditorPendingDeleteObjectName(name)) {
                mesh = tm;
                index = static_cast<int>(i);
                break;
            }
        }
    }
    if (!mesh) return Result::fail("object not found: " + name);

    auto rep = std::make_shared<Triangle>(mesh, 0u);
    g_ctx->selection.clearSelection();
    g_ctx->selection.selectObject(mesh, index, name, 0u, rep);

    ui.triggerDuplicate(*g_ctx);

    if (g_ctx->selection.hasSelection() && g_ctx->selection.selected.name != name) {
        out_new_name = g_ctx->selection.selected.name;
        return Result::success();
    }
    return Result::fail("duplicate failed for: " + name);
}

Result importModel(const std::string& filepath) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    const bool ok = ProjectManager::getInstance().importModel(
        filepath, g_ctx->scene, g_ctx->renderer, g_ctx->backend_ptr);
    if (!ok) return Result::fail("import failed: " + filepath);
    ui.mesh_cache_valid = false;   // new nodes need fresh selection caches
    return Result::success();
}

namespace {

std::shared_ptr<TriangleMesh> createPrimitiveMesh(const std::string& type_in, float size, const std::string& name) {
    std::string type = type_in;
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    auto mesh = std::make_shared<TriangleMesh>();
    mesh->nodeName = name;
    mesh->transform = std::make_shared<Transform>();

    mesh->geometry->add_attribute<Vec3>("P");
    mesh->geometry->add_attribute<Vec3>("N");
    mesh->geometry->add_attribute<Vec3>("P_orig");
    mesh->geometry->add_attribute<Vec3>("N_orig");
    mesh->geometry->add_attribute<Vec2>("uv");
    mesh->geometry->add_attribute<uint16_t>("materialID");

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    auto& indices = mesh->geometry->indices;
    indices.clear();

    const float hs = size * 0.5f;
    constexpr float M_PI_F = 3.14159265358979323846f;

    if (type == "cube" || type == "box") {
        static const Vec3 face_normals[6] = {
            Vec3(0, 0, 1), Vec3(0, 0, -1),
            Vec3(1, 0, 0), Vec3(-1, 0, 0),
            Vec3(0, 1, 0), Vec3(0, -1, 0)
        };
        const Vec3 face_verts[6][4] = {
            { Vec3(-hs, -hs,  hs), Vec3( hs, -hs,  hs), Vec3( hs,  hs,  hs), Vec3(-hs,  hs,  hs) },
            { Vec3( hs, -hs, -hs), Vec3(-hs, -hs, -hs), Vec3(-hs,  hs, -hs), Vec3( hs,  hs, -hs) },
            { Vec3( hs, -hs,  hs), Vec3( hs, -hs, -hs), Vec3( hs,  hs, -hs), Vec3( hs,  hs,  hs) },
            { Vec3(-hs, -hs, -hs), Vec3(-hs, -hs,  hs), Vec3(-hs,  hs,  hs), Vec3(-hs,  hs, -hs) },
            { Vec3(-hs,  hs,  hs), Vec3( hs,  hs,  hs), Vec3( hs,  hs, -hs), Vec3(-hs,  hs, -hs) },
            { Vec3(-hs, -hs, -hs), Vec3( hs, -hs, -hs), Vec3( hs, -hs,  hs), Vec3(-hs, -hs,  hs) }
        };
        static const Vec2 face_uvs[4] = {
            Vec2(0.0f, 0.0f), Vec2(1.0f, 0.0f), Vec2(1.0f, 1.0f), Vec2(0.0f, 1.0f)
        };

        for (int f = 0; f < 6; ++f) {
            uint32_t base_idx = static_cast<uint32_t>(positions.size());
            for (int v = 0; v < 4; ++v) {
                positions.push_back(face_verts[f][v]);
                normals.push_back(face_normals[f]);
                uvs.push_back(face_uvs[v]);
            }
            indices.push_back(base_idx + 0);
            indices.push_back(base_idx + 1);
            indices.push_back(base_idx + 2);
            indices.push_back(base_idx + 0);
            indices.push_back(base_idx + 2);
            indices.push_back(base_idx + 3);
        }
    } else if (type == "plane") {
        positions = {
            Vec3(-hs, 0.0f, -hs), Vec3( hs, 0.0f, -hs),
            Vec3( hs, 0.0f,  hs), Vec3(-hs, 0.0f,  hs)
        };
        normals = {
            Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)
        };
        uvs = {
            Vec2(0.0f, 0.0f), Vec2(1.0f, 0.0f), Vec2(1.0f, 1.0f), Vec2(0.0f, 1.0f)
        };
        indices = { 0, 2, 1, 0, 3, 2 };
    } else if (type == "sphere") {
        const int stacks = 20;
        const int slices = 32;
        const float radius = hs;

        for (int i = 0; i <= stacks; ++i) {
            float phi = M_PI_F * static_cast<float>(i) / static_cast<float>(stacks);
            float cos_phi = std::cos(phi);
            float sin_phi = std::sin(phi);

            for (int j = 0; j <= slices; ++j) {
                float theta = 2.0f * M_PI_F * static_cast<float>(j) / static_cast<float>(slices);
                float cos_theta = std::cos(theta);
                float sin_theta = std::sin(theta);

                Vec3 norm(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
                Vec3 pos = norm * radius;
                Vec2 uv(static_cast<float>(j) / static_cast<float>(slices), 1.0f - static_cast<float>(i) / static_cast<float>(stacks));

                positions.push_back(pos);
                normals.push_back(norm);
                uvs.push_back(uv);
            }
        }

        for (int i = 0; i < stacks; ++i) {
            for (int j = 0; j < slices; ++j) {
                uint32_t first = static_cast<uint32_t>(i * (slices + 1) + j);
                uint32_t second = first + static_cast<uint32_t>(slices + 1);

                indices.push_back(first);
                indices.push_back(second);
                indices.push_back(first + 1);

                indices.push_back(second);
                indices.push_back(second + 1);
                indices.push_back(first + 1);
            }
        }
    } else if (type == "cylinder") {
        const int slices = 32;
        const float radius = hs;
        const float half_h = hs;

        for (int i = 0; i <= slices; ++i) {
            float theta = 2.0f * M_PI_F * static_cast<float>(i) / static_cast<float>(slices);
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            Vec3 n(cos_t, 0.0f, sin_t);
            float u = static_cast<float>(i) / static_cast<float>(slices);

            positions.push_back(Vec3(radius * cos_t, -half_h, radius * sin_t));
            normals.push_back(n);
            uvs.push_back(Vec2(u, 0.0f));

            positions.push_back(Vec3(radius * cos_t, half_h, radius * sin_t));
            normals.push_back(n);
            uvs.push_back(Vec2(u, 1.0f));
        }

        for (int i = 0; i < slices; ++i) {
            uint32_t b0 = static_cast<uint32_t>(i * 2);
            uint32_t t0 = b0 + 1;
            uint32_t b1 = b0 + 2;
            uint32_t t1 = b0 + 3;

            indices.push_back(b0);
            indices.push_back(b1);
            indices.push_back(t0);

            indices.push_back(t0);
            indices.push_back(b1);
            indices.push_back(t1);
        }

        uint32_t top_center_idx = static_cast<uint32_t>(positions.size());
        positions.push_back(Vec3(0.0f, half_h, 0.0f));
        normals.push_back(Vec3(0.0f, 1.0f, 0.0f));
        uvs.push_back(Vec2(0.5f, 0.5f));

        uint32_t top_ring_start = static_cast<uint32_t>(positions.size());
        for (int i = 0; i <= slices; ++i) {
            float theta = 2.0f * M_PI_F * static_cast<float>(i) / static_cast<float>(slices);
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            positions.push_back(Vec3(radius * cos_t, half_h, radius * sin_t));
            normals.push_back(Vec3(0.0f, 1.0f, 0.0f));
            uvs.push_back(Vec2(0.5f + 0.5f * cos_t, 0.5f + 0.5f * sin_t));
        }

        for (int i = 0; i < slices; ++i) {
            indices.push_back(top_center_idx);
            indices.push_back(top_ring_start + i + 1);
            indices.push_back(top_ring_start + i);
        }

        uint32_t bot_center_idx = static_cast<uint32_t>(positions.size());
        positions.push_back(Vec3(0.0f, -half_h, 0.0f));
        normals.push_back(Vec3(0.0f, -1.0f, 0.0f));
        uvs.push_back(Vec2(0.5f, 0.5f));

        uint32_t bot_ring_start = static_cast<uint32_t>(positions.size());
        for (int i = 0; i <= slices; ++i) {
            float theta = 2.0f * M_PI_F * static_cast<float>(i) / static_cast<float>(slices);
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            positions.push_back(Vec3(radius * cos_t, -half_h, radius * sin_t));
            normals.push_back(Vec3(0.0f, -1.0f, 0.0f));
            uvs.push_back(Vec2(0.5f + 0.5f * cos_t, 0.5f + 0.5f * sin_t));
        }

        for (int i = 0; i < slices; ++i) {
            indices.push_back(bot_center_idx);
            indices.push_back(bot_ring_start + i);
            indices.push_back(bot_ring_start + i + 1);
        }
    } else if (type == "torus") {
        const int main_segments = 32;
        const int tube_segments = 16;
        const float R = hs * 0.7f;
        const float r = hs * 0.3f;

        for (int i = 0; i <= main_segments; ++i) {
            float u = 2.0f * M_PI_F * static_cast<float>(i) / static_cast<float>(main_segments);
            float cos_u = std::cos(u);
            float sin_u = std::sin(u);

            for (int j = 0; j <= tube_segments; ++j) {
                float v = 2.0f * M_PI_F * static_cast<float>(j) / static_cast<float>(tube_segments);
                float cos_v = std::cos(v);
                float sin_v = std::sin(v);

                Vec3 pos(
                    (R + r * cos_v) * cos_u,
                    r * sin_v,
                    (R + r * cos_v) * sin_u
                );
                Vec3 norm(cos_v * cos_u, sin_v, cos_v * sin_u);
                Vec2 uv(static_cast<float>(i) / static_cast<float>(main_segments), static_cast<float>(j) / static_cast<float>(tube_segments));

                positions.push_back(pos);
                normals.push_back(norm);
                uvs.push_back(uv);
            }
        }

        for (int i = 0; i < main_segments; ++i) {
            for (int j = 0; j < tube_segments; ++j) {
                uint32_t first = static_cast<uint32_t>(i * (tube_segments + 1) + j);
                uint32_t second = first + static_cast<uint32_t>(tube_segments + 1);

                indices.push_back(first);
                indices.push_back(second);
                indices.push_back(first + 1);

                indices.push_back(second);
                indices.push_back(second + 1);
                indices.push_back(first + 1);
            }
        }
    } else {
        return nullptr;
    }

    size_t vCount = positions.size();
    mesh->geometry->resize_vertices(vCount);

    Vec3* pos = mesh->geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* norm = mesh->geometry->get_attribute_data_mut<Vec3>("N");
    Vec3* origP = mesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* origN = mesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
    Vec2* uv_ptr = mesh->geometry->get_attribute_data_mut<Vec2>("uv");
    uint16_t* matIDs = mesh->geometry->get_attribute_data_mut<uint16_t>("materialID");

    if (pos && origP) {
        std::memcpy(pos, positions.data(), vCount * sizeof(Vec3));
        std::memcpy(origP, positions.data(), vCount * sizeof(Vec3));
    }
    if (norm && origN) {
        std::memcpy(norm, normals.data(), vCount * sizeof(Vec3));
        std::memcpy(origN, normals.data(), vCount * sizeof(Vec3));
    }
    if (uv_ptr) {
        std::memcpy(uv_ptr, uvs.data(), vCount * sizeof(Vec2));
    }
    if (matIDs) {
        std::memset(matIDs, 0, vCount * sizeof(uint16_t));
    }

    mesh->build_local_bvh();
    return mesh;
}

} // namespace

Result addPrimitive(const std::string& type, const std::string& requested_name, float size, std::string& out_new_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");
    if (size <= 0.0f) return Result::fail("size must be positive");

    std::string base_name = requested_name;
    if (base_name.empty()) {
        base_name = type;
        if (!base_name.empty()) base_name[0] = static_cast<char>(std::toupper(base_name[0]));
    }

    std::string candidate_name = base_name;
    int index = 1;
    while (objectExists(candidate_name)) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), ".%03d", index++);
        candidate_name = base_name + buf;
    }

    auto mesh = createPrimitiveMesh(type, size, candidate_name);
    if (!mesh) return Result::fail("unknown primitive type '" + type + "' (expected cube|sphere|plane|cylinder|torus)");

    auto cmd = std::make_unique<AddObjectCommand>(mesh);
    cmd->execute(*g_ctx);
    g_history->record(std::move(cmd));

    out_new_name = candidate_name;
    return Result::success();
}

Result getMaterialParam(const std::string& object_name, const std::string& param,
                        MaterialParamValue& out) {
    if (!g_ctx) return notBound();
    if (!objectExists(object_name)) return Result::fail("object not found: " + object_name);

    MaterialParamKind kind;
    bool is_color = false;
    if (!parseMaterialParam(param, kind, is_color)) {
        return Result::fail("unknown material parameter: " + param);
    }

    const auto ids = objectMaterialIds(*g_ctx, object_name);
    for (uint16_t id : ids) {
        auto* material = dynamic_cast<PrincipledBSDF*>(MaterialManager::getInstance().getMaterial(id));
        if (!material) continue;
        const MaterialValue value = readMaterialValue(*material, kind);
        out = {};
        out.is_color = is_color;
        out.scalar = value.scalar;
        out.color = value.color;
        return Result::success();
    }
    return Result::fail("object has no Principled BSDF material: " + object_name);
}

namespace {

Result setMaterialParamValue(const std::string& object_name, const std::string& param,
                             const MaterialValue& requested, bool supplied_color) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");
    if (!objectExists(object_name)) return Result::fail("object not found: " + object_name);

    MaterialParamKind kind;
    bool expects_color = false;
    if (!parseMaterialParam(param, kind, expects_color)) {
        return Result::fail("unknown material parameter: " + param);
    }
    if (expects_color != supplied_color) {
        return Result::fail(param + (expects_color ? " expects an RGB value" : " expects a scalar value"));
    }

    if (supplied_color) {
        if (!std::isfinite(requested.color.x) || !std::isfinite(requested.color.y) ||
            !std::isfinite(requested.color.z) || requested.color.x < 0.0f ||
            requested.color.y < 0.0f || requested.color.z < 0.0f) {
            return Result::fail(param + " components must be finite and non-negative");
        }
    } else {
        const float value = requested.scalar;
        if (!std::isfinite(value)) return Result::fail(param + " must be finite");
        const bool unit_range = kind == MaterialParamKind::Roughness ||
                                kind == MaterialParamKind::Metallic ||
                                kind == MaterialParamKind::Specular ||
                                kind == MaterialParamKind::Transmission ||
                                kind == MaterialParamKind::Opacity;
        if (unit_range && (value < 0.0f || value > 1.0f)) {
            return Result::fail(param + " must be in the range [0, 1]");
        }
        if (kind == MaterialParamKind::EmissionStrength && value < 0.0f) {
            return Result::fail("emission_strength must be non-negative");
        }
        if (kind == MaterialParamKind::Ior && (value < 1.0f || value > 10.0f)) {
            return Result::fail("ior must be in the range [1, 10]");
        }
    }

    std::vector<MaterialEdit> edits;
    for (uint16_t id : objectMaterialIds(*g_ctx, object_name)) {
        auto* material = dynamic_cast<PrincipledBSDF*>(MaterialManager::getInstance().getMaterial(id));
        if (!material) continue;
        MaterialEdit edit;
        edit.material_id = id;
        edit.before = readMaterialValue(*material, kind);
        edit.after = requested;
        edits.push_back(edit);
    }
    if (edits.empty()) {
        return Result::fail("object has no Principled BSDF material: " + object_name);
    }

    auto command = std::make_unique<MaterialParamCommand>(object_name, param, kind, std::move(edits));
    command->execute(*g_ctx);
    g_history->record(std::move(command));
    return Result::success();
}

} // namespace

Result setMaterialParam(const std::string& object_name, const std::string& param, float value) {
    MaterialValue requested;
    requested.scalar = value;
    return setMaterialParamValue(object_name, param, requested, false);
}

Result setMaterialParam(const std::string& object_name, const std::string& param, const Vec3& value) {
    MaterialValue requested;
    requested.color = value;
    return setMaterialParamValue(object_name, param, requested, true);
}

namespace {

const char* lightTypeName(LightType t) {
    switch (t) {
        case LightType::Point:       return "point";
        case LightType::Directional: return "directional";
        case LightType::Spot:        return "spot";
        case LightType::Area:        return "area";
        default:                     return "unknown";
    }
}

} // namespace

std::vector<LightInfo> listLights() {
    std::vector<LightInfo> out;
    if (!g_ctx) return out;
    int i = 0;
    for (auto& l : g_ctx->scene.lights) {
        if (l) {
            LightInfo info;
            info.index = i;
            info.name = l->nodeName;
            info.type = lightTypeName(l->type());
            info.position = l->position;
            out.push_back(std::move(info));
        }
        ++i;
    }
    return out;
}

Result addLight(const std::string& type, const Vec3& position, std::string& out_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");

    // Same defaults as the UI "Add > Light" menu.
    std::shared_ptr<Light> light;
    std::string prefix;
    if (type == "point") {
        light = std::make_shared<PointLight>(position, Vec3(10, 10, 10), 0.1f);
        prefix = "Point_";
    } else if (type == "directional") {
        auto l = std::make_shared<DirectionalLight>(Vec3(-1, -1, -0.5), Vec3(5, 5, 5), 0.1f);
        l->position = position;
        light = l;
        prefix = "Directional_";
    } else if (type == "spot") {
        light = std::make_shared<SpotLight>(position, Vec3(0, -1, 0), Vec3(10, 10, 10), 45.0f, 60.0f);
        prefix = "Spot_";
    } else if (type == "area") {
        light = std::make_shared<AreaLight>(position, Vec3(1, 0, 0), Vec3(0, 0, 1), 2.0f, 2.0f, Vec3(10, 10, 10));
        prefix = "Area_";
    } else {
        return Result::fail("unknown light type (point|directional|spot|area): " + type);
    }
    light->nodeName = prefix + std::to_string(g_ctx->scene.lights.size() + 1);

    auto cmd = std::make_unique<AddLightCommand>(light);
    cmd->execute(*g_ctx);   // idempotent push + backend light sync
    g_history->record(std::move(cmd));
    ProjectManager::getInstance().markModified();
    out_name = light->nodeName;
    return Result::success();
}

Result deleteLight(int index) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");
    auto& lights = g_ctx->scene.lights;
    if (index < 0 || index >= static_cast<int>(lights.size()) || !lights[index])
        return Result::fail("light index out of range: " + std::to_string(index));

    auto cmd = std::make_unique<DeleteLightCommand>(lights[index]);
    cmd->execute(*g_ctx);
    g_history->record(std::move(cmd));
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result setLightPosition(int index, const Vec3& position) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");
    auto& lights = g_ctx->scene.lights;
    if (index < 0 || index >= static_cast<int>(lights.size()) || !lights[index])
        return Result::fail("light index out of range: " + std::to_string(index));

    auto& light = lights[index];
    LightState old_state = LightState::capture(*light);
    LightState new_state = old_state;
    new_state.position = position;

    auto cmd = std::make_unique<TransformLightCommand>(light, old_state, new_state);
    cmd->execute(*g_ctx);   // applies new state + backend light sync
    g_history->record(std::move(cmd));
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result undo() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history || !g_history->canUndo()) return Result::fail("nothing to undo");
    g_history->undo(*g_ctx);
    return Result::success();
}

Result redo() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history || !g_history->canRedo()) return Result::fail("nothing to redo");
    g_history->redo(*g_ctx);
    return Result::success();
}

std::string undoDescription() {
    return g_history ? g_history->getUndoDescription() : std::string();
}

std::string redoDescription() {
    return g_history ? g_history->getRedoDescription() : std::string();
}

Result requestRender() {
    if (!g_ctx) return notBound();
    g_ctx->start_render = true;
    return Result::success();
}

Result resetAccumulation() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("accumulation is owned by the final render job");
    if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
    g_ctx->renderer.resetCPUAccumulation();
    return Result::success();
}

Result renderFrame(const std::string& output_path, int spp) {
    if (!g_ctx) return notBound();
    if (output_path.empty()) return Result::fail("render output path is empty");
    if (spp <= 0) return Result::fail("render spp must be greater than zero");
    if (g_render_job.state == RenderJobState::Rendering) {
        return Result::fail("a final render job is already running");
    }
    if (rendering_in_progress.load() || g_ctx->is_animation_mode) {
        return Result::fail("cannot start a final render while animation rendering is active");
    }

    std::error_code path_error;
    std::filesystem::path path = std::filesystem::absolute(output_path, path_error);
    if (path_error) return Result::fail("invalid render output path: " + output_path);
    if (!path.has_extension()) path += ".png";
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension != ".png") return Result::fail("final render output must be a .png file");

    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, path_error);
        if (path_error) return Result::fail("cannot create render output directory: " + parent.string());
    }

    g_render_job = {};
    g_render_job.state = RenderJobState::Rendering;
    g_render_job.output_path = path.lexically_normal().string();
    g_render_job.target_samples = spp;

    rendering_stopped_cpu = false;
    rendering_stopped_gpu = false;
    g_ctx->renderer.resetCPUAccumulation();
    if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
    g_ctx->render_settings.final_render_samples = spp;
    g_ctx->render_settings.render_target_samples = spp;
    g_ctx->render_settings.render_current_samples = 0;
    g_ctx->render_settings.render_progress = 0.0f;
    g_ctx->render_settings.render_elapsed_seconds = 0.0f;
    g_ctx->render_settings.render_estimated_remaining = 0.0f;
    g_ctx->render_settings.is_rendering_active = true;
    g_ctx->render_settings.is_render_paused = false;
    g_ctx->render_settings.is_final_render_mode = true;

    // Final output must come from the path-traced Rendered viewport. The normal
    // mode-transition block owns backend creation/full sync if we came from Solid.
    ui.viewport_settings.shading_mode = 2;
    g_ctx->start_render = true;
    return Result::success();
}

RenderJobInfo renderStatus() {
    RenderJobInfo info = g_render_job;
    if (g_ctx && info.state == RenderJobState::Rendering) {
        info.current_samples = g_ctx->render_settings.render_current_samples;
        info.target_samples = std::max(1, g_render_job.target_samples);
        info.progress = std::clamp(
            static_cast<float>(info.current_samples) / static_cast<float>(info.target_samples),
            0.0f, 1.0f);
    }
    return info;
}

Result cancelRender() {
    if (!g_ctx) return notBound();
    if (g_render_job.state != RenderJobState::Rendering) {
        return Result::fail("no final render job is running");
    }
    rendering_stopped_cpu = true;
    rendering_stopped_gpu = true;
    g_ctx->render_settings.is_final_render_mode = false;
    g_ctx->render_settings.is_rendering_active = false;
    g_ctx->start_render = false;
    g_render_job.state = RenderJobState::Cancelled;
    return Result::success();
}

bool renderOutputPending() {
    return g_render_job.state == RenderJobState::Rendering;
}

std::string renderOutputPath() {
    return g_render_job.output_path;
}

void completeRenderOutput(bool ok, const std::string& error) {
    if (g_render_job.state != RenderJobState::Rendering) return;
    if (g_ctx) {
        g_render_job.current_samples = g_ctx->render_settings.render_current_samples;
        g_render_job.target_samples = std::max(1, g_render_job.target_samples);
        g_render_job.progress = ok ? 1.0f : g_ctx->render_settings.render_progress;
        g_ctx->render_settings.is_final_render_mode = false;
        g_ctx->render_settings.is_rendering_active = false;
        g_ctx->start_render = false;
    }
    g_render_job.state = ok ? RenderJobState::Completed : RenderJobState::Failed;
    g_render_job.error = ok ? std::string() : error;
}

// ---------------------------------------------------------------------------
// Sequence render (multi-frame). Hooks into the g_seq_save_active state machine
// that already exists in Main.cpp; the main loop drives frame accumulation and
// PNG writes, then sets quit when running in CLI mode.
// ---------------------------------------------------------------------------
namespace {
struct SequenceState {
    int spp = 128;
    int start_frame = 0;
};
SequenceState g_seq_state;
} // namespace

Result renderSequence(const std::string& output_dir, int spp,
                      int start_frame, int end_frame) {
    if (!g_ctx) return notBound();
    if (output_dir.empty()) return Result::fail("sequence output directory is empty");
    if (spp <= 0) return Result::fail("sequence spp must be greater than zero");
    if (start_frame < 0) return Result::fail("start_frame must be non-negative");
    if (end_frame < start_frame)
        return Result::fail("end_frame must be >= start_frame");
    if (g_seq_save_active)
        return Result::fail("a sequence render is already running");
    if (g_render_job.state == RenderJobState::Rendering)
        return Result::fail("a single-frame render job is already running");
    if (rendering_in_progress.load() && g_ctx->is_animation_mode && !g_seq_save_active)
        return Result::fail("cannot start a sequence while animation rendering is active");

    std::error_code dir_error;
    std::filesystem::create_directories(output_dir, dir_error);
    if (dir_error)
        return Result::fail("cannot create sequence output directory: " + output_dir);

    // Mirror the exact sequence the UI uses (Main.cpp ~L4469) so the backend
    // and state machine behave identically for CLI vs interactive launches.
    g_seq_state.spp = spp;
    g_seq_state.start_frame = start_frame;

    // Set the active frame before engaging the sequence machine.
    const rtapi::Result frame_result = setFrame(start_frame);
    if (!frame_result.ok) return frame_result;

    // Configure the render quality to match the requested spp before the
    // state machine starts accumulating the first frame.
    g_ctx->render_settings.final_render_samples = spp;
    g_ctx->render_settings.render_target_samples = spp;
    g_ctx->render_settings.render_current_samples = 0;
    g_ctx->render_settings.render_progress = 0.0f;
    g_ctx->render_settings.is_rendering_active = true;
    g_ctx->render_settings.is_render_paused = false;
    g_ctx->render_settings.is_final_render_mode = false;  // sequence uses interactive path

    // Switch to Rendered mode so g_seq_save_active state machine can accumulate.
    ui.viewport_settings.shading_mode = 2;

    // Arm the state machine (same variables used by the UI export button).
    g_seq_save_dir = output_dir;
    g_seq_save_frame = start_frame;
    g_seq_save_end = end_frame;
    g_seq_save_denoise = false;
    g_seq_save_active = true;

    rendering_in_progress = true;
    g_ctx->is_animation_mode = true;
    g_ctx->render_settings.animation_render_locked = true;
    rendering_stopped_gpu = false;
    rendering_stopped_cpu = false;
    g_ctx->start_render = true;

    return Result::success();
}

SequenceJobInfo sequenceStatus() {
    SequenceJobInfo info;
    info.active = g_seq_save_active;
    info.current_frame = g_seq_save_frame;
    info.start_frame = g_seq_state.start_frame;
    info.end_frame = g_seq_save_end;
    info.output_dir = g_seq_save_dir;
    if (g_seq_save_active && info.end_frame >= info.start_frame) {
        const int total_frames = info.end_frame - info.start_frame + 1;
        const int done_frames = info.current_frame - info.start_frame;
        info.total_progress = std::clamp(
            static_cast<float>(done_frames) / static_cast<float>(total_frames),
            0.0f, 1.0f);
        if (g_ctx) {
            info.frame_progress = std::clamp(
                g_ctx->render_settings.render_progress, 0.0f, 1.0f);
        }
    } else if (!g_seq_save_active && g_seq_save_frame > g_seq_save_end) {
        // Completed.
        info.total_progress = 1.0f;
        info.frame_progress = 1.0f;
    }
    return info;
}

Result cancelSequence() {
    if (!g_ctx) return notBound();
    if (!g_seq_save_active)
        return Result::fail("no sequence render is running");
    // Same path as the UI's "Stop Anim" button: signal the stop atomics;
    // the state machine in Main.cpp detects them next frame and cleans up.
    rendering_stopped_cpu = true;
    rendering_stopped_gpu = true;
    return Result::success();
}

std::string currentProjectPath() {
    return ProjectManager::getInstance().getCurrentFilePath();
}

Result saveProject(const std::string& filepath) {
    if (!g_ctx) return notBound();
    if (rendering_in_progress.load() || g_render_job.state == RenderJobState::Rendering)
        return Result::fail("cannot save project while rendering");
    ProjectManager& projects = ProjectManager::getInstance();
    const bool ok = filepath.empty()
        ? projects.saveProject(g_ctx->scene, g_ctx->render_settings, g_ctx->renderer)
        : projects.saveProject(filepath, g_ctx->scene, g_ctx->render_settings, g_ctx->renderer);
    if (!ok) {
        return Result::fail(filepath.empty()
            ? "project has no current path or save failed"
            : "project save failed: " + filepath);
    }
    return Result::success();
}

Result openProject(const std::string& filepath) {
    if (!g_ctx) return notBound();
    if (rendering_in_progress.load() || g_render_job.state == RenderJobState::Rendering)
        return Result::fail("cannot open project while rendering");
    if (filepath.empty()) return Result::fail("project path is empty");
    std::error_code file_error;
    if (!std::filesystem::is_regular_file(filepath, file_error)) {
        return Result::fail("project file not found: " + filepath);
    }

    const bool ok = ProjectManager::getInstance().openProject(
        filepath, g_ctx->scene, g_ctx->render_settings, g_ctx->renderer, g_ctx->backend_ptr);
    if (!ok) return Result::fail("project open failed: " + filepath);

    // Commands and selections retain scene-owned pointers, so neither may cross
    // a project boundary. Backend/geometry dirty flags are scheduled by
    // ProjectManager::openProject and consumed by the normal main-loop path.
    g_ctx->selection.clearSelection();
    if (g_history) g_history->clear();
    ui.timeline.reset();
    ui.timeline.setCurrentFrame(g_ctx->scene.timeline.current_frame);
    g_ctx->render_settings.animation_current_frame = g_ctx->scene.timeline.current_frame;
    g_ctx->render_settings.animation_playback_frame = g_ctx->scene.timeline.current_frame;
    g_ctx->active_model_path = filepath;
    ui.invalidateCache();
    ui.mesh_cache_valid = false;
    g_ctx->start_render = true;
    notifySceneLoaded();
    return Result::success();
}

int currentFrame() {
    if (!g_ctx) return 0;
    return ui.timeline.getCurrentFrame();
}

Result setFrame(int frame) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (frame < 0) return Result::fail("frame must be non-negative");
    ui.timeline.setCurrentFrame(frame);
    g_ctx->scene.timeline.current_frame = frame;
    g_ctx->render_settings.animation_current_frame = frame;
    g_ctx->render_settings.animation_playback_frame = frame;
    g_ctx->start_render = true;
    return Result::success();
}

// ---------------------------------------------------------------------------
// Keyframes (Faz 3c).
// ---------------------------------------------------------------------------
Result insertKeyframe(const std::string& object_name, const std::string& channel,
                      int frame, const Vec3& value) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (frame < 0) return Result::fail("frame must be non-negative");
    if (!objectExists(object_name)) return Result::fail("object not found: " + object_name);

    enum Channel { LOC, ROT, SCL } which;
    if (channel == "location")      which = LOC;
    else if (channel == "rotation") which = ROT;
    else if (channel == "scale")    which = SCL;
    else return Result::fail("unknown channel '" + channel + "' (expected location|rotation|scale)");

    auto& timeline = g_ctx->scene.timeline;

    Keyframe kf(frame);
    kf.has_transform = true;
    kf.transform.clearAllChannels();
    // Preserve any channels already keyed at this frame — addKeyframe replaces the
    // whole TransformKeyframe on a merge, so we must carry the existing one forward.
    auto track_it = timeline.tracks.find(object_name);
    if (track_it != timeline.tracks.end()) {
        for (auto& existing : track_it->second.keyframes) {
            if (existing.frame == frame && existing.has_transform) {
                kf.transform = existing.transform;
                break;
            }
        }
    }

    switch (which) {
        case LOC:
            kf.transform.position = value;
            kf.transform.has_pos_x = kf.transform.has_pos_y = kf.transform.has_pos_z = true;
            break;
        case ROT:
            kf.transform.rotation = value;
            kf.transform.has_rot_x = kf.transform.has_rot_y = kf.transform.has_rot_z = true;
            break;
        case SCL:
            kf.transform.scale = value;
            kf.transform.has_scl_x = kf.transform.has_scl_y = kf.transform.has_scl_z = true;
            break;
    }
    kf.transform.refreshCompoundFlags();

    timeline.insertKeyframe(object_name, kf);
    g_ctx->start_render = true;
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result removeKeyframe(const std::string& object_name, int frame) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& timeline = g_ctx->scene.timeline;
    if (timeline.tracks.find(object_name) == timeline.tracks.end()) {
        return Result::fail("no animation track for object: " + object_name);
    }
    timeline.removeKeyframe(object_name, frame);
    g_ctx->start_render = true;
    ProjectManager::getInstance().markModified();
    return Result::success();
}

std::vector<int> listKeyframes(const std::string& object_name) {
    std::vector<int> frames;
    if (!g_ctx) return frames;
    auto it = g_ctx->scene.timeline.tracks.find(object_name);
    if (it == g_ctx->scene.timeline.tracks.end()) return frames;
    for (const auto& k : it->second.keyframes) frames.push_back(k.frame);
    return frames;
}

// ---------------------------------------------------------------------------
// Node graphs (Faz 3d).
// ---------------------------------------------------------------------------
namespace {

NodeSystem::GraphBase* findNodeGraph(UIContext& ctx, const std::string& graph_type,
                                     const std::string& graph_name, Result& err) {
    if (graph_type == "material") {
        auto it = ctx.scene.material_node_graphs.find(graph_name);
        if (it == ctx.scene.material_node_graphs.end() || !it->second) {
            err = Result::fail("material node graph not found: " + graph_name);
            return nullptr;
        }
        return it->second.get();
    }
    if (graph_type == "geometry") {
        auto it = ctx.scene.geometry_node_graphs.find(graph_name);
        if (it == ctx.scene.geometry_node_graphs.end() || !it->second) {
            err = Result::fail("geometry node graph not found: " + graph_name);
            return nullptr;
        }
        return it->second.get();
    }
    if (graph_type == "terrain") {
        TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(graph_name);
        if (!terrain || !terrain->nodeGraph) {
            err = Result::fail("terrain node graph not found: " + graph_name);
            return nullptr;
        }
        return terrain->nodeGraph.get();
    }
    err = Result::fail("unknown graph_type '" + graph_type + "' (expected material|geometry|terrain)");
    return nullptr;
}

} // namespace

std::vector<NodeTypeDesc> listNodeTypes() {
    std::vector<NodeTypeDesc> out;
    for (const auto& info : NodeSystem::NodeRegistry::instance().getAllTypes()) {
        NodeTypeDesc d;
        d.type_id = info.typeId;
        d.category = info.category;
        d.display_name = info.displayName;
        d.description = info.description;
        out.push_back(std::move(d));
    }
    return out;
}

Result addNode(const std::string& graph_type, const std::string& graph_name,
               const std::string& type_id, unsigned int& out_node_id) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;

    std::shared_ptr<NodeSystem::NodeBase> node =
        NodeSystem::NodeRegistry::instance().create(type_id);
    if (!node) return Result::fail("unknown node type: " + type_id);

    NodeSystem::NodeBase* added = graph->registerNode(std::move(node));
    out_node_id = added->id;
    graph->markAllDirty();
    return Result::success();
}

Result removeNode(const std::string& graph_type, const std::string& graph_name,
                  unsigned int node_id) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;
    if (!graph->getNode(node_id)) return Result::fail("node id not found: " + std::to_string(node_id));
    graph->removeNode(node_id);
    graph->markAllDirty();
    return Result::success();
}

Result linkNodes(const std::string& graph_type, const std::string& graph_name,
                 unsigned int from_node, int from_output, unsigned int to_node,
                 int to_input, unsigned int& out_link_id) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;

    NodeSystem::NodeBase* src = graph->getNode(from_node);
    NodeSystem::NodeBase* dst = graph->getNode(to_node);
    if (!src) return Result::fail("from_node not found: " + std::to_string(from_node));
    if (!dst) return Result::fail("to_node not found: " + std::to_string(to_node));
    if (from_output < 0 || from_output >= static_cast<int>(src->outputs.size()))
        return Result::fail("from_output index out of range");
    if (to_input < 0 || to_input >= static_cast<int>(dst->inputs.size()))
        return Result::fail("to_input index out of range");

    const uint32_t start_pin = src->outputs[from_output].id;
    const uint32_t end_pin = dst->inputs[to_input].id;
    const uint32_t link_id = graph->addLink(start_pin, end_pin);
    if (link_id == 0) {
        return Result::fail("link rejected (type/semantic mismatch, would create a cycle, "
                            "or invalid pins)");
    }
    out_link_id = link_id;
    return Result::success();
}

Result listNodes(const std::string& graph_type, const std::string& graph_name,
                 std::vector<NodeDesc>& out) {
    out.clear();
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;

    for (const auto& n : graph->nodes) {
        if (!n) continue;
        NodeDesc d;
        d.id = n->id;
        d.type_id = n->metadata.typeId.empty() ? n->getTypeId() : n->metadata.typeId;
        d.display_name = n->metadata.displayName.empty() ? n->metadata.typeId : n->metadata.displayName;
        d.input_count = static_cast<int>(n->inputs.size());
        d.output_count = static_cast<int>(n->outputs.size());
        out.push_back(std::move(d));
    }
    return Result::success();
}

// ---------------------------------------------------------------------------
// Node parameters (Faz 5.1b). A node's input-pin default values are its
// scriptable parameters. findNodeGraph (above) locates the graph.
// ---------------------------------------------------------------------------
namespace {

const char* nodeDataTypeName(NodeSystem::DataType t) {
    using DT = NodeSystem::DataType;
    switch (t) {
        case DT::Float:    return "float";
        case DT::Int:      return "int";
        case DT::Bool:     return "bool";
        case DT::Vector2:  return "vector2";
        case DT::Vector3:  return "vector3";
        case DT::Vector4:  return "vector4";
        case DT::Color:    return "color";
        case DT::String:   return "string";
        case DT::Image2D:  return "image2d";
        case DT::Geometry: return "geometry";
        case DT::Material: return "material";
        default:           return "none";
    }
}

// PinValue (variant) -> NodeParamValue. Reports whichever alternative is stored,
// regardless of the pin's declared type (an unset default reads back as None).
NodeParamValue pinValueToParam(const NodeSystem::PinValue& v) {
    NodeParamValue out;
    using K = NodeParamValue::Kind;
    if (auto* f = std::get_if<float>(&v)) {
        out.kind = K::Float; out.floats[0] = *f;
    } else if (auto* i = std::get_if<int>(&v)) {
        out.kind = K::Int; out.int_value = *i; out.floats[0] = static_cast<float>(*i);
    } else if (auto* b = std::get_if<bool>(&v)) {
        out.kind = K::Bool; out.bool_value = *b; out.floats[0] = *b ? 1.0f : 0.0f;
    } else if (auto* v2 = std::get_if<std::array<float, 2>>(&v)) {
        out.kind = K::Vector2; out.floats[0] = (*v2)[0]; out.floats[1] = (*v2)[1];
    } else if (auto* v3 = std::get_if<std::array<float, 3>>(&v)) {
        out.kind = K::Vector3; out.floats[0] = (*v3)[0]; out.floats[1] = (*v3)[1]; out.floats[2] = (*v3)[2];
    } else if (auto* v4 = std::get_if<std::array<float, 4>>(&v)) {
        out.kind = K::Vector4;
        for (int k = 0; k < 4; ++k) out.floats[k] = (*v4)[k];
    } else if (auto* s = std::get_if<std::string>(&v)) {
        out.kind = K::String; out.string_value = *s;
    }
    return out;
}

// Write a NodeParamValue into a pin's defaultValue, COERCED to the pin's declared
// data type. The graph evaluator reads defaultValue expecting a specific variant
// alternative (tryGetFloat etc.), so we must store the alternative that matches
// the pin — not whatever the caller happened to pass.
Result setPinDefault(NodeSystem::Pin& pin, const NodeParamValue& val) {
    using DT = NodeSystem::DataType;
    using K = NodeParamValue::Kind;
    auto scalar = [&]() -> float {
        switch (val.kind) {
            case K::Int:  return static_cast<float>(val.int_value);
            case K::Bool: return val.bool_value ? 1.0f : 0.0f;
            default:      return val.floats[0];  // Float / Vector* use component 0
        }
    };
    switch (pin.dataType) {
        case DT::Float:
            pin.defaultValue = scalar();
            break;
        case DT::Int:
            pin.defaultValue = (val.kind == K::Int) ? val.int_value
                                                    : static_cast<int>(scalar());
            break;
        case DT::Bool:
            pin.defaultValue = (val.kind == K::Bool) ? val.bool_value : (scalar() != 0.0f);
            break;
        case DT::Vector2:
            pin.defaultValue = std::array<float, 2>{ val.floats[0], val.floats[1] };
            break;
        case DT::Vector3:
            pin.defaultValue = std::array<float, 3>{ val.floats[0], val.floats[1], val.floats[2] };
            break;
        case DT::Vector4:
        case DT::Color:
            pin.defaultValue = std::array<float, 4>{ val.floats[0], val.floats[1],
                                                     val.floats[2], val.floats[3] };
            break;
        case DT::String:
            if (val.kind != K::String)
                return Result::fail("pin '" + pin.name + "' is a string; provide a string value");
            pin.defaultValue = val.string_value;
            break;
        default:
            return Result::fail(std::string("pin '") + pin.name + "' (type " +
                                nodeDataTypeName(pin.dataType) + ") has no scriptable default value");
    }
    return Result::success();
}

} // namespace

Result listNodeParams(const std::string& graph_type, const std::string& graph_name,
                      unsigned int node_id, std::vector<NodeParamInfo>& out) {
    out.clear();
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;
    NodeSystem::NodeBase* node = graph->getNode(node_id);
    if (!node) return Result::fail("node id not found: " + std::to_string(node_id));

    for (size_t i = 0; i < node->inputs.size(); ++i) {
        const NodeSystem::Pin& pin = node->inputs[i];
        NodeParamInfo info;
        info.index = static_cast<int>(i);
        info.name = pin.name;
        info.data_type = nodeDataTypeName(pin.dataType);
        info.connected = graph->getInputSource(pin.id) != nullptr;
        info.value = pinValueToParam(pin.defaultValue);
        out.push_back(std::move(info));
    }
    return Result::success();
}

Result getNodeParam(const std::string& graph_type, const std::string& graph_name,
                    unsigned int node_id, int pin_index, NodeParamValue& out) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;
    NodeSystem::NodeBase* node = graph->getNode(node_id);
    if (!node) return Result::fail("node id not found: " + std::to_string(node_id));
    if (pin_index < 0 || pin_index >= static_cast<int>(node->inputs.size()))
        return Result::fail("pin_index out of range (node has " +
                            std::to_string(node->inputs.size()) + " inputs)");
    out = pinValueToParam(node->inputs[pin_index].defaultValue);
    return Result::success();
}

Result setNodeParam(const std::string& graph_type, const std::string& graph_name,
                    unsigned int node_id, int pin_index, const NodeParamValue& value) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;
    NodeSystem::NodeBase* node = graph->getNode(node_id);
    if (!node) return Result::fail("node id not found: " + std::to_string(node_id));
    if (pin_index < 0 || pin_index >= static_cast<int>(node->inputs.size()))
        return Result::fail("pin_index out of range (node has " +
                            std::to_string(node->inputs.size()) + " inputs)");
    Result r = setPinDefault(node->inputs[pin_index], value);
    if (!r) return r;
    // Mark this node and everything downstream dirty so the next evaluation picks
    // up the new default (graph-construction scope, same as Faz 3d — the editor's
    // Live path or a future rt.nodes.evaluate performs the actual apply).
    node->dirty = true;
    graph->markDirtyDownstream(node_id);
    return Result::success();
}

namespace {

bool isStructuralNodeProperty(const std::string& name) {
    return name == "id" || name == "typeId" || name == "name" || name == "position" ||
           name == "size" || name == "inputs" || name == "outputs" || name == "metadata";
}

void collectNodeProperties(const nlohmann::json& value, const std::string& prefix,
                           std::vector<NodePropertyInfo>& out) {
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (prefix.empty() && isStructuralNodeProperty(it.key())) continue;
            const std::string path = prefix.empty() ? it.key() : prefix + "." + it.key();
            collectNodeProperties(it.value(), path, out);
        }
        return;
    }
    if (prefix.empty() || value.is_array() || value.is_null()) return;
    NodePropertyInfo info;
    info.name = prefix;
    if (value.is_boolean()) {
        info.data_type = "bool";
        info.value.kind = NodeParamValue::Kind::Bool;
        info.value.bool_value = value.get<bool>();
    } else if (value.is_number_integer() || value.is_number_unsigned()) {
        info.data_type = "int";
        info.value.kind = NodeParamValue::Kind::Int;
        info.value.int_value = value.get<int>();
    } else if (value.is_number_float()) {
        info.data_type = "float";
        info.value.kind = NodeParamValue::Kind::Float;
        info.value.floats[0] = value.get<float>();
    } else if (value.is_string()) {
        info.data_type = "string";
        info.value.kind = NodeParamValue::Kind::String;
        info.value.string_value = value.get<std::string>();
    } else {
        return;
    }
    out.push_back(std::move(info));
}

nlohmann::json* findNodePropertyJson(nlohmann::json& root, const std::string& path) {
    if (path.empty()) return nullptr;
    nlohmann::json* current = &root;
    size_t start = 0;
    while (start < path.size()) {
        const size_t dot = path.find('.', start);
        const std::string key = path.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        if (!current->is_object() || !current->contains(key)) return nullptr;
        current = &(*current)[key];
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    return current;
}

Result serializedNode(UIContext& ctx, const std::string& graph_type, const std::string& graph_name,
                      unsigned int node_id, NodeSystem::GraphBase*& graph,
                      NodeSystem::NodeBase*& node, nlohmann::json& serialized) {
    Result err;
    graph = findNodeGraph(ctx, graph_type, graph_name, err);
    if (!graph) return err;
    node = graph->getNode(node_id);
    if (!node) return Result::fail("node not found: " + std::to_string(node_id));
    auto* terrain_node = dynamic_cast<TerrainNodesV2::TerrainNodeBase*>(node);
    if (!terrain_node)
        return Result::fail("serialized property API currently supports terrain nodes only");
    terrain_node->serializeToJson(serialized);
    return Result::success();
}

} // namespace

Result listNodeProperties(const std::string& graph_type, const std::string& graph_name,
                          unsigned int node_id, std::vector<NodePropertyInfo>& out) {
    out.clear();
    if (!g_ctx) return notBound();
    NodeSystem::GraphBase* graph = nullptr;
    NodeSystem::NodeBase* node = nullptr;
    nlohmann::json serialized;
    Result r = serializedNode(*g_ctx, graph_type, graph_name, node_id, graph, node, serialized);
    if (!r.ok) return r;
    collectNodeProperties(serialized, "", out);
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    return Result::success();
}

Result getNodeProperty(const std::string& graph_type, const std::string& graph_name,
                       unsigned int node_id, const std::string& property, NodeParamValue& out) {
    std::vector<NodePropertyInfo> properties;
    Result r = listNodeProperties(graph_type, graph_name, node_id, properties);
    if (!r.ok) return r;
    for (const auto& item : properties) {
        if (item.name == property) { out = item.value; return Result::success(); }
    }
    return Result::fail("node property not found: " + property);
}

Result setNodeProperty(const std::string& graph_type, const std::string& graph_name,
                       unsigned int node_id, const std::string& property,
                       const NodeParamValue& value) {
    if (!g_ctx) return notBound();
    NodeSystem::GraphBase* graph = nullptr;
    NodeSystem::NodeBase* node = nullptr;
    nlohmann::json serialized;
    Result r = serializedNode(*g_ctx, graph_type, graph_name, node_id, graph, node, serialized);
    if (!r.ok) return r;
    nlohmann::json* target = findNodePropertyJson(serialized, property);
    if (!target || target->is_array() || target->is_object() || target->is_null())
        return Result::fail("node property not found or not scalar: " + property);
    if (target->is_boolean() && value.kind == NodeParamValue::Kind::Bool) *target = value.bool_value;
    else if ((target->is_number_integer() || target->is_number_unsigned()) && value.kind == NodeParamValue::Kind::Int) *target = value.int_value;
    else if (target->is_number() && value.kind == NodeParamValue::Kind::Float) *target = value.floats[0];
    else if (target->is_string() && value.kind == NodeParamValue::Kind::String) *target = value.string_value;
    else return Result::fail("node property type mismatch: " + property);
    auto* terrain_node = dynamic_cast<TerrainNodesV2::TerrainNodeBase*>(node);
    if (!terrain_node)
        return Result::fail("serialized property API currently supports terrain nodes only");
    terrain_node->deserializeFromJson(serialized);
    graph->markDirtyDownstream(node_id);
    return Result::success();
}

// ---------------------------------------------------------------------------
// Camera (Faz 5.1a).
// ---------------------------------------------------------------------------
namespace {

Camera* activeCamera() {
    return (g_ctx && g_ctx->scene.camera) ? g_ctx->scene.camera.get() : nullptr;
}

// Shared post-edit path: re-derive basis vectors, arm the backend camera resync,
// reset accumulation (a real visual change, not a post-process tweak) and render.
void cameraChanged(Camera& cam) {
    cam.update_camera_vectors();
    g_camera_dirty = true;
    resetAccumulation();
    if (g_ctx) g_ctx->start_render = true;
}

} // namespace

Result getCamera(CameraState& out) {
    if (!g_ctx) return notBound();
    Camera* cam = activeCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    out.position = cam->lookfrom;
    out.target = cam->lookat;
    out.up = cam->vup;
    out.fov = cam->vfov;                 // UI treats vfov as the authoritative field
    out.focus_distance = cam->focus_dist;
    out.aperture = cam->aperture;
    return Result::success();
}

Result setCameraPosition(const Vec3& position) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    Camera* cam = activeCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    cam->lookfrom = position;
    cameraChanged(*cam);
    return Result::success();
}

Result setCameraTarget(const Vec3& target) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    Camera* cam = activeCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    cam->lookat = target;
    cameraChanged(*cam);
    return Result::success();
}

Result setCameraFov(float fov) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (fov <= 0.0f || fov >= 180.0f) return Result::fail("fov must be in (0, 180) degrees");
    Camera* cam = activeCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    cam->vfov = fov;   // keep both fields in sync, matching the UI camera panel
    cam->fov = fov;
    cameraChanged(*cam);
    return Result::success();
}

Result setCameraFocusDistance(float focus_distance) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (focus_distance <= 0.0f) return Result::fail("focus_distance must be positive");
    Camera* cam = activeCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    cam->focus_dist = focus_distance;
    cameraChanged(*cam);
    return Result::success();
}

Result setCameraAperture(float aperture) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (aperture < 0.0f) return Result::fail("aperture must be non-negative");
    Camera* cam = activeCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    cam->aperture = aperture;
    cameraChanged(*cam);
    return Result::success();
}

// ---------------------------------------------------------------------------
// World / environment (Faz 5.1c).
// ---------------------------------------------------------------------------
namespace {

void worldChanged() {
    g_world_dirty = true;
    resetAccumulation();
    if (g_ctx) g_ctx->start_render = true;
}

// Re-derive sun_direction from elevation/azimuth (same formula as World.cpp),
// avoiding CUDA make_float3/normalize by writing the components directly.
void recomputeSunDirection(NishitaSkyParams& p) {
    const float elevRad = p.sun_elevation * 3.14159265f / 180.0f;
    const float azimRad = p.sun_azimuth * 3.14159265f / 180.0f;
    float dx = std::cos(elevRad) * std::sin(azimRad);
    float dy = std::sin(elevRad);
    float dz = std::cos(elevRad) * std::cos(azimRad);
    const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (len > 1e-8f) { dx /= len; dy /= len; dz /= len; }
    p.sun_direction.x = dx;
    p.sun_direction.y = dy;
    p.sun_direction.z = dz;
}

// Push the world sun onto the scene's first directional light. WHY THIS EXISTS:
// SceneUI::sync_sun_with_light defaults ON, and SceneUI::processSunSync() runs
// every UI frame reading the directional light back INTO the Nishita params
// (reverse sync). Without a matching forward push, a scripted sun_elevation/
// azimuth would be reverted on the very next frame — the round-trip smoke passes
// (same script call) but the change never sticks visually. Pushing the sun here
// makes the light agree, so reverse-sync finds no discrepancy and leaves the
// scripted value alone. It also makes the change actually visible: the
// directional light is what casts shadows / lights the scene, not the sky disc.
// Mirrors the timeline "world drives light" branch in scene_ui_world.cpp.
const char* worldModeName(WorldMode m) {
    switch (m) {
        case WORLD_MODE_COLOR:   return "solid";
        case WORLD_MODE_HDRI:    return "hdri";
        case WORLD_MODE_NISHITA: return "nishita";
        default:                 return "solid";
    }
}

bool parseWorldMode(const std::string& s, WorldMode& out) {
    if (s == "solid" || s == "color")   { out = WORLD_MODE_COLOR;   return true; }
    if (s == "hdri" || s == "env")      { out = WORLD_MODE_HDRI;    return true; }
    if (s == "nishita" || s == "sky")   { out = WORLD_MODE_NISHITA; return true; }
    return false;
}

void syncDirectionalLightToWorldSun(const NishitaSkyParams& p) {
    if (!g_ctx) return;
    for (auto& light : g_ctx->scene.lights) {
        if (light && light->type() == LightType::Directional) {
            // light->direction is the direction light TRAVELS (sun -> ground),
            // i.e. the negated direction TO the sun.
            light->direction = Vec3(-p.sun_direction.x, -p.sun_direction.y, -p.sun_direction.z);
            light->intensity = p.sun_intensity;
            g_lights_dirty = true;
            break;  // first directional light only, matching the UI sync
        }
    }
}

} // namespace

Result getWorld(WorldState& out) {
    if (!g_ctx) return notBound();
    const NishitaSkyParams p = g_ctx->renderer.world.getNishitaParams();
    out.mode = worldModeName(g_ctx->renderer.world.getMode());
    out.background_color = g_ctx->renderer.world.getColor();
    out.sun_elevation = p.sun_elevation;
    out.sun_azimuth = p.sun_azimuth;
    out.sun_intensity = p.sun_intensity;
    out.atmosphere_intensity = p.atmosphere_intensity;
    out.sun_size = p.sun_size;
    return Result::success();
}

Result setWorldMode(const std::string& mode) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    WorldMode m;
    if (!parseWorldMode(mode, m))
        return Result::fail("unknown world mode '" + mode + "' (expected solid|hdri|nishita)");
    g_ctx->renderer.world.setMode(m);
    worldChanged();
    return Result::success();
}

Result setWorldBackgroundColor(const Vec3& color) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    // The UI writes BOTH: world.setColor() is what the renderer/miss shader reads
    // (also the Nishita sun tint); scene.background_color is the serialized field.
    // Setting only the latter left the render source unchanged.
    g_ctx->renderer.world.setColor(color);
    g_ctx->scene.background_color = color;
    worldChanged();
    return Result::success();
}

Result setWorldSunElevation(float degrees) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    NishitaSkyParams p = g_ctx->renderer.world.getNishitaParams();
    p.sun_elevation = degrees;
    recomputeSunDirection(p);
    g_ctx->renderer.world.setNishitaParams(p);
    syncDirectionalLightToWorldSun(p);  // else processSunSync reverts it next frame
    worldChanged();
    return Result::success();
}

Result setWorldSunAzimuth(float degrees) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    NishitaSkyParams p = g_ctx->renderer.world.getNishitaParams();
    p.sun_azimuth = degrees;
    recomputeSunDirection(p);
    g_ctx->renderer.world.setNishitaParams(p);
    syncDirectionalLightToWorldSun(p);  // else processSunSync reverts it next frame
    worldChanged();
    return Result::success();
}

Result setWorldSunIntensity(float intensity) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (intensity < 0.0f) return Result::fail("sun intensity must be non-negative");
    NishitaSkyParams p = g_ctx->renderer.world.getNishitaParams();
    p.sun_intensity = intensity;
    g_ctx->renderer.world.setNishitaParams(p);
    // Also drive the directional light so the scene actually brightens/dims and
    // the UI Light-Sync path (which mirrors light->intensity into the sun) agrees.
    g_ctx->renderer.world.setSunIntensity(intensity);
    syncDirectionalLightToWorldSun(p);
    worldChanged();
    return Result::success();
}

Result setWorldAtmosphereIntensity(float intensity) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (intensity < 0.0f) return Result::fail("atmosphere intensity must be non-negative");
    NishitaSkyParams p = g_ctx->renderer.world.getNishitaParams();
    p.atmosphere_intensity = intensity;
    g_ctx->renderer.world.setNishitaParams(p);
    worldChanged();
    return Result::success();
}

Result setWorldSunSize(float degrees) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (degrees < 0.0f) return Result::fail("sun size must be non-negative");
    NishitaSkyParams p = g_ctx->renderer.world.getNishitaParams();
    p.sun_size = degrees;
    g_ctx->renderer.world.setNishitaParams(p);
    worldChanged();
    return Result::success();
}

// ---------------------------------------------------------------------------
// Post-processing helpers (Faz 5.1d).
// CRITICAL RULE: Post-processing changes MUST NEVER call resetAccumulation.
// ---------------------------------------------------------------------------
namespace {

std::string toneMapTypeName(ToneMappingType type) {
    switch (type) {
        case ToneMappingType::AGX: return "agx";
        case ToneMappingType::ACES: return "aces";
        case ToneMappingType::Uncharted: return "uncharted";
        case ToneMappingType::Filmic: return "filmic";
        case ToneMappingType::None: return "none";
    }
    return "none";
}

bool parseToneMapType(const std::string& name, ToneMappingType& out) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "agx") { out = ToneMappingType::AGX; return true; }
    if (s == "aces") { out = ToneMappingType::ACES; return true; }
    if (s == "uncharted") { out = ToneMappingType::Uncharted; return true; }
    if (s == "filmic") { out = ToneMappingType::Filmic; return true; }
    if (s == "none") { out = ToneMappingType::None; return true; }
    return false;
}

void postChanged() {
    if (g_ctx) {
        g_ctx->apply_tonemap = true;
        g_ctx->render_settings.persistent_tonemap = true;
    }
}

void stylizeChanged() {
    if (g_ctx) {
        g_ctx->render_settings.stylize_enabled = g_ctx->renderer.stylizeMode.enabled;
        stylize_redisplay = true;
    }
}

} // namespace

Result getPost(PostState& out) {
    if (!g_ctx) return notBound();
    const auto& params = g_ctx->color_processor.params;
    out.exposure = params.global_exposure;
    out.gamma = params.global_gamma;
    out.saturation = params.saturation;
    out.color_temperature = params.color_temperature;
    out.tone_mapping = toneMapTypeName(params.tone_mapping_type);
    out.vignette_enabled = params.enable_vignette;
    out.vignette_strength = params.vignette_strength;
    out.stylize_enabled = g_ctx->renderer.stylizeMode.enabled;
    out.stylize_strength = g_ctx->renderer.stylizeMode.profile.global_strength;
    return Result::success();
}

Result setPostExposure(float exposure) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (exposure < 0.0f) return Result::fail("exposure must be non-negative");
    g_ctx->color_processor.params.global_exposure = exposure;
    postChanged();
    return Result::success();
}

Result setPostGamma(float gamma) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (gamma <= 0.0f) return Result::fail("gamma must be positive");
    g_ctx->color_processor.params.global_gamma = gamma;
    postChanged();
    return Result::success();
}

Result setPostSaturation(float saturation) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (saturation < 0.0f) return Result::fail("saturation must be non-negative");
    g_ctx->color_processor.params.saturation = saturation;
    postChanged();
    return Result::success();
}

Result setPostColorTemperature(float temp_k) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (temp_k <= 0.0f) return Result::fail("color temperature must be positive");
    g_ctx->color_processor.params.color_temperature = temp_k;
    postChanged();
    return Result::success();
}

Result setPostToneMapping(const std::string& type) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    ToneMappingType t;
    if (!parseToneMapType(type, t))
        return Result::fail("unknown tone mapping type '" + type + "' (expected agx|aces|uncharted|filmic|none)");
    g_ctx->color_processor.params.tone_mapping_type = t;
    postChanged();
    return Result::success();
}

Result setPostVignetteEnabled(bool enabled) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    g_ctx->color_processor.params.enable_vignette = enabled;
    postChanged();
    return Result::success();
}

Result setPostVignetteStrength(float strength) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (strength < 0.0f) return Result::fail("vignette strength must be non-negative");
    g_ctx->color_processor.params.vignette_strength = strength;
    postChanged();
    return Result::success();
}

Result setPostStylizeEnabled(bool enabled) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    g_ctx->renderer.stylizeMode.enabled = enabled;
    stylizeChanged();
    return Result::success();
}

Result setPostStylizeStrength(float strength) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (strength < 0.0f) return Result::fail("stylize strength must be non-negative");
    g_ctx->renderer.stylizeMode.profile.global_strength = strength;
    stylizeChanged();
    return Result::success();
}

// ---------------------------------------------------------------------------
// Mesh Modifiers (Faz 5.2b) — Implemented in RtApiModifiers.cpp
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scatter & Foliage System (Faz 5.2c)
// ---------------------------------------------------------------------------

namespace {

InstanceGroup* findScatterGroupHelper(const std::string& key) {
    InstanceManager& im = InstanceManager::getInstance();
    if (key.empty()) return nullptr;
    try {
        size_t idx = 0;
        int id = std::stoi(key, &idx);
        if (idx == key.size()) {
            if (auto g = im.getGroup(id)) return g;
        }
    } catch (...) {}
    return im.findGroupByName(key);
}

} // namespace

Result listScatterGroups(std::vector<rtapi::ScatterGroupInfo>& out_groups) {
    out_groups.clear();
    if (!g_ctx) return notBound();

    InstanceManager& im = InstanceManager::getInstance();
    const auto& groups = im.getGroups();
    out_groups.reserve(groups.size());

    for (const auto& g : groups) {
        if (g.transient) continue;
        rtapi::ScatterGroupInfo info;
        info.id = g.id;
        info.name = g.name;
        info.target_type = (g.target_type == InstanceGroup::TargetType::TERRAIN) ? "terrain" : "mesh";
        info.target_node_name = g.target_node_name;
        info.instance_count = g.getInstanceCount();
        info.triangle_count = g.getTriangleCount();

        for (const auto& src : g.sources) {
            rtapi::ScatterSourceInfo sinfo;
            sinfo.name = src.name;
            sinfo.weight = src.weight;
            sinfo.scale_min = src.settings.scale_min;
            sinfo.scale_max = src.settings.scale_max;
            sinfo.rotation_random_y = src.settings.rotation_random_y;
            sinfo.align_to_normal = src.settings.align_to_normal;
            info.sources.push_back(sinfo);
        }
        out_groups.push_back(info);
    }
    return Result::success();
}

Result createScatterGroup(const std::string& name, const std::string& target_node_name,
                           const std::string& target_type_in, rtapi::ScatterGroupInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (name.empty()) return Result::fail("scatter group name cannot be empty");

    InstanceManager& im = InstanceManager::getInstance();
    std::string type_lower = target_type_in;
    std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    int new_id = im.createGroup(name, target_node_name, {});
    InstanceGroup* group = im.getGroup(new_id);
    if (!group) return Result::fail("failed to create scatter group: " + name);

    group->target_type = (type_lower == "terrain") ? InstanceGroup::TargetType::TERRAIN : InstanceGroup::TargetType::MESH;
    group->target_node_name = target_node_name;

    out_info.id = group->id;
    out_info.name = group->name;
    out_info.target_type = (group->target_type == InstanceGroup::TargetType::TERRAIN) ? "terrain" : "mesh";
    out_info.target_node_name = group->target_node_name;
    out_info.instance_count = 0;
    out_info.triangle_count = 0;
    return Result::success();
}

Result deleteScatterGroup(const std::string& group_id_or_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    InstanceManager& im = InstanceManager::getInstance();
    InstanceGroup* group = findScatterGroupHelper(group_id_or_name);
    if (!group) return Result::fail("scatter group not found: " + group_id_or_name);

    int id = group->id;
    SceneUI::syncInstancesToScene(*g_ctx, *group, true);
    im.deleteGroup(id);

    scheduleSceneMutationRebuilds(*g_ctx, true);
    ui.mesh_cache_valid = false;
    return Result::success();
}

Result clearScatterGroup(const std::string& group_id_or_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    InstanceGroup* group = findScatterGroupHelper(group_id_or_name);
    if (!group) return Result::fail("scatter group not found: " + group_id_or_name);

    group->clearInstances();
    SceneUI::syncInstancesToScene(*g_ctx, *group, true);

    scheduleSceneMutationRebuilds(*g_ctx, true);
    ui.mesh_cache_valid = false;
    return Result::success();
}

Result addScatterSource(const std::string& group_id_or_name, const std::string& mesh_name,
                        float weight, float scale_min, float scale_max,
                        float rotation_y, bool align_to_normal) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    InstanceGroup* group = findScatterGroupHelper(group_id_or_name);
    if (!group) return Result::fail("scatter group not found: " + group_id_or_name);

    if (!objectExists(mesh_name)) return Result::fail("source mesh object not found: " + mesh_name);

    ScatterSource source;
    source.name = mesh_name;
    source.weight = std::clamp(weight, 0.0f, 100.0f);
    source.settings.scale_min = std::max(0.01f, scale_min);
    source.settings.scale_max = std::max(source.settings.scale_min, scale_max);
    source.settings.rotation_random_y = std::clamp(rotation_y, 0.0f, 360.0f);
    source.settings.align_to_normal = align_to_normal;

    std::unordered_set<TriangleMesh*> seenMeshes;
    for (auto& obj : g_ctx->scene.world.objects) {
        if (auto tmesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (tmesh->nodeName != mesh_name || !tmesh->geometry) continue;
            if (!seenMeshes.insert(tmesh.get()).second) continue;
            source.flat_meshes.push_back(tmesh);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (tri->getNodeName() == mesh_name) source.triangles.push_back(tri);
        }
    }
    if (!source.flat_meshes.empty()) source.triangles.clear();
    source.computeCenter();

    group->sources.push_back(std::move(source));
    return Result::success();
}

Result removeScatterSource(const std::string& group_id_or_name, int source_index) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    InstanceGroup* group = findScatterGroupHelper(group_id_or_name);
    if (!group) return Result::fail("scatter group not found: " + group_id_or_name);

    if (source_index < 0 || source_index >= static_cast<int>(group->sources.size())) {
        return Result::fail("invalid scatter source index: " + std::to_string(source_index));
    }

    group->sources.erase(group->sources.begin() + source_index);
    group->clearInstances();
    SceneUI::syncInstancesToScene(*g_ctx, *group, true);

    scheduleSceneMutationRebuilds(*g_ctx, true);
    ui.mesh_cache_valid = false;
    return Result::success();
}

Result setScatterGroupSettings(const std::string& group_id_or_name,
                                const int* target_count, const int* seed,
                                const float* min_distance, const float* slope_max,
                                const float* height_min, const float* height_max,
                                const std::string* density_mask, const std::string* scale_mask) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    InstanceGroup* group = findScatterGroupHelper(group_id_or_name);
    if (!group) return Result::fail("scatter group not found: " + group_id_or_name);

    auto& bs = group->brush_settings;
    if (target_count) bs.target_count = std::clamp(*target_count, 0, 5000000);
    if (seed) bs.seed = *seed;
    if (min_distance) bs.min_distance = std::max(0.0f, *min_distance);
    if (slope_max) bs.slope_max = std::clamp(*slope_max, 0.0f, 90.0f);
    if (height_min) bs.height_min = *height_min;
    if (height_max) bs.height_max = *height_max;
    if (density_mask) bs.density_mask_attribute = *density_mask;
    if (scale_mask) bs.scale_mask_attribute = *scale_mask;

    return Result::success();
}

Result fillScatterGroup(const std::string& group_id_or_name, int& out_spawned) {
    out_spawned = 0;
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    InstanceGroup* group = findScatterGroupHelper(group_id_or_name);
    if (!group) return Result::fail("scatter group not found: " + group_id_or_name);

    if (group->sources.empty()) return Result::fail("scatter group has no source meshes: " + group->name);

    group->clearInstances();

    if (group->target_type == InstanceGroup::TargetType::MESH) {
        if (group->target_node_name.empty()) return Result::fail("scatter group has no target mesh node name");

        std::vector<std::shared_ptr<Triangle>> surfaceTris;
        for (auto& obj : g_ctx->scene.world.objects) {
            if (auto tmesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                if (tmesh->nodeName != group->target_node_name || !tmesh->geometry) continue;
                for (size_t f = 0; f < tmesh->num_triangles(); ++f) {
                    surfaceTris.push_back(std::make_shared<Triangle>(tmesh, static_cast<uint32_t>(f)));
                }
            } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                if (tri->getNodeName() == group->target_node_name) {
                    surfaceTris.push_back(tri);
                }
            }
        }
        if (surfaceTris.empty()) return Result::fail("target surface mesh has no triangles: " + group->target_node_name);

        out_spawned = group->scatterFillMesh(surfaceTris);
    } else {
        return Result::fail("terrain fill via API requires selecting a valid terrain node name");
    }

    SceneUI::syncInstancesToScene(*g_ctx, *group, false);

    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_bvh_rebuild_pending = true;
    g_optix_rebuild_pending = true;
    g_vulkan_rebuild_pending = true;
    g_viewport_raster_rebuild_pending = true;
    ui.mesh_cache_valid = false;

    scheduleSceneMutationRebuilds(*g_ctx, true);
    return Result::success();
}

Result addScatterInstance(const std::string& group_id_or_name, Vec3 pos, Vec3 rot, Vec3 scale, int source_index) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    InstanceGroup* group = findScatterGroupHelper(group_id_or_name);
    if (!group) return Result::fail("scatter group not found: " + group_id_or_name);

    InstanceTransform transform(pos, rot, scale);
    transform.source_index = source_index;
    group->addInstance(transform);

    SceneUI::syncInstancesToScene(*g_ctx, *group, false);

    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_bvh_rebuild_pending = true;
    g_optix_rebuild_pending = true;
    g_vulkan_rebuild_pending = true;
    g_viewport_raster_rebuild_pending = true;
    ui.mesh_cache_valid = false;

    scheduleSceneMutationRebuilds(*g_ctx, true);
    return Result::success();
}

// ---------------------------------------------------------------------------
// Physics Engine (Faz 5.3a)
// ---------------------------------------------------------------------------

static RayTrophiSim::RigidBodyObject* findRigidBodyForNode(const std::string& object_name) {
    if (!g_ctx) return nullptr;
    for (auto& rb : g_ctx->scene.rigid_bodies) {
        if (rb.source_name == object_name) return &rb;
    }
    return nullptr;
}

Result getPhysicsBody(const std::string& object_name, rtapi::PhysicsBodyInfo& out_info) {
    if (!g_ctx) return notBound();
    if (!objectExists(object_name)) return Result::fail("object not found: " + object_name);

    auto rb = findRigidBodyForNode(object_name);
    if (!rb) return Result::fail("no physics body attached to object: " + object_name);

    out_info.object_name = object_name;
    out_info.kind = (rb->kind == RayTrophiSim::BodyKind::Cloth) ? "cloth" :
                    (rb->kind == RayTrophiSim::BodyKind::SoftBody) ? "soft" : "rigid";
    out_info.motion_type = (rb->motion_type == RayTrophiSim::RigidBodyMotionType::Static) ? "static" :
                           (rb->motion_type == RayTrophiSim::RigidBodyMotionType::Kinematic) ? "kinematic" : "dynamic";
    out_info.shape = (rb->shape == RayTrophiSim::RigidBodyShape::Sphere) ? "sphere" :
                     (rb->shape == RayTrophiSim::RigidBodyShape::Capsule) ? "capsule" :
                     (rb->shape == RayTrophiSim::RigidBodyShape::Mesh) ? "mesh" : "box";
    out_info.enabled = rb->enabled;
    out_info.mass = rb->mass;
    out_info.friction = rb->friction;
    out_info.restitution = rb->restitution;
    out_info.linear_damping = rb->linear_damping;
    out_info.angular_damping = rb->angular_damping;
    out_info.gravity_scale = rb->gravity_scale;
    out_info.soft_stiffness = rb->getSoftStiffness();
    out_info.soft_pressure = rb->getSoftPressure();
    out_info.soft_damping = rb->getSoftDamping();
    return Result::success();
}

Result addPhysicsBody(const std::string& object_name, const std::string& kind_in,
                      const std::string& motion_type_in, const std::string& shape_in,
                      float mass, rtapi::PhysicsBodyInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!objectExists(object_name)) return Result::fail("object not found: " + object_name);

    std::string kind_str = kind_in;
    std::transform(kind_str.begin(), kind_str.end(), kind_str.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    std::string motion_str = motion_type_in;
    std::transform(motion_str.begin(), motion_str.end(), motion_str.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    std::string shape_str = shape_in;
    std::transform(shape_str.begin(), shape_str.end(), shape_str.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    RayTrophiSim::BodyKind bkind = RayTrophiSim::BodyKind::Rigid;
    if (kind_str == "soft" || kind_str == "softbody" || kind_str == "soft_body") {
        bkind = RayTrophiSim::BodyKind::SoftBody;
    } else if (kind_str == "cloth") {
        bkind = RayTrophiSim::BodyKind::Cloth;
    }

    bool is_dynamic = (motion_str != "static");
    RayTrophiSim::RigidBodyObject* rb = nullptr;

    if (bkind == RayTrophiSim::BodyKind::Rigid) {
        rb = g_ctx->scene.addRigidBodyForObject(object_name, is_dynamic);
    } else {
        rb = g_ctx->scene.addSoftBodyForObject(object_name, bkind);
    }

    if (!rb) return Result::fail("failed to add physics body for object: " + object_name);

    if (motion_str == "static") rb->motion_type = RayTrophiSim::RigidBodyMotionType::Static;
    else if (motion_str == "kinematic") rb->motion_type = RayTrophiSim::RigidBodyMotionType::Kinematic;
    else rb->motion_type = RayTrophiSim::RigidBodyMotionType::Dynamic;

    if (shape_str == "sphere") rb->shape = RayTrophiSim::RigidBodyShape::Sphere;
    else if (shape_str == "capsule") rb->shape = RayTrophiSim::RigidBodyShape::Capsule;
    else if (shape_str == "mesh") rb->shape = RayTrophiSim::RigidBodyShape::Mesh;
    else rb->shape = RayTrophiSim::RigidBodyShape::Box;

    if (mass > 0.0f) rb->mass = mass;

    g_ctx->scene.captureRigidBodyRestPose(*rb);
    if (g_ctx->scene.rigid_body_system) {
        g_ctx->scene.rigid_body_system->resetRuntime(true);
    }

    return getPhysicsBody(object_name, out_info);
}

Result removePhysicsBody(const std::string& object_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!objectExists(object_name)) return Result::fail("object not found: " + object_name);

    if (!g_ctx->scene.removeRigidBodyForObject(object_name)) {
        return Result::fail("no physics body attached to object: " + object_name);
    }
    return Result::success();
}

Result updatePhysicsBody(const std::string& object_name,
                        const std::string* kind, const std::string* motion_type,
                        const std::string* shape, const bool* enabled,
                        const float* mass, const float* friction, const float* restitution,
                        const float* linear_damping, const float* angular_damping,
                        const float* gravity_scale, const float* soft_stiffness,
                        const float* soft_pressure, const float* soft_damping) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!objectExists(object_name)) return Result::fail("object not found: " + object_name);

    auto rb = findRigidBodyForNode(object_name);
    if (!rb) return Result::fail("no physics body attached to object: " + object_name);

    if (kind) {
        std::string k = *kind;
        std::transform(k.begin(), k.end(), k.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (k == "soft" || k == "softbody" || k == "soft_body") rb->kind = RayTrophiSim::BodyKind::SoftBody;
        else if (k == "cloth") rb->kind = RayTrophiSim::BodyKind::Cloth;
        else rb->kind = RayTrophiSim::BodyKind::Rigid;
    }
    if (motion_type) {
        std::string m = *motion_type;
        std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (m == "static") { rb->motion_type = RayTrophiSim::RigidBodyMotionType::Static; rb->dynamic = false; }
        else if (m == "kinematic") { rb->motion_type = RayTrophiSim::RigidBodyMotionType::Kinematic; rb->dynamic = true; }
        else { rb->motion_type = RayTrophiSim::RigidBodyMotionType::Dynamic; rb->dynamic = true; }
    }
    if (shape) {
        std::string s = *shape;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (s == "sphere") rb->shape = RayTrophiSim::RigidBodyShape::Sphere;
        else if (s == "capsule") rb->shape = RayTrophiSim::RigidBodyShape::Capsule;
        else if (s == "mesh") rb->shape = RayTrophiSim::RigidBodyShape::Mesh;
        else rb->shape = RayTrophiSim::RigidBodyShape::Box;
    }
    if (enabled) rb->enabled = *enabled;
    if (mass) rb->mass = std::max(0.001f, *mass);
    if (friction) rb->friction = std::clamp(*friction, 0.0f, 10.0f);
    if (restitution) rb->restitution = std::clamp(*restitution, 0.0f, 1.0f);
    if (linear_damping) rb->linear_damping = std::clamp(*linear_damping, 0.0f, 10.0f);
    if (angular_damping) rb->angular_damping = std::clamp(*angular_damping, 0.0f, 10.0f);
    if (gravity_scale) rb->gravity_scale = *gravity_scale;
    if (soft_stiffness) rb->setSoftStiffness(std::clamp(*soft_stiffness, 0.0f, 1.0f));
    if (soft_pressure) rb->setSoftPressure(std::max(0.0f, *soft_pressure));
    if (soft_damping) rb->setSoftDamping(std::clamp(*soft_damping, 0.0f, 1.0f));

    g_ctx->scene.captureRigidBodyRestPose(*rb);
    if (g_ctx->scene.rigid_body_system) {
        g_ctx->scene.rigid_body_system->resetRuntime(true);
    }
    return Result::success();
}

Result resetPhysicsSimulation() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    g_ctx->scene.ensureRigidBodySystem();
    if (g_ctx->scene.rigid_body_system) {
        g_ctx->scene.rigid_body_system->resetRuntime(true);
    }
    return Result::success();
}

Result stepPhysicsSimulation(float dt) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (dt <= 0.0f) dt = 0.0166667f;

    g_ctx->scene.ensureRigidBodySystem();
    if (g_ctx->scene.rigid_body_system) {
        RayTrophiSim::SimulationContext simCtx = g_ctx->scene.simulation_world.makeContext(dt, 0, 1);
        simCtx.dt = dt;
        g_ctx->scene.rigid_body_system->step(simCtx);
    }
    return Result::success();
}

Result setPhysicsGravity(Vec3 gravity) {
    if (!g_ctx) return notBound();
    g_ctx->scene.ensureRigidBodySystem();
    if (g_ctx->scene.rigid_body_system) {
        g_ctx->scene.rigid_body_system->setGravity(gravity);
    }
    return Result::success();
}

Result getPhysicsGravity(Vec3& out_gravity) {
    out_gravity = Vec3(0.0f, -9.81f, 0.0f);
    if (!g_ctx) return notBound();
    g_ctx->scene.ensureRigidBodySystem();
    return Result::success();
}

// ---------------------------------------------------------------------------
// Fluid Simulation Engine (Faz 5.3b) — Implemented in RtApiFluid.cpp
// ---------------------------------------------------------------------------

Result runScriptFile(const std::string& filepath) {
    if (!g_ctx) return notBound();
    if (!rtpython::isInitialized()) return Result::fail("Python runtime is not initialized");
    const rtpython::ExecutionResult result = rtpython::executeFile(filepath);
    return result.ok ? Result::success() : Result::fail(result.error);
}

} // namespace rtapi
