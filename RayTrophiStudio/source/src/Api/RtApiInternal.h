/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiInternal.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * Internal shared declarations for modular RtApi translation units.
 */

#pragma once

#include "Api/RtApi.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "SceneCommand.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Global C symbols from Main.cpp / Scene
extern SceneUI ui;
extern bool g_solid_viewport_active;
extern bool g_geometry_dirty;
extern std::atomic<uint64_t> g_scene_geometry_generation;
extern bool g_bvh_rebuild_pending;
extern bool g_optix_rebuild_pending;
extern bool g_vulkan_rebuild_pending;
extern bool g_viewport_raster_rebuild_pending;

class PrincipledBSDF;

namespace rtapi {

extern UIContext* g_ctx;
extern SceneHistory* g_history;
extern RenderJobInfo g_render_job;

inline Result notBound() { return Result::fail("rtapi is not bound to a UIContext"); }
inline bool renderJobActive() { return g_render_job.state == RenderJobState::Rendering; }
bool objectExists(const std::string& name);
void pollTerrainEvaluations();

// Distinct material ids referenced by an object's flat meshes, in first-seen
// order. Defined in RtApi.cpp; shared with RtApiMaterial.cpp.
std::vector<uint16_t> objectMaterialIds(UIContext& ctx, const std::string& object_name);

// Principled BSDF parameter read/write vocabulary. Defined in RtApi.cpp, where
// the OBJECT-scoped rt.material.get/set live; RtApiMaterial.cpp reuses the same
// enum/parse/read/write so a MATERIAL-scoped setter (material.set_param) cannot
// drift from the object-scoped one's field mapping or validation range.
enum class MaterialParamKind {
    BaseColor, Roughness, Metallic, Specular, Emission, EmissionStrength,
    Transmission, Ior, Opacity,
    IsBubble, BubbleIor, BubbleFilm,
    ResinDensity, ResinColor, ResinRoughness, ResinInclusion, ResinDirt,
    ResinDirtColor, ResinInclusionScale, ResinShard, ResinShardHue, ResinObjectSpace,
    DustStyle, DustColorA, DustColorB, ShardShape,
    UvScaleX, UvScaleY, UvOffsetX, UvOffsetY
};

struct MaterialValue {
    float scalar = 0.0f;
    Vec3 color;
};

bool parseMaterialParam(const std::string& name, MaterialParamKind& out, bool& is_color);
MaterialValue readMaterialValue(const PrincipledBSDF& material, MaterialParamKind kind);
void writeMaterialValue(PrincipledBSDF& material, MaterialParamKind kind, const MaterialValue& value);
// Range/finiteness checks shared by both scoped setters (unit range for
// roughness/metallic/specular/transmission/opacity, ior in [1,10], emission
// strength non-negative). Returns Result::success() when `value` is acceptable
// for `kind`.
Result validateMaterialParamValue(MaterialParamKind kind, bool is_color, const MaterialValue& value);

// Particle emitters, particle colliders and grid domains all live on ONE
// runtime, so RtApiFluid.cpp and RtApiParticle.cpp must reach it through the
// same accessor and invalidate it the same way — a facade that edits the
// runtime without invalidateScriptSimulation() leaves the cached sim frames
// and the timeline resync stale. Defined in RtApiFluid.cpp.
RayTrophiSim::ParticleSimulationSystem& scriptSimulationRuntime();
void invalidateScriptSimulation();

} // namespace rtapi
