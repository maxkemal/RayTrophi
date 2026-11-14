// SceneUI in the Player: state carrier + explicit "editor not available" stubs.
//
// PLAYER BUILD ONLY. Studio links the real editor (scene_ui*.cpp) and defines
// `ui` in Main.cpp; the two must never be linked into the same target (Studio
// lists this file as None).
//
// WHY THIS FILE EXISTS: shared API code (rtapi, ~18 files, ~170 accesses) reads
// and writes state that lives on the SceneUI object: the timeline cursor,
// viewport shading mode, mesh-cache validity, panel visibility flags. That is
// the debt CLAUDE.md rule 1 names ("the UI must not own state"). Until that
// state moves to the core, the Player owns a SceneUI object that is NEVER drawn
// and only carries those fields, so the shared code links and behaves the same.
//
// Operations whose LOGIC still lives in the editor (mesh edit operators,
// fracture, duplicate, geometry-graph evaluation, scatter/VDB sync) are defined
// here as explicit refusals: bool operations return false (callers turn that
// into an API error), void operations log once. Nothing here pretends to
// succeed. Each entry is a to-do: move the logic to the core, delete the stub.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md (Faz 3d).

#include "scene_ui.h"
#include "Core/VulkanSceneSync.h"
#include "UI/UvEditorUI.h"
#include "UI/TemplateHubUI.h"
#include "globals.h"

#include <atomic>
#include <string>

// The Player's SceneUI state object (see header comment). Never drawn.
SceneUI ui;

namespace {

// One log line per operation name, so a script that hammers an unavailable
// method does not flood SceneLog.
void editorUnavailable(const char* operation) {
    static std::atomic<unsigned> logged_mask{0};
    static const char* const kNames[] = {
        "mesh edit operators", "fracture", "duplicate", "geometry graph evaluation",
        "animation processing", "scatter scene sync", "VDB volume GPU sync",
        "fracture recipe replay", "UV editor", "mesh cache sync"};
    unsigned bit = 0;
    for (unsigned i = 0; i < sizeof(kNames) / sizeof(kNames[0]); ++i) {
        if (std::string(kNames[i]) == operation) { bit = 1u << i; break; }
    }
    if (bit != 0 && (logged_mask.fetch_or(bit) & bit) != 0) return;
    SCENE_LOG_WARN(std::string("[Player] '") + operation +
                   "' is implemented in the Studio editor and is not available in RayTrophi Player.");
}

} // namespace

// The real destructor releases an editor preview texture; the Player never creates one.
SceneUI::~SceneUI() {
}

// ------------------------------------------------------------- mesh edit mode

bool SceneUI::ensureEditableMeshCache(UIContext& /*ctx*/, const std::string& /*objectName*/) {
    editorUnavailable("mesh edit operators");
    return false;
}

bool SceneUI::ensureEditableHalfEdge() {
    return false;
}

bool SceneUI::extrudeSelectedMeshFaces(UIContext& /*ctx*/, float /*distance*/) {
    editorUnavailable("mesh edit operators");
    return false;
}

bool SceneUI::insetSelectedMeshFaces(UIContext& /*ctx*/, float /*amount*/) {
    editorUnavailable("mesh edit operators");
    return false;
}

bool SceneUI::bevelSelectedEdges(UIContext& /*ctx*/, float /*width*/, int /*segments*/, bool /*roundProfile*/) {
    editorUnavailable("mesh edit operators");
    return false;
}

bool SceneUI::loopCutSelectedEdges(UIContext& /*ctx*/, float /*t*/) {
    editorUnavailable("mesh edit operators");
    return false;
}

bool SceneUI::dissolveSelectedEdges(UIContext& /*ctx*/) {
    editorUnavailable("mesh edit operators");
    return false;
}

bool SceneUI::dissolveSelectedVertices(UIContext& /*ctx*/) {
    editorUnavailable("mesh edit operators");
    return false;
}

bool SceneUI::mergeSelectedVerticesToCenter(UIContext& /*ctx*/) {
    editorUnavailable("mesh edit operators");
    return false;
}

bool SceneUI::weldSelectedVerticesByDistance(UIContext& /*ctx*/, float /*distance*/) {
    editorUnavailable("mesh edit operators");
    return false;
}

// Refusing means "not handled incrementally": SceneCommand falls back to its
// own full resync path, which is the documented behaviour for a UI that cannot
// service the object.
bool SceneUI::adoptExternalFlatSoaEdit(UIContext& /*ctx*/, const std::string& /*objectName*/) {
    return false;
}

// ------------------------------------------------------------------ fracture

void SceneUI::fractureSelectedMesh(UIContext& /*ctx*/, const std::string& /*node*/,
                                   int /*site_count*/, uint32_t /*seed*/, int /*pattern*/,
                                   const SceneData::FractureRecipe* /*settings*/) {
    // No shard record is written, so rtapi reports the fracture as failed.
    editorUnavailable("fracture");
}

void SceneUI::unfractureMesh(UIContext& /*ctx*/, const std::string& /*node*/) {
    editorUnavailable("fracture");
}

void SceneUI::replayFractureRecipes(UIContext& ctx) {
    // A project that recorded fractures opens UNFRACTURED here. Say so loudly:
    // a whole mesh where shards were saved is exactly the quiet, plausible
    // result nobody would report as a bug.
    if (!ctx.scene.fracture_recipes.empty()) {
        SCENE_LOG_WARN("[Player] Project records " + std::to_string(ctx.scene.fracture_recipes.size()) +
                       " fracture recipe(s); the fracture cutter lives in the Studio editor, so these "
                       "objects are loaded UNFRACTURED.");
    }
}

// ------------------------------------------------------- scene / cache sync

void SceneUI::invalidateCache() {
    mesh_cache_valid = false;
}

void SceneUI::syncMeshCacheVisibility(const std::string& /*objectName*/, bool /*visible*/) {
    // Editor selection cache only; the scene object's own visibility is set by the caller.
}

void SceneUI::triggerDuplicate(UIContext& /*ctx*/) {
    // Selection is left unchanged, so rtapi reports "duplicate failed".
    editorUnavailable("duplicate");
}

void SceneUI::processAnimations(UIContext& /*ctx*/) {
    editorUnavailable("animation processing");
}

bool SceneUI::evaluateGeometryGraph(UIContext& /*ctx*/, const std::string& /*objectName*/,
                                    GeometryNodesV2::GeometryNodeGraphV2& /*graph*/,
                                    bool /*livePreview*/, bool /*deferScenePublish*/) {
    editorUnavailable("geometry graph evaluation");
    return false;
}

void SceneUI::syncInstancesToScene(UIContext& /*ctx*/, InstanceGroup& /*group*/, bool /*clear_only*/,
                                   bool /*force_cpu_compat*/) {
    editorUnavailable("scatter scene sync");
}

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

// Not an editor feature: core API (fluid domains, volume materials, VDB import)
// republishes the volume table through this entry. The Player has one Vulkan
// device, so only the viewport backend receives it (Core/VulkanSceneSync).
void SceneUI::syncVDBVolumesToGPU(UIContext& ctx) {
    AppRuntime::publishVolumes(ctx.scene, ctx.renderer, nullptr, g_viewport_backend.get());
}

// Studio drops to the Solid viewport before a scene swap so a heavy mode is not
// bound while geometry is torn down. The Player has no raster/Rendered viewport
// switch yet; record the same state so `viewport.shading` reads consistently.
void enterSolidViewportForSceneLoad(SceneUI& shell_ui, const char* /*reason*/) {
    const int previous = shell_ui.viewport_settings.shading_mode;
    // 0 = Solid and 3 = Matcap are already light raster modes (same rule as Studio).
    if (previous != 1 && previous != 2) return;
    if (!g_scene_load_solid_guard) return;
    shell_ui.viewport_settings.shading_mode = 0;
}

// ------------------------------------------------------------ editor windows

namespace UvEditorUI {
void open(const std::string& /*object*/) {
    editorUnavailable("UV editor");
}
} // namespace UvEditorUI

namespace raytrophi::templates {

// Recent-project list is a Studio hub feature. The Player keeps an empty hub
// object so ProjectManager's bookkeeping call has somewhere to go.
TemplateHubUI::TemplateHubUI() {
}

TemplateHubUI& TemplateHubUI::instance() {
    static TemplateHubUI hub;
    return hub;
}

void TemplateHubUI::addRecentProject(const std::filesystem::path& /*project_path*/) {
}

} // namespace raytrophi::templates
