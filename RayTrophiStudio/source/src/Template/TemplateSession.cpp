#include "Template/TemplateSession.h"

#include "Template/TemplateLoader.h"
#include "Template/TemplateRegistry.h"
#include "Template/TemplateRecipeStage.h"
#include "Template/TemplateUiStateAdapter.h"
#include "Template/PathUtils.h"
#include "UI/TemplateHubUI.h"
#include "ProjectManager.h"
#include "MaterialManager.h"
#include "SceneCommand.h"
#include "scene_ui.h"

#include <algorithm>
#include <utility>

namespace raytrophi::templates {
using pathutils::pathToUtf8;
namespace {

TemplateOpenResult rejected(const std::string& id, const std::string& code,
                            std::vector<std::string> errors,
                            std::vector<std::string> warnings = {}) {
    TemplateOpenResult result;
    result.template_id = id;
    result.code = code;
    result.errors = std::move(errors);
    result.warnings = std::move(warnings);
    return result;
}

TemplateOpenResult ok(const std::string& id, const std::string& path) {
    TemplateOpenResult result;
    result.template_id = id;
    result.state = "opened";
    result.code = "opened";
    result.opened = true;
    return result;
}

} // namespace

TemplateSession& TemplateSession::instance() {
    static TemplateSession session;
    return session;
}

TemplateOpenResult TemplateSession::open(const std::string& template_id,
                                         const std::string& conflict_policy,
                                         UIContext& context,
                                         SceneUI& ui,
                                         SceneHistory* history) const {
    TemplateConflictPolicy policy;
    if (!TemplateLoader::parseConflictPolicy(conflict_policy, policy)) {
        return rejected(template_id, "invalid_parameter",
                        {"conflict_policy must be 'reject' or 'discard'"});
    }

    const auto plan = TemplateLoader::instance().prepare(
        template_id, ProjectManager::getInstance().hasUnsavedChanges(), policy);
    if (!plan.ready) return rejected(template_id, plan.code, plan.errors, plan.warnings);

    // Support opening custom user templates and project-based templates
    if (plan.scene_type == "project") {
        bool opened_proj = ProjectManager::getInstance().openProject(
            pathToUtf8(plan.scene_path), context.scene, context.render_settings, context.renderer, context.backend_ptr);
        if (!opened_proj) {
            return rejected(template_id, "project_open_failed", {"failed to open project template"});
        }
        if (!context.scene.camera) {
            auto fallback_camera = std::make_shared<Camera>(
                Vec3(11.0f, 8.0f, 14.0f), Vec3(0.0f, 0.25f, 0.0f),
                Vec3(0.0f, 1.0f, 0.0f), 40.0f, 16.0f / 9.0f,
                0.0f, (Vec3(11.0f, 8.0f, 14.0f) - Vec3(0.0f, 0.25f, 0.0f)).length(), 0);
            fallback_camera->nodeName = "Startup Camera";
            context.scene.addCamera(fallback_camera);
            context.scene.setActiveCamera(context.scene.cameras.size() - 1);
        }
        context.scene.initialized = true;
        context.selection.clearSelection();
        if (history) history->clear();
        ui.timeline.reset();
        ui.invalidateCache();
        ui.rebuildMeshCache(context.scene.world.objects);
        return ok(template_id, pathToUtf8(plan.scene_path));
    }

    if (plan.scene_type != "recipe") {
        return rejected(template_id, "unsupported_scene_type",
                        {"unsupported scene type: " + plan.scene_type});
    }

    const TemplateMetadata* metadata = TemplateRegistry::instance().find(template_id);
    if (!metadata) {
        return rejected(template_id, "template_not_found",
                        {"template not found: " + template_id});
    }

    // Construct every fallible CPU-side recipe object before crossing the active
    // scene mutation boundary. A staging failure leaves the current project intact.
    const auto staged = TemplateRecipeStager::stage(plan.scene_path);
    if (!staged.ready) return rejected(template_id, staged.code, staged.errors);

    ProjectManager::getInstance().newProject(context.scene, context.renderer);
    if (staged.mesh && staged.material && staged.mesh->geometry) {
        const uint16_t material_id = MaterialManager::getInstance().getOrCreateMaterialID(
            staged.material->materialName, staged.material);
        uint16_t* material_ids = staged.mesh->geometry->get_attribute_data_mut<uint16_t>("materialID");
        if (material_ids) {
            std::fill_n(material_ids, staged.mesh->geometry->get_vertex_count(), material_id);
        }
    }
    if (staged.mesh) context.scene.world.objects.push_back(staged.mesh);
    if (staged.light) context.scene.lights.push_back(staged.light);
    if (staged.camera) {
        context.scene.cameras.clear();
        context.scene.camera = nullptr;
        context.scene.addCamera(staged.camera);
        // addCamera() only auto-activates the first entry. Template creation
        // can follow a transient/default camera, so make the staged camera the
        // authoritative active camera before any backend or CPU sync occurs.
        context.scene.setActiveCamera(context.scene.cameras.size() - 1);
    }
    if (!context.scene.camera) {
        // Startup must always have an active camera. This fallback keeps an
        // incomplete/legacy template from producing a black first viewport.
        auto fallback_camera = std::make_shared<Camera>(
            Vec3(11.0f, 8.0f, 14.0f), Vec3(0.0f, 0.25f, 0.0f),
            Vec3(0.0f, 1.0f, 0.0f), 40.0f, 16.0f / 9.0f,
            0.0f, (Vec3(11.0f, 8.0f, 14.0f) - Vec3(0.0f, 0.25f, 0.0f)).length(), 0);
        fallback_camera->nodeName = "Startup Camera";
        context.scene.cameras.clear();
        context.scene.camera = nullptr;
        context.scene.addCamera(fallback_camera);
        context.scene.setActiveCamera(context.scene.cameras.size() - 1);
    }
    if (staged.preset == "general_scene") {
        context.renderer.world.setMode(WORLD_MODE_COLOR);
        context.renderer.world.setColor(Vec3(0.224f, 0.224f, 0.224f));
        context.renderer.world.setColorIntensity(1.0f);
        context.scene.background_color = Vec3(0.224f, 0.224f, 0.224f);
    }
    // `initialized` gates the entire interactive render loop, including an
    // empty raster viewport and world-only rendered frames. It means the
    // editor session is ready here; it must not be treated as "has geometry".
    context.scene.initialized = true;
    context.selection.clearSelection();
    if (history) history->clear();
    ui.timeline.reset();
    context.scene.timeline.current_frame = 0;
    context.render_settings.animation_current_frame = 0;
    context.render_settings.animation_playback_frame = 0;
    context.active_model_path.clear();
    ui.invalidateCache();
    ui.rebuildMeshCache(context.scene.world.objects);
    // Match the canonical Open/Import picking preparation order. Brush tools
    // need a real surface HitRecord (face, barycentrics, UV and normal), not the
    // projected-bounds fallback used by object selection. Synchronize the flat
    // mesh caches first and only then build the CPU BVH so paint, sculpt, hair
    // and placement all see the same staged geometry on their first ray.
    ui.syncAllTransformedVertices(context.scene);
    ui.picking_vertices_synced = true;
    context.renderer.rebuildBVH(context.scene, context.render_settings.UI_use_embree);
    if (staged.mesh && metadata->ui_state.frame_target == staged.mesh->nodeName) {
        context.selection.selectObject(staged.mesh, 0, staged.mesh->nodeName);
    }

    const auto ui_result = TemplateUiStateAdapter::apply(metadata->ui_state, ui);
    // UI-state application may adjust the active camera's look target/focus.
    // Rebuild the derived camera basis after those fields are final; otherwise
    // get_ray() can retain the default -Z direction while lookfrom/lookat report
    // the staged procedural camera, making CPU center rays miss the cube.
    if (context.scene.camera) {
        context.scene.camera->update_camera_vectors();
    }
    context.sample_count = 0;
    context.renderer.resetCPUAccumulation();
    if (context.backend_ptr) context.backend_ptr->resetAccumulation();

    g_camera_dirty = true;
    g_lights_dirty = true;
    g_world_dirty = true;
    g_geometry_dirty = true;
    g_materials_dirty = true;
    g_gas_volumes_dirty = true;
    // Besides arming every backend, this invalidates any async CPU-BVH snapshot
    // still running for the scene that newProject just discarded.
    scheduleSceneMutationRebuilds(context, true);
    // The recipe commit already built the complete CPU BVH synchronously above.
    // Do not let an async snapshot captured during newProject/empty-scene
    // teardown replace that valid staged BVH on the first CPU frame. Advance the
    // generation once more and consume the pending flag; later Add/Import
    // mutations can use the normal lazy path again.
    extern uint64_t g_cpu_bvh_requested_generation;
    ++g_cpu_bvh_requested_generation;
    g_bvh_rebuild_pending = false;
    g_viewport_raster_rebuild_pending = true;
    g_vulkan_rebuild_pending = context.render_settings.use_vulkan;
    g_optix_rebuild_pending = context.render_settings.use_optix;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    if (context.render_settings.use_vulkan) context.render_settings.backend_changed = true;
    context.start_render = true;

    TemplateHubUI::instance().addRecentProject(metadata->display_name);

    TemplateOpenResult result;
    result.template_id = template_id;
    result.state = "opened";
    result.code = "opened";
    result.opened = true;
    result.ui_state_applied = ui_result.applied;
    result.warnings = ui_result.warnings;
    return result;
}

} // namespace raytrophi::templates
