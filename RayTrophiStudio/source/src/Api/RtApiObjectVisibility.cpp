/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiObjectVisibility.cpp
* Author:        Kemal Demirtas
* Date:          October 2026
* License:       MIT
* =========================================================================
*
* Object visibility (the outliner eye). This used to live only in
* scene_ui_hierarchy.cpp, so a script could not hide a ground plane to measure
* what it costs -- the outliner now calls setObjectVisible and keeps no
* visibility logic of its own.
*
* Two layers are written and both matter:
*   - Hittable::visible on every scene entry carrying the name. Backend
*     REBUILDS read it (updateGeometry, raster rebuild, OptiX snapshot), so a
*     flag left stale resurrects the object on the next topology change.
*   - setVisibilityByNodeName on the viewport AND render backend. This is the
*     immediate effect (TLAS / raster instance masks) without a rebuild.
*/

#include "RtApiInternal.h"

#include <memory>
#include <string>
#include <unordered_set>

#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"
#include "HittableInstance.h"
#include "globals.h"           // SCENE_LOG_INFO
#include "MeshEdit/SplineObject.h"
#include "ProjectManager.h"
#include "Renderer.h"
#include "Triangle.h"
#include "TriangleMesh.h"

// Global scope, NOT inside namespace rtapi (a declaration there would be a
// second, never-defined symbol). See the same note in RtApi.cpp.
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;

namespace rtapi {
namespace {

// Same role resolution the outliner used: the viewport backend draws the
// editor view, the render backend is the RT/final path. In some modes one
// object serves both roles; it must then be told only once.
Backend::IViewportBackend* viewportBackend(UIContext& ctx) {
    if (g_viewport_backend) return g_viewport_backend.get();
    return dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
}

Backend::IBackend* renderBackend(UIContext& ctx) {
    if (g_backend) return g_backend.get();
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr)
        return ctx.backend_ptr;
    return nullptr;
}

// Visits every scene entry that carries `name`, including a facade's
// parentMesh (a flat mesh reached through a facade is the real geometry; the
// facade alone is a handle). Flat SoA meshes are visited directly -- a
// facade-only scan would report a project-sidecar object as absent.
template <typename Fn>
void forEachEntry(SceneData& scene, const std::string& name, Fn&& fn) {
    std::unordered_set<const Hittable*> seen;
    auto visit = [&](Hittable* h) {
        if (h && seen.insert(h).second) fn(*h);
    };
    for (const auto& obj : scene.world.objects) {
        if (!obj) continue;
        if (obj->isTriangleMesh()) {
            auto* tm = static_cast<TriangleMesh*>(obj.get());
            if (tm->nodeName == name) visit(tm);
        } else if (obj->isTriangle()) {
            auto* tri = static_cast<Triangle*>(obj.get());
            if (tri->getNodeName() != name) continue;
            visit(tri);
            if (tri->parentMesh) visit(tri->parentMesh.get());
        } else if (auto* inst = dynamic_cast<HittableInstance*>(obj.get())) {
            if (inst->node_name == name) visit(inst);
        } else if (auto* spline = dynamic_cast<MeshEdit::SplineObject*>(obj.get())) {
            if (spline->nodeName == name) visit(spline);
        }
    }
}

} // namespace

Result setObjectVisible(const std::string& name, bool visible) {
    if (!g_ctx) return notBound();
    if (name.empty()) return Result::fail("object name must not be empty");
    UIContext& ctx = *g_ctx;

    const bool pending = ctx.scene.isEditorPendingDeleteObjectName(name);
    if (pending && visible)
        return Result::fail("object is pending delete and stays hidden: " + name);

    int entries = 0;
    forEachEntry(ctx.scene, name, [&](Hittable& h) {
        h.visible = visible;
        ++entries;
    });
    if (entries == 0) return Result::fail("object not found: " + name);

    // The outliner reads its eye from its own representative facades; for a
    // flat mesh that facade is a throwaway handle NOT in world.objects, so the
    // sweep above never reaches it.
    ui.syncMeshCacheVisibility(name, visible);

    Backend::IViewportBackend* viewport = viewportBackend(ctx);
    Backend::IBackend* render = renderBackend(ctx);
    if (viewport) viewport->setVisibilityByNodeName(name, visible);
    if (render && render != static_cast<Backend::IBackend*>(viewport))
        render->setVisibilityByNodeName(name, visible);

    // ★★ setVisibilityByNodeName can only UNMASK an instance that exists.
    // Every backend rebuild (Vulkan updateGeometry, raster rebuild, OptiX
    // snapshot) SKIPS hidden objects entirely, so an object that was hidden
    // when anything rebuilt has no instance left, and "show" was a silent
    // no-op until some unrelated rebuild happened (field report 2026-10-25:
    // Plane_1 stayed absent from RT after the first eye toggle). On show,
    // detect the missing instance and schedule the rebuild that recreates it.
    if (visible) {
        bool vulkanMissing = false;
        bool rasterMissing = false;
        bool nonVulkanBackend = false;
        auto inspect = [&](Backend::IBackend* backend) {
            if (!backend) return;
            auto* vba = dynamic_cast<Backend::VulkanBackendAdapter*>(backend);
            if (!vba) { nonVulkanBackend = true; return; }
            const auto c = vba->countInstanceMasksByNodeName(name);
            if (c.rt_instances == 0 && c.rt_instances_all > 0) vulkanMissing = true;
            if (c.raster_instances == 0 && c.raster_instances_all > 0) rasterMissing = true;
        };
        inspect(viewport);
        if (render != static_cast<Backend::IBackend*>(viewport)) inspect(render);
        // OptiX keeps no read-back, and its rebuild snapshot drops hidden
        // objects the same way: a show has to assume the instance is gone.
        if (vulkanMissing || rasterMissing || nonVulkanBackend) {
            g_geometry_dirty = true;
            g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
            if (vulkanMissing || rasterMissing) {
                g_vulkan_rebuild_pending = true;
                g_viewport_raster_rebuild_pending = true;
            }
            if (nonVulkanBackend) g_optix_rebuild_pending = true;
            if (vulkanMissing || rasterMissing)
                SCENE_LOG_INFO("[Visibility] '" + name +
                               "' had no Vulkan instance (dropped by a rebuild while hidden); "
                               "geometry rebuild scheduled");
        }
    }

    ProjectManager::getInstance().markModified();
    ctx.renderer.resetCPUAccumulation();
    if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
    return Result::success();
}

Result getObjectVisibility(const std::string& name, ObjectVisibility& out) {
    if (!g_ctx) return notBound();
    UIContext& ctx = *g_ctx;
    out = {};
    out.name = name;
    out.pending_delete = ctx.scene.isEditorPendingDeleteObjectName(name);
    forEachEntry(ctx.scene, name, [&](Hittable& h) {
        ++out.entries;
        if (h.visible) ++out.visible_entries;
    });
    if (out.entries == 0) return Result::fail("object not found: " + name);
    out.visible = !out.pending_delete && out.visible_entries == out.entries;

    Backend::IViewportBackend* viewport = viewportBackend(ctx);
    Backend::IBackend* render = renderBackend(ctx);
    out.render_is_viewport = render && render == static_cast<Backend::IBackend*>(viewport);

    if (auto* vba = dynamic_cast<Backend::VulkanBackendAdapter*>(static_cast<Backend::IBackend*>(viewport))) {
        const auto c = vba->countInstanceMasksByNodeName(name);
        out.viewport_raster_instances = c.raster_instances;
        out.viewport_raster_visible = c.raster_visible;
        out.viewport_rt_instances = c.rt_instances;
        out.viewport_rt_visible = c.rt_visible;
    }
    if (out.render_is_viewport) {
        out.render_rt_instances = out.viewport_rt_instances;
        out.render_rt_visible = out.viewport_rt_visible;
    } else if (auto* vba = dynamic_cast<Backend::VulkanBackendAdapter*>(render)) {
        const auto c = vba->countInstanceMasksByNodeName(name);
        out.render_rt_instances = c.rt_instances;
        out.render_rt_visible = c.rt_visible;
    }
    return Result::success();
}

} // namespace rtapi
