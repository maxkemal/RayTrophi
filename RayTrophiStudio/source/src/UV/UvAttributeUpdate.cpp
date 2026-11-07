// SPDX-License-Identifier: MIT
#include "UV/UvAttributeUpdate.h"
#include "scene_ui.h"
#include "Backend/IViewportBackend.h"
#include "PerfProfile.h"

#include <memory>

extern std::unique_ptr<Backend::IBackend> g_backend;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern bool g_vulkan_rebuild_pending;
extern bool g_optix_rebuild_pending;
extern bool g_viewport_raster_rebuild_pending;

namespace UV {

void publishAttributeUpdate(UIContext& ctx, const TriangleMesh* mesh) {
    RTPERF_SCOPE("uv.attribute_publish");
    auto* render = g_backend ? g_backend.get() : ctx.backend_ptr;
    auto* viewport = g_viewport_backend ? g_viewport_backend.get() :
        dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
    if (render && render != viewport) {
        const auto type = render->getInfo().type;
        if ((type == Backend::BackendType::VULKAN_RT ||
             type == Backend::BackendType::VULKAN_COMPUTE ||
             type == Backend::BackendType::OPTIX) && !render->updateFlatMeshUVs(mesh)) {
            // Legacy/merged/weld-conflicting layouts require their normal rebuild once.
            // Do not invalidate the successful viewport or bump geometry generation.
            RTPERF_SCOPE("uv.attribute_rebuild_fallback");
            if (type == Backend::BackendType::OPTIX) {
                g_optix_rebuild_pending = true;
            } else {
                g_vulkan_rebuild_pending = true;
            }
        }
        render->resetAccumulation();
    }
    if (viewport) {
        if (!viewport->updateFlatMeshUVs(mesh)) {
            RTPERF_SCOPE("uv.attribute_rebuild_fallback");
            g_viewport_raster_rebuild_pending = true;
        }
        viewport->resetAccumulation();
    }
    // CPU Embree resolves UVs from the live flat mesh at hit time; no BVH update.
    ctx.renderer.resetCPUAccumulation();
    ctx.start_render = true;
}

} // namespace UV
