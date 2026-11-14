#include "Api/RtApiVdbVolume.h"

#include "Api/RtApiInternal.h"
#include "Core/RuntimeProjectHost.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"
#include "ProjectManager.h"
#include "Renderer.h"
#include "VDBVolume.h"
#include "VolumeShader.h"
#include "globals.h"
#include "scene_ui.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>

extern std::unique_ptr<Backend::IBackend> g_backend;
extern bool g_viewport_raster_rebuild_pending;

namespace rtapi {

bool shouldApplySpecialVdbOrientation(const std::string& source_hint) {
    std::string lowered = source_hint;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered.find("embergen") != std::string::npos;
}

void applyVdbImportOrientation(VDBVolume& vdb, int orientation_preset,
                               const std::string& source_hint) {
    bool apply_axis_conversion = false;
    if (orientation_preset == 2) {
        apply_axis_conversion = true;
    } else if (orientation_preset == 0) {
        apply_axis_conversion = shouldApplySpecialVdbOrientation(source_hint);
    }
    if (apply_axis_conversion) {
        vdb.setRotation(Vec3(-90.0f, 0.0f, 0.0f));
    } else if (orientation_preset == 1) {
        vdb.setRotation(Vec3(0.0f, 0.0f, 0.0f));
    }
}

namespace {

// The render backend that owns the volume TLAS entry: Studio's separate render
// backend when one exists, otherwise a non-viewport context backend. Same rule
// as SceneUI's getVdbRenderBackend.
Backend::IBackend* vdbRenderBackend() {
    if (g_backend) {
        return g_backend.get();
    }
    if (g_ctx->backend_ptr && dynamic_cast<Backend::IViewportBackend*>(g_ctx->backend_ptr) == nullptr) {
        return g_ctx->backend_ptr;
    }
    return nullptr;
}

} // namespace

Result importVdbVolume(const std::string& path, int orientation_preset, VdbImportInfo& out) {
    out = VdbImportInfo{};
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (path.empty()) return Result::fail("path is empty");
    if (orientation_preset < 0 || orientation_preset > 2) {
        return Result::fail("orientation must be 0 (auto), 1 (keep) or 2 (Z-up to Y-up)");
    }
    std::error_code exists_error;
    if (!std::filesystem::exists(AppRuntime::projectPathFromUtf8(path), exists_error)) {
        return Result::fail("VDB file not found: " + path);
    }

    auto vdb = std::make_shared<VDBVolume>();
    if (!vdb->loadVDB(path)) {
        return Result::fail("failed to load VDB: " + path);
    }

    // Auto-scale huge VDBs (likely centimetres instead of metres).
    const Vec3 bmin = vdb->getLocalBoundsMin();
    const Vec3 bmax = vdb->getLocalBoundsMax();
    const float max_dim = (std::max)(bmax.x - bmin.x, (std::max)(bmax.y - bmin.y, bmax.z - bmin.z));
    if (max_dim > 50.0f) {
        const float target_size = 5.0f;
        vdb->setScale(Vec3(target_size / max_dim));
        out.auto_scaled = true;
        SCENE_LOG_INFO("[Import] Auto-scaled VDB from " + std::to_string(max_dim) + " to " +
                       std::to_string(target_size) + " units.");
    }

    // Centre on X/Z and sit the bottom of the volume on Y = 0.
    const Vec3 scale = vdb->getScale();
    const Vec3 center = (bmin + bmax) * 0.5f;
    vdb->setPosition(Vec3(-center.x * scale.x, -bmin.y * scale.y, -center.z * scale.z));

    out.gpu_uploaded = vdb->uploadToGPU();
    if (!out.gpu_uploaded) {
        SCENE_LOG_WARN("VDB uploaded to CPU only (GPU upload failed)");
    }
    out.fire_shader = vdb->hasGrid("temperature");
    vdb->setShader(out.fire_shader ? VolumeShader::createFirePreset()
                                   : VolumeShader::createSmokePreset());
    applyVdbImportOrientation(*vdb, orientation_preset, path);

    g_ctx->scene.addVDBVolume(vdb);
    // Also a Hittable for CPU ray intersection.
    g_ctx->scene.world.objects.push_back(vdb);
    g_ctx->renderer.rebuildBVH(g_ctx->scene, g_ctx->render_settings.UI_use_embree);
    g_ctx->renderer.resetCPUAccumulation();

    Backend::IBackend* render_backend = vdbRenderBackend();
    if (dynamic_cast<Backend::VulkanBackendAdapter*>(render_backend) != nullptr) {
        // Vulkan RT must rebuild its TLAS to add the new AABB instance.
        g_viewport_raster_rebuild_pending = true;
        g_vulkan_rebuild_pending = true;
    } else {
        // Raster viewport (and OptiX): publish the volume table now.
        SceneUI::syncVDBVolumesToGPU(*g_ctx);
        if (render_backend) {
            render_backend->resetAccumulation();
        }
        g_viewport_raster_rebuild_pending = true;
    }
    ProjectManager::getInstance().markModified();
    ui.mesh_cache_valid = false;

    out.name = vdb->name;
    out.local_bounds_min = bmin;
    out.local_bounds_max = bmax;
    out.position = vdb->getPosition();
    out.scale = vdb->getScale();
    out.scene_vdb_count = static_cast<int>(g_ctx->scene.vdb_volumes.size());
    SCENE_LOG_INFO("Imported VDB Volume: " + path);
    return Result::success();
}

} // namespace rtapi
