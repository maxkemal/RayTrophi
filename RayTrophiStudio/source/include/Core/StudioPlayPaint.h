#pragma once

#include "Backend/VulkanBackend.h"
#include "Game/GameRuntime.h"
#include "Paint/WetPaintGpu.h"
#include "globals.h"
#include "scene_data.h"

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;

namespace AppRuntime {

inline bool studioPlayPaintActive() {
    return Game::playtestActive() && !Paint::wetPaintGpu().empty();
}

inline bool usesPaintDevice(Backend::IBackend* backend, void* paint_device) {
    auto* vulkan = dynamic_cast<Backend::VulkanBackendAdapter*>(backend);
    const auto* device = vulkan ? vulkan->getVulkanDevice() : nullptr;
    return paint_device && device && device->getDevice() == paint_device;
}

inline bool bindStudioGamePaintDevice(SceneData& scene) {
    auto* vulkan = dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get());
    auto* device = vulkan ? vulkan->getVulkanDevice() : nullptr;
    if (!device || !device->isInitialized() || Paint::wetPaintGpu().publicationDevice()) {
        return false;
    }
    auto& compute = scene.simulation_world.compute();
    const auto previous = g_vulkan_sim_compute_ctx;
    device->publishAsSimComputeContext();
    const auto target = g_vulkan_sim_compute_ctx;
    g_vulkan_sim_compute_ctx = previous;
    if (compute.nativeDevice() != target.device) {
        auto backend = RayTrophiSim::createVulkanSimulationComputeBackend(target);
        if (!backend) {
            return false;
        }
        compute.synchronize();
        if (usesPaintDevice(g_backend.get(), compute.nativeDevice())) {
            g_backend->waitForCompletion();
        }
        scene.invalidateSimulationDenseGpuAddresses();
        compute.setBackend(std::move(backend));
    }
    if (previous.device != target.device) {
        RayTrophiSim::releaseSharedMeshComputeBackend();
    }
    g_vulkan_sim_compute_ctx = target;
    g_hasVulkanComputeSim = true;
    return true;
}

// The dedicated raster adapter cannot present an offline render from a
// separate device. During Play, use material preview if it owns the paint.
inline bool usePaintRasterShading(int& shading, Backend::IBackend* render,
                                  Backend::IViewportBackend* raster, void* paint_device) {
    if (!studioPlayPaintActive() || shading != 2 || usesPaintDevice(render, paint_device) ||
        !usesPaintDevice(raster, paint_device)) {
        return false;
    }
    shading = 1;
    Game::playtest().restore_rendered_shading = true;
    return true;
}

inline int publishStudioPaint(Backend::IBackend* render, Backend::IBackend* raster,
                              Backend::IBackend* active) {
    auto& paint = Paint::wetPaintGpu();
    if (!studioPlayPaintActive()) {
        return paint.publish(render, raster);
    }
    if (!usesPaintDevice(active, paint.publicationDevice())) {
        return 0;
    }
    // Keep the other backend's dirty rectangles for its next edit-mode update.
    return active == render ? paint.publish(active, nullptr, false)
                            : paint.publish(nullptr, active, false);
}

} // namespace AppRuntime
