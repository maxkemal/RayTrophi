/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          RenderStateManager.cpp
 * Description:   Phase 1 implementation — owns viewport mode authoritatively
 *                and bridges dirty scopes to the existing g_* globals while
 *                later phases migrate callers off the globals entirely.
 * =========================================================================
 */
#include "Core/RenderStateManager.h"
#include "globals.h"

namespace Core {

RenderStateManager& RenderStateManager::instance() {
    static RenderStateManager s;
    return s;
}

RenderStateManager::RenderStateManager() = default;

// ---------------------------------------------------------------------------
// Viewport mode — passive authoritative value.
// Backends push their own mode here; this class never notifies anyone.
// ---------------------------------------------------------------------------
void RenderStateManager::setViewportMode(ViewportMode mode) {
    m_viewportMode.store(mode, std::memory_order_release);
}

RenderStateManager::ViewportMode RenderStateManager::viewportMode() const {
    return m_viewportMode.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Dirty flags — Phase 1 bridge to legacy globals.
// Later phases will store state inside RenderStateManager directly and
// deprecate these extern bools.
// ---------------------------------------------------------------------------
void RenderStateManager::markDirty(DirtyScope scope) {
    switch (scope) {
        case DirtyScope::Geometry:
            g_geometry_dirty = true;
            g_bvh_rebuild_pending = true;
            g_vulkan_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            g_viewport_raster_rebuild_pending = true;
            g_scene_geometry_generation.fetch_add(1, std::memory_order_acq_rel);
            break;
        case DirtyScope::Materials:
            g_materials_dirty = true;
            break;
        case DirtyScope::Textures:
            g_materials_dirty = true;
            break;
        case DirtyScope::Lights:
            g_lights_dirty = true;
            break;
        case DirtyScope::Camera:
            g_camera_dirty = true;
            break;
        case DirtyScope::World:
            g_world_dirty = true;
            break;
        case DirtyScope::Transforms:
            g_gpu_refit_pending = true;
            g_cpu_bvh_refit_pending = true;
            break;
        case DirtyScope::PaintLayer:
            g_geometry_dirty = true;
            g_bvh_rebuild_pending = true;
            break;
        case DirtyScope::GasVolumes:
            g_gas_volumes_dirty = true;
            break;
        case DirtyScope::Count:
            break;
    }
}

bool RenderStateManager::isDirty(DirtyScope scope) const {
    switch (scope) {
        case DirtyScope::Geometry:   return g_geometry_dirty;
        case DirtyScope::Materials:  return g_materials_dirty;
        case DirtyScope::Textures:   return g_materials_dirty;
        case DirtyScope::Lights:     return g_lights_dirty;
        case DirtyScope::Camera:     return g_camera_dirty;
        case DirtyScope::World:      return g_world_dirty;
        case DirtyScope::Transforms: return g_gpu_refit_pending || g_cpu_bvh_refit_pending;
        case DirtyScope::PaintLayer: return g_geometry_dirty;
        case DirtyScope::GasVolumes: return g_gas_volumes_dirty;
        case DirtyScope::Count:      return false;
    }
    return false;
}

void RenderStateManager::clearDirty(DirtyScope scope) {
    switch (scope) {
        case DirtyScope::Geometry:   g_geometry_dirty = false; break;
        case DirtyScope::Materials:  g_materials_dirty = false; break;
        case DirtyScope::Textures:   g_materials_dirty = false; break;
        case DirtyScope::Lights:     g_lights_dirty = false; break;
        case DirtyScope::Camera:     g_camera_dirty = false; break;
        case DirtyScope::World:      g_world_dirty = false; break;
        case DirtyScope::Transforms:
            g_gpu_refit_pending = false;
            g_cpu_bvh_refit_pending = false;
            break;
        case DirtyScope::PaintLayer: g_geometry_dirty = false; break;
        case DirtyScope::GasVolumes: g_gas_volumes_dirty = false; break;
        case DirtyScope::Count:      break;
    }
}

uint64_t RenderStateManager::geometryGeneration() const {
    return g_scene_geometry_generation.load(std::memory_order_acquire);
}

} // namespace Core
