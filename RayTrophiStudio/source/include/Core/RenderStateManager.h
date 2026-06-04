/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          RenderStateManager.h
 * Description:   Centralized render state (viewport mode, dirty flags,
 *                rebuild coordination). Replaces the scattered g_* globals
 *                and per-backend m_viewportMode copies.
 * =========================================================================
 */
#ifndef RENDER_STATE_MANAGER_H
#define RENDER_STATE_MANAGER_H

#include "Backend/IBackend.h"

#include <atomic>
#include <cstdint>

namespace Core {

// ----------------------------------------------------------------------------
// Scopes identify which subsystem has become stale. One flag per scope lets
// flushToBackends() triage rebuilds without every call site knowing which
// backend is active.
// ----------------------------------------------------------------------------
enum class DirtyScope : uint8_t {
    Geometry = 0,   // mesh/triangle topology — triggers BVH + GPU blas rebuild
    Materials,      // material properties / textures bound to surfaces
    Textures,       // texture content only (no material param change)
    Lights,         // scene lights
    Camera,         // camera transform / parameters
    World,          // sky / environment / world shader
    Transforms,     // instance TRS without topology change (refit path)
    PaintLayer,     // paint-layer stack updated (mesh paint / sculpt)
    GasVolumes,     // VDB / gas volume data
    Count
};

class RenderStateManager {
public:
    using ViewportMode = Backend::ViewportMode;

    static RenderStateManager& instance();

    // ---- Viewport mode (passive authoritative value) ----------------------
    // Backends push their current mode here from their own setViewportMode.
    // This class stores the value but never notifies anyone — side effects
    // and cached mirrors live in the backends, matching the pre-RSM design.
    void setViewportMode(ViewportMode mode);
    ViewportMode viewportMode() const;

    // ---- Dirty flags ------------------------------------------------------
    // markDirty() is the single entry point. Internally it maps each scope
    // to the legacy g_* globals (Phase 1 compat) and increments the scene
    // geometry generation counter when topology actually changes.
    void markDirty(DirtyScope scope);
    bool isDirty(DirtyScope scope) const;
    void clearDirty(DirtyScope scope);

    // Scene geometry monotonic counter. Backends compare against their last
    // seen value to skip redundant rebuilds on mode switches.
    uint64_t geometryGeneration() const;

private:
    RenderStateManager();
    RenderStateManager(const RenderStateManager&) = delete;
    RenderStateManager& operator=(const RenderStateManager&) = delete;

    std::atomic<ViewportMode> m_viewportMode{ViewportMode::Rendered};
};

} // namespace Core

#endif // RENDER_STATE_MANAGER_H
