// Shared runtime state (Studio + Player).
//
// Definitions moved verbatim out of Main.cpp and the Studio UI sources so the
// Player links the same render/scene dirty flags, deferred-rebuild flags, active
// backends and loader state that shared code (rtapi, ProjectManager, Vulkan
// backend, simulation) reads and writes. Declarations stay where they were
// (globals.h, scene_ui.h, scene_data.h and the call sites' own externs); only
// the single definition moved. Do not add a second definition elsewhere.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md (Faz 3b).

#include "globals.h"
#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "ColorProcessingParams.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct AnimationData;

// Request a display rebuild + stylize re-apply WITHOUT re-rendering and WITHOUT forcing
// tonemap on — honors the current persistent_tonemap setting. Set by Stylize param changes.
bool stylize_redisplay = false;

// ===========================================================================
// DIRTY FLAGS - Prevents unnecessary GPU buffer updates when data unchanged
// ===========================================================================
bool g_camera_dirty = true;
bool g_lights_dirty = true;
// Two consumers, two flags (rationale in globals.h): a single flag was eaten by
// whichever consumer ran first and the other adapter (separate VkDevice) never
// received the world. Producers call markWorldDirty(); new code that sets the
// flags one by one leaves a consumer starving again.
bool g_world_dirty = true;
bool g_viewport_world_dirty = true;
void markWorldDirty() {
    g_world_dirty = true;
    g_viewport_world_dirty = true;
}
bool g_original_surface_needs_sync = true;

// Granular scene-sync dirty flags
bool g_geometry_dirty = true;        // Mesh/triangle data changed
bool g_materials_dirty = true;       // Material properties changed
bool g_gas_volumes_dirty = true;     // Gas/VDB volumes changed
bool g_texture_pool_dirty = false;   // Evict SceneTextureManager pool inside uploadMaterials (after waitIdle)

// Scene geometry generation counter
std::atomic<uint64_t> g_scene_geometry_generation{1};

// ===========================================================================
// DEFERRED REBUILD FLAGS - For optimized batched rebuilds
// Set these instead of calling rebuild immediately, Main loop handles them
// ===========================================================================
bool g_bvh_rebuild_pending = false;      // CPU BVH needs rebuild
bool g_gpu_refit_pending = false;        // GPU Geometry needs update (Deferred)
// A transform refit consumed by the Solid/Matcap raster backend still has to be
// applied to the inactive RT backend when Rendered mode resumes. Keeping this
// separate avoids either losing the RT update or refitting raster every idle frame.
bool g_render_backend_refit_pending = false;
bool g_vulkan_rebuild_pending = false;    // GPU Vulkan geometry needs rebuild
bool g_vulkan_device_cc_refit_pending = false; // Slice 4: device-resident CC position-only edit - MODE_UPDATE-refit the BLAS instead of a full rebuild
bool g_vulkan_geometry_append_pending = false; // Additive-only mutation (scatter etc.) - try incremental TLAS refit first
bool g_geometry_deform_pending = false; // Physics body baked new verts - refit only those nodes in place on the active path (raster/OptiX/Vulkan) + CPU Embree refit (vs full rebuild)
bool g_viewport_raster_rebuild_pending = false; // Interactive raster viewport needs rebuild
bool g_viewport_raster_cache_invalid = false;   // Raster-only representation change; see globals.h
bool g_viewport_raster_scatter_update_pending = false; // Foliage-only instance-table refresh
bool g_vulkan_scatter_refresh_pending = false; // InstanceGroup topology only; preserve BLASes
bool g_optix_rebuild_pending = false;
bool g_optix_scatter_refresh_pending = false;
bool g_particle_cpu_geometry_dirty = false; // Particle bridge changed; CPU-active path re-expands particles into the BVH

// -- Viewport-driven sequence save -------------------------------------------
// Renders an animation to disk through the SAME fast interactive viewport path
// (scrub each frame -> converge to the interactive quality -> save) instead of the
// separate render_Animation worker. Driven by a state machine in the main loop.
bool        g_seq_save_active = false;
int         g_seq_save_frame = 0;
int         g_seq_save_end = 0;
std::string g_seq_save_dir;
bool        g_seq_save_denoise = false;

// True while the active viewport is an interactive raster shading mode, i.e.
// NOT Rendered. All raster modes share the native NanoVDB SurfaceSDF buffer.
bool g_solid_viewport_active = false;
bool g_native_surface_sdf_viewport_available = false;
bool g_material_preview_viewport_active = false;
bool g_dense_gas_host_mirror_needed = false;
bool g_sim_timeline_mode = true;  // true = timeline-driven (bake/scrub, idle when stopped); false = live free-run preview
bool g_sim_use_gpu_solver = false; // GPU simulation compute (Vulkan, or CUDA where present): grid solver + APIC fluid + NanoVDB bridge
bool g_gpu_subdivide_enabled = true; // GPU linear mesh subdivision (CPU fallback when unavailable / small mesh)
RayTrophiSim::SimulationComputeVulkanContext g_vulkan_sim_compute_ctx{};
std::atomic<bool> g_optix_rebuild_in_progress{false}; // True while TLAS rebuild is happening
std::atomic<bool> g_viewport_rebuild_in_progress{false}; // True while viewport backend resources are being torn down/rebuilt
std::atomic<bool> g_anim_backend_needs_full_rebuild{false}; // See globals.h - set by aborted-anim cleanup and backend-swap paths
bool g_mesh_cache_dirty = false;         // UI mesh cache needs rebuild
bool g_cpu_sync_pending = false;         // CPU data needs sync after TLAS mode changes
bool g_cpu_bvh_refit_pending = false;    // CPU BVH fast refit (Embree only)
bool g_cpu_bvh_stale = false;            // Lazy CPU BVH refit (see globals.h)
int g_bvh_rebuild_deferred_frames = 0;
uint64_t g_cpu_bvh_requested_generation = 0;
uint64_t g_cpu_bvh_future_generation = 0;
bool g_viewport_hovered = false;         // True when mouse is over the main RenderView
int g_backend_switch_cooldown_frames = 0; // Skip a few GPU animation/render ticks right after backend switch

// ===========================================================================
// SCENE LOADING FLAGS - Thread safety for project load/save operations
// ===========================================================================
std::atomic<bool> g_scene_loading_in_progress{false};  // Prevents concurrent load operations
std::atomic<bool> g_needs_geometry_rebuild{false};   // Set by loader thread, main loop does actual rebuild
std::atomic<bool> g_needs_optix_sync{false};         // Set by loader thread, main loop syncs backend buffers
std::atomic<bool> g_deferred_render_backend_prepare_pending{false};
int g_deferred_render_backend_prepare_delay_frames = 0;

// Active render backend and the dedicated interactive viewport backend. Both
// shells reset them explicitly during shutdown, before process exit.
std::unique_ptr<Backend::IBackend> g_backend;
std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
ColorProcessor color_processor(image_width, image_height);
std::string active_model_path;

// Animation-graph clip list pointer (set by the Studio anim-graph panel; read by
// AnimationNodes). Null in a shell without that panel.
std::vector<std::shared_ptr<AnimationData>>* g_uiClipsRef = nullptr;

// Scene-load raster guard (rationale in scene_ui.h, next to its declaration).
bool g_scene_load_solid_guard = true;

// Dense subdivide output is ONE TriangleMesh hittable instead of a facade soup
// (rationale where the flag is read in scene_ui_menu.hpp).
bool g_dense_mesh_as_hittable = true;

// System RAM query for the "RAM sim cache is large, bake to disk" nudge and the
// sim frame-cache budget (declared in scene_data.h / scene_ui_forcefield.hpp).
std::uint64_t queryAvailablePhysicalRamBytes() {
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return static_cast<std::uint64_t>(ms.ullAvailPhys);
    return 0;
}

std::size_t simFrameCacheBudgetFromHardware() {
    // Total RAM is constant for the process; query once.
    static const std::size_t budget = [] {
        constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;
        MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
        if (!GlobalMemoryStatusEx(&ms) || ms.ullTotalPhys == 0) return std::size_t(4) * kGiB;
        const std::uint64_t total = ms.ullTotalPhys;
        const std::uint64_t by_fraction = total / 10ull * 7ull;
        const std::uint64_t by_headroom = total > 2ull * kGiB ? total - 2ull * kGiB : 0ull;
        return static_cast<std::size_t>((std::max)(kGiB, (std::min)(by_fraction, by_headroom)));
    }();
    return budget;
}
