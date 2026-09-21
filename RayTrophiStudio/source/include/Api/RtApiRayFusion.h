#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "RayFusion/ProbeBounce.h"
#include "RayFusion/ProbeField.h"

namespace rtapi {
struct RayFusionCoreInfo {
    std::string stage = "probe_control_plane";
    bool core_available = true;
    bool renderer_available = false;
    bool gi_active = false;
    std::string inactive_reason = "probe field status has not been queried";
    std::string quality;
    uint32_t probe_abi_version = 0;
    uint32_t directional_texels_per_probe = 0;
    uint32_t payload_bytes_per_probe = 0;
    uint32_t planned_rays_per_probe = 0;
    uint32_t planned_probes_per_update = 0;
    uint32_t planned_rays_per_update = 0;
};
// Probe field residency; gi_active additionally requires material shading and
// a published traced field. This does not promise visible surface coverage.
struct RayFusionProbeFieldInfo {
    bool overlay_requested = false, overlay_ready = false, follow_camera = false;
    uint32_t overlay_markers = 0;
    std::string overlay_reason;
    RayFusion::BounceStatus bounce;
    bool supported = false;
    bool configured = false;
    bool uploaded = false;
    bool bound = false;
    // What actually RAN: "sky_bake" (1a), "traced" (1b), or "none". Never the
    // request -- an instrument that echoes the request cannot show a refusal.
    std::string producer = "none";
    bool producer_traced_requested = false;
    std::string producer_reason;
    std::string budget_preset;
    uint64_t producer_signature = 0;
    float hit_fraction = 0.0f;
    uint32_t rejected_inside = 0;
    bool auto_fit = false;
    std::string auto_fit_mode = "off";
    std::string auto_fit_reason;
    // Cok arka yuz gordu AMA mesafe kapisi katinin icinde OLMADIGINI soyledi.
    // Kapali bir odada bu sayi buyuk olmali; sifirsa kapi hic devreye girmiyor.
    uint32_t backface_enclosed = 0;
    float mean_hit_distance = 0.0f;   // dunya birimi, yonler uzerinden ortalama
    double trace_ms = 0.0;
    uint64_t traced_publishes = 0;
    uint32_t total = 0, valid = 0, pending = 0, in_flight = 0;
    uint64_t accepted = 0, rejected = 0;
    // The APPLIED window, read back from the field. counts/minimum are cells;
    // spacing is world units; max_slots is the ceiling on counts.x*y*z.
    uint32_t counts[3]{};
    int32_t minimum[3]{};
    double minimum_world[3]{};
    float spacing = 0.0f;
    uint32_t max_slots = 0;
};

// Acceleration structure residency in the raster viewport. This is the bill
// RayFusion pays every frame the scene changes, whether or not a ray is traced,
// so it is reported in bytes and milliseconds -- not as a feature flag.
struct RayFusionSceneASInfo {
    bool hardware_rt = false;   // false = every ray-based step is blocked here
    bool ready = false;
    uint32_t blas_count = 0;
    uint32_t instance_count = 0;
    uint32_t instances_skipped = 0;
    uint32_t meshes_skipped = 0;
    uint64_t as_bytes = 0;
    double last_build_ms = 0.0;
    uint64_t builds = 0;
    uint64_t built_geometry_generation = 0;
    // What the gate actually watches. The global generation is reported next to
    // it but is NOT the gate: scene.delete leaves that counter unchanged.
    uint64_t geometry_signature = 0;
    uint64_t instance_signature = 0;
    double signature_ms = 0.0;
    uint64_t tlas_only_refreshes = 0;
    uint32_t instances_hidden = 0;
    // Of blas_count, how many were built from a WELDED (indexed) raster mesh
    // and how many from genuine flat SoA. They must sum to blas_count. A scene
    // whose large static meshes are welded (terrain, imported props) reporting
    // blas_indexed == 0 means the triangulation was dropped and the traced
    // world is fabricated from vertex storage order.
    uint32_t blas_indexed = 0;
    uint32_t blas_flat = 0;
    // Of blas_count, how many were built from a GPU-SKINNED mesh and are
    // therefore re-fit as the character deforms. GPU skinning rewrites the
    // CONTENTS of the vertex buffer the BLAS borrows while its handle, its
    // device address and its vertex count all stay put -- so an AS with every
    // other counter green can still be describing the pose of the frame it was
    // built in, and the deforming mesh casts a frozen shadow.
    //
    // blas_skinned says the refit is POSSIBLE; skin_refits says it is
    // HAPPENING. In an animated scene skin_refits must climb while the timeline
    // plays and hold still while it is paused. blas_skinned > 0 with
    // skin_refits stuck at 0 is the stale-shadow bug back again.
    uint32_t blas_skinned = 0;
    uint64_t skin_refits = 0;
    uint64_t skin_refit_failures = 0;
    // Total plus its three parts. Act on the PARTS -- drain (the GPU still owed
    // us a frame), blas (one submit + fence around the refits), tlas (a full
    // top-level rebuild, which drains again). Each has a different fix.
    double last_skin_refit_ms = 0.0;
    double last_skin_drain_ms = 0.0;
    double last_skin_blas_ms = 0.0;
    double last_skin_tlas_ms = 0.0;
    // The AS was released while the viewport shows Rendered so the render
    // backend gets the VRAM; yields counts those releases.
    bool yielded = false;
    uint64_t yields = 0;
    // Driver-reported device-local VRAM of the whole process (every VkDevice).
    // vram_measured=false: VK_EXT_memory_budget missing, zeros are not a reading.
    bool vram_measured = false;
    uint64_t vram_usage_bytes = 0;
    uint64_t vram_budget_bytes = 0;
    std::string inactive_reason;
};

struct RayFusionCoreCheck { std::string name; bool passed = false; std::string detail; };
struct RayFusionCoreValidation {
    bool passed = false;
    bool gpu_tested = false;
    std::vector<RayFusionCoreCheck> checks;
};
// Producer lever: traced (step 1b) or the sky bake (1a), so the two can be
// compared on the SAME scene. Returns false when no viewport backend took it.
bool setRayFusionProbeProducer(bool traced);
bool setRayFusionProbeBounce(bool enabled);
bool setRayFusionProbeOverlay(bool enabled);
bool setRayFusionProbeFollowCamera(bool enabled);
// Move or reshape the probe window at runtime. Partial: unsent fields keep what
// the field has. Fail-closed -- false leaves the window untouched and fills
// @p error. Read the APPLIED window back from rayFusionProbeFieldStatus().
bool setRayFusionProbeGrid(const RayFusion::GridRequest& request, std::string& error);
RayFusionCoreInfo rayFusionCoreStatus();
RayFusionProbeFieldInfo rayFusionProbeFieldStatus();
RayFusionSceneASInfo rayFusionSceneASStatus();
RayFusionCoreValidation validateRayFusionCore();
} // namespace rtapi
