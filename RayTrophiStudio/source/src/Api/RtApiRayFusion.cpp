#include "Api/RtApi.h"
#include "Backend/IBackend.h"
#include "RayFusion/ProbeField.h"
#include <exception>
#include <functional>

namespace rtapi {
void forEachViewportBackend(const std::function<void(Backend::IBackend&)>& fn);

// ★★★★★ RAYFUSION SETTER'LARI VIEWPORT'U BACKEND ICINDE DIRTY EDER, burada
//   DEGIL. Bu dosyada bir kez `g_ctx->start_render = true` yazan bir yardimci
//   vardi ve OLCULDU ki hicbir sey yapmiyor: o bayrak yol izleyiciyi surer,
//   raster viewport'un kapisi ise `needsViewportRender()` ve orasi yalnizca
//   `m_interactiveViewport.dirty` / `pump` bakar.
//
// ★★★★ Belirtisi bir no-op'tan beterdi: `set_probe_bounce` "applied: True,
//   requested: True" donuyor, `traced_publishes` 1201'de duruyor ve
//   `bounce_active` False kaliyordu. Yani duzeltme UYGULANMIS GORUNUYOR,
//   olcum ise eski kareyi okuyordu.
//
// ★★★ Yanlis kol BIRAKILMADI, SOKULDU. Iki farkli "yeniden ciz" kavramini
//   ayni isimle yan yana yasatmak, bu depoda tekrar tekrar sessiz arizaya
//   dondu: dogru olani eklenince yanlis olani zararsiz gorunur ve bir
//   sonraki setter yine ona baglanir.

RayFusionProbeFieldInfo rayFusionProbeFieldStatus() {
    RayFusionProbeFieldInfo out;
    forEachViewportBackend([&](Backend::IBackend& backend) {
        Backend::RayFusionProbeStatus probe{};
        if (!backend.getRayFusionProbeStatus(probe)) return;
        // First backend that actually owns a field wins. Merging two fields
        // would report a grid that neither of them has.
        if (out.configured || !probe.supported) return;
        out.supported = probe.supported;
        out.overlay_requested = probe.overlay_requested;
        out.overlay_ready = probe.overlay_ready;
        out.overlay_markers = probe.overlay_markers;
        out.overlay_reason = probe.overlay_reason;
        out.follow_camera = probe.follow_camera;
        out.bounce = probe.bounce;
        out.configured = probe.configured;
        out.uploaded = probe.uploaded;
        out.bound = probe.bound;
        out.producer = probe.producer;
        out.producer_traced_requested = probe.producer_traced_requested;
        out.producer_reason = probe.producer_reason;
        out.hit_fraction = probe.hit_fraction;
        out.rejected_inside = probe.rejected_inside;
        out.auto_fit = probe.auto_fit;
        out.auto_fit_mode = probe.auto_fit_mode;
        out.auto_fit_reason = probe.auto_fit_reason;
        out.backface_enclosed = probe.backface_enclosed;
        out.mean_hit_distance = probe.mean_hit_distance;
        out.trace_ms = probe.trace_ms;
        out.traced_publishes = probe.traced_publishes;
        out.budget_preset = probe.budget_preset;
        out.producer_signature = probe.producer_signature;
        out.total = probe.total;
        out.valid = probe.valid;
        out.pending = probe.pending;
        out.in_flight = probe.in_flight;
        out.accepted = probe.accepted;
        out.rejected = probe.rejected;
        for (int i = 0; i < 3; ++i) {
            out.counts[i] = probe.counts[i];
            out.minimum[i] = probe.minimum[i];
            out.minimum_world[i] = double(probe.minimum[i]) * double(probe.spacing);
        }
        out.spacing = probe.spacing;
        out.max_slots = probe.max_slots;
    });
    return out;
}

RayFusionSceneASInfo rayFusionSceneASStatus() {
    RayFusionSceneASInfo out;
    bool found = false;
    forEachViewportBackend([&](Backend::IBackend& backend) {
        Backend::RayFusionSceneASStatus as{};
        const bool answered = backend.getRayFusionSceneASStatus(as);
        // hardware_rt is a device fact, so take it from any backend that knows.
        // A backend that has never been asked to build reports the capability
        // but not a grid -- and that distinction is the whole point of the UI
        // gate: "cannot" and "has not yet" are different sentences.
        if (as.hardware_rt) out.hardware_rt = true;
        // Yield state and the VRAM reading are valid without a built AS --
        // "released" is exactly the state in which nothing is built.
        if (as.yielded) out.yielded = true;
        out.yields += as.yields;
        if (as.vram_measured && !out.vram_measured) {
            out.vram_measured = true;
            out.vram_usage_bytes = as.vram_usage_bytes;
            out.vram_budget_bytes = as.vram_budget_bytes;
        }
        if (!answered || found) return;
        found = true;
        out.ready = as.ready;
        out.blas_count = as.blas_count;
        out.instance_count = as.instance_count;
        out.instances_skipped = as.instances_skipped;
        out.meshes_skipped = as.meshes_skipped;
        out.as_bytes = as.as_bytes;
        out.last_build_ms = as.last_build_ms;
        out.builds = as.builds;
        out.built_geometry_generation = as.built_geometry_generation;
        out.geometry_signature = as.geometry_signature;
        out.instance_signature = as.instance_signature;
        out.signature_ms = as.signature_ms;
        out.tlas_only_refreshes = as.tlas_only_refreshes;
        out.instances_hidden = as.instances_hidden;
        out.blas_indexed = as.blas_indexed;
        out.blas_flat = as.blas_flat;
        out.blas_skinned = as.blas_skinned;
        out.skin_refits = as.skin_refits;
        out.skin_refit_failures = as.skin_refit_failures;
        out.last_skin_refit_ms = as.last_skin_refit_ms;
        out.last_skin_drain_ms = as.last_skin_drain_ms;
        out.last_skin_blas_ms = as.last_skin_blas_ms;
        out.last_skin_tlas_ms = as.last_skin_tlas_ms;
        out.inactive_reason = as.inactive_reason;
    });
    if (!found && out.inactive_reason.empty()) {
        out.inactive_reason = out.hardware_rt
            ? "no viewport backend has built a scene acceleration structure yet"
            : "this GPU or driver reports no hardware ray tracing support";
    }
    return out;
}

bool setRayFusionProbeProducer(bool traced) {
    bool applied = false;
    forEachViewportBackend([&](Backend::IBackend& backend) {
        Backend::RayFusionProbeStatus probe{};
        // Only a backend that actually owns a probe field can honour this. A
        // silent success on a backend with no field would report a producer
        // change that never happened.
        if (!backend.getRayFusionProbeStatus(probe) || !probe.supported) return;
        backend.setRayFusionProbeProducer(traced);
        applied = true;
    });
    return applied;
}

bool setRayFusionProbeBounce(bool enabled) {
    bool applied = false;
    forEachViewportBackend([&](Backend::IBackend& backend) {
        Backend::RayFusionProbeStatus probe{};
        if (backend.getRayFusionProbeStatus(probe) && probe.supported)
            applied = backend.setRayFusionProbeBounce(enabled) || applied;
    });
    return applied;
}

bool setRayFusionProbeOverlay(bool enabled) {
    bool applied = false;
    forEachViewportBackend([&](Backend::IBackend& backend) {
        Backend::RayFusionProbeStatus probe{};
        if (!backend.getRayFusionProbeStatus(probe) || !probe.supported) return;
        applied = backend.setRayFusionProbeOverlay(enabled) || applied;
    });
    return applied;
}

bool setRayFusionProbeFollowCamera(bool enabled) {
    bool applied = false;
    forEachViewportBackend([&](Backend::IBackend& backend) {
        Backend::RayFusionProbeStatus probe{};
        if (!backend.getRayFusionProbeStatus(probe) || !probe.supported) return;
        applied = backend.setRayFusionProbeFollowCamera(enabled) || applied;
    });
    return applied;
}

bool setRayFusionProbeGrid(const RayFusion::GridRequest& request, std::string& error) {
    bool applied = false;
    error.clear();
    forEachViewportBackend([&](Backend::IBackend& backend) {
        Backend::RayFusionProbeStatus probe{};
        if (!backend.getRayFusionProbeStatus(probe) || !probe.supported) return;
        std::string backendError;
        // Report the FIRST refusal rather than the last: a second backend that
        // simply has no field would otherwise overwrite the real reason with
        // "no viewport backend owns a probe field".
        if (backend.setRayFusionProbeGrid(request, backendError)) applied = true;
        else if (error.empty()) error = backendError;
    });
    if (!applied && error.empty()) error = "no viewport backend owns a probe field";
    return applied;
}

RayFusionCoreInfo rayFusionCoreStatus() {
    RayFusionCoreInfo out;
    const auto field = rayFusionProbeFieldStatus();
    const auto screen = screenGiStatus();
    const auto shading = viewportShading();
    const bool sceneLighting = viewportPreviewLighting().preset == "scene";
    out.stage = screen.settings.enabled ? "raster_screen_gi" : "raster_probe_gi";
    out.renderer_available = field.supported || screen.supported;
    out.gi_active = shading.mode == "material" && sceneLighting && field.configured &&
        field.uploaded && field.bound && field.valid > 0 && field.producer == "traced";
    out.gi_active = out.gi_active || (screen.ready && shading.mode == "material" && sceneLighting);
    if (out.gi_active) out.inactive_reason.clear();
    else if (!field.supported) out.inactive_reason = "no available probe field renderer";
    else if (shading.mode != "material") out.inactive_reason = "material viewport is not active";
    else if (!sceneLighting) out.inactive_reason = "scene lighting is not active";
    else if (!field.configured || !field.uploaded || !field.bound || field.valid == 0)
        out.inactive_reason = "no valid probe field is published and bound";
    else out.inactive_reason = "published producer is sky bake, not traced GI";
    out.quality = viewportQuality().preset;
    const auto budget = RayFusion::budgetForQuality(out.quality);
    out.probe_abi_version = RayFusion::kProbeAbiVersion;
    out.directional_texels_per_probe = RayFusion::kProbeTexels;
    out.payload_bytes_per_probe = static_cast<uint32_t>(sizeof(RayFusion::ProbePacket));
    out.planned_rays_per_probe = budget.raysPerProbe;
    out.planned_probes_per_update = budget.maxProbes;
    out.planned_rays_per_update = budget.maxRays;
    return out;
}
RayFusionCoreValidation validateRayFusionCore() {
    RayFusionCoreValidation out;
    try {
        const auto checks = RayFusion::validateProbeCore();
        out.passed = !checks.empty();
        for (const auto& check : checks) {
            out.checks.push_back({check.name, check.passed, check.detail});
            out.passed = out.passed && check.passed;
        }
    } catch (const std::exception& error) {
        out.passed = false;
        out.checks.push_back({"core_exception", false, error.what()});
    } catch (...) {
        out.passed = false;
        out.checks.push_back({"core_exception", false, "unknown failure"});
    }
    return out;
}
} // namespace rtapi
