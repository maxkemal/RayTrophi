#include "RtIpcRayFusion.h"
#include "RtIpcScreenGi.h"
#include "RtIpcReflection.h"
#include "Api/RtApiRayFusion.h"
#include <stdexcept>

namespace {
bool requireBool(const nlohmann::json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_boolean())
        throw std::invalid_argument(std::string("required boolean parameter: ") + key);
    return params[key].get<bool>();
}
}

bool dispatchRayFusionIpc(const std::string& method, const nlohmann::json& params,
                         const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    using json = nlohmann::json;
    if (dispatchScreenGiIpc(method,params,enqueue,result)) return true;
    if (dispatchReflectionIpc(method,params,enqueue,result)) return true;
    if (method == "rayfusion.set_probe_overlay") {
        if (params.size() != 1) throw std::invalid_argument("requires only boolean 'enabled'");
        const bool enabled = requireBool(params, "enabled");
        result = enqueue([enabled](UIContext&) {
            const bool applied = rtapi::setRayFusionProbeOverlay(enabled);
            const auto p = rtapi::rayFusionProbeFieldStatus();
            return json{{"applied", applied}, {"requested", p.overlay_requested},
                {"ready", p.overlay_ready}, {"reason", p.overlay_reason}};
        });
        return true;
    }
    if (method == "rayfusion.set_probe_follow_camera") {
        if (params.size() != 1) throw std::invalid_argument("requires only boolean 'enabled'");
        const bool enabled = requireBool(params, "enabled");
        result = enqueue([enabled](UIContext&) {
            const bool applied = rtapi::setRayFusionProbeFollowCamera(enabled);
            return json{{"applied", applied}, {"enabled", rtapi::rayFusionProbeFieldStatus().follow_camera}};
        });
        return true;
    }
    if (method == "rayfusion.set_probe_grid") {
        // Partial by design: a caller moving the window must not have to restate
        // its shape. Unknown keys are still rejected -- a typo that is silently
        // ignored reads as "the grid did not move", which is the same symptom
        // as the field being broken.
        for (const auto& item : params.items())
            if (item.key() != "counts" && item.key() != "spacing" &&
                item.key() != "minimum" && item.key() != "center" &&
                item.key() != "auto_fit")
                throw std::invalid_argument("unknown parameter: " + item.key());
        RayFusion::GridRequest request;
        // is_number(), not is_number_float(): JSON 3 and 3.0 are the same
        // spacing, and rejecting the whole number sends the caller hunting for
        // a syntax problem that does not exist.
        if (params.contains("spacing")) {
            if (!params["spacing"].is_number())
                throw std::invalid_argument("'spacing' must be a number in world units");
            request.hasSpacing = true;
            request.spacing = params["spacing"].get<float>();
        }
        if (params.contains("counts")) {
            const auto& counts = params["counts"];
            if (!counts.is_array() || counts.size() != 3)
                throw std::invalid_argument("'counts' must be three cell counts [x,y,z]");
            for (size_t axis = 0; axis < 3; ++axis) {
                if (!counts[axis].is_number_unsigned())
                    throw std::invalid_argument("'counts' entries must be positive integers");
                request.counts[axis] = counts[axis].get<uint32_t>();
            }
            request.hasCounts = true;
        }
        if (params.contains("minimum")) {
            const auto& minimum = params["minimum"];
            if (!minimum.is_array() || minimum.size() != 3)
                throw std::invalid_argument("'minimum' must be three cell indices [x,y,z]");
            for (size_t axis = 0; axis < 3; ++axis) {
                if (!minimum[axis].is_number_integer())
                    throw std::invalid_argument("'minimum' entries are CELLS, so integers");
                request.minimum[axis] = minimum[axis].get<int32_t>();
            }
            request.hasMinimum = true;
        }
        if (params.contains("center")) {
            const auto& center = params["center"];
            if (!center.is_array() || center.size() != 3)
                throw std::invalid_argument("'center' must be three world coordinates [x,y,z]");
            for (size_t axis = 0; axis < 3; ++axis) {
                if (!center[axis].is_number())
                    throw std::invalid_argument("'center' entries must be numbers");
                request.center[axis] = center[axis].get<float>();
            }
            request.hasCenter = true;
        }
        if (params.contains("auto_fit")) {
            if (!params["auto_fit"].is_boolean())
                throw std::invalid_argument("'auto_fit' must be a boolean");
            request.hasAutoFit = true;
            request.autoFit = params["auto_fit"].get<bool>();
        }
        if (!request.hasCounts && !request.hasSpacing && !request.hasMinimum &&
            !request.hasCenter && !request.hasAutoFit)
            throw std::invalid_argument(
                "send at least one of 'counts', 'spacing', 'minimum', 'center' or 'auto_fit'");
        result = enqueue([request](UIContext&) {
            std::string error;
            const bool applied = rtapi::setRayFusionProbeGrid(request, error);
            const auto p = rtapi::rayFusionProbeFieldStatus();
            // The reported window is the APPLIED one, and it is still the OLD
            // one until a viewport frame services the field. Read it back after
            // a frame, exactly as with the producer and bounce toggles.
            return json{{"applied", applied}, {"error", error},
                {"counts", {p.counts[0], p.counts[1], p.counts[2]}},
                {"minimum", {p.minimum[0], p.minimum[1], p.minimum[2]}},
                {"minimum_cell", {p.minimum[0], p.minimum[1], p.minimum[2]}},
                {"minimum_world", {p.minimum_world[0], p.minimum_world[1], p.minimum_world[2]}},
                {"spacing", p.spacing}, {"max_slots", p.max_slots},
                {"follow_camera", p.follow_camera}};
        });
        return true;
    }
    if (method == "rayfusion.core_status") {
        if (!params.empty()) throw std::invalid_argument("rayfusion.core_status takes no parameters");
        result = enqueue([](UIContext&) {
            const auto s = rtapi::rayFusionCoreStatus();
            return json{{"stage", s.stage}, {"core_available", s.core_available},
                {"renderer_available", s.renderer_available}, {"gi_active", s.gi_active},
                {"inactive_reason", s.inactive_reason}, {"quality", s.quality},
                {"probe_abi_version", s.probe_abi_version},
                {"directional_texels_per_probe", s.directional_texels_per_probe},
                {"payload_bytes_per_probe", s.payload_bytes_per_probe},
                {"planned_rays_per_probe", s.planned_rays_per_probe},
                {"planned_probes_per_update", s.planned_probes_per_update},
                {"planned_rays_per_update", s.planned_rays_per_update}};
        });
        return true;
    }
    if (method == "rayfusion.scene_as") {
        if (!params.empty()) throw std::invalid_argument("rayfusion.scene_as takes no parameters");
        result = enqueue([](UIContext&) {
            const auto a = rtapi::rayFusionSceneASStatus();
            return json{{"hardware_rt", a.hardware_rt}, {"ready", a.ready},
                {"blas_count", a.blas_count}, {"instance_count", a.instance_count},
                {"instances_skipped", a.instances_skipped},
                {"meshes_skipped", a.meshes_skipped},
                {"as_bytes", a.as_bytes}, {"last_build_ms", a.last_build_ms},
                {"builds", a.builds},
                {"built_geometry_generation", a.built_geometry_generation},
                {"geometry_signature", a.geometry_signature},
                {"instance_signature", a.instance_signature},
                {"signature_ms", a.signature_ms},
                {"tlas_only_refreshes", a.tlas_only_refreshes},
                {"instances_hidden", a.instances_hidden},
                {"blas_indexed", a.blas_indexed},
                {"blas_flat", a.blas_flat},
                // blas_skinned says the refit is POSSIBLE; skin_refits says it
                // is HAPPENING. An animated scene must show skin_refits climb
                // while the timeline plays; stuck at 0 with blas_skinned > 0 is
                // the frozen-shadow bug back. WRITE-THEN-MEASURE: put a frame
                // between stepping the timeline and reading this.
                {"blas_skinned", a.blas_skinned},
                {"skin_refits", a.skin_refits},
                {"skin_refit_failures", a.skin_refit_failures},
                {"last_skin_refit_ms", a.last_skin_refit_ms},
                // Read the PARTS, not the total: drain is the GPU still owing a
                // frame, blas is one submit + fence around the refits, tlas is a
                // full top-level rebuild that drains again. Different fixes.
                {"last_skin_drain_ms", a.last_skin_drain_ms},
                {"last_skin_blas_ms", a.last_skin_blas_ms},
                {"last_skin_tlas_ms", a.last_skin_tlas_ms},
                {"yielded", a.yielded}, {"yields", a.yields},
                {"vram_measured", a.vram_measured},
                {"vram_usage_bytes", a.vram_usage_bytes},
                {"vram_budget_bytes", a.vram_budget_bytes},
                {"inactive_reason", a.inactive_reason}};
        });
        return true;
    }
    if (method == "rayfusion.probe_field") {
        if (!params.empty()) throw std::invalid_argument("rayfusion.probe_field takes no parameters");
        result = enqueue([](UIContext&) {
            const auto p = rtapi::rayFusionProbeFieldStatus();
            return json{{"overlay_requested", p.overlay_requested}, {"overlay_ready", p.overlay_ready},
                {"overlay_markers", p.overlay_markers}, {"overlay_reason", p.overlay_reason},
                {"follow_camera", p.follow_camera},
                {"supported", p.supported}, {"configured", p.configured},
                {"uploaded", p.uploaded}, {"bound", p.bound},
                {"producer", p.producer},
                {"bounce_requested", p.bounce.requested}, {"bounce_ready", p.bounce.ready},
                {"bounce_active", p.bounce.active}, {"bounce_reason", p.bounce.reason},
                {"bounce_signature", p.bounce.signature},
                {"bounce_instances", p.bounce.instances}, {"bounce_materials", p.bounce.materials},
                {"bounce_lights", p.bounce.lights},
                {"bounce_supported_materials", p.bounce.supportedMaterials},
                {"bounce_unsupported_materials", p.bounce.unsupportedMaterials},
                {"bounce_rejected_textured", p.bounce.rejectedTextured},
                {"bounce_rejected_transparent", p.bounce.rejectedTransparent},
                {"bounce_rejected_layered", p.bounce.rejectedLayered},
                {"bounce_rejected_flagged", p.bounce.rejectedFlagged},
                {"bounce_rejected_flag_bits", p.bounce.rejectedFlagBits},
                {"bounce_unsupported_lights", p.bounce.unsupportedLights},
                {"bounce_hits", p.bounce.hits},
                {"bounce_shaded_hits", p.bounce.shadedHits},
                // ★★★★★ Elenme sebepleri. `bounce_shaded_hits == 0` tek basina
                //   "bounce goruntuyu degistirmedi" der ama NEDEN'i soylemez ve
                //   bes ayri yol ayni sifiri uretir. Bunlarin toplami +
                //   shaded_hits == hits olmali.
                {"bounce_backface_shaded", p.bounce.backFaceShaded},
                {"bounce_skipped_disabled", p.bounce.skippedBounceDisabled},
                {"bounce_rejected_unresolved", p.bounce.rejectedUnresolved},
                {"bounce_rejected_unsupported_hit", p.bounce.rejectedUnsupported},
                {"bounce_rejected_degenerate", p.bounce.rejectedDegenerate},
                // ★★★ Emissive NEE olcusu. `emissive_triangles` kac ucgenin ISIK
                //   olarak orneklenebilir oldugunu soyler; digerleri KLOZU
                //   adlandirir. Tek bir "yok" sayisi, emissive bir arazinin cap
                //   yuzunden mi, welded mesh oldugu icin mi, yoksa transparan
                //   oldugu icin mi disarida kaldigini asla soyleyemez.
                {"emissive_triangles", p.bounce.emissiveTriangles},
                {"emissive_dropped", p.bounce.emissiveDropped},
                {"emissive_skipped_indexed", p.bounce.emissiveSkippedIndexed},
                {"emissive_rejected_transparent", p.bounce.emissiveRejectedTransparent},
                {"emissive_area", p.bounce.emissiveArea},
                {"bounce_alpha_tested", p.bounce.alphaTested},
                {"bounce_alpha_occluded", p.bounce.alphaOccluded},
                // CPU cost of rebuilding the hit/material/light tables, EVERY
                // raster frame. Read it next to scene_as signature_ms: together
                // they are what RayFusion spends before a single ray is cast.
                {"bounce_prepare_ms", p.bounce.prepareMs},
                // Toplamin FAZLARI: hangi fazin kapiya ihtiyaci oldugu
                // tahminle degil olcuyle secilsin.
                {"bounce_prepare_instances_ms", p.bounce.prepareInstancesMs},
                {"bounce_prepare_emissive_ms", p.bounce.prepareEmissiveMs},
                {"bounce_prepare_materials_ms", p.bounce.prepareMaterialsMs},
                {"bounce_prepare_upload_ms", p.bounce.prepareUploadMs},
                {"bounce_upload_skipped", p.bounce.uploadSkipped},
                {"bounce_emissive_cached", p.bounce.emissiveCached},
                {"bounce_sun_included", p.bounce.sunInBounce},
                {"bounce_sun_tint_from_lut", p.bounce.sunTintFromLut},
                {"producer_traced_requested", p.producer_traced_requested},
                {"producer_reason", p.producer_reason},
                {"hit_fraction", p.hit_fraction},
                {"rejected_inside", p.rejected_inside},
                {"auto_fit", p.auto_fit},
                {"auto_fit_mode", p.auto_fit_mode},
                {"auto_fit_reason", p.auto_fit_reason},
                {"backface_enclosed", p.backface_enclosed},
                {"mean_hit_distance", p.mean_hit_distance},
                {"trace_ms", p.trace_ms},
                {"traced_publishes", p.traced_publishes},
                {"budget_preset", p.budget_preset},
                {"producer_signature", p.producer_signature},
                {"total", p.total}, {"valid", p.valid}, {"pending", p.pending},
                {"in_flight", p.in_flight}, {"accepted", p.accepted},
                {"rejected", p.rejected},
                {"counts", {p.counts[0], p.counts[1], p.counts[2]}},
                {"minimum", {p.minimum[0], p.minimum[1], p.minimum[2]}},
                {"minimum_cell", {p.minimum[0], p.minimum[1], p.minimum[2]}},
                {"minimum_world", {p.minimum_world[0], p.minimum_world[1], p.minimum_world[2]}},
                {"spacing", p.spacing}, {"max_slots", p.max_slots}};
        });
        return true;
    }
    if (method == "rayfusion.set_probe_bounce") {
        if (params.size() != 1)
            throw std::invalid_argument("rayfusion.set_probe_bounce requires only boolean 'enabled'");
        const bool enabled = requireBool(params, "enabled");
        result = enqueue([enabled](UIContext&) {
            const bool applied = rtapi::setRayFusionProbeBounce(enabled);
            const auto p = rtapi::rayFusionProbeFieldStatus();
            return json{{"applied", applied}, {"requested", p.bounce.requested},
                {"active", p.bounce.active}, {"reason", p.bounce.reason}};
        });
        return true;
    }
    if (method == "rayfusion.set_probe_producer") {
        if (!params.contains("traced") || !params["traced"].is_boolean())
            throw std::invalid_argument("rayfusion.set_probe_producer needs boolean 'traced'");
        const bool traced = params["traced"].get<bool>();
        result = enqueue([traced](UIContext&) {
            const bool applied = rtapi::setRayFusionProbeProducer(traced);
            const auto p = rtapi::rayFusionProbeFieldStatus();
            // Report what the field ACTUALLY runs after the change, not the
            // request that was just made: this method exists to measure a
            // producer swap, and echoing the input would measure nothing.
            return json{{"applied", applied}, {"requested_traced", traced},
                {"producer", p.producer}, {"producer_reason", p.producer_reason}};
        });
        return true;
    }
    if (method == "rayfusion.validate_core") {
        if (!params.empty()) throw std::invalid_argument("rayfusion.validate_core takes no parameters");
        result = enqueue([](UIContext&) {
            const auto r = rtapi::validateRayFusionCore();
            json checks = json::array();
            for (const auto& c : r.checks)
                checks.push_back({{"name", c.name}, {"passed", c.passed}, {"detail", c.detail}});
            return json{{"passed", r.passed}, {"gpu_tested", r.gpu_tested}, {"checks", checks}};
        });
        return true;
    }
    return false;
}
