#include "RtPythonRayFusion.h"
#include "RtPythonScreenGi.h"
#include "RtPythonReflection.h"
#include "Api/RtApiRayFusion.h"

#include <stdexcept>
#include <string>

namespace rtpy {
void registerRayFusionBindings(pybind11::module_& root) {
    namespace py = pybind11;
    auto module = root.def_submodule("rayfusion", "RayFusion development status and isolated core checks");
    registerScreenGiBindings(module);
    registerReflectionBindings(module);
    module.def("set_probe_overlay", [](bool enabled) {
        const bool applied = rtapi::setRayFusionProbeOverlay(enabled);
        const auto p = rtapi::rayFusionProbeFieldStatus();
        py::dict out;
        out["applied"] = applied; out["requested"] = p.overlay_requested;
        out["ready"] = p.overlay_ready; out["reason"] = p.overlay_reason;
        return out;
    }, py::arg("enabled").noconvert());
    module.def("set_probe_follow_camera", [](bool enabled) {
        const bool applied = rtapi::setRayFusionProbeFollowCamera(enabled);
        py::dict out;
        out["applied"] = applied;
        out["enabled"] = rtapi::rayFusionProbeFieldStatus().follow_camera;
        return out;
    }, py::arg("enabled").noconvert());
    module.def("core_status", [] {
        const auto s = rtapi::rayFusionCoreStatus();
        py::dict out;
        out["stage"] = s.stage;
        out["core_available"] = s.core_available;
        out["renderer_available"] = s.renderer_available;
        out["gi_active"] = s.gi_active;
        out["inactive_reason"] = s.inactive_reason;
        out["quality"] = s.quality;
        out["probe_abi_version"] = s.probe_abi_version;
        out["directional_texels_per_probe"] = s.directional_texels_per_probe;
        out["payload_bytes_per_probe"] = s.payload_bytes_per_probe;
        out["planned_rays_per_probe"] = s.planned_rays_per_probe;
        out["planned_probes_per_update"] = s.planned_probes_per_update;
        out["planned_rays_per_update"] = s.planned_rays_per_update;
        return out;
    });
    module.def("scene_as", [] {
        const auto a = rtapi::rayFusionSceneASStatus();
        py::dict out;
        out["hardware_rt"] = a.hardware_rt;
        out["ready"] = a.ready;
        out["blas_count"] = a.blas_count;
        out["instance_count"] = a.instance_count;
        out["instances_skipped"] = a.instances_skipped;
        out["meshes_skipped"] = a.meshes_skipped;
        out["as_bytes"] = a.as_bytes;
        out["last_build_ms"] = a.last_build_ms;
        out["builds"] = a.builds;
        out["built_geometry_generation"] = a.built_geometry_generation;
        out["geometry_signature"] = a.geometry_signature;
        out["instance_signature"] = a.instance_signature;
        out["signature_ms"] = a.signature_ms;
        out["tlas_only_refreshes"] = a.tlas_only_refreshes;
        out["instances_hidden"] = a.instances_hidden;
        out["blas_indexed"] = a.blas_indexed;
        out["blas_flat"] = a.blas_flat;
        out["blas_skinned"] = a.blas_skinned;
        out["skin_refits"] = a.skin_refits;
        out["skin_refit_failures"] = a.skin_refit_failures;
        out["last_skin_refit_ms"] = a.last_skin_refit_ms;
        out["last_skin_drain_ms"] = a.last_skin_drain_ms;
        out["last_skin_blas_ms"] = a.last_skin_blas_ms;
        out["last_skin_tlas_ms"] = a.last_skin_tlas_ms;
        out["yielded"] = a.yielded;
        out["yields"] = a.yields;
        out["vram_measured"] = a.vram_measured;
        out["vram_usage_bytes"] = a.vram_usage_bytes;
        out["vram_budget_bytes"] = a.vram_budget_bytes;
        out["inactive_reason"] = a.inactive_reason;
        return out;
    });
    module.def("probe_field", [] {
        const auto p = rtapi::rayFusionProbeFieldStatus();
        py::dict out;
        out["supported"] = p.supported;
        out["overlay_requested"] = p.overlay_requested;
        out["overlay_ready"] = p.overlay_ready;
        out["overlay_markers"] = p.overlay_markers;
        out["overlay_reason"] = p.overlay_reason;
        out["follow_camera"] = p.follow_camera;
        out["configured"] = p.configured;
        out["uploaded"] = p.uploaded;
        out["bound"] = p.bound;
        out["producer"] = p.producer;
        out["bounce_requested"] = p.bounce.requested;
        out["bounce_ready"] = p.bounce.ready;
        out["bounce_active"] = p.bounce.active;
        out["bounce_reason"] = p.bounce.reason;
        out["bounce_signature"] = p.bounce.signature;
        out["bounce_instances"] = p.bounce.instances;
        out["bounce_materials"] = p.bounce.materials;
        out["bounce_lights"] = p.bounce.lights;
        out["bounce_supported_materials"] = p.bounce.supportedMaterials;
        out["bounce_unsupported_materials"] = p.bounce.unsupportedMaterials;
        out["bounce_rejected_textured"] = p.bounce.rejectedTextured;
        out["bounce_rejected_transparent"] = p.bounce.rejectedTransparent;
        out["bounce_rejected_layered"] = p.bounce.rejectedLayered;
        out["bounce_rejected_flagged"] = p.bounce.rejectedFlagged;
        out["bounce_rejected_flag_bits"] = p.bounce.rejectedFlagBits;
        out["bounce_unsupported_lights"] = p.bounce.unsupportedLights;
        out["bounce_hits"] = p.bounce.hits;
        out["bounce_shaded_hits"] = p.bounce.shadedHits;
        out["bounce_backface_shaded"] = p.bounce.backFaceShaded;
        out["bounce_skipped_disabled"] = p.bounce.skippedBounceDisabled;
        out["bounce_rejected_unresolved"] = p.bounce.rejectedUnresolved;
        out["bounce_rejected_unsupported_hit"] = p.bounce.rejectedUnsupported;
        out["bounce_rejected_degenerate"] = p.bounce.rejectedDegenerate;
        out["emissive_triangles"] = p.bounce.emissiveTriangles;
        out["emissive_dropped"] = p.bounce.emissiveDropped;
        out["emissive_skipped_indexed"] = p.bounce.emissiveSkippedIndexed;
        out["emissive_rejected_transparent"] = p.bounce.emissiveRejectedTransparent;
        out["emissive_area"] = p.bounce.emissiveArea;
        out["bounce_alpha_tested"] = p.bounce.alphaTested;
        out["bounce_alpha_occluded"] = p.bounce.alphaOccluded;
        out["bounce_prepare_ms"] = p.bounce.prepareMs;
        out["bounce_prepare_instances_ms"] = p.bounce.prepareInstancesMs;
        out["bounce_prepare_emissive_ms"] = p.bounce.prepareEmissiveMs;
        out["bounce_prepare_materials_ms"] = p.bounce.prepareMaterialsMs;
        out["bounce_prepare_upload_ms"] = p.bounce.prepareUploadMs;
        out["bounce_upload_skipped"] = p.bounce.uploadSkipped;
        out["bounce_emissive_cached"] = p.bounce.emissiveCached;
        out["bounce_sun_included"] = p.bounce.sunInBounce;
        out["bounce_sun_tint_from_lut"] = p.bounce.sunTintFromLut;
        out["producer_traced_requested"] = p.producer_traced_requested;
        out["producer_reason"] = p.producer_reason;
        out["hit_fraction"] = p.hit_fraction;
        out["rejected_inside"] = p.rejected_inside;
        out["auto_fit"] = p.auto_fit;
        out["auto_fit_mode"] = p.auto_fit_mode;
        out["auto_fit_reason"] = p.auto_fit_reason;
        out["backface_enclosed"] = p.backface_enclosed;
        out["mean_hit_distance"] = p.mean_hit_distance;
        out["trace_ms"] = p.trace_ms;
        out["traced_publishes"] = p.traced_publishes;
        out["budget_preset"] = p.budget_preset;
        out["producer_signature"] = p.producer_signature;
        out["total"] = p.total;
        out["valid"] = p.valid;
        out["pending"] = p.pending;
        out["in_flight"] = p.in_flight;
        out["accepted"] = p.accepted;
        out["rejected"] = p.rejected;
        out["counts"] = py::make_tuple(p.counts[0], p.counts[1], p.counts[2]);
        out["minimum"] = py::make_tuple(p.minimum[0], p.minimum[1], p.minimum[2]);
        out["minimum_cell"] = out["minimum"];
        out["minimum_world"] = py::make_tuple(p.minimum_world[0], p.minimum_world[1], p.minimum_world[2]);
        out["spacing"] = p.spacing;
        out["max_slots"] = p.max_slots;
        return out;
    });
    // Partial grid edit. Every argument is optional and None means "keep what
    // the field has" -- restating a shape you did not mean to change is how a
    // caller silently resets a count. 'minimum' is cells, 'center' is world
    // units; sending both is an error rather than a precedence rule nobody
    // would remember.
    module.def("set_probe_grid", [](py::object counts, py::object spacing,
                                    py::object minimum, py::object center,
                                    py::object auto_fit) {
        RayFusion::GridRequest request;
        const auto readTriple = [](const py::object& value, const char* name) {
            const auto sequence = value.cast<py::sequence>();
            if (py::len(sequence) != 3)
                throw std::invalid_argument(std::string(name) + " needs three components");
            return sequence;
        };
        if (!counts.is_none()) {
            const auto sequence = readTriple(counts, "counts");
            for (size_t axis = 0; axis < 3; ++axis) {
                const long long value = sequence[axis].cast<long long>();
                if (value < 1 || value > 128)
                    throw std::invalid_argument("each grid count must be 1..128");
                request.counts[axis] = static_cast<uint32_t>(value);
            }
            request.hasCounts = true;
        }
        if (!spacing.is_none()) {
            request.spacing = spacing.cast<float>();
            request.hasSpacing = true;
        }
        if (!minimum.is_none()) {
            const auto sequence = readTriple(minimum, "minimum");
            for (size_t axis = 0; axis < 3; ++axis)
                request.minimum[axis] = sequence[axis].cast<int32_t>();
            request.hasMinimum = true;
        }
        if (!center.is_none()) {
            const auto sequence = readTriple(center, "center");
            for (size_t axis = 0; axis < 3; ++axis)
                request.center[axis] = sequence[axis].cast<float>();
            request.hasCenter = true;
        }
        if (!auto_fit.is_none()) {
            request.autoFit = auto_fit.cast<bool>();
            request.hasAutoFit = true;
        }
        if (!request.hasCounts && !request.hasSpacing && !request.hasMinimum &&
            !request.hasCenter && !request.hasAutoFit)
            throw std::invalid_argument(
                "set_probe_grid needs at least one of counts, spacing, minimum, "
                "center or auto_fit");
        std::string error;
        const bool applied = rtapi::setRayFusionProbeGrid(request, error);
        const auto p = rtapi::rayFusionProbeFieldStatus();
        py::dict out;
        out["applied"] = applied;
        out["error"] = error;
        // The APPLIED window, which is still the old one until the viewport
        // services the field on its next frame.
        out["counts"] = py::make_tuple(p.counts[0], p.counts[1], p.counts[2]);
        out["minimum"] = py::make_tuple(p.minimum[0], p.minimum[1], p.minimum[2]);
        out["minimum_cell"] = out["minimum"];
        out["minimum_world"] = py::make_tuple(p.minimum_world[0], p.minimum_world[1], p.minimum_world[2]);
        out["spacing"] = p.spacing;
        out["max_slots"] = p.max_slots;
        out["follow_camera"] = p.follow_camera;
        return out;
    }, py::arg("counts") = py::none(), py::arg("spacing") = py::none(),
       py::arg("minimum") = py::none(), py::arg("center") = py::none(),
       py::arg("auto_fit") = py::none());
    module.def("set_probe_bounce", [](bool enabled) {
        const bool applied = rtapi::setRayFusionProbeBounce(enabled);
        const auto p = rtapi::rayFusionProbeFieldStatus();
        py::dict out;
        out["applied"] = applied;
        out["requested"] = p.bounce.requested;
        out["active"] = p.bounce.active;
        out["reason"] = p.bounce.reason;
        return out;
    }, py::arg("enabled").noconvert());
    module.def("set_probe_producer", [](bool traced) {
        const bool applied = rtapi::setRayFusionProbeProducer(traced);
        const auto p = rtapi::rayFusionProbeFieldStatus();
        py::dict out;
        out["applied"] = applied;
        out["requested_traced"] = traced;
        out["producer"] = p.producer;
        out["producer_reason"] = p.producer_reason;
        return out;
    }, py::arg("traced"));
    module.def("validate_core", [] {
        const auto r = rtapi::validateRayFusionCore();
        py::dict out;
        out["passed"] = r.passed;
        out["gpu_tested"] = r.gpu_tested;
        py::list checks;
        for (const auto& c : r.checks) {
            py::dict item;
            item["name"] = c.name;
            item["passed"] = c.passed;
            item["detail"] = c.detail;
            checks.append(item);
        }
        out["checks"] = checks;
        return out;
    });
}
} // namespace rtpy
