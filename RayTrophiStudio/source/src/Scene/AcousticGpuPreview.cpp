#include "Audio/AcousticScene.h"
#include "Audio/AcousticRenderDebug.h"
#include "Audio/AcousticGpu.h"
#include "Audio/AcousticPreview.h"
#include "Audio/AcousticGeometryPolicy.h"
#include "Audio/AcousticSurface.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "Backend/VulkanBackend.h"
#include "ObjectSubstance.h"
#include "TriangleMesh.h"
#include "Camera.h"
#include "scene_ui.h"
#include "scene_data.h"

#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <typeinfo>

namespace audio {
namespace {
using json = nlohmann::json;

struct PreviewTick {
    const void* owner = nullptr;
    const void* backend = nullptr;
    std::uint64_t epoch = 0;
    std::string current;
    std::string applied;
    std::chrono::steady_clock::time_point next;
    std::chrono::steady_clock::time_point requested;
};

json solver(const char* state) {
    return {{"backend", "active_vulkan_as"}, {"state", state},
        {"cpu_bvh", false}, {"result_bytes_per_source", sizeof(AcousticGpuSummary)}};
}

void publish(UIContext& ctx, const json& entries, const char* status,
    const char* state, const std::string& error = {}) {
    for (const auto& entry : entries) {
        json report = {{"transmission_status", status}, {"transmission_energy", nullptr},
            {"solver", solver(state)}};
        if (!error.empty()) {
            report["error"] = error;
        }
        ctx.scene.audio->acousticPreviewResult(entry.at("id").get<std::string>(), report);
    }
}

void responses(UIContext& ctx, AcousticGpuGeometry& geometry) {
    std::set<std::string> names(geometry.names.begin(), geometry.names.end());
    std::map<std::string, std::uint64_t> gpuTriangles;
    std::map<std::string, std::uint64_t> authoredTriangles;
    std::set<std::string> simulationNames;
    for (std::size_t i = 0; i < geometry.names.size(); ++i) {
        gpuTriangles[geometry.names[i]] += geometry.instances[i].triangleCount;
    }
    // Check coverage by canonical flat-object identity; never read face facades
    // or force CPU position synchronization/BVH construction.
    for (const auto& object : ctx.scene.world.objects) {
        if (!object) {
            continue;
        }
        if (isAcousticSimulationVisual(*object)) {
            ++geometry.ignoredSimulationObjects;
            if (const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
                simulationNames.insert(mesh->nodeName);
            }
            continue;
        }
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh) {
            throw std::runtime_error(std::string("Non-flat scene geometry is outside acoustic coverage: ") +
                typeid(*object).name());
        }
        if (ctx.scene.isEditorPendingDeleteObjectName(mesh->nodeName)) {
            continue;
        }
        if (!mesh->geometry || mesh->geometry->indices.empty()) {
            continue;
        }
        if (!names.contains(mesh->nodeName)) {
            throw std::runtime_error("Active AS omits authored acoustic geometry, including hidden walls");
        }
        if (mesh->geometry->indices.size() % 3 != 0) {
            throw std::runtime_error("Canonical acoustic mesh has incomplete triangle indices");
        }
        authoredTriangles[mesh->nodeName] += mesh->geometry->indices.size() / 3;
    }
    for (const auto& [name, count] : authoredTriangles) {
        if (gpuTriangles[name] != count) {
            throw std::runtime_error("Active AS triangle coverage differs from canonical flat objects");
        }
    }
    std::map<std::string, std::pair<std::uint32_t, json>> objects;
    for (std::size_t i = 0; i < geometry.names.size(); ++i) {
        const auto& name = geometry.names[i];
        auto& item = geometry.instances[i];
        if (simulationNames.contains(name) && item.defined != kAcousticIgnoredVisual) {
            item.defined = kAcousticIgnoredVisual;
            ++geometry.ignoredSimulationInstances;
        }
        if (item.defined == kAcousticIgnoredVisual) {
            continue;
        }
        if (name.empty()) {
            continue;
        } // Unused sparse customIndex slot.
        auto found = objects.find(name);
        if (found == objects.end()) {
            const auto state = matter::objectSubstanceState(ctx.scene, name, true);
            found = objects.emplace(name, std::make_pair(
                static_cast<std::uint32_t>(objects.size()),
                state.value("acoustics", json::object()))).first;
        }
        item.object = found->second.first;
        const auto& response = found->second.second;
        item.defined = response.value("defined", false) ? 1u : 0u;
        if (item.defined) {
            for (unsigned band = 0; band < 6; ++band) {
                item.energy[band] = response.at("bands").at(band).at("transmission").get<float>();
                item.reflection[band] = response.at("bands").at(band)
                    .at("specular_reflection").get<float>();
                item.diffuse[band] = response.at("bands").at(band)
                    .at("diffuse_reflection").get<float>();
                if (!std::isfinite(item.energy[band]) || item.energy[band] < 0 ||
                    item.energy[band] > 1 || !std::isfinite(item.reflection[band]) ||
                    item.reflection[band] < 0 || item.reflection[band] > 1 ||
                    !std::isfinite(item.diffuse[band]) || item.diffuse[band] < 0 ||
                    item.diffuse[band] > 1 ||
                    item.energy[band] + item.reflection[band] + item.diffuse[band] > 1.00001f) {
                    throw std::runtime_error("Invalid acoustic transmission energy");
                }
            }
        }
        item.energy[6] = item.energy[7] = 1;
    }
}

AcousticGpuRay ray(const json& source, const Vec3& listener, const json& entry) {
    if (!source.is_array() || source.size() != 3) {
        throw std::runtime_error("Acoustic source position needs three components");
    }
    AcousticGpuRay result;
    double squared = 0;
    const double end[] = {listener.x, listener.y, listener.z};
    for (unsigned axis = 0; axis < 3; ++axis) {
        const double start = source.at(axis).get<double>();
        if (!std::isfinite(start) || !std::isfinite(end[axis]) ||
            std::abs(start) > 1e9 || std::abs(end[axis]) > 1e9) {
            throw std::runtime_error("Acoustic GPU position is not finite or exceeds bounds");
        }
        result.source[axis] = static_cast<float>(start);
        result.listener[axis] = static_cast<float>(end[axis]);
        squared += (start - end[axis]) * (start - end[axis]);
    }
    result.source[3] = entry.at("meters_per_unit").get<float>();
    result.listener[3] = entry.at("boundary_mode") == "object_barrier" ? 1.0f : 0.0f;
    result.options[0] = entry.value("reflections", false) ? 1.0f : 0.0f;
    result.options[1] = 1.0f;
    result.options[2] = entry.value("diffuse", false) ? 1.0f : 0.0f;
    if (std::sqrt(squared) * result.source[3] < 1e-4) {
        throw std::runtime_error("Acoustic segment must be at least 0.1 mm long");
    }
    return result;
}

void collectPreview(UIContext& ctx, Backend::VulkanBackendAdapter& backend,
    PreviewTick& tick) {
    AcousticGpuCompletion completion;
    if (!backend.pollAcousticGpu(completion) || completion.stamp != tick.current) {
        return;
    }
    const auto identity = json::parse(completion.stamp);
    const auto entries = ctx.scene.audio->invoke("acoustics_preview_status", json::object());
    const auto age = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tick.requested).count();
    const auto geometry = backend.acousticGpuGeometry();
    if (!geometry.error.empty()) {
        throw std::runtime_error("Acoustic completion coverage unavailable: " + geometry.error);
    }
    if (identity.at("boundaries") != geometry.boundaryRevision) {
        throw std::runtime_error("Acoustic completion environment boundary revision changed");
    }
    if (identity.at("materials") != matter::objectSubstanceFingerprint(ctx.scene)) {
        throw std::runtime_error("Acoustic completion material revision changed");
    }
    if (identity.at("audio_generation") != ctx.scene.audio->generation()) {
        throw std::runtime_error("Acoustic completion audio ownership changed");
    }
    if (age > 1000) {
        throw std::runtime_error("Acoustic completion exceeded 1000 ms: " + std::to_string(age));
    }
    if (completion.summaries.size() != completion.ids.size() ||
        identity.at("sources").size() != completion.ids.size()) {
        throw std::runtime_error("Acoustic GPU completion has inconsistent source counts");
    }
    const char* errors[] = {"", "unknown_material", "hit_budget_exceeded",
        "ambiguous_boundary", "unsupported_geometry", "invalid_gpu_geometry"};
    for (std::size_t i = 0; i < completion.ids.size(); ++i) {
        const auto& source = identity.at("sources").at(i);
        bool current = false;
        for (const auto& entry : entries) {
            current = current || (entry.at("id") == completion.ids[i] &&
                source.at("id") == completion.ids[i] &&
                entry.at("preview_revision") == source.at("preview_revision"));
        }
        if (!current) {
            continue;
        }
        const auto emitter = ctx.scene.audio->invoke("get", {{"id", completion.ids[i]}});
        if (source.value("diffuse", false) && !supportsDiffuseSource(emitter)) {
            continue;
        }
        if (source.at("reaction") != emitter.at("reaction")) {
            continue;
        }
        const auto runtime = ctx.scene.audio->invoke("source_runtime",
            {{"id", completion.ids[i]}});
        if (!runtime.at("valid").get<bool>() || !runtime.at("spatial").get<bool>()) {
            ctx.scene.audio->acousticPreviewResult(completion.ids[i],
                {{"transmission_status", "source_unavailable"}, {"transmission_energy", nullptr},
                 {"solver", solver("failed")}});
            continue;
        }
        const auto& summary = completion.summaries[i];
        const auto code = summary.state[0];
        json report = {{"source_position", source.at("position")},
            {"position_origin", source.at("position_origin")},
            {"listener_position", identity.at("listener")},
            {"pose_policy", "sampled_runtime_world"}, {"result_age_ms", age},
            {"coverage", "authored_flat_environment"}, {"snapshot_complete", code == 0},
            {"simulation_visuals_are_boundaries", false},
            {"ignored_simulation_instances", source.value("ignored_simulation_instances", 0u)},
            {"ignored_simulation_objects", source.value("ignored_simulation_objects", 0u)},
            {"transmission_energy", nullptr}, {"distance_meters", summary.path[0]},
            {"delay_seconds", summary.path[1]}, {"crossings", summary.state[1]},
            {"raw_candidates", summary.state[2]},
            {"solver", solver(code == 0 ? "ready" : "failed")}, {"reflections", json::array()}};
        report["diffuse"] = {{"model", "single_bounce_lambert_nee"},
            {"enabled", summary.diffuseState[0] != 0},
            {"samples", summary.diffuseState[0]}, {"contributions", summary.diffuseState[1]},
            {"rejected_queries", summary.diffuseState[2]},
            {"near_receiver_rejections", summary.diffuseState[3]},
            {"complete", summary.diffuseState[2] == 0 && summary.diffuseState[3] == 0},
            {"quadrature_samples", 64}, {"delay_bins", kAcousticDiffuseBins},
            {"maximum_excess_delay_seconds", 1.0},
            {"receiver_energy_normalization", "free_field_direct"},
            {"phase_resolved", false}, {"receiver_exclusion_radius_m", 0.1}};
        for (const auto& reflection : summary.reflections) {
            if (reflection.path[2] < 0.5f) {
                continue;
            }
            json energy = json::array();
            for (unsigned band = 0; band < 6; ++band) {
                energy.push_back(reflection.energy[band]);
            }
            bool finite = std::isfinite(reflection.path[0]) &&
                std::isfinite(reflection.path[1]) && reflection.path[0] >= 0 &&
                reflection.path[1] >= 0 && reflection.path[1] <= 1;
            for (const auto& band : energy) {
                finite = finite && std::isfinite(band.get<float>()) && band.get<float>() >= 0;
            }
            if (!finite) {
                report["diffuse"]["complete"] = false;
                report["diffuse"]["invalid_response"] = true;
                continue;
            }
            report["reflections"].push_back({{"energy", energy},
                {"kind", reflection.path[3] > 0.5f ? "diffuse" : "specular"},
                {"energy_reference", reflection.path[3] > 0.5f ?
                    "receiver_relative_to_free_field_direct" : "material_fraction"},
                {"distance_meters", reflection.path[0]},
                {"excess_delay_seconds", reflection.path[1]}});
        }
        if (code == 0) {
            report["transmission_status"] = summary.state[1] == 0 ? "clear" :
                source.at("mode") == "object_barrier" ? "evaluated_object_barrier" :
                "evaluated_thin_surface";
            report["transmission_energy"] = json::array();
            for (unsigned band = 0; band < 6; ++band) {
                report["transmission_energy"].push_back(summary.energy[band]);
            }
        } else {
            report["transmission_status"] = code < 6 ? errors[code] : "gpu_error";
        }
        ctx.scene.audio->acousticPreviewResult(completion.ids[i], report);
    }
    tick.applied = tick.current;
}

bool sameRequestOwners(const json& first, const json& second) {
    for (const char* key : {"epoch", "boundaries", "materials", "audio_generation"}) {
        if (first.at(key) != second.at(key)) {
            return false;
        }
    }
    const auto& a = first.at("sources");
    const auto& b = second.at("sources");
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (const char* key : {"id", "preview_revision", "position_origin", "reaction"}) {
            if (a.at(i).at(key) != b.at(i).at(key)) {
                return false;
            }
        }
    }
    return true;
}
}

void resolveAcousticGpuResponses(UIContext& ctx, AcousticGpuGeometry& geometry) {
    responses(ctx, geometry);
}

void tickSceneAcoustics(UIContext& ctx) {
    updateAcousticRenderDebug(ctx);
    static PreviewTick tick;
    auto* backend = dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr);
    if (tick.owner != ctx.scene.audio.get() || tick.backend != backend) {
        tick.owner = ctx.scene.audio.get();
        tick.backend = backend;
        ++tick.epoch;
        tick.current.clear();
        tick.applied.clear();
        tick.next = {};
    }
    const auto now = std::chrono::steady_clock::now();
    try {
        if (now < tick.next && (!backend || !ctx.scene.camera)) {
            return;
        }
        if (!backend) {
            throw std::runtime_error("Active backend does not provide GPU acoustic queries");
        }
        if (!ctx.scene.camera) {
            throw std::runtime_error("Acoustic preview needs an active camera");
        }
        // Poll every UI frame, before throttling submissions or replacing the pose stamp.
        // Otherwise a continuously moving physical source cancels every completed query.
        collectPreview(ctx, *backend, tick);
        if (now < tick.next) {
            return;
        }
        tick.next = now + std::chrono::milliseconds(250);
        const auto entries = ctx.scene.audio->invoke("acoustics_preview_status", json::object());
        if (entries.empty()) {
            backend->cancelAcousticGpu();
            tick.current.clear();
            tick.applied.clear();
            return;
        }
        auto geometry = backend->acousticGpuGeometry();
        if (!geometry.error.empty()) {
            throw std::runtime_error(geometry.error);
        }
        responses(ctx, geometry);
        AcousticGpuBatch batch;
        batch.instances = std::move(geometry.instances);
        json identity = {{"epoch", tick.epoch}, {"geometry", geometry.revision},
            {"boundaries", geometry.boundaryRevision},
            {"materials", matter::objectSubstanceFingerprint(ctx.scene)},
            {"audio_generation", ctx.scene.audio->generation()},
            {"frame", ctx.scene.timeline.current_frame}, {"sources", json::array()}};
        json active = json::array();
        for (const auto& entry : entries) {
            const auto id = entry.at("id").get<std::string>();
            const auto emitter = ctx.scene.audio->invoke("get", {{"id", id}});
            const auto runtime = ctx.scene.audio->invoke("source_runtime", {{"id", id}});
            if (entry.value("diffuse", false) && !supportsDiffuseSource(emitter)) {
                ctx.scene.audio->acousticPreviewResult(id,
                    {{"transmission_status", "source_unavailable"},
                     {"error", "Diffuse transport requires an omnidirectional source"},
                     {"solver", solver("failed")}});
                continue;
            }
            if (!runtime.at("spatial").get<bool>() || !runtime.at("valid").get<bool>()) {
                ctx.scene.audio->acousticPreviewResult(id,
                    {{"transmission_status", "source_unavailable"}, {"transmission_energy", nullptr},
                     {"solver", solver("failed")}});
                continue;
            }
            const auto position = runtime.at("position");
            try {
                batch.rays.push_back(ray(position, ctx.scene.camera->lookfrom, entry));
            } catch (const std::exception& error) {
                ctx.scene.audio->acousticPreviewResult(id, {{"transmission_status", "invalid_pose"},
                    {"transmission_energy", nullptr}, {"error", error.what()},
                    {"solver", solver("failed")}});
                continue;
            }
            batch.ids.push_back(id);
            active.push_back(entry);
            identity["sources"].push_back({{"id", id}, {"position", position},
                {"position_origin", runtime.at("position_origin")},
                {"diffuse", entry.value("diffuse", false)},
                {"ignored_simulation_instances", geometry.ignoredSimulationInstances},
                {"ignored_simulation_objects", geometry.ignoredSimulationObjects},
                {"reaction", emitter.at("reaction")},
                {"preview_revision", entry.at("preview_revision")},
                {"units", entry.at("meters_per_unit")}, {"mode", entry.at("boundary_mode")}});
        }
        if (batch.ids.empty()) {
            backend->cancelAcousticGpu();
            tick.current.clear();
            tick.applied.clear();
            return;
        }
        const auto& listener = ctx.scene.camera->lookfrom;
        identity["listener"] = {listener.x, listener.y, listener.z};
        batch.stamp = identity.dump();
        if (tick.current != batch.stamp) {
            const auto age = std::chrono::duration<double>(now - tick.requested).count();
            if (!tick.current.empty() && tick.applied != tick.current && age < 1 &&
                sameRequestOwners(json::parse(tick.current), identity)) {
                // Finish the bounded in-flight sample instead of starving a moving source.
                // The next submission samples its latest world pose.
                return;
            }
            backend->cancelAcousticGpu();
            tick.current = batch.stamp;
            tick.applied.clear();
            tick.requested = now;
            publish(ctx, active, "pending", "pending");
        }
        if (tick.applied != tick.current && !backend->queueAcousticGpu(
                std::move(batch), geometry.revision)) {
            throw std::runtime_error(backend->acousticGpuError());
        }
    } catch (const std::exception& error) {
        if (backend) {
            backend->cancelAcousticGpu();
        }
        tick.current.clear();
        tick.applied.clear();
        tick.next = now + std::chrono::milliseconds(250);
        publish(ctx, ctx.scene.audio->invoke("acoustics_preview_status", json::object()),
            "gpu_unavailable", "failed", error.what());
    }
}
} // namespace audio
