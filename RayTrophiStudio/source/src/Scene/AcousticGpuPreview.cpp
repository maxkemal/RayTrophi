#include "Audio/AcousticScene.h"
#include "Audio/AcousticRenderDebug.h"
#include "Audio/AcousticGpu.h"
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
    for (std::size_t i = 0; i < geometry.names.size(); ++i) {
        gpuTriangles[geometry.names[i]] += geometry.instances[i].triangleCount;
    }
    // Check coverage by canonical flat-object identity; never read face facades
    // or force CPU position synchronization/BVH construction.
    for (const auto& object : ctx.scene.world.objects) {
        if (!object) {
            continue;
        }
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh) {
            throw std::runtime_error("Non-flat scene geometry is outside acoustic coverage");
        }
        if (ctx.scene.isEditorPendingDeleteObjectName(mesh->nodeName)) {
            continue;
        }
        if (!mesh->geometry || mesh->geometry->indices.empty()) {
            continue;
        }
        if (mesh->transient || !names.contains(mesh->nodeName)) {
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
        auto& item = geometry.instances[i];
        item.object = found->second.first;
        const auto& response = found->second.second;
        item.defined = response.value("defined", false) ? 1u : 0u;
        if (item.defined) {
            for (unsigned band = 0; band < 6; ++band) {
                item.energy[band] = response.at("bands").at(band).at("transmission").get<float>();
                if (!std::isfinite(item.energy[band]) || item.energy[band] < 0 ||
                    item.energy[band] > 1) {
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
    if (std::sqrt(squared) * result.source[3] < 1e-4) {
        throw std::runtime_error("Acoustic segment must be at least 0.1 mm long");
    }
    return result;
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
    if (now < tick.next) {
        return;
    }
    tick.next = now + std::chrono::milliseconds(250);
    const auto entries = ctx.scene.audio->invoke("acoustics_preview_status", json::object());
    if (entries.empty()) {
        if (backend) {
            backend->cancelAcousticGpu();
        }
        tick.current.clear();
        tick.applied.clear();
        return;
    }
    try {
        if (!backend) {
            throw std::runtime_error("Active backend does not provide GPU acoustic queries");
        }
        if (!ctx.scene.camera) {
            throw std::runtime_error("Acoustic preview needs an active camera");
        }
        auto geometry = backend->acousticGpuGeometry();
        if (!geometry.error.empty()) {
            throw std::runtime_error(geometry.error);
        }
        responses(ctx, geometry);
        AcousticGpuBatch batch;
        batch.instances = std::move(geometry.instances);
        json identity = {{"epoch", tick.epoch}, {"geometry", geometry.revision},
            {"materials", matter::objectSubstanceFingerprint(ctx.scene)},
            {"frame", ctx.scene.timeline.current_frame}, {"sources", json::array()}};
        for (const auto& entry : entries) {
            const auto id = entry.at("id").get<std::string>();
            const auto emitter = ctx.scene.audio->invoke("get", {{"id", id}});
            if (!emitter.value("spatial", false) ||
                emitter.at("reaction").value("driver", "manual") != "manual") {
                throw std::runtime_error("Acoustic preview requires spatial manual sources");
            }
            const auto position = worldPose(ctx, emitter).at("position");
            batch.rays.push_back(ray(position, ctx.scene.camera->lookfrom, entry));
            batch.ids.push_back(id);
            identity["sources"].push_back({{"id", id}, {"position", position},
                {"preview_revision", entry.at("preview_revision")},
                {"units", entry.at("meters_per_unit")}, {"mode", entry.at("boundary_mode")}});
        }
        const auto& listener = ctx.scene.camera->lookfrom;
        identity["listener"] = {listener.x, listener.y, listener.z};
        batch.stamp = identity.dump();
        if (tick.current != batch.stamp) {
            backend->cancelAcousticGpu();
            tick.current = batch.stamp;
            tick.applied.clear();
            publish(ctx, entries, "pending", "pending");
        }
        AcousticGpuCompletion completion;
        if (backend->pollAcousticGpu(completion) && completion.stamp == tick.current &&
            completion.summaries.size() == entries.size() && completion.ids == batch.ids) {
            const char* errors[] = {"", "unknown_material", "hit_budget_exceeded",
                "ambiguous_boundary", "unsupported_geometry", "invalid_gpu_geometry"};
            for (std::size_t i = 0; i < completion.ids.size(); ++i) {
                const auto& summary = completion.summaries[i];
                const auto code = summary.state[0];
                json report = {{"source_position", identity.at("sources").at(i).at("position")},
                    {"listener_position", identity.at("listener")},
                    {"coverage", "active_gpu_as_flat_mesh"},
                    {"snapshot_complete", code == 0}, {"transmission_energy", nullptr},
                    {"distance_meters", summary.path[0]}, {"delay_seconds", summary.path[1]},
                    {"crossings", summary.state[1]}, {"raw_candidates", summary.state[2]},
                    {"solver", solver(code == 0 ? "ready" : "failed")}};
                if (code == 0) {
                    report["transmission_status"] = summary.state[1] == 0 ? "clear" :
                        batch.rays[i].listener[3] > 0.5f ? "evaluated_object_barrier" :
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
        publish(ctx, entries, "gpu_unavailable", "failed", error.what());
    }
}
} // namespace audio
