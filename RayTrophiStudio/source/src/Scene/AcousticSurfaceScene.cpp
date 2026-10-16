#include "Audio/AcousticSurface.h"
#include "Audio/AudioService.h"
#include "Backend/VulkanBackend.h"
#include "ObjectSubstance.h"
#include "scene_data.h"
#include "scene_ui.h"
#include <chrono>
#include <map>
#include <stdexcept>

namespace audio {
void updateAcousticRenderDebug(UIContext& ctx) {
    auto* backend = dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr);
    if (!backend) {
        return;
    }
    if (ctx.render_settings.debug_view != kAcousticSurfaceDebugView) {
        backend->disableAcousticSurface();
        return;
    }
    struct Cache {
        const void* owner = nullptr;
        const void* backend = nullptr;
        std::chrono::steady_clock::time_point next;
        std::string revision, error;
        std::uint64_t materials = 0;
        std::shared_ptr<const std::vector<AcousticSurfaceInstance>> instances;
    };
    static Cache cache;
    if (cache.owner != ctx.scene.audio.get() || cache.backend != backend) {
        cache = {};
        cache.owner = ctx.scene.audio.get();
        cache.backend = backend;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= cache.next) {
        cache.next = now + std::chrono::milliseconds(250);
        try {
            auto geometry = backend->acousticGpuGeometry();
            if (!geometry.error.empty()) {
                throw std::runtime_error(geometry.error);
            }
            const auto materials = matter::objectSubstanceFingerprint(ctx.scene);
            if (!cache.instances || cache.revision != geometry.revision ||
                cache.materials != materials) {
                resolveAcousticGpuResponses(ctx, geometry);
                auto instances = std::make_shared<std::vector<AcousticSurfaceInstance>>();
                instances->resize(geometry.instances.size());
                std::map<std::string, nlohmann::json> responses;
                for (std::size_t i = 0; i < instances->size(); ++i) {
                    auto& item = instances->at(i);
                    item.geometry = geometry.instances[i];
                    if (!item.geometry.defined) {
                        continue;
                    }
                    const auto& name = geometry.names[i];
                    auto found = responses.find(name);
                    if (found == responses.end()) {
                        found = responses.emplace(name,
                            matter::objectSubstanceState(ctx.scene, name, true).at("acoustics")).first;
                    }
                    for (unsigned band = 0; band < 6; ++band) {
                        item.reflection[band] = found->second.at("bands").at(band)
                            .at("diffuse_reflection").get<float>();
                        item.specular[band] = found->second.at("bands").at(band)
                            .at("specular_reflection").get<float>();
                    }
                }
                cache.instances = std::move(instances);
                cache.revision = geometry.revision;
                cache.materials = materials;
            }
            cache.error.clear();
        } catch (const std::exception& exception) {
            cache.instances.reset();
            cache.error = exception.what();
        }
    }
    AcousticSurfaceBatch batch;
    batch.instances = cache.instances;
    batch.error = cache.error;
    const auto workspace = ctx.scene.audio->invoke("workspace_get", nlohmann::json::object());
    batch.band = workspace.value("acoustic_band", -1);
    batch.reflectionSamples = workspace.value("acoustic_reflection_samples", 1u);
    if (workspace.value("acoustic_all_sources", false)) {
        const auto data = ctx.scene.audio->snapshot();
        for (const auto& emitter : data.at("emitters")) {
            const auto source = acousticRenderSource(ctx, emitter);
            if (source[7] < 0.5f) {
                continue;
            }
            if (batch.sourceCount == batch.sources.size()) {
                batch.error = "Surface source budget exceeded (maximum four); isolate a source";
                break;
            }
            batch.sources[batch.sourceCount++] = source;
        }
    } else {
        batch.sources[0] = acousticRenderSource(ctx);
        batch.sourceCount = batch.sources[0][7] > 0.5f ? 1u : 0u;
    }
    const auto revision = cache.revision;
    batch.stillCurrent = [backend, revision] {
        const auto current = backend->acousticGpuGeometry();
        return current.error.empty() && current.revision == revision;
    };
    batch.stamp = nlohmann::json{{"geometry", cache.revision}, {"materials", cache.materials},
        {"sources", batch.sources}, {"count", batch.sourceCount}, {"band", batch.band},
        {"reflections", batch.reflectionSamples}, {"error", batch.error}}.dump();
    if (backend->setAcousticSurface(std::move(batch))) {
        ctx.start_render = true;
    }
}

nlohmann::json acousticSurfaceStatus(UIContext& ctx) {
    const auto workspace = ctx.scene.audio->invoke("workspace_get", nlohmann::json::object());
    auto* backend = dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr);
    return {{"view", kAcousticSurfaceDebugView},
        {"active", ctx.render_settings.debug_view == kAcousticSurfaceDebugView},
        {"all_sources", workspace.value("acoustic_all_sources", false)},
        {"band", workspace.value("acoustic_band", -1)},
        {"reflection_samples", workspace.value("acoustic_reflection_samples", 1)},
        {"model", "six_band_direct_and_first_bounce"},
        {"specular_model", "bounded_image_source_candidates"},
        {"phase_resolved", false}, {"refraction", false}, {"diffraction", false},
        {"source_budget", 4}, {"barrier_budget", 32}, {"candidate_budget", 256},
        {"max_queries_per_pixel", 30},
        {"calibrated_spl", false}, {"db_range", {-60, 0}},
        {"extra_gpu_readback_bytes", 0}, {"temporal_accumulation", false},
        {"error", backend ? backend->acousticSurfaceError() : "Vulkan RT backend required"}};
}
} // namespace audio
