#include "Audio/AudioThunderBridge.h"
#include "Audio/AudioService.h"
#include "Backend/CloudParams.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "renderer.h"
#include "Camera.h"
#include "Api/RtApi.h"
#include <algorithm>
#include <cmath>
#include <memory>

namespace audio {
void tickCloudThunder(UIContext& ctx) {
    // Observe the authoritative visual clock once per editor update. Geometry and storm-cell
    // selection come from the same CPU producer used by the renderer, never from RT samples.
    static std::weak_ptr<AudioService> owner;
    static std::uint64_t generation = 0;
    static double previousTime = 0;
    static bool previousPlaying = false;
    auto service = ctx.scene.audio;
    const auto& world = ctx.renderer.world;
    const auto& clouds = world.getClouds();
    const float fps = static_cast<float>(std::max(1, ctx.render_settings.animation_fps));
    const double time = static_cast<float>(std::max(0, ctx.scene.timeline.current_frame)) / fps;
    const float rate = Backend::lightningEffectiveRate(clouds.lightning);
    const double period = rate > 0 ? 240.0 / rate : 0;
    const bool playing = ctx.scene_ui_ptr && ctx.scene_ui_ptr->timeline.isPlaying();
    const bool stoppedScrub = !playing && !previousPlaying && time != previousTime;
    previousPlaying = playing;
    if (!std::isfinite(time) || !ctx.scene.camera) {
        service->resetThunder();
        owner.reset();
        return;
    }
    const bool reset = owner.lock() != service || generation != service->generation() ||
        time < previousTime || time - previousTime > 2 ||
        stoppedScrub;
    if (reset) {
        service->resetThunder();
        owner = service;
        generation = service->generation();
        previousTime = time;
    }
    const auto config = service->snapshot().at("thunder");
    if (!reset && time > previousTime && period > 0 && clouds.lightning.enabled &&
        config.at("enabled").get<bool>() && world.data.mode == WORLD_MODE_NISHITA &&
        !rtapi::renderOutputPending()) {
        const auto camera = ctx.scene.camera->lookfrom;
        const auto wind = world.cloudWindVelocity();
        for (int lane = 0; lane < Backend::kCloudLightningEvents; ++lane) {
            const double offset = period * lane / Backend::kCloudLightningEvents;
            const double first = std::floor((previousTime - offset) / period);
            const double last = std::floor((time - offset) / period);
            for (double slot = first; slot <= last; slot += 1) {
                const double wrapped = slot - std::floor(slot / 4294967296.0) * 4294967296.0;
                std::uint32_t seed = Backend::lightningHash(static_cast<std::uint32_t>(wrapped) ^
                    clouds.weather.seed ^ Backend::lightningHash(static_cast<std::uint32_t>(lane + 1)));
                // Mirror only the producer's lane clock; all eligibility, cell selection and
                // bolt geometry are evaluated by makeSharedCloudParamsGPU below.
                const double onset = offset + slot * period +
                    period * (0.03 + 0.07 * Backend::lightningRandom(seed));
                const float size = 0.65f + Backend::lightningRandom(seed) * 0.70f;
                if (onset <= previousTime || onset > time) {
                    continue;
                }
                const float probeAge = clouds.lightning.duration_s * size * 0.5f;
                const float sampleTime = static_cast<float>(onset + probeAge);
                const auto packet = Backend::makeSharedCloudParamsGPU(clouds, sampleTime,
                    world.cloudWindOffset(), wind, world.data, sampleTime);
                const auto& event = packet.lightning[lane];
                if (event.start[3] <= 0 || event.color[3] <= 0) {
                    continue;
                }
                // Closest point on the channel trunk is a bounded approximation to the first
                // audible wavefront. Freeze it at onset; later camera movement affects pan/gain.
                const Vec3 start(event.start[0] - wind.x * probeAge, event.start[1],
                                 event.start[2] - wind.z * probeAge);
                const Vec3 end(event.end[0] - wind.x * probeAge, event.end[1],
                               event.end[2] - wind.z * probeAge);
                const Vec3 delta = end - start;
                const float squared = delta.dot(delta);
                const float u = squared > 0 ?
                    std::clamp((camera - start).dot(delta) / squared, 0.0f, 1.0f) : 0;
                const Vec3 position = start + delta * u;
                const std::string key = "cloud/" + std::to_string(clouds.weather.seed) + "/" +
                    std::to_string(period) + "/" + std::to_string(lane) + "/" +
                    std::to_string(static_cast<long long>(slot));
                try {
                    service->invoke("thunder_event", {{"key", key},
                        {"position", nlohmann::json::array({position.x, position.y, position.z})},
                        {"intensity", clouds.lightning.intensity},
                        {"age_seconds", std::clamp(time - onset, 0.0, 2.0)}});
                } catch (const std::exception&) {
                    // A full bounded queue may drop new events; the editor must keep rendering.
                }
            }
        }
    }
    previousTime = time;
    service->tickThunder();
}
}
