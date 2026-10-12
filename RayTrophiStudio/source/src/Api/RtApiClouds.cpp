/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiClouds.cpp
* License:       MIT
* =========================================================================
*
* Cloud authority surface (docs/dev/ATMOSPHERE_CLOUDS.md §6).
*
* JSON in, JSON out, and the schema is atmosphere::cloudsToJson -- ONE
* definition. A field-by-field mirror struct here would be a second copy that
* drifts the first time a field is added.
*/

#include "RtApiInternal.h"

#include <string>

#include "Atmosphere/AtmosphereClouds.h"
#include "Backend/VulkanBackend.h"
#include "ProjectManager.h"
#include "json.hpp"

#include <vector>

extern void markWorldDirty();
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;

namespace rtapi {

namespace {

// Merge `patch` into `cur`. Objects merge recursively (RFC 7386); the
// `layers` array merges ELEMENT-WISE, so {"layers":[{}, {"coverage":0.4}]}
// edits layer 1 only. A plain merge-patch would replace the whole array.
void mergeCloudPatch(nlohmann::json& cur, const nlohmann::json& patch) {
    for (auto it = patch.begin(); it != patch.end(); ++it) {
        if (it.key() == "layers" && it.value().is_array() && cur["layers"].is_array()) {
            auto& layers = cur["layers"];
            const auto& p = it.value();
            for (size_t i = 0; i < p.size() && i < layers.size(); ++i) {
                if (p[i].is_object()) layers[i].merge_patch(p[i]);
            }
        } else {
            cur[it.key()].merge_patch(it.value());
        }
    }
}

Result commit(const atmosphere::CloudState& next) {
    std::string error;
    if (!g_ctx->renderer.world.setClouds(next, &error)) return Result::fail(error);
    markWorldDirty();
    resetAccumulation();
    ProjectManager::getInstance().markModified();
    return Result::success();
}

} // namespace

Result getWeatherJson(std::string& out_json) {
    if (!g_ctx) return notBound();
    const World& w = g_ctx->renderer.world;
    nlohmann::json j, climate, derived;
    atmosphere::climateToJson(w.getClimate(), climate);
    atmosphere::derivedWeatherToJson(w.derivedWeather(), derived);
    j["climate"] = climate;
    j["derived"] = derived;
    j["derive_from_climate"] = w.getClouds().derive_from_climate;
    out_json = j.dump();
    return Result::success();
}

Result getCloudsJson(std::string& out_json) {
    if (!g_ctx) return notBound();
    const World& w = g_ctx->renderer.world;
    nlohmann::json j;
    atmosphere::cloudsToJson(w.getClouds(), j);
    // Read-only context so a script sees what the renderer is fed.
    const Vec3 drift = w.cloudWindOffset();
    j["time_seconds"] = w.getCloudTime();
    j["lightning_effective_rate_per_minute"] = Backend::lightningEffectiveRate(
        w.getClouds().lightning);
    j["wind_offset_m"] = {drift.x, drift.z};
    j["revision"] = w.cloudRevision();
    j["any_enabled"] = w.getClouds().anyEnabled();
    j["presets"] = atmosphere::cloudPresetNames();
    out_json = j.dump();
    return Result::success();
}

Result setCloudsJson(const std::string& patch_json) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    nlohmann::json patch;
    try {
        patch = nlohmann::json::parse(patch_json);
    } catch (const std::exception& e) {
        return Result::fail(std::string("patch is not valid JSON: ") + e.what());
    }
    if (!patch.is_object()) return Result::fail("patch must be a JSON object");
    // Read-only keys from get are accepted and ignored, so get -> edit -> set
    // round-trips without the caller having to strip them.
    for (const char* ro : {"time_seconds", "wind_offset_m", "revision", "any_enabled", "presets"})
        patch.erase(ro);
    patch.erase("lightning_effective_rate_per_minute");

    // storms.freeze_automatic: an action, not a field -- the live automatic
    // cells become authored ones (atmosphere::freezeStormCells), after the
    // rest of the patch is applied.
    bool freeze = false;
    if (patch.contains("storms") && patch["storms"].is_object() &&
        patch["storms"].contains("freeze_automatic")) {
        const auto& f = patch["storms"]["freeze_automatic"];
        if (!f.is_boolean()) return Result::fail("storms.freeze_automatic must be a boolean");
        freeze = f.get<bool>();
        patch["storms"].erase("freeze_automatic");
    }

    nlohmann::json cur;
    atmosphere::cloudsToJson(g_ctx->renderer.world.getClouds(), cur);
    mergeCloudPatch(cur, patch);
    atmosphere::CloudState next;
    std::string error;
    if (!atmosphere::cloudsFromJson(cur, next, &error)) return Result::fail(error);
    if (freeze) {
        const World& world = g_ctx->renderer.world;
        atmosphere::freezeStormCells(next, world.getCloudTime(), world.cloudWindVelocity());
        if (!atmosphere::validateClouds(next, &error)) return Result::fail(error);
    }
    return commit(next);
}

Result applyCloudPreset(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    atmosphere::CloudState preset;
    if (!atmosphere::cloudPreset(name, preset)) {
        std::string list;
        for (const auto& n : atmosphere::cloudPresetNames()) list += (list.empty() ? "" : "|") + n;
        return Result::fail("unknown cloud preset '" + name + "' (" + list + ")");
    }
    // A look preset keeps the quality settings, like the panel.
    preset.quality = g_ctx->renderer.world.getClouds().quality;
    preset.material = g_ctx->renderer.world.getClouds().material;
    return commit(preset);
}

Result cloudStatsJson(std::string& out_json) {
    if (!g_ctx) return notBound();
    nlohmann::json rows = nlohmann::json::array();
    // Same role rule as atmosphereStats: name both ROLES, even when both point
    // at one object -- "one device or two" is part of the answer.
    auto describe = [&](const char* role, Backend::IBackend* backend) {
        nlohmann::json r;
        r["role"] = role;
        auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(backend);
        r["is_vulkan"] = vk != nullptr;
        r["present"] = backend != nullptr;
        if (vk) {
            r["resources_available"] = vk->cloudResourcesAvailable();
            r["noise_ready"] = vk->cloudNoiseReady();
            r["weather_ready"] = vk->cloudWeatherReady();
            r["noise_generations"] = vk->cloudNoiseGenerations();
            r["weather_generations"] = vk->cloudWeatherGenerations();
            r["sample_dispatches"] = vk->cloudSampleDispatches();
            r["state_updates"] = vk->cloudStateRevisionsSeen();
            // The flag the RT shaders read: false while textures build, in a
            // non-Nishita world, or with every layer off.
            r["rt_rendering"] = vk->cloudRtRendering();
            r["scene_depth_pipeline_available"] = vk->cloudSceneDepthAvailable();
            // Lightning as the shaders receive it at this instant: channel,
            // growth, pulse radiance, and the lit cloud base that lights the
            // ground and rain (flash_*; radius 0 = not lighting anything,
            // e.g. during the stepped leader). Events are time-driven, so
            // poll repeatedly to catch one.
            const Backend::CloudParamsGPU params = vk->cloudParamsGPU();
            nlohmann::json events = nlohmann::json::array();
            for (int i = 0; i < Backend::kCloudLightningEvents; ++i) {
                const auto& e = params.lightning[i];
                const auto& f = params.flash[i];
                // Channel network: [a xyz, b xyz, weight (1 main), leader
                // progress at which it starts to grow].
                nlohmann::json segments = nlohmann::json::array();
                const int count = (std::min)(static_cast<int>(e.color[3] + 0.5f),
                                             Backend::kCloudBoltSegments);
                for (int k = 0; k < count; ++k) {
                    const auto& s = params.bolt[i * Backend::kCloudBoltSegments + k];
                    segments.push_back({{s.a[0], s.a[1], s.a[2]}, {s.b[0], s.b[1], s.b[2]},
                                        s.a[3], s.b[3]});
                }
                events.push_back({
                    {"active", e.start[3] > 0.0f},
                    {"pulse_radiance", e.start[3]},
                    {"start", {e.start[0], e.start[1], e.start[2]}},
                    {"end", {e.end[0], e.end[1], e.end[2]}},
                    {"growth", e.stage[0]},
                    {"ground_strike", e.stage[2] < 0.5f || e.stage[2] > 1.5f},
                    {"kind", e.stage[2] > 1.5f ? "upward" : (e.stage[2] < 0.5f ? "ground" : "cloud")},
                    {"glow_radius_m", e.stage[3]},
                    {"flash_position", {f.position[0], f.position[1], f.position[2]}},
                    {"flash_radius_m", f.position[3]},
                    {"flash_radiance", {f.radiance[0], f.radiance[1], f.radiance[2]}},
                    {"channel_brightness", e.stage[1]},
                    {"segments", segments}});
            }
            r["lightning"] = events;
        }
        rows.push_back(r);
    };
    describe("render", ::g_backend.get());
    describe("viewport", ::g_viewport_backend.get());
    nlohmann::json j;
    j["backends"] = rows;
    j["revision"] = g_ctx->renderer.world.cloudRevision();
    j["any_enabled"] = g_ctx->renderer.world.getClouds().anyEnabled();
    {
        // Live storm cells (ATMOSPHERE_STORMS.md): exactly what the GPU cell
        // map and the lightning placement read at this cloud time.
        const World& world = g_ctx->renderer.world;
        atmosphere::StormCellState cells[atmosphere::kMaxStormCells];
        const int n = atmosphere::evaluateStormCells(world.getClouds(), world.getCloudTime(),
                                                     world.cloudWindVelocity(), cells);
        nlohmann::json list = nlohmann::json::array();
        for (int i = 0; i < n; ++i) {
            const auto& c = cells[i];
            const char* phase = c.life < 0.25f ? "growing" : (c.life < 0.7f ? "mature" : "dissipating");
            list.push_back({{"position_m", {c.x, 0.0f, c.z}}, {"radius_m", c.radius_m},
                            {"life", c.life}, {"phase", phase}, {"coverage", c.coverage},
                            {"tower", c.tower}, {"convective_rain", c.convective_rain},
                            {"stratiform_rain", c.stratiform_rain}, {"lightning", c.lightning},
                            {"manual", c.manual}, {"manual_index", c.manual_index},
                            {"age_s", c.age_s}});
        }
        j["storms"] = {{"enabled", world.getClouds().storms.enabled},
                       {"cloud_time_s", world.getCloudTime()},
                       {"cells", list}};
    }
    // Stated, not implied: Vulkan RT path-traces the cloud field (Faz 3b);
    // the frozen OptiX path still draws the legacy procedural volume fed by
    // the derived nishita.cloud_* packet (decision a).
    j["renderer"] = {{"vulkan_rt", "path_traced"}, {"optix", "legacy_volume_mapping"}};
    out_json = j.dump();
    return Result::success();
}

Result sampleCloudsJson(const std::string& request_json, std::string& out_json) {
    if (!g_ctx) return notBound();
    nlohmann::json req;
    try {
        req = nlohmann::json::parse(request_json);
    } catch (const std::exception& e) {
        return Result::fail(std::string("request is not valid JSON: ") + e.what());
    }
    const std::string mode = req.value("mode", std::string("density"));
    uint32_t modeId = 0;
    if (mode == "density") modeId = 0;
    else if (mode == "transmittance") modeId = 1;
    else if (mode == "base_density") modeId = 2;
    else if (mode == "precipitation") modeId = 3;   // mm/h at points
    else return Result::fail("mode must be density|transmittance|base_density|precipitation");
    const int steps = req.value("steps", 256);
    if (steps < 1 || steps > 100000) return Result::fail("steps must be within 1..100000");

    auto readVec = [](const nlohmann::json& v, float out[4]) {
        if (!v.is_array() || v.size() != 3) return false;
        for (int k = 0; k < 3; ++k) {
            if (!v[k].is_number()) return false;
            out[k] = v[k].get<float>();
        }
        out[3] = 0.0f;
        return true;
    };
    std::vector<Backend::CloudQueryGPU> queries;
    if (modeId == 1) {
        if (!req.contains("segments") || !req["segments"].is_array())
            return Result::fail("transmittance needs segments: [[[ax,ay,az],[bx,by,bz]], ...]");
        for (const auto& s : req["segments"]) {
            Backend::CloudQueryGPU q{};
            if (!s.is_array() || s.size() != 2 || !readVec(s[0], q.a) || !readVec(s[1], q.b))
                return Result::fail("each segment is [[ax,ay,az],[bx,by,bz]]");
            queries.push_back(q);
        }
    } else {
        if (!req.contains("points") || !req["points"].is_array())
            return Result::fail(mode + " needs points: [[x,y,z], ...]");
        for (const auto& p : req["points"]) {
            Backend::CloudQueryGPU q{};
            if (!readVec(p, q.a)) return Result::fail("each point is [x,y,z]");
            queries.push_back(q);
        }
    }
    if (queries.empty()) return Result::fail("no points/segments given");
    if (queries.size() > 65536) return Result::fail("at most 65536 queries per call");

    const std::string role = req.value("backend", std::string("render"));
    Backend::IBackend* backend = role == "viewport"
        ? static_cast<Backend::IBackend*>(::g_viewport_backend.get()) : ::g_backend.get();
    auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(backend);
    if (!vk) return Result::fail("backend '" + role + "' is not a Vulkan adapter (clouds are Vulkan-only)");

    // Push the CURRENT authority first: the adapter's copy only refreshes on
    // the next world sync, and set_clouds -> sample_clouds in one script must
    // measure the new clouds, not last frame's.
    const World& w = g_ctx->renderer.world;
    vk->setCloudState(w.getClouds(), w.getCloudTime(), w.cloudWindOffset(), w.cloudWindVelocity());

    std::vector<float> values;
    std::string why;
    if (!vk->sampleClouds(modeId, static_cast<uint32_t>(steps), queries, values, why))
        return Result::fail(why);
    nlohmann::json j;
    j["mode"] = mode;
    j["backend"] = role;
    j["values"] = values;
    out_json = j.dump();
    return Result::success();
}

} // namespace rtapi
