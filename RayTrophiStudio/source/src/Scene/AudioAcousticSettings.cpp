#include "Audio/AudioAcousticSettings.h"
#include "Audio/AudioScope.h"
#include <stdexcept>
#include <utility>

namespace audio {
namespace {
using json = nlohmann::json;
void require(bool valid, const char* message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
json& source(json& scene, const json& id) {
    for (auto& emitter : scene.at("emitters")) {
        if (emitter.at("id") == id) {
            return emitter;
        }
    }
    throw std::invalid_argument("Unknown acoustic source id");
}
json effective(const json& scene, const json& emitter) {
    const auto& policy = emitter.at("acoustics");
    auto config = policy.at("mode") == "custom" ? policy.at("config") : scene.at("acoustics");
    if (policy.at("mode") == "off" || !emitter.at("spatial").get<bool>()) {
        config["enabled"] = false;
    }
    // Inherited settings remain useful for directional sources: only the
    // unsupported diffuse component is excluded. Authored defaults are preserved.
    if (policy.at("mode") == "scene" && !supportsDiffuseSource(emitter)) {
        config["diffuse"] = false;
    }
    return config;
}
json view(const json& scene) {
    json entries = json::array();
    for (const auto& emitter : scene.at("emitters")) {
        entries.push_back({{"id", emitter.at("id")}, {"policy", emitter.at("acoustics")},
            {"effective", effective(scene, emitter)},
            {"spatial_eligible", emitter.at("spatial")},
            {"diffuse_eligible", supportsDiffuseSource(emitter)}});
    }
    return {{"config", scene.at("acoustics")}, {"sources", entries}};
}
}

nlohmann::json validateAcousticSettings(const json& input) {
    require(input.is_object(), "Acoustic settings must be an object");
    json config = {{"enabled", false}, {"meters_per_unit", 1.0},
        {"boundary_mode", "object_barrier"}, {"spectral", true},
        {"reflections", false}, {"diffuse", false}};
    for (auto field = input.begin(); field != input.end(); ++field) {
        require(config.contains(field.key()), "Unknown acoustic settings field");
        config[field.key()] = field.value();
    }
    require(config.at("enabled").is_boolean(), "Acoustic enabled must be boolean");
    auto validation = config;
    validation["enabled"] = true;
    AcousticPreview validator;
    validator.configure({{"id", "validation"}, {"spatial", true},
        {"cone_inner", 360}, {"cone_outer", 360}}, validation);
    return config;
}

nlohmann::json validateSourceAcoustics(const json& input) {
    require(input.is_object(), "Source acoustics must be an object");
    for (auto field = input.begin(); field != input.end(); ++field) {
        require(field.key() == "mode" || field.key() == "config",
            "Unknown source acoustic policy field");
    }
    const auto mode = input.value("mode", json("scene"));
    require(mode.is_string() && (mode == "scene" || mode == "custom" || mode == "off"),
        "Source acoustic mode must be scene, custom or off");
    return {{"mode", mode},
        {"config", validateAcousticSettings(input.value("config", json::object()))}};
}

void reconcileAcousticSettings(json& scene, AcousticPreview& preview) {
    scene["acoustics"] = validateAcousticSettings(scene.value("acoustics", json::object()));
    auto staged = preview;
    const auto previous = preview.status();
    // Remove disabled entries first so replacement never temporarily exceeds budget.
    for (const auto& entry : previous) {
        bool keep = false;
        for (const auto& emitter : scene.at("emitters")) {
            if (emitter.at("id") == entry.at("id")) {
                keep = effective(scene, emitter).at("enabled").get<bool>();
            }
        }
        if (!keep) {
            staged.forget(entry.at("id"));
        }
    }
    for (auto& emitter : scene.at("emitters")) {
        emitter["acoustics"] = validateSourceAcoustics(
            emitter.value("acoustics", json::object()));
        const auto config = effective(scene, emitter);
        if (!config.at("enabled").get<bool>()) {
            continue;
        }
        require(emitter.at("acoustics").at("mode") != "custom" ||
            !config.at("diffuse").get<bool>() || supportsDiffuseSource(emitter),
            "Custom diffuse processing requires an omnidirectional source");
        bool same = false;
        for (const auto& entry : previous) {
            if (entry.at("id") != emitter.at("id")) {
                continue;
            }
            same = true;
            for (auto field = config.begin(); field != config.end(); ++field) {
                same = same && entry.contains(field.key()) && entry.at(field.key()) == field.value();
            }
        }
        if (!same) {
            staged.configure(emitter, config);
        }
    }
    preview = std::move(staged);
}

AcousticPreview stageAcousticEmitter(const json& scene, const AcousticPreview& preview,
                                    const json& emitter) {
    auto candidate = scene;
    bool replaced = false;
    for (auto& item : candidate.at("emitters")) {
        if (item.at("id") == emitter.at("id")) {
            item = emitter;
            replaced = true;
        }
    }
    if (!replaced) {
        candidate["emitters"].push_back(emitter);
    }
    auto staged = preview;
    reconcileAcousticSettings(candidate, staged);
    return staged;
}

nlohmann::json acousticSettingsOperation(json& scene, AcousticPreview& preview,
                                        const std::string& operation, const json& params) {
    if (operation == "acoustics_get") {
        return view(scene);
    }
    auto candidate = scene;
    auto staged = preview;
    json ids = json::array();
    if (operation == "acoustics_update") {
        require(params.contains("config") && params.at("config").is_object(),
            "Scene acoustic update requires config object");
        auto config = candidate.at("acoustics");
        config.update(params.at("config"));
        candidate["acoustics"] = validateAcousticSettings(config);
    } else if (operation == "acoustics_reset_sources") {
        ids = resolveAudioScope(candidate.at("emitters"), params);
        for (const auto& id : ids) {
            source(candidate, id)["acoustics"]["mode"] = "scene";
        }
    } else if (operation == "acoustics_preview") {
        require(params.contains("id"), "Acoustic source configuration requires id");
        auto config = params;
        config.erase("id");
        require(config.contains("enabled"), "Acoustic source configuration requires enabled");
        config = validateAcousticSettings(config);
        auto& emitter = source(candidate, params.at("id"));
        require(!config.at("enabled").get<bool>() || emitter.at("spatial").get<bool>(),
            "Custom acoustic processing requires a spatial source");
        emitter["acoustics"] = {{"mode", "custom"}, {"config", config}};
        ids.push_back(params.at("id"));
    } else {
        require(operation == "acoustics_batch", "Unknown acoustic settings operation");
        if (params.value("action", json()) != "configure") {
            return preview.batch(scene.at("emitters"), params);
        }
        require(params.contains("config"), "Acoustic batch configuration requires config");
        require(params.at("config").is_object() && params.at("config").contains("enabled"),
            "Acoustic batch configuration requires enabled");
        const auto config = validateAcousticSettings(params.at("config"));
        ids = resolveAudioScope(candidate.at("emitters"), params);
        for (const auto& id : ids) {
            auto& emitter = source(candidate, id);
            require(!config.at("enabled").get<bool>() || emitter.at("spatial").get<bool>(),
                "Custom acoustic processing requires spatial sources");
            emitter["acoustics"] = {{"mode", "custom"}, {"config", config}};
        }
    }
    reconcileAcousticSettings(candidate, staged);
    scene = std::move(candidate);
    preview = std::move(staged);
    if (operation == "acoustics_preview") {
        for (const auto& entry : preview.status()) {
            if (entry.at("id") == params.at("id")) {
                return entry;
            }
        }
        return {{"id", params.at("id")}, {"enabled", false}, {"amplitude_gain", 1}};
    }
    if (operation == "acoustics_batch") {
        json entries = json::array();
        const auto runtime = preview.status();
        for (const auto& id : ids) {
            json result = {{"id", id}, {"enabled", false}, {"amplitude_gain", 1}};
            for (const auto& entry : runtime) {
                if (entry.at("id") == id) {
                    result = entry;
                    break;
                }
            }
            entries.push_back(std::move(result));
        }
        return {{"ok", true}, {"ids", ids}, {"action", "configure"},
            {"scope", params.value("scope", "selected")}, {"entries", entries}};
    }
    return view(scene);
}
}
