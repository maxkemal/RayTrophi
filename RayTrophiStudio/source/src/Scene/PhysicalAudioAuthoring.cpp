#include "Audio/PhysicalAudioAuthoring.h"
#include "Audio/PhysicalAudioSource.h"
#include "Audio/PhysicalAudioTimeline.h"
#include "Audio/AudioReaction.h"
#include <stdexcept>
#include <string>

namespace audio {
namespace {
using json = nlohmann::json;
}

json physicalGeneratorCatalogue() {
    json models = json::array();
    for (const auto* name : {"liquid_flow", "gas_flow", "friction", "rolling", "impact",
                            "contact_activity", "deformation", "fracture_activity"}) {
        const std::string model(name);
        const bool resonant = model == "impact" || model == "contact_activity" ||
            model == "deformation" || model == "fracture_activity";
        const char* label = model == "liquid_flow" ? "Liquid / flow" :
            model == "gas_flow" ? "Gas / airflow" : model == "friction" ? "Friction" :
            model == "rolling" ? "Rolling" : model == "impact" ? "Impact" :
            model == "contact_activity" ? "Contact texture" :
            model == "deformation" ? "Deformation" : "Fracture texture";
        models.push_back({{"model", model}, {"label", label}, {"resonant", resonant},
            {"supports_surface", model == "friction"},
            {"default_generator", {{"version", 2}, {"character_revision", 2},
                {"model", model}, {"seed", 1}, {"brightness", 0.5}, {"texture", 0.5}, {"variation", 0.5}, {"body", 0.5}}}});
    }
    return {{"versions", {1, 2}}, {"default_version", 1}, {"recommended_version", 2},
        {"character_revisions", {1, 2}}, {"default_character_revision", 1},
        {"recommended_character_revision", 2},
        {"friction_surfaces", {"organic", "mineral", "metal", "glass"}},
        {"models", models}, {"sample_rate", 48000}, {"channels", 1},
        {"quality_scope", "artistic_activity_sonification"},
        {"measured_discrete_events", false},
        {"controls_v2", {{"texture", {0, 1}}, {"variation", {0, 1}}, {"body", {0, 1}}}},
        {"resonators", {"wood", "metal", "stone", "glass"}},
        {"legacy_preserved", true}, {"shared_live_replay_bake", true}};
}

json physicalGeneratorPatch(const json& source, const json& params) {
    if (!params.contains("generator") || !params.at("generator").is_object()) {
        throw std::invalid_argument("physical_configure requires a generator object");
    }
    auto candidate = source;
    candidate["generator"] = params.at("generator");
    candidate["asset_id"] = "";
    candidate["loop"] = true;
    if (params.contains("reaction")) {
        candidate["reaction"] = AudioReaction::validate(params.at("reaction"));
    } else if (candidate["generator"].value("model", json("liquid_flow")) == "impact") {
        candidate["reaction"] = AudioReaction::validate(json::object());
    } else if (candidate.at("reaction").at("driver") == "manual" ||
               candidate.at("reaction").at("driver") == "wind" ||
               candidate.at("reaction").at("driver") == "rain") {
        // An explicit conversion never invents domain measurements or enables telemetry.
        candidate["reaction"] = AudioReaction::validate({{"driver", "signal"},
            {"signal_key", "physical/" + source.at("id").get<std::string>()}});
    }
    if (params.value("clear_history", false)) {
        candidate["generator_history"] = nullptr;
    }
    validatePhysicalAudioSource(candidate);
    validateStoredPhysicalTimeline(candidate);
    json patch = {{"id", source.at("id")}, {"asset_id", ""}, {"loop", true},
        {"generator", candidate.at("generator")}, {"reaction", candidate.at("reaction")}};
    if (params.value("clear_history", false)) {
        patch["generator_history"] = nullptr;
    }
    return patch;
}
}
