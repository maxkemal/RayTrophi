#include "Audio/AudioEnvironment.h"
#include "Audio/AudioService.h"
#include <fstream>
#include <stdexcept>

namespace audio {
nlohmann::json installEnvironment(AudioService& candidate, const std::filesystem::path& assetsRoot) {
    using json = nlohmann::json;
    json result = json::object();
    const auto folder = assetsRoot / "audio" / "cc0_starter";
    for (const std::string role : {"wind", "rain"}) {
        const auto data = candidate.snapshot();
        json source;
        for (const auto& e : data.at("emitters")) {
            if (e.value("environment_role", std::string()) == role) {
                if (!source.is_null()) {
                    throw std::invalid_argument("Duplicate environment role: " + role);
                }
                source = e;
            }
        }
        if (source.is_null()) {
            const std::string stem = role == "wind" ? "wind_gust_stereo" : "rain_loop_stereo";
            std::ifstream metadata(folder / (stem + ".asset.json"));
            if (!metadata) {
                throw std::runtime_error("CC0 starter pack not found; choose its Assets root");
            }
            json record;
            metadata >> record;
            const auto asset = candidate.invoke("asset_import", {
                {"path", pathUtf8(folder / (stem + ".wav"))},
                {"license", record.value("license", std::string())},
                {"author", record.value("author", std::string())},
                {"source", record.value("source", std::string())},
                {"tags", record.value("tags", json::array())}});
            source = candidate.invoke("create", {{"asset_id", asset.at("id")},
                {"name", "Environment " + role}, {"environment_role", role},
                {"spatial", false}, {"loop", true}, {"bus", "Weather"},
                {"reaction", {{"driver", role}, {"automatic", true}}}});
        } else {
            auto reaction = source.at("reaction");
            reaction["driver"] = role;
            reaction["automatic"] = true;
            source = candidate.invoke("update", {{"id", source.at("id")},
                {"loop", true}, {"reaction", reaction}});
        }
        result[role + "_source_id"] = source.at("id");
    }
    return result;
}
}
