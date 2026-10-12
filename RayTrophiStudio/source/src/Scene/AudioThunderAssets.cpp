#include "Audio/AudioEnvironment.h"
#include "Audio/AudioService.h"
#include <fstream>
#include <stdexcept>

namespace audio {
nlohmann::json installThunder(AudioService& candidate, const std::filesystem::path& assetsRoot) {
    using json = nlohmann::json;
    const auto existing = candidate.snapshot().value("thunder", json::object());
    if (!existing.value("near_assets", json::array()).empty() &&
        !existing.value("far_assets", json::array()).empty()) {
        return candidate.invoke("thunder_update", {{"config", {{"enabled", true}}}});
    }
    const auto folder = assetsRoot / "audio" / "cc0_thunder";
    json nearAssets = json::array();
    json farAssets = json::array();
    for (const std::string index : {"01", "02", "03", "04", "06", "07", "08"}) {
        const std::string stem = "thunder_" + index + "_mono";
        std::ifstream metadata(folder / (stem + ".asset.json"));
        if (!metadata) {
            throw std::runtime_error("CC0 thunder pack not found; choose its Assets root");
        }
        json record;
        metadata >> record;
        const auto asset = candidate.invoke("asset_import", {
            {"path", pathUtf8(folder / (stem + ".wav"))},
            {"license", record.at("license")}, {"author", record.at("author")},
            {"source", record.at("source")}, {"tags", record.at("tags")}});
        const bool near = index == "06" || index == "07" || index == "08";
        (near ? nearAssets : farAssets).push_back(asset.at("id"));
    }
    return candidate.invoke("thunder_update", {{"config", {
        {"enabled", true}, {"near_assets", nearAssets}, {"far_assets", farAssets}}}});
}
}
