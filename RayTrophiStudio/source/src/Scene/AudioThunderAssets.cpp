#include "Audio/AudioEnvironment.h"
#include "Audio/AudioService.h"
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace audio {
nlohmann::json installThunder(AudioService& candidate, const std::filesystem::path& assetsRoot) {
    using json = nlohmann::json;
    const auto existing = candidate.snapshot().value("thunder", json::object());
    if (!existing.value("near_assets", json::array()).empty() &&
        !existing.value("far_assets", json::array()).empty()) {
        return candidate.invoke("thunder_update", {{"config", {{"enabled", true}}}});
    }
    const auto stormFolder = assetsRoot / "audio" / "cc0_storm_thunder";
    const bool stormPack = std::filesystem::exists(stormFolder / "manifest.json");
    const auto folder = stormPack ? stormFolder : assetsRoot / "audio" / "cc0_thunder";
    const std::vector<std::pair<std::string, bool>> variants = stormPack
        ? std::vector<std::pair<std::string, bool>>{
            {"storm_crack_tail_a_mono", true}, {"storm_crack_tail_b_mono", true},
            {"storm_deep_roll_a_mono", false}, {"storm_deep_roll_b_mono", false}}
        : std::vector<std::pair<std::string, bool>>{
            {"thunder_01_mono", false}, {"thunder_02_mono", false},
            {"thunder_03_mono", false}, {"thunder_04_mono", false},
            {"thunder_06_mono", true}, {"thunder_07_mono", true}, {"thunder_08_mono", true}};
    json nearAssets = json::array();
    json farAssets = json::array();
    for (const auto& [stem, near] : variants) {
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
        (near ? nearAssets : farAssets).push_back(asset.at("id"));
    }
    return candidate.invoke("thunder_update", {{"config", {
        {"enabled", true}, {"near_assets", nearAssets}, {"far_assets", farAssets}}}});
}
}
