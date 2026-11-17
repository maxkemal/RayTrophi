#pragma once

#include "json.hpp"
#include <string>

namespace AudioUI {

// Keep authored names intact, while making even legacy duplicate names distinct.
inline std::string sourceLabel(const nlohmann::json& source,
                               const nlohmann::json& sceneAudio) {
    const auto id = source.value("id", std::string());
    auto label = source.value("name", std::string());
    if (label.empty()) {
        label = "Audio Source";
    }
    const auto assetId = source.value("asset_id", std::string());
    if (!assetId.empty() && sceneAudio.contains("assets")) {
        for (const auto& asset : sceneAudio.at("assets")) {
            if (asset.value("id", std::string()) != assetId) {
                continue;
            }
            const auto assetName = asset.value("name", std::string());
            if (!assetName.empty() && assetName != label) {
                label += " - " + assetName;
            }
            break;
        }
    }
    return label + " [" + id + "]";
}

} // namespace AudioUI
