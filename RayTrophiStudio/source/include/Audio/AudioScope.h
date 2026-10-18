#pragma once
#include "json.hpp"
#include <set>
#include <stdexcept>
#include <string>

namespace audio {
inline nlohmann::json resolveAudioScope(const nlohmann::json& emitters,
                                        const nlohmann::json& params) {
    using json = nlohmann::json;
    const auto scope = params.value("scope", json("selected"));
    const auto ids = params.value("ids", json::array());
    if (!scope.is_string() || (scope != "selected" && scope != "group" && scope != "scene") ||
        !ids.is_array()) {
        throw std::invalid_argument("scope must be selected, group or scene; ids must be an array");
    }
    json result = json::array();
    std::set<std::string> known;
    for (const auto& source : emitters) {
        known.insert(source.at("id").get<std::string>());
        if (scope == "scene") {
            result.push_back(source.at("id"));
        }
    }
    if (scope == "scene") {
        if (!ids.empty()) {
            throw std::invalid_argument("scene scope must omit ids");
        }
        return result;
    }
    if (ids.empty() || ids.size() > 32 || (scope == "selected" && ids.size() != 1)) {
        throw std::invalid_argument("selected requires one ID; group requires 1..32 unique IDs");
    }
    std::set<std::string> seen;
    for (const auto& id : ids) {
        if (!id.is_string() || !known.contains(id.get<std::string>()) ||
            !seen.insert(id.get<std::string>()).second) {
            throw std::invalid_argument("Scope IDs must be unique existing source IDs");
        }
        result.push_back(id);
    }
    return result;
}
}
