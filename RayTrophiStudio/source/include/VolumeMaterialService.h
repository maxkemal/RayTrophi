#pragma once
#include "json.hpp"
#include <string>
struct UIContext;
namespace volume_material {
nlohmann::json execute(UIContext& ctx, const std::string& operation,
    const std::string& target, const nlohmann::json& patch = nlohmann::json::object());
}
