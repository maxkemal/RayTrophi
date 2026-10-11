#pragma once
#include "json.hpp"
#include <string>

struct UIContext;
class SceneHistory;

namespace audio {
nlohmann::json author(UIContext& ctx, SceneHistory* history, const std::string& operation,
                      const nlohmann::json& params);
void recordEdit(UIContext& ctx, SceneHistory* history, const nlohmann::json& before,
                const std::string& description);
void syncScene(UIContext& ctx);
nlohmann::json worldPose(UIContext& ctx, const nlohmann::json& emitter);
nlohmann::json localPosition(UIContext& ctx, const nlohmann::json& emitter,
                             const nlohmann::json& worldPosition);
}
