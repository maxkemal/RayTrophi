#pragma once

#include "Api/RtApi.h"
#include "imgui.h"

#include <string>

namespace AnimGraphTriggerActions {

// UI and external clients address the same per-character runtime service.
inline void drawFireButton(const std::string& character, const std::string& name) {
    static std::string error;
    if (ImGui::Button("Fire Trigger")) {
        const auto result = rtapi::triggerAnimGraphParam(character, name);
        error = result.ok ? std::string{} : result.error;
    }
    if (!error.empty()) {
        ImGui::TextWrapped("Trigger failed: %s", error.c_str());
    }
}

} // namespace AnimGraphTriggerActions
