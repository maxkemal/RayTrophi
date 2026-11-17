#pragma once

#include "Api/RtApiCharacter.h"
#include "imgui.h"
#include <string>

namespace CharacterUI {
inline std::string error;

inline bool deleteButton(const std::string &character) {
    if (!ImGui::Button("Delete character and meshes")) {
        return false;
    }
    const auto result = rtapi::deleteCharacter(character);
    error = result.ok ? "" : result.error;
    return result.ok;
}

inline void showError() {
    if (!error.empty()) {
        ImGui::TextWrapped("Character deletion: %s", error.c_str());
    }
}
} // namespace CharacterUI
