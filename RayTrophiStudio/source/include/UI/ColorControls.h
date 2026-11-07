#pragma once

#include "imgui.h"

// Presentation only. Callers retain their setters, validation and change commits.
namespace ColorControls {

inline bool field3(const char* id, float* value) {
    // Keep the original Fog View layout: native RGB inputs, square preview
    // and label in one row. No custom width threshold or forced line break.
    return ImGui::ColorEdit3(id, value);
}

} // namespace ColorControls
