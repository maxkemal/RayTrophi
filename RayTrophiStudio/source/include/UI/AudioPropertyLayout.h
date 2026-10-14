#pragma once
#include "imgui.h"
#include <cfloat>
#include <string>

namespace AudioUI {
// A label above its full-width field never consumes a fixed right-side gutter.
inline std::string property(const char* label) {
    ImGui::TextWrapped("%s", label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    return std::string("##") + label;
}
}
