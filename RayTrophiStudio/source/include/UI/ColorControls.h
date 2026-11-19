#pragma once

#include "imgui.h"

// Presentation only. Callers retain their setters, validation and change commits.
namespace ColorControls {

inline bool field3(const char* id, float* value) {
    // Keep the original Fog View layout: native RGB inputs, square preview
    // and label in one row. No custom width threshold or forced line break.
    //
    // ★★★ "Changed" means the value CHANGED (2026-11-19). ImGui's picker
    //   reports a change on EVERY frame its hue bar or saturation square is
    //   held - it rebuilds RGB from HSV and only compares against the colour
    //   the popup opened with - so a mouse resting on the bar re-committed the
    //   material each frame: RT and RayFusion accumulation reset non-stop and
    //   the per-edit GPU upload ran every frame.
    const float before[3] = {value[0], value[1], value[2]};
    const bool reported = ImGui::ColorEdit3(id, value);
    return reported &&
           (value[0] != before[0] || value[1] != before[1] || value[2] != before[2]);
}

} // namespace ColorControls
