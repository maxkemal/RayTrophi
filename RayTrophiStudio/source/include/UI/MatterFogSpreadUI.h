#pragma once

#include "Api/RtApi.h"
#include "Fluid/FluidFogDensity.h"
#include "imgui.h"
#include <string>

namespace RayTrophiSim::Fluid {

inline void drawMatterFogSpread(const std::string& domain, float currentSpread) {
    if (!ImGui::CollapsingHeader("Fog View##MatterFogSpread",
            ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    float spread = currentSpread;
    if (ImGui::SliderFloat("Fog Spread (voxels)##MatterFogSpread", &spread,
            0.0f, kFogSpreadMaxVoxels, "%.2f")) {
        rtapi::FluidFogPatch patch;
        patch.spread_voxels = spread;
        const auto result = rtapi::setFluidFog(domain, patch);
        if (!result.ok) {
            ImGui::TextWrapped("%s", result.error.c_str());
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Fog carrier spread (Gaussian sigma in simulation voxels).\n"
            "0 = raw splat. Larger values make wider, softer fog particles.\n"
            "Render only; does not change grain radius or gas density.");
    }
}

} // namespace RayTrophiSim::Fluid
