#pragma once
#include "Api/RtApi.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "imgui.h"

namespace MaterialCoverageUI {
inline void draw(uint16_t materialId, const PrincipledBSDF& material) {
    bool cutout = material.coverage.alphaCutout;
    if (ImGui::Checkbox("Alpha Cutout", &cutout)) {
        const auto result = rtapi::setMaterialParamByName(
            MaterialManager::getInstance().getMaterialName(materialId),
            "alpha_cutout", cutout ? 1.0f : 0.0f);
        if (!result.ok) ImGui::TextWrapped("%s", result.error.c_str());
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Use alpha as a leaf/card coverage mask (threshold 0.5).\n"
        "Fractional alpha does not create glass. Transmission remains independent.\n"
        "Off preserves legacy transparency.");
}
}
