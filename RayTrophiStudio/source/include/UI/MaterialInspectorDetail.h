#pragma once

#include "UI/MaterialSurfaceBasicsUI.h"
#include "UI/MaterialCoverageUI.h"
#include "UI/MaterialInspectorControls.h"
#include "ui_modern.h"

namespace MaterialInspectorSections {

inline bool drawDetail(PrincipledBSDF* pbsdf, uint16_t mat_id,
                       const MaterialSurfaceBasicsUI::IsKeyed& isMatKeyed,
                       const MaterialSurfaceBasicsUI::InsertKey& insertMatPropKey) {
    bool changed = false;

    float det_str = pbsdf->micro_detail_strength;
    if (MaterialInspectorControls::smartFloat("pd_str", "Strength", &det_str, 0.0f, 1.0f, "%.3f",
                                              false, nullptr, 12)) {
        pbsdf->micro_detail_strength = det_str;
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("World-space color variation + dirt + roughness.\nDoes NOT warp UV — "
                          "texture stays intact.\n0 = off, 0.2–0.5 = subtle, 1.0 = strong");
    }

    float det_sc = pbsdf->micro_detail_scale;
    if (MaterialInspectorControls::smartFloat("pd_sc", "Scale", &det_sc, 0.1f, 10.0f, "%.2f", false,
                                              nullptr, 12)) {
        pbsdf->micro_detail_scale = det_sc;
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("World-space frequency\n1–2 = large patches\n4–8 = fine grain");
    }

    ImGui::Separator();

    float tb_str = pbsdf->tile_break_strength;
    if (MaterialInspectorControls::smartFloat("pd_tb", "Tile Break", &tb_str, 0.0f, 0.35f, "%.3f",
                                              false, nullptr, 12)) {
        pbsdf->tile_break_strength = tb_str;
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("UV warp to break repeating tiling seams.\nLeave at 0 for "
                          "unique/non-tiling albedo maps.\n0.05–0.15 = subtle, 0.25–0.35 = strong");
    }

    ImGui::Separator();
    if (ImGui::SmallButton("Worn Stone")) {
        pbsdf->micro_detail_strength = 0.45f;
        pbsdf->micro_detail_scale = 3.5f;
        pbsdf->tile_break_strength = 0.0f;
        changed = true;
    }
    MaterialInspectorControls::nextPreset();
    if (ImGui::SmallButton("Dusty")) {
        pbsdf->micro_detail_strength = 0.30f;
        pbsdf->micro_detail_scale = 2.0f;
        pbsdf->tile_break_strength = 0.0f;
        changed = true;
    }
    MaterialInspectorControls::nextPreset();
    if (ImGui::SmallButton("Tiled Tex")) {
        pbsdf->micro_detail_strength = 0.20f;
        pbsdf->micro_detail_scale = 2.5f;
        pbsdf->tile_break_strength = 0.15f;
        changed = true;
    }
    MaterialInspectorControls::nextPreset();
    if (ImGui::SmallButton("Off")) {
        pbsdf->micro_detail_strength = 0.0f;
        pbsdf->tile_break_strength = 0.0f;
        changed = true;
    }
    return changed;
}

} // namespace MaterialInspectorSections
