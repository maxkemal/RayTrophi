#pragma once

#include "UI/PanelControls.h"

#include "UI/MaterialInspectorSurface.h"
#include "UI/MaterialInspectorTransparency.h"
#include "UI/MaterialInspectorInterior.h"
#include "UI/MaterialInspectorSubsurface.h"
#include "UI/MaterialInspectorCoat.h"
#include "UI/MaterialInspectorEmission.h"
#include "UI/MaterialInspectorDetail.h"

#include <string>

namespace MaterialInspectorUI {

inline bool draw(PrincipledBSDF* material, uint16_t materialId,
                 const MaterialSurfaceBasicsUI::IsKeyed& isKeyed,
                 const MaterialSurfaceBasicsUI::InsertKey& insertKey) {
    using DrawSection = bool (*)(PrincipledBSDF*, uint16_t, const MaterialSurfaceBasicsUI::IsKeyed&,
                                 const MaterialSurfaceBasicsUI::InsertKey&);
    struct Section {
        const char* label;
        const char* description;
        DrawSection draw;
    };
    using namespace MaterialInspectorSections;
    static const Section sections[] = {
        {"Surface", "Color, roughness and reflection", drawSurface},
        {"Transparency", "Transmission, coverage and thin shells", drawTransparency},
        {"Interior", "Depth, absorption and suspended detail", drawInterior},
        {"Subsurface", "Light scattering beneath the surface", drawSubsurface},
        {"Coat", "A second reflective layer", drawCoat},
        {"Emission", "Light emitted by the material", drawEmission},
        {"Detail", "Surface variation and texture repetition", drawDetail}};

    ImGui::PushID("material_properties");
    ImGui::PushID(static_cast<int>(materialId));
    ImGuiStorage* storage = ImGui::GetStateStorage();
    const int count = IM_ARRAYSIZE(sections);
    const int allSections = (1 << count) - 1;
    const ImGuiID visibleId = ImGui::GetID("visible_sections");
    int previous = storage->GetInt(ImGui::GetID("section"), 0);
    if (previous < 0 || previous >= count) {
        previous = 0;
    }
    int visible = storage->GetInt(visibleId, 1 << previous) & allSections;
    int visibleCount = 0;
    std::string preview;
    for (int i = 0; i < count; ++i) {
        if ((visible & (1 << i)) != 0) {
            ++visibleCount;
            preview = sections[i].label;
        }
    }
    if (visibleCount == 0) {
        preview = "Choose sections...";
    } else if (visibleCount > 1) {
        preview = std::to_string(visibleCount) + " sections visible";
    }
    PanelControls::heading("Material properties");
    if (PanelControls::beginCombo("##section", preview.c_str())) {
        for (int i = 0; i < count; ++i) {
            bool enabled = (visible & (1 << i)) != 0;
            if (ImGui::Checkbox(sections[i].label, &enabled)) {
                if (enabled) {
                    visible |= 1 << i;
                } else {
                    visible &= ~(1 << i);
                }
                storage->SetInt(visibleId, visible);
            }
        }
        ImGui::EndCombo();
    }
    bool changed = false;
    for (int i = 0; i < count; ++i) {
        if ((visible & (1 << i)) == 0) {
            continue;
        }
        ImGui::PushID(i);
        ImGui::Spacing();
        ImGui::SeparatorText(sections[i].label);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", sections[i].description);
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        changed |= sections[i].draw(material, materialId, isKeyed, insertKey);
        ImGui::PopID();
    }
    ImGui::PopID();
    ImGui::PopID();
    return changed;
}

} // namespace MaterialInspectorUI
