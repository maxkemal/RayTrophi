#pragma once

#include "PrincipledBSDF.h"
#include "Paint/PaintTextureSet.h"
#include "imgui.h"

namespace MaterialTextureGridUI {

template <typename DrawSlot>
void draw(PrincipledBSDF& material, DrawSlot&& drawSlot,
          bool& changed, bool& textureChanged) {
    struct Slot {
        const char* label;
        std::shared_ptr<Texture>* texture;
        TextureType type;
        Paint::PaintChannel channel;
        int* packedChannel;
        bool normal;
    };
    Slot slots[] = {
        {"Albedo", &material.albedoProperty.texture, TextureType::Albedo,
         Paint::PaintChannel::BaseColor, nullptr, false},
        {"Normal", &material.normalProperty.texture, TextureType::Normal,
         Paint::PaintChannel::Normal, nullptr, true},
        {"Roughness", &material.roughnessProperty.texture, TextureType::Roughness,
         Paint::PaintChannel::Roughness, &material.roughness_tex_channel, false},
        {"Metallic", &material.metallicProperty.texture, TextureType::Metallic,
         Paint::PaintChannel::Metallic, &material.metallic_tex_channel, false},
        {"Specular", &material.specularProperty.texture, TextureType::Specular,
         Paint::PaintChannel::Mask, nullptr, false},
        {"Emission", &material.emissionProperty.texture, TextureType::Emission,
         Paint::PaintChannel::Emission, nullptr, false},
        {"Transmission", &material.transmissionProperty.texture, TextureType::Transmission,
         Paint::PaintChannel::Transmission, nullptr, false},
        {"Height", &material.heightProperty.texture, TextureType::Unknown,
         Paint::PaintChannel::Mask, nullptr, false},
        {"Opacity", &material.opacityProperty.texture, TextureType::Opacity,
         Paint::PaintChannel::Opacity, nullptr, false}
    };
    const float minimumCell = ImGui::GetFontSize() * 10.5f;
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const int columns = ImGui::GetContentRegionAvail().x >= minimumCell * 2.0f + gap ? 2 : 1;
    if (!ImGui::BeginTable("texture_grid", columns, ImGuiTableFlags_SizingStretchSame)) {
        return;
    }
    for (auto& slot : slots) {
        ImGui::TableNextColumn();
        ImGui::PushID(slot.label);
        ImGui::BeginGroup();
        drawSlot(slot.label, *slot.texture, slot.type, slot.channel);
        if (slot.normal) {
            float strength = material.get_normal_strength();
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::SliderFloat("##strength", &strength, 0.0f, 8.0f, "Strength %.2f")) {
                material.set_normal_strength(strength);
                material.normalProperty.intensity = strength;
                changed = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Normal strength; Ctrl+click for numeric entry");
            }
        }
        if (slot.packedChannel && *slot.texture) {
            const char* items[] = {"Auto (ORM)", "R", "G", "B"};
            int selected = *slot.packedChannel;
            if (selected < 0 || selected > 3) {
                selected = 0;
            }
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::Combo("##channel", &selected, items, IM_ARRAYSIZE(items))) {
                *slot.packedChannel = selected;
                changed = true;
                textureChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Texture channel; Auto uses ORM packing");
            }
        }
        ImGui::EndGroup();
        ImGui::Spacing();
        ImGui::PopID();
    }
    ImGui::EndTable();
}

} // namespace MaterialTextureGridUI
