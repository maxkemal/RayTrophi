/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          ui_section_style.cpp
* Description:   Implementation of flat monochrome & theme-controlled section headers.
* =========================================================================
*/
#include "ui_section_style.h"
#include "imgui_internal.h"

void SectionStyleManager::saveSettings(std::ostream& os) const {
    os << static_cast<int>(settings_.style) << " "
       << (settings_.overrideSectionAccentsWithTheme ? 1 : 0) << " "
       << (settings_.showLeftBar ? 1 : 0) << " "
       << settings_.headerPaddingY << " "
       << settings_.headerBgAlpha << " "
       << settings_.borderAlpha << "\n";
}

void SectionStyleManager::loadSettings(std::istream& is) {
    int secStyleVal = 0;
    int overrideSecAccents = 1;
    int showLeftBarVal = 1;
    if (is >> secStyleVal >> overrideSecAccents >> showLeftBarVal >> settings_.headerPaddingY >> settings_.headerBgAlpha >> settings_.borderAlpha) {
        settings_.style = static_cast<SectionHeaderStyle>(secStyleVal);
        settings_.overrideSectionAccentsWithTheme = (overrideSecAccents != 0);
        settings_.showLeftBar = (showLeftBarVal != 0);
    }
}

bool SectionStyleManager::beginSection(const char* title, const ImVec4& accentColor, bool defaultOpen,
                                         const ImVec4& themeAccent, const ImVec4& themeBorder, const ImVec4& themeText) {
    ImGui::PushID(title);
    
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    float width = ImGui::GetContentRegionAvail().x;
    
    // Effective accent/highlight color based on global theme control
    ImVec4 finalAccent = accentColor;
    if (settings_.overrideSectionAccentsWithTheme || settings_.style == SectionHeaderStyle::FlatMonochrome) {
        if (settings_.style == SectionHeaderStyle::FlatMonochrome) {
            // Renksiz / Flat Monochrome: Use neutral theme border tint
            finalAccent = themeBorder;
        } else if (settings_.style == SectionHeaderStyle::ThemeAccent) {
            // Theme Accent: Use current theme's primary accent
            finalAccent = themeAccent;
        }
    }

    // Set frame padding before querying height to ensure exact, slim header height
    float padY = (settings_.style == SectionHeaderStyle::LegacyRainbow) ? 5.0f : settings_.headerPaddingY;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, padY));
    float height = ImGui::GetFrameHeight();
    
    // Header Background
    float bgAlpha = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? settings_.headerBgAlpha : 0.16f;
    ImU32 headerBg = ImGui::ColorConvertFloat4ToU32(ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, bgAlpha));
    
    float bAlpha = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? settings_.borderAlpha : 0.40f;
    ImU32 borderColor = ImGui::ColorConvertFloat4ToU32(ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, bAlpha));

    float rounding = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? 3.0f : 4.0f;
    
    // Draw Header Background (Sleek, rounded corners top if open, all if closed)
    drawList->AddRectFilled(cursorPos, ImVec2(cursorPos.x + width, cursorPos.y + height), headerBg, rounding, defaultOpen ? ImDrawFlags_RoundCornersTop : ImDrawFlags_RoundCornersAll);
    
    if (settings_.style == SectionHeaderStyle::LegacyRainbow) {
        // Draw Top Accent Line (Thin, distinct)
        drawList->AddLine(
            ImVec2(cursorPos.x, cursorPos.y), 
            ImVec2(cursorPos.x + width, cursorPos.y), 
            ImGui::ColorConvertFloat4ToU32(finalAccent), 2.0f
        );
    } else if (settings_.showLeftBar) {
        // Draw Left Accent Bar (Minimalist 2px accent indicator)
        ImU32 leftBarColor = ImGui::ColorConvertFloat4ToU32(
            (settings_.style == SectionHeaderStyle::FlatMonochrome) ? themeAccent : finalAccent
        );
        drawList->AddRectFilled(
            cursorPos, 
            ImVec2(cursorPos.x + 2.5f, cursorPos.y + height), 
            leftBarColor, 1.0f
        );
    }
    
    // Override TreeNode style colors so ImGui doesn't render built-in chunky frame background
    float hoverAlpha = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? 0.08f : 0.14f;
    float activeAlpha = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? 0.14f : 0.24f;

    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, hoverAlpha));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, activeAlpha));
    ImGui::PushStyleColor(ImGuiCol_Text, themeText); 
    
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth |
                               ImGuiTreeNodeFlags_AllowOverlap |
                               ImGuiTreeNodeFlags_FramePadding |
                               ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (defaultOpen) flags |= ImGuiTreeNodeFlags_DefaultOpen;

    bool opened = ImGui::TreeNodeEx(title, flags);
    
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar();

    if (opened) {
        // Push state for EndSection to draw the surrounding border
        sectionStack_.push_back({cursorPos, width, borderColor, opened});
        
        // Add indent and vertical spacing for content
        ImGui::Indent(6.0f);
        ImGui::Spacing();
    } else {
        ImGui::PopID();
    }
    
    return opened;
}

void SectionStyleManager::endSection() {
    if (sectionStack_.empty()) return;
    
    SectionStackItem state = sectionStack_.back();
    sectionStack_.pop_back();
    
    if (state.isOpen) {
        ImGui::Spacing();
        ImGui::Unindent(6.0f);
        
        // Draw the Border around the whole open section
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 endPos = ImGui::GetCursorScreenPos();
        
        float rounding = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? 3.0f : 4.0f;

        drawList->AddRect(
            state.startPos,
            ImVec2(state.startPos.x + state.width, endPos.y),
            state.borderColor,
            rounding
        );

        ImGui::PopID();
    }
}

bool SectionStyleManager::beginColoredSection(const char* title, const ImVec4& titleColor, bool defaultOpen, const ImVec4& themeText) {
    ImVec4 finalTextColor = (settings_.overrideSectionAccentsWithTheme || settings_.style == SectionHeaderStyle::FlatMonochrome) 
        ? themeText 
        : titleColor;

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth |
                               ImGuiTreeNodeFlags_AllowOverlap |
                               ImGuiTreeNodeFlags_FramePadding |
                               ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (defaultOpen) flags |= ImGuiTreeNodeFlags_DefaultOpen;

    ImGui::PushStyleColor(ImGuiCol_Text, finalTextColor);
    bool opened = ImGui::TreeNodeEx(title, flags);
    ImGui::PopStyleColor();
    return opened;
}

bool SectionStyleManager::beginCollapsingHeader(const char* label, ImGuiTreeNodeFlags flags,
                                                const ImVec4& accentColor, const ImVec4& themeAccent,
                                                const ImVec4& themeBorder, const ImVec4& themeText) {
    ImVec4 finalAccent = accentColor;
    if (settings_.overrideSectionAccentsWithTheme || settings_.style == SectionHeaderStyle::FlatMonochrome) {
        if (settings_.style == SectionHeaderStyle::FlatMonochrome) {
            finalAccent = themeBorder;
        } else if (settings_.style == SectionHeaderStyle::ThemeAccent) {
            finalAccent = themeAccent;
        }
    }

    float padY = (settings_.style == SectionHeaderStyle::LegacyRainbow) ? 5.0f : settings_.headerPaddingY;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, padY));

    float bgAlpha = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? settings_.headerBgAlpha : 0.16f;
    float hoverAlpha = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? 0.08f : 0.16f;
    float activeAlpha = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? 0.14f : 0.24f;
    float bAlpha = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? settings_.borderAlpha : 0.30f;

    ImVec4 bgCol = ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, bgAlpha);
    ImVec4 hovCol = ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, hoverAlpha);
    ImVec4 actCol = ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, activeAlpha);
    ImVec4 borderCol = ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, bAlpha);
    ImVec4 textCol = themeText;

    ImGui::PushStyleColor(ImGuiCol_Header, bgCol);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, hovCol);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, actCol);
    ImGui::PushStyleColor(ImGuiCol_Border, borderCol);
    ImGui::PushStyleColor(ImGuiCol_Text, textCol);

    float rounding = (settings_.style == SectionHeaderStyle::FlatMonochrome) ? 3.0f : 4.0f;
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, rounding);

    bool opened = ImGui::CollapsingHeader(label, flags | ImGuiTreeNodeFlags_SpanAvailWidth);

    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(5);

    return opened;
}

void SectionStyleManager::drawSectionSettingsUI(float& panelAlpha) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.65f, 1.0f), "Sub-Panel & Section Header Settings:");
    ImGui::Indent();

    bool changed = false;

    const char* headerStyleNames[] = {
        "Flat Monochrome (Clean & Thin)",
        "Active Theme Accent",
        "Legacy Rainbow Colors"
    };
    int secStyleIdx = static_cast<int>(settings_.style);
    if (ImGui::Combo("Sub-Panel Header Style", &secStyleIdx, headerStyleNames, IM_ARRAYSIZE(headerStyleNames))) {
        settings_.style = static_cast<SectionHeaderStyle>(secStyleIdx);
        changed = true;
    }

    changed |= ImGui::Checkbox("Override Sub-Panel Colors with Theme", &settings_.overrideSectionAccentsWithTheme);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Unifies all collapsible sub-panels with the active theme color palette instead of per-panel disparate colors.");
    }

    changed |= ImGui::Checkbox("Show Left Accent Indicator Bar", &settings_.showLeftBar);
    changed |= ImGui::SliderFloat("Header Slim Padding (Y)", &settings_.headerPaddingY, 1.0f, 6.0f, "%.1f px");
    changed |= ImGui::SliderFloat("Header Background Alpha", &settings_.headerBgAlpha, 0.0f, 0.40f, "%.2f");
    changed |= ImGui::SliderFloat("Section Border Opacity", &settings_.borderAlpha, 0.05f, 0.60f, "%.2f");

    if (changed) {
        // Will be persisted by saveThemeSettings
    }
    ImGui::Unindent();
}
