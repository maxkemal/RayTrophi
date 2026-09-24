/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          ui_section_style.h
* Description:   Flat monochrome & theme-controlled sub-panel section system.
* =========================================================================
*/
#pragma once

#include "imgui.h"
#include <vector>
#include <string>
#include <iostream>

enum class SectionHeaderStyle {
    FlatMonochrome = 0,   // Renksiz, ince, flat, minimalist başlık (Önerilen)
    ThemeAccent = 1,      // Aktif tema vurgu rengi ile ince başlık
    LegacyRainbow = 2     // Klasik çok renkli kalın başlıklar
};

struct SectionSettings {
    SectionHeaderStyle style = SectionHeaderStyle::FlatMonochrome;
    bool overrideSectionAccentsWithTheme = true; // Tüm alt bölmeler genel temaya tabidir
    bool showLeftBar = true;                      // İnce sol gösterge çizgisi
    float headerPaddingY = 3.0f;                  // Thinner flat başlık dolgusu (varsayılan 3.0px)
    float borderAlpha = 0.16f;                    // Zarif kenarlık alpha
    float headerBgAlpha = 0.06f;                  // Flat renksiz arka plan alpha
};

struct SectionStackItem {
    ImVec2 startPos;
    float width;
    ImU32 borderColor;
    bool isOpen;
};

class SectionStyleManager {
public:
    static SectionStyleManager& instance() {
        static SectionStyleManager inst;
        return inst;
    }

    SectionSettings& settings() { return settings_; }
    const SectionSettings& settings() const { return settings_; }

    void saveSettings(std::ostream& os) const;
    void loadSettings(std::istream& is);

    bool beginSection(const char* title, const ImVec4& accentColor, bool defaultOpen,
                      const ImVec4& themeAccent, const ImVec4& themeBorder, const ImVec4& themeText);
    void endSection();
    bool beginColoredSection(const char* title, const ImVec4& titleColor, bool defaultOpen, const ImVec4& themeText);
    bool beginCollapsingHeader(const char* label, ImGuiTreeNodeFlags flags = 0,
                               const ImVec4& accentColor = ImVec4(0.4f, 0.6f, 1.0f, 1.0f),
                               const ImVec4& themeAccent = ImVec4(0.4f, 0.6f, 1.0f, 1.0f),
                               const ImVec4& themeBorder = ImVec4(0.3f, 0.3f, 0.35f, 1.0f),
                               const ImVec4& themeText = ImVec4(0.95f, 0.95f, 0.95f, 1.0f));

    void drawSectionSettingsUI(float& panelAlpha);

private:
    SectionStyleManager() = default;
    SectionSettings settings_;
    std::vector<SectionStackItem> sectionStack_;
};
