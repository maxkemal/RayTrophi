/*
 * RayTrophi Modern UI System - Implementation
 * ============================================
 * ThemeManager, UIWidgets ve PanelManager implementasyonlarÄ±.
 */

#include "ui_modern.h"
#include "globals.h"
#include <fstream>
#include <algorithm>
#include <imgui_internal.h>
#include <unordered_map>

// ============================================================================
// THEME MANAGER IMPLEMENTATION
// ============================================================================

void ThemeManager::registerDefaultThemes() {
    themes_.clear();

    // --- 0: RayTrophi Pro Dark (Varsayılan) ---
    {
        Theme t;
        t.name = "RayTrophi Pro Dark";
        t.style.frameRounding = 4.0f;
        t.style.windowRounding = 4.0f;
        t.style.scrollbarRounding = 9.0f;
        t.style.grabRounding = 4.0f;
        t.style.popupRounding = 4.0f;
        t.style.tabRounding = 4.0f;

        t.colors.primary    = ImVec4(0.32f, 0.32f, 0.34f, 1.0f); // Button gray, stands out on sandy bg
        t.colors.secondary  = ImVec4(0.18f, 0.18f, 0.19f, 1.0f); // Headers and active tabs
        t.colors.accent     = ImVec4(0.90f, 0.52f, 0.18f, 1.0f); // Blender Orange
        t.colors.background = ImVec4(0.24f, 0.24f, 0.25f, 1.0f); // Blender warm dark sand background
        t.colors.surface    = ImVec4(0.14f, 0.14f, 0.15f, 1.0f); // Dark recessed inputs/lists/child frames
        t.colors.text       = ImVec4(0.88f, 0.88f, 0.88f, 1.0f); // Soft grey text (eye-friendly)
        t.colors.textMuted  = ImVec4(0.55f, 0.55f, 0.56f, 1.0f);
        t.colors.success    = ImVec4(0.26f, 0.65f, 0.36f, 1.0f);
        t.colors.warning    = ImVec4(0.85f, 0.60f, 0.15f, 1.0f);
        t.colors.error      = ImVec4(0.80f, 0.30f, 0.30f, 1.0f);
        t.colors.border     = ImVec4(0.11f, 0.11f, 0.12f, 0.50f);
        themes_.push_back(t);
    }

    // --- 1: Custom Theme ---
    {
        Theme t;
        t.name = "Custom Theme";
        t.style.frameRounding = 4.0f;
        t.style.windowRounding = 4.0f;
        t.style.scrollbarRounding = 9.0f;
        t.style.grabRounding = 4.0f;
        t.style.popupRounding = 4.0f;
        t.style.tabRounding = 4.0f;

        // Initialize to RayTrophi Pro Dark colors as a starting point
        t.colors.primary    = ImVec4(0.32f, 0.32f, 0.34f, 1.0f);
        t.colors.secondary  = ImVec4(0.18f, 0.18f, 0.19f, 1.0f);
        t.colors.accent     = ImVec4(0.90f, 0.52f, 0.18f, 1.0f);
        t.colors.background = ImVec4(0.24f, 0.24f, 0.25f, 1.0f);
        t.colors.surface    = ImVec4(0.14f, 0.14f, 0.15f, 1.0f);
        t.colors.text       = ImVec4(0.88f, 0.88f, 0.88f, 1.0f);
        t.colors.textMuted  = ImVec4(0.55f, 0.55f, 0.56f, 1.0f);
        t.colors.success    = ImVec4(0.26f, 0.65f, 0.36f, 1.0f);
        t.colors.warning    = ImVec4(0.85f, 0.60f, 0.15f, 1.0f);
        t.colors.error      = ImVec4(0.80f, 0.30f, 0.30f, 1.0f);
        t.colors.border     = ImVec4(0.11f, 0.11f, 0.12f, 0.50f);
        themes_.push_back(t);
    }

    currentIndex_ = 0; // RayTrophi Pro Dark varsayılan
    loadCustomThemes("custom_themes.cfg");
}

void ThemeManager::addTheme(const Theme& theme) {
    themes_.push_back(theme);
    SCENE_LOG_INFO("[addTheme] Added theme: " + theme.name + ". themes_.size(): " + std::to_string(themes_.size()));
}

void ThemeManager::deleteTheme(int index) {
    if (index >= 2 && index < static_cast<int>(themes_.size())) {
        themes_.erase(themes_.begin() + index);
        currentIndex_ = 0; // Reset to RayTrophi Pro Dark
    }
}

void ThemeManager::setTheme(int index) {
    if (index >= 0 && index < static_cast<int>(themes_.size())) {
        currentIndex_ = index;
    }
}

void ThemeManager::setTheme(const std::string& name) {
    for (int i = 0; i < static_cast<int>(themes_.size()); ++i) {
        if (themes_[i].name == name) {
            currentIndex_ = i;
            return;
        }
    }
}

void ThemeManager::applyCurrentTheme(float panelAlpha) {
    const Theme& t = themes_[currentIndex_];
    ImGuiStyle& style = ImGui::GetStyle();
    
    // Always apply dark style as base
    ImGui::StyleColorsDark();

    // Stil ayarlari
    style.WindowRounding    = t.style.windowRounding;
    style.ChildRounding     = t.style.windowRounding;
    style.FrameRounding     = t.style.frameRounding;
    style.GrabRounding      = t.style.grabRounding;
    style.ScrollbarSize     = 7.0f; // Blender-like thin width
    style.ScrollbarRounding = 9.0f; // Blender-like rounded capsule shape
    style.TabRounding       = t.style.tabRounding;
    style.PopupRounding     = t.style.popupRounding;
    style.FramePadding      = t.style.framePadding;
    style.ItemSpacing       = t.style.itemSpacing;
    style.WindowPadding     = t.style.windowPadding;

    // Renkler
    ImVec4* c = style.Colors;
    
    c[ImGuiCol_WindowBg]          = ImVec4(t.colors.background.x, t.colors.background.y, 
                                            t.colors.background.z, panelAlpha);
    c[ImGuiCol_ChildBg]           = ImVec4(t.colors.surface.x, t.colors.surface.y,
                                            t.colors.surface.z, panelAlpha);
    c[ImGuiCol_PopupBg]           = ImVec4(t.colors.background.x, t.colors.background.y,
                                            t.colors.background.z, 0.98f);
    c[ImGuiCol_MenuBarBg]         = ImVec4(t.colors.secondary.x, t.colors.secondary.y,
                                            t.colors.secondary.z, 1.0f);
    
    c[ImGuiCol_Text]              = t.colors.text;
    c[ImGuiCol_TextDisabled]      = t.colors.textMuted;
    
    c[ImGuiCol_FrameBg]           = t.colors.surface;
    c[ImGuiCol_FrameBgHovered]    = UIWidgets::ScaleColor(t.colors.surface, 1.3f);
    c[ImGuiCol_FrameBgActive]     = UIWidgets::ScaleColor(t.colors.surface, 1.5f);
    
    c[ImGuiCol_Button]            = t.colors.primary;
    c[ImGuiCol_ButtonHovered]     = UIWidgets::ScaleColor(t.colors.primary, 1.2f);
    c[ImGuiCol_ButtonActive]      = UIWidgets::ScaleColor(t.colors.primary, 0.8f);
    
    c[ImGuiCol_Header]            = ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.22f);
    c[ImGuiCol_HeaderHovered]     = ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.48f);
    c[ImGuiCol_HeaderActive]      = ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.70f);
    
    c[ImGuiCol_SliderGrab]        = t.colors.accent;
    c[ImGuiCol_SliderGrabActive]  = UIWidgets::ScaleColor(t.colors.accent, 1.2f);
    
    c[ImGuiCol_Border]            = t.colors.border;
    
    c[ImGuiCol_Tab]               = UIWidgets::ScaleColor(t.colors.secondary, 0.8f);
    c[ImGuiCol_TabHovered]        = UIWidgets::ScaleColor(t.colors.secondary, 1.3f);
    c[ImGuiCol_TabActive]         = t.colors.secondary;
    
    c[ImGuiCol_TitleBg]           = t.colors.secondary;
    c[ImGuiCol_TitleBgActive]     = UIWidgets::ScaleColor(t.colors.secondary, 1.15f);
    c[ImGuiCol_TitleBgCollapsed]  = UIWidgets::ScaleColor(t.colors.secondary, 0.85f);
    
    c[ImGuiCol_CheckMark]         = t.colors.accent;
    c[ImGuiCol_TextSelectedBg]    = ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.35f);
    
    c[ImGuiCol_Separator]         = t.colors.border;
    c[ImGuiCol_SeparatorHovered]  = ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.50f);
    c[ImGuiCol_SeparatorActive]   = t.colors.accent;
    
    c[ImGuiCol_ResizeGrip]        = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.60f);
    c[ImGuiCol_ResizeGripActive]  = ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.90f);
    
    // Blender-like Flat & Thin Scrollbar Styles
    c[ImGuiCol_ScrollbarBg]          = ImVec4(0.0f, 0.0f, 0.0f, 0.0f); // Transparent track background
    c[ImGuiCol_ScrollbarGrab]        = ImVec4(t.colors.textMuted.x, t.colors.textMuted.y, t.colors.textMuted.z, 0.30f); // Flat, subtle handle
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(t.colors.textMuted.x, t.colors.textMuted.y, t.colors.textMuted.z, 0.55f); // Highlighted on hover
    c[ImGuiCol_ScrollbarGrabActive]  = ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.75f); // Active colored grab
}

const char* ThemeManager::getThemeName(int index) const {
    if (index >= 0 && index < static_cast<int>(themes_.size())) {
        return themes_[index].name.c_str();
    }
    return "Unknown";
}

std::vector<const char*> ThemeManager::getAllThemeNames() const {
    std::vector<const char*> names;
    for (const auto& t : themes_) {
        names.push_back(t.name.c_str());
    }
    return names;
}

void ThemeManager::saveThemeSettings(const std::string& filepath, float panelAlpha) {
    std::ofstream file(filepath);
    if (!file.is_open()) return;
    file.imbue(std::locale::classic());
    
    std::string themeName = themes_[currentIndex_].name;
    std::replace(themeName.begin(), themeName.end(), ' ', '_');
    file << themeName << " " << panelAlpha << "\n";
    
    // Save icon settings
    file << static_cast<int>(iconSettings_.style) << " "
         << iconSettings_.scaleMultiplier << " "
         << iconSettings_.thicknessMultiplier << " "
         << (iconSettings_.overridePanelAccentsWithTheme ? 1 : 0) << "\n";
         
    file << iconSettings_.customColor.x << " " << iconSettings_.customColor.y << " "
         << iconSettings_.customColor.z << " " << iconSettings_.customColor.w << "\n";
         
    file << iconSettings_.customBgColor.x << " " << iconSettings_.customBgColor.y << " "
         << iconSettings_.customBgColor.z << " " << iconSettings_.customBgColor.w << "\n";
         
    file << iconSettings_.customShadowColor.x << " " << iconSettings_.customShadowColor.y << " "
         << iconSettings_.customShadowColor.z << " " << iconSettings_.customShadowColor.w << "\n";
         
    file << iconSettings_.matcapColor.x << " " << iconSettings_.matcapColor.y << " "
         << iconSettings_.matcapColor.z << " " << iconSettings_.matcapColor.w << "\n";
}

bool ThemeManager::loadThemeSettings(const std::string& filepath, float& panelAlpha) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    file.imbue(std::locale::classic());
    
    std::string themeToken;
    float alpha = 0.90f;
    if (file >> themeToken >> alpha) {
        panelAlpha = (alpha < 0.65f) ? 0.65f : ((alpha > 1.0f) ? 1.0f : alpha);
        
        bool isIndex = !themeToken.empty() && std::all_of(themeToken.begin(), themeToken.end(), ::isdigit);
        if (isIndex) {
            int idx = std::stoi(themeToken);
            setTheme(idx);
        } else {
            std::replace(themeToken.begin(), themeToken.end(), '_', ' ');
            setTheme(themeToken);
        }
        
        int styleVal = 0;
        int overrideAccents = 0;
        if (file >> styleVal >> iconSettings_.scaleMultiplier >> iconSettings_.thicknessMultiplier >> overrideAccents) {
            iconSettings_.style = static_cast<IconStyle>(styleVal);
            iconSettings_.overridePanelAccentsWithTheme = (overrideAccents != 0);
            
            file >> iconSettings_.customColor.x >> iconSettings_.customColor.y 
                 >> iconSettings_.customColor.z >> iconSettings_.customColor.w;
                 
            file >> iconSettings_.customBgColor.x >> iconSettings_.customBgColor.y 
                 >> iconSettings_.customBgColor.z >> iconSettings_.customBgColor.w;
                 
            file >> iconSettings_.customShadowColor.x >> iconSettings_.customShadowColor.y 
                 >> iconSettings_.customShadowColor.z >> iconSettings_.customShadowColor.w;

            file >> iconSettings_.matcapColor.x >> iconSettings_.matcapColor.y 
                 >> iconSettings_.matcapColor.z >> iconSettings_.matcapColor.w;
        }
        applyCurrentTheme(panelAlpha);
        return true;
    }
    return false;
}

void ThemeManager::saveCustomThemes(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        SCENE_LOG_ERROR("[saveCustomThemes] Failed to open " + filepath + " for writing.");
        return;
    }
    file.imbue(std::locale::classic());
    
    int numCustom = 0;
    if (themes_.size() > 1) {
        numCustom = static_cast<int>(themes_.size()) - 1;
    }
    file << numCustom << "\n";
   // SCENE_LOG_INFO("[saveCustomThemes] Saving " + std::to_string(numCustom) + " custom themes to " + filepath);
    
    for (size_t i = 1; i < themes_.size(); ++i) {
        const auto& t = themes_[i];
        std::string name = t.name;
        std::replace(name.begin(), name.end(), ' ', '_');
        file << name << "\n";
        
        const auto& tc = t.colors;
        file << tc.primary.x << " " << tc.primary.y << " " << tc.primary.z << " " << tc.primary.w << " "
             << tc.secondary.x << " " << tc.secondary.y << " " << tc.secondary.z << " " << tc.secondary.w << " "
             << tc.accent.x << " " << tc.accent.y << " " << tc.accent.z << " " << tc.accent.w << " "
             << tc.background.x << " " << tc.background.y << " " << tc.background.z << " " << tc.background.w << " "
             << tc.surface.x << " " << tc.surface.y << " " << tc.surface.z << " " << tc.surface.w << " "
             << tc.text.x << " " << tc.text.y << " " << tc.text.z << " " << tc.text.w << " "
             << tc.textMuted.x << " " << tc.textMuted.y << " " << tc.textMuted.z << " " << tc.textMuted.w << " "
             << tc.success.x << " " << tc.success.y << " " << tc.success.z << " " << tc.success.w << " "
             << tc.warning.x << " " << tc.warning.y << " " << tc.warning.z << " " << tc.warning.w << " "
             << tc.error.x << " " << tc.error.y << " " << tc.error.z << " " << tc.error.w << " "
             << tc.border.x << " " << tc.border.y << " " << tc.border.z << " " << tc.border.w << "\n";
             
        const auto& ts = t.style;
        file << ts.windowRounding << " " << ts.frameRounding << " " << ts.grabRounding << " "
             << ts.scrollbarRounding << " " << ts.tabRounding << " " << ts.popupRounding << "\n";
    }
}

void ThemeManager::loadCustomThemes(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return;
    file.imbue(std::locale::classic());
    
    int numCustom = 0;
    if (file >> numCustom) {
        SCENE_LOG_INFO("[loadCustomThemes] Loading " + std::to_string(numCustom) + " custom themes from " + filepath);
        for (int i = 0; i < numCustom; ++i) {
            Theme t;
            if (!(file >> t.name)) {
                SCENE_LOG_ERROR("[loadCustomThemes] Failed to read theme name at index " + std::to_string(i));
                break;
            }
            std::replace(t.name.begin(), t.name.end(), '_', ' ');
            
            auto& tc = t.colors;
            file >> tc.primary.x >> tc.primary.y >> tc.primary.z >> tc.primary.w
                 >> tc.secondary.x >> tc.secondary.y >> tc.secondary.z >> tc.secondary.w
                 >> tc.accent.x >> tc.accent.y >> tc.accent.z >> tc.accent.w
                 >> tc.background.x >> tc.background.y >> tc.background.z >> tc.background.w
                 >> tc.surface.x >> tc.surface.y >> tc.surface.z >> tc.surface.w
                 >> tc.text.x >> tc.text.y >> tc.text.z >> tc.text.w
                 >> tc.textMuted.x >> tc.textMuted.y >> tc.textMuted.z >> tc.textMuted.w
                 >> tc.success.x >> tc.success.y >> tc.success.z >> tc.success.w
                 >> tc.warning.x >> tc.warning.y >> tc.warning.z >> tc.warning.w
                 >> tc.error.x >> tc.error.y >> tc.error.z >> tc.error.w
                 >> tc.border.x >> tc.border.y >> tc.border.z >> tc.border.w;
                 
            auto& ts = t.style;
            file >> ts.windowRounding >> ts.frameRounding >> ts.grabRounding
                 >> ts.scrollbarRounding >> ts.tabRounding >> ts.popupRounding;
                 
            if (file.fail()) {
                SCENE_LOG_ERROR("[loadCustomThemes] Failed to read theme values for index " + std::to_string(i) + " (locale mismatch or corrupt file)");
                break;
            }
                 
            // Check if a theme with this name already exists in default themes (like "Custom Theme"),
            // and overwrite its properties rather than appending a duplicate theme.
            bool found = false;
            for (auto& existing : themes_) {
                if (existing.name == t.name) {
                    existing = t;
                    found = true;
                    break;
                }
            }
            if (!found) {
                themes_.push_back(t);
            }
        }
    } else {
        SCENE_LOG_WARN("[loadCustomThemes] Failed to read numCustom from " + filepath);
    }
}

// ============================================================================
// UI WIDGETS IMPLEMENTATION
// ============================================================================

namespace UIWidgets {

void HelpMarker(const char* desc) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 36.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

ImVec4 ScaleColor(const ImVec4& c, float scale) {
    return ImVec4(
        std::min(c.x * scale, 1.0f),
        std::min(c.y * scale, 1.0f),
        std::min(c.z * scale, 1.0f),
        c.w
    );
}

ImGuiTreeNodeFlags GetSectionFlags(bool defaultOpen) {
    ImGuiTreeNodeFlags flags =
        ImGuiTreeNodeFlags_Framed |
        ImGuiTreeNodeFlags_SpanFullWidth |
        ImGuiTreeNodeFlags_AllowOverlap |
        ImGuiTreeNodeFlags_FramePadding |
        ImGuiTreeNodeFlags_NoTreePushOnOpen; // Disable default tree indentation
    
    if (defaultOpen)
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    
    return flags;
}

// Helper for the border rect
struct SectionState {
    ImVec2 startPos;
    float width;
    ImU32 borderColor;
    bool isOpen;
};
static std::vector<SectionState> s_SectionStack;

bool BeginSection(const char* title, const ImVec4& accentColor, bool defaultOpen) {
    ImGui::PushID(title);
    
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    float width = ImGui::GetContentRegionAvail().x;
    float height = ImGui::GetFrameHeight(); // Standard frame height (thinner than before)
    
    // Convert accent color
    ImU32 accentU32 = ImGui::ColorConvertFloat4ToU32(accentColor);
    ImU32 headerBg = ImGui::ColorConvertFloat4ToU32(ImVec4(accentColor.x, accentColor.y, accentColor.z, 0.15f));
    ImU32 borderColor = ImGui::ColorConvertFloat4ToU32(ImVec4(accentColor.x, accentColor.y, accentColor.z, 0.5f));

    // Draw Header Background (Sleek)
    drawList->AddRectFilled(cursorPos, ImVec2(cursorPos.x + width, cursorPos.y + height), headerBg, 4.0f, ImDrawFlags_RoundCornersTop);
    
    // Draw Top Accent Line (Thin, distinct)
    drawList->AddLine(
        ImVec2(cursorPos.x, cursorPos.y), 
        ImVec2(cursorPos.x + width, cursorPos.y), 
        accentU32, 2.0f
    );
    
    // Modernized TreeNode
    // Use FramePadding 4,4 to make it slimmer (default is often 4,3 or 8,6)
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));
    // Color text matching accent for consistency
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1,1,1,0.95f)); 
    
    bool opened = ImGui::TreeNodeEx(title, GetSectionFlags(defaultOpen));
    
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    if (opened) {
        // Push state for EndSection to draw the surrounding border
        s_SectionStack.push_back({cursorPos, width, borderColor, opened});
        
        // Add a small indent for content (reduced from 8.0f to 4.0f)
        ImGui::Indent(4.0f);
    } else {
        ImGui::PopID();
    }
    
    return opened;
}

void EndSection() {
    if (s_SectionStack.empty()) return;
    
    SectionState state = s_SectionStack.back();
    s_SectionStack.pop_back();
    
    if (state.isOpen) {
        ImGui::Unindent(4.0f);
        // TreePop is NOT needed when using ImGuiTreeNodeFlags_NoTreePushOnOpen
        
        // Draw the Border around the whole open section
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 endPos = ImGui::GetCursorScreenPos();
        
        drawList->AddRect(
            state.startPos,
            ImVec2(state.startPos.x + state.width, endPos.y),
            state.borderColor,
            4.0f
        );

        ImGui::PopID();
    }
}

bool BeginColoredSection(const char* title, const ImVec4& titleColor, bool defaultOpen) {
    ImGui::PushStyleColor(ImGuiCol_Text, titleColor);
    bool opened = ImGui::TreeNodeEx(title, GetSectionFlags(defaultOpen));
    ImGui::PopStyleColor();
    return opened;
}

bool StateButton(const char* label, bool isActive, 
                 const ImVec4& activeColor, const ImVec4& inactiveColor,
                 const ImVec2& size) {
    ImVec4 baseColor = isActive ? activeColor : inactiveColor;
    ImVec4 textColor = isActive 
        ? ImVec4(0.0f, 0.0f, 0.0f, 1.0f)  // Aktifken siyah yazÄ±
        : ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Pasifken beyaz yazÄ±
    
    ImGui::PushStyleColor(ImGuiCol_Button, baseColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ScaleColor(baseColor, 1.2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ScaleColor(baseColor, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_Text, textColor);
    
    bool clicked = ImGui::Button(label, size);
    
    ImGui::PopStyleColor(4);
    
    return clicked;
}

bool PrimaryButton(const char* label, const ImVec2& size, bool enabled) {
    const auto& theme = ThemeManager::instance().current();
    
    if (!enabled)
        ImGui::BeginDisabled();
    
    ImGui::PushStyleColor(ImGuiCol_Button, theme.colors.primary);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ScaleColor(theme.colors.primary, 1.2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ScaleColor(theme.colors.primary, 0.8f));
    
    bool clicked = ImGui::Button(label, size);
    
    ImGui::PopStyleColor(3);
    
    if (!enabled)
        ImGui::EndDisabled();
    
    return clicked && enabled;
}

bool SecondaryButton(const char* label, const ImVec2& size, bool enabled) {
    const auto& theme = ThemeManager::instance().current();
    
    if (!enabled)
        ImGui::BeginDisabled();
    
    ImGui::PushStyleColor(ImGuiCol_Button, theme.colors.secondary);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ScaleColor(theme.colors.secondary, 1.3f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ScaleColor(theme.colors.secondary, 0.7f));
    
    bool clicked = ImGui::Button(label, size);
    
    ImGui::PopStyleColor(3);
    
    if (!enabled)
        ImGui::EndDisabled();
    
    return clicked && enabled;
}

bool DangerButton(const char* label, const ImVec2& size, bool enabled) {
    const auto& theme = ThemeManager::instance().current();

    if (!enabled)
        ImGui::BeginDisabled();

    ImGui::PushStyleColor(ImGuiCol_Button, theme.colors.error);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ScaleColor(theme.colors.error, 1.2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ScaleColor(theme.colors.error, 0.8f));

    bool clicked = ImGui::Button(label, size);

    ImGui::PopStyleColor(3);

    if (!enabled)
        ImGui::EndDisabled();

    return clicked && enabled;
}

bool SliderWithHelp(const char* label, float* value, float min, float max,
                    const char* tooltip, const char* format) {
    bool changed = ImGui::SliderFloat(label, value, min, max, format);
    if (tooltip)
        HelpMarker(tooltip);
    return changed;
}

bool DragIntWithHelp(const char* label, int* value, float speed, int min, int max,
                     const char* tooltip) {
    bool changed = ImGui::DragInt(label, value, speed, min, max);
    if (tooltip)
        HelpMarker(tooltip);
    return changed;
}

bool DragFloatWithHelp(const char* label, float* value, float speed, float min, float max,
                       const char* tooltip, const char* format) {
    bool changed = ImGui::DragFloat(label, value, speed, min, max, format);
    if (tooltip)
        HelpMarker(tooltip);
    return changed;
}

void ColoredHeader(const char* text, const ImVec4& color) {
    ImGui::TextColored(color, "%s", text);
}

void Divider() {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
}

void StatusIndicator(const char* text, StatusType status) {
    ImVec4 color;
    switch (status) {
        case StatusType::Success: color = ImVec4(0.3f, 1.0f, 0.3f, 1.0f); break;
        case StatusType::Warning: color = ImVec4(1.0f, 0.8f, 0.0f, 1.0f); break;
        case StatusType::Error:   color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f); break;
        case StatusType::Info:    
        default:                  color = ImVec4(0.4f, 0.7f, 1.0f, 1.0f); break;
    }
    
    // KÃ¼Ã§Ã¼k renkli daire
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    float radius = 4.0f;
    drawList->AddCircleFilled(
        ImVec2(pos.x + radius + 2, pos.y + ImGui::GetTextLineHeight() * 0.5f),
        radius,
        ImGui::ColorConvertFloat4ToU32(color)
    );
    
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + radius * 2 + 8);
    ImGui::TextColored(color, "%s", text);
}

void ProgressBarEx(float fraction, const ImVec2& size, const char* overlay,
                   const ImVec4& barColor) {
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barColor);
    ImGui::ProgressBar(fraction, size, overlay);
    ImGui::PopStyleColor();
}

float GetInspectorItemWidth() {
    float avail = ImGui::GetContentRegionAvail().x;
    float w = avail * 0.45f;
    if (w > 260.0f) w = 260.0f;
    if (w < 160.0f) w = 160.0f;
    return w;
}

float GetInspectorActionWidth() {
    float avail = ImGui::GetContentRegionAvail().x;
    float w = avail * 0.45f;;
    if (w > 260.0f) w = 260.0f; // Action buttons max width
    if (w < 160.0f) w = 160.0f;
    return w;
}

float GetRightAlignOffset(float widgetWidth) {
    return ImGui::GetContentRegionAvail().x - widgetWidth;
}

void BeginLabelValuePair(const char* label, float labelWidth) {
    ImGui::Text("%s", label);
    ImGui::SameLine(labelWidth);
}

void EndLabelValuePair() {
    // Åu an iÃ§in boÅŸ - gelecekte ek iÅŸlevsellik iÃ§in
}

} // namespace UIWidgets

// ============================================================================
// PANEL MANAGER IMPLEMENTATION
// ============================================================================

PanelState& PanelManager::getState(const std::string& panelId) {
    return states_[panelId];
}

void PanelManager::saveStates(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) return;
    file.imbue(std::locale::classic());
    
    for (const auto& [id, state] : states_) {
        file << id << " "
             << state.isVisible << " "
             << state.isCollapsed << " "
             << state.lastPosition.x << " "
             << state.lastPosition.y << " "
             << state.lastSize.x << " "
             << state.lastSize.y << "\n";
    }
}

bool PanelManager::loadStates(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    file.imbue(std::locale::classic());
    
    std::string id;
    while (file >> id) {
        PanelState state;
        file >> state.isVisible
             >> state.isCollapsed
             >> state.lastPosition.x
             >> state.lastPosition.y
             >> state.lastSize.x
             >> state.lastSize.y;
        states_[id] = state;
    }
    
    return true;
}

// ============================================================================
// THEME UI IMPLEMENTATION
// ============================================================================
namespace UIWidgets {

void DrawThemeSelector(float& panel_alpha) {
    // Note: BeginSection is UIWidgets::BeginSection
    if (BeginSection("Interface / Theme", ImVec4(0.5f, 0.7f, 1.0f, 1.0f))) {
        
        auto& themeManager = ThemeManager::instance();
        auto themeNames = themeManager.getAllThemeNames();
        int currentThemeIdx = themeManager.currentIndex();

        if (ImGui::Combo("Select Theme", &currentThemeIdx, 
                         themeNames.data(), static_cast<int>(themeNames.size()))) {
            themeManager.setTheme(currentThemeIdx);
            themeManager.applyCurrentTheme(panel_alpha);
            themeManager.saveThemeSettings("theme.cfg", panel_alpha);
        }

        // Duplicate / Delete custom themes controls
        static char newThemeName[64] = "My Custom Theme";
        ImGui::PushItemWidth(-1.0f);
        ImGui::InputTextWithHint("##NewThemeName", "Theme Name...", newThemeName, IM_ARRAYSIZE(newThemeName));
        ImGui::PopItemWidth();

        float btn_w = ImGui::GetContentRegionAvail().x;
        bool isDeletable = (currentThemeIdx >= 2);
        if (isDeletable) {
            btn_w = (btn_w - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        }

        if (ImGui::Button("Duplicate Theme", ImVec2(btn_w, 0.0f))) {
            std::string nameStr(newThemeName);
            if (!nameStr.empty()) {
                Theme newTheme = themeManager.current();
                newTheme.name = nameStr;
                themeManager.addTheme(newTheme);
                
                int newIdx = themeManager.themeCount() - 1;
                themeManager.setTheme(newIdx);
                themeManager.applyCurrentTheme(panel_alpha);
                themeManager.saveCustomThemes("custom_themes.cfg");
                themeManager.saveThemeSettings("theme.cfg", panel_alpha);
                
                // Clear the theme name input text buffer after a successful duplication
                newThemeName[0] = '\0';
            }
        }
        if (isDeletable) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.70f, 0.20f, 0.20f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.30f, 0.30f, 1.0f));
            if (ImGui::Button("Delete Theme", ImVec2(btn_w, 0.0f))) {
                themeManager.deleteTheme(currentThemeIdx);
                themeManager.applyCurrentTheme(panel_alpha);
                themeManager.saveCustomThemes("custom_themes.cfg");
                themeManager.saveThemeSettings("theme.cfg", panel_alpha);
            }
            ImGui::PopStyleColor(2);
        }

        // Panel Transparency   
        if (ImGui::SliderFloat("Panel Transparency", &panel_alpha, 0.65f, 1.0f, "%.2f")) {
            ImGuiStyle& style = ImGui::GetStyle();
            style.Colors[ImGuiCol_WindowBg].w = panel_alpha;
            style.Colors[ImGuiCol_ChildBg].w = panel_alpha;
            themeManager.saveThemeSettings("theme.cfg", panel_alpha);
        }

        // Custom theme editing
        if (currentThemeIdx >= 1) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.90f, 0.65f, 0.20f, 1.0f), "Customize Theme Colors:");
            ImGui::Indent();
            
            auto& customTheme = themeManager.getThemeMutable(currentThemeIdx);
            bool changed = false;
            
            // Allow renaming user-created custom themes (index >= 2)
            if (currentThemeIdx >= 2) {
                char renameBuf[64] = "";
                strncpy(renameBuf, customTheme.name.c_str(), sizeof(renameBuf) - 1);
                renameBuf[sizeof(renameBuf) - 1] = '\0';
                if (ImGui::InputText("Theme Name", renameBuf, sizeof(renameBuf))) {
                    if (strlen(renameBuf) > 0) {
                        customTheme.name = renameBuf;
                        changed = true;
                    }
                }
            }
            
            // Roundings
            changed |= ImGui::SliderFloat("Window Rounding", &customTheme.style.windowRounding, 0.0f, 16.0f, "%.0f px");
            changed |= ImGui::SliderFloat("Frame Rounding", &customTheme.style.frameRounding, 0.0f, 16.0f, "%.0f px");
            changed |= ImGui::SliderFloat("Tab Rounding", &customTheme.style.tabRounding, 0.0f, 16.0f, "%.0f px");
            changed |= ImGui::SliderFloat("Grab Rounding", &customTheme.style.grabRounding, 0.0f, 16.0f, "%.0f px");
            
            // Colors
            changed |= ImGui::ColorEdit4("Primary (Buttons)", &customTheme.colors.primary.x);
            changed |= ImGui::ColorEdit4("Secondary (Tabs)", &customTheme.colors.secondary.x);
            changed |= ImGui::ColorEdit4("Accent Color", &customTheme.colors.accent.x);
            changed |= ImGui::ColorEdit4("Background", &customTheme.colors.background.x);
            changed |= ImGui::ColorEdit4("Surface (Panels)", &customTheme.colors.surface.x);
            changed |= ImGui::ColorEdit4("Text Color", &customTheme.colors.text.x);
            changed |= ImGui::ColorEdit4("Muted Text", &customTheme.colors.textMuted.x);
            changed |= ImGui::ColorEdit4("Border Color", &customTheme.colors.border.x);
            
            if (changed) {
                themeManager.applyCurrentTheme(panel_alpha);
                themeManager.saveCustomThemes("custom_themes.cfg");
                themeManager.saveThemeSettings("theme.cfg", panel_alpha);
            }
            ImGui::Unindent();
        }

        // Icon settings
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.0f), "Icon & Accent Settings:");
        ImGui::Indent();
        
        auto& iconSettings = themeManager.getIconSettings();
        bool iconChanged = false;
        
        const char* styleNames[] = { "Clay Matcap (Premium)", "Flat Minimalist", "Neon Glow", "Custom Palette" };
        int styleIdx = static_cast<int>(iconSettings.style);
        if (ImGui::Combo("Icon Style", &styleIdx, styleNames, IM_ARRAYSIZE(styleNames))) {
            iconSettings.style = static_cast<IconStyle>(styleIdx);
            iconChanged = true;
        }
        
        iconChanged |= ImGui::SliderFloat("Icon Size Scale", &iconSettings.scaleMultiplier, 0.5f, 2.0f, "%.2fx");
        iconChanged |= ImGui::SliderFloat("Icon Line Width", &iconSettings.thicknessMultiplier, 0.5f, 3.0f, "%.2fx");
        
        if (iconSettings.style == IconStyle::CustomPalette || iconSettings.style == IconStyle::FlatMinimalist || iconSettings.style == IconStyle::NeonGlow) {
            iconChanged |= ImGui::ColorEdit4("Icon Base Color", &iconSettings.customColor.x);
            if (iconSettings.style == IconStyle::CustomPalette || iconSettings.style == IconStyle::FlatMinimalist) {
                iconChanged |= ImGui::ColorEdit4("Icon Background", &iconSettings.customBgColor.x);
            }
        }
        else if (iconSettings.style == IconStyle::ClayMatcap) {
            iconChanged |= ImGui::ColorEdit3("Matcap Tint Color", &iconSettings.matcapColor.x);
            
            const char* matcapPresets[] = { "Neutral Gray", "Terracotta Red", "Sculpting Wax", "Jade Green", "Golden Metal" };
            ImVec4 matcapColors[] = {
                ImVec4(0.51f, 0.53f, 0.55f, 1.0f),
                ImVec4(0.70f, 0.32f, 0.24f, 1.0f),
                ImVec4(0.75f, 0.45f, 0.40f, 1.0f),
                ImVec4(0.24f, 0.62f, 0.40f, 1.0f),
                ImVec4(0.85f, 0.65f, 0.22f, 1.0f)
            };
            
            int currentPresetIdx = -1;
            for (int i = 0; i < 5; ++i) {
                if (std::abs(iconSettings.matcapColor.x - matcapColors[i].x) < 0.01f &&
                    std::abs(iconSettings.matcapColor.y - matcapColors[i].y) < 0.01f &&
                    std::abs(iconSettings.matcapColor.z - matcapColors[i].z) < 0.01f) {
                    currentPresetIdx = i;
                    break;
                }
            }
            
            const char* comboPreviewText = (currentPresetIdx >= 0) ? matcapPresets[currentPresetIdx] : "Custom Tint";
            
            if (ImGui::BeginCombo("Matcap Material", comboPreviewText)) {
                for (int i = 0; i < 5; ++i) {
                    bool isSelected = (currentPresetIdx == i);
                    
                    ImGui::PushID(i);
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    ImGui::Dummy(ImVec2(16.0f, 16.0f));
                    ImGui::SameLine();
                    
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddCircleFilled(ImVec2(pos.x + 8.0f, pos.y + 8.0f), 6.0f, ImGui::ColorConvertFloat4ToU32(matcapColors[i]));
                    dl->AddCircle(ImVec2(pos.x + 8.0f, pos.y + 8.0f), 6.0f, IM_COL32(200, 200, 200, 180), 12, 1.0f);
                    
                    if (ImGui::Selectable(matcapPresets[i], isSelected)) {
                        iconSettings.matcapColor = matcapColors[i];
                        iconChanged = true;
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                    ImGui::PopID();
                }
                ImGui::EndCombo();
            }
        }

        // Accent override checkmark
        iconChanged |= ImGui::Checkbox("Override Panel Accents with Active Theme Accent", &iconSettings.overridePanelAccentsWithTheme);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Unifies checkbox/slider colors across all side panels by matching the active theme's accent color.");
        }
        
        if (iconChanged) {
            themeManager.saveThemeSettings("theme.cfg", panel_alpha);
        }
        ImGui::Unindent();

        EndSection();
    }
}

void DrawIcon(IconType type, ImVec2 p, float s, ImU32 col, float thickness) {
    const auto& settings = ThemeManager::instance().getIconSettings();
    float s_orig = s;
    s *= settings.scaleMultiplier;
    float offset = (s_orig - s) * 0.5f;
    p.x += offset;
    p.y += offset;
    thickness *= settings.thicknessMultiplier;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    float pading = s * 0.08f;
    float is = s - pading * 2;
    ImVec2 cp = ImVec2(p.x + s * 0.5f, p.y + s * 0.5f);
    p.x += pading; p.y += pading;

    // Define macro-wrapper helper for IM_COL32
    auto getIconColor = [&](int r, int g, int b, int a) -> ImU32 {
        if (settings.style == IconStyle::FlatMinimalist) {
            ImVec4 cVec = ImGui::ColorConvertU32ToFloat4(col);
            float opacity = a / 255.0f;
            return ImGui::ColorConvertFloat4ToU32(ImVec4(cVec.x, cVec.y, cVec.z, cVec.w * opacity));
        }
        else if (settings.style == IconStyle::NeonGlow) {
            ImVec4 cVec = ImGui::ColorConvertU32ToFloat4(col);
            float opacity = a / 255.0f;
            return ImGui::ColorConvertFloat4ToU32(ImVec4(cVec.x * 1.3f, cVec.y * 1.3f, cVec.z * 1.3f, cVec.w * opacity));
        }
        else if (settings.style == IconStyle::CustomPalette) {
            float opacity = a / 255.0f;
            return ImGui::ColorConvertFloat4ToU32(ImVec4(settings.customColor.x, settings.customColor.y, settings.customColor.z, settings.customColor.w * opacity));
        }
        // ClayMatcap (standard)
        return (((ImU32)(a)<<24) | ((ImU32)(b)<<16) | ((ImU32)(g)<<8) | ((ImU32)(r)));
    };

    // Push macro override locally for IM_COL32
#pragma push_macro("IM_COL32")
#undef IM_COL32
#define IM_COL32(r,g,b,a) getIconColor(r,g,b,a)

    // Dynamic Clay Matcap Palette based on settings.matcapColor
    ImVec4 mc = settings.matcapColor;
    auto makeColor = [](float r, float g, float b, float a) -> ImU32 {
        return (((ImU32)(a * 255.0f)<<24) | ((ImU32)(b * 255.0f)<<16) | ((ImU32)(g * 255.0f)<<8) | ((ImU32)(r * 255.0f)));
    };

    const ImU32 clay_shadow          = makeColor(mc.x * 0.38f, mc.y * 0.39f, mc.z * 0.42f, 1.0f);    // Dark slate shadow base
    const ImU32 clay_diffuse         = makeColor(mc.x, mc.y, mc.z, 1.0f); // Middle neutral gray matcap base
    const ImU32 clay_specular        = makeColor(1.0f, 1.0f, 1.0f, 0.63f); // Specular white highlight
    const ImU32 clay_buildup         = makeColor((std::min)(1.0f, mc.x * 1.15f), (std::min)(1.0f, mc.y * 1.15f), (std::min)(1.0f, mc.z * 1.15f), 1.0f); // Middle-light gray for buildup shapes
    const ImU32 clay_highlight       = makeColor((std::min)(1.0f, mc.x * 1.54f), (std::min)(1.0f, mc.y * 1.54f), (std::min)(1.0f, mc.z * 1.54f), 1.0f); // Edge highlight lines
    const ImU32 clay_detail_shadow   = makeColor(mc.x * 0.27f, mc.y * 0.27f, mc.z * 0.29f, 1.0f);    // Deep indent/detail shadow
    const ImU32 clay_trans_shadow    = makeColor(mc.x * 0.38f, mc.y * 0.39f, mc.z * 0.42f, 0.70f);   // Semi-transparent shadow
    const ImU32 clay_trans_shadow_150 = makeColor(mc.x * 0.38f, mc.y * 0.39f, mc.z * 0.42f, 0.59f);  // Semi-transparent shadow (lighter)
    const ImU32 clay_trans_highlight = makeColor((std::min)(1.0f, mc.x * 1.54f), (std::min)(1.0f, mc.y * 1.54f), (std::min)(1.0f, mc.z * 1.54f), 0.47f); // Semi-transparent highlight


    auto drawBaseClaySphere = [&](ImVec2 center, float radius) {
        if (settings.style == IconStyle::FlatMinimalist) {
            dl->AddCircleFilled(center, radius, ImGui::ColorConvertFloat4ToU32(settings.customBgColor));
            dl->AddCircle(center, radius, col, 32, thickness * 0.7f);
        }
        else if (settings.style == IconStyle::NeonGlow) {
            dl->AddCircleFilled(center, radius, IM_COL32(10, 15, 25, 120));
            ImVec4 neon = ImGui::ColorConvertU32ToFloat4(col);
            for (int i = 1; i <= 3; ++i) {
                dl->AddCircle(center, radius + i * 0.8f, ImGui::ColorConvertFloat4ToU32(ImVec4(neon.x, neon.y, neon.z, 0.18f / i)), 32, thickness * 1.5f);
            }
            dl->AddCircle(center, radius, col, 32, thickness);
        }
        else if (settings.style == IconStyle::CustomPalette) {
            dl->AddCircleFilled(center, radius, ImGui::ColorConvertFloat4ToU32(settings.customBgColor));
            dl->AddCircle(center, radius, ImGui::ColorConvertFloat4ToU32(settings.customColor), 32, thickness);
        }
        else {
            dl->AddCircleFilled(center, radius, clay_shadow);
            dl->AddCircleFilled(ImVec2(center.x - radius * 0.12f, center.y - radius * 0.12f), radius * 0.85f, clay_diffuse);
            dl->AddCircleFilled(ImVec2(center.x - radius * 0.25f, center.y - radius * 0.25f), radius * 0.22f, clay_specular);
        }
    };

    switch (type) {
        case IconType::Scene:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Glowing orange/cyan node tree hierarchy on top of the sphere
                ImVec2 n0 = ImVec2(cp.x, cp.y - r * 0.4f);
                ImVec2 n1 = ImVec2(cp.x - r * 0.4f, cp.y + r * 0.2f);
                ImVec2 n2 = ImVec2(cp.x + r * 0.4f, cp.y + r * 0.2f);
                
                dl->AddLine(n0, ImVec2(n0.x, cp.y), IM_COL32(100, 200, 255, 180), thickness);
                dl->AddLine(ImVec2(n1.x, cp.y), ImVec2(n2.x, cp.y), IM_COL32(100, 200, 255, 180), thickness);
                dl->AddLine(ImVec2(n1.x, cp.y), n1, IM_COL32(100, 200, 255, 180), thickness);
                dl->AddLine(ImVec2(n2.x, cp.y), n2, IM_COL32(100, 200, 255, 180), thickness);
                
                dl->AddCircleFilled(n0, 3.5f, IM_COL32(255, 150, 50, 255)); // Orange parent node
                dl->AddCircleFilled(n1, 3.0f, IM_COL32(100, 220, 255, 255)); // Cyan child 1
                dl->AddCircleFilled(n2, 3.0f, IM_COL32(100, 220, 255, 255)); // Cyan child 2
            }
            break;
        case IconType::Render:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Camera aperture rings and blades in gold/white
                float apR = r * 0.55f;
                dl->AddCircle(cp, apR, IM_COL32(255, 200, 50, 200), 16, thickness);
                
                for (int i = 0; i < 6; ++i) {
                    float a = i * (6.28318f / 6.0f);
                    ImVec2 p0(cp.x + cosf(a) * apR, cp.y + sinf(a) * apR);
                    float inner_a = a + 1.1f;
                    ImVec2 p1(cp.x + cosf(inner_a) * (apR * 0.4f), cp.y + sinf(inner_a) * (apR * 0.4f));
                    dl->AddLine(p0, p1, IM_COL32(255, 200, 50, 200), thickness * 0.9f);
                }
                
                dl->AddCircleFilled(cp, apR * 0.25f, IM_COL32(255, 255, 255, 120)); // glowing center lens
            }
            break;
        case IconType::Terrain:
            {
                float r = is * 0.48f;
                // Deformed mountain/terrain shape instead of a clean sphere
                ImVec2 t_top(cp.x, cp.y - r * 0.6f);
                ImVec2 t_left(cp.x - r * 0.8f, cp.y + r * 0.6f);
                ImVec2 t_right(cp.x + r * 0.8f, cp.y + r * 0.6f);
                ImVec2 t_mid(cp.x - r * 0.2f, cp.y + r * 0.2f);
                
                // Base mountain shadow (left side dark, right side light)
                ImVec2 faceL[] = { t_top, t_mid, t_left };
                ImVec2 faceR[] = { t_top, t_mid, t_right };
                
                dl->AddConvexPolyFilled(faceL, 3, IM_COL32(80, 110, 80, 255)); // Forest green shadow side
                dl->AddConvexPolyFilled(faceR, 3, IM_COL32(110, 140, 100, 255)); // Lighter green side
                
                // Wireframe terrain grid lines on top of the mountain
                dl->AddTriangle(t_top, t_left, t_mid, IM_COL32(255, 255, 255, 60), 0.8f);
                dl->AddTriangle(t_top, t_right, t_mid, IM_COL32(255, 255, 255, 80), 0.8f);
                
                // Draw mountain outlines
                dl->AddLine(t_top, t_left, IM_COL32(200, 220, 200, 255), thickness);
                dl->AddLine(t_top, t_right, IM_COL32(200, 220, 200, 255), thickness);
                dl->AddLine(t_left, t_right, IM_COL32(150, 180, 150, 255), thickness);
                dl->AddLine(t_top, t_mid, IM_COL32(200, 220, 200, 255), thickness);
            }
            break;
        case IconType::Sculpt:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Golden carving wire loop tool scraping the sphere
                ImVec2 loopCenter = ImVec2(cp.x + r * 0.1f, cp.y - r * 0.1f);
                float loopR = r * 0.45f;
                
                // Scraped clay crease/mark
                dl->PathClear();
                dl->PathArcTo(cp, r * 0.8f, 1.8f, 3.5f);
                dl->PathStroke(clay_detail_shadow, false, thickness * 2.0f);
                
                // Tool metal wire loop
                dl->AddCircle(loopCenter, loopR, IM_COL32(255, 200, 50, 255), 16, thickness * 1.5f);
                
                // Tool shaft/handle
                ImVec2 handleStart = ImVec2(loopCenter.x + cosf(0.78f) * loopR, loopCenter.y - sinf(0.78f) * loopR);
                ImVec2 handleEnd = ImVec2(handleStart.x + r * 0.6f, handleStart.y - r * 0.6f);
                dl->AddLine(handleStart, handleEnd, IM_COL32(180, 180, 180, 255), thickness * 2.0f);
            }
            break;
        case IconType::Hair:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Three flowing golden strands wrapping the sphere
                ImU32 hairCol = IM_COL32(255, 180, 70, 230);
                dl->AddBezierQuadratic(ImVec2(cp.x - r * 0.8f, cp.y + r * 0.4f),
                                       ImVec2(cp.x - r * 0.2f, cp.y - r * 0.6f),
                                       ImVec2(cp.x + r * 0.6f, cp.y - r * 0.4f), hairCol, thickness * 1.5f);
                                       
                dl->AddBezierQuadratic(ImVec2(cp.x - r * 0.6f, cp.y + r * 0.7f),
                                       ImVec2(cp.x + r * 0.1f, cp.y - r * 0.5f),
                                       ImVec2(cp.x + r * 0.7f, cp.y - r * 0.1f), hairCol, thickness * 1.2f);
                                       
                dl->AddBezierQuadratic(ImVec2(cp.x - r * 0.8f, cp.y - r * 0.1f),
                                       ImVec2(cp.x - r * 0.3f, cp.y - r * 0.8f),
                                       ImVec2(cp.x + r * 0.4f, cp.y - r * 0.7f), IM_COL32(255, 210, 120, 180), thickness * 0.9f);
            }
            break;
        case IconType::Brush: // Used as Stylize Mode / Mix Behavior
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Mix blend strokes (Violet and Orange semi-transparent waves)
                dl->PathClear();
                dl->PathArcTo(cp, r * 0.65f, 2.2f, 4.2f);
                dl->PathStroke(IM_COL32(180, 100, 255, 180), false, thickness * 2.2f);
                
                dl->PathClear();
                dl->PathArcTo(cp, r * 0.65f, 3.2f, 5.2f);
                dl->PathStroke(IM_COL32(255, 170, 50, 180), false, thickness * 2.2f);
                
                // Magic Wand overlay (slanted gold stick with white star spark)
                ImVec2 wandStart = ImVec2(cp.x - r * 0.5f, cp.y + r * 0.5f);
                ImVec2 wandEnd = ImVec2(cp.x + r * 0.3f, cp.y - r * 0.3f);
                dl->AddLine(wandStart, wandEnd, IM_COL32(220, 220, 220, 255), thickness * 1.5f); // metal shaft
                dl->AddLine(ImVec2(wandEnd.x - r * 0.15f, wandEnd.y + r * 0.15f), wandEnd, IM_COL32(255, 215, 0, 255), thickness * 1.5f); // golden tip
                
                // Sparkle at wand tip
                dl->AddCircleFilled(wandEnd, 1.8f, IM_COL32(255, 255, 255, 255));
                dl->AddCircle(wandEnd, 3.0f, IM_COL32(255, 255, 255, 150), 8, 1.0f);
            }
            break;
        case IconType::Move:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // RGB translate axes
                float axLen = r * 0.80f;
                float arrSz = is * 0.08f;
                
                // X Axis - Red (Right)
                ImVec2 xEnd(cp.x + axLen, cp.y);
                dl->AddLine(cp, xEnd, IM_COL32(255, 75, 75, 240), thickness * 1.6f);
                dl->AddTriangleFilled(xEnd, ImVec2(xEnd.x - arrSz, xEnd.y - arrSz * 0.5f), ImVec2(xEnd.x - arrSz, xEnd.y + arrSz * 0.5f), IM_COL32(255, 75, 75, 240));
                
                // Y Axis - Green (Up)
                ImVec2 yEnd(cp.x, cp.y - axLen);
                dl->AddLine(cp, yEnd, IM_COL32(75, 220, 75, 240), thickness * 1.6f);
                dl->AddTriangleFilled(yEnd, ImVec2(yEnd.x - arrSz * 0.5f, yEnd.y + arrSz), ImVec2(yEnd.x + arrSz * 0.5f, yEnd.y + arrSz), IM_COL32(75, 220, 75, 240));
                
                // Z Axis - Blue (Down-Left)
                ImVec2 zEnd(cp.x - axLen * 0.707f, cp.y + axLen * 0.707f);
                dl->AddLine(cp, zEnd, IM_COL32(75, 140, 255, 240), thickness * 1.6f);
                float aZ = 2.356f;
                dl->AddTriangleFilled(zEnd, 
                    ImVec2(zEnd.x - arrSz * cosf(aZ - 0.4f), zEnd.y - arrSz * sinf(aZ - 0.4f)),
                    ImVec2(zEnd.x - arrSz * cosf(aZ + 0.4f), zEnd.y - arrSz * sinf(aZ + 0.4f)),
                    IM_COL32(75, 140, 255, 240));
            }
            break;
        case IconType::Rotate:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // RGB rotation track rings
                // X Ring - Red (Horizontal-ish ellipse)
                dl->AddEllipse(cp, ImVec2(r * 0.85f, r * 0.30f), IM_COL32(255, 75, 75, 200), -0.2f, 0, thickness * 1.4f);
                
                // Y Ring - Green (Vertical-ish ellipse)
                dl->AddEllipse(cp, ImVec2(r * 0.30f, r * 0.85f), IM_COL32(75, 220, 75, 200), 0.2f, 0, thickness * 1.4f);
                
                // Z Ring - Blue (Outer circular boundary)
                dl->AddCircle(cp, r * 0.85f, IM_COL32(75, 140, 255, 220), 32, thickness * 1.4f);
                
                // Active rotation arrow on the outer Z ring
                float arrAngle = -0.5f;
                ImVec2 arrPt(cp.x + cosf(arrAngle) * r * 0.85f, cp.y + sinf(arrAngle) * r * 0.85f);
                float headAngle = arrAngle + 1.57f;
                float arrSz = is * 0.08f;
                dl->AddTriangleFilled(arrPt,
                    ImVec2(arrPt.x - arrSz * cosf(headAngle - 0.4f), arrPt.y - arrSz * sinf(headAngle - 0.4f)),
                    ImVec2(arrPt.x - arrSz * cosf(headAngle + 0.4f), arrPt.y - arrSz * sinf(headAngle + 0.4f)),
                    IM_COL32(75, 140, 255, 255));
            }
            break;
        case IconType::ScaleAxis:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // RGB Scale axes (ending with box handles)
                float axLen = r * 0.80f;
                float boxSz = is * 0.08f;
                
                // X Axis - Red
                ImVec2 xEnd(cp.x + axLen, cp.y);
                dl->AddLine(cp, xEnd, IM_COL32(255, 75, 75, 240), thickness * 1.6f);
                dl->AddRectFilled(ImVec2(xEnd.x - boxSz*0.5f, xEnd.y - boxSz*0.5f), ImVec2(xEnd.x + boxSz*0.5f, xEnd.y + boxSz*0.5f), IM_COL32(255, 75, 75, 240), 1.0f);
                
                // Y Axis - Green
                ImVec2 yEnd(cp.x, cp.y - axLen);
                dl->AddLine(cp, yEnd, IM_COL32(75, 220, 75, 240), thickness * 1.6f);
                dl->AddRectFilled(ImVec2(yEnd.x - boxSz*0.5f, yEnd.y - boxSz*0.5f), ImVec2(yEnd.x + boxSz*0.5f, yEnd.y + boxSz*0.5f), IM_COL32(75, 220, 75, 240), 1.0f);
                
                // Z Axis - Blue
                ImVec2 zEnd(cp.x - axLen * 0.707f, cp.y + axLen * 0.707f);
                dl->AddLine(cp, zEnd, IM_COL32(75, 140, 255, 240), thickness * 1.6f);
                dl->AddRectFilled(ImVec2(zEnd.x - boxSz*0.5f, zEnd.y - boxSz*0.5f), ImVec2(zEnd.x + boxSz*0.5f, zEnd.y + boxSz*0.5f), IM_COL32(75, 140, 255, 240), 1.0f);
            }
            break;
        case IconType::Gizmo:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Combined layout: outer white boundary, X/Y axes lines, amber center dot
                dl->AddCircle(cp, r * 0.85f, IM_COL32(240, 240, 240, 150), 32, thickness * 0.9f);
                dl->AddLine(ImVec2(cp.x, cp.y - r * 0.85f), ImVec2(cp.x, cp.y + r * 0.85f), IM_COL32(240, 240, 240, 150), thickness * 0.9f);
                dl->AddLine(ImVec2(cp.x - r * 0.85f, cp.y), ImVec2(cp.x + r * 0.85f, cp.y), IM_COL32(240, 240, 240, 150), thickness * 0.9f);
                
                // Center point glow (amber)
                dl->AddCircleFilled(cp, is * 0.12f, IM_COL32(255, 180, 50, 240));
                dl->AddCircle(cp, is * 0.12f, IM_COL32(255, 225, 100, 255), 12, 1.0f);
            }
            break;
        case IconType::ViewSolid:
            {
                // Solid Mode: 3D Matte Clay Cube
                ImVec2 V_top(cp.x, cp.y - is * 0.42f);
                ImVec2 V_bottom(cp.x, cp.y + is * 0.42f);
                ImVec2 V_tl(cp.x - is * 0.36f, cp.y - is * 0.21f);
                ImVec2 V_tr(cp.x + is * 0.36f, cp.y - is * 0.21f);
                ImVec2 V_bl(cp.x - is * 0.36f, cp.y + is * 0.21f);
                ImVec2 V_br(cp.x + is * 0.36f, cp.y + is * 0.21f);

                ImVec2 top_face[] = { cp, V_tl, V_top, V_tr };
                ImVec2 left_face[] = { cp, V_tl, V_bl, V_bottom };
                ImVec2 right_face[] = { cp, V_tr, V_br, V_bottom };

                // Shaded faces using the Matcap Clay Palette
                dl->AddConvexPolyFilled(top_face, 4, clay_buildup);
                dl->AddConvexPolyFilled(left_face, 4, clay_diffuse);
                dl->AddConvexPolyFilled(right_face, 4, clay_shadow);

                // Highlight/Border lines
                ImVec2 hex[] = { V_top, V_tr, V_br, V_bottom, V_bl, V_tl };
                dl->AddPolyline(hex, 6, clay_highlight, ImDrawFlags_Closed, thickness * 0.8f);
                dl->AddLine(cp, V_bottom, clay_highlight, thickness * 0.8f);
                dl->AddLine(cp, V_tl, clay_highlight, thickness * 0.8f);
                dl->AddLine(cp, V_tr, clay_highlight, thickness * 0.8f);
                
                // Add a small specular highlight dot on the top-left face to simulate the matcap light source
                dl->AddCircleFilled(ImVec2(cp.x - is * 0.12f, cp.y - is * 0.18f), is * 0.05f, clay_specular);
            }
            break;
        case IconType::ViewMatcap:
            {
                float rad = is * 0.44f;
                // Terracotta clay matcap colors
                const ImU32 terracotta_shadow = IM_COL32(90, 40, 30, 255);
                const ImU32 terracotta_diffuse = IM_COL32(185, 95, 75, 255);
                const ImU32 terracotta_specular = IM_COL32(255, 215, 195, 180);
                
                // Draw shaded sphere
                dl->AddCircleFilled(cp, rad, terracotta_shadow);
                dl->AddCircleFilled(ImVec2(cp.x - rad * 0.12f, cp.y - rad * 0.12f), rad * 0.85f, terracotta_diffuse);
                dl->AddCircleFilled(ImVec2(cp.x - rad * 0.25f, cp.y - rad * 0.25f), rad * 0.22f, terracotta_specular);
                
                // Thin golden-orange outline
                dl->AddCircle(cp, rad, IM_COL32(255, 140, 80, 100), 32, thickness * 0.8f);
            }
            break;
        case IconType::ViewPreview:
            {
                float rad = is * 0.44f;
                // Base shadow
                dl->AddCircleFilled(cp, rad, clay_shadow);
                
                // Left half: Glossy blue metallic
                dl->PathClear();
                dl->PathArcTo(cp, rad, 0.5f * 3.14159f, 1.5f * 3.14159f);
                dl->PathFillConvex(IM_COL32(20, 80, 160, 255));
                
                dl->PathClear();
                dl->PathArcTo(ImVec2(cp.x - rad * 0.12f, cp.y - rad * 0.12f), rad * 0.85f, 0.5f * 3.14159f, 1.5f * 3.14159f);
                dl->PathFillConvex(IM_COL32(60, 160, 250, 255));
                
                // Right half: Neutral clay gray
                dl->PathClear();
                dl->PathArcTo(ImVec2(cp.x - rad * 0.12f, cp.y - rad * 0.12f), rad * 0.85f, -0.5f * 3.14159f, 0.5f * 3.14159f);
                dl->PathFillConvex(clay_diffuse);
                
                // Grid lines (UV coordinate layout)
                ImU32 gridCol = IM_COL32(255, 255, 255, 80);
                dl->AddLine(ImVec2(cp.x - rad, cp.y), ImVec2(cp.x + rad, cp.y), gridCol, thickness * 0.7f);
                dl->AddLine(ImVec2(cp.x, cp.y - rad), ImVec2(cp.x, cp.y + rad), gridCol, thickness * 0.7f);
                
                // Curved longitude grid lines
                dl->AddBezierQuadratic(ImVec2(cp.x, cp.y - rad), ImVec2(cp.x - rad * 0.4f, cp.y), ImVec2(cp.x, cp.y + rad), gridCol, thickness * 0.7f);
                dl->AddBezierQuadratic(ImVec2(cp.x, cp.y - rad), ImVec2(cp.x + rad * 0.4f, cp.y), ImVec2(cp.x, cp.y + rad), gridCol, thickness * 0.7f);
                
                // Specular highlight overlapping the boundary
                dl->AddCircleFilled(ImVec2(cp.x - rad * 0.25f, cp.y - rad * 0.25f), rad * 0.22f, clay_specular);
                
                // Outer ring
                dl->AddCircle(cp, rad, clay_highlight, 32, thickness * 0.8f);
            }
            break;
        case IconType::ViewRendered:
            {
                float rad = is * 0.42f;
                // Soft ground shadow under the sphere
                dl->PathClear();
                const int num_segments = 16;
                const float rx = is * 0.35f;
                const float ry = is * 0.08f;
                const ImVec2 shadow_center(cp.x, cp.y + is * 0.38f);
                for (int i = 0; i < num_segments; ++i) {
                    float a = i * (6.283185f / num_segments);
                    dl->PathLineTo(ImVec2(shadow_center.x + cosf(a) * rx, shadow_center.y + sinf(a) * ry));
                }
                dl->PathFillConvex(IM_COL32(0, 0, 0, 150));
                
                // Base clay sphere shadow
                dl->AddCircleFilled(cp, rad, clay_shadow);
                
                // Warm studio light diffuse overlay (offset top-left)
                dl->AddCircleFilled(ImVec2(cp.x - rad * 0.12f, cp.y - rad * 0.12f), rad * 0.85f, IM_COL32(145, 140, 135, 255));
                
                // Cool cyan bounce light on the bottom-right edge
                dl->PathClear();
                dl->PathArcTo(cp, rad - 1.0f, 0.0f, 1.57f, 16);
                dl->PathStroke(IM_COL32(100, 220, 255, 120), false, thickness * 1.8f);
                
                // Warm golden-white specular highlight spot (offset top-left)
                dl->AddCircleFilled(ImVec2(cp.x - rad * 0.25f, cp.y - rad * 0.25f), rad * 0.22f, IM_COL32(255, 245, 220, 200));
                
                // Outer ring
                dl->AddCircle(cp, rad, clay_highlight, 32, thickness * 0.8f);
                
                // Beautiful 4-point golden star/sparkle at top-right (monochrome ray reflection)
                ImVec2 sc = ImVec2(cp.x + rad * 0.45f, cp.y - rad * 0.45f);
                float sr = is * 0.24f;
                ImVec2 star_pts[] = {
                    ImVec2(sc.x, sc.y - sr),
                    ImVec2(sc.x + sr * 0.22f, sc.y - sr * 0.22f),
                    ImVec2(sc.x + sr, sc.y),
                    ImVec2(sc.x + sr * 0.22f, sc.y + sr * 0.22f),
                    ImVec2(sc.x, sc.y + sr),
                    ImVec2(sc.x - sr * 0.22f, sc.y + sr * 0.22f),
                    ImVec2(sc.x - sr, sc.y),
                    ImVec2(sc.x - sr * 0.22f, sc.y - sr * 0.22f)
                };
                dl->AddConvexPolyFilled(star_pts, 8, IM_COL32(255, 220, 100, 240));
                dl->AddPolyline(star_pts, 8, IM_COL32(255, 240, 180, 200), ImDrawFlags_Closed, 1.0f);
            }
            break;
        case IconType::CameraHud:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 hud_col = IM_COL32(80, 220, 255, 220); // Cyan/blue-green
                // Camera outer frame (corner ticks or thin rect)
                float rect_w = r * 0.7f;
                float rect_h = r * 0.5f;
                dl->AddRect(ImVec2(cp.x - rect_w * 0.5f, cp.y - rect_h * 0.5f), ImVec2(cp.x + rect_w * 0.5f, cp.y + rect_h * 0.5f), hud_col, 1.0f, 0, thickness * 0.8f);
                // Lens circle in center
                dl->AddCircle(cp, r * 0.18f, hud_col, 12, thickness * 0.8f);
                // Tiny top-right led/dot
                dl->AddCircleFilled(ImVec2(cp.x + rect_w * 0.35f, cp.y - rect_h * 0.35f), 1.5f, IM_COL32(255, 75, 75, 240));
            }
            break;
        case IconType::ViewOverlays:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 overlay_col = IM_COL32(255, 180, 50, 220); // Orange/Amber
                // Four vertical bar lines at different heights on the sphere
                float w = r * 0.5f;
                dl->AddLine(ImVec2(cp.x - w, cp.y + r * 0.25f), ImVec2(cp.x - w, cp.y - r * 0.1f), overlay_col, thickness * 0.9f);
                dl->AddLine(ImVec2(cp.x - w * 0.33f, cp.y + r * 0.35f), ImVec2(cp.x - w * 0.33f, cp.y - r * 0.4f), overlay_col, thickness * 0.9f);
                dl->AddLine(ImVec2(cp.x + w * 0.33f, cp.y + r * 0.35f), ImVec2(cp.x + w * 0.33f, cp.y - r * 0.2f), overlay_col, thickness * 0.9f);
                dl->AddLine(ImVec2(cp.x + w, cp.y + r * 0.25f), ImVec2(cp.x + w, cp.y - r * 0.5f), overlay_col, thickness * 0.9f);
            }
            break;
        case IconType::PivotEdit:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 pivot_col = IM_COL32(255, 215, 0, 240); // Gold
                float l = r * 0.75f;
                dl->AddCircle(cp, r * 0.25f, pivot_col, 16, thickness * 0.9f);
                dl->AddLine(ImVec2(cp.x, cp.y - l), ImVec2(cp.x, cp.y + l), pivot_col, thickness * 0.9f);
                dl->AddLine(ImVec2(cp.x - l, cp.y), ImVec2(cp.x + l, cp.y), pivot_col, thickness * 0.9f);
            }
            break;
        case IconType::PivotCenter:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 pivot_col = IM_COL32(255, 215, 0, 240); // Gold
                dl->AddCircle(cp, r * 0.20f, pivot_col, 12, thickness * 0.9f);
                dl->AddCircle(cp, r * 0.50f, pivot_col, 18, thickness * 0.8f);
                // Center ticks
                float t0 = r * 0.60f;
                float t1 = r * 0.85f;
                dl->AddLine(ImVec2(cp.x, cp.y - t1), ImVec2(cp.x, cp.y - t0), pivot_col, thickness * 0.8f);
                dl->AddLine(ImVec2(cp.x, cp.y + t1), ImVec2(cp.x, cp.y + t0), pivot_col, thickness * 0.8f);
                dl->AddLine(ImVec2(cp.x - t1, cp.y), ImVec2(cp.x - t0, cp.y), pivot_col, thickness * 0.8f);
                dl->AddLine(ImVec2(cp.x + t1, cp.y), ImVec2(cp.x + t0, cp.y), pivot_col, thickness * 0.8f);
            }
            break;
        case IconType::Sensitivity:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 sens_col = IM_COL32(230, 230, 230, 240); // Off-white
                dl->AddCircle(cp, r * 0.65f, sens_col, 24, thickness * 0.8f);
                // Pointer needle in bright orange/red
                ImU32 needle_col = IM_COL32(255, 90, 50, 255);
                dl->AddLine(cp, ImVec2(cp.x + r * 0.45f * cosf(-0.785f), cp.y + r * 0.45f * sinf(-0.785f)), needle_col, thickness * 1.2f);
                dl->AddCircleFilled(cp, 2.0f, needle_col);
                // Gauge markings
                for (int i = 0; i < 4; ++i) {
                    float angle = -3.14159f * 0.5f + i * (3.14159f * 0.5f);
                    dl->AddLine(ImVec2(cp.x + r * 0.5f * cosf(angle), cp.y + r * 0.5f * sinf(angle)),
                                ImVec2(cp.x + r * 0.65f * cosf(angle), cp.y + r * 0.65f * sinf(angle)),
                                sens_col, thickness * 0.8f);
                }
            }
            break;
        case IconType::Settings:
            dl->AddLine(ImVec2(p.x + is * 0.22f, p.y + is * 0.34f), ImVec2(p.x + is * 0.78f, p.y + is * 0.34f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.22f, p.y + is * 0.50f), ImVec2(p.x + is * 0.78f, p.y + is * 0.50f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.22f, p.y + is * 0.66f), ImVec2(p.x + is * 0.78f, p.y + is * 0.66f), col, thickness);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.38f, p.y + is * 0.34f), is * 0.07f, col, 12);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.62f, p.y + is * 0.50f), is * 0.07f, col, 12);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.46f, p.y + is * 0.66f), is * 0.07f, col, 12);
            break;
        case IconType::Play:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 play_col = IM_COL32(75, 220, 75, 240); // Bright green
                dl->AddTriangleFilled(
                    ImVec2(cp.x - r * 0.30f, cp.y - r * 0.50f),
                    ImVec2(cp.x - r * 0.30f, cp.y + r * 0.50f),
                    ImVec2(cp.x + r * 0.55f, cp.y),
                    play_col
                );
            }
            break;
        case IconType::Pause:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 pause_col = IM_COL32(75, 220, 75, 240); // Bright green
                dl->AddRectFilled(ImVec2(cp.x - r * 0.35f, cp.y - r * 0.50f), ImVec2(cp.x - r * 0.08f, cp.y + r * 0.50f), pause_col, 1.5f);
                dl->AddRectFilled(ImVec2(cp.x + r * 0.08f, cp.y - r * 0.50f), ImVec2(cp.x + r * 0.35f, cp.y + r * 0.50f), pause_col, 1.5f);
            }
            break;
        case IconType::Stop:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 stop_col = IM_COL32(255, 75, 75, 240); // Red
                dl->AddRectFilled(ImVec2(cp.x - r * 0.40f, cp.y - r * 0.40f), ImVec2(cp.x + r * 0.40f, cp.y + r * 0.40f), stop_col, 2.0f);
            }
            break;
        case IconType::Duplicate:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 key_col1 = IM_COL32(180, 180, 185, 240); // Back diamond (grayish/silver)
                ImVec2 cp1 = ImVec2(cp.x - r * 0.18f, cp.y - r * 0.18f);
                ImVec2 pts1[] = {
                    ImVec2(cp1.x, cp1.y - r * 0.35f),
                    ImVec2(cp1.x + r * 0.35f, cp1.y),
                    ImVec2(cp1.x, cp1.y + r * 0.35f),
                    ImVec2(cp1.x - r * 0.35f, cp1.y)
                };
                dl->AddConvexPolyFilled(pts1, 4, key_col1);
                dl->AddPolyline(pts1, 4, IM_COL32(255, 255, 255, 180), ImDrawFlags_Closed, thickness * 0.8f);
                
                ImU32 key_col2 = IM_COL32(255, 215, 0, 240); // Front diamond (golden)
                ImVec2 cp2 = ImVec2(cp.x + r * 0.18f, cp.y + r * 0.18f);
                ImVec2 pts2[] = {
                    ImVec2(cp2.x, cp2.y - r * 0.35f),
                    ImVec2(cp2.x + r * 0.35f, cp2.y),
                    ImVec2(cp2.x, cp2.y + r * 0.35f),
                    ImVec2(cp2.x - r * 0.35f, cp2.y)
                };
                dl->AddConvexPolyFilled(pts2, 4, key_col2);
                dl->AddPolyline(pts2, 4, IM_COL32(255, 255, 255, 200), ImDrawFlags_Closed, thickness * 0.8f);
            }
            break;
        case IconType::Help:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 q_col = IM_COL32(255, 255, 255, 240); // White
                // Top curve of the question mark
                dl->PathClear();
                dl->PathArcTo(ImVec2(cp.x, cp.y - r * 0.15f), r * 0.30f, -3.14159f, 0.0f, 16);
                dl->PathArcTo(ImVec2(cp.x + r * 0.15f, cp.y - r * 0.15f), r * 0.15f, 0.0f, 1.57f, 8);
                dl->PathLineTo(ImVec2(cp.x, cp.y + r * 0.15f));
                dl->PathStroke(q_col, false, thickness * 1.2f);
                
                // Dot
                dl->AddCircleFilled(ImVec2(cp.x, cp.y + r * 0.45f), thickness * 1.3f, q_col);
            }
            break;
        case IconType::AddKey:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 key_col = IM_COL32(80, 220, 255, 240); // Cyan/blue
                ImVec2 pts[] = {
                    ImVec2(cp.x, cp.y - r * 0.45f),
                    ImVec2(cp.x + r * 0.45f, cp.y),
                    ImVec2(cp.x, cp.y + r * 0.45f),
                    ImVec2(cp.x - r * 0.45f, cp.y)
                };
                dl->AddConvexPolyFilled(pts, 4, key_col);
                dl->AddPolyline(pts, 4, IM_COL32(255, 255, 255, 200), ImDrawFlags_Closed, thickness * 0.8f);
                
                // Plus sign overlay
                ImU32 sign_col = IM_COL32(255, 255, 255, 255);
                float pl = r * 0.18f;
                dl->AddLine(ImVec2(cp.x, cp.y - pl), ImVec2(cp.x, cp.y + pl), sign_col, thickness * 1.2f);
                dl->AddLine(ImVec2(cp.x - pl, cp.y), ImVec2(cp.x + pl, cp.y), sign_col, thickness * 1.2f);
            }
            break;
        case IconType::RemoveKey:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                ImU32 key_col = IM_COL32(255, 90, 50, 240); // Red-Orange
                ImVec2 pts[] = {
                    ImVec2(cp.x, cp.y - r * 0.45f),
                    ImVec2(cp.x + r * 0.45f, cp.y),
                    ImVec2(cp.x, cp.y + r * 0.45f),
                    ImVec2(cp.x - r * 0.45f, cp.y)
                };
                dl->AddConvexPolyFilled(pts, 4, key_col);
                dl->AddPolyline(pts, 4, IM_COL32(255, 255, 255, 200), ImDrawFlags_Closed, thickness * 0.8f);
                
                // Minus sign overlay
                ImU32 sign_col = IM_COL32(255, 255, 255, 255);
                float pl = r * 0.18f;
                dl->AddLine(ImVec2(cp.x - pl, cp.y), ImVec2(cp.x + pl, cp.y), sign_col, thickness * 1.2f);
            }
            break;
        case IconType::PaintTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Curved paint stroke in vibrant cyan/magenta across the sphere
                dl->PathClear();
                dl->PathArcTo(cp, r * 0.65f, 2.5f, 5.0f);
                dl->PathStroke(IM_COL32(0, 220, 255, 255), false, thickness * 2.2f);
                
                // Brush tip at the end of the stroke (top-right)
                ImVec2 brushPos = ImVec2(cp.x + cosf(5.0f) * r * 0.65f, cp.y + sinf(5.0f) * r * 0.65f);
                // Paint brush handle and bristle shape
                dl->AddLine(brushPos, ImVec2(brushPos.x + r * 0.4f, brushPos.y - r * 0.4f), IM_COL32(230, 160, 50, 255), thickness * 1.5f);
                dl->AddCircleFilled(brushPos, thickness * 1.8f, IM_COL32(0, 220, 255, 255));
            }
            break;
        case IconType::EraseTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Faded stroke under the eraser
                dl->PathClear();
                dl->PathArcTo(cp, r * 0.65f, 2.2f, 3.8f);
                dl->PathStroke(IM_COL32(255, 60, 60, 90), false, thickness * 2.0f);
                
                // Shaded Eraser block at top-right
                ImVec2 er = ImVec2(cp.x + r * 0.2f, cp.y - r * 0.2f);
                float esw = is * 0.24f;
                float esh = is * 0.14f;
                
                ImVec2 p0 = ImVec2(er.x, er.y);
                ImVec2 p1 = ImVec2(er.x + esw, er.y - esh);
                ImVec2 p2 = ImVec2(er.x + esw * 0.7f, er.y - esh - esh * 0.5f);
                ImVec2 p3 = ImVec2(er.x - esw * 0.3f, er.y - esh * 0.5f);
                
                ImVec2 topFace[] = { p0, p1, p2, p3 };
                dl->AddConvexPolyFilled(topFace, 4, IM_COL32(255, 120, 150, 255));
                dl->AddPolyline(topFace, 4, IM_COL32(255, 150, 180, 255), ImDrawFlags_Closed, 1.0f);
                
                ImVec2 p0_b = ImVec2(er.x, er.y + esh * 0.4f);
                ImVec2 p1_b = ImVec2(er.x + esw, er.y - esh + esh * 0.4f);
                ImVec2 sideFace[] = { p0, p1, p1_b, p0_b };
                dl->AddConvexPolyFilled(sideFace, 4, IM_COL32(240, 240, 240, 255));
                dl->AddPolyline(sideFace, 4, IM_COL32(255, 255, 255, 255), ImDrawFlags_Closed, 1.0f);
            }
            break;
        case IconType::SoftenTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Soft blue and purple waves representing blending
                ImVec2 wCenter = ImVec2(cp.x, cp.y + r * 0.1f);
                float wr = r * 0.65f;
                
                dl->AddCircleFilled(wCenter, wr * 0.7f, IM_COL32(100, 120, 255, 50));
                
                dl->AddBezierQuadratic(ImVec2(wCenter.x - wr, wCenter.y - wr * 0.2f),
                                       wCenter,
                                       ImVec2(wCenter.x + wr, wCenter.y - wr * 0.2f),
                                       IM_COL32(100, 150, 255, 180), thickness * 1.5f);
                                       
                dl->AddBezierQuadratic(ImVec2(wCenter.x - wr * 0.8f, wCenter.y + wr * 0.2f),
                                       ImVec2(wCenter.x, wCenter.y + wr * 0.4f),
                                       ImVec2(wCenter.x + wr * 0.8f, wCenter.y + wr * 0.2f),
                                       IM_COL32(160, 100, 255, 140), thickness * 1.2f);
            }
            break;
        case IconType::StampTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Star imprint on the sphere in gold
                ImVec2 starC = cp;
                float starR = r * 0.4f;
                ImVec2 star_pts[] = {
                    ImVec2(starC.x, starC.y - starR),
                    ImVec2(starC.x + starR * 0.25f, starC.y - starR * 0.25f),
                    ImVec2(starC.x + starR, starC.y),
                    ImVec2(starC.x + starR * 0.25f, starC.y + starR * 0.25f),
                    ImVec2(starC.x, starC.y + starR),
                    ImVec2(starC.x - starR * 0.25f, starC.y + starR * 0.25f),
                    ImVec2(starC.x - starR, starC.y),
                    ImVec2(starC.x - starR * 0.25f, starC.y - starR * 0.25f)
                };
                dl->AddConvexPolyFilled(star_pts, 8, IM_COL32(255, 200, 50, 180));
                
                // Wooden stamp handle
                ImVec2 st = ImVec2(cp.x - r * 0.1f, cp.y - r * 0.5f);
                dl->AddRectFilled(ImVec2(st.x - r * 0.2f, st.y - r * 0.3f), ImVec2(st.x + r * 0.2f, st.y), IM_COL32(160, 100, 60, 255), 3.0f);
                dl->AddCircleFilled(ImVec2(st.x, st.y - r * 0.35f), r * 0.18f, IM_COL32(130, 80, 40, 255));
            }
            break;
        case IconType::FillTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Half filled overlay
                dl->PathClear();
                dl->PathArcTo(cp, r, 0.25f * 3.14159f, 1.25f * 3.14159f);
                dl->PathFillConvex(IM_COL32(255, 140, 0, 130));
                
                // Paint bucket pouring paint
                ImVec2 bucket = ImVec2(cp.x + r * 0.4f, cp.y - r * 0.6f);
                dl->AddLine(bucket, ImVec2(bucket.x - r * 0.3f, bucket.y + r * 0.3f), IM_COL32(255, 140, 0, 255), 2.5f);
                
                ImVec2 b0 = bucket;
                ImVec2 b1 = ImVec2(bucket.x + r * 0.25f, bucket.y - r * 0.25f);
                ImVec2 b2 = ImVec2(bucket.x + r * 0.45f, bucket.y - r * 0.05f);
                ImVec2 b3 = ImVec2(bucket.x + r * 0.20f, bucket.y + r * 0.20f);
                ImVec2 bFace[] = { b0, b1, b2, b3 };
                dl->AddConvexPolyFilled(bFace, 4, IM_COL32(200, 200, 200, 255));
                dl->AddPolyline(bFace, 4, IM_COL32(255, 255, 255, 255), ImDrawFlags_Closed, 1.0f);
            }
            break;
        case IconType::CloneTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImVec2 cSource = ImVec2(cp.x - r * 0.4f, cp.y + r * 0.3f);
                ImVec2 cTarget = ImVec2(cp.x + r * 0.4f, cp.y - r * 0.3f);
                float cr = r * 0.24f;
                
                // Link line
                dl->AddLine(cSource, cTarget, IM_COL32(255, 255, 255, 120), thickness * 0.8f);
                
                // Source (Green-ish cyan)
                dl->AddCircle(cSource, cr, IM_COL32(50, 230, 200, 255), 12, thickness * 1.2f);
                dl->AddLine(ImVec2(cSource.x - cr * 1.4f, cSource.y), ImVec2(cSource.x + cr * 1.4f, cSource.y), IM_COL32(50, 230, 200, 255), thickness * 0.8f);
                dl->AddLine(ImVec2(cSource.x, cSource.y - cr * 1.4f), ImVec2(cSource.x, cSource.y + cr * 1.4f), IM_COL32(50, 230, 200, 255), thickness * 0.8f);
                
                // Target (Orange)
                dl->AddCircle(cTarget, cr, IM_COL32(255, 140, 50, 255), 12, thickness * 1.2f);
                dl->AddLine(ImVec2(cTarget.x - cr * 1.4f, cTarget.y), ImVec2(cTarget.x + cr * 1.4f, cTarget.y), IM_COL32(255, 140, 50, 255), thickness * 0.8f);
                dl->AddLine(ImVec2(cTarget.x, cTarget.y - cr * 1.4f), ImVec2(cTarget.x, cTarget.y + cr * 1.4f), IM_COL32(255, 140, 50, 255), thickness * 0.8f);
            }
            break;
        case IconType::SprayTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Nozzle
                ImVec2 nozzle = ImVec2(cp.x + r * 0.6f, cp.y - r * 0.6f);
                dl->AddLine(nozzle, ImVec2(nozzle.x - r * 0.2f, nozzle.y + r * 0.2f), IM_COL32(180, 180, 180, 255), thickness * 2.0f);
                
                // Colorful droplets
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.4f, cp.y - r * 0.2f), 1.8f, IM_COL32(255, 80, 80, 230));
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.1f, cp.y + r * 0.3f), 2.2f, IM_COL32(80, 255, 100, 230));
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.5f, cp.y + r * 0.2f), 1.2f, IM_COL32(80, 180, 255, 230));
                dl->AddCircleFilled(ImVec2(cp.x + r * 0.1f, cp.y - r * 0.1f), 1.5f, IM_COL32(255, 220, 80, 230));
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.2f, cp.y - r * 0.5f), 1.6f, IM_COL32(255, 100, 255, 230));
                dl->AddCircleFilled(ImVec2(cp.x + r * 0.2f, cp.y + r * 0.4f), 2.0f, IM_COL32(80, 255, 255, 230));
            }
            break;
        case IconType::SmudgeTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Smudge lines (curved trails representing blending)
                dl->PathClear();
                dl->PathArcTo(cp, r * 0.65f, 2.5f, 4.8f);
                dl->PathStroke(IM_COL32(180, 100, 255, 180), false, thickness * 2.2f);
                
                dl->PathClear();
                dl->PathArcTo(ImVec2(cp.x - r * 0.1f, cp.y + r * 0.1f), r * 0.6f, 2.5f, 4.8f);
                dl->PathStroke(IM_COL32(255, 100, 180, 140), false, thickness * 1.8f);
                
                // Finger tip circle smudge
                ImVec2 fingerPos = ImVec2(cp.x + cosf(4.8f) * r * 0.65f, cp.y + sinf(4.8f) * r * 0.65f);
                dl->AddCircleFilled(fingerPos, thickness * 2.5f, IM_COL32(230, 230, 250, 255));
                dl->AddCircle(fingerPos, thickness * 2.5f, IM_COL32(180, 100, 255, 255), 12, 1.0f);
            }
            break;
        case IconType::DodgeTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Glowing sun representing lightening exposure
                float sunR = r * 0.35f;
                dl->AddCircleFilled(cp, sunR, IM_COL32(255, 235, 100, 240));
                dl->AddCircle(cp, sunR, IM_COL32(255, 255, 255, 255), 16, thickness * 0.8f);
                
                for (int i = 0; i < 8; ++i) {
                    float angle = i * (6.2831853f / 8.0f);
                    ImVec2 rayStart(cp.x + cosf(angle) * (sunR + 1.5f), cp.y + sinf(angle) * (sunR + 1.5f));
                    ImVec2 rayEnd(cp.x + cosf(angle) * (sunR + r * 0.35f), cp.y + sinf(angle) * (sunR + r * 0.35f));
                    dl->AddLine(rayStart, rayEnd, IM_COL32(255, 220, 50, 240), thickness * 1.1f);
                }
            }
            break;
        case IconType::BurnTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Crescent moon representing darkening
                float moonR = r * 0.5f;
                ImVec2 mOffset = ImVec2(cp.x - r * 0.15f, cp.y - r * 0.15f);
                
                dl->PathClear();
                dl->PathArcTo(cp, moonR, -1.8f, 1.8f);
                dl->PathArcTo(mOffset, moonR * 0.9f, 1.8f, -1.8f, 16);
                dl->PathFillConvex(IM_COL32(110, 110, 240, 220));
                
                dl->PathClear();
                dl->PathArcTo(cp, moonR, -1.8f, 1.8f);
                dl->PathStroke(IM_COL32(200, 200, 255, 255), false, thickness * 0.8f);
            }
            break;
        case IconType::SharpenTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Sharp high contrast star representation
                ImVec2 sc = cp;
                float sr = r * 0.6f;
                ImVec2 star_pts[] = {
                    ImVec2(sc.x, sc.y - sr),
                    ImVec2(sc.x + sr * 0.20f, sc.y - sr * 0.20f),
                    ImVec2(sc.x + sr, sc.y),
                    ImVec2(sc.x + sr * 0.20f, sc.y + sr * 0.20f),
                    ImVec2(sc.x, sc.y + sr),
                    ImVec2(sc.x - sr * 0.20f, sc.y + sr * 0.20f),
                    ImVec2(sc.x - sr, sc.y),
                    ImVec2(sc.x - sr * 0.20f, sc.y - sr * 0.20f)
                };
                dl->AddConvexPolyFilled(star_pts, 8, IM_COL32(0, 240, 255, 180));
                dl->AddPolyline(star_pts, 8, IM_COL32(255, 255, 255, 255), ImDrawFlags_Closed, thickness * 1.2f);
                
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.4f, cp.y - r * 0.4f), 1.5f, IM_COL32(255, 255, 255, 255));
            }
            break;
        case IconType::EyedropperTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Pipette pipette slanted from top-right to bottom-left
                float pr = r * 0.8f;
                ImVec2 tip = ImVec2(cp.x - pr * 0.4f, cp.y + pr * 0.4f);
                ImVec2 top = ImVec2(cp.x + pr * 0.4f, cp.y - pr * 0.4f);
                
                dl->AddLine(tip, top, IM_COL32(220, 220, 220, 255), thickness * 3.0f);
                dl->AddCircleFilled(top, pr * 0.25f, IM_COL32(255, 80, 80, 255));
                
                ImVec2 ringCenter = ImVec2(cp.x + pr * 0.2f, cp.y - pr * 0.2f);
                dl->AddCircleFilled(ringCenter, pr * 0.15f, IM_COL32(100, 100, 100, 255));
                
                ImVec2 dropPos = ImVec2(tip.x - pr * 0.25f, tip.y + pr * 0.25f);
                dl->AddCircleFilled(dropPos, 2.0f, IM_COL32(50, 220, 255, 255));
            }
            break;
        case IconType::GrabTool:
            {
                float r = is * 0.48f;
                ImVec2 lobeCenter = ImVec2(cp.x - r * 0.2f, cp.y + r * 0.2f);
                ImVec2 pullPeak = ImVec2(cp.x + r * 1.0f, cp.y - r * 1.0f);
                
                // Custom Grab Shape Path (combining lobe and pulled peak)
                dl->PathClear();
                dl->PathArcTo(lobeCenter, r, 0.75f * 3.14159f, 1.75f * 3.14159f); // bottom-left arc
                dl->PathLineTo(pullPeak);
                // Fill base shadow
                dl->PathFillConvex(clay_shadow);
                
                // Diffuse overlay
                ImVec2 lobeCenterDiff = ImVec2(lobeCenter.x - r * 0.1f, lobeCenter.y - r * 0.1f);
                ImVec2 pullPeakDiff = ImVec2(pullPeak.x - r * 0.15f, pullPeak.y + r * 0.15f);
                dl->PathClear();
                dl->PathArcTo(lobeCenterDiff, r * 0.82f, 0.75f * 3.14159f, 1.75f * 3.14159f);
                dl->PathLineTo(pullPeakDiff);
                dl->PathFillConvex(clay_diffuse);
                
                // Specular highlight on the lobe
                dl->AddCircleFilled(ImVec2(lobeCenter.x - r * 0.25f, lobeCenter.y - r * 0.25f), r * 0.22f, clay_specular);
                
                // Yellow grab/pull arrow
                ImVec2 arrowStart = pullPeak;
                ImVec2 arrowEnd = ImVec2(pullPeak.x + r * 0.5f, pullPeak.y - r * 0.5f);
                dl->AddLine(arrowStart, arrowEnd, IM_COL32(255, 215, 0, 255), thickness * 1.5f);
                
                float angle = -0.78539f; // -45 deg
                float arrow_sz = is * 0.15f;
                dl->AddTriangleFilled(
                    arrowEnd,
                    ImVec2(arrowEnd.x - arrow_sz * cosf(angle - 0.5f), arrowEnd.y - arrow_sz * sinf(angle - 0.5f)),
                    ImVec2(arrowEnd.x - arrow_sz * cosf(angle + 0.5f), arrowEnd.y - arrow_sz * sinf(angle + 0.5f)),
                    IM_COL32(255, 215, 0, 255)
                );
            }
            break;
        case IconType::InflateTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Secondary ballooning sphere in top-right
                ImVec2 inflateCenter = ImVec2(cp.x + r * 0.4f, cp.y - r * 0.4f);
                float inflateR = r * 0.65f;
                
                // Inflated shadow
                dl->AddCircleFilled(inflateCenter, inflateR, clay_shadow);
                // Inflated diffuse
                dl->AddCircleFilled(ImVec2(inflateCenter.x - inflateR * 0.12f, inflateCenter.y - inflateR * 0.12f), inflateR * 0.85f, clay_diffuse);
                // Inflated specular
                dl->AddCircleFilled(ImVec2(inflateCenter.x - inflateR * 0.25f, inflateCenter.y - inflateR * 0.25f), inflateR * 0.22f, clay_specular);
                
                // Radial orange arrows pointing outwards
                float arrowLen = is * 0.22f;
                for (int i = 0; i < 3; ++i) {
                    float a = -0.78539f + (i - 1) * 0.6f;
                    float c = cosf(a);
                    float s_val = sinf(a);
                    ImVec2 start(inflateCenter.x + c * inflateR, inflateCenter.y + s_val * inflateR);
                    ImVec2 end(inflateCenter.x + c * (inflateR + arrowLen), inflateCenter.y + s_val * (inflateR + arrowLen));
                    dl->AddLine(start, end, IM_COL32(255, 130, 30, 255), thickness);
                    
                    float arrow_sz = is * 0.08f;
                    dl->AddTriangleFilled(
                        end,
                        ImVec2(end.x - arrow_sz * cosf(a - 0.4f), end.y - arrow_sz * sinf(a - 0.4f)),
                        ImVec2(end.x - arrow_sz * cosf(a + 0.4f), end.y - arrow_sz * sinf(a + 0.4f)),
                        IM_COL32(255, 130, 30, 255)
                    );
                }
            }
            break;
        case IconType::SmoothTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Add rough texture details on the left-bottom of the sphere
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.6f, cp.y + r * 0.3f), 1.5f, clay_detail_shadow);
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.4f, cp.y + r * 0.5f), 1.0f, clay_detail_shadow);
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.5f, cp.y + r * 0.1f), 1.2f, clay_detail_shadow);
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.2f, cp.y + r * 0.6f), 1.3f, clay_detail_shadow);
                
                // Curved smoothing action arrow in bright green
                dl->PathClear();
                dl->PathArcTo(cp, r * 1.15f, -1.8f, 0.5f);
                dl->PathStroke(IM_COL32(40, 220, 110, 255), false, thickness * 1.2f);
                
                // Arrow head at the end
                float endAngle = 0.5f;
                float arrow_sz = is * 0.11f;
                ImVec2 endPt(cp.x + cosf(endAngle) * r * 1.15f, cp.y + sinf(endAngle) * r * 1.15f);
                float a_head = endAngle + 1.57f; // perpendicular
                dl->AddTriangleFilled(
                    endPt,
                    ImVec2(endPt.x - arrow_sz * cosf(a_head - 0.4f), endPt.y - arrow_sz * sinf(a_head - 0.4f)),
                    ImVec2(endPt.x - arrow_sz * cosf(a_head + 0.4f), endPt.y - arrow_sz * sinf(a_head + 0.4f)),
                    IM_COL32(40, 220, 110, 255)
                );
            }
            break;
        case IconType::FlattenTool:
            {
                float r = is * 0.48f;
                float cutY = cp.y - r * 0.35f;
                float halfWidth = r * cosf(asinf(-0.35f));
                
                // Base shadow (flat top sphere)
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - halfWidth, cutY));
                dl->PathLineTo(ImVec2(cp.x + halfWidth, cutY));
                dl->PathArcTo(cp, r, -0.357f, 3.14159f + 0.357f);
                dl->PathFillConvex(clay_shadow);
                
                // Diffuse overlay (clipped)
                ImVec2 diffCp(cp.x - r * 0.12f, cp.y - r * 0.12f);
                float diffCutY = diffCp.y - r * 0.85f * 0.35f;
                float diffHalfWidth = r * 0.85f * cosf(asinf(-0.35f));
                dl->PathClear();
                dl->PathLineTo(ImVec2(diffCp.x - diffHalfWidth, diffCutY));
                dl->PathLineTo(ImVec2(diffCp.x + diffHalfWidth, diffCutY));
                dl->PathArcTo(diffCp, r * 0.85f, -0.357f, 3.14159f + 0.357f);
                dl->PathFillConvex(clay_diffuse);
                
                // Specular highlight
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.25f, cp.y - r * 0.25f), r * 0.22f, clay_specular);
                
                // Flat orange tool line resting on flat cut
                dl->AddLine(ImVec2(cp.x - r * 1.1f, cutY), ImVec2(cp.x + r * 1.1f, cutY), IM_COL32(255, 130, 30, 255), thickness * 1.5f);
                
                // Downward pressure arrow
                ImVec2 arrStart(cp.x, cutY - is * 0.22f);
                ImVec2 arrEnd(cp.x, cutY - is * 0.04f);
                dl->AddLine(arrStart, arrEnd, IM_COL32(255, 130, 30, 255), thickness);
                dl->AddTriangleFilled(arrEnd, ImVec2(arrEnd.x - is * 0.06f, arrEnd.y - is * 0.06f), ImVec2(arrEnd.x + is * 0.06f, arrEnd.y - is * 0.06f), IM_COL32(255, 130, 30, 255));
            }
            break;
        case IconType::DrawTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Draw smooth curved stroke shadow
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - r * 0.6f, cp.y + r * 0.4f));
                dl->PathBezierQuadraticCurveTo(ImVec2(cp.x - r * 0.1f, cp.y - r * 0.1f), ImVec2(cp.x + r * 0.6f, cp.y - r * 0.4f), 10);
                dl->PathStroke(clay_trans_shadow, false, r * 0.38f);
                
                // Main buildup stroke in light clay
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - r * 0.6f, cp.y + r * 0.4f));
                dl->PathBezierQuadraticCurveTo(ImVec2(cp.x - r * 0.1f, cp.y - r * 0.1f), ImVec2(cp.x + r * 0.6f, cp.y - r * 0.4f), 10);
                dl->PathStroke(clay_buildup, false, r * 0.28f);
                
                // Top stroke highlight
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - r * 0.6f, cp.y + r * 0.4f));
                dl->PathBezierQuadraticCurveTo(ImVec2(cp.x - r * 0.1f, cp.y - r * 0.1f), ImVec2(cp.x + r * 0.6f, cp.y - r * 0.4f), 10);
                dl->PathStroke(clay_highlight, false, r * 0.10f);
                
                // Yellow build up arrow
                ImVec2 arrStart(cp.x + r * 0.3f, cp.y + r * 0.4f);
                ImVec2 arrEnd(cp.x + r * 0.6f, cp.y + r * 0.1f);
                dl->AddLine(arrStart, arrEnd, IM_COL32(255, 190, 40, 255), thickness);
                
                float a = -0.78539f; // -45 deg
                float arrow_sz = is * 0.08f;
                dl->AddTriangleFilled(
                    arrEnd,
                    ImVec2(arrEnd.x - arrow_sz * cosf(a - 0.4f), arrEnd.y - arrow_sz * sinf(a - 0.4f)),
                    ImVec2(arrEnd.x - arrow_sz * cosf(a + 0.4f), arrEnd.y - arrow_sz * sinf(a + 0.4f)),
                    IM_COL32(255, 190, 40, 255)
                );
            }
            break;
        case IconType::LayerTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Plateau polygon (raised step)
                ImVec2 p0(cp.x - r * 0.5f, cp.y + r * 0.2f);
                ImVec2 p1(cp.x + r * 0.2f, cp.y - r * 0.5f);
                ImVec2 p2(cp.x + r * 0.7f, cp.y - r * 0.1f);
                ImVec2 p3(cp.x + r * 0.1f, cp.y + r * 0.5f);
                
                float raiseX = -r * 0.18f;
                float raiseY = -r * 0.18f;
                
                ImVec2 p0_r(p0.x + raiseX, p0.y + raiseY);
                ImVec2 p1_r(p1.x + raiseX, p1.y + raiseY);
                ImVec2 p2_r(p2.x + raiseX, p2.y + raiseY);
                ImVec2 p3_r(p3.x + raiseX, p3.y + raiseY);
                
                // Side walls (shadows)
                dl->PathClear();
                dl->PathLineTo(p0); dl->PathLineTo(p1); dl->PathLineTo(p1_r); dl->PathLineTo(p0_r);
                dl->PathFillConvex(clay_shadow);
                
                dl->PathClear();
                dl->PathLineTo(p1); dl->PathLineTo(p2); dl->PathLineTo(p2_r); dl->PathLineTo(p1_r);
                dl->PathFillConvex(clay_shadow);
                
                // Flat top surface
                dl->PathClear();
                dl->PathLineTo(p0_r); dl->PathLineTo(p1_r); dl->PathLineTo(p2_r); dl->PathLineTo(p3_r);
                dl->PathFillConvex(clay_buildup);
                
                // Highlight the lips
                dl->AddLine(p0_r, p1_r, clay_highlight, thickness);
                dl->AddLine(p1_r, p2_r, clay_highlight, thickness);
            }
            break;
        case IconType::PinchTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Vertical pinch line
                dl->AddLine(ImVec2(cp.x, cp.y - r * 0.8f), ImVec2(cp.x, cp.y + r * 0.8f), IM_COL32(35, 36, 38, 230), thickness * 1.5f);
                
                // Soft edges/highlights of pinch
                dl->AddLine(ImVec2(cp.x - 2.0f, cp.y - r * 0.7f), ImVec2(cp.x - 2.0f, cp.y + r * 0.7f), clay_trans_highlight, 1.0f);
                dl->AddLine(ImVec2(cp.x + 2.0f, cp.y - r * 0.7f), ImVec2(cp.x + 2.0f, cp.y + r * 0.7f), clay_trans_highlight, 1.0f);
                
                // Inward pointing blue arrows
                float arrowY = cp.y;
                float arrow_sz = is * 0.10f;
                
                ImVec2 arrLeftStart(cp.x - r * 1.1f, arrowY);
                ImVec2 arrLeftEnd(cp.x - r * 0.2f, arrowY);
                dl->AddLine(arrLeftStart, arrLeftEnd, IM_COL32(60, 170, 255, 255), thickness * 1.5f);
                dl->AddTriangleFilled(arrLeftEnd, ImVec2(arrLeftEnd.x - arrow_sz, arrowY - arrow_sz * 0.6f), ImVec2(arrLeftEnd.x - arrow_sz, arrowY + arrow_sz * 0.6f), IM_COL32(60, 170, 255, 255));
                
                ImVec2 arrRightStart(cp.x + r * 1.1f, arrowY);
                ImVec2 arrRightEnd(cp.x + r * 0.2f, arrowY);
                dl->AddLine(arrRightStart, arrRightEnd, IM_COL32(60, 170, 255, 255), thickness * 1.5f);
                dl->AddTriangleFilled(arrRightEnd, ImVec2(arrRightEnd.x + arrow_sz, arrowY - arrow_sz * 0.6f), ImVec2(arrRightEnd.x + arrow_sz, arrowY + arrow_sz * 0.6f), IM_COL32(60, 170, 255, 255));
            }
            break;
        case IconType::ClayTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Draw a soft, round organic clay buildup (mound) in the center-right
                ImVec2 moundCenter(cp.x + r * 0.22f, cp.y - r * 0.15f);
                float moundR = r * 0.58f;
                
                // Mound shadow
                dl->AddCircleFilled(moundCenter, moundR, clay_trans_shadow);
                
                // Mound diffuse (soft build up in light clay color)
                ImVec2 moundCenterDiff(moundCenter.x - moundR * 0.08f, moundCenter.y - moundR * 0.08f);
                dl->AddCircleFilled(moundCenterDiff, moundR * 0.88f, clay_buildup);
                
                // Mound specular highlight
                dl->AddCircleFilled(ImVec2(moundCenterDiff.x - moundR * 0.2f, moundCenterDiff.y - moundR * 0.2f), moundR * 0.22f, clay_specular);
            }
            break;
        case IconType::ClayStripsTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Strip 1
                {
                    ImVec2 p0(cp.x - r * 0.6f, cp.y + r * 0.42f);
                    ImVec2 p1(cp.x + r * 0.05f, cp.y - r * 0.05f);
                    float dx = r * 0.12f;
                    float dy = r * 0.18f;
                    ImVec2 s0(p0.x - dx, p0.y - dy);
                    ImVec2 s1(p1.x - dx, p1.y - dy);
                    ImVec2 s2(p1.x + dx, p1.y + dy);
                    ImVec2 s3(p0.x + dx, p0.y + dy);
                    
                    dl->PathClear();
                    dl->PathLineTo(s0); dl->PathLineTo(s1); dl->PathLineTo(s2); dl->PathLineTo(s3);
                    dl->PathFillConvex(clay_trans_shadow_150);
                    
                    dl->PathClear();
                    dl->PathLineTo(ImVec2(s0.x+1, s0.y+1)); dl->PathLineTo(ImVec2(s1.x+1, s1.y+1));
                    dl->PathLineTo(ImVec2(s2.x-1, s2.y-1)); dl->PathLineTo(ImVec2(s3.x-1, s3.y-1));
                    dl->PathFillConvex(clay_buildup);
                    dl->AddLine(ImVec2(s0.x+1, s0.y+1), ImVec2(s1.x+1, s1.y+1), clay_highlight, 1.0f);
                }
                
                // Strip 2
                {
                    ImVec2 p0(cp.x - r * 0.05f, cp.y + r * 0.05f);
                    ImVec2 p1(cp.x + r * 0.6f, cp.y - r * 0.42f);
                    float dx = r * 0.12f;
                    float dy = r * 0.18f;
                    ImVec2 s0(p0.x - dx, p0.y - dy);
                    ImVec2 s1(p1.x - dx, p1.y - dy);
                    ImVec2 s2(p1.x + dx, p1.y + dy);
                    ImVec2 s3(p0.x + dx, p0.y + dy);
                    
                    dl->PathClear();
                    dl->PathLineTo(s0); dl->PathLineTo(s1); dl->PathLineTo(s2); dl->PathLineTo(s3);
                    dl->PathFillConvex(clay_trans_shadow);
                    
                    dl->PathClear();
                    dl->PathLineTo(ImVec2(s0.x+1, s0.y+1)); dl->PathLineTo(ImVec2(s1.x+1, s1.y+1));
                    dl->PathLineTo(ImVec2(s2.x-1, s2.y-1)); dl->PathLineTo(ImVec2(s3.x-1, s3.y-1));
                    dl->PathFillConvex(clay_buildup);
                    dl->AddLine(ImVec2(s0.x+1, s0.y+1), ImVec2(s1.x+1, s1.y+1), clay_highlight, 1.0f);
                }
            }
            break;
        case IconType::CreaseTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Crease valley
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - r * 0.6f, cp.y + r * 0.4f));
                dl->PathBezierQuadraticCurveTo(ImVec2(cp.x, cp.y), ImVec2(cp.x + r * 0.6f, cp.y - r * 0.4f), 10);
                dl->PathStroke(clay_detail_shadow, false, thickness * 2.0f);
                
                // Crease raised lips
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - r * 0.6f - 2.0f, cp.y + r * 0.4f - 2.0f));
                dl->PathBezierQuadraticCurveTo(ImVec2(cp.x - 2.0f, cp.y - 2.0f), ImVec2(cp.x + r * 0.6f - 2.0f, cp.y - r * 0.4f - 2.0f), 10);
                dl->PathStroke(clay_trans_highlight, false, thickness);
                
                // Orange/red action curve
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - r * 0.3f, cp.y + r * 0.5f));
                dl->PathBezierQuadraticCurveTo(ImVec2(cp.x + r * 0.2f, cp.y + r * 0.3f), ImVec2(cp.x + r * 0.5f, cp.y - r * 0.2f), 10);
                dl->PathStroke(IM_COL32(255, 80, 40, 255), false, thickness * 1.2f);
            }
            break;
        case IconType::ScrapeTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Flat beveled polygon
                ImVec2 bp0(cp.x - r * 0.2f, cp.y - r * 0.7f);
                ImVec2 bp1(cp.x + r * 0.7f, cp.y - r * 0.2f);
                ImVec2 bp2(cp.x + r * 0.3f, cp.y + r * 0.3f);
                ImVec2 bp3(cp.x - r * 0.5f, cp.y - r * 0.1f);
                
                dl->PathClear();
                dl->PathLineTo(bp0); dl->PathLineTo(bp1); dl->PathLineTo(bp2); dl->PathLineTo(bp3);
                dl->PathFillConvex(clay_shadow);
                
                ImVec2 bp0_in(bp0.x + 1, bp0.y + 1);
                ImVec2 bp1_in(bp1.x - 1, bp1.y + 1);
                ImVec2 bp2_in(bp2.x - 1, bp2.y - 1);
                ImVec2 bp3_in(bp3.x + 1, bp3.y - 1);
                dl->PathClear();
                dl->PathLineTo(bp0_in); dl->PathLineTo(bp1_in); dl->PathLineTo(bp2_in); dl->PathLineTo(bp3_in);
                dl->PathFillConvex(clay_buildup);
                
                // Scraper blade line
                dl->AddLine(ImVec2(bp3.x - r * 0.2f, bp3.y - r * 0.1f), ImVec2(bp1.x + r * 0.1f, bp1.y + r * 0.2f), IM_COL32(255, 255, 255, 230), thickness * 1.5f);
                
                // Scraping pressure arrow
                ImVec2 arrStart(cp.x + r * 0.8f, cp.y + r * 0.5f);
                ImVec2 arrEnd(cp.x + r * 0.3f, cp.y + r * 0.1f);
                dl->AddLine(arrStart, arrEnd, IM_COL32(255, 255, 255, 230), thickness);
                float a = atan2f(arrEnd.y - arrStart.y, arrEnd.x - arrStart.x);
                float arrow_sz = is * 0.08f;
                dl->AddTriangleFilled(
                    arrEnd,
                    ImVec2(arrEnd.x - arrow_sz * cosf(a - 0.4f), arrEnd.y - arrow_sz * sinf(a - 0.4f)),
                    ImVec2(arrEnd.x - arrow_sz * cosf(a + 0.4f), arrEnd.y - arrow_sz * sinf(a + 0.4f)),
                    IM_COL32(255, 255, 255, 230)
                );
            }
            break;
        case IconType::MaskTool:
            {
                // Clay sphere half-covered by a cool "frozen" mask region.
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                // Masked half: a cool-blue half disc with a soft boundary, evoking
                // a protected/frozen region of the surface.
                const ImU32 maskFill = IM_COL32(80, 140, 210, 200);
                const ImU32 maskEdge = IM_COL32(170, 205, 245, 230);
                dl->PathClear();
                dl->PathArcTo(cp, r * 0.92f, IM_PI * 0.5f, IM_PI * 1.5f, 24);
                dl->PathLineTo(ImVec2(cp.x, cp.y - r * 0.92f));
                dl->PathFillConvex(maskFill);
                // Boundary line down the middle.
                dl->AddLine(ImVec2(cp.x, cp.y - r * 0.92f), ImVec2(cp.x, cp.y + r * 0.92f),
                            maskEdge, thickness);
                // Small lock-dot to read as "protected".
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.45f, cp.y), r * 0.16f,
                                    IM_COL32(255, 255, 255, 230), 12);
            }
            break;
        case IconType::DrawSharpTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Draw a very sharp peak/crease ridge
                ImVec2 p0(cp.x - r * 0.6f, cp.y + r * 0.4f);
                ImVec2 p1(cp.x + r * 0.6f, cp.y - r * 0.4f);
                
                // Crease valley/ridge shadow
                dl->PathClear();
                dl->PathLineTo(p0);
                dl->PathBezierQuadraticCurveTo(ImVec2(cp.x - r * 0.05f, cp.y - r * 0.05f), p1, 10);
                dl->PathStroke(clay_detail_shadow, false, r * 0.22f);
                
                // Sharp peak line (bright highlight)
                dl->PathClear();
                dl->PathLineTo(p0);
                dl->PathBezierQuadraticCurveTo(ImVec2(cp.x, cp.y), p1, 10);
                dl->PathStroke(clay_highlight, false, r * 0.10f);
                
                // Sharp yellow arrow
                ImVec2 arrStart(cp.x + r * 0.25f, cp.y + r * 0.42f);
                ImVec2 arrEnd(cp.x + r * 0.58f, cp.y + r * 0.08f);
                dl->AddLine(arrStart, arrEnd, IM_COL32(255, 215, 0, 255), thickness * 1.5f);
                float a = atan2f(arrEnd.y - arrStart.y, arrEnd.x - arrStart.x);
                float arrow_sz = is * 0.10f;
                dl->AddTriangleFilled(
                    arrEnd,
                    ImVec2(arrEnd.x - arrow_sz * cosf(a - 0.4f), arrEnd.y - arrow_sz * sinf(a - 0.4f)),
                    ImVec2(arrEnd.x - arrow_sz * cosf(a + 0.4f), arrEnd.y - arrow_sz * sinf(a + 0.4f)),
                    IM_COL32(255, 215, 0, 255)
                );
            }
            break;
        case IconType::NudgeTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Draw a wave/nudge deformation
                // A lateral push representation (curved blue arrow along the sphere's top-right horizon)
                dl->PathClear();
                dl->PathArcTo(cp, r * 1.15f, -1.9f, -0.4f);
                dl->PathStroke(IM_COL32(50, 160, 255, 255), false, thickness * 1.4f);
                
                float endAngle = -0.4f;
                float arrow_sz = is * 0.11f;
                ImVec2 endPt(cp.x + cosf(endAngle) * r * 1.15f, cp.y + sinf(endAngle) * r * 1.15f);
                float a_head = endAngle + 1.57f;
                dl->AddTriangleFilled(
                    endPt,
                    ImVec2(endPt.x - arrow_sz * cosf(a_head - 0.4f), endPt.y - arrow_sz * sinf(a_head - 0.4f)),
                    ImVec2(endPt.x - arrow_sz * cosf(a_head + 0.4f), endPt.y - arrow_sz * sinf(a_head + 0.4f)),
                    IM_COL32(50, 160, 255, 255)
                );
                
                // Add a small smudge/slide mark on the clay
                dl->AddCircleFilled(ImVec2(cp.x + r * 0.2f, cp.y - r * 0.2f), r * 0.3f, clay_trans_shadow);
                dl->AddCircleFilled(ImVec2(cp.x + r * 0.35f, cp.y - r * 0.35f), r * 0.22f, clay_buildup);
            }
            break;
        case IconType::BlobTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Spherical swell overlay in magenta/orange
                ImVec2 blobC(cp.x + r * 0.3f, cp.y - r * 0.2f);
                float blobR = r * 0.52f;
                
                dl->AddCircleFilled(blobC, blobR, IM_COL32(230, 80, 150, 180));
                dl->AddCircleFilled(ImVec2(blobC.x - blobR * 0.12f, blobC.y - blobR * 0.12f), blobR * 0.85f, IM_COL32(255, 120, 180, 220));
                dl->AddCircleFilled(ImVec2(blobC.x - blobR * 0.28f, blobC.y - blobR * 0.28f), blobR * 0.25f, clay_specular);
            }
            break;
        case IconType::SculptFillTool:
            {
                float r = is * 0.48f;
                // Base clay sphere with a flat cut bottom or valley
                float cutY = cp.y + r * 0.2f;
                float halfWidth = r * cosf(asinf(0.2f));
                
                // Draw base with valley
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - halfWidth, cutY));
                dl->PathLineTo(ImVec2(cp.x + halfWidth, cutY));
                dl->PathArcTo(cp, r, 0.201f, 3.14159f - 0.201f);
                dl->PathFillConvex(clay_shadow);
                
                // Draw filling material (translucent green/cyan filling up the lower cut)
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - r, cp.y + r * 0.2f));
                dl->PathArcTo(cp, r, 0.2f, 2.94f);
                dl->PathFillConvex(IM_COL32(50, 220, 140, 140));
                
                // Horizontal spade line
                dl->AddLine(ImVec2(cp.x - r * 0.9f, cp.y + r * 0.2f), ImVec2(cp.x + r * 0.9f, cp.y + r * 0.2f), IM_COL32(50, 220, 140, 240), thickness * 1.5f);
            }
            break;
        case IconType::SnakeHookTool:
            {
                float r = is * 0.48f;
                
                // Clay base (smaller, since we pull out a long horn)
                drawBaseClaySphere(cp, r * 0.75f);
                
                // Pull a hook/tentacle from center to top right
                ImVec2 hookStart = cp;
                ImVec2 hookMid = ImVec2(cp.x + r * 0.4f, cp.y - r * 0.3f);
                ImVec2 hookEnd = ImVec2(cp.x + r * 1.1f, cp.y - r * 1.1f);
                
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - r * 0.4f, cp.y + r * 0.2f));
                dl->PathBezierQuadraticCurveTo(hookMid, hookEnd, 10);
                dl->PathBezierQuadraticCurveTo(hookMid, ImVec2(cp.x + r * 0.2f, cp.y + r * 0.4f), 10);
                dl->PathFillConvex(clay_shadow);
                
                dl->PathClear();
                dl->PathLineTo(ImVec2(cp.x - r * 0.3f, cp.y + r * 0.1f));
                dl->PathBezierQuadraticCurveTo(hookMid, hookEnd, 10);
                dl->PathBezierQuadraticCurveTo(hookMid, ImVec2(cp.x + r * 0.1f, cp.y + r * 0.3f), 10);
                dl->PathFillConvex(clay_diffuse);
                
                // Specular highlight at peak
                dl->AddCircleFilled(ImVec2(hookMid.x - r * 0.1f, hookMid.y - r * 0.1f), r * 0.15f, clay_specular);
                
                // Swirling yellow trajectory line
                dl->PathClear();
                dl->PathBezierQuadraticCurveTo(ImVec2(cp.x + r * 0.6f, cp.y - r * 0.1f), hookEnd, 10);
                dl->PathStroke(IM_COL32(255, 200, 50, 230), false, thickness);
            }
            break;
        case IconType::ElasticDeformTool:
            {
                float r = is * 0.48f;
                // Stretched/distorted clay sphere (drawn as an ellipse or sheared path)
                ImVec2 ellipseR = ImVec2(r * 1.25f, r * 0.85f);
                
                // Draw base sheared ellipse shadow
                dl->PathClear();
                for (int i = 0; i <= 24; ++i) {
                    float a = i * (6.28318f / 24.0f);
                    float rot = -0.3f; // rotation angle
                    float x = cp.x + cosf(a) * ellipseR.x * cosf(rot) - sinf(a) * ellipseR.y * sinf(rot);
                    float y = cp.y + cosf(a) * ellipseR.x * sinf(rot) + sinf(a) * ellipseR.y * cosf(rot);
                    dl->PathLineTo(ImVec2(x, y));
                }
                dl->PathFillConvex(clay_shadow);
                
                // Inner stretched body (diffuse)
                dl->PathClear();
                ImVec2 diffCp(cp.x - r * 0.12f, cp.y - r * 0.12f);
                for (int i = 0; i <= 24; ++i) {
                    float a = i * (6.28318f / 24.0f);
                    float rot = -0.3f;
                    float x = diffCp.x + cosf(a) * ellipseR.x * 0.85f * cosf(rot) - sinf(a) * ellipseR.y * 0.85f * sinf(rot);
                    float y = diffCp.y + cosf(a) * ellipseR.x * 0.85f * sinf(rot) + sinf(a) * ellipseR.y * 0.85f * cosf(rot);
                    dl->PathLineTo(ImVec2(x, y));
                }
                dl->PathFillConvex(clay_diffuse);
                
                // Specular highlight
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.3f, cp.y - r * 0.2f), r * 0.22f, clay_specular);
                
                // Double-headed green pull arrow
                ImVec2 pA = ImVec2(cp.x - r * 0.9f, cp.y + r * 0.4f);
                ImVec2 pB = ImVec2(cp.x + r * 0.9f, cp.y - r * 0.4f);
                dl->AddLine(pA, pB, IM_COL32(40, 220, 100, 255), thickness * 1.5f);
                
                float a = atan2f(pB.y - pA.y, pB.x - pA.x);
                float arrow_sz = is * 0.08f;
                dl->AddTriangleFilled(pB, ImVec2(pB.x - arrow_sz * cosf(a - 0.4f), pB.y - arrow_sz * sinf(a - 0.4f)), ImVec2(pB.x - arrow_sz * cosf(a + 0.4f), pB.y - arrow_sz * sinf(a + 0.4f)), IM_COL32(40, 220, 100, 255));
                dl->AddTriangleFilled(pA, ImVec2(pA.x + arrow_sz * cosf(a - 0.4f), pA.y + arrow_sz * sinf(a - 0.4f)), ImVec2(pA.x + arrow_sz * cosf(a + 0.4f), pA.y + arrow_sz * sinf(a + 0.4f)), IM_COL32(40, 220, 100, 255));
            }
            break;
        case IconType::VertexMode:
            {
                ImVec2 p0(p.x + is * 0.20f, p.y + is * 0.72f);
                ImVec2 p1(p.x + is * 0.80f, p.y + is * 0.72f);
                ImVec2 p2(p.x + is * 0.50f, p.y + is * 0.52f);
                ImVec2 apex(p.x + is * 0.50f, p.y + is * 0.20f);

                dl->AddLine(p0, p2, IM_COL32(140, 145, 155, 120), thickness * 0.8f);
                dl->AddLine(p1, p2, IM_COL32(140, 145, 155, 120), thickness * 0.8f);
                dl->AddLine(apex, p2, IM_COL32(140, 145, 155, 120), thickness * 0.8f);

                dl->AddLine(p0, p1, col, thickness);
                dl->AddLine(p0, apex, col, thickness);
                dl->AddLine(p1, apex, col, thickness);

                dl->AddCircleFilled(p0, is * 0.05f, col, 10);
                dl->AddCircleFilled(p1, is * 0.05f, col, 10);

                dl->AddCircleFilled(apex, is * 0.13f, IM_COL32(245, 145, 30, 80), 12);
                dl->AddCircleFilled(apex, is * 0.08f, IM_COL32(255, 160, 40, 255), 12);
                dl->AddCircle(apex, is * 0.08f, IM_COL32(255, 215, 150, 255), 12, 1.0f);
            }
            break;
        case IconType::EdgeMode:
            {
                ImVec2 vTop(cp.x, cp.y - is * 0.35f);
                ImVec2 vLeft(cp.x - is * 0.35f, cp.y - is * 0.15f);
                ImVec2 vRight(cp.x + is * 0.35f, cp.y - is * 0.15f);
                ImVec2 vCenter(cp.x, cp.y + is * 0.05f);
                ImVec2 vBLeft(cp.x - is * 0.35f, cp.y + is * 0.30f);
                ImVec2 vBRight(cp.x + is * 0.35f, cp.y + is * 0.30f);
                ImVec2 vBottom(cp.x, cp.y + is * 0.45f);

                ImU32 wCol = col;
                dl->AddLine(vTop, vLeft, wCol, thickness * 0.8f);
                dl->AddLine(vTop, vRight, wCol, thickness * 0.8f);
                dl->AddLine(vLeft, vCenter, wCol, thickness * 0.8f);
                dl->AddLine(vRight, vCenter, wCol, thickness * 0.8f);
                dl->AddLine(vLeft, vBLeft, wCol, thickness * 0.8f);
                dl->AddLine(vRight, vBRight, wCol, thickness * 0.8f);
                dl->AddLine(vBLeft, vBottom, wCol, thickness * 0.8f);
                dl->AddLine(vBRight, vBottom, wCol, thickness * 0.8f);

                dl->AddLine(vCenter, vBottom, IM_COL32(0, 220, 220, 90), thickness * 3.5f);
                dl->AddLine(vCenter, vBottom, IM_COL32(40, 255, 255, 255), thickness * 1.8f);

                dl->AddCircleFilled(vCenter, thickness * 0.9f, wCol, 8);
                dl->AddCircleFilled(vBottom, thickness * 0.9f, wCol, 8);
            }
            break;
        case IconType::FaceMode:
            {
                ImVec2 vTop(cp.x, cp.y - is * 0.35f);
                ImVec2 vLeft(cp.x - is * 0.35f, cp.y - is * 0.15f);
                ImVec2 vRight(cp.x + is * 0.35f, cp.y - is * 0.15f);
                ImVec2 vCenter(cp.x, cp.y + is * 0.05f);
                ImVec2 vBLeft(cp.x - is * 0.35f, cp.y + is * 0.30f);
                ImVec2 vBRight(cp.x + is * 0.35f, cp.y + is * 0.30f);
                ImVec2 vBottom(cp.x, cp.y + is * 0.45f);

                ImVec2 topFace[] = { vTop, vRight, vCenter, vLeft };
                dl->AddConvexPolyFilled(topFace, 4, IM_COL32(40, 160, 255, 90));
                dl->AddPolyline(topFace, 4, IM_COL32(60, 180, 255, 255), ImDrawFlags_Closed, thickness * 1.8f);

                ImU32 wCol = col;
                dl->AddLine(vLeft, vBLeft, wCol, thickness * 0.8f);
                dl->AddLine(vRight, vBRight, wCol, thickness * 0.8f);
                dl->AddLine(vCenter, vBottom, wCol, thickness * 0.8f);
                dl->AddLine(vBLeft, vBottom, wCol, thickness * 0.8f);
                dl->AddLine(vBRight, vBottom, wCol, thickness * 0.8f);
            }
            break;
        case IconType::AddFace:
            {
                ImVec2 p0(p.x + is * 0.15f, p.y + is * 0.25f);
                ImVec2 p1(p.x + is * 0.65f, p.y + is * 0.20f);
                ImVec2 p2(p.x + is * 0.55f, p.y + is * 0.70f);
                ImVec2 p3(p.x + is * 0.10f, p.y + is * 0.75f);
                
                ImVec2 pts[] = { p0, p1, p2, p3 };
                dl->AddConvexPolyFilled(pts, 4, IM_COL32(40, 220, 100, 45));
                dl->AddPolyline(pts, 4, col, ImDrawFlags_Closed, thickness);

                ImVec2 plusCp(p.x + is * 0.78f, p.y + is * 0.72f);
                dl->AddCircleFilled(plusCp, is * 0.18f, IM_COL32(30, 40, 50, 200), 12);
                dl->AddCircle(plusCp, is * 0.18f, IM_COL32(40, 220, 100, 250), 12, 1.2f);
                dl->AddLine(ImVec2(plusCp.x - is * 0.10f, plusCp.y), ImVec2(plusCp.x + is * 0.10f, plusCp.y), IM_COL32(40, 220, 100, 255), 1.8f);
                dl->AddLine(ImVec2(plusCp.x, plusCp.y - is * 0.10f), ImVec2(plusCp.x, plusCp.y + is * 0.10f), IM_COL32(40, 220, 100, 255), 1.8f);
            }
            break;
        case IconType::MergeVertices:
            {
                ImVec2 outer0(cp.x - is * 0.32f, cp.y - is * 0.22f);
                ImVec2 outer1(cp.x + is * 0.32f, cp.y - is * 0.22f);
                ImVec2 outer2(cp.x, cp.y + is * 0.34f);

                ImU32 lineCol = IM_COL32(140, 145, 155, 160);
                dl->AddLine(outer0, cp, lineCol, thickness * 0.8f);
                dl->AddLine(outer1, cp, lineCol, thickness * 0.8f);
                dl->AddLine(outer2, cp, lineCol, thickness * 0.8f);

                dl->AddCircleFilled(outer0, is * 0.06f, IM_COL32(255, 140, 40, 255), 10);
                dl->AddCircleFilled(outer1, is * 0.06f, IM_COL32(255, 140, 40, 255), 10);
                dl->AddCircleFilled(outer2, is * 0.06f, IM_COL32(255, 140, 40, 255), 10);

                dl->AddCircleFilled(cp, is * 0.12f, IM_COL32(255, 160, 40, 100), 12);
                dl->AddCircleFilled(cp, is * 0.07f, IM_COL32(255, 215, 0, 255), 12);
            }
            break;
        case IconType::WeldVertices:
            {
                ImVec2 p0(cp.x - is * 0.28f, cp.y);
                ImVec2 p1(cp.x + is * 0.28f, cp.y);

                dl->AddLine(p0, p1, IM_COL32(140, 145, 155, 180), thickness * 0.9f);
                
                dl->AddCircle(cp, is * 0.22f, IM_COL32(40, 180, 255, 160), 16, 1.2f);
                dl->AddCircleFilled(cp, is * 0.08f, IM_COL32(40, 180, 255, 255), 10);

                dl->AddCircleFilled(p0, is * 0.06f, IM_COL32(255, 140, 40, 255), 10);
                dl->AddCircleFilled(p1, is * 0.06f, IM_COL32(255, 140, 40, 255), 10);
            }
            break;
        case IconType::DissolveTopology:
            {
                ImVec2 p0(p.x + is * 0.15f, p.y + is * 0.20f);
                ImVec2 p2(p.x + is * 0.85f, p.y + is * 0.80f);
                
                dl->AddRect(p0, p2, col, 1.0f, 0, thickness);
                dl->AddLine(p0, p2, IM_COL32(200, 200, 200, 100), thickness * 0.8f);

                dl->AddLine(ImVec2(cp.x - is * 0.15f, cp.y - is * 0.15f), ImVec2(cp.x + is * 0.15f, cp.y + is * 0.15f), IM_COL32(255, 60, 60, 220), 2.0f);
                dl->AddLine(ImVec2(cp.x + is * 0.15f, cp.y - is * 0.15f), ImVec2(cp.x - is * 0.15f, cp.y + is * 0.15f), IM_COL32(255, 60, 60, 220), 2.0f);
            }
            break;
        case IconType::LoopCutTool:
            {
                float radX = is * 0.32f;
                float radY = is * 0.10f;
                ImVec2 topCenter(cp.x, cp.y - is * 0.26f);
                ImVec2 botCenter(cp.x, cp.y + is * 0.26f);

                dl->AddEllipse(topCenter, ImVec2(radX, radY), col, 0.0f, 0, thickness);
                dl->AddEllipse(botCenter, ImVec2(radX, radY), col, 0.0f, 0, thickness);
                dl->AddLine(ImVec2(cp.x - radX, topCenter.y), ImVec2(cp.x - radX, botCenter.y), col, thickness);
                dl->AddLine(ImVec2(cp.x + radX, topCenter.y), ImVec2(cp.x + radX, botCenter.y), col, thickness);

                dl->AddEllipse(cp, ImVec2(radX, radY), IM_COL32(255, 215, 0, 255), 0.0f, 0, thickness * 2.0f);
                
                dl->AddTriangleFilled(ImVec2(cp.x, cp.y - is * 0.08f),
                                      ImVec2(cp.x - is * 0.04f, cp.y),
                                      ImVec2(cp.x + is * 0.04f, cp.y),
                                      IM_COL32(255, 215, 0, 255));
                dl->AddTriangleFilled(ImVec2(cp.x, cp.y + is * 0.08f),
                                      ImVec2(cp.x - is * 0.04f, cp.y),
                                      ImVec2(cp.x + is * 0.04f, cp.y),
                                      IM_COL32(255, 215, 0, 255));
            }
            break;
        case IconType::ExtrudeFaceTool:
            {
                ImVec2 bTop(cp.x, cp.y - is * 0.05f);
                ImVec2 bLeft(cp.x - is * 0.32f, cp.y + is * 0.12f);
                ImVec2 bRight(cp.x + is * 0.32f, cp.y + is * 0.12f);
                ImVec2 bCenter(cp.x, cp.y + is * 0.28f);

                float upShift = is * 0.35f;
                ImVec2 eTop(bTop.x, bTop.y - upShift);
                ImVec2 eLeft(bLeft.x, bLeft.y - upShift);
                ImVec2 eRight(bRight.x, bRight.y - upShift);
                ImVec2 eCenter(bCenter.x, bCenter.y - upShift);

                ImVec2 extFace[] = { eTop, eRight, eCenter, eLeft };
                dl->AddConvexPolyFilled(extFace, 4, IM_COL32(40, 160, 255, 120));
                dl->AddPolyline(extFace, 4, IM_COL32(60, 180, 255, 255), ImDrawFlags_Closed, thickness * 1.5f);

                ImVec2 baseFace[] = { bTop, bRight, bCenter, bLeft };
                dl->AddPolyline(baseFace, 4, col, ImDrawFlags_Closed, thickness * 0.8f);

                dl->AddLine(bLeft, eLeft, IM_COL32(200, 200, 200, 120), thickness * 0.8f);
                dl->AddLine(bRight, eRight, IM_COL32(200, 200, 200, 120), thickness * 0.8f);
                dl->AddLine(bCenter, eCenter, IM_COL32(200, 200, 200, 120), thickness * 0.8f);

                dl->AddLine(bCenter, eCenter, IM_COL32(40, 255, 140, 220), thickness * 1.4f);
                dl->AddTriangleFilled(ImVec2(eCenter.x, eCenter.y - is * 0.06f),
                                      ImVec2(eCenter.x - is * 0.04f, eCenter.y + is * 0.02f),
                                      ImVec2(eCenter.x + is * 0.04f, eCenter.y + is * 0.02f),
                                      IM_COL32(40, 255, 140, 255));
            }
            break;
        case IconType::DeleteFaceTool:
            {
                ImVec2 vTop(cp.x, cp.y - is * 0.35f);
                ImVec2 vLeft(cp.x - is * 0.35f, cp.y - is * 0.15f);
                ImVec2 vRight(cp.x + is * 0.35f, cp.y - is * 0.15f);
                ImVec2 vCenter(cp.x, cp.y + is * 0.05f);
                ImVec2 vBLeft(cp.x - is * 0.35f, cp.y + is * 0.30f);
                ImVec2 vBRight(cp.x + is * 0.35f, cp.y + is * 0.30f);
                ImVec2 vBottom(cp.x, cp.y + is * 0.45f);

                ImVec2 topFace[] = { vTop, vRight, vCenter, vLeft };
                dl->AddConvexPolyFilled(topFace, 4, IM_COL32(255, 60, 60, 35));
                dl->AddPolyline(topFace, 4, IM_COL32(255, 80, 80, 180), ImDrawFlags_Closed, thickness * 0.9f);

                ImU32 wCol = col;
                dl->AddLine(vLeft, vBLeft, wCol, thickness * 0.6f);
                dl->AddLine(vRight, vBRight, wCol, thickness * 0.6f);
                dl->AddLine(vCenter, vBottom, wCol, thickness * 0.6f);
                dl->AddLine(vBLeft, vBottom, wCol, thickness * 0.6f);
                dl->AddLine(vBRight, vBottom, wCol, thickness * 0.6f);

                ImVec2 fc = cp;
                fc.y -= is * 0.15f;
                dl->AddLine(ImVec2(fc.x - is * 0.14f, fc.y - is * 0.10f), ImVec2(fc.x + is * 0.14f, fc.y + is * 0.10f), IM_COL32(255, 50, 50, 255), 2.2f);
                dl->AddLine(ImVec2(fc.x + is * 0.14f, fc.y - is * 0.10f), ImVec2(fc.x - is * 0.14f, fc.y + is * 0.10f), IM_COL32(255, 50, 50, 255), 2.2f);
            }
            break;
        case IconType::ShadeFlatTool:
            {
                ImVec2 top(cp.x, cp.y - is * 0.38f);
                ImVec2 bot(cp.x, cp.y + is * 0.38f);
                ImVec2 left(cp.x - is * 0.38f, cp.y);
                ImVec2 right(cp.x + is * 0.38f, cp.y);

                ImVec2 triL[] = { top, left, bot };
                ImVec2 triR[] = { top, right, bot };

                dl->AddConvexPolyFilled(triL, 3, IM_COL32(255, 255, 255, 30));
                dl->AddConvexPolyFilled(triR, 3, IM_COL32(0, 0, 0, 60));

                dl->AddTriangle(top, left, bot, col, thickness);
                dl->AddTriangle(top, right, bot, col, thickness);
                dl->AddLine(top, bot, col, thickness);
            }
            break;
        case IconType::ShadeSmoothTool:
            {
                float rad = is * 0.38f;
                dl->AddCircle(cp, rad, col, 32, thickness);
                
                dl->AddBezierQuadratic(ImVec2(cp.x - rad * 0.8f, cp.y - rad * 0.2f),
                                       ImVec2(cp.x, cp.y + rad * 0.5f),
                                       ImVec2(cp.x + rad * 0.8f, cp.y - rad * 0.2f),
                                       IM_COL32(255, 255, 255, 100), thickness * 1.2f);
                                       
                dl->AddBezierQuadratic(ImVec2(cp.x - rad * 0.7f, cp.y + rad * 0.2f),
                                       ImVec2(cp.x, cp.y + rad * 0.75f),
                                       ImVec2(cp.x + rad * 0.7f, cp.y + rad * 0.2f),
                                       IM_COL32(255, 255, 255, 60), thickness * 0.8f);
            }
            break;
        case IconType::Water:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Draw blue-cyan shaded water waves overlay
                dl->PathClear();
                dl->PathArcTo(cp, r, 0.15f * 3.14159f, 0.85f * 3.14159f);
                dl->PathFillConvex(IM_COL32(30, 160, 255, 140)); // Blue transparent bottom overlay
                
                // Water ripples (curved wave lines)
                dl->PathClear();
                for (int i = 0; i <= 10; ++i) {
                    float t = i / 10.0f;
                    float x = cp.x - r * 0.8f + t * r * 1.6f;
                    float y = cp.y + r * 0.2f + sinf(t * 4.5f) * (r * 0.12f);
                    dl->PathLineTo(ImVec2(x, y));
                }
                dl->PathStroke(IM_COL32(100, 220, 255, 200), false, thickness * 1.2f);
                
                // Falling water drop in blue-cyan
                ImVec2 drop_tip = ImVec2(cp.x, cp.y - r * 0.7f);
                float drop_rad = r * 0.28f;
                dl->PathClear();
                dl->PathLineTo(drop_tip);
                dl->PathArcTo(ImVec2(cp.x, cp.y - r * 0.2f), drop_rad, -0.1f * 3.1415f, 1.1f * 3.1415f, 8);
                dl->PathFillConvex(IM_COL32(80, 200, 255, 240));
                dl->PathStroke(IM_COL32(150, 230, 255, 255), true, 1.0f);
            }
            break;
        case IconType::Volumetric:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Soft translucent clouds layered on top of the sphere
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.3f, cp.y + r * 0.2f), r * 0.45f, IM_COL32(255, 255, 255, 90));
                dl->AddCircleFilled(ImVec2(cp.x + r * 0.3f, cp.y + r * 0.2f), r * 0.40f, IM_COL32(255, 255, 255, 90));
                dl->AddCircleFilled(ImVec2(cp.x, cp.y - r * 0.2f), r * 0.50f, IM_COL32(255, 255, 255, 120));
                
                dl->AddCircle(ImVec2(cp.x, cp.y - r * 0.2f), r * 0.50f, IM_COL32(255, 255, 255, 180), 16, 1.0f);
            }
            break;
        case IconType::Force:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Orbiting / swirling force field rings in glowing orange/red
                float rx1 = r * 1.1f;
                float ry1 = r * 0.35f;
                
                dl->PathClear();
                for (int i = 0; i <= 20; ++i) {
                    float a = i * (6.28318f / 20.0f);
                    float rot = 0.6f;
                    float x = cp.x + cosf(a) * rx1 * cosf(rot) - sinf(a) * ry1 * sinf(rot);
                    float y = cp.y + cosf(a) * rx1 * sinf(rot) + sinf(a) * ry1 * cosf(rot);
                    dl->PathLineTo(ImVec2(x, y));
                }
                dl->PathStroke(IM_COL32(255, 120, 40, 220), false, thickness * 1.2f);
                
                dl->PathClear();
                for (int i = 0; i <= 20; ++i) {
                    float a = i * (6.28318f / 20.0f);
                    float rot = -0.6f;
                    float x = cp.x + cosf(a) * rx1 * cosf(rot) - sinf(a) * ry1 * sinf(rot);
                    float y = cp.y + cosf(a) * rx1 * sinf(rot) + sinf(a) * ry1 * cosf(rot);
                    dl->PathLineTo(ImVec2(x, y));
                }
                dl->PathStroke(IM_COL32(255, 60, 40, 180), false, thickness * 1.0f);
            }
            break;
        case UIWidgets::IconType::World:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Latitude/longitude glow lines on the globe
                dl->AddEllipse(cp, ImVec2(r * 0.4f, r * 0.95f), IM_COL32(0, 220, 255, 130), 0.0f, 0, thickness * 0.8f); // vertical grid
                dl->AddLine(ImVec2(cp.x - r, cp.y), ImVec2(cp.x + r, cp.y), IM_COL32(0, 220, 255, 130), thickness * 0.8f); // equator
                
                // Sun crescent glow on top-right edge
                dl->PathClear();
                dl->PathArcTo(cp, r, -1.2f, 0.2f);
                dl->PathStroke(IM_COL32(255, 220, 60, 240), false, thickness * 1.5f);
            }
            break;
        case UIWidgets::IconType::System:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                // Glowing cyan HUD border/screen overlay
                dl->AddRect(ImVec2(cp.x - r * 0.7f, cp.y - r * 0.5f), ImVec2(cp.x + r * 0.7f, cp.y + r * 0.5f), IM_COL32(100, 200, 255, 220), 2.0f, 0, thickness);
                
                // Monitor stand
                dl->AddLine(ImVec2(cp.x, cp.y + r * 0.5f), ImVec2(cp.x, cp.y + r * 0.8f), IM_COL32(100, 200, 255, 220), thickness);
                dl->AddLine(ImVec2(cp.x - r * 0.3f, cp.y + r * 0.8f), ImVec2(cp.x + r * 0.3f, cp.y + r * 0.8f), IM_COL32(100, 200, 255, 220), thickness);
                
                // Inner green checkbox/dot
                dl->AddCircleFilled(ImVec2(cp.x - r * 0.3f, cp.y - r * 0.1f), 2.0f, IM_COL32(100, 255, 100, 255));
                dl->AddLine(ImVec2(cp.x - r * 0.1f, cp.y - r * 0.1f), ImVec2(cp.x + r * 0.4f, cp.y - r * 0.1f), IM_COL32(240, 240, 240, 255), 1.0f);
            }
            break;
        case IconType::Wind:
            dl->AddLine(ImVec2(p.x, p.y + is*0.3f), ImVec2(p.x + is*0.7f, p.y + is*0.3f), col, thickness);
            dl->AddLine(ImVec2(p.x + is*0.2f, p.y + is*0.6f), ImVec2(p.x + is*0.9f, p.y + is*0.6f), col, thickness);
            dl->AddBezierQuadratic(ImVec2(p.x + is*0.7f, p.y + is*0.3f), ImVec2(p.x + is*0.85f, p.y + is*0.1f), ImVec2(p.x + is, p.y + is*0.3f), col, thickness);
            break;
        case IconType::Gravity:
            dl->AddCircleFilled(cp, is * 0.4f, col);
            dl->AddCircle(cp, is * 0.5f, col, 0, thickness);
            break;
        case IconType::Vortex:
            for(int i=0; i<3; i++) dl->AddCircle(cp, is*0.15f*(i+1), col, 0, thickness);
            dl->AddLine(cp, ImVec2(cp.x + is*0.5f, cp.y - is*0.5f), col, thickness);
            break;
        case IconType::Physics:
            dl->AddCircleFilled(ImVec2(cp.x-4, cp.y-4), 4, col);
            dl->AddRect(ImVec2(cp.x+2, cp.y+2), ImVec2(cp.x+10, cp.y+10), col, 0, 0, thickness);
            break;
        case IconType::Magnet:
            dl->AddRect(p, ImVec2(p.x+is, p.y+is*0.4f), col, 5.0f, ImDrawFlags_RoundCornersTop, thickness*2);
            dl->AddRectFilled(ImVec2(p.x, p.y), ImVec2(p.x+is*0.3f, p.y+is*0.2f), col);
            dl->AddRectFilled(ImVec2(p.x+is*0.7f, p.y), ImVec2(p.x+is, p.y+is*0.2f), col);
            break;
        case IconType::Noise:
            {
                const int steps = 10;
                ImVec2 prevPt(p.x, cp.y);
                for (int i = 1; i <= steps; ++i) {
                    float t = (float)i / (float)steps;
                    float x = p.x + t * is;
                    float offset = (i % 2 == 0 ? 1.0f : -1.0f) * (is * 0.22f) * (t > 0.1f && t < 0.9f ? 1.0f : 0.2f);
                    ImVec2 currPt(x, cp.y + offset);
                    dl->AddLine(prevPt, currPt, col, thickness * 1.2f);
                    prevPt = currPt;
                }
            }
            break;
        case IconType::Camera:
            dl->AddRect(ImVec2(p.x, p.y+is*0.2f), ImVec2(p.x+is*0.7f, p.y+is*0.8f), col, 2.0f, 0, thickness);
            dl->AddTriangleFilled(ImVec2(p.x+is*0.7f, cp.y), ImVec2(p.x+is, p.y), ImVec2(p.x+is, p.y+is), col);
            break;
        case IconType::Light:
            dl->AddCircle(cp, is*0.3f, col, 0, thickness);
            for(int i=0; i<8; i++) {
                float a = i*6.28f/8.f;
                dl->AddLine(ImVec2(cp.x+cosf(a)*is*0.35f, cp.y+sinf(a)*is*0.35f), ImVec2(cp.x+cosf(a)*is*0.55f, cp.y+sinf(a)*is*0.55f), col, thickness);
            }
            break;
        case IconType::Mesh:
            {
                ImVec2 vTop(cp.x, cp.y - is * 0.35f);
                ImVec2 vLeft(cp.x - is * 0.35f, cp.y - is * 0.15f);
                ImVec2 vRight(cp.x + is * 0.35f, cp.y - is * 0.15f);
                ImVec2 vCenter(cp.x, cp.y + is * 0.05f);
                ImVec2 vBLeft(cp.x - is * 0.35f, cp.y + is * 0.30f);
                ImVec2 vBRight(cp.x + is * 0.35f, cp.y + is * 0.30f);
                ImVec2 vBottom(cp.x, cp.y + is * 0.45f);

                ImVec2 topFace[] = { vTop, vRight, vCenter, vLeft };
                dl->AddConvexPolyFilled(topFace, 4, IM_COL32(100, 130, 255, 65));
                dl->AddPolyline(topFace, 4, col, ImDrawFlags_Closed, thickness * 0.8f);

                dl->AddLine(vCenter, vBottom, IM_COL32(0, 220, 220, 90), thickness * 3.0f);
                dl->AddLine(vCenter, vBottom, IM_COL32(40, 255, 255, 255), thickness * 1.5f);

                dl->AddCircleFilled(vBottom, is * 0.08f, IM_COL32(255, 140, 40, 255), 10);
                dl->AddCircle(vBottom, is * 0.08f, IM_COL32(255, 200, 100, 255), 10, 1.0f);

                ImU32 wCol = col;
                dl->AddLine(vLeft, vBLeft, wCol, thickness * 0.8f);
                dl->AddLine(vRight, vBRight, wCol, thickness * 0.8f);
                dl->AddLine(vBLeft, vBottom, wCol, thickness * 0.8f);
                dl->AddLine(vBRight, vBottom, wCol, thickness * 0.8f);
            }
            break;
        case UIWidgets::IconType::Timeline:
            for(int i=0; i<3; i++) dl->AddLine(ImVec2(p.x, p.y + is*0.3f*i + 2), ImVec2(p.x + is, p.y + is*0.3f*i + 2), col, thickness);
            dl->AddRectFilled(ImVec2(cp.x-1, p.y), ImVec2(cp.x+1, p.y+is), col); // Needle
            break;
        case UIWidgets::IconType::Console:
            dl->AddLine(ImVec2(p.x, p.y+2), ImVec2(p.x + is*0.4f, cp.y), col, thickness*1.2f);
            dl->AddLine(ImVec2(p.x + is*0.4f, cp.y), ImVec2(p.x, p.y+is-2), col, thickness*1.2f);
            dl->AddLine(ImVec2(cp.x, p.y+is-2), ImVec2(p.x+is, p.y+is-2), col, thickness*1.2f);
            break;
        case UIWidgets::IconType::Graph:
            dl->AddCircle(ImVec2(p.x+is*0.2f, cp.y), is*0.15f, col, 0, thickness);
            dl->AddCircle(ImVec2(p.x+is*0.8f, p.y+is*0.3f), is*0.15f, col, 0, thickness);
            dl->AddCircle(ImVec2(p.x+is*0.8f, p.y+is*0.7f), is*0.15f, col, 0, thickness);
            dl->AddLine(ImVec2(p.x+is*0.35f, cp.y), ImVec2(p.x+is*0.65f, p.y+is*0.3f), col, thickness*0.5f);
            dl->AddLine(ImVec2(p.x+is*0.35f, cp.y), ImVec2(p.x+is*0.65f, p.y+is*0.7f), col, thickness*0.5f);
            break;
        case UIWidgets::IconType::AnimGraph:
            dl->AddRect(ImVec2(p.x, p.y+is*0.1f), ImVec2(p.x+is*0.4f, p.y+is*0.4f), col, 1.0f, 0, thickness);
            dl->AddRect(ImVec2(p.x+is*0.6f, p.y+is*0.6f), ImVec2(p.x+is, p.y+is*0.9f), col, 1.0f, 0, thickness);
            dl->AddLine(ImVec2(p.x+is*0.4f, p.y+is*0.25f), ImVec2(p.x+is*0.6f, p.y+is*0.75f), col, thickness);
            break;
        case UIWidgets::IconType::Assets:
            dl->AddRect(ImVec2(p.x + is*0.10f, p.y + is*0.28f), ImVec2(p.x + is*0.90f, p.y + is*0.82f), col, 2.0f, 0, thickness);
            {
                ImVec2 folderTab[5] = {
                    ImVec2(p.x + is*0.10f, p.y + is*0.34f),
                    ImVec2(p.x + is*0.22f, p.y + is*0.16f),
                    ImVec2(p.x + is*0.46f, p.y + is*0.16f),
                    ImVec2(p.x + is*0.56f, p.y + is*0.28f),
                    ImVec2(p.x + is*0.90f, p.y + is*0.28f)
                };
                dl->AddPolyline(folderTab, 5, col, false, thickness);
            }
            break;
        case UIWidgets::IconType::LightPoint:
            dl->AddCircleFilled(cp, is*0.15f, col);
            for(int i=0; i<8; i++) {
                float a = i*6.28f/8.f;
                dl->AddLine(ImVec2(cp.x+cosf(a)*is*0.25f, cp.y+sinf(a)*is*0.25f), ImVec2(cp.x+cosf(a)*is*0.5f, cp.y+sinf(a)*is*0.5f), col, thickness*0.8f);
            }
            break;
        case UIWidgets::IconType::LightDir:
            for(int i=0; i<3; i++) {
                dl->AddLine(ImVec2(p.x+is*0.3f*i + 2, p.y), ImVec2(p.x+is*0.3f*i + 2, p.y+is*0.7f), col, thickness);
                dl->AddLine(ImVec2(p.x+is*0.3f*i + 2, p.y+is*0.7f), ImVec2(p.x+is*0.3f*i - 2, p.y+is*0.5f), col, thickness*0.8f);
                dl->AddLine(ImVec2(p.x+is*0.3f*i + 2, p.y+is*0.7f), ImVec2(p.x+is*0.3f*i + 6, p.y+is*0.5f), col, thickness*0.8f);
            }
            break;
        case UIWidgets::IconType::LightSpot:
            dl->AddCircle(ImVec2(cp.x, p.y+is*0.2f), is*0.2f, col, 0, thickness);
            dl->AddLine(ImVec2(cp.x-is*0.2f, p.y+is*0.2f), ImVec2(p.x, p.y+is*0.9f), col, thickness);
            dl->AddLine(ImVec2(cp.x+is*0.2f, p.y+is*0.2f), ImVec2(p.x+is, p.y+is*0.9f), col, thickness);
            dl->AddLine(ImVec2(p.x, p.y+is*0.9f), ImVec2(p.x+is, p.y+is*0.9f), col, thickness*0.5f);
            break;
        case UIWidgets::IconType::LightArea:
            dl->AddRect(ImVec2(p.x+is*0.1f, p.y+is*0.2f), ImVec2(p.x+is*0.9f, p.y+is*0.6f), col, 1.0f, 0, thickness);
            for(int i=0; i<4; i++) dl->AddLine(ImVec2(p.x+is*(0.2f+0.2f*i), p.y+is*0.6f), ImVec2(p.x+is*(0.2f+0.2f*i), p.y+is*0.9f), col, thickness*0.5f);
            break;
        case UIWidgets::IconType::HairAddTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                for (int i = 0; i < 3; ++i) {
                    dl->AddBezierQuadratic(
                        ImVec2(xs[i], cp.y + r*0.8f),
                        ImVec2(xs[i] - r*0.2f, cp.y),
                        ImVec2(xs[i], cp.y - r*0.5f),
                        hairCol, thickness * 1.2f
                    );
                }
                float px = cp.x + r*0.6f;
                float py = cp.y - r*0.6f;
                float sz = r*0.22f;
                dl->AddLine(ImVec2(px - sz, py), ImVec2(px + sz, py), col, thickness * 1.5f);
                dl->AddLine(ImVec2(px, py - sz), ImVec2(px, py + sz), col, thickness * 1.5f);
            }
            break;
        case UIWidgets::IconType::HairRemoveTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                for (int i = 0; i < 3; ++i) {
                    if (i == 1) {
                        dl->AddLine(ImVec2(xs[i], cp.y + r*0.8f), ImVec2(xs[i] - r*0.1f, cp.y + r*0.2f), hairCol, thickness * 1.2f);
                        dl->AddLine(ImVec2(xs[i] - r*0.15f, cp.y - r*0.2f), ImVec2(xs[i], cp.y - r*0.5f), hairCol, thickness * 1.2f);
                    } else {
                        dl->AddBezierQuadratic(
                            ImVec2(xs[i], cp.y + r*0.8f),
                            ImVec2(xs[i] - r*0.2f, cp.y),
                            ImVec2(xs[i], cp.y - r*0.5f),
                            hairCol, thickness * 1.2f
                        );
                    }
                }
                float px = cp.x + r*0.6f;
                float py = cp.y - r*0.6f;
                float sz = r*0.22f;
                dl->AddLine(ImVec2(px - sz, py), ImVec2(px + sz, py), col, thickness * 1.5f);
            }
            break;
        case UIWidgets::IconType::HairCutTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                for (int i = 0; i < 3; ++i) {
                    dl->AddLine(ImVec2(xs[i], cp.y + r*0.8f), ImVec2(xs[i], cp.y - r*0.1f), hairCol, thickness * 1.3f);
                }
                ImVec2 s_center(cp.x, cp.y - r * 0.35f);
                float sc = r * 0.42f;
                float scissorThickness = thickness * 1.8f;
                dl->AddLine(ImVec2(s_center.x - sc, s_center.y - sc * 0.8f), ImVec2(s_center.x + sc, s_center.y + sc * 0.8f), col, scissorThickness);
                dl->AddLine(ImVec2(s_center.x + sc, s_center.y - sc * 0.8f), ImVec2(s_center.x - sc, s_center.y + sc * 0.8f), col, scissorThickness);
                dl->AddCircle(ImVec2(s_center.x - sc * 0.9f, s_center.y + sc * 0.9f), sc * 0.35f, col, 0, scissorThickness);
                dl->AddCircle(ImVec2(s_center.x + sc * 0.9f, s_center.y + sc * 0.9f), sc * 0.35f, col, 0, scissorThickness);
            }
            break;
        case UIWidgets::IconType::HairCombTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                for (int i = 0; i < 3; ++i) {
                    dl->AddBezierQuadratic(
                        ImVec2(xs[i], cp.y + r*0.8f),
                        ImVec2(xs[i] + r*0.3f, cp.y),
                        ImVec2(xs[i] - r*0.2f, cp.y - r*0.6f),
                        hairCol, thickness * 1.2f
                    );
                }
                float comb_y = cp.y - r * 0.45f;
                float combThickness = thickness * 2.0f;
                dl->AddLine(ImVec2(cp.x - r * 0.75f, comb_y), ImVec2(cp.x + r * 0.75f, comb_y), col, combThickness);
                for (int i = 0; i < 6; ++i) {
                    float cx = cp.x - r * 0.6f + r * 0.24f * i;
                    dl->AddLine(ImVec2(cx, comb_y), ImVec2(cx, comb_y + r * 0.45f), col, thickness * 1.2f);
                }
            }
            break;
        case UIWidgets::IconType::HairLengthTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                float lens[] = { r*0.6f, r*1.1f, r*0.8f };
                for (int i = 0; i < 3; ++i) {
                    dl->AddLine(
                        ImVec2(xs[i], cp.y + r*0.8f),
                        ImVec2(xs[i], cp.y + r*0.8f - lens[i]),
                        hairCol, thickness * 1.2f
                    );
                }
                float ax = cp.x + r*0.6f;
                float ay = cp.y - r*0.5f;
                dl->AddLine(ImVec2(ax, ay + r*0.4f), ImVec2(ax, ay - r*0.3f), col, thickness * 1.5f);
                dl->AddLine(ImVec2(ax, ay - r*0.3f), ImVec2(ax - r*0.15f, ay - r*0.1f), col, thickness * 1.5f);
                dl->AddLine(ImVec2(ax, ay - r*0.3f), ImVec2(ax + r*0.15f, ay - r*0.1f), col, thickness * 1.5f);
            }
            break;
        case UIWidgets::IconType::HairDensityTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 dotCol = IM_COL32(255, 180, 70, 240);
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        float dx = cp.x - r*0.4f + r*0.4f*i + (j%2 ? 2.0f : -2.0f);
                        float dy = cp.y - r*0.4f + r*0.4f*j;
                        dl->AddCircleFilled(ImVec2(dx, dy), 1.6f, dotCol);
                        dl->AddLine(ImVec2(dx, dy), ImVec2(dx + 2.0f, dy - r*0.25f), dotCol, thickness);
                    }
                }
            }
            break;
        case UIWidgets::IconType::HairClumpTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                ImVec2 tip = ImVec2(cp.x, cp.y - r * 0.6f);
                dl->AddLine(ImVec2(cp.x - r * 0.5f, cp.y + r * 0.8f), tip, hairCol, thickness * 1.2f);
                dl->AddLine(ImVec2(cp.x, cp.y + r * 0.8f), tip, hairCol, thickness * 1.2f);
                dl->AddLine(ImVec2(cp.x + r * 0.5f, cp.y + r * 0.8f), tip, hairCol, thickness * 1.2f);
                
                dl->AddLine(ImVec2(cp.x - r * 0.22f, cp.y - r * 0.1f), ImVec2(cp.x + r * 0.22f, cp.y - r * 0.1f), col, thickness * 2.0f);
            }
            break;
        case UIWidgets::IconType::HairPuffTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                dl->AddBezierQuadratic(ImVec2(xs[0], cp.y + r*0.8f), ImVec2(xs[0] - r*0.5f, cp.y), ImVec2(xs[0], cp.y - r*0.6f), hairCol, thickness*1.2f);
                dl->AddBezierQuadratic(ImVec2(xs[2], cp.y + r*0.8f), ImVec2(xs[2] + r*0.5f, cp.y), ImVec2(xs[2], cp.y - r*0.6f), hairCol, thickness*1.2f);
                dl->AddLine(ImVec2(xs[1], cp.y + r*0.8f), ImVec2(xs[1], cp.y - r*0.6f), hairCol, thickness*1.2f);
                
                dl->AddLine(ImVec2(cp.x - r*0.4f, cp.y), ImVec2(cp.x - r*0.75f, cp.y), col, thickness);
                dl->AddLine(ImVec2(cp.x + r*0.4f, cp.y), ImVec2(cp.x + r*0.75f, cp.y), col, thickness);
            }
            break;
        case UIWidgets::IconType::HairWaveTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                for (int i = 0; i < 3; ++i) {
                    ImVec2 p0(xs[i], cp.y + r*0.8f);
                    ImVec2 p1(xs[i] + r*0.3f, cp.y + r*0.35f);
                    ImVec2 p2(xs[i] - r*0.3f, cp.y - r*0.1f);
                    ImVec2 p3(xs[i], cp.y - r*0.6f);
                    dl->AddBezierCubic(p0, p1, p2, p3, hairCol, thickness * 1.2f);
                }
            }
            break;
        case UIWidgets::IconType::HairFrizzTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                for (int j = 0; j < 3; ++j) {
                    ImVec2 prev(xs[j], cp.y + r*0.8f);
                    for (int i = 1; i <= 4; ++i) {
                        float t = i / 4.f;
                        float rx = xs[j] + (i%2 ? r*0.18f : -r*0.18f);
                        ImVec2 curr(rx, cp.y + r*0.8f - t * r*1.3f);
                        dl->AddLine(prev, curr, hairCol, thickness);
                        prev = curr;
                    }
                }
            }
            break;
        case UIWidgets::IconType::HairSmoothTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                for (int i = 0; i < 3; ++i) {
                    dl->AddLine(ImVec2(xs[i], cp.y + r*0.8f), ImVec2(xs[i], cp.y - r*0.6f), hairCol, thickness * 1.5f);
                }
            }
            break;
        case UIWidgets::IconType::HairPinchTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                dl->AddBezierQuadratic(ImVec2(xs[0], cp.y + r*0.8f), ImVec2(cp.x - 2.0f, cp.y), ImVec2(xs[0], cp.y - r*0.6f), hairCol, thickness*1.2f);
                dl->AddBezierQuadratic(ImVec2(xs[2], cp.y + r*0.8f), ImVec2(cp.x + 2.0f, cp.y), ImVec2(xs[2], cp.y - r*0.6f), hairCol, thickness*1.2f);
                dl->AddLine(ImVec2(xs[1], cp.y + r*0.8f), ImVec2(xs[1], cp.y - r*0.6f), hairCol, thickness*1.2f);
                
                float ax = r * 0.22f;
                dl->AddLine(ImVec2(cp.x - r * 0.8f, cp.y), ImVec2(cp.x - r * 0.8f + ax, cp.y), col, thickness * 1.5f);
                dl->AddLine(ImVec2(cp.x - r * 0.8f + ax, cp.y), ImVec2(cp.x - r * 0.8f + ax - 3, cp.y - 3), col, thickness * 1.5f);
                dl->AddLine(ImVec2(cp.x - r * 0.8f + ax, cp.y), ImVec2(cp.x - r * 0.8f + ax - 3, cp.y + 3), col, thickness * 1.5f);
                
                dl->AddLine(ImVec2(cp.x + r * 0.8f, cp.y), ImVec2(cp.x + r * 0.8f - ax, cp.y), col, thickness * 1.5f);
                dl->AddLine(ImVec2(cp.x + r * 0.8f - ax, cp.y), ImVec2(cp.x + r * 0.8f - ax + 3, cp.y - 3), col, thickness * 1.5f);
                dl->AddLine(ImVec2(cp.x + r * 0.8f - ax, cp.y), ImVec2(cp.x + r * 0.8f - ax + 3, cp.y + 3), col, thickness * 1.5f);
            }
            break;
        case UIWidgets::IconType::HairSpreadTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float xs[] = { cp.x - r*0.4f, cp.x, cp.x + r*0.4f };
                dl->AddBezierQuadratic(ImVec2(xs[0], cp.y + r*0.8f), ImVec2(xs[0] - r*0.2f, cp.y), ImVec2(xs[0], cp.y - r*0.6f), hairCol, thickness*1.2f);
                dl->AddBezierQuadratic(ImVec2(xs[2], cp.y + r*0.8f), ImVec2(xs[2] + r*0.2f, cp.y), ImVec2(xs[2], cp.y - r*0.6f), hairCol, thickness*1.2f);
                dl->AddLine(ImVec2(xs[1], cp.y + r*0.8f), ImVec2(xs[1], cp.y - r*0.6f), hairCol, thickness*1.2f);
                
                float ax = r * 0.22f;
                dl->AddLine(ImVec2(cp.x - r * 0.3f, cp.y), ImVec2(cp.x - r * 0.3f - ax, cp.y), col, thickness * 1.5f);
                dl->AddLine(ImVec2(cp.x - r * 0.3f - ax, cp.y), ImVec2(cp.x - r * 0.3f - ax + 3, cp.y - 3), col, thickness * 1.5f);
                dl->AddLine(ImVec2(cp.x - r * 0.3f - ax, cp.y), ImVec2(cp.x - r * 0.3f - ax + 3, cp.y + 3), col, thickness * 1.5f);
                
                dl->AddLine(ImVec2(cp.x + r * 0.3f, cp.y), ImVec2(cp.x + r * 0.3f + ax, cp.y), col, thickness * 1.5f);
                dl->AddLine(ImVec2(cp.x + r * 0.3f + ax, cp.y), ImVec2(cp.x + r * 0.3f + ax - 3, cp.y - 3), col, thickness * 1.5f);
                dl->AddLine(ImVec2(cp.x + r * 0.3f + ax, cp.y), ImVec2(cp.x + r * 0.3f + ax - 3, cp.y + 3), col, thickness * 1.5f);
            }
            break;
        case UIWidgets::IconType::HairBraidTool:
            {
                float r = is * 0.48f;
                drawBaseClaySphere(cp, r);
                
                ImU32 hairCol = IM_COL32(255, 180, 70, 220);
                float sc = r * 0.6f;
                for (int i = 0; i < 3; ++i) {
                    float dy = cp.y - r * 0.5f + i * r * 0.4f;
                    dl->AddBezierQuadratic(ImVec2(cp.x - sc, dy), ImVec2(cp.x, dy + r*0.18f), ImVec2(cp.x + sc, dy + r*0.35f), hairCol, thickness * 1.3f);
                    dl->AddBezierQuadratic(ImVec2(cp.x + sc, dy + r*0.1f), ImVec2(cp.x, dy + r*0.27f), ImVec2(cp.x - sc, dy + r*0.45f), hairCol, thickness * 1.3f);
                }
            }
            break;
        default: break;
    }
#pragma pop_macro("IM_COL32")
}

bool IconActionButton(const char* id,
                      UIWidgets::IconType icon,
                      const char* label,
                      bool active,
                      const ImVec4& accent,
                      const ImVec2& requested_size,
                      const char* tooltip,
                      bool enabled) {
    const ImVec2 text_size = (label && label[0] != '\0') ? ImGui::CalcTextSize(label) : ImVec2(0.0f, 0.0f);
    const ImVec2 size(
        requested_size.x > 0.0f ? requested_size.x : (56.0f + text_size.x),
        requested_size.y > 0.0f ? requested_size.y : 34.0f);

    if (!enabled) ImGui::BeginDisabled();
    ImGui::PushID(id);

    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##iconAction", size);
    const bool hovered = ImGui::IsItemHovered();
    const float blend = active ? 1.0f : (hovered ? 0.72f : 0.0f);

    const ImVec4 base(0.12f, 0.135f, 0.16f, enabled ? 0.94f : 0.50f);
    const ImVec4 bg(
        base.x + (accent.x - base.x) * (0.14f + 0.12f * blend),
        base.y + (accent.y - base.y) * (0.12f + 0.10f * blend),
        base.z + (accent.z - base.z) * (0.10f + 0.08f * blend),
        active ? 0.98f : (enabled ? 0.92f : 0.50f));

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), ImGui::ColorConvertFloat4ToU32(bg), 5.0f);
    dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                ImGui::ColorConvertFloat4ToU32(ImVec4(accent.x, accent.y, accent.z, active ? 0.58f : (hovered ? 0.32f : 0.14f))),
                5.0f, 0, active ? 1.4f : 1.0f);
    dl->AddRectFilled(ImVec2(pos.x + 1.0f, pos.y + 1.0f),
                      ImVec2(pos.x + size.x - 1.0f, pos.y + size.y * 0.50f),
                      ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, hovered ? 0.028f : 0.016f)),
                      4.0f, ImDrawFlags_RoundCornersTop);

    const bool has_label = label && label[0] != '\0';
    const float icon_size = has_label
        ? ((std::min)(size.y - 12.0f, 24.0f))
        : ((std::min)(size.x, size.y) * 0.84f);
    const ImVec2 icon_pos = has_label
        ? ImVec2(pos.x + 8.0f, pos.y + (size.y - icon_size) * 0.5f)
        : ImVec2(pos.x + (size.x - icon_size) * 0.5f, pos.y + (size.y - icon_size) * 0.5f);
    const ImVec4 icon_color = active
        ? ImVec4((std::min)(1.0f, accent.x + 0.12f), (std::min)(1.0f, accent.y + 0.12f), (std::min)(1.0f, accent.z + 0.12f), enabled ? 1.0f : 0.50f)
        : ImLerp(ImVec4(0.60f, 0.64f, 0.70f, enabled ? 1.0f : 0.45f), accent, hovered ? 0.72f : 0.0f);
    DrawIcon(icon, icon_pos, icon_size, ImGui::ColorConvertFloat4ToU32(icon_color), 1.7f);

    if (has_label) {
        dl->AddText(ImVec2(icon_pos.x + icon_size + 8.0f, pos.y + (size.y - text_size.y) * 0.5f),
                    ImGui::ColorConvertFloat4ToU32(active ? ImVec4(0.96f, 0.98f, 1.0f, enabled ? 1.0f : 0.50f)
                                                          : ImVec4(0.80f, 0.84f, 0.90f, enabled ? 0.96f : 0.45f)),
                    label);
    }

    if (hovered) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (tooltip && tooltip[0] != '\0') {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
            ImGui::TextUnformatted(tooltip);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    ImGui::PopID();
    if (!enabled) ImGui::EndDisabled();
    return clicked && enabled;
}

void PushControlSurfaceStyle(const ImVec4& accent) {
    const auto& t = ThemeManager::instance().current();
    const auto& settings = ThemeManager::instance().getIconSettings();

    ImVec4 finalAccent = accent;
    if (settings.overridePanelAccentsWithTheme) {
        finalAccent = t.colors.accent;
    }

    float frame_round = 10.0f;
    float grab_round = 10.0f;
    float popup_round = 12.0f;

    if (settings.overridePanelAccentsWithTheme) {
        frame_round = t.style.frameRounding;
        grab_round = t.style.grabRounding;
        popup_round = t.style.popupRounding;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, frame_round);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, grab_round);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, popup_round);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 8.0f));
    
    // Dynamically derive from active theme colors
    ImGui::PushStyleColor(ImGuiCol_FrameBg, t.colors.surface);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ScaleColor(t.colors.surface, 1.3f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ScaleColor(t.colors.surface, 1.5f));
    
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, 0.92f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4((std::min)(1.0f, finalAccent.x + 0.10f), (std::min)(1.0f, finalAccent.y + 0.10f), (std::min)(1.0f, finalAccent.z + 0.10f), 1.0f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, ImVec4((std::min)(1.0f, finalAccent.x + 0.10f), (std::min)(1.0f, finalAccent.y + 0.10f), (std::min)(1.0f, finalAccent.z + 0.10f), 1.0f));
    
    ImGui::PushStyleColor(ImGuiCol_Button, t.colors.primary);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ScaleColor(t.colors.primary, 1.2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ScaleColor(t.colors.primary, 0.8f));
    
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(finalAccent.x, finalAccent.y, finalAccent.z, 0.18f));
    
    ImGui::PushStyleColor(ImGuiCol_Header, t.colors.secondary);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ScaleColor(t.colors.secondary, 1.2f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ScaleColor(t.colors.secondary, 1.4f));
}

void PopControlSurfaceStyle() {
    ImGui::PopStyleColor(13);
    ImGui::PopStyleVar(5);
}

bool HorizontalTab(const char* label, UIWidgets::IconType icon, bool active, float width) {
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    static std::unordered_map<ImGuiID, float> hover_anim;

    ImVec2 size = ImVec2(width > 0 ? width : ImGui::CalcTextSize(label).x + 44.0f, 22.0f);
    ImVec2 p = ImGui::GetCursorScreenPos();
    
    ImGui::PushID(label);
    const ImGuiID tab_id = ImGui::GetID("##htab");
    bool clicked = ImGui::InvisibleButton("##htab", size);
    bool hovered = ImGui::IsItemHovered();
    ImGui::PopID();

    float& anim = hover_anim[tab_id];
    const float target_hover = (hovered || active) ? 1.0f : 0.0f;
    const float anim_speed = 12.0f * io.DeltaTime;
    anim += (target_hover - anim) * ImClamp(anim_speed, 0.0f, 1.0f);

    auto getHoverTint = [&](UIWidgets::IconType iconType) -> ImVec4 {
        switch (iconType) {
            case UIWidgets::IconType::Timeline:  return ImVec4(0.98f, 0.78f, 0.36f, 1.0f);
            case UIWidgets::IconType::Console:   return ImVec4(0.52f, 0.90f, 0.62f, 1.0f);
            case UIWidgets::IconType::Graph:     return ImVec4(0.56f, 0.84f, 1.00f, 1.0f);
            case UIWidgets::IconType::AnimGraph: return ImVec4(0.98f, 0.58f, 0.82f, 1.0f);
            case UIWidgets::IconType::Assets:    return ImVec4(1.00f, 0.72f, 0.42f, 1.0f);
            default:                             return ImVec4(0.50f, 0.92f, 0.72f, 1.0f);
        }
    };

    if (active) {
        dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,0.08f)), 3.0f);
        dl->AddRectFilled(ImVec2(p.x + 4, p.y + size.y - 2), ImVec2(p.x + size.x - 4, p.y + size.y), ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.9f, 0.8f, 1.0f)), 1.0f);
    } else if (anim > 0.01f) {
        dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,0.04f * anim)), 3.0f);
    }

    ImVec4 idleIcon(0.6f, 0.6f, 0.65f, 1.0f);
    ImVec4 activeIcon(0.1f, 0.9f, 0.8f, 1.0f);
    ImVec4 hoverIcon = getHoverTint(icon);
    ImVec4 iconTint = active ? activeIcon : ImLerp(idleIcon, hoverIcon, anim);
    const float iconSize = 16.0f + 2.0f * anim;
    DrawIcon(icon, ImVec2(p.x + 8.0f - (iconSize - 16.0f) * 0.5f, p.y + 3.0f - (iconSize - 16.0f) * 0.5f), iconSize, ImGui::ColorConvertFloat4ToU32(iconTint), 1.5f + 0.25f * anim);
    
    ImVec4 idleText(0.6f, 0.6f, 0.65f, 1.0f);
    ImVec4 hoverText = getHoverTint(icon);
    ImVec4 textTint = active ? ImVec4(1,1,1,1) : ImLerp(idleText, hoverText, anim * 0.85f);
    dl->AddText(ImVec2(p.x + 30, p.y + 3), ImGui::ColorConvertFloat4ToU32(textTint), label);

    if (active) {
        // Vertical Bridge Indicator (Alignment with sidebar language)
        dl->AddRectFilled(
            ImVec2(p.x + 4, p.y + size.y - 3), 
            ImVec2(p.x + size.x - 4, p.y + size.y), 
            ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.9f, 0.8f, 1.0f)),
            1.5f
        );

        // Subtle bottom glow
        dl->AddRectFilledMultiColor(
            ImVec2(p.x, p.y + size.y - 8),
            ImVec2(p.x + size.x, p.y + size.y),
            IM_COL32(26, 230, 204, 0), IM_COL32(26, 230, 204, 0),
            IM_COL32(26, 230, 204, 30), IM_COL32(26, 230, 204, 30)
        );
    }

    if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

    ImGui::SameLine(0, 4);
    return clicked;
}

} // namespace UIWidgets

