/*
 * RayTrophi Modern UI System - Implementation
 * ============================================
 * ThemeManager, UIWidgets ve PanelManager implementasyonlarÄ±.
 */

#include "ui_modern.h"
#include <fstream>
#include <algorithm>
#include <imgui_internal.h>
#include <unordered_map>

// ============================================================================
// THEME MANAGER IMPLEMENTATION
// ============================================================================

void ThemeManager::registerDefaultThemes() {
    themes_.clear();

    // --- 0: Dark (ImGui Default) ---
    {
        Theme t;
        t.name = "Dark";
        t.colors.primary    = ImVec4(0.26f, 0.59f, 0.98f, 1.0f);
        t.colors.secondary  = ImVec4(0.20f, 0.20f, 0.22f, 1.0f);
        t.colors.accent     = ImVec4(0.40f, 0.70f, 1.00f, 1.0f);
        t.colors.background = ImVec4(0.06f, 0.06f, 0.06f, 0.94f);
        t.colors.surface    = ImVec4(0.10f, 0.10f, 0.10f, 1.0f);
        t.colors.text       = ImVec4(1.00f, 1.00f, 1.00f, 1.0f);
        t.colors.textMuted  = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);
        t.colors.success    = ImVec4(0.30f, 1.00f, 0.30f, 1.0f);
        t.colors.warning    = ImVec4(1.00f, 0.80f, 0.00f, 1.0f);
        t.colors.error      = ImVec4(1.00f, 0.30f, 0.30f, 1.0f);
        t.colors.border     = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
        themes_.push_back(t);
    }

    // --- 1: Light ---
    {
        Theme t;
        t.name = "Light";
        t.colors.primary    = ImVec4(0.26f, 0.59f, 0.98f, 1.0f);
        t.colors.secondary  = ImVec4(0.90f, 0.90f, 0.90f, 1.0f);
        t.colors.accent     = ImVec4(0.20f, 0.50f, 0.90f, 1.0f);
        t.colors.background = ImVec4(0.94f, 0.94f, 0.94f, 1.0f);
        t.colors.surface    = ImVec4(1.00f, 1.00f, 1.00f, 1.0f);
        t.colors.text       = ImVec4(0.00f, 0.00f, 0.00f, 1.0f);
        t.colors.textMuted  = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);
        t.colors.success    = ImVec4(0.10f, 0.70f, 0.10f, 1.0f);
        t.colors.warning    = ImVec4(0.90f, 0.60f, 0.00f, 1.0f);
        t.colors.error      = ImVec4(0.80f, 0.10f, 0.10f, 1.0f);
        t.colors.border     = ImVec4(0.00f, 0.00f, 0.00f, 0.30f);
        themes_.push_back(t);
    }

    // --- 2: Classic ---
    {
        Theme t;
        t.name = "Classic";
        t.colors.primary    = ImVec4(0.24f, 0.52f, 0.88f, 1.0f);
        t.colors.secondary  = ImVec4(0.20f, 0.22f, 0.27f, 1.0f);
        t.colors.accent     = ImVec4(0.40f, 0.44f, 0.64f, 1.0f);
        t.colors.background = ImVec4(0.00f, 0.00f, 0.00f, 0.85f);
        t.colors.surface    = ImVec4(0.11f, 0.11f, 0.14f, 1.0f);
        t.colors.text       = ImVec4(0.90f, 0.90f, 0.90f, 1.0f);
        t.colors.textMuted  = ImVec4(0.60f, 0.60f, 0.60f, 1.0f);
        t.colors.success    = ImVec4(0.40f, 0.80f, 0.40f, 1.0f);
        t.colors.warning    = ImVec4(0.90f, 0.70f, 0.00f, 1.0f);
        t.colors.error      = ImVec4(0.90f, 0.40f, 0.40f, 1.0f);
        t.colors.border     = ImVec4(0.50f, 0.50f, 0.50f, 0.50f);
        themes_.push_back(t);
    }

    // --- 3: Studio Dark ---
    {
        Theme t;
        t.name = "Studio Dark";
        t.style.frameRounding = 3.0f;
        t.style.windowRounding = 2.0f;
        t.style.scrollbarRounding = 3.0f;
        t.style.grabRounding = 3.0f;
        t.style.popupRounding = 2.0f;
        t.style.tabRounding = 3.0f;

        t.colors.primary    = ImVec4(0.20f, 0.45f, 0.85f, 0.80f);
        t.colors.secondary  = ImVec4(0.22f, 0.24f, 0.28f, 1.0f);
        t.colors.accent     = ImVec4(0.28f, 0.56f, 1.00f, 0.90f);
        t.colors.background = ImVec4(0.10f, 0.10f, 0.12f, 0.94f);
        t.colors.surface    = ImVec4(0.16f, 0.16f, 0.18f, 1.0f);
        t.colors.text       = ImVec4(1.00f, 1.00f, 1.00f, 1.0f);
        t.colors.textMuted  = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);
        t.colors.success    = ImVec4(0.30f, 1.00f, 0.30f, 1.0f);
        t.colors.warning    = ImVec4(1.00f, 1.00f, 0.00f, 1.0f);
        t.colors.error      = ImVec4(1.00f, 0.00f, 0.00f, 1.0f);
        t.colors.border     = ImVec4(0.05f, 0.05f, 0.05f, 0.50f);
        themes_.push_back(t);
    }

    // --- 4: RayTrophi Pro Dark (VarsayÄ±lan) ---
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

    // --- 5: Neon Cyber ---
    {
        Theme t;
        t.name = "Neon Cyber";
        t.style.windowRounding = 4.0f;
        t.style.frameRounding = 4.0f;
        t.style.grabRounding = 4.0f;

        t.colors.primary    = ImVec4(0.05f, 0.25f, 0.05f, 1.0f);
        t.colors.secondary  = ImVec4(0.10f, 0.25f, 0.10f, 1.0f);
        t.colors.accent     = ImVec4(0.40f, 1.00f, 0.40f, 1.0f);
        t.colors.background = ImVec4(0.02f, 0.02f, 0.04f, 0.94f);
        t.colors.surface    = ImVec4(0.05f, 0.13f, 0.05f, 1.0f);
        t.colors.text       = ImVec4(0.65f, 1.00f, 0.65f, 1.0f);
        t.colors.textMuted  = ImVec4(0.30f, 0.50f, 0.30f, 1.0f);
        t.colors.success    = ImVec4(0.40f, 1.00f, 0.40f, 1.0f);
        t.colors.warning    = ImVec4(1.00f, 1.00f, 0.40f, 1.0f);
        t.colors.error      = ImVec4(1.00f, 0.40f, 0.40f, 1.0f);
        t.colors.border     = ImVec4(0.00f, 1.00f, 0.00f, 0.20f);
        themes_.push_back(t);
    }

    // --- 6: High Contrast ---
    {
        Theme t;
        t.name = "High Contrast";
        t.style.windowRounding = 0.0f;
        t.style.frameRounding = 0.0f;
        t.style.grabRounding = 0.0f;

        t.colors.primary    = ImVec4(0.00f, 0.00f, 0.00f, 1.0f);
        t.colors.secondary  = ImVec4(0.20f, 0.20f, 0.20f, 1.0f);
        t.colors.accent     = ImVec4(1.00f, 1.00f, 1.00f, 1.0f);
        t.colors.background = ImVec4(0.00f, 0.00f, 0.00f, 0.94f);
        t.colors.surface    = ImVec4(0.05f, 0.05f, 0.05f, 1.0f);
        t.colors.text       = ImVec4(1.00f, 1.00f, 1.00f, 1.0f);
        t.colors.textMuted  = ImVec4(0.60f, 0.60f, 0.60f, 1.0f);
        t.colors.success    = ImVec4(0.00f, 1.00f, 0.00f, 1.0f);
        t.colors.warning    = ImVec4(1.00f, 1.00f, 0.00f, 1.0f);
        t.colors.error      = ImVec4(1.00f, 0.00f, 0.00f, 1.0f);
        t.colors.border     = ImVec4(1.00f, 1.00f, 1.00f, 0.40f);
        themes_.push_back(t);
    }

    currentIndex_ = 4; // RayTrophi Pro Dark varsayÄ±lan
}

void ThemeManager::addTheme(const Theme& theme) {
    themes_.push_back(theme);
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
    
    // Once ImGui'nin varsayilan stillerini uygula
    switch (currentIndex_) {
        case 0: ImGui::StyleColorsDark(); break;
        case 1: ImGui::StyleColorsLight(); break;
        case 2: ImGui::StyleColorsClassic(); break;
        default: ImGui::StyleColorsDark(); break;
    }

    // Stil ayarlari
    style.WindowRounding    = t.style.windowRounding;
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
                                            t.colors.surface.z, 0.95f);
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
    file << currentIndex_ << " " << panelAlpha << "\n";
}

bool ThemeManager::loadThemeSettings(const std::string& filepath, float& panelAlpha) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    int index = 4;
    float alpha = 0.75f;
    if (file >> index >> alpha) {
        setTheme(index);
        panelAlpha = alpha;
        return true;
    }
    return false;
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

        // Panel Transparency   
        if (ImGui::SliderFloat("Panel Transparency", &panel_alpha, 0.1f, 1.0f, "%.2f")) {
            ImGuiStyle& style = ImGui::GetStyle();
            style.Colors[ImGuiCol_WindowBg].w = panel_alpha;
            themeManager.saveThemeSettings("theme.cfg", panel_alpha);
        }

        EndSection();
    }
}

void DrawIcon(IconType type, ImVec2 p, float s, ImU32 col, float thickness) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float pading = s * 0.2f;
    float is = s - pading * 2;
    ImVec2 cp = ImVec2(p.x + s * 0.5f, p.y + s * 0.5f);
    p.x += pading; p.y += pading;

    switch (type) {
        case IconType::Scene:
            // Hierarchy Tree
            dl->AddRect(ImVec2(p.x + is * 0.12f, p.y + is * 0.12f), ImVec2(p.x + is * 0.34f, p.y + is * 0.34f), col, 1.0f, 0, thickness);
            dl->AddRect(ImVec2(p.x + is * 0.56f, p.y + is * 0.44f), ImVec2(p.x + is * 0.78f, p.y + is * 0.66f), col, 1.0f, 0, thickness);
            dl->AddRect(ImVec2(p.x + is * 0.56f, p.y + is * 0.74f), ImVec2(p.x + is * 0.78f, p.y + is * 0.96f), col, 1.0f, 0, thickness);
            {
                float x1 = p.x + is * 0.23f;
                float y1 = p.y + is * 0.34f;
                float x2 = p.x + is * 0.56f;
                dl->AddLine(ImVec2(x1, y1), ImVec2(x1, p.y + is * 0.85f), col, thickness);
                dl->AddLine(ImVec2(x1, p.y + is * 0.55f), ImVec2(x2, p.y + is * 0.55f), col, thickness);
                dl->AddLine(ImVec2(x1, p.y + is * 0.85f), ImVec2(x2, p.y + is * 0.85f), col, thickness);
            }
            break;
        case IconType::Render:
            // Camera Aperture / Shutter
            {
                float rad = is * 0.45f;
                dl->AddCircle(cp, rad, col, 24, thickness);
                for (int i = 0; i < 6; ++i) {
                    float a = i * (6.28318f / 6.0f);
                    float x1 = cp.x + cosf(a) * rad;
                    float y1 = cp.y + sinf(a) * rad;
                    float inner_a = a + 0.8f;
                    float x2 = cp.x + cosf(inner_a) * (rad * 0.4f);
                    float y2 = cp.y + sinf(inner_a) * (rad * 0.4f);
                    dl->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), col, thickness);
                }
            }
            break;
        case IconType::Terrain:
            // Overlapping Mountains
            dl->PathClear();
            dl->PathLineTo(ImVec2(p.x + is * 0.3f, p.y + is * 0.85f));
            dl->PathLineTo(ImVec2(p.x + is * 0.65f, p.y + is * 0.3f));
            dl->PathLineTo(ImVec2(p.x + is * 0.95f, p.y + is * 0.85f));
            dl->PathStroke(col, 0, thickness);

            dl->PathClear();
            dl->PathLineTo(ImVec2(p.x + is * 0.05f, p.y + is * 0.85f));
            dl->PathLineTo(ImVec2(p.x + is * 0.40f, p.y + is * 0.42f));
            dl->PathLineTo(ImVec2(p.x + is * 0.75f, p.y + is * 0.85f));
            dl->PathStroke(col, 0, thickness);

            dl->AddLine(ImVec2(p.x + is * 0.05f, p.y + is * 0.85f), ImVec2(p.x + is * 0.95f, p.y + is * 0.85f), col, thickness);
            break;
        case IconType::Sculpt: // Used as Modifiers/Modeling
            // Wrench
            {
                ImVec2 p1 = ImVec2(p.x + is * 0.18f, p.y + is * 0.82f);
                ImVec2 p2 = ImVec2(p.x + is * 0.62f, p.y + is * 0.38f);
                dl->AddLine(p1, p2, col, thickness * 2.5f);
                
                ImVec2 head_center = ImVec2(p.x + is * 0.72f, p.y + is * 0.28f);
                float head_rad = is * 0.22f;
                
                dl->PathClear();
                float start_angle = -0.15f * 3.14159f;
                float end_angle = 1.35f * 3.14159f;
                dl->PathArcTo(head_center, head_rad, start_angle, end_angle, 12);
                
                ImVec2 cut1 = ImVec2(head_center.x + cosf(start_angle) * head_rad, head_center.y + sinf(start_angle) * head_rad);
                ImVec2 inner_center = ImVec2(head_center.x - cosf(0.6f) * (head_rad * 0.3f), head_center.y + sinf(0.6f) * (head_rad * 0.3f));
                dl->PathLineTo(inner_center);
                dl->PathLineTo(cut1);
                dl->PathStroke(col, ImDrawFlags_Closed, thickness * 1.5f);
                
                dl->AddCircle(ImVec2(p.x + is * 0.18f, p.y + is * 0.82f), is * 0.08f, col, 8, thickness);
            }
            break;
        case IconType::Hair:
            dl->AddBezierQuadratic(ImVec2(p.x + is*0.24f, p.y + is*0.86f),
                                   ImVec2(p.x + is*0.12f, p.y + is*0.44f),
                                   ImVec2(p.x + is*0.34f, p.y + is*0.14f),
                                   col, thickness);
            dl->AddBezierQuadratic(ImVec2(p.x + is*0.48f, p.y + is*0.88f),
                                   ImVec2(p.x + is*0.34f, p.y + is*0.48f),
                                   ImVec2(p.x + is*0.54f, p.y + is*0.12f),
                                   col, thickness * 1.15f);
            dl->AddBezierQuadratic(ImVec2(p.x + is*0.74f, p.y + is*0.84f),
                                   ImVec2(p.x + is*0.88f, p.y + is*0.46f),
                                   ImVec2(p.x + is*0.66f, p.y + is*0.18f),
                                   col, thickness);
            break;
        case IconType::Brush: // Used as Stylize Mode
        {
            // Magic Wand
            dl->AddLine(ImVec2(p.x + is * 0.15f, p.y + is * 0.85f), ImVec2(p.x + is * 0.60f, p.y + is * 0.40f), col, thickness * 1.8f);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.60f, p.y + is * 0.40f), thickness * 1.5f, col);
            
            ImVec2 sc = ImVec2(p.x + is * 0.75f, p.y + is * 0.25f);
            float sr = is * 0.18f;
            dl->PathClear();
            dl->PathLineTo(ImVec2(sc.x, sc.y - sr));
            dl->PathBezierQuadraticCurveTo(ImVec2(sc.x + sr*0.2f, sc.y - sr*0.2f), ImVec2(sc.x + sr, sc.y), 6);
            dl->PathBezierQuadraticCurveTo(ImVec2(sc.x + sr*0.2f, sc.y + sr*0.2f), ImVec2(sc.x, sc.y + sr), 6);
            dl->PathBezierQuadraticCurveTo(ImVec2(sc.x - sr*0.2f, sc.y + sr*0.2f), ImVec2(sc.x - sr, sc.y), 6);
            dl->PathBezierQuadraticCurveTo(ImVec2(sc.x - sr*0.2f, sc.y - sr*0.2f), ImVec2(sc.x, sc.y - sr), 6);
            dl->PathStroke(col, ImDrawFlags_Closed, thickness);

            dl->AddCircleFilled(ImVec2(p.x + is * 0.82f, p.y + is * 0.50f), thickness * 0.8f, col);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.48f, p.y + is * 0.18f), thickness * 0.8f, col);
            break;
        }
        case IconType::Move:
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.14f), ImVec2(cp.x, p.y + is * 0.86f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.14f, cp.y), ImVec2(p.x + is * 0.86f, cp.y), col, thickness);
            dl->AddTriangleFilled(ImVec2(cp.x, p.y + is * 0.08f), ImVec2(cp.x - is * 0.08f, p.y + is * 0.24f), ImVec2(cp.x + is * 0.08f, p.y + is * 0.24f), col);
            dl->AddTriangleFilled(ImVec2(cp.x, p.y + is * 0.92f), ImVec2(cp.x - is * 0.08f, p.y + is * 0.76f), ImVec2(cp.x + is * 0.08f, p.y + is * 0.76f), col);
            dl->AddTriangleFilled(ImVec2(p.x + is * 0.08f, cp.y), ImVec2(p.x + is * 0.24f, cp.y - is * 0.08f), ImVec2(p.x + is * 0.24f, cp.y + is * 0.08f), col);
            dl->AddTriangleFilled(ImVec2(p.x + is * 0.92f, cp.y), ImVec2(p.x + is * 0.76f, cp.y - is * 0.08f), ImVec2(p.x + is * 0.76f, cp.y + is * 0.08f), col);
            break;
        case IconType::Rotate:
            dl->PathArcTo(cp, is * 0.28f, 0.35f, 5.5f, 24);
            dl->PathStroke(col, 0, thickness * 1.1f);
            dl->AddLine(ImVec2(cp.x + is * 0.18f, p.y + is * 0.20f), ImVec2(cp.x + is * 0.34f, p.y + is * 0.28f), col, thickness);
            dl->AddLine(ImVec2(cp.x + is * 0.18f, p.y + is * 0.20f), ImVec2(cp.x + is * 0.28f, p.y + is * 0.08f), col, thickness);
            break;
        case IconType::ScaleAxis:
            dl->AddRect(ImVec2(p.x + is * 0.24f, p.y + is * 0.24f), ImVec2(p.x + is * 0.72f, p.y + is * 0.72f), col, 2.0f, 0, thickness);
            dl->AddRectFilled(ImVec2(p.x + is * 0.12f, p.y + is * 0.68f), ImVec2(p.x + is * 0.26f, p.y + is * 0.82f), col, 1.5f);
            dl->AddRectFilled(ImVec2(p.x + is * 0.70f, p.y + is * 0.10f), ImVec2(p.x + is * 0.84f, p.y + is * 0.24f), col, 1.5f);
            break;
        case IconType::Gizmo:
            dl->AddCircle(cp, is * 0.12f, col, 18, thickness);
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.16f), ImVec2(cp.x, p.y + is * 0.84f), col, thickness * 0.9f);
            dl->AddLine(ImVec2(p.x + is * 0.16f, cp.y), ImVec2(p.x + is * 0.84f, cp.y), col, thickness * 0.9f);
            break;
        case IconType::ViewSolid:
            dl->AddCircleFilled(cp, is * 0.40f, IM_COL32(126, 132, 142, 190), 32);
            dl->AddCircleFilled(ImVec2(cp.x - is * 0.12f, cp.y - is * 0.12f), is * 0.14f, IM_COL32(255, 255, 255, 68), 18);
            dl->AddCircleFilled(ImVec2(cp.x + is * 0.10f, cp.y + is * 0.11f), is * 0.24f, IM_COL32(36, 40, 48, 96), 20);
            dl->AddCircle(cp, is * 0.40f, col, 28, thickness);
            dl->AddLine(ImVec2(cp.x - is * 0.24f, cp.y), ImVec2(cp.x + is * 0.24f, cp.y), IM_COL32(220, 224, 230, 120), thickness * 0.75f);
            dl->AddLine(ImVec2(cp.x - is * 0.08f, cp.y - is * 0.24f), ImVec2(cp.x - is * 0.08f, cp.y + is * 0.24f), IM_COL32(220, 224, 230, 105), thickness * 0.75f);
            dl->AddLine(ImVec2(cp.x + is * 0.08f, cp.y - is * 0.24f), ImVec2(cp.x + is * 0.08f, cp.y + is * 0.24f), IM_COL32(220, 224, 230, 105), thickness * 0.75f);
            break;
        case IconType::ViewMatcap:
            dl->AddCircleFilled(cp, is * 0.40f, IM_COL32(152, 156, 168, 210), 32);
            dl->AddCircleFilled(ImVec2(cp.x - is * 0.13f, cp.y - is * 0.13f), is * 0.15f, IM_COL32(255, 255, 255, 86), 18);
            dl->AddCircleFilled(ImVec2(cp.x + is * 0.11f, cp.y + is * 0.12f), is * 0.24f, IM_COL32(52, 54, 62, 102), 20);
            dl->AddCircle(cp, is * 0.40f, col, 28, thickness);
            dl->AddBezierQuadratic(ImVec2(cp.x - is * 0.22f, cp.y + is * 0.10f), ImVec2(cp.x, cp.y + is * 0.25f), ImVec2(cp.x + is * 0.22f, cp.y + is * 0.10f), IM_COL32(235, 238, 244, 118), thickness * 0.8f);
            break;
        case IconType::ViewPreview:
            dl->AddCircleFilled(cp, is * 0.40f, IM_COL32(118, 126, 136, 182), 32);
            dl->AddCircleFilled(ImVec2(cp.x + is * 0.08f, cp.y), is * 0.32f, IM_COL32(84, 168, 150, 138), 26);
            dl->AddCircleFilled(ImVec2(cp.x - is * 0.11f, cp.y - is * 0.11f), is * 0.13f, IM_COL32(255, 255, 255, 68), 16);
            dl->AddCircle(cp, is * 0.40f, col, 28, thickness);
            dl->AddLine(ImVec2(cp.x, cp.y - is * 0.26f), ImVec2(cp.x, cp.y + is * 0.26f), IM_COL32(236, 240, 246, 112), thickness * 0.8f);
            break;
        case IconType::ViewRendered:
            dl->AddCircleFilled(cp, is * 0.40f, IM_COL32(72, 138, 255, 200), 32);
            dl->AddCircleFilled(ImVec2(cp.x + is * 0.08f, cp.y + is * 0.06f), is * 0.28f, IM_COL32(242, 144, 72, 144), 22);
            dl->AddCircleFilled(ImVec2(cp.x - is * 0.13f, cp.y - is * 0.13f), is * 0.14f, IM_COL32(255, 255, 255, 96), 18);
            dl->AddCircle(cp, is * 0.40f, col, 28, thickness);
            dl->AddBezierQuadratic(ImVec2(cp.x - is * 0.22f, cp.y + is * 0.12f), ImVec2(cp.x + is * 0.02f, cp.y + is * 0.26f), ImVec2(cp.x + is * 0.24f, cp.y + is * 0.04f), IM_COL32(255, 244, 210, 120), thickness * 0.78f);
            break;
        case IconType::CameraHud:
            dl->AddRect(ImVec2(p.x + is * 0.18f, p.y + is * 0.24f), ImVec2(p.x + is * 0.82f, p.y + is * 0.76f), col, 2.0f, 0, thickness);
            dl->AddCircle(ImVec2(p.x + is * 0.34f, p.y + is * 0.40f), is * 0.04f, col, 10, thickness * 0.8f);
            dl->AddLine(ImVec2(p.x + is * 0.24f, p.y + is * 0.62f), ImVec2(p.x + is * 0.76f, p.y + is * 0.62f), col, thickness * 0.85f);
            break;
        case IconType::ViewOverlays:
            dl->AddLine(ImVec2(p.x + is * 0.18f, p.y + is * 0.78f), ImVec2(p.x + is * 0.18f, p.y + is * 0.28f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.40f, p.y + is * 0.78f), ImVec2(p.x + is * 0.40f, p.y + is * 0.18f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.62f, p.y + is * 0.78f), ImVec2(p.x + is * 0.62f, p.y + is * 0.44f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.84f, p.y + is * 0.78f), ImVec2(p.x + is * 0.84f, p.y + is * 0.12f), col, thickness);
            break;
        case IconType::PivotEdit:
            dl->AddCircle(cp, is * 0.08f, col, 12, thickness);
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.16f), ImVec2(cp.x, p.y + is * 0.84f), col, thickness * 0.85f);
            dl->AddLine(ImVec2(p.x + is * 0.16f, cp.y), ImVec2(p.x + is * 0.84f, cp.y), col, thickness * 0.85f);
            break;
        case IconType::PivotCenter:
            dl->AddCircle(cp, is * 0.08f, col, 12, thickness);
            dl->AddCircle(cp, is * 0.24f, col, 18, thickness * 0.85f);
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.08f), ImVec2(cp.x, p.y + is * 0.18f), col, thickness * 0.8f);
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.92f), ImVec2(cp.x, p.y + is * 0.82f), col, thickness * 0.8f);
            dl->AddLine(ImVec2(p.x + is * 0.08f, cp.y), ImVec2(p.x + is * 0.18f, cp.y), col, thickness * 0.8f);
            dl->AddLine(ImVec2(p.x + is * 0.92f, cp.y), ImVec2(p.x + is * 0.82f, cp.y), col, thickness * 0.8f);
            break;
        case IconType::Sensitivity:
            dl->AddCircle(cp, is * 0.24f, col, 24, thickness);
            dl->AddLine(cp, ImVec2(cp.x + is * 0.16f, cp.y - is * 0.08f), col, thickness * 1.1f);
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.16f), ImVec2(cp.x, p.y + is * 0.22f), col, thickness * 0.8f);
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.84f), ImVec2(cp.x, p.y + is * 0.78f), col, thickness * 0.8f);
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
            dl->AddTriangleFilled(ImVec2(p.x + is*0.34f, p.y + is*0.22f),
                                  ImVec2(p.x + is*0.34f, p.y + is*0.78f),
                                  ImVec2(p.x + is*0.78f, p.y + is*0.50f),
                                  col);
            break;
        case IconType::Pause:
            dl->AddRectFilled(ImVec2(p.x + is*0.28f, p.y + is*0.20f), ImVec2(p.x + is*0.44f, p.y + is*0.80f), col, 2.0f);
            dl->AddRectFilled(ImVec2(p.x + is*0.56f, p.y + is*0.20f), ImVec2(p.x + is*0.72f, p.y + is*0.80f), col, 2.0f);
            break;
        case IconType::Stop:
            dl->AddRectFilled(ImVec2(p.x + is*0.26f, p.y + is*0.26f), ImVec2(p.x + is*0.74f, p.y + is*0.74f), col, 3.0f);
            break;
        case IconType::Duplicate:
            dl->AddRect(ImVec2(p.x + is*0.18f, p.y + is*0.24f), ImVec2(p.x + is*0.58f, p.y + is*0.64f), col, 2.0f, 0, thickness);
            dl->AddRect(ImVec2(p.x + is*0.38f, p.y + is*0.38f), ImVec2(p.x + is*0.78f, p.y + is*0.78f), col, 2.0f, 0, thickness);
            break;
        case IconType::Help:
            dl->AddCircle(cp, is*0.34f, col, 20, thickness);
            dl->AddBezierQuadratic(ImVec2(cp.x - is*0.10f, p.y + is*0.34f),
                                   ImVec2(cp.x - is*0.02f, p.y + is*0.16f),
                                   ImVec2(cp.x + is*0.14f, p.y + is*0.28f),
                                   col, thickness);
            dl->AddLine(ImVec2(cp.x + is*0.10f, p.y + is*0.44f), ImVec2(cp.x + is*0.02f, p.y + is*0.58f), col, thickness);
            dl->AddCircleFilled(ImVec2(cp.x, p.y + is*0.74f), thickness * 1.15f, col);
            break;
        case IconType::AddKey:
            dl->AddRect(ImVec2(p.x + is*0.16f, p.y + is*0.42f), ImVec2(p.x + is*0.44f, p.y + is*0.58f), col, 2.0f, 0, thickness);
            dl->AddCircle(cp, is*0.14f, col, 16, thickness);
            dl->AddLine(ImVec2(p.x + is*0.70f, p.y + is*0.28f), ImVec2(p.x + is*0.70f, p.y + is*0.72f), col, thickness);
            dl->AddLine(ImVec2(p.x + is*0.48f, cp.y), ImVec2(p.x + is*0.92f, cp.y), col, thickness);
            break;
        case IconType::RemoveKey:
            dl->AddRect(ImVec2(p.x + is*0.16f, p.y + is*0.42f), ImVec2(p.x + is*0.44f, p.y + is*0.58f), col, 2.0f, 0, thickness);
            dl->AddCircle(cp, is*0.14f, col, 16, thickness);
            dl->AddLine(ImVec2(p.x + is*0.48f, cp.y), ImVec2(p.x + is*0.92f, cp.y), col, thickness);
            break;
        case IconType::PaintTool:
            dl->AddLine(ImVec2(p.x + is*0.22f, p.y + is*0.80f), ImVec2(p.x + is*0.64f, p.y + is*0.38f), col, thickness * 1.8f);
            dl->AddRect(ImVec2(p.x + is*0.58f, p.y + is*0.20f), ImVec2(p.x + is*0.82f, p.y + is*0.42f), col, 2.0f, 0, thickness);
            dl->AddQuadFilled(ImVec2(p.x + is*0.12f, p.y + is*0.90f),
                              ImVec2(p.x + is*0.24f, p.y + is*0.70f),
                              ImVec2(p.x + is*0.34f, p.y + is*0.80f),
                              ImVec2(p.x + is*0.24f, p.y + is*0.98f), col);
            break;
        case IconType::EraseTool:
            dl->AddQuadFilled(ImVec2(p.x + is*0.18f, p.y + is*0.68f),
                              ImVec2(p.x + is*0.38f, p.y + is*0.34f),
                              ImVec2(p.x + is*0.74f, p.y + is*0.52f),
                              ImVec2(p.x + is*0.54f, p.y + is*0.86f), col);
            dl->AddLine(ImVec2(p.x + is*0.54f, p.y + is*0.86f), ImVec2(p.x + is*0.76f, p.y + is*0.86f), col, thickness);
            break;
        case IconType::SoftenTool:
            dl->AddBezierQuadratic(ImVec2(p.x + is*0.14f, p.y + is*0.62f),
                                   ImVec2(cp.x, p.y + is*0.24f),
                                   ImVec2(p.x + is*0.86f, p.y + is*0.62f),
                                   col, thickness);
            dl->AddBezierQuadratic(ImVec2(p.x + is*0.18f, p.y + is*0.76f),
                                   ImVec2(cp.x, p.y + is*0.42f),
                                   ImVec2(p.x + is*0.82f, p.y + is*0.76f),
                                   col, thickness * 0.9f);
            break;
        case IconType::StampTool:
            dl->AddRect(ImVec2(p.x + is*0.24f, p.y + is*0.40f), ImVec2(p.x + is*0.74f, p.y + is*0.78f), col, 2.0f, 0, thickness);
            dl->AddRect(ImVec2(p.x + is*0.38f, p.y + is*0.18f), ImVec2(p.x + is*0.60f, p.y + is*0.38f), col, 2.0f, 0, thickness);
            dl->AddLine(ImVec2(p.x + is*0.49f, p.y + is*0.38f), ImVec2(p.x + is*0.49f, p.y + is*0.24f), col, thickness);
            break;
        case IconType::FillTool:
            dl->AddRect(ImVec2(p.x + is*0.22f, p.y + is*0.20f), ImVec2(p.x + is*0.70f, p.y + is*0.50f), col, 2.0f, 0, thickness);
            dl->AddLine(ImVec2(p.x + is*0.70f, p.y + is*0.50f), ImVec2(p.x + is*0.82f, p.y + is*0.62f), col, thickness);
            dl->AddLine(ImVec2(p.x + is*0.82f, p.y + is*0.62f), ImVec2(p.x + is*0.54f, p.y + is*0.88f), col, thickness);
            dl->AddLine(ImVec2(p.x + is*0.54f, p.y + is*0.88f), ImVec2(p.x + is*0.42f, p.y + is*0.76f), col, thickness);
            break;
        case IconType::CloneTool:
            dl->AddCircle(ImVec2(p.x + is*0.38f, p.y + is*0.42f), is*0.16f, col, 18, thickness);
            dl->AddCircle(ImVec2(p.x + is*0.62f, p.y + is*0.58f), is*0.16f, col, 18, thickness);
            dl->AddLine(ImVec2(p.x + is*0.52f, p.y + is*0.48f), ImVec2(p.x + is*0.48f, p.y + is*0.52f), col, thickness);
            break;
        case IconType::SprayTool: // Used as Scatter (Foliage & Mesh)
        {
            dl->AddLine(ImVec2(p.x + is * 0.1f, p.y + is * 0.85f), ImVec2(p.x + is * 0.9f, p.y + is * 0.85f), col, thickness);
            
            ImVec2 g1 = ImVec2(p.x + is * 0.35f, p.y + is * 0.85f);
            dl->AddBezierQuadratic(g1, ImVec2(g1.x - is*0.12f, g1.y - is*0.25f), ImVec2(g1.x - is*0.18f, g1.y - is*0.42f), col, thickness);
            dl->AddBezierQuadratic(g1, ImVec2(g1.x, g1.y - is*0.30f), ImVec2(g1.x - is*0.02f, g1.y - is*0.50f), col, thickness);
            dl->AddBezierQuadratic(g1, ImVec2(g1.x + is*0.10f, g1.y - is*0.22f), ImVec2(g1.x + is*0.15f, g1.y - is*0.35f), col, thickness);

            dl->AddCircle(ImVec2(p.x + is * 0.70f, p.y + is * 0.77f), is * 0.08f, col, 6, thickness);
            
            dl->AddCircleFilled(ImVec2(p.x + is * 0.20f, p.y + is * 0.35f), thickness * 0.8f, col);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.55f, p.y + is * 0.22f), thickness * 0.8f, col);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.80f, p.y + is * 0.40f), thickness * 0.8f, col);
            break;
        }
        case IconType::GrabTool:
            dl->AddCircle(cp, is * 0.22f, col, 20, thickness);
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.06f), ImVec2(cp.x, p.y + is * 0.30f), col, thickness);
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.70f), ImVec2(cp.x, p.y + is * 0.94f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.06f, cp.y), ImVec2(p.x + is * 0.30f, cp.y), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.70f, cp.y), ImVec2(p.x + is * 0.94f, cp.y), col, thickness);
            dl->AddTriangleFilled(ImVec2(cp.x, p.y + is * 0.02f), ImVec2(cp.x - is*0.05f, p.y + is * 0.12f), ImVec2(cp.x + is*0.05f, p.y + is * 0.12f), col);
            break;
        case IconType::InflateTool:
            dl->AddCircle(cp, is * 0.30f, col, 24, thickness);
            dl->AddLine(ImVec2(cp.x, cp.y - is * 0.16f), ImVec2(cp.x, cp.y + is * 0.16f), col, thickness);
            dl->AddLine(ImVec2(cp.x - is * 0.16f, cp.y), ImVec2(cp.x + is * 0.16f, cp.y), col, thickness);
            break;
        case IconType::SmoothTool:
            dl->AddBezierQuadratic(ImVec2(p.x + is * 0.14f, p.y + is * 0.60f),
                                   ImVec2(cp.x, p.y + is * 0.22f),
                                   ImVec2(p.x + is * 0.86f, p.y + is * 0.60f),
                                   col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.16f, p.y + is * 0.74f),
                        ImVec2(p.x + is * 0.84f, p.y + is * 0.74f),
                        col, thickness);
            break;
        case IconType::FlattenTool:
            dl->AddLine(ImVec2(p.x + is * 0.14f, p.y + is * 0.70f),
                        ImVec2(p.x + is * 0.86f, p.y + is * 0.70f),
                        col, thickness * 1.3f);
            dl->AddTriangle(ImVec2(p.x + is * 0.24f, p.y + is * 0.56f),
                            ImVec2(cp.x, p.y + is * 0.22f),
                            ImVec2(p.x + is * 0.76f, p.y + is * 0.56f),
                            col, thickness);
            break;
        case IconType::DrawTool:
            dl->AddLine(ImVec2(p.x + is * 0.24f, p.y + is * 0.82f),
                        ImVec2(p.x + is * 0.66f, p.y + is * 0.40f),
                        col, thickness * 1.5f);
            dl->AddTriangleFilled(ImVec2(p.x + is * 0.68f, p.y + is * 0.18f),
                                  ImVec2(p.x + is * 0.84f, p.y + is * 0.42f),
                                  ImVec2(p.x + is * 0.56f, p.y + is * 0.34f),
                                  col);
            break;
        case IconType::LayerTool:
            dl->AddRect(ImVec2(p.x + is * 0.18f, p.y + is * 0.28f),
                        ImVec2(p.x + is * 0.72f, p.y + is * 0.46f),
                        col, 2.0f, 0, thickness);
            dl->AddRect(ImVec2(p.x + is * 0.28f, p.y + is * 0.50f),
                        ImVec2(p.x + is * 0.82f, p.y + is * 0.68f),
                        col, 2.0f, 0, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.78f, p.y + is * 0.18f),
                        ImVec2(p.x + is * 0.92f, p.y + is * 0.32f),
                        col, thickness * 1.2f);
            dl->AddLine(ImVec2(p.x + is * 0.78f, p.y + is * 0.18f),
                        ImVec2(p.x + is * 0.86f, p.y + is * 0.08f),
                        col, thickness * 1.2f);
            break;
        case IconType::PinchTool:
            dl->AddLine(ImVec2(p.x + is * 0.18f, cp.y), ImVec2(p.x + is * 0.82f, cp.y), col, thickness);
            dl->AddTriangleFilled(ImVec2(p.x + is * 0.18f, cp.y),
                                  ImVec2(p.x + is * 0.34f, cp.y - is * 0.12f),
                                  ImVec2(p.x + is * 0.34f, cp.y + is * 0.12f),
                                  col);
            dl->AddTriangleFilled(ImVec2(p.x + is * 0.82f, cp.y),
                                  ImVec2(p.x + is * 0.66f, cp.y - is * 0.12f),
                                  ImVec2(p.x + is * 0.66f, cp.y + is * 0.12f),
                                  col);
            break;
        case IconType::ClayTool:
            dl->AddRect(ImVec2(p.x + is * 0.16f, p.y + is * 0.56f),
                        ImVec2(p.x + is * 0.84f, p.y + is * 0.78f),
                        col, 3.0f, 0, thickness);
            dl->AddBezierQuadratic(ImVec2(p.x + is * 0.18f, p.y + is * 0.56f),
                                   ImVec2(cp.x, p.y + is * 0.18f),
                                   ImVec2(p.x + is * 0.82f, p.y + is * 0.56f),
                                   col, thickness);
            break;
        case IconType::ClayStripsTool:
            dl->AddLine(ImVec2(p.x + is * 0.18f, p.y + is * 0.34f), ImVec2(p.x + is * 0.82f, p.y + is * 0.34f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.18f, p.y + is * 0.52f), ImVec2(p.x + is * 0.82f, p.y + is * 0.52f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.18f, p.y + is * 0.70f), ImVec2(p.x + is * 0.82f, p.y + is * 0.70f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.28f, p.y + is * 0.24f), ImVec2(p.x + is * 0.28f, p.y + is * 0.80f), col, thickness * 0.7f);
            dl->AddLine(ImVec2(p.x + is * 0.54f, p.y + is * 0.24f), ImVec2(p.x + is * 0.54f, p.y + is * 0.80f), col, thickness * 0.7f);
            break;
        case IconType::CreaseTool:
            dl->AddLine(ImVec2(p.x + is * 0.18f, p.y + is * 0.30f), ImVec2(cp.x, p.y + is * 0.76f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.82f, p.y + is * 0.30f), ImVec2(cp.x, p.y + is * 0.76f), col, thickness);
            dl->AddLine(ImVec2(cp.x, p.y + is * 0.20f), ImVec2(cp.x, p.y + is * 0.88f), col, thickness);
            break;
        case IconType::ScrapeTool:
            dl->AddLine(ImVec2(p.x + is * 0.16f, p.y + is * 0.72f), ImVec2(p.x + is * 0.84f, p.y + is * 0.52f), col, thickness * 1.5f);
            dl->AddRect(ImVec2(p.x + is * 0.30f, p.y + is * 0.22f), ImVec2(p.x + is * 0.62f, p.y + is * 0.36f), col, 2.0f, 0, thickness);
            break;
        case IconType::VertexMode:
            dl->AddCircleFilled(ImVec2(p.x + is * 0.22f, p.y + is * 0.72f), is * 0.10f, col, 14);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.50f, p.y + is * 0.24f), is * 0.10f, col, 14);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.78f, p.y + is * 0.72f), is * 0.10f, col, 14);
            dl->AddLine(ImVec2(p.x + is * 0.22f, p.y + is * 0.72f), ImVec2(p.x + is * 0.50f, p.y + is * 0.24f), col, thickness * 0.8f);
            dl->AddLine(ImVec2(p.x + is * 0.50f, p.y + is * 0.24f), ImVec2(p.x + is * 0.78f, p.y + is * 0.72f), col, thickness * 0.8f);
            break;
        case IconType::EdgeMode:
            dl->AddLine(ImVec2(p.x + is * 0.18f, p.y + is * 0.78f), ImVec2(p.x + is * 0.50f, p.y + is * 0.22f), col, thickness * 1.3f);
            dl->AddLine(ImVec2(p.x + is * 0.50f, p.y + is * 0.22f), ImVec2(p.x + is * 0.82f, p.y + is * 0.78f), col, thickness * 1.3f);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.18f, p.y + is * 0.78f), thickness * 0.9f, col, 10);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.82f, p.y + is * 0.78f), thickness * 0.9f, col, 10);
            break;
        case IconType::FaceMode:
            dl->AddQuad(ImVec2(p.x + is * 0.24f, p.y + is * 0.26f),
                        ImVec2(p.x + is * 0.76f, p.y + is * 0.20f),
                        ImVec2(p.x + is * 0.82f, p.y + is * 0.74f),
                        ImVec2(p.x + is * 0.18f, p.y + is * 0.80f),
                        col, thickness);
            break;
        case IconType::AddFace:
            dl->AddQuad(ImVec2(p.x + is * 0.18f, p.y + is * 0.30f),
                        ImVec2(p.x + is * 0.62f, p.y + is * 0.24f),
                        ImVec2(p.x + is * 0.68f, p.y + is * 0.68f),
                        ImVec2(p.x + is * 0.14f, p.y + is * 0.74f),
                        col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.74f, cp.y), ImVec2(p.x + is * 0.94f, cp.y), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.84f, p.y + is * 0.30f), ImVec2(p.x + is * 0.84f, p.y + is * 0.70f), col, thickness);
            break;
        case IconType::MergeVertices:
            dl->AddCircleFilled(ImVec2(p.x + is * 0.24f, cp.y), is * 0.09f, col, 12);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.76f, cp.y), is * 0.09f, col, 12);
            dl->AddLine(ImVec2(p.x + is * 0.34f, cp.y), ImVec2(p.x + is * 0.66f, cp.y), col, thickness);
            dl->AddTriangleFilled(ImVec2(cp.x, p.y + is * 0.34f),
                                  ImVec2(cp.x - is * 0.08f, p.y + is * 0.52f),
                                  ImVec2(cp.x + is * 0.08f, p.y + is * 0.52f),
                                  col);
            break;
        case IconType::WeldVertices:
            dl->AddCircleFilled(ImVec2(p.x + is * 0.28f, p.y + is * 0.68f), is * 0.08f, col, 12);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.50f, p.y + is * 0.34f), is * 0.08f, col, 12);
            dl->AddCircleFilled(ImVec2(p.x + is * 0.74f, p.y + is * 0.68f), is * 0.08f, col, 12);
            dl->AddLine(ImVec2(p.x + is * 0.28f, p.y + is * 0.68f), ImVec2(p.x + is * 0.50f, p.y + is * 0.34f), col, thickness * 0.8f);
            dl->AddLine(ImVec2(p.x + is * 0.74f, p.y + is * 0.68f), ImVec2(p.x + is * 0.50f, p.y + is * 0.34f), col, thickness * 0.8f);
            dl->AddCircle(cp, is * 0.30f, col, 20, thickness * 0.8f);
            break;
        case IconType::DissolveTopology:
            dl->AddLine(ImVec2(p.x + is * 0.18f, p.y + is * 0.24f), ImVec2(p.x + is * 0.82f, p.y + is * 0.78f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.18f, p.y + is * 0.78f), ImVec2(p.x + is * 0.82f, p.y + is * 0.24f), col, thickness);
            dl->AddCircle(cp, is * 0.12f, col, 14, thickness);
            break;
        case IconType::LoopCutTool:
            dl->AddLine(ImVec2(p.x + is * 0.20f, p.y + is * 0.26f), ImVec2(p.x + is * 0.20f, p.y + is * 0.78f), col, thickness * 0.9f);
            dl->AddLine(ImVec2(p.x + is * 0.50f, p.y + is * 0.20f), ImVec2(p.x + is * 0.50f, p.y + is * 0.84f), col, thickness * 1.5f);
            dl->AddLine(ImVec2(p.x + is * 0.80f, p.y + is * 0.26f), ImVec2(p.x + is * 0.80f, p.y + is * 0.78f), col, thickness * 0.9f);
            dl->AddTriangleFilled(ImVec2(p.x + is * 0.50f, p.y + is * 0.08f),
                                  ImVec2(p.x + is * 0.42f, p.y + is * 0.24f),
                                  ImVec2(p.x + is * 0.58f, p.y + is * 0.24f),
                                  col);
            break;
        case IconType::ExtrudeFaceTool:
            dl->AddQuad(ImVec2(p.x + is * 0.18f, p.y + is * 0.58f),
                        ImVec2(p.x + is * 0.54f, p.y + is * 0.52f),
                        ImVec2(p.x + is * 0.58f, p.y + is * 0.84f),
                        ImVec2(p.x + is * 0.12f, p.y + is * 0.88f),
                        col, thickness);
            dl->AddQuad(ImVec2(p.x + is * 0.34f, p.y + is * 0.22f),
                        ImVec2(p.x + is * 0.70f, p.y + is * 0.16f),
                        ImVec2(p.x + is * 0.74f, p.y + is * 0.48f),
                        ImVec2(p.x + is * 0.28f, p.y + is * 0.54f),
                        col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.20f, p.y + is * 0.58f), ImVec2(p.x + is * 0.36f, p.y + is * 0.22f), col, thickness * 0.9f);
            dl->AddLine(ImVec2(p.x + is * 0.54f, p.y + is * 0.52f), ImVec2(p.x + is * 0.70f, p.y + is * 0.16f), col, thickness * 0.9f);
            break;
        case IconType::DeleteFaceTool:
            dl->AddQuad(ImVec2(p.x + is * 0.20f, p.y + is * 0.26f),
                        ImVec2(p.x + is * 0.78f, p.y + is * 0.22f),
                        ImVec2(p.x + is * 0.82f, p.y + is * 0.78f),
                        ImVec2(p.x + is * 0.16f, p.y + is * 0.82f),
                        col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.28f, p.y + is * 0.34f), ImVec2(p.x + is * 0.72f, p.y + is * 0.68f), col, thickness * 1.2f);
            dl->AddLine(ImVec2(p.x + is * 0.72f, p.y + is * 0.34f), ImVec2(p.x + is * 0.28f, p.y + is * 0.68f), col, thickness * 1.2f);
            break;
        case IconType::ShadeFlatTool:
            dl->AddTriangle(ImVec2(p.x + is * 0.18f, p.y + is * 0.76f),
                            ImVec2(p.x + is * 0.50f, p.y + is * 0.22f),
                            ImVec2(p.x + is * 0.82f, p.y + is * 0.76f),
                            col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.50f, p.y + is * 0.22f), ImVec2(p.x + is * 0.50f, p.y + is * 0.76f), col, thickness);
            break;
        case IconType::ShadeSmoothTool:
            dl->AddBezierQuadratic(ImVec2(p.x + is * 0.16f, p.y + is * 0.72f),
                                   ImVec2(cp.x, p.y + is * 0.22f),
                                   ImVec2(p.x + is * 0.84f, p.y + is * 0.72f),
                                   col, thickness * 1.1f);
            dl->AddBezierQuadratic(ImVec2(p.x + is * 0.20f, p.y + is * 0.56f),
                                   ImVec2(cp.x, p.y + is * 0.38f),
                                   ImVec2(p.x + is * 0.80f, p.y + is * 0.56f),
                                   col, thickness * 0.9f);
            break;
        case IconType::Water:
            {
                float yoff1 = is * 0.65f;
                float yoff2 = is * 0.82f;
                dl->PathClear();
                for (int i = 0; i <= 16; ++i) {
                    float t = i / 16.0f;
                    float x = p.x + t * is;
                    float y = p.y + yoff1 + sinf(t * 6.28f) * (is * 0.08f);
                    dl->PathLineTo(ImVec2(x, y));
                }
                dl->PathStroke(col, 0, thickness);

                dl->PathClear();
                for (int i = 0; i <= 16; ++i) {
                    float t = i / 16.0f;
                    float x = p.x + t * is;
                    float y = p.y + yoff2 + sinf(t * 6.28f + 1.5f) * (is * 0.08f);
                    dl->PathLineTo(ImVec2(x, y));
                }
                dl->PathStroke(col, 0, thickness);

                ImVec2 drop_tip = ImVec2(cp.x, p.y + is * 0.14f);
                float drop_rad = is * 0.14f;
                dl->PathClear();
                dl->PathLineTo(drop_tip);
                dl->PathArcTo(ImVec2(cp.x, p.y + is * 0.38f), drop_rad, -0.1f * 3.1415f, 1.1f * 3.1415f, 10);
                dl->PathStroke(col, ImDrawFlags_Closed, thickness);
            }
            break;
        case IconType::Volumetric:
            {
                dl->AddLine(ImVec2(p.x + is * 0.2f, p.y + is * 0.75f), ImVec2(p.x + is * 0.8f, p.y + is * 0.75f), col, thickness);
                dl->PathArcTo(ImVec2(p.x + is * 0.3f, p.y + is * 0.58f), is * 0.18f, 0.7f * 3.1415f, 1.7f * 3.1415f, 8);
                dl->PathArcTo(ImVec2(p.x + is * 0.52f, p.y + is * 0.42f), is * 0.24f, 1.3f * 3.1415f, -0.2f * 3.1415f, 8);
                dl->PathArcTo(ImVec2(p.x + is * 0.72f, p.y + is * 0.58f), is * 0.18f, -0.7f * 3.1415f, 0.3f * 3.1415f, 8);
                dl->PathStroke(col, 0, thickness);
            }
            break;
        case IconType::Force:
            {
                dl->AddCircleFilled(cp, is * 0.08f, col);
                dl->PathClear();
                for (int i = 0; i <= 24; ++i) {
                    float a = i * (6.28318f / 24.0f);
                    float rx = is * 0.45f;
                    float ry = is * 0.15f;
                    float rot = 0.5f;
                    float x = cp.x + cosf(a) * rx * cosf(rot) - sinf(a) * ry * sinf(rot);
                    float y = cp.y + cosf(a) * rx * sinf(rot) + sinf(a) * ry * cosf(rot);
                    dl->PathLineTo(ImVec2(x, y));
                }
                dl->PathStroke(col, 0, thickness);
                
                dl->PathClear();
                for (int i = 0; i <= 24; ++i) {
                    float a = i * (6.28318f / 24.0f);
                    float rx = is * 0.45f;
                    float ry = is * 0.15f;
                    float rot = -0.5f;
                    float x = cp.x + cosf(a) * rx * cosf(rot) - sinf(a) * ry * sinf(rot);
                    float y = cp.y + cosf(a) * rx * sinf(rot) + sinf(a) * ry * cosf(rot);
                    dl->PathLineTo(ImVec2(x, y));
                }
                dl->PathStroke(col, 0, thickness);
            }
            break;
        case UIWidgets::IconType::World:
            dl->AddCircle(cp, is*0.4f, col, 0, thickness);
            dl->AddEllipse(cp, ImVec2(is*0.2f, is*0.4f), col, 0.0f, 0, thickness*0.8f); // Vertical axis
            dl->AddLine(ImVec2(cp.x-is*0.4f, cp.y), ImVec2(cp.x+is*0.4f, cp.y), col, thickness*0.8f); // Horizontal
            break;
        case UIWidgets::IconType::System:
            dl->AddRect(p, ImVec2(p.x + is, p.y + is * 0.7f), col, 2.0f, 0, thickness);
            dl->AddLine(ImVec2(p.x + is*0.3f, p.y + is), ImVec2(p.x + is*0.7f, p.y + is), col, thickness);
            dl->AddLine(ImVec2(p.x + is*0.5f, p.y + is*0.7f), ImVec2(p.x + is*0.5f, p.y + is), col, thickness);
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
            dl->AddRect(p, ImVec2(p.x+is, p.y+is), col, 1.0f, 0, thickness);
            dl->AddLine(p, ImVec2(p.x+is, p.y+is), col, thickness*0.5f);
            dl->AddLine(ImVec2(p.x+is, p.y), ImVec2(p.x, p.y+is), col, thickness*0.5f);
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
        default: break;
    }
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
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), ImGui::ColorConvertFloat4ToU32(bg), 10.0f);
    dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                ImGui::ColorConvertFloat4ToU32(ImVec4(accent.x, accent.y, accent.z, active ? 0.58f : (hovered ? 0.32f : 0.14f))),
                10.0f, 0, active ? 1.4f : 1.0f);
    dl->AddRectFilled(ImVec2(pos.x + 1.0f, pos.y + 1.0f),
                      ImVec2(pos.x + size.x - 1.0f, pos.y + size.y * 0.50f),
                      ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, hovered ? 0.028f : 0.016f)),
                      9.0f, ImDrawFlags_RoundCornersTop);

    const bool has_label = label && label[0] != '\0';
    const float icon_size = has_label
        ? ((std::min)(size.y - 12.0f, 24.0f))
        : ((std::min)(size.x, size.y) - 10.0f);
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

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 12.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 8.0f));
    
    // Dynamically derive from active theme colors
    ImGui::PushStyleColor(ImGuiCol_FrameBg, t.colors.surface);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ScaleColor(t.colors.surface, 1.3f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ScaleColor(t.colors.surface, 1.5f));
    
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(accent.x, accent.y, accent.z, 0.92f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4((std::min)(1.0f, accent.x + 0.10f), (std::min)(1.0f, accent.y + 0.10f), (std::min)(1.0f, accent.z + 0.10f), 1.0f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, ImVec4((std::min)(1.0f, accent.x + 0.10f), (std::min)(1.0f, accent.y + 0.10f), (std::min)(1.0f, accent.z + 0.10f), 1.0f));
    
    ImGui::PushStyleColor(ImGuiCol_Button, t.colors.primary);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ScaleColor(t.colors.primary, 1.2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ScaleColor(t.colors.primary, 0.8f));
    
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(accent.x, accent.y, accent.z, 0.18f));
    
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

