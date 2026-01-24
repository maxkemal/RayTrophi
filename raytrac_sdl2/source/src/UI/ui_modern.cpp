/*
 * RayTrophi Modern UI System - Implementation
 * ============================================
 * ThemeManager, UIWidgets ve PanelManager implementasyonlarÄ±.
 */

#include "ui_modern.h"
#include <fstream>
#include <algorithm>

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
        t.style.frameRounding = 3.0f;
        t.style.windowRounding = 2.0f;
        t.style.scrollbarRounding = 3.0f;
        t.style.grabRounding = 3.0f;
        t.style.popupRounding = 2.0f;
        t.style.tabRounding = 3.0f;

        t.colors.primary    = ImVec4(0.25f, 0.25f, 0.27f, 1.0f);
        t.colors.secondary  = ImVec4(0.22f, 0.22f, 0.24f, 1.0f);
        t.colors.accent     = ImVec4(0.40f, 0.40f, 0.45f, 1.0f);
        t.colors.background = ImVec4(0.11f, 0.11f, 0.12f, 0.94f);
        t.colors.surface    = ImVec4(0.18f, 0.18f, 0.19f, 1.0f);
        t.colors.text       = ImVec4(0.95f, 0.95f, 0.95f, 1.0f);
        t.colors.textMuted  = ImVec4(0.45f, 0.45f, 0.45f, 1.0f);
        t.colors.success    = ImVec4(0.30f, 1.00f, 0.30f, 1.0f);
        t.colors.warning    = ImVec4(1.00f, 0.80f, 0.00f, 1.0f);
        t.colors.error      = ImVec4(1.00f, 0.15f, 0.15f, 1.0f);
        t.colors.border     = ImVec4(0.00f, 0.00f, 0.00f, 0.40f);
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
    
    // Ã–nce ImGui'nin varsayÄ±lan stillerini uygula
    switch (currentIndex_) {
        case 0: ImGui::StyleColorsDark(); break;
        case 1: ImGui::StyleColorsLight(); break;
        case 2: ImGui::StyleColorsClassic(); break;
        default: ImGui::StyleColorsDark(); break;
    }

    // Stil ayarlarÄ±
    style.WindowRounding    = t.style.windowRounding;
    style.FrameRounding     = t.style.frameRounding;
    style.GrabRounding      = t.style.grabRounding;
    style.ScrollbarRounding = t.style.scrollbarRounding;
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
    c[ImGuiCol_PopupBg]           = ImVec4(t.colors.surface.x * 1.1f, t.colors.surface.y * 1.1f,
                                            t.colors.surface.z * 1.1f, 0.98f);
    
    c[ImGuiCol_Text]              = t.colors.text;
    c[ImGuiCol_TextDisabled]      = t.colors.textMuted;
    
    c[ImGuiCol_FrameBg]           = t.colors.surface;
    c[ImGuiCol_FrameBgHovered]    = UIWidgets::ScaleColor(t.colors.surface, 1.3f);
    c[ImGuiCol_FrameBgActive]     = UIWidgets::ScaleColor(t.colors.surface, 1.5f);
    
    c[ImGuiCol_Button]            = t.colors.primary;
    c[ImGuiCol_ButtonHovered]     = UIWidgets::ScaleColor(t.colors.primary, 1.2f);
    c[ImGuiCol_ButtonActive]      = UIWidgets::ScaleColor(t.colors.primary, 0.8f);
    
    c[ImGuiCol_Header]            = t.colors.secondary;
    c[ImGuiCol_HeaderHovered]     = UIWidgets::ScaleColor(t.colors.secondary, 1.2f);
    c[ImGuiCol_HeaderActive]      = UIWidgets::ScaleColor(t.colors.secondary, 1.4f);
    
    c[ImGuiCol_SliderGrab]        = t.colors.accent;
    c[ImGuiCol_SliderGrabActive]  = UIWidgets::ScaleColor(t.colors.accent, 1.2f);
    
    c[ImGuiCol_Border]            = t.colors.border;
    
    c[ImGuiCol_Tab]               = UIWidgets::ScaleColor(t.colors.secondary, 0.8f);
    c[ImGuiCol_TabHovered]        = UIWidgets::ScaleColor(t.colors.secondary, 1.3f);
    c[ImGuiCol_TabActive]         = t.colors.secondary;
    
    c[ImGuiCol_ScrollbarBg]       = ImVec4(t.colors.background.x * 0.8f, t.colors.background.y * 0.8f,
                                            t.colors.background.z * 0.8f, 0.6f);
    c[ImGuiCol_ScrollbarGrab]     = UIWidgets::ScaleColor(t.colors.secondary, 1.2f);
    c[ImGuiCol_ScrollbarGrabHovered] = UIWidgets::ScaleColor(t.colors.secondary, 1.5f);
    c[ImGuiCol_ScrollbarGrabActive]  = UIWidgets::ScaleColor(t.colors.secondary, 1.8f);
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

// ============================================================================
// UI WIDGETS IMPLEMENTATION
// ============================================================================

namespace UIWidgets {

void HelpMarker(const char* desc) {
    ImGui::SameLine();
    
    // Use theme accent color for better visibility instead of disabled gray
    ImVec4 helpColor = ThemeManager::instance().current().colors.accent;
    ImGui::TextColored(helpColor, "[?]");
    
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        
        // Add a stylistic header
        ImGui::TextColored(helpColor, "Info");
        ImGui::Separator();
        
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
        ImGuiTreeNodeFlags_AllowItemOverlap |
        ImGuiTreeNodeFlags_FramePadding;
    
    if (defaultOpen)
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    
    return flags;
}

bool BeginSection(const char* title, const ImVec4& accentColor, bool defaultOpen) {
    ImGui::PushID(title);
    
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    float width = ImGui::GetContentRegionAvail().x;
    float height = ImGui::GetFrameHeight();
    
    // Sol tarafta accent bar Ã§iz
    ImU32 accentU32 = ImGui::ColorConvertFloat4ToU32(accentColor);
    drawList->AddRectFilled(
        cursorPos,
        ImVec2(cursorPos.x + 3, cursorPos.y + height),
        accentU32
    );
    
    // Ä°Ã§eriÄŸi biraz saÄŸa kaydÄ±r
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 8);
    
    bool opened = ImGui::TreeNodeEx(title, GetSectionFlags(defaultOpen));
    
    if (!opened)
        ImGui::PopID();
    
    return opened;
}

void EndSection() {
    ImGui::TreePop();
    ImGui::PopID();
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
        }

        // Panel Transparency   
        if (ImGui::SliderFloat("Panel Transparency", &panel_alpha, 0.1f, 1.0f, "%.2f")) {
            ImGuiStyle& style = ImGui::GetStyle();
            style.Colors[ImGuiCol_WindowBg].w = panel_alpha;
        }

        EndSection();
    }
}

} // namespace UIWidgets

