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
        
        // Add a bit of spacing after header
        ImGui::Indent(8.0f); // Slight indent for content
        ImGui::Spacing();
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
        ImGui::Unindent(8.0f);
        ImGui::Spacing();
        ImGui::TreePop(); 
        
        // Draw the Border around the whole open section
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 endPos = ImGui::GetCursorScreenPos();
        
        // Close the box from header start to current cursor position
        // We exclude the top edge because we already drew a styled header there
        // Actually, let's draw a full rect border excluding top? Or just a full rect with rounding?
        // Let's do a full rect from Header Start to End Content
        
        // Header height was approx GetFrameHeight()
        // We want the border to encompass the header + content.
        
        // Rect logic:
        // Top-Left: state.startPos
        // Bottom-Right: (state.startPos.x + state.width, endPos.y)
        
        drawList->AddRect(
            state.startPos,
            ImVec2(state.startPos.x + state.width, endPos.y),
            state.borderColor,
            4.0f
        );

        ImGui::PopID();
    }
    
    // Extra spacing between sections
    ImGui::Spacing();
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

void DrawIcon(IconType type, ImVec2 p, float s, ImU32 col, float thickness) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float pading = s * 0.2f;
    float is = s - pading * 2;
    ImVec2 cp = ImVec2(p.x + s * 0.5f, p.y + s * 0.5f);
    p.x += pading; p.y += pading;

    switch (type) {
        case IconType::Scene:
            dl->AddRect(p, ImVec2(p.x + is * 0.7f, p.y + is * 0.7f), col, 0, 0, thickness);
            dl->AddRect(ImVec2(p.x + is * 0.3f, p.y + is * 0.3f), ImVec2(p.x + is, p.y + is), col, 0, 0, thickness);
            dl->AddLine(p, ImVec2(p.x + is * 0.3f, p.y + is * 0.3f), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.7f, p.y), ImVec2(p.x + is, p.y + is * 0.3f), col, thickness);
            dl->AddLine(ImVec2(p.x, p.y + is * 0.7f), ImVec2(p.x + is * 0.3f, p.y + is), col, thickness);
            dl->AddLine(ImVec2(p.x + is * 0.7f, p.y + is * 0.7f), ImVec2(p.x + is, p.y + is), col, thickness);
            break;
        case IconType::Render:
            dl->AddCircle(cp, is * 0.4f, col, 0, thickness * 1.5f);
            for(int i=0; i<8; i++) {
                float ang = i * (6.28f / 8.0f);
                dl->AddLine(ImVec2(cp.x + cosf(ang)*is*0.35f, cp.y + sinf(ang)*is*0.35f),
                           ImVec2(cp.x + cosf(ang)*is*0.55f, cp.y + sinf(ang)*is*0.55f), col, thickness * 1.5f);
            }
            break;
        case IconType::Terrain:
            dl->AddTriangleFilled(ImVec2(p.x, p.y + is), ImVec2(p.x + is*0.5f, p.y), ImVec2(p.x + is, p.y + is), col);
            break;
        case IconType::Water:
            for(int i=0; i<3; i++) {
                float yoff = i * (is/3.0f);
                dl->AddBezierQuadratic(ImVec2(p.x, p.y + yoff), ImVec2(p.x + is*0.25f, p.y + yoff + 5), ImVec2(p.x + is*0.5f, p.y + yoff), col, thickness);
                dl->AddBezierQuadratic(ImVec2(p.x + is*0.5f, p.y + yoff), ImVec2(p.x + is*0.75f, p.y + yoff - 5), ImVec2(p.x + is, p.y + yoff), col, thickness);
            }
            break;
        case IconType::Volumetric:
            dl->AddCircleFilled(ImVec2(cp.x - is*0.25f, cp.y + is*0.1f), is*0.25f, col);
            dl->AddCircleFilled(ImVec2(cp.x + is*0.25f, cp.y + is*0.1f), is*0.25f, col);
            dl->AddCircleFilled(ImVec2(cp.x, cp.y - is*0.15f), is*0.35f, col);
            break;
        case IconType::Force:
            dl->AddCircle(cp, is * 0.45f, col, 0, thickness);
            dl->AddLine(cp, ImVec2(cp.x, cp.y - is*0.4f), col, thickness * 1.5f);
            dl->AddTriangleFilled(ImVec2(cp.x - 4, cp.y - is*0.4f), ImVec2(cp.x + 4, cp.y - is*0.4f), ImVec2(cp.x, cp.y - is*0.55f), col);
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

bool HorizontalTab(const char* label, UIWidgets::IconType icon, bool active, float width) {
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    ImVec2 size = ImVec2(width > 0 ? width : ImGui::CalcTextSize(label).x + 44.0f, 22.0f);
    ImVec2 p = ImGui::GetCursorScreenPos();
    
    ImGui::PushID(label);
    bool clicked = ImGui::InvisibleButton("##htab", size);
    bool hovered = ImGui::IsItemHovered();
    ImGui::PopID();

    if (active) {
        dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,0.08f)), 3.0f);
        dl->AddRectFilled(ImVec2(p.x + 4, p.y + size.y - 2), ImVec2(p.x + size.x - 4, p.y + size.y), ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.9f, 0.8f, 1.0f)), 1.0f);
    } else if (hovered) {
        dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,0.04f)), 3.0f);
    }

    ImU32 iconCol = active ? ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.9f, 0.8f, 1.0f)) : ImGui::ColorConvertFloat4ToU32(ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
    DrawIcon(icon, ImVec2(p.x + 8, p.y + 3), 16.0f, iconCol);
    
    ImU32 textCol = active ? ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,1)) : ImGui::ColorConvertFloat4ToU32(ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
    dl->AddText(ImVec2(p.x + 30, p.y + 3), textCol, label);

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

