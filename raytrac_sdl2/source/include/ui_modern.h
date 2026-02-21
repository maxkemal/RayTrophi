/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          ui_modern.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
/*
 * RayTrophi Modern UI System
 * ==========================
 * Data-driven panel yapısı ve yeniden kullanılabilir widget bileşenleri.
 * Mevcut SceneUI ile tam uyumlu - kademeli geçiş için tasarlandı.
 */

#include "imgui.h"
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

 // ============================================================================
 // TEMA SİSTEMİ
 // ============================================================================

struct ThemeColors {
    ImVec4 primary;       // Ana renk (butonlar, vurgu)
    ImVec4 secondary;     // İkincil renk
    ImVec4 accent;        // Öne çıkan öğeler için
    ImVec4 background;    // Pencere arka planı
    ImVec4 surface;       // Kart/panel yüzeyleri
    ImVec4 text;          // Normal metin
    ImVec4 textMuted;     // Soluk metin
    ImVec4 success;       // Başarı durumu (yeşil)
    ImVec4 warning;       // Uyarı durumu (sarı)
    ImVec4 error;         // Hata durumu (kırmızı)
    ImVec4 border;        // Kenar renkleri
};

struct ThemeStyle {
    float windowRounding = 2.0f;
    float frameRounding = 3.0f;
    float grabRounding = 3.0f;
    float scrollbarRounding = 3.0f;
    float tabRounding = 3.0f;
    float popupRounding = 2.0f;
    ImVec2 framePadding = ImVec2(6, 4);
    ImVec2 itemSpacing = ImVec2(8, 6);
    ImVec2 windowPadding = ImVec2(10, 10);
};

struct Theme {
    std::string name;
    ThemeColors colors;
    ThemeStyle style;
};

class ThemeManager {
public:
    static ThemeManager& instance() {
        static ThemeManager inst;
        return inst;
    }

    void registerDefaultThemes();
    void addTheme(const Theme& theme);
    void setTheme(int index);
    void setTheme(const std::string& name);
    void applyCurrentTheme(float panelAlpha = 0.75f);

    const Theme& current() const { return themes_[currentIndex_]; }
    int currentIndex() const { return currentIndex_; }
    int themeCount() const { return static_cast<int>(themes_.size()); }
    const char* getThemeName(int index) const;
    std::vector<const char*> getAllThemeNames() const;

private:
    ThemeManager() { registerDefaultThemes(); }
    std::vector<Theme> themes_;
    int currentIndex_ = 4;
};

// ============================================================================
// MODERN WİDGET'LAR
// ============================================================================

namespace UIWidgets {

    // -------- YARDIMCI FONKSİYONLAR --------

    // Tooltip ile birlikte soru işareti gösterir
    void HelpMarker(const char* desc);

    // Rengi scale eder (hover/active durumlar için)
    ImVec4 ScaleColor(const ImVec4& c, float scale);

    // -------- SECTION BAŞLIKLARI --------

    // Modern gradient arka planlı section header
    // Döndürülen değer: section açık mı?
    bool BeginSection(const char* title,
        const ImVec4& accentColor = ImVec4(0.4f, 0.6f, 1.0f, 1.0f),
        bool defaultOpen = true);
    void EndSection();

    // Standart ImGui TreeNode flagları (tutarlılık için)
    ImGuiTreeNodeFlags GetSectionFlags(bool defaultOpen = true);

    // Renkli başlıklı section (mevcut kodla uyumlu)
    bool BeginColoredSection(const char* title,
        const ImVec4& titleColor,
        bool defaultOpen = true);

    // -------- BUTONLAR --------

    // Durum bazlı renkli buton (loaded/unloaded gibi)
    bool StateButton(const char* label, bool isActive,
        const ImVec4& activeColor = ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
        const ImVec4& inactiveColor = ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
        const ImVec2& size = ImVec2(0, 0));

    // Birincil aksiyon butonu
    bool PrimaryButton(const char* label,
        const ImVec2& size = ImVec2(0, 0),
        bool enabled = true);

    // İkincil buton (daha az vurgulu)
    bool SecondaryButton(const char* label,
        const ImVec2& size = ImVec2(0, 0),
        bool enabled = true);

    // Kırmızı/Tehlike butonu
    bool DangerButton(const char* label,
        const ImVec2& size = ImVec2(0, 0),
        bool enabled = true);

    void DrawThemeSelector(float& panel_alpha);

    // -------- GİRİŞ ALANLARI --------

    // Label + Slider + Tooltip tek satırda
    bool SliderWithHelp(const char* label, float* value,
        float min, float max,
        const char* tooltip,
        const char* format = "%.2f");

    // Label + DragInt + Tooltip
    bool DragIntWithHelp(const char* label, int* value,
        float speed, int min, int max,
        const char* tooltip);

    // Label + DragFloat + Tooltip
    bool DragFloatWithHelp(const char* label, float* value,
        float speed, float min, float max,
        const char* tooltip,
        const char* format = "%.2f");

    // -------- GÖRSEL ELEMANLAR --------

    // Renkli metin başlığı (section başlığı olarak kullanılabilir)
    void ColoredHeader(const char* text, const ImVec4& color);

    // İnce ayırıcı çizgi
    void Divider();

    // Durum göstergesi (yeşil/sarı/kırmızı nokta + metin)
    enum class StatusType { Success, Warning, Error, Info };
    void StatusIndicator(const char* text, StatusType status);

    // İlerleme çubuğu (modern stil)
    void ProgressBarEx(float fraction, const ImVec2& size,
        const char* overlay = nullptr,
        const ImVec4& barColor = ImVec4(0.3f, 0.6f, 1.0f, 1.0f));

    // -------- LAYOUT YARDIMCILARI --------

    // Returns the ideal width for property items (sliders/inputs) in a panel
    float GetInspectorItemWidth();
    // Returns the ideal width for full-width action buttons in a panel
    float GetInspectorActionWidth();

    // Sağa hizalı widget için boşluk hesapla
    float GetRightAlignOffset(float widgetWidth);

    // İki widget'ı yatay hizala (label-value gibi)
    void BeginLabelValuePair(const char* label, float labelWidth = 120.0f);
    void EndLabelValuePair();

    // -------- PROGRAMMATIC ICONS --------
    // High-stability geometric icons drawn via DrawList

    enum class IconType {
        Scene, Render, Terrain, Water, Volumetric, Force, World, System, Sculpt,
        Wind, Gravity, Physics, Vortex, Noise, Magnet,
        Camera, Light, Mesh,
        Timeline, Console, Graph, AnimGraph,
        LightPoint, LightDir, LightSpot, LightArea
    };

    void DrawIcon(IconType type, ImVec2 pos, float size, ImU32 color, float thickness = 1.5f);
    bool HorizontalTab(const char* label, IconType icon, bool active, float width = 0);

} // namespace UIWidgets

// ============================================================================
// PANEL YÖNETİMİ
// ============================================================================

struct PanelState {
    bool isVisible = true;
    bool isCollapsed = false;
    ImVec2 lastPosition = ImVec2(0, 0);
    ImVec2 lastSize = ImVec2(400, 500);
};

class PanelManager {
public:
    static PanelManager& instance() {
        static PanelManager inst;
        return inst;
    }

    PanelState& getState(const std::string& panelId);
    void saveStates(const std::string& filepath);
    bool loadStates(const std::string& filepath);

private:
    std::unordered_map<std::string, PanelState> states_;
};



