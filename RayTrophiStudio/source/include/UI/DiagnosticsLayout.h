#pragma once
#include "imgui.h"
#include <cstdarg>
#include <cstdio>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════
// TANILAMA SATIRI DÜZENİ — gruplanmış, hizalı, kopyalanabilir
// ═══════════════════════════════════════════════════════════════════════════
//
// ★★★★★ TEK KURAL, ve bu dosyanın var olma sebebi: BİR SATIR HEM ÇİZİLİR HEM
//   KAYDEDİLİR, aynı çağrıda. "Kopyala" düğmesi için metni ikinci bir yerde
//   kurmak, ekranda yazan ile panoya giden değerin ayrışabilmesi demektir --
//   ve bu deponun en pahalı hata sınıfı tam olarak panelin yalan söylemesidir.
//   Ayrı bir kopya yolu, o yalanı taşınabilir hale getirirdi.
//
// ★★★ DÜZYAZI SATIR DEĞİLDİR. Bir cümlenin yeri `Help()` (hover) veya kod
//   yorumudur, panelin gövdesi değil. Her kadranın altına paragraf yazmak,
//   aranan sayıyı bulunamaz hale getiriyordu; ölçüm yüzeyi zaten IPC.
//
// ★★ Grup başlığında bir DURUM NOKTASI var: bölümü açmadan sağlığı görürsün.
//   Kapalı bir bölüm "sorun yok" demek değildir -- nokta onu söyler.
namespace RtDiag {

enum class State { Ok, Warn, Fault, Off };

inline std::string& buffer() { static std::string s; return s; }
inline float labelWidth() { return 168.0f; }

inline void record(const char* prefix, const char* text) {
    buffer().append(prefix).append(text).append("\n");
}

inline const char* vformat(const char* fmt, va_list args) {
    static char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, args);
    return buf;
}

inline ImVec4 stateColor(State s) {
    switch (s) {
        case State::Ok:    return ImVec4(0.45f, 0.80f, 0.50f, 1.0f);
        case State::Warn:  return ImVec4(1.00f, 0.75f, 0.30f, 1.0f);
        case State::Fault: return ImVec4(1.00f, 0.48f, 0.35f, 1.0f);
        default:           return ImVec4(0.45f, 0.45f, 0.45f, 1.0f);
    }
}

// Durum noktası + başlık. Dönüş: bölüm açık mı.
inline bool BeginGroup(const char* id, const char* title, State state,
                       const char* stateText) {
    ImGui::PushStyleColor(ImGuiCol_Text, stateColor(state));
    ImGui::TextUnformatted("*");
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, 6.0f);
    const bool open = ImGui::TreeNodeEx(id, ImGuiTreeNodeFlags_SpanAvailWidth, "%s", title);
    if (stateText && *stateText) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%s)", stateText);
    }
    buffer().append("\n[").append(title).append("] ")
            .append(stateText ? stateText : "").append("\n");
    return open;
}
inline void EndGroup() { ImGui::TreePop(); }

// Hizalı ana satır: etiket solda, değer sabit sütunda.
inline void Row(const char* label, const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    const char* value = vformat(fmt, args);
    va_end(args);
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine(labelWidth());
    ImGui::TextUnformatted(value);
    buffer().append("  ").append(label).append(" = ").append(value).append("\n");
}

// İkincil satır: sayının kırılımı. Hâlâ satır, hâlâ kopyalanır.
inline void Note(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    const char* value = vformat(fmt, args);
    va_end(args);
    ImGui::TextDisabled("    %s", value);
    record("    ", value);
}

inline void Warn(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    const char* value = vformat(fmt, args);
    va_end(args);
    ImGui::TextColored(stateColor(State::Warn), "  ! %s", value);
    record("  ! ", value);
}

inline void Fault(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    const char* value = vformat(fmt, args);
    va_end(args);
    ImGui::TextColored(stateColor(State::Fault), "  !! %s", value);
    record("  !! ", value);
}

// ★ Düzyazının TEK meşru yeri. Panelde yer kaplamaz, hover'da açılır.
//   Panoya da gitmez: kopyalanan şey ÖLÇÜM olmalı, açıklama değil.
inline void Help(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

inline void ResetBuffer(const char* header) {
    buffer().assign(header).append("\n");
}

inline void CopyAllButton(const char* id) {
    if (ImGui::SmallButton(id)) ImGui::SetClipboardText(buffer().c_str());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Copy every line in this panel, exactly as shown.");
}

} // namespace RtDiag
