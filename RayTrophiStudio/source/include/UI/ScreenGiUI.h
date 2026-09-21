#pragma once
#include "Api/RtApiScreenGi.h"
#include "UI/DiagnosticsLayout.h"
#include "imgui.h"

// Ekran-uzayı difüz GI: kontroller ve ÖLÇÜM aynı grupta.
// Ayrıntılı gerekçe: docs/dev/NEXT_BUILD_CHECKS.md
inline void DrawScreenGiGroup() {
    auto status = rtapi::screenGiStatus();
    auto settings = status.settings;
    static std::string error;

    // Grup durumu: kapalıyken nokta gri, çalışırken yeşil, istenip
    // çalışmıyorsa amber -- bölümü açmadan görülür.
    RtDiag::State state = RtDiag::State::Off;
    const char* stateText = "off";
    if (settings.enabled) {
        state = status.ready ? RtDiag::State::Ok : RtDiag::State::Warn;
        stateText = status.ready ? "running" : "not active";
    }
    if (!RtDiag::BeginGroup("##ScreenGiGroup", "Diffuse GI (screen)", state, stateText))
        return;

    bool changed = ImGui::Checkbox("Enabled##ScreenGi", &settings.enabled);
    if (settings.enabled) {
        int samples = settings.samples == 4 ? 2 : (settings.samples == 2 ? 1 : 0);
        ImGui::SetNextItemWidth(RtDiag::labelWidth());
        if (ImGui::Combo("Rays / pixel##ScreenGi", &samples, "1\0" "2\0" "4\0")) {
            settings.samples = 1u << samples; changed = true;
        }
        int radius = int(settings.filterRadius);
        ImGui::SetNextItemWidth(RtDiag::labelWidth());
        if (ImGui::SliderInt("Filter radius##ScreenGi", &radius, 0, 2)) {
            settings.filterRadius = uint32_t(radius); changed = true;
        }
        ImGui::SetNextItemWidth(RtDiag::labelWidth());
        changed = ImGui::DragFloat("Ray distance##ScreenGi", &settings.maxDistance,
                                   0.5f, 0.1f, 10000.0f, "%.1f") || changed;
        RtDiag::Help("Geometry beyond this distance is approximated by the environment, "
                     "so a short distance reads open sky where a wall actually stands.");
    }
    if (changed) rtapi::setScreenGi(settings, error);

    if (settings.enabled) {
        const auto& m = status.measurement;
        if (!m.measured) {
            RtDiag::Row("measurement", "none yet");
        } else {
            RtDiag::Row("sky visibility", "%.3f  (%.0f%% of px)",
                        m.mean_sky_visibility, m.visibility_coverage * 100.0f);
            RtDiag::Help("Cosine-weighted fraction of the hemisphere that reached sky, measured "
                         "from rays already traced this frame. In a closed room this must be far "
                         "below 1; near 1 means the occlusion measurement is not reaching the "
                         "shader. One frame behind, sampled from 1/64 of pixels.");
            RtDiag::Row("GI luminance", "%.3f", m.mean_gi_luminance);
            RtDiag::Row("coverage", "%.0f%% any / %.0f%% full",
                        m.any_confidence_fraction * 100.0f,
                        m.full_confidence_fraction * 100.0f);
            // ★★★★★ AYIRICI SATIR: "full" yüksekse tüketici ambient fallback'ini
            //   hiç çalıştırmıyor, yani gökyüzü görünürlüğü o piksellere
            //   UYGULANMIYOR ve fazla parlaklık GI radyansındadır. Düşükse
            //   fazlalık fallback kolundadır. İki kökün düzeltmesi farklı.
            RtDiag::Help("'full' pixels take screen GI radiance directly and skip the ambient "
                         "fallback, so the sky-visibility term does not apply to them. If ambient "
                         "looks too bright while this is high, the excess is in the GI radiance.");
            RtDiag::Note("%u px sampled, 1 frame behind", m.stats_pixels);
            if (m.full_confidence_fraction > 0.9f)
                RtDiag::Warn("visibility term is bypassed on most pixels");
            // ★ Kapsama dusukse ortalama dogru ama TUKETICIYE ULASMIYOR:
            //   verdikti olmayan pikselde shader 1.0'a duser ve engelsiz
            //   gokyuzu geri gelir. Ikinci satir bu delige ne kadarinin
            //   komsudan yamandigini soyler -- ve KAPSAMAYA EKLENMEZ, cunku
            //   yamanmis bir deger olculmus bir deger degildir.
            RtDiag::Row("visibility filled", "%.1f%% from neighbours",
                        m.visibility_filled_fraction * 100.0f);
            RtDiag::Help("Pixels with no hemisphere verdict of their own. Their visibility is "
                         "carried in from neighbours that do have one -- without that they fall "
                         "back to unoccluded sky, which is the brightest possible answer and the "
                         "reason unlit surfaces looked sky-painted with probes off.");
            if (m.visibility_coverage + m.visibility_filled_fraction < 0.9f)
                RtDiag::Warn("only %.0f%% measured + %.0f%% filled carry a visibility verdict; "
                             "the rest still fall back to unoccluded sky",
                             m.visibility_coverage * 100.0f,
                             m.visibility_filled_fraction * 100.0f);
        }
        if (!status.ready) RtDiag::Note("%s", status.reason.c_str());
    }
    if (!error.empty()) RtDiag::Fault("%s", error.c_str());
    RtDiag::EndGroup();
}
