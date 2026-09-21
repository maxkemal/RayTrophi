#pragma once
#include "Api/RtApiReflection.h"
#include "imgui.h"

// ★★★★★ BU PANEL 2026-09-13'te SADELESTIRILDI, ve sebebi yazili bir doktrini
//   ihlal etmesiydi. `viewport_realtime_quality_panel.hpp` panelin uc katmanini
//   tanimliyor: (1) kullanici ayarlari gorunur, (2) rapor kapali, (3) gelistirici
//   tanilama kapali. Ilk hali gelistirici agacina DORT ciplak kadran, surekli
//   gorunen bir paragraf ve dort sayac koymustu.
//
// ★★★ Kadranlar kaldirildi, YERINE KONULMADI: butce artik kalite preset'inden
//   turer (`rasterReflectionSamples` vb.), tam olarak golge karosu / PCF tap /
//   isik butcesi gibi. Bu ayni zamanda bir TUTARSIZLIGI kapatiyor -- baska her
//   raster kalite dugmesi preset'ten turerken yalnizca yansima ciplak
//   sayilarla ayarlaniyordu.
//
// ★★ Script tarafinda hicbir sey kaybedilmedi: `rayfusion.set_reflections`
//   sayisal kontrollerin hepsini override edebiliyor (bir sayiyi adlandirmak
//   manuel kontrole gecirir). A/B olcumu panelin isi degil.
//
// ★ Panelde SUREKLI gorunen tek tanilama satiri, kullanicinin baska yoldan
//   OGRENEMEYECEGI olan: "her isin elendi, goruntu kapaliyken aynisi". Geri
//   kalan sayilar kapali bir alt agacta.
inline void DrawReflectionHelpMarker(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

inline void DrawReflectionControls() {
    auto status = rtapi::reflectionStatus();
    auto settings = status.settings;
    static std::string error;

    ImGui::Separator();
    if (!status.supported) ImGui::BeginDisabled();
    // TEK kullanici kontrolu.
    const bool toggled = ImGui::Checkbox("Ray-traced reflections##RfReflect",
                                         &settings.enabled);
    if (!status.supported) ImGui::EndDisabled();
    DrawReflectionHelpMarker(
        "Traces one ray per pixel for glossy surfaces instead of looking the "
        "reflection up in the environment map, so a metal or a polished floor "
        "shows the room rather than only the sky.\n\n"
        "Not metals only: the gate is the specular lobe, so varnished wood, "
        "painted floors and ceramic get it through the same Fresnel term -- "
        "and more of it at grazing angles.\n\n"
        "Ray count and the roughness cut-off follow the Realtime PBR Quality "
        "preset above. Scripts can override them "
        "(rayfusion.set_reflections).\n\n"
        "One bounce: no mirror inside a mirror. A ray that reaches the sky "
        "changes nothing, so an open outdoor scene looks the same as with this "
        "off.");

    if (toggled) {
        // ★ Toggle ederken butceyi PRESET'e birak. Panelden gelen bir acma
        //   islemi, script'in birakmis oldugu manuel sayilari devralmamali --
        //   kullanici bir kutu isaretledi, bir butce SECMEDI.
        settings.followQualityPreset = true;
        rtapi::setReflection(settings, error);
    }

    if (!status.settings.enabled) {
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
        return;
    }

    // ★★★ Surekli gorunen TEK tanilama: gecisin kostugu ama goruntuyu HIC
    //   degistirmedigi durum. Kullanici bunu ekrana bakarak ayirt edemez --
    //   "yansima yok" ile "yansima yok CUNKU her isin elendi" ayni gorunur.
    if (status.rays > 0 && status.shadedHits == 0)
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.35f, 1.0f),
            "Every ray was rejected: the image is the same as with reflections off.");

    if (ImGui::TreeNode("Reflection diagnostics##RfReflect")) {
        ImGui::TextDisabled("%s", status.ready ? "Dispatch recorded" : "Not active");
        ImGui::Text("budget: %u ray%s, roughness < %.2f, weight > %.3f",
                    status.settings.samples, status.settings.samples == 1u ? "" : "s",
                    status.settings.roughnessGate, status.settings.weightGate);
        // Uygulanan butcenin NEREDEN geldigi. "Preset'i takip ediyor" ile
        // "script manuel ayarlamis" ayni sayilari gosterebilir; hangisi oldugu
        // ancak burada yaziyorsa anlasilir.
        ImGui::TextDisabled(status.followedQualityPreset
            ? "from quality preset '%s'" : "manual override (script), preset '%s'",
            status.qualityPreset.c_str());
        // ★★★★★ SIRA ONEMLI. `shaded_hits` kabul sayisidir; `gated` yalnizca
        //   kac pikselin yansima ISTEDIGINI olcer, aldigini degil.
        ImGui::Text("gated %llu px | rays %llu | shaded %llu | sky %llu",
                    (unsigned long long)status.gatedPixels,
                    (unsigned long long)status.rays,
                    (unsigned long long)status.shadedHits,
                    (unsigned long long)status.skyMisses);
        if (status.rays > 0 && status.skyMisses == status.rays)
            ImGui::TextDisabled("Every ray reached the sky: nothing in the scene to "
                                "reflect, so the image is unchanged.");
        if (status.countersLagOneFrame)
            ImGui::TextDisabled("Counters describe the LAST dispatch, not this frame.");
        if (!status.reason.empty()) ImGui::TextWrapped("%s", status.reason.c_str());
        ImGui::TreePop();
    }
    if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
}
