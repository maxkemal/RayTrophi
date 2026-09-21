#pragma once
#include "Api/RtApiReflection.h"
// ★★★ Tek sozluk, uc tuketici (IPC, Python, panel). Ayri kopyalar yazmak,
//   ayni alanin iki yerde farkli isimle gorunmesi demek olurdu.
template<class Dict> Dict reflectionDictionary(
    const RayFusion::ReflectionStatus& s = rtapi::reflectionStatus()) {
    Dict out;
    out["enabled"] = s.settings.enabled;
    // ★★★ Bunlar ETKIN degerler. `follow_quality_preset` true ise bunlari
    //   preset belirledi, depolanan degerler degil -- ve `quality_preset` hangi
    //   preset oldugunu soyler. Istenen degeri raporlamak, panelin yalan
    //   soylemesi olurdu.
    out["follow_quality_preset"] = s.followedQualityPreset;
    out["quality_preset"] = s.qualityPreset;
    out["samples"] = s.settings.samples;
    out["roughness_gate"] = s.settings.roughnessGate;
    out["weight_gate"] = s.settings.weightGate;
    out["max_distance"] = s.settings.maxDistance;
    out["supported"] = s.supported;
    out["ready"] = s.ready;
    out["width"] = s.width;
    out["height"] = s.height;

    // ★★★★★ KABUL ALETI. Sirasi ve anlami:
    //   gated_pixels : kac piksel yansima ISTEDI (GPU'da olculdu, CPU kapisi degil)
    //   rays         : kac isin ATILDI
    //   shaded_hits  : kac isin gercekten bir yuzey GOLGELENDIRDI
    //   sky_misses   : kac isin gokyuzune gitti
    // `rays > 0` iken `shaded_hits == 0`, goruntunun yansima KAPALIYKEN
    // uretilenle birebir ayni oldugunu soyler. `gated_pixels` bunu asla
    // soyleyemez: o yalnizca kac pikselin yansima istedigini olcer, aldigini
    // degil. `sky_misses == rays` ise goruntu de degismemistir -- ama sebebi
    // kapsam degil, sahnede yansiyacak bir sey olmamasidir.
    out["gated_pixels"] = s.gatedPixels;
    out["rays"] = s.rays;
    out["shaded_hits"] = s.shadedHits;
    out["sky_misses"] = s.skyMisses;
    // ★★ Sayaclar bir KARE GERIDEN gelir. IPC ile ayar yazip HEMEN olcen bir
    //   test ONCEKI partinin sayilarini gorur; arada bir kare uretilmeli.
    out["counters_lag_one_frame"] = s.countersLagOneFrame;
    // Yazili kapsam: tek sicrama, aynada ayna yok; isabet yuzeyinin kendi
    // spekuleri yok. Clearcoat ikinci lobu env lookup'inda kalir.
    out["bounces"] = 1u;
    out["clearcoat_lobe"] = "environment_lookup";
    out["gate"] = "lobe_weight_and_roughness";
    out["reason"] = s.reason;
    return out;
}
