#pragma once
#include <cmath>
#include <cstdint>
#include <string>

namespace RayFusion {
struct ScreenGiSettings {
    bool enabled = false;
    uint32_t samples = 1;
    uint32_t filterRadius = 2;
    float maxDistance = 100.0f;
};
inline bool validateScreenGi(const ScreenGiSettings& s, std::string& error) {
    error.clear();
    if (s.samples != 1 && s.samples != 2 && s.samples != 4)
        error = "samples must be 1, 2 or 4";
    else if (s.filterRadius > 2) error = "filter_radius must be 0, 1 or 2";
    else if (!std::isfinite(s.maxDistance) || s.maxDistance < 0.1f || s.maxDistance > 10000.0f)
        error = "max_distance must be finite and in [0.1, 10000] world units";
    return error.empty();
}
// ★★★★★ OLCUM, ayar degil. Bu alanlar olmadan ambient'in neden fazla parlak
//   oldugu SORULAMIYORDU: iki bambaska kok ayni belirtiyi uretiyor ve ikisinin
//   duzeltmesi farkli.
//
//   1) `full_confidence_fraction` yuksekse tuketici fallback blogunu HIC
//      calistirmiyor demektir -- yani gorunurluk terimi o piksellere
//      DOKUNMUYOR ve fazlalik ekran GI'nin kendi radyansindadir.
//   2) Dusukse fazlalik gorunurlukle zayiflatilmis fallback'tedir.
//
// ★★ `measured` ayri bir bayrak: sifirlar "olctum ve sifir cikti" ile
//   "hic olcemedim"i ayirmak zorunda. Varsayilan bir olcum degildir.
// ★ Degerler BIR KARE GECIKMELIDIR (GPU'yu durdurmamak icin beklemeden
//   okunur) ve piksellerin 1/64'unden ornekenir -- `stats_pixels` kac
//   pikselin katkida bulundugunu soyler ki kimse bunu tam sayim sanmasin.
struct ScreenGiMeasurement {
    bool measured = false;
    float mean_sky_visibility = 0.0f;   // [0,1], kosinus agirlikli
    float full_confidence_fraction = 0.0f; // light.w >= 1 olan piksel orani
    float any_confidence_fraction = 0.0f;  // light.w > 0 olan piksel orani
    float mean_gi_luminance = 0.0f;     // sahne-lineer, suzulmus GI
    // ★★★★★ KAC PIKSELIN gorunurluk verdikti VAR. `mean_sky_visibility` yalnizca
    //   BU piksellerin ortalamasidir -- kapsama dusukse ortalama dogru ama
    //   ALAKASIZ olur: tuketici verdikti olmayan pikselde olcumu hic goremez,
    //   `rfSpecularSkyVisibility` kanit bulamayip 1.0 doner ve ENGELSIZ gokyuzu
    //   geri gelir. Yani "gorunurluk 0.047" ile "gokyuzu engelleniyor" ayni sey
    //   DEGIL; arasindaki fark tam olarak bu alan.
    float visibility_coverage = 0.0f;
    // Kendi olcumu olmayip KOMSUDAN doldurulan piksel orani. `visibility_coverage`
    // ile TOPLANMAZ ve onun yerine gecmez: biri "olctum", digeri "olcemedim ama
    // yakinindan tasidim" der. Ikisini tek sayiya katlamak, doldurmanin ne kadar
    // is yaptigini ve nerede yetmedigini birden gorunmez yapardi.
    float visibility_filled_fraction = 0.0f;
    uint32_t stats_pixels = 0;          // ornege giren piksel sayisi
};
struct ScreenGiStatus {
    ScreenGiSettings settings;
    bool supported = false, ready = false;
    uint32_t width = 0, height = 0;
    uint64_t primaryRayBudget = 0;
    std::string reason = "not prepared";
    ScreenGiMeasurement measurement;
};
}
