#pragma once
#include <cmath>
#include <cstdint>
#include <string>

namespace RayFusion {

// RayFusion — piksel basina spekuler yansima.
//
// ★★★ Bu gecis env lookup'ini DEGISTIRIR, metalik bir terim EKLEMEZ. Fragment
//   shader spekuler agirligi (F0*brdf.x + brdf.y) G-buffer'a yazar, gecis de
//   `agirlik * (izlenenRadyans - env(R))` ekler. Iki sonucu var: iskalayan isin
//   goruntuyu HIC degistirmez (dikis yapisal olarak olusamaz) ve gecis hic
//   kosmazsa goruntu bugunkunun aynisi olur.
//
// ★★ Kapi LOBA kurulu, malzeme sinifina DEGIL. Agirlik bir Fresnel terimi
//   oldugu icin verniklenmis ahsap, boyali zemin, seramik ve plastik ayni
//   yoldan gecer; `metallic > x` kapisi tam olarak o yuzeyleri elerdi.
struct ReflectionSettings {
    bool enabled = false;
    // ★★★★★ VARSAYILAN: butce PRESET'ten turer. Asagidaki sayilar o zaman
    //   YOK SAYILIR ve `ReflectionStatus::settings` ETKIN degerleri raporlar --
    //   depolanani degil. Panelin/IPC'nin istenen degeri gostermesi, bu deponun
    //   en pahali hata sinifidir (olcu aleti yalan soyluyor).
    // ★★ Bir script sayisal kontrollerden BIRINI adlandirdiginda bu bayrak o
    //   istekte otomatik olarak false olur: "samples=4 yazdim ama degismedi"
    //   sessiz bir sasirtma olurdu.
    bool followQualityPreset = true;
    // Probe basina degil PIKSEL basina isin: 1 varsayilan. 2/4 gurultuyu
    // dusurur ama maliyet dogrusal buyur.
    uint32_t samples = 1;
    // roughness bu degerin USTUNDE olan piksel elenir.
    // ★ 0,30 -> 0,15 (2026-09-13): 0,30'da alpha 0,09 ve lob ~20 derece; o
    //   konidan tek ornek yapisal olarak gurultuludur. Filtre yaricapi kapiya
    //   gore olceklendigi icin kapiyi buyutmek artik guvenli -- ama varsayilan
    //   gurultusuz olani secer.
    float roughnessGate = 0.15f;
    // Split-sum agirliginin en buyuk kanali bu degerin ALTINDA ise elenir.
    // Grazing acida agirlik Fresnel ile yukseldigi icin bu kapi kaba-ish bir
    // dielektrigi tam gorunur oldugu yerde geciriyor, tepeden bakista eliyor.
    float weightGate = 0.01f;
    float maxDistance = 200.0f;
};

inline bool validateReflection(const ReflectionSettings& s, std::string& error) {
    error.clear();
    if (s.samples != 1 && s.samples != 2 && s.samples != 4)
        error = "samples must be 1, 2 or 4";
    else if (!std::isfinite(s.roughnessGate) || s.roughnessGate <= 0.0f || s.roughnessGate > 1.0f)
        error = "roughness_gate must be finite and in (0, 1]";
    else if (!std::isfinite(s.weightGate) || s.weightGate < 0.0f || s.weightGate >= 1.0f)
        error = "weight_gate must be finite and in [0, 1)";
    else if (!std::isfinite(s.maxDistance) || s.maxDistance < 0.1f || s.maxDistance > 10000.0f)
        error = "max_distance must be finite and in [0.1, 10000] world units";
    return error.empty();
}

struct ReflectionStatus {
    ReflectionSettings settings;
    bool supported = false, ready = false;
    uint32_t width = 0, height = 0;

    // ★★★★★ BU DILIMIN KABUL ALETI, ve sirasi onemli.
    //   `gatedPixels`  : kac piksel yansima ISTEDI (CPU kapisi degil, GPU'da
    //                    olculmus -- G-buffer agirligi ve roughness'i gecen).
    //   `rays`         : kac isin ATILDI.
    //   `shadedHits`   : kac isin gercekten bir yuzey GOLGELENDIRDI.
    //   `skyMisses`    : kac isin gokyuzune gitti.
    //
    // ★ `rays > 0` iken `shadedHits == 0`, her isinin elendigi ve yayinlanan
    //   goruntunun yansima KAPALIYKEN uretilenle birebir ayni oldugu anlamina
    //   gelir. `gatedPixels` bunu asla soyleyemez: o yalnizca kac pikselin
    //   yansima istedigini olcer, aldigini degil.
    // ★ `skyMisses == rays` ise goruntu de degismemistir -- ama sebebi
    //   farklidir: kapsam degil, sahnede yansiyacak bir sey olmamasi.
    uint64_t gatedPixels = 0, rays = 0, shadedHits = 0, skyMisses = 0;

    // Butce preset'ten mi turedi. `settings` her iki durumda da ETKIN
    // degerleri tasir; bu alan NEDENINI soyler.
    bool followedQualityPreset = true;
    std::string qualityPreset;  // etkin butceyi belirleyen preset adi
    // ★★ Sayaclar bir KARE GERIDEN gelir: kare komut tamponunda sifirlanir,
    //   GPU doldurur, CPU sonraki karede okur. IPC ile bir ayar yazip HEMEN
    //   olcen bir test, ONCEKI partinin sayilarini gorur.
    bool countersLagOneFrame = true;
    std::string reason = "not prepared";
};

}
