#pragma once
#include "Api/RtApiScreenGi.h"
template<class Dict> Dict screenGiDictionary(const RayFusion::ScreenGiStatus& s=rtapi::screenGiStatus()) {
    Dict out;
    out["enabled"]=s.settings.enabled;
    out["samples"]=s.settings.samples;
    out["filter_radius"]=s.settings.filterRadius;
    out["max_distance"]=s.settings.maxDistance;
    out["supported"]=s.supported;out["ready"]=s.ready;
    out["width"]=s.width;out["height"]=s.height;
    out["primary_ray_budget"]=s.primaryRayBudget;
    // ★★ Sabit `false` YANILTICIYDI. GI'nin KENDI gecmis tamponu hala yok, ama
    //   2026-09-14'ten beri ornek tohumu TAA birikim indisiyle donuyor, yani
    //   gurultu TAA'nin ortalayabildigi bir sey. Ikisi ayni sey degil ve ayri
    //   raporlanmalari sart: biri "GI kendi yakinsiyor", digeri "GI baskasinin
    //   yakinsamasina biniyor".
    out["temporal_accumulation"]=false;          // GI'ye ait gecmis tamponu
    out["temporal_seed_rotation"]="taa_frame_index"; // gurultuyu ortalanabilir yapan sey
    out["normal_source"]="depth_geometric";
    out["ready_scope"]="last_recorded_raster_frame";
    out["reason"]=s.reason;
    // ── OLCUM (bir kare gecikmeli, piksellerin 1/64'unden) ──────────────────
    out["measured"]=s.measurement.measured;
    out["stats_pixels"]=s.measurement.stats_pixels;
    out["mean_sky_visibility"]=s.measurement.mean_sky_visibility;
    // ★★ Ortalamanin YANINDA okunmak zorunda: kapsama dusukse ortalama dogru
    //   ama tuketiciye ulasmiyor demektir.
    out["visibility_coverage"]=s.measurement.visibility_coverage;
    out["visibility_filled_fraction"]=s.measurement.visibility_filled_fraction;
    // ★★★★★ AYIRICI ALAN. Yuksekse tuketici fallback blogunu hic calistirmiyor,
    //   yani gorunurluk terimi o piksellere DOKUNMUYOR ve fazla parlaklik ekran
    //   GI'nin kendi radyansindadir. Dusukse fazlalik fallback kolundadir.
    out["full_confidence_fraction"]=s.measurement.full_confidence_fraction;
    out["any_confidence_fraction"]=s.measurement.any_confidence_fraction;
    out["mean_gi_luminance"]=s.measurement.mean_gi_luminance;
    return out;
}
