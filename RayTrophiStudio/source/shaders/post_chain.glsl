#ifndef RT_POST_CHAIN_GLSL_INCLUDED
#define RT_POST_CHAIN_GLSL_INCLUDED
#include "../include/PostProcess/ColorMath.h"

// Shared CPU/CUDA/GLSL math: exposure -> Bradford white balance ->
// view transform -> saturation/look gamma -> vignette -> sRGB.
struct RtPostParams {
    // ★★ IKI ayri pozlama, bilerek ayri: `exposure` renk derecelendirmesi
    //   (post.*), `cameraExposure` fiziksel pozlama ucgeni (ISO/enstantane/
    //   diyafram). Ayni noktada carpilirlar ama ayri sahipleri var ve ayri
    //   raporlanirlar; tek alanda toplamak hangisinin oynadigini gizlerdi.
    float cameraExposure;
    float exposure;
    float gamma;
    float saturation;
    float colorTemperature;
    float vignetteStrength;
    uint  toneMapping;
    uint  vignetteEnabled;
};

vec3 rtToneMap(vec3 c,uint type) { return rtPcTone(c,int(type)); }
float rtLinearToSRGB(float c) {
    c = clamp(c, 0.0, 1.0);
    return (c <= 0.0031308) ? (12.92 * c) : (1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

vec3 rtLinearToSRGB(vec3 c) {
    return vec3(rtLinearToSRGB(c.r), rtLinearToSRGB(c.g), rtLinearToSRGB(c.b));
}

float rtSanitize(float v) {
    if (isnan(v)) return 0.0;
    if (isinf(v)) return v > 0.0 ? 65504.0 : 0.0;
    return max(v, 0.0);
}

vec3 rtSanitize(vec3 c) {
    return vec3(rtSanitize(c.r), rtSanitize(c.g), rtSanitize(c.b));
}

// ── Zincir ─────────────────────────────────────────────────────────────────
// `uv01`: pikselin 0..1 ekran konumu. Vignette CPU'daki applyVignette ile ayni
// formulu kullanir: u=(x/w-0.5)*2, falloff = 1 - strength*(u^2+v^2).
//
// Donus: sRGB kodlanmis 0..1 deger, yani dogrudan 8-bit hedefe yazilabilir.
vec3 rtApplyPost(vec3 linearColor, RtPostParams p, vec2 uv01) {
    vec3 c = rtSanitize(linearColor) * p.exposure * p.cameraExposure;

    c = rtPcGrade(c,int(p.toneMapping),p.colorTemperature,p.saturation,p.gamma);

    if (p.vignetteEnabled != 0u && p.vignetteStrength > 0.0) {
        vec2 uv = (uv01 - vec2(0.5)) * 2.0;
        c *= clamp(1.0 - p.vignetteStrength * dot(uv, uv), 0.0, 1.0);
    }

    return rtLinearToSRGB(c);
}

#endif  // RT_POST_CHAIN_GLSL_INCLUDED
