#ifndef RAYFUSION_REFLECTION_GBUFFER_GLSL
#define RAYFUSION_REFLECTION_GBUFFER_GLSL

// RayFusion reflection G-buffer -- ORTAK kodlama.
//
// *** Bu dosya TEK bir sebeple var: kodlayan (material_preview_frag.frag) ile
//   kod cozen (reflection_trace.comp) AYNI fonksiyonu kullansin. Iki kopya
//   yazmak, bu deponun en sik tekrarlayan hata sinifini ("uretici != tuketici")
//   davet etmek olurdu: yansima, goruntudeki yuzeyin gosterdiginden BASKA bir
//   normali izler ve belirti "yansima biraz kaymis" olur -- kimsenin bug diye
//   raporlamadigi turden.
//
// Kanal sozlesmesi:
//   normal   (R16G16_SFLOAT)       .xy = oct(shading normal), dunya uzayi
//   specular (R16G16B16A16_SFLOAT) .rgb = F0*brdf.x + brdf.y (split-sum,
//                                        ambientBaseWeight dahil)
//                                  .a   = reflectionRoughness
//
// ★ Kapi .rgb'dedir, roughness'ta DEGIL. `.rgb == 0` "bu piksel yansima
//   istemiyor" demektir ve temizlenmis (dokunulmamis) piksel tam olarak budur.
//   Kapiyi roughness'a kurmak, sifirlanmis her pikseli "mukemmel ayna" ilan
//   etmek olurdu -- gokyuzunun kendisi dahil.

vec2 rfOctEncodeNormal(vec3 n) {
    if (!(dot(n, n) > 1e-12)) return vec2(0.0);
    n = normalize(n);
    vec2 p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    if (n.z > 0.0) return p;
    return (vec2(1.0) - abs(p.yx)) *
           vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
}

vec3 rfOctDecodeNormal(vec2 e) {
    vec3 n = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
        n.xy = (vec2(1.0) - abs(n.yx)) *
               vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    // Dejenere kodlama (temizlenmis texel) sifir uzunluk verir. Buradan bir
    // yon UYDURMAK, kapisi kapali bir pikselde yansima izlemek olurdu.
    return dot(n, n) > 1e-12 ? normalize(n) : vec3(0.0);
}

#endif
