// Only included by the raster mesh consumer, never volume/SDF ray marching.
layout(set=0, binding=23, std430) readonly buffer RtScreenShadow {
    uvec4 rtShadowMeta;     // magic, width, height, PRIMARY scene-light index
    // ★★★★ Maske bir isik icin degil bir ISIK KUMESI icin gecerlidir. Dunya
    //   gunesi ile ona senkronlanmis directional sahne isigi AYNI yondedir;
    //   gorunurluk de aynidir, o yuzden tek dispatch ikisini de karsilar ve
    //   cascade'lerin IKISI de devredilir. CPU tarafi yonleri OLCEREK doldurur.
    uvec4 rtShadowCoverage; // x = sahne isigi bit maskesi, y = dunya gunesi
    vec2 rtShadowPixels[];  // visibility, camera depth
};
bool rtScreenShadow(uint light, out float visibility) {
    visibility = 1.0;
    if ((vMaterialID & 0x80000000u) != 0u) return false;
    if (rtShadowMeta.x != 0x52545348u) return false;
    // 32 = dunya gunesi slotu (kMaterialPreviewMaxSceneLights).
    bool covered = light < 32u ? (rtShadowCoverage.x & (1u << light)) != 0u
                              : rtShadowCoverage.y != 0u;
    if (!covered) return false;
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (any(lessThan(pixel, ivec2(0))) ||
        any(greaterThanEqual(pixel, ivec2(rtShadowMeta.yz)))) return false;
    vec2 sampleValue = rtShadowPixels[uint(pixel.y)*rtShadowMeta.y+uint(pixel.x)];
    // Transparent replay, impostors and uncovered prepass surfaces use the atlas.
    // ★★★★★ Bu esik ~2 ULP'dir, yani derinlik on gecisi ile ana gecisin
    //   gl_Position'i BIT BAZINDA esit olmak zorundadir. Iki vertex shader da
    //   `invariant gl_Position` ve AYNI ifade ile yazilmistir; birini degistirip
    //   otekini birakmak bu satiri sessizce her fragman icin false yapar ve
    //   cascade'i devretmis bir isik tamamen aydinlik kalir.
    if (sampleValue.y >= 1.0 || abs(sampleValue.y-gl_FragCoord.z) > 2e-7) return false;
    visibility = sampleValue.x;
    return true;
}
