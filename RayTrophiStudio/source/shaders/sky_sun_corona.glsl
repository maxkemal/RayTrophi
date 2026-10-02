// Sharp Mie core around the sun (aureole) -- shared by miss.rmiss (Vulkan RT)
// and material_preview_sky.frag (RayFusion).
//
// The sky-view LUT clamps the Mie phase at 2.0 (atmosphere_lut.comp): a
// 256x128 LUT cannot hold the forward peak, it would smear it over several
// texels. What the clamp cuts off is added back here, per pixel, analytically.
// ★ This used to live ONLY in miss.rmiss: RayFusion showed the broad halo (in
//   the LUT, identical in both) but no core -- the kind of difference that
//   reads as "realtime looks a bit softer", never as a bug.
//
// Magnitude matches the CPU calculateNishitaSky reference verbatim:
//   mieScat = mie_scattering * (mie_density * 0.15), mie_scattering default
//   3.996e-6 (not in either world block yet, hence the constant).

#ifndef SKY_SUN_CORONA_GLSL
#define SKY_SUN_CORONA_GLSL

vec3 skyMieCorona(float mu, float mieAnisotropy, float mieScaleHeight,
                  float atmosphereIntensity, vec3 transSun) {
    float g = clamp(mieAnisotropy, 0.0, 0.99);
    float phaseFull = (1.0 - g * g) /
        (4.0 * 3.14159265359 * pow(max(1.0 + g * g - 2.0 * g * mu, 0.0001), 1.5));
    float excessPhase = max(0.0, phaseFull - 2.0);
    const float MIE_SCAT = 3.996e-6;
    return transSun * (vec3(MIE_SCAT) * (mieScaleHeight * 0.15) * excessPhase * atmosphereIntensity);
}

#endif // SKY_SUN_CORONA_GLSL
