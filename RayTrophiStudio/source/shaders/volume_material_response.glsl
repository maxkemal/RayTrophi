#ifndef RT_VOLUME_MATERIAL_RESPONSE
#define RT_VOLUME_MATERIAL_RESPONSE

// Uniform response is deliberately opt-in, preserving old scene equations.
vec3 volumeEmissionSource(vec3 color, float legacyRadiance, float density,
                          float temperatureU, float strength, float response[5],
                          bool temperatureDriven) {
    if (response[0] < 0.5) {
        return color * legacyRadiance * density * strength;
    }
    if (density <= 0.0) {
        return vec3(0.0);
    }
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = max(mix(vec3(luminance), color, response[4]), vec3(0.0));
    float thermal = temperatureDriven
        ? (temperatureU > 0.0 ? pow(clamp(temperatureU, 0.0, 1.0), response[1]) : 0.0)
        : 1.0;
    return color * thermal * pow(density, response[2]) * strength * exp2(response[3]);
}

// Poisson probability of at least two interactions along the measured sky
// chord. Unlike a per-camera-step optical depth this is invariant to march dt.
float volumeHigherOrderSkyVisibility(float skyT, float amount, float albedo) {
    float transmittance = clamp(skyT, 0.0, 1.0);
    float tau = -log(max(transmittance, 1e-8));
    float twoOrMore = clamp(1.0 - (1.0 + tau) * transmittance, 0.0, 1.0);
    return clamp(amount, 0.0, 1.0) * clamp(albedo, 0.0, 1.0) *
           (1.0 - transmittance) * twoOrMore;
}

#endif
