#ifndef LENS_WATER_PHASE_GLSL
#define LENS_WATER_PHASE_GLSL
// All stored masses use liquid-equivalent radius cubed. Multiplication by
// 4*pi/3 yields water-equivalent volume in m^3 for every phase.
float lensLiquidMass(LensDrop drop) {
    return max(drop.a.z - drop.b.z - drop.b.w, 0.0);
}

float lensParticleRadius(LensDrop drop) {
    float volume = lensLiquidMass(drop) + drop.b.z * 10.0 + drop.b.w / 0.917;
    return pow(max(volume, 0.0), 1.0 / 3.0);
}

float lensRetention(uint id, float temperature) {
    if (id == 0u || id > 8u) return 0.0;
    if (id == 3u) return temperature <= 1.0 ? 1.0 : 0.35;
    if (id == 4u) return 0.5;
    if (id == 5u || id == 8u) return 0.15;
    if (id == 6u) return 0.08; // Hail rebounds; only a small impact residue stays.
    return 1.0;
}

// xyz = water-equivalent volume factor, snow mass share, ice mass share.
vec3 lensImpactPhase(uint id, float temperature) {
    if (id == 3u) return vec3(0.1, 1.0, 0.0);
    if (id == 4u) return vec3(0.5, 0.5, 0.0);
    if (id == 5u) return vec3(0.4, 0.0, 1.0);
    if (id == 8u) return vec3(0.917, 0.0, 1.0);
    if (id == 6u) return vec3(0.917, 0.0, temperature < 0.0 ? 1.0 : 0.0);
    if (id == 7u && temperature < 0.0) return vec3(1.0, 0.0, 1.0);
    return vec3(1.0, 0.0, 0.0);
}

void lensExchangePhase(inout LensDrop drop, float temperature, float dt) {
    // Empirical first-order heat transfer. No mass is created or lost by phase
    // exchange; evaporation and drainage are handled separately by the caller.
    if (temperature > 0.0) {
        drop.b.z *= exp(-0.04 * temperature * dt);
        drop.b.w *= exp(-0.012 * temperature * dt);
    } else if (temperature < 0.0) {
        float frozen = lensLiquidMass(drop) * (1.0 - exp(0.05 * temperature * dt));
        drop.b.w += frozen;
    }
}
#endif
