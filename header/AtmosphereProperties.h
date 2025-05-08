// AtmosphereProperties.h
#pragma once

struct AtmosphereProperties {
    float sigma_s;       // Scattering coefficient (hacim içi saçýlma oraný)
    float sigma_a;       // Absorption coefficient (hacim içi zayýflama oraný)
    float g;             // Phase function anisotropy (-1 geri, 0 izotropik, 1 ileri)
    float base_density;  // Yoðunluk (isteðe baðlý)
    float temperature;   // Ortam sýcaklýðý (isteðe baðlý entropi etkileri için)
    bool  active;
};
