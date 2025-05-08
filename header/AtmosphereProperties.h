// AtmosphereProperties.h
#pragma once

struct AtmosphereProperties {
    float sigma_s;       // Scattering coefficient (hacim i�i sa��lma oran�)
    float sigma_a;       // Absorption coefficient (hacim i�i zay�flama oran�)
    float g;             // Phase function anisotropy (-1 geri, 0 izotropik, 1 ileri)
    float base_density;  // Yo�unluk (iste�e ba�l�)
    float temperature;   // Ortam s�cakl��� (iste�e ba�l� entropi etkileri i�in)
    bool  active;
};
