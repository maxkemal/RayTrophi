#ifndef ATMOSPHERE_TEMPERATURE_GLSL
#define ATMOSPHERE_TEMPERATURE_GLSL
// GPU mirror of atmosphere::sampleClimate: Kelvin/Celsius differ by a constant;
// the lapse rate stops at the same 11 km tropopause. Altitude includes scene offset.
float atmosphereAirTemperatureC(float seaLevelC, float lapse, float altitudeM) {
    return seaLevelC - lapse * min(altitudeM, 11000.0);
}
float atmosphereFrozenSurfaceShare(float temperatureC) {
    return 1.0 - smoothstep(0.0, 2.0, temperatureC);
}
#endif
