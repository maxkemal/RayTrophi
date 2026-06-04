#pragma once

#include <cmath>

#ifdef __CUDACC__
#define RT_GODRAYS_HD __host__ __device__ __forceinline__
#else
#define RT_GODRAYS_HD inline
#endif

namespace GodRaysModel {

constexpr float kMinIntensity = 0.001f;
constexpr float kMinDensity = 0.0f;
constexpr float kSunBelowHorizonCutoff = -0.05f;
constexpr float kMaxMarchDistance = 5000.0f;
constexpr int kMinSteps = 8;
constexpr int kMaxSteps = 48;
constexpr float kPhaseClamp = 6.0f;
constexpr float kSunRadianceScale = 0.15f;
constexpr float kMediaDensityScale = 0.02f;
constexpr float kHeightFalloff = 0.0002f;
constexpr float kTransmittanceCutoff = 0.01f;
constexpr float kPi = 3.14159265358979323846f;

RT_GODRAYS_HD float maxf(float a, float b) {
    return a > b ? a : b;
}

RT_GODRAYS_HD float minf(float a, float b) {
    return a < b ? a : b;
}

RT_GODRAYS_HD float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

RT_GODRAYS_HD int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

RT_GODRAYS_HD bool isEnabled(int enabled, float intensity, float density) {
    return enabled != 0 && intensity > kMinIntensity && density > kMinDensity;
}

RT_GODRAYS_HD float anisotropyFade(float sunDot, float mieAnisotropy) {
    float base = sunDot > 0.0f ? sunDot : 0.0f;
    float exponent = 1.0f + (1.0f - mieAnisotropy) * 10.0f;
#ifdef __CUDACC__
    return powf(base, exponent);
#else
    return std::pow(base, exponent);
#endif
}

RT_GODRAYS_HD bool isSunBelowHorizon(float sunY) {
    return sunY < kSunBelowHorizonCutoff;
}

RT_GODRAYS_HD float computeMarchDistance(float maxDistance, float fogDistance) {
    return minf(minf(maxDistance, fogDistance), kMaxMarchDistance);
}

RT_GODRAYS_HD int computeStepCount(float sunDot, int requestedSamples) {
    int steps = (sunDot > 0.98f) ? requestedSamples : (requestedSamples / 2);
    return clampi(steps, kMinSteps, kMaxSteps);
}

RT_GODRAYS_HD float computeStepSize(float marchDistance, int numSteps) {
    return (numSteps > 0) ? (marchDistance / static_cast<float>(numSteps)) : 0.0f;
}

RT_GODRAYS_HD float computeMiePhase(float sunDot, float mieAnisotropy) {
    float g = clampf(mieAnisotropy, 0.0f, 0.999f);
    float g2 = g * g;
    float denom = maxf(1.0f + g2 - 2.0f * g * sunDot, 0.0001f);
#ifdef __CUDACC__
    float phase = (1.0f - g2) / (4.0f * 3.14159265358979323846f * powf(denom, 1.5f));
#else
    float phase = (1.0f - g2) / (4.0f * 3.14159265358979323846f * std::pow(denom, 1.5f));
#endif
    return minf(phase, kPhaseClamp);
}

RT_GODRAYS_HD float computeMediaDensity(float godRaysDensity) {
    return godRaysDensity * kMediaDensityScale;
}

RT_GODRAYS_HD float computeNearFade(float t) {
    return clampf((t - 0.05f) * 8.0f, 0.0f, 1.0f);
}

RT_GODRAYS_HD float computeHeightFactor(float sampleY, float altitude) {
    float h = maxf(0.0f, sampleY + altitude);
#ifdef __CUDACC__
    return expf(-h * kHeightFalloff);
#else
    return std::exp(-h * kHeightFalloff);
#endif
}

RT_GODRAYS_HD float computeStepTransmittance(float sigmaT, float stepSize) {
#ifdef __CUDACC__
    return expf(-sigmaT * stepSize);
#else
    return std::exp(-sigmaT * stepSize);
#endif
}

RT_GODRAYS_HD float smoothstepf(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

RT_GODRAYS_HD float computeSolarCoreFade(float sunDot, float sunSizeDegrees) {
    float radius = sunSizeDegrees * (kPi / 180.0f) * 0.5f;
    radius = maxf(radius, 1e-4f);
    float mu = clampf(sunDot, -1.0f, 1.0f);
#ifdef __CUDACC__
    float angle = acosf(mu);
#else
    float angle = std::acos(mu);
#endif
    float inner = radius * 1.5f;
    float outer = radius * 6.0f;
    return smoothstepf(inner, outer, angle);
}

} // namespace GodRaysModel

#undef RT_GODRAYS_HD
