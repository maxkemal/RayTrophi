#define INTERNAL_VOLUME_TYPE uint
#define INTERNAL_DENSITY(v, p) sampleField(v, volAddress(v, 232u), p)
#define INTERNAL_EMISSION(v, p, d) emissionAt(v, p, d)
#include "volume_internal_integral.glsl"
#undef INTERNAL_VOLUME_TYPE
#undef INTERNAL_DENSITY
#undef INTERNAL_EMISSION

vec3 volumeInternalLighting(uint vi, vec3 p, vec3 viewDir) {
    int steps = clamp(volInt(vi, 280u), 0, 32);
    if (steps == 0 || volInt(vi, 256u) == 0) return vec3(0.0);
    vec3 absorption = max(volVec3(vi, 120u), vec3(0.0));
    float peak = max(absorption.r, max(absorption.g, absorption.b));
    vec3 weights = peak > 1e-5 ? absorption / peak : vec3(1.0);
    vec3 extinction = vec3(max(volFloat(vi, 100u), 0.0)) +
                      weights * max(volFloat(vi, 132u), 0.0);
    vec3 radiance = vec3(0.0);
    float weightSum = 0.0;
    // Stable six-direction quadrature, explicitly a realtime approximation.
    for (int axis = 0; axis < 3; ++axis) {
        for (int signIndex = 0; signIndex < 2; ++signIndex) {
            vec3 direction = vec3(0.0);
            direction[axis] = signIndex == 0 ? -1.0 : 1.0;
            float nearT, farT;
            if (!rayVolumeInterval(vi, p, direction, nearT, farT) || farT <= 0.0) continue;
            float phase = phaseDual(dot(viewDir, direction), vi);
            radiance += phase * volumeInternalRay(vi, p, direction, farT, steps, extinction);
            weightSum += phase;
        }
    }
    return weightSum > 1e-8 ? radiance / weightSum : vec3(0.0);
}
