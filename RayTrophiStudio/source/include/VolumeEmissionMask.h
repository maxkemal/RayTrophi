#ifndef RT_VOLUME_EMISSION_MASK_H
#define RT_VOLUME_EMISSION_MASK_H

#if defined(__CUDACC__)
#define RT_MASK_FN __host__ __device__ inline
#elif defined(__cplusplus)
#define RT_MASK_FN inline
#else
#define RT_MASK_FN
#endif

// Shared by CPU, CUDA and GLSL. The mask input is evaluated shading density
// before its effective multiplier. Independent emission skips the smoke curve;
// remap/cutoff remain shared. Legacy emission includes the smoke curve.
// Optical coefficients remain untouched. Apply to emission BEFORE limiting.
// Packed parameters: enabled, center, width, strength.
RT_MASK_FN float volumeEmissionMaskFactor(float density, float multiplier,
    const float mask[4]) {
    if (mask[0] < 0.5f || mask[3] <= 0.0f) {
        return 1.0f;
    }
    if (!(multiplier > 0.0f) || !(density >= 0.0f)) {
        return 1.0f;
    }
    float width = mask[2] > 0.000001f ? mask[2] : 0.000001f;
    float x = (density / multiplier - (mask[1] - width * 0.5f)) / width;
    x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    float factor = 1.0f - mask[3] * x * x * (3.0f - 2.0f * x);
    return factor < 0.0f ? 0.0f : (factor > 1.0f ? 1.0f : factor);
}

#undef RT_MASK_FN
#endif
