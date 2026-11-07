#ifndef RT_SURFACE_SUPPORT
#define RT_SURFACE_SUPPORT

float surfaceSupportBlend(float weight, float squaredWeight) {
    float effective = weight * weight / max(squaredWeight, 1e-12);
    float t = clamp((effective - 4.0) / 12.0, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float constrainSparseSurface(float reconstructed, float nearest, float supportBlend) {
    return reconstructed + (1.0 - supportBlend) *
        (max(reconstructed, nearest) - reconstructed);
}

#endif
