#ifndef VOLUME_DENSITY_CURVE_GLSL
#define VOLUME_DENSITY_CURVE_GLSL
// Apply after input range normalization, before cutoff and multiplier.
// Monotone curve with zero origin; above-range values retain a linear tail.
float volumeDensityCurve(float x, float curve[12]) {
    x = max(x, 0.0);
    if (curve[0] < 0.5 || x <= 0.0) return x;
    if (x >= 1.0) return x * curve[8];
    float t = x * 7.0;
    int i = min(int(t), 6);
    return mix(curve[i+1], curve[i+2], t-float(i));
}
#endif
