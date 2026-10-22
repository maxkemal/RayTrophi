// Included once per renderer after its sampling adapters. Uses exactly the
// camera emission source and density; no live-only emitter list or heat guess.
vec3 volumeInternalRay(INTERNAL_VOLUME_TYPE volume, vec3 origin, vec3 direction,
                       float distance, int steps, vec3 extinctionPerDensity) {
    if (distance <= 0.0 || steps <= 0) return vec3(0.0);
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    float dt = distance / float(steps);
    for (int i = 0; i < 32; ++i) {
        if (i >= steps || max(throughput.r, max(throughput.g, throughput.b)) < 0.001) break;
        vec3 p = origin + direction * ((float(i) + 0.5) * dt);
        float density = INTERNAL_DENSITY(volume, p);
        if (density <= 0.0 || isnan(density) || isinf(density)) continue;
        vec3 segmentT = exp(-density * extinctionPerDensity * dt);
        vec3 source = INTERNAL_EMISSION(volume, p, density);
        if (any(isnan(source)) || any(isinf(source))) source = vec3(0.0);
        radiance += throughput * source * (vec3(1.0) - segmentT);
        throughput *= segmentT;
    }
    return radiance;
}
