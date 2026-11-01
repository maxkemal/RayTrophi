// Included once per renderer after its sampling adapters. Uses exactly the
// camera emission source and density; no live-only emitter list or heat guess.
// Emission is a source function or an independent coefficient j, and
// volumeEmissionIntegrate weighs each correctly.
//
// Segments start at one voxel and grow, so steps sets reach and never spacing.
// distance/steps made energy a function of the budget: a long first segment read
// hot gas deep inside the smoke at no attenuation (frame 69 of the nuke, median
// .641 at 32 steps, .798 at 4). Now fewer steps can only lose far light.
const float INTERNAL_SEGMENT_GROWTH = 1.35;

vec3 volumeInternalRay(INTERNAL_VOLUME_TYPE volume, vec3 origin, vec3 direction,
                       float distance, int steps, vec3 extinctionPerDensity) {
    if (distance <= 0.0 || steps <= 0) return vec3(0.0);
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    float voxel = INTERNAL_VOXEL(volume);
    float segment = voxel > 0.0 ? voxel : distance / float(steps);
    float flags = INTERNAL_FLAGS(volume);
    float t = 0.0;
    for (int i = 0; i < 32; ++i) {
        if (i >= steps || t >= distance ||
            max(throughput.r, max(throughput.g, throughput.b)) < 0.001) break;
        float dt = min(segment, distance - t);
        vec3 p = origin + direction * (t + 0.5 * dt);
        t += dt;
        segment *= INTERNAL_SEGMENT_GROWTH;
        float emissionDensity = 0.0;
        float density = INTERNAL_DENSITY(volume, p, emissionDensity);
        if (isnan(density) || isinf(density) ||
            !volumeEmissionSegmentActive(density, emissionDensity, flags)) continue;
        vec3 sigma = INTERNAL_OPTICAL(volume, p, density, emissionDensity) * extinctionPerDensity;
        vec3 source = INTERNAL_EMISSION(volume, p, emissionDensity);
        if (any(isnan(source)) || any(isinf(source))) source = vec3(0.0);
        radiance += throughput * volumeEmissionIntegrate(source, source, sigma, dt, flags);
        throughput *= exp(-sigma * dt);
    }
    return radiance;
}
