#ifndef RAYFUSION_SPECULAR_VISIBILITY_GLSL
#define RAYFUSION_SPECULAR_VISIBILITY_GLSL

// Directional, conservative sky visibility proxy, NOT reflected radiance.
// The producer's finite horizon represents a miss. Nearby walls have moments
// far below that horizon. Use the reflection direction, not irradiance or AO.
// This is a probe-origin estimate: parallax, small windows and narrow glossy
// lobes still need surface-origin rays. No additional scene geometry authority.
float rfSpecularSkyVisibility(vec3 worldPos, vec3 normal, vec3 reflection) {
    float horizon = rfGridSpacing.y;
    float spacing = rfGridSpacing.x;
    if (rfGridMinimum.w == 0 || rfGridCounts.w <= 0 ||
        !(horizon > 0.0) || !(spacing > 0.0) ||
        any(lessThanEqual(rfGridCounts.xyz, ivec3(0)))) return 1.0;
    if (any(isnan(worldPos)) || any(isinf(worldPos)) ||
        any(isnan(normal)) || any(isinf(normal)) ||
        any(isnan(reflection)) || any(isinf(reflection)) ||
        dot(normal, normal) < 1e-12 || dot(reflection, reflection) < 1e-12) return 1.0;
    ivec3 lo = rfGridMinimum.xyz;
    ivec3 hi = lo + rfGridCounts.xyz;
    // ★★★★ AYNI PAY, ve ayni sinirla: bu fonksiyon ile `rfSampleProbeField`
    //   ayni yerde susarsa piksel hem isimasini hem gorunurlugunu birden
    //   kaybeder ve engelsiz gokyuzune duser. Ikisini farkli sinirlarda
    //   birakmak, dikisi kucultup YERINI DEGISTIRMEKTEN ibaret olurdu.
    ivec3 cell = ivec3(floor(worldPos / spacing));
    if (any(lessThan(cell, lo - ivec3(1))) || any(greaterThan(cell, hi))) return 1.0;
    vec3 grid = worldPos / spacing - vec3(0.5);
    ivec3 base = ivec3(floor(grid));
    vec3 blend = fract(grid);
    vec3 n = normalize(normal);
    vec3 r = normalize(reflection);
    float total = 0.0;
    float visible = 0.0;
    for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 2; ++x) {
        ivec3 offset = ivec3(x, y, z);
        ivec3 neighbour = base + offset;
        if (any(lessThan(neighbour, lo)) || any(greaterThanEqual(neighbour, hi))) continue;
        vec3 weights = mix(vec3(1.0) - blend, blend, vec3(offset));
        float weight = weights.x * weights.y * weights.z;
        if (weight <= 0.0) continue;
        uint slot = rfSlotFor(neighbour);
        if (slot >= uint(rfGridCounts.w)) continue;
        RfProbeTexel packet = rfProbeTexels[rfDirectionalIndex(slot, r)];
        if (packet.irradiance.a < 0.5) continue;
        vec3 delta = worldPos - (vec3(neighbour) + vec3(0.5)) * spacing;
        float distanceToSurface = length(delta);
        vec3 direction = distanceToSurface > 1e-6 ? delta / distanceToSurface : n;
        float facing = distanceToSurface > 1e-6 ? 0.5 * dot(n, -direction) + 0.5 : 1.0;
        weight *= max(facing * facing, 0.01);
        vec2 connection = rfProbeTexels[rfDirectionalIndex(slot, direction)].distance.xy;
        float surfaceVisibility = rfMomentVisibility(connection, distanceToSurface);
        // Small tolerance avoids treating float accumulation error on an all-
        // miss packet as an occluder. This does not shorten the trace itself.
        float skyVisibility = rfMomentVisibility(packet.distance.xy, horizon * 0.999);
        total += weight;
        visible += weight * surfaceVisibility * skyVisibility;
    }
    // No evidence preserves the existing fallback. Blocked evidence is zero;
    // do not normalize by visibility and resurrect an occluded sky reflection.
    return total > 0.0 ? clamp(visible / total, 0.0, 1.0) : 1.0;
}
#endif
