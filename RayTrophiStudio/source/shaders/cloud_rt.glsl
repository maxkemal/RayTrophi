// cloud_rt.glsl -- Vulkan RT cloud reference (docs/dev/ATMOSPHERE_CLOUDS.md §3.1).
//
// Clouds are not geometry: they are solved where a ray leaves the scene
// (miss.rmiss) by volumetric path tracing through the cloud field of
// cloud_common.glsl -- delta tracking for free flights, ratio tracking for
// transmittance, sun NEE at every scattering event, the sky picked up where a
// scattered path escapes, Jendersie-d'Eon phase. No artistic multipliers:
// silver lining, dark bases and overcast grey come out of the transport.
//
// Surfaces receive the cloud shadow through cloudSunTransmittance() in their
// directional-light NEE (closesthit), so ground and sky agree by construction.
//
// Bindings (RT set 0): 30-34 = cloud_common.glsl block, 35 = majorant map.
//
// Includer contract:
//   - always: nothing (the field + tracking need only bindings 30-35)
//   - CLOUD_RT_LIGHTING defined: `worldData` (VkWorldDataExtended) and
//     `atmosphereLUTs` declared before the include, and the includer
//     implements `vec3 cloudSkyAmbient(vec3 dir)` (sky radiance WITHOUT the
//     sun disk -- the sun arrives through NEE).

#ifndef CLOUD_RT_GLSL
#define CLOUD_RT_GLSL

// RT defaults; cloud_raster.comp (RayFusion) overrides all three.
#ifndef CLOUD_SET
#define CLOUD_SET 0
#endif
#ifndef CLOUD_BINDING_BASE
#define CLOUD_BINDING_BASE 30
#endif
#ifndef CLOUD_MAJORANT_BINDING
#define CLOUD_MAJORANT_BINDING 35
#endif
#ifndef CLOUD_CELL_BINDING
#define CLOUD_CELL_BINDING 36
#endif
#include "cloud_common.glsl"
#include "cloud_cirrus_shape.glsl"

layout(set = CLOUD_SET, binding = CLOUD_MAJORANT_BINDING) uniform sampler2D cloudMajorantMap;

const float CLOUD_RT_MAX_DIST = 1000000.0; // tall storm/cirrus horizons exceed 200 km
const int   CLOUD_RT_TRACK_BUDGET = 2048;   // tentative collisions + segments per tracking
const int   CLOUD_RT_MAX_INTERVALS = 6;     // 3 layers x (up to) 2 shell pieces

bool cloudRtEnabled()        { return (int(cloudParams.misc.w + 0.5) & 1) != 0; }
bool cloudRtSecondaryFull()  { return (int(cloudParams.misc.w + 0.5) & 2) != 0; }
int  cloudRtMaxBounces()     { return max(1, int(cloudParams.misc.z + 0.5)); }

// PCG hash stream, private to the cloud tracker (the path's own seed is not
// advanced: the tracker's sample count varies per pixel).
float cloudRand(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w) * (1.0 / 4294967296.0);
}

uint cloudSeed(uint a, vec3 p) {
    uvec3 q = floatBitsToUint(p);
    uint h = a ^ (q.x * 73856093u) ^ (q.y * 19349663u) ^ (q.z * 83492791u);
    h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
    return h;
}

// Ray/sphere around the planet centre. Planet-scale radii make the textbook
// b^2 - c form cancel catastrophically in float, so c is formed as
// (|oc| - r)(|oc| + r) and the roots through the stable quadratic.
vec2 cloudSphere(vec3 o, vec3 d, float r) {
    vec3 oc = o - cloudPlanetCenter();
    float len = length(oc);
    float b = dot(oc, d);
    float c = (len - r) * (len + r);
    vec3 q = oc - b * d;
    float ql = length(q);
    float h = (r - ql) * (r + ql);
    if (h < 0.0) return vec2(1.0, -1.0);
    float s = sqrt(h);
    float tBig = (b > 0.0) ? (-b - s) : (-b + s);
    float tSmall = (abs(tBig) > 1e-6) ? c / tBig : 0.0;
    return vec2(min(tBig, tSmall), max(tBig, tSmall));
}

// Segments of [0, tLimit] inside each enabled layer's shell, sorted by start.
// .x = t0, .y = t1, .z = layer index.
int cloudIntervals(vec3 o, vec3 d, float tLimit, out vec3 iv[CLOUD_RT_MAX_INTERVALS]) {
    int n = 0;
    float R = cloudParams.wind.z;
    // The ground occludes the shell behind it.
    vec2 g = cloudSphere(o, d, R);
    if (g.x <= g.y && g.x > 0.0) tLimit = min(tLimit, g.x);
    for (int li = 0; li < 3; ++li) {
        CloudLayerG L = cloudParams.layers[li];
        if (L.b.w < 0.5 || L.b.x <= 0.0 || L.a.z <= 0.0) continue;
        vec2 a = cloudSphere(o, d, R + L.a.x + L.a.y);
        if (a.x > a.y || a.y <= 0.0) continue;
        float s0 = max(a.x, 0.0);
        float s1 = min(a.y, tLimit);
        vec2 b = cloudSphere(o, d, R + L.a.x);
        bool inner = b.x <= b.y && b.y > 0.0;
        if (!inner) {
            if (s1 > s0) iv[n++] = vec3(s0, s1, float(li));
        } else {
            float e0 = min(b.x, s1);
            if (e0 > s0) iv[n++] = vec3(s0, e0, float(li));
            float e1 = max(b.y, s0);
            if (s1 > e1) iv[n++] = vec3(e1, s1, float(li));
        }
    }
    for (int i = 1; i < n; ++i) {
        vec3 k = iv[i];
        int j = i - 1;
        while (j >= 0 && iv[j].x > k.x) { iv[j + 1] = iv[j]; --j; }
        iv[j + 1] = k;
    }
    return n;
}

// Tile length in metres: one majorant fetch bounds a segment this long.
float cloudTileWorld() {
    return 1.0 / max(cloudParams.weather.x * float(textureSize(cloudMajorantMap, 0).x), 1e-9);
}

// Segment length for which ONE majorant fetch is still a bound. The map holds
// the max over the 3x3 tiles around the start tile, so the lookup (ray travel
// plus the wind-shear shift, cloudShear) may wander at most one tile. Shear
// shifts the lookup by up to 5.12*|shear| m per metre of altitude gained (the
// quadratic lean x its 1.6 modulation x the 1.6 curl scale), so a steep ray
// in wind left the neighbourhood and was clipped along straight tile edges.
float cloudSegmentLen(vec3 p, vec3 d) {
    float tile = cloudTileWorld();
    float sh = length(cloudParams.precip2.zw);
    if (sh <= 0.0) return tile;
    float dy = abs(dot(normalize(p - cloudPlanetCenter()), d));
    float travel = sqrt(max(1.0 - dy * dy, 0.0)) + 5.12 * sh * dy;
    return tile / max(travel, 1.0);
}

// Upper bound of cloudLayerDensity() on a segment <= one tile starting at p:
// density <= local coverage * extinction (cloud_common.glsl), and the map
// holds the max coverage input over the 3x3-tile neighbourhood.
float cloudLayerMajorant(int li, vec3 p) {
    CloudLayerG L = cloudParams.layers[li];
    p = cloudShear(L, p, cloudAltitude(p));
    vec2 uv = cloudWeatherUV(L, p);
    ivec2 sz = textureSize(cloudMajorantMap, 0);
    ivec2 tx = ivec2(floor(fract(uv) * vec2(sz))) % sz;
    float m = min(1.0, texelFetch(cloudMajorantMap, tx, 0).r + 0.004);
    // Interpolated linked density can exceed the active region's own knobs.
    // Bound by the group's largest coverage and extinction; interpolation
    // is convex and weather coordinates are those of the active region.
    if (L.d.z > 0.5 && L.d.w > 0.5) {
        for (int i = 0; i < 3; ++i) {
            CloudLayerG member = cloudParams.layers[i];
            if (member.b.w < 0.5 || member.d.z < 0.5 || member.d.w != L.d.w) continue;
            L.a.z = max(L.a.z, member.a.z);
            L.b.x = max(L.b.x, member.b.x);
        }
    }
    float cov = clamp(cloudRemap(m, 1.0 - L.a.z, 1.0, 0.0, 1.0), 0.0, 1.0);
    // Storm cells raise the cover; their map carries its own dilated bound.
    cov = max(cov, cloudStormCellAt(p).a);
    return cov * L.b.x;
}

// Delta tracking. True = a real collision at distance tHit in layer layerHit.
// False with tHit < 0 means the work cap was reached, not escape to clear sky.
bool cloudFreeFlight(vec3 o, vec3 d, float tLimit, inout uint rng, out float tHit, out int layerHit) {
    tHit = 0.0;
    layerHit = 0;
    vec3 iv[CLOUD_RT_MAX_INTERVALS];
    int n = cloudIntervals(o, d, tLimit, iv);
    int budget = CLOUD_RT_TRACK_BUDGET;
    for (int k = 0; k < n; ++k) {
        int li = int(iv[k].z);
        float t = iv[k].x;
        float tEnd = iv[k].y;
        while (t < tEnd && budget > 0) {
            float sEnd = min(t + cloudSegmentLen(o + d * t, d), tEnd);
            float maj = cloudLayerMajorant(li, o + d * t);
            --budget;
            if (maj <= 0.0) { t = sEnd; continue; }
            // Memoryless: restarting at the segment end is exact.
            for (;;) {
                t -= log(1.0 - cloudRand(rng)) / maj;
                if (t >= sEnd) { t = sEnd; break; }
                if (budget <= 0) { tHit = -1.0; return false; }
                --budget;
                vec3 p = o + d * t;
                float dens = cloudLayerDensity(li, p, cloudAltitude(p), true);
                if (cloudRand(rng) * maj < dens) {
                    tHit = t;
                    layerHit = li;
                    return true;
                }
            }
        }
        if (t < tEnd) { tHit = -1.0; return false; }
    }
    return false;
}

// Ratio tracking with Russian roulette on the running estimate. Unbiased
// transmittance of the cloud field along [0, tLimit] unless the hard work cap
// truncates it; capped paths use conservative zero rather than leak sky light.
float cloudTransmittance(vec3 o, vec3 d, float tLimit, inout uint rng) {
    vec3 iv[CLOUD_RT_MAX_INTERVALS];
    int n = cloudIntervals(o, d, tLimit, iv);
    if (n == 0) return 1.0;
    float T = 1.0;
    int budget = CLOUD_RT_TRACK_BUDGET;
    for (int k = 0; k < n; ++k) {
        int li = int(iv[k].z);
        float t = iv[k].x;
        float tEnd = iv[k].y;
        while (t < tEnd && budget > 0) {
            float sEnd = min(t + cloudSegmentLen(o + d * t, d), tEnd);
            float maj = cloudLayerMajorant(li, o + d * t);
            --budget;
            if (maj <= 0.0) { t = sEnd; continue; }
            for (;;) {
                t -= log(1.0 - cloudRand(rng)) / maj;
                if (t >= sEnd) { t = sEnd; break; }
                // Unknown remaining extinction cannot be published as clear
                // sky. Conservative truncation is dark-biased only on cap hits.
                if (budget <= 0) return 0.0;
                --budget;
                vec3 p = o + d * t;
                T *= 1.0 - cloudLayerDensity(li, p, cloudAltitude(p), true) / maj;
                if (T < 0.1) {
                    if (cloudRand(rng) > 0.5) return 0.0;
                    T *= 2.0;
                }
            }
        }
        if (t < tEnd) return 0.0;
    }
    return T;
}

bool cloudRtReference() { return (int(cloudParams.misc.w + 0.5) & 4) != 0; }
int  cloudRtSteps()     { return clamp(int(cloudParams.weather.w + 0.5), 16, 1024); }

// Optical depth along [0, tLimit] by a jittered fixed-count march (base
// shape, cheap). Bounded cost; accumulation removes the jitter.
float cloudOpticalDepth(vec3 o, vec3 d, float tLimit, int steps, float jitter, bool detail) {
    vec3 iv[CLOUD_RT_MAX_INTERVALS];
    int n = cloudIntervals(o, d, tLimit, iv);
    float tau = 0.0;
    for (int k = 0; k < n; ++k) {
        int li = int(iv[k].z);
        float len = iv[k].y - iv[k].x;
        int ns = clamp(int(float(steps) * min(1.0, len / max(cloudParams.layers[li].a.y, 1.0))), 2, steps);
        float dt = len / float(ns);
        for (int i = 0; i < ns; ++i) {
            vec3 p = o + d * (iv[k].x + (float(i) + jitter) * dt);
            tau += cloudLayerDensity(li, p, cloudAltitude(p), detail) * dt;
        }
    }
    return tau;
}

// Surface NEE: transmittance of the cloud field toward a directional light.
float cloudSunTransmittance(vec3 p, vec3 wi, uint seed) {
    if (!cloudRtEnabled() || wi.y <= -0.2) return 1.0;
    uint rng = cloudSeed(seed ^ 0x9e3779b9u, p);
    if (cloudRtReference()) return cloudTransmittance(p, wi, CLOUD_RT_MAX_DIST, rng);
    return exp(-cloudOpticalDepth(p, wi, CLOUD_RT_MAX_DIST, 24, cloudRand(rng), false));
}

#include "cloud_lightning.glsl"

#ifdef CLOUD_RT_LIGHTING

#ifdef CLOUD_SEPARATE_FLASH_HISTORY
// The same march integrates flash energy, but only stable cloud radiance is
// accumulated in RayFusion history. No extra field samples or buffers.
vec3 cloudFlashInscatter = vec3(0.0);
#endif

// Mixture of the two HG lobes of the Jendersie-d'Eon fit as the sampling
// density; the full phase (HG + Draine) is the target, the ratio is ~1.
vec3 cloudSampleHG(vec3 w, float g, float u1, float u2) {
    float cosT;
    if (abs(g) < 1e-3) {
        cosT = 1.0 - 2.0 * u1;
    } else {
        float s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
        cosT = clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
    }
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float phi = 2.0 * CLOUD_PI * u2;
    vec3 t = normalize(abs(w.y) < 0.999 ? cross(w, vec3(0.0, 1.0, 0.0)) : cross(w, vec3(1.0, 0.0, 0.0)));
    vec3 b = cross(w, t);
    return normalize(sinT * cos(phi) * t + sinT * sin(phi) * b + cosT * w);
}

// Returns the new direction and the phase / pdf weight.
vec3 cloudSamplePhase(int li, float diameterUm, vec3 w, inout uint rng, out float weight) {
    float phaseOverride = cloudParams.layers[li].e.w;
    if (phaseOverride <= 0.99) {
        float u1 = cloudRand(rng);
        float u2 = cloudRand(rng);
        weight = 1.0;
        return cloudSampleHG(w, phaseOverride, u1, u2);
    }
    CloudPhaseFit f = cloudPhaseFit(diameterUm);
    float u0 = cloudRand(rng), u1 = cloudRand(rng), u2 = cloudRand(rng);
    vec3 nd = cloudSampleHG(w, u0 < f.wD ? f.gD : f.gHG, u1, u2);
    float mu = dot(w, nd);
    float pdf = (1.0 - f.wD) * cloudPhaseHG(f.gHG, mu) + f.wD * cloudPhaseHG(f.gD, mu);
    weight = cloudLayerPhase(li, diameterUm, mu) / max(pdf, 1e-8);
    return nd;
}

vec3 cloudTransmittanceLUT(float altitude, float cosZenith) {
    if (worldData.w._pad5 == 0) return vec3(1.0);
    float u = (max(cosZenith, 0.01) + 0.2) / 1.2;
    float v = clamp(altitude / max(1.0, worldData.w.atmosphereHeight), 0.0, 1.0);
    return texture(atmosphereLUTs[0], vec2(u, v)).rgb;
}

// Sun irradiance arriving at a cloud point, before cloud self-shadowing.
// Same scale as the directional "Sun" light (irradiance = sun_intensity), so
// clouds and the ground they shade stay in one exposure; the atmosphere
// transmittance at the cloud's own altitude reddens it at low sun.
vec3 cloudSunIrradiance(vec3 p, vec3 sunDir) {
    vec3 up = normalize(p - cloudPlanetCenter());
    float cosS = dot(up, sunDir);
    if (cosS < -0.05) return vec3(0.0);
    float horizon = smoothstep(-0.05, 0.02, cosS);
    return cloudTransmittanceLUT(cloudAltitude(p), cosS) * worldData.w.sunIntensity * horizon;
}

// Atmosphere between the camera and a cloud at distance t (Bruneton ratio of
// two transmittance-LUT lookups; clamped because the LUT is not exact for
// downward rays).
vec3 cloudAerialTransmittance(vec3 o, vec3 d, float t) {
    vec3 p = o + d * t;
    vec3 upO = normalize(o - cloudPlanetCenter());
    vec3 upP = normalize(p - cloudPlanetCenter());
    vec3 tO = cloudTransmittanceLUT(cloudAltitude(o), dot(upO, d));
    vec3 tP = cloudTransmittanceLUT(cloudAltitude(p), dot(upP, d));
    return clamp(tO / max(tP, vec3(1e-4)), vec3(0.0), vec3(1.0));
}

vec3 cloudSkyAmbient(vec3 dir);

// ── Production marcher (Schneider 2015/2022 "Nubis", Hillaire 2016) ────────
// One jittered march per sample: adaptive steps (fine in cloud, coarse and
// map-skipped in empty sky, growing with distance), energy-conserving step
// integration, 6-sample light march toward the sun, multiple scattering by
// Wrenninge octaves, height-graded sky ambient. Bounded cost; the octave
// constants are what the reference path tracer calibrates.
const int   CLOUD_MS_OCTAVES = 3;
const float CLOUD_MS_A = 0.5;    // scattering per octave
const float CLOUD_MS_B = 0.5;    // extinction per octave (light reaches deeper)
const float CLOUD_MS_C = 0.5;    // phase anisotropy per octave

// Cirrus (ice, 8-12 km): an optically thin sheet on its own shell. Streaks
// stretched along the wind, single scattering with a forward ice lobe (HG
// g = 0.75) + sky ambient; it sits behind the volumetric layers, so it is
// attenuated by what the march left in T.
void cloudCirrus(vec3 o, vec3 d, float tLimit, vec3 sunDir, float mu, vec3 ambTop,
                 inout vec3 inscatter, inout float T) {
    if (cloudParams.cirrus.x < 0.5 || cloudParams.cirrus.w <= 0.0 || T <= 0.0) return;
    vec2 hit = cloudSphere(o, d, cloudParams.wind.z + cloudParams.cirrus.y);
    float t = hit.x > 0.0 ? hit.x : hit.y;
    if (t <= 0.0 || t > tLimit) return;
    vec2 g = cloudSphere(o, d, cloudParams.wind.z);
    if (g.x <= g.y && g.x > 0.0 && g.x < t) return;     // behind the ground
    vec3 p = o + d * t;
    vec2 wdir = cloudParams.cirrusDirection.xy;
    vec2 xz = p.xz - cloudParams.wind.xy;
    // Open-ended wisps share two existing reads. Smooth gradient warp avoids
    // the cusp-shaped bends produced by using cellular distance as a spine.
    vec2 st = vec2(dot(xz, wdir), dot(xz, vec2(-wdir.y, wdir.x))) *
              cloudParams.misc.x;
    vec3 phase = cloudShapePhase(2);
    float evo = cloudParams.weather.z;
    vec4 n = textureLod(cloudBaseNoise,
                        vec3(st.x * 0.23, 0.37, st.y * 0.75) + phase +
                        vec3(evo * 0.08, 0.0, evo * 0.12), 0.0);
    float bend = (n.a - 0.5) * 0.65;
    float trail = st.y + bend;
    vec4 n2 = textureLod(cloudDetailNoise,
                         vec3(st.x * 0.35 + (n.a - 0.5) * 0.2,
                              0.61 + (n.a - 0.5) * 0.35,
                              trail * 3.0) + phase + vec3(evo * 0.15), 0.0);
    float cov = cloudCirrusWisps(n, n2);
    if (cov <= 0.0) return;
    // Slant path through a thin sheet: optical depth grows toward the horizon.
    vec3 up = normalize(p - cloudPlanetCenter());
    float tau = cloudParams.cirrus.w * cov / max(abs(dot(up, d)), 0.08);
    float a = 1.0 - exp(-tau);
    vec3 Es = cloudSunIrradiance(p, sunDir);
    float iceG = cloudParams.scatterAlbedo.w <= 0.99 ?
                 cloudParams.scatterAlbedo.w : 0.75;
    vec3 L = cloudParams.scatterAlbedo.rgb *
             (Es * cloudPhaseHG(iceG, mu) + ambTop) + cloudParams.emissionSource.rgb;
    vec3 Tap = cloudAerialTransmittance(o, d, t);
    inscatter += T * a * (L * Tap + cloudSkyAmbient(d) * (vec3(1.0) - Tap));
    T *= 1.0 - a;
}

void cloudPrecipMarch(vec3 o, vec3 d, float tLimit, int steps, uint seed, inout vec3 inscatter, inout float T,
                      inout float depth);

// Work level of the next march. 0 = camera / RayFusion sky (full lighting),
// 1 = a path's secondary ray (reflections, sky light reaching the ground):
// its result is averaged over the hemisphere by the surface it lights, so it
// gets base-shape density, a two-tap light march, sparse ambient occlusion
// and an earlier cut-off. Under a sky-wide storm every bounce ray climbs into
// the cloud: that, not the camera ray, was the cost that exploded.
int cloudMarchLod = 0;

void cloudMarchLimited(vec3 o, vec3 d, float tLimit, int steps, uint seed,
                out vec3 inscatter, out float T, out float depth) {
#ifdef CLOUD_SEPARATE_FLASH_HISTORY
    cloudFlashInscatter = vec3(0.0);
#endif
    uint rng = cloudSeed(seed ^ 0x51ed270bu, o + d);
    vec4 lightningActivity = cloudLightningActivities();
    inscatter = vec3(0.0);
    T = 1.0;
    depth = -1.0;
    vec3 iv[CLOUD_RT_MAX_INTERVALS];
    int n = cloudIntervals(o, d, tLimit, iv);
    // Empty liquid intervals may still contain cirrus or precipitation.
    if (n == 0 && (cloudParams.cirrus.x < 0.5 || cloudParams.cirrus.w <= 0.0) &&
        cloudParams.precip.x <= 0.0) return;

    vec3 sunDir = normalize(worldData.w.sunDir);
    float mu = dot(d, sunDir);
    // Sky ambient = the hemisphere, not the zenith texel: zenith and horizon
    // averaged. The zenith alone is deep blue and painted every shadowed
    // cloud underside saturated navy.
    vec3 horiz = normalize(vec3(d.x, 0.08, d.z) + vec3(1e-4, 0.0, 0.0));
    vec3 ambTop = 0.5 * (cloudSkyAmbient(normalize(vec3(0.0, 1.0, 0.0) + d * 0.2)) +
                         cloudSkyAmbient(horiz));
    // Light from below = the GROUND, not the sky LUT's below-horizon texel
    // (which can be as bright as the sky and whitened cloud bases). Lambert
    // ground, albedo 0.25, lit by sky + sun; the sun part is shadowed per
    // sample by the cloud column above (see groundSun below).
    const float kGroundAlbedo = 0.25;
    vec3 ambBottom = kGroundAlbedo * ambTop;
    vec3 groundSun = kGroundAlbedo / CLOUD_PI * worldData.w.sunIntensity *
                     cloudTransmittanceLUT(cloudAltitude(o), sunDir.y) * max(sunDir.y, 0.0);
    float jitter = cloudRand(rng);
    float depthW = 0.0, depthSum = 0.0;
    bool secondary = cloudMarchLod != 0;
    int budget = secondary ? steps * 2 : steps * 4;
    // A sample this deep behind the cloud adds < this fraction of the pixel.
    float cutoff = secondary ? 0.03 : 0.005;
    // Night (or the sun below the horizon): the light march has nothing to
    // carry; the flash and the ambient light the cloud. Skip it outright.
    bool sunUp = worldData.w.sunIntensity > 1e-5 && sunDir.y > -0.1;

    for (int k = 0; k < n && budget > 0; ++k) {
        int li = int(iv[k].z);
        CloudLayerG L = cloudParams.layers[li];
        vec3 Es = vec3(-1.0);
        // Reserve bounded work for later regions. As a region nears its work
        // limit, complete it with coarser steps instead of exposing an abrupt
        // unprocessed tail as clear sky. The global step budget is unchanged.
        int reserve = (n - k - 1) * 8;
        float thick = max(L.a.y, 1.0);
        // Fine step: the shell in `steps` samples; it grows with distance so
        // horizon clouds do not eat the budget.
        float baseDt = thick / float(steps) * 2.0;
        // Phase lobes for this layer.
        float ph[CLOUD_MS_OCTAVES];
        float pIso = 1.0 / (4.0 * CLOUD_PI);
        float full = cloudLayerPhase(li, L.b.y, mu);
        for (int oc = 0; oc < CLOUD_MS_OCTAVES; ++oc)
            ph[oc] = mix(pIso, full, pow(CLOUD_MS_C, float(oc)));

        float t = iv[k].x;
        float tEnd = iv[k].y;
        float dt = baseDt * (1.0 + t / 5000.0);
        t += dt * jitter;
        // Ambient occlusion varies slowly: refreshed every `aoEvery` lit
        // samples once the view is mostly occluded (and on secondary rays).
        float tauTop = 0.0, tauBot = 0.0;
        int aoTick = 0;
        while (t < tEnd && budget > reserve) {
            int remaining = budget - reserve;
            --budget;
            dt = baseDt * (1.0 + t / 5000.0);
            float completionStep = (tEnd - t) / float(remaining);
            vec3 p = o + d * t;
            if (cloudLayerMajorant(li, p) <= 0.0) {
                // Only this segment is certified empty by the neighbourhood
                // majorant. A distance-grown dt can exceed it and erase the
                // next cloud along straight map-tile boundaries.
                float safeSkip = min(cloudSegmentLen(p, d), tEnd - t);
                if (completionStep <= safeSkip) {
                    t += safeSkip;
                    continue;
                }
                // At the budget tail use a coarse density sample instead of
                // applying the empty-region proof beyond its valid support.
                dt = max(dt, completionStep);
                p = o + d * min(t + completionStep * 0.5, tEnd);
            }
            float alt = cloudAltitude(p);
            // Detailed evaluation already rejects empty base shape before
            // erosion. Avoid solving all storm cells twice at the same point.
            // Detail erosion only shapes edges the viewer can still see: once
            // T is small the rest of the pixel cannot show it (Nubis LOD).
            bool wantDetail = !secondary && T > 0.15;
            float cheap;
            float sigma = cloudLayerDensityResolved(li, p, alt, wantDetail, cheap);
            if (cheap <= 0.0) {
                t += min(max(dt, completionStep), tEnd - t);
                continue;
            }
            // Inside cloud: finer steps (vertical structure), >= 25 m.
            // Deep in (low T) the remaining samples matter little: longer
            // steps, the integration below is energy-conserving per step.
            float deepGrow = 1.0 + 1.5 * (1.0 - smoothstep(0.05, 0.35, T));
            if (secondary) deepGrow *= 2.0;
            float stepLen = min(max(max(dt * 0.6 * deepGrow, 25.0), completionStep), tEnd - t);
            if (sigma > 0.0) {
                if (Es.r < 0.0) Es = cloudSunIrradiance(p, sunDir);
                // Light march: 6 samples doubling from 40 m (2.5 km), detail
                // in the first 3 -- the gaps between turrets are tens of
                // metres, a thickness-relative first step jumped over them --
                // plus one far base-shape sample for the rest of a tall tower.
                // LOD: 6 taps while the sample is clearly visible, fewer as T
                // falls (secondary: 2). The last tap stretches to the same
                // 2520 m, so the total path is unchanged -- only resolution.
                float tau = 0.0;
                if (sunUp) {
                    int lightTaps = secondary ? 2 : (T > 0.35 ? 6 : (T > 0.12 ? 4 : 3));
                    bool lightDetail = !secondary && T > 0.35;
                    float lt = 0.0;
                    for (int j = 0; j < lightTaps; ++j) {
                        float seg = (j == lightTaps - 1) ? 2520.0 - lt : 40.0 * pow(2.0, float(j));
                        // Stratify the existing light samples to soften fixed
                        // quadrature bands without adding another shaft march.
                        vec3 q = p + sunDir * (lt + seg * mix(0.2, 0.8, cloudRand(rng)));
                        tau += cloudDensity(q, lightDetail && j < 3 && j < lightTaps - 1) * seg;
                        lt += seg;
                    }
                    {
                        vec3 q = p + sunDir * (lt + 1500.0);
                        tau += cloudDensity(q, false) * 3000.0;
                    }
                }
                float h = clamp((alt - L.a.x) / thick, 0.0, 1.0);
                // Sky ambient, occluded by the cloud above / below (two cheap
                // base-shape samples). Without this the ambient term is the
                // same at every depth and a thick cloud reads as flat white.
                vec3 upN = normalize(p - cloudPlanetCenter());
                float dTop = thick * (1.0 - h), dBot = thick * h;
                if (L.d.z > 0.5) {
                    dTop = max(L.g.x + L.g.y - alt, 0.0);
                    dBot = max(alt - L.g.x, 0.0);
                }
                int aoEvery = secondary ? 4 : (T > 0.25 ? 1 : 2);
                if (aoTick++ % aoEvery == 0) {
                    vec3 qt = p + upN * (dTop * 0.5), qb = p - upN * (dBot * 0.5);
                    tauTop = (L.d.z > 0.5 ? cloudDensity(qt, false) :
                              cloudLayerDensity(li, qt, cloudAltitude(qt), false)) * dTop;
                    tauBot = (L.d.z > 0.5 ? cloudDensity(qb, false) :
                              cloudLayerDensity(li, qb, cloudAltitude(qb), false)) * dBot;
                }
                // The ground under this cloud sits in the cloud's own shadow:
                // approximate by the sun transmittance of this column.
                // The base sees a wide patch of ground, most of it outside
                // this cloud's shadow: floor of 0.35 on the shadowing.
                vec3 below = ambBottom + groundSun * (0.35 + 0.65 * exp(-tau));
                // Occlusion of the ambient by diffusion, not Beer: light
                // multiply-scattered through a thick cloud falls ~1/(1+k tau)
                // (k ~ 0.75 (1 - g)), so a tower's interior is dim grey, not
                // the exp(-tau) = 0 that left only the zenith colour.
                vec3 amb = 0.5 * (ambTop / (1.0 + 0.2 * tauTop) + below / (1.0 + 0.2 * tauBot));
                vec3 S = vec3(0.0);
                float a = 1.0, b = 1.0;
                float meanAlbedo = dot(L.e.rgb, vec3(1.0 / 3.0));
                for (int oc = 0; oc < CLOUD_MS_OCTAVES; ++oc) {
                    S += a * Es * exp(-tau * b) * ph[oc];
                    a *= CLOUD_MS_A * meanAlbedo;
                    b *= CLOUD_MS_B;
                }
                // Diffusion tail of the sun light (Faz 3e): the octaves above
                // are exp(-tau/8) at best and die inside a cumulonimbus; real
                // multiple scattering carries sunlight through it as
                // ~1/(1 + k tau). Weighted in only where the octaves have
                // faded, so thin cloud is unchanged.
                S += Es * pIso * 0.5 * (1.0 - exp(-0.125 * tau)) /
                     (1.0 + 0.1 * tau) * meanAlbedo * meanAlbedo * meanAlbedo;
                // "Powder" (Schneider 2015): thin sunward edges have little
                // in-scattered light yet, so crevices and turret rims darken
                // and towers separate. Faded toward the sun (silver lining).
                float powder = 1.0 - exp(-sigma * 80.0);
                S *= mix(1.0, powder, 0.6 * (0.5 - 0.5 * mu));
                // Ambient: sky radiance scattered isotropically, dimmed by
                // the cloud above/below (cheap: height gradient).
                S += amb;
                S = S * L.e.rgb + L.f.rgb;
                vec3 flash = cloudLightningSource(p, lightningActivity, !secondary);
                // Energy-conserving integration over the step (Hillaire 2016).
                float Ts = exp(-sigma * stepLen);
#ifdef CLOUD_SEPARATE_FLASH_HISTORY
                cloudFlashInscatter += T * flash * (1.0 - Ts);
#else
                S += flash;
#endif
                inscatter += T * S * (1.0 - Ts);
                depthSum += t * T * (1.0 - Ts);
                depthW += T * (1.0 - Ts);
                T *= Ts;
                if (T < cutoff) { T = 0.0; break; }
            }
            t += stepLen;
        }
        if (T <= 0.0) break;
    }
    if (depthW > 1e-4) {
        depth = depthSum / depthW;
        vec3 Tap = cloudAerialTransmittance(o, d, depth);
        inscatter = inscatter * Tap + cloudSkyAmbient(d) * (vec3(1.0) - Tap) * (1.0 - T);
#ifdef CLOUD_SEPARATE_FLASH_HISTORY
        cloudFlashInscatter *= Tap;
#endif
    }
    cloudCirrus(o, d, tLimit, sunDir, mu, ambTop, inscatter, T);
    cloudPrecipMarch(o, d, tLimit, steps, seed, inscatter, T, depth);
}

void cloudMarch(vec3 o, vec3 d, int steps, uint seed,
                out vec3 inscatter, out float T, out float depth) {
    cloudMarchLimited(o, d, CLOUD_RT_MAX_DIST, steps, seed, inscatter, T, depth);
}

// Precipitation shafts in front of the clouds (§3.1): the part of the ray
// under layer 0's base, within kPrecipMaxDist. Light = sun through the cloud
// column above (cheap base-shape optical depth) with a mild forward lobe for
// rain, isotropic snow, plus the sky / ground ambient the base lets through.
// Shafts hang below the base: seen from below they are in front of the
// cloud they fall from, from above behind it. Every ray that carries clouds
// gets the shafts: sky rays (miss shader, RayFusion sky layer) and finite
// segments in front of a surface (cloud_hit_segment RT, cloud_scene.comp
// RayFusion), so a shaft hangs in front of a mountain and ends on it.
const float kPrecipMaxDist = 40000.0;
void cloudPrecipMarch(vec3 o, vec3 d, float tLimit, int steps, uint seed, inout vec3 inscatter, inout float T,
                      inout float depth) {
    if (cloudParams.precip.x <= 0.0 || cloudParams.precip.y <= 0.0) return;
    CloudLayerG L = cloudParams.layers[0];
    if (L.b.w < 0.5) return;
    float R = cloudParams.wind.z;
    float tEnd = min(kPrecipMaxDist, tLimit);
    float t0 = 0.0;
    // A ray that met a surface ends the shaft there (terrain, even below
    // y = 0). A ray that met NOTHING ends at the implied scene ground (altitude
    // offset), not sea level; a camera under it (no terrain, y < 0) starts
    // where the ray climbs out -- looking down it sees no rain instead of
    // 40 km of it.
    if (tLimit >= CLOUD_RT_MAX_DIST * 0.999) {
        vec2 g = cloudSphere(o, d, R + cloudParams.wind.w);
        if (cloudAltitude(o) < cloudParams.wind.w) {
            if (g.x > g.y || g.y <= 0.0) return;
            t0 = g.y;
        } else if (g.x <= g.y && g.x > 0.0) {
            tEnd = min(tEnd, g.x);
        }
    }
    vec2 b = cloudSphere(o, d, R + L.a.x);
    bool fromAbove = cloudAltitude(o) >= L.a.x;
    if (fromAbove) {           // above the base: enter at it
        if (b.x > b.y || b.x <= 0.0) return;
        t0 = b.x;
        tEnd = min(tEnd, b.y);
    } else if (b.x <= b.y && b.y > 0.0) {
        tEnd = min(tEnd, b.y);                 // below: leave through it
    }
    if (tEnd <= t0) return;

    uint rng = cloudSeed(seed ^ 0x2c1b3c6du, o + d * 3.0);
    vec3 sunDir = normalize(worldData.w.sunDir);
    float mu = dot(d, sunDir);
    bool snow = cloudParams.precip2.x > 0.5;
    float ph = snow ? 1.0 / (4.0 * CLOUD_PI) : mix(1.0 / (4.0 * CLOUD_PI), cloudPhaseHG(0.6, mu), 0.5);
    vec3 ambTop = cloudSkyAmbient(normalize(vec3(0.0, 1.0, 0.0) + d * 0.2));
    vec3 Es = vec3(-1.0);
    bool secondary = cloudMarchLod != 0;
    int n = max(steps / 3, secondary ? 8 : 16);
    // Lit cloud bases light the curtain during a flash.
    vec4 flashActivity = cloudFlashAny() ? cloudLightningActivities() : vec4(0.0);
    bool flashOn = any(greaterThan(flashActivity, vec4(0.0)));
    // The sun's optical depth through the column above changes slowly along
    // the ray: one estimate serves `shadeEvery` neighbouring samples while
    // they are close together (the per-sample 4-step march was the most
    // expensive part of the curtain).
    int shadeEvery = secondary ? 4 : 2;
    int shadeTick = 0;
    float shade = 1.0;
    // Steps grow with distance: near shafts resolved, far ones a veil.
    float len = tEnd - t0;
    float jitter = cloudRand(rng);
    vec3 pin = vec3(0.0);
    float pT = 1.0, dSum = 0.0, dW = 0.0;
    for (int i = 0; i < n; ++i) {
        float u0 = (float(i) + jitter) / float(n);
        float u1 = (float(i) + 1.0 + jitter) / float(n);
        float ta = t0 + len * u0 * u0, tb = t0 + len * min(u1 * u1, 1.0);
        float dt = tb - ta;
        if (dt <= 0.0) continue;
        vec3 p = o + d * ta;
        // Streaks only where the step resolves them (~110 m fine octave):
        // a short segment in front of a nearby slope keeps them at any
        // distance, a long sky ray turns far shafts into their mean veil.
        float sigma = cloudPrecipDensityLod(p, 1.0 - smoothstep(120.0, 600.0, dt));
        if (sigma <= 0.0) continue;
        if (Es.r < 0.0) Es = cloudSunIrradiance(p, sunDir);
        if (shadeTick++ % shadeEvery == 0 || dt > 400.0) {
            float tauSun = cloudOpticalDepth(p, sunDir, CLOUD_RT_MAX_DIST, secondary ? 2 : 4,
                                             jitter, false);
            shade = exp(-tauSun);
        }
        // Under a cloud the sky above is mostly blocked (scaled with the sun
        // shade as a proxy); ~0.125 of it comes back off the ground.
        vec3 S = Es * shade * ph + ambTop * (0.275 + 0.35 * shade);
        if (flashOn) S += cloudFlashMediumLight(p, flashActivity) * (1.0 / (4.0 * CLOUD_PI));
        float Ts = exp(-sigma * dt);
        pin += pT * S * 0.95 * (1.0 - Ts);
        dSum += ta * pT * (1.0 - Ts);
        dW += pT * (1.0 - Ts);
        pT *= Ts;
        if (pT < 0.01) break;
    }
    if (dW <= 1e-4) return;
    float pd = dSum / dW;
    vec3 Tap = cloudAerialTransmittance(o, d, pd);
    pin = pin * Tap + cloudSkyAmbient(d) * (vec3(1.0) - Tap) * (1.0 - pT);
    // From below the shaft is in front of the cloud; from above, behind it.
    if (fromAbove) inscatter += T * pin;
    else           inscatter = pin + pT * inscatter;
#ifdef CLOUD_SEPARATE_FLASH_HISTORY
    if (!fromAbove) cloudFlashInscatter *= pT;
#endif
    T *= pT;
    if (depth < 0.0 || pd < depth) depth = pd;
}

// One path-traced estimate of the clouds along a ray that left the scene
// (REFERENCE mode, quality.rt_reference_path_trace).
//   inscatter = light scattered toward the viewer by the clouds (aerial
//               perspective applied), T = cloud transmittance (multiplies
//               the sky behind), depth = first-collision distance (-1 none).
void cloudRenderLimited(vec3 o, vec3 d, float tLimit, bool cameraRay, uint seed,
                 out vec3 inscatter, out float T, out float depth) {
    if (!cloudRtReference()) {
        // Secondary rays: half the steps (reflections, sky light on the ground).
        int steps = cloudRtSteps();
        cloudMarchLod = cameraRay ? 0 : 1;
        cloudMarchLimited(o, d, tLimit, cameraRay ? steps : max(16, steps / 2),
                          seed, inscatter, T, depth);
        cloudMarchLod = 0;
        return;
    }
    uint rng = cloudSeed(seed, o + d);
    vec4 lightningActivity = cloudLightningActivities();
    inscatter = vec3(0.0);
    depth = -1.0;
    T = cloudTransmittance(o, d, tLimit, rng);

    // Secondary rays: capped walk until the cloud-aware sky panorama (3c)
    // replaces it; quality.rt_secondary_full_march lifts the cap. The cap is
    // a named bias: fewer orders of scattering = darker clouds seen in
    // reflections and in the sky light reaching the ground.
    int maxB = cloudRtMaxBounces();
    if (!cameraRay && !cloudRtSecondaryFull()) maxB = min(maxB, 3);

    vec3 sunDir = normalize(worldData.w.sunDir);
    vec3 pos = o;
    vec3 dir = d;
    vec3 thr = vec3(1.0);
    for (int b = 0; b < maxB; ++b) {
        float t;
        int li;
        if (!cloudFreeFlight(pos, dir, b == 0 ? tLimit : CLOUD_RT_MAX_DIST, rng, t, li)) {
            // Escaped after scattering: the sky it sees lights the cloud.
            if (b > 0 && t >= 0.0) inscatter += thr * cloudSkyAmbient(dir);
            break;
        }
        pos += dir * t;
        if (b == 0) depth = t;
        // Emission is a source per optical-depth event; absorption reduces
        // the scattering throughput before both NEE and continuation.
        inscatter += thr * (cloudParams.layers[li].f.rgb +
                           cloudLightningSource(pos, lightningActivity, true));
        thr *= cloudParams.layers[li].e.rgb;
        if (max(thr.r, max(thr.g, thr.b)) <= 0.0) break;
        float dUm = cloudParams.layers[li].b.y;
        // Sun NEE (single-scattering albedo of water droplets ~1 in the visible).
        vec3 Es = cloudSunIrradiance(pos, sunDir);
        if (max(Es.r, max(Es.g, Es.b)) > 0.0) {
            float Ts = cloudTransmittance(pos, sunDir, CLOUD_RT_MAX_DIST, rng);
            inscatter += thr * Es * Ts * cloudLayerPhase(li, dUm, dot(dir, sunDir));
        }
        float w;
        dir = cloudSamplePhase(li, dUm, dir, rng, w);
        thr *= w;
        if (b >= 3) {
            float q = clamp(max(thr.r, max(thr.g, thr.b)), 0.05, 1.0);
            if (cloudRand(rng) > q) break;
            thr /= q;
        }
    }
    if (depth > 0.0) {
        // Haze between the viewer and the cloud: the transmittance ratio dims
        // the cloud, and the sky's own in-scatter (no sun disk) fills in the
        // part of the view the cloud covers.
        vec3 Tap = cloudAerialTransmittance(o, d, depth);
        inscatter = inscatter * Tap + cloudSkyAmbient(d) * (vec3(1.0) - Tap) * (1.0 - T);
    }
    // Cirrus uses the same bounded thin-sheet model in both renderer modes.
    // It is not part of the liquid-cloud random walk. Previously selecting
    // reference transport omitted this existing authored layer altogether.
    if (cloudParams.cirrus.x >= 0.5 && cloudParams.cirrus.w > 0.0) {
        vec3 horiz = normalize(vec3(d.x, 0.08, d.z) + vec3(1e-4, 0.0, 0.0));
        vec3 ambTop = 0.5 *
            (cloudSkyAmbient(normalize(vec3(0.0, 1.0, 0.0) + d * 0.2)) +
             cloudSkyAmbient(horiz));
        cloudCirrus(o, d, tLimit, sunDir, dot(d, sunDir), ambTop, inscatter, T);
    }
    // The authored curtain remains the same bounded approximation when the
    // liquid-cloud reference transport is selected.
    cloudPrecipMarch(o, d, tLimit, cloudRtSteps(), seed, inscatter, T, depth);
}

void cloudRender(vec3 o, vec3 d, bool cameraRay, uint seed,
                 out vec3 inscatter, out float T, out float depth) {
    cloudRenderLimited(o, d, CLOUD_RT_MAX_DIST, cameraRay, seed, inscatter, T, depth);
}

#endif // CLOUD_RT_LIGHTING

#endif // CLOUD_RT_GLSL
