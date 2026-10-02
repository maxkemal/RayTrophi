// Cloud field: density, phase and parameter block (docs/dev/ATMOSPHERE_CLOUDS.md).
// ONE definition shared by every consumer -- the IPC sampler (cloud_sample.comp),
// the RT reference (Faz 3b) and RayFusion (Faz 3c). Parity between the two
// Vulkan devices comes from this file plus Backend/CloudParams.h, not from
// sharing a texture (they cannot: two VkDevices).
//
// Includer defines, before including:
//   CLOUD_SET           descriptor set
//   CLOUD_BINDING_BASE  first of five consecutive bindings:
//                       +0 base noise (sampler3D), +1 detail noise (sampler3D),
//                       +2 curl (sampler2D), +3 weather map (sampler2D),
//                       +4 CloudParams (std430 buffer)
//
// Units: world metres (scene Y up), extinction 1/m. Altitude above sea level
// = distance from the planet centre - planet radius; the scene sits at
// `altitude_offset_m` above sea level, so clouds are spherical shells and the
// horizon is right.

#ifndef CLOUD_COMMON_GLSL
#define CLOUD_COMMON_GLSL

#ifndef CLOUD_SET
#error "cloud_common.glsl: define CLOUD_SET"
#endif
#ifndef CLOUD_BINDING_BASE
#error "cloud_common.glsl: define CLOUD_BINDING_BASE"
#endif

struct CloudLayerG {
    vec4 a;   // base_altitude_m, thickness_m, coverage, type
    vec4 b;   // extinction_per_m, droplet_diameter_um, erosion, enabled
    vec4 c;   // 1/cell_size_m, 1/detail_size_m, weather offset u, v
};

layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 0) uniform sampler3D cloudBaseNoise;
layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 1) uniform sampler3D cloudDetailNoise;
layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 2) uniform sampler2D cloudCurlNoise;
layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 3) uniform sampler2D cloudWeatherMap;
layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 4, std430) readonly buffer CloudParamsBlock {
    CloudLayerG layers[3];
    vec4 weather;   // 1/extent_m, type_variation, evolution, rt_steps
    vec4 wind;      // drift_x_m, drift_z_m, planet_radius_m, altitude_offset_m
    vec4 cirrus;    // enabled, altitude_m, coverage, optical depth
    vec4 misc;      // 1/cirrus_scale_m, seed, rt_max_bounces, rt_flags (cloud_rt.glsl)
    vec4 precip;    // rate_mm_h, extinction_per_m at that rate, slant_x, slant_z
    vec4 precip2;   // snow (0/1), ground_fraction, shear_x, shear_z (m per m of height)
} cloudParams;

const float CLOUD_PI = 3.14159265358979;

float cloudRemap(float v, float lo, float hi, float nlo, float nhi) {
    return nlo + (v - lo) * (nhi - nlo) / max(hi - lo, 1e-5);
}

vec3 cloudPlanetCenter() {
    return vec3(0.0, -(cloudParams.wind.z + cloudParams.wind.w), 0.0);
}

float cloudAltitude(vec3 p) {
    return length(p - cloudPlanetCenter()) - cloudParams.wind.z;
}

// Vertical density profile for a cloud type at a height fraction (0 base,
// 1 top of the cloud -- not of the shell, see topFrac). Four genera, blended:
//   0.00 stratus        thin sheet, soft top
//   0.33 stratocumulus  low lumpy cells, flattened tops
//   0.66 cumulus        sharp flat base, rounded dome
//   1.00 cumulonimbus   fills the column; the anvil is a coverage term
float cloudProfile(float type, float h) {
    float st  = smoothstep(0.0, 0.05, h) * (1.0 - smoothstep(0.18, 0.40, h));
    float sc  = smoothstep(0.0, 0.08, h) * (1.0 - smoothstep(0.35, 0.60, h));
    float cu  = smoothstep(0.0, 0.05, h) * (1.0 - smoothstep(0.75, 1.00, h));
    float cb  = smoothstep(0.0, 0.04, h) * (1.0 - smoothstep(0.88, 1.00, h));
    float t = clamp(type, 0.0, 1.0) * 3.0;
    if (t < 1.0) return mix(st, sc, t);
    if (t < 2.0) return mix(sc, cu, t - 1.0);
    return mix(cu, cb, t - 2.0);
}

// Wind shear (Faz 3e): a parcel higher in the cloud has drifted further
// downwind, so towers lean and anvils spread one way. Applied to every
// lookup of a layer (weather, noise, majorant) as a height-dependent shift.
// Quadratic in height (the base is anchored, crowns lean) and modulated by a
// low-frequency field (strength 0.4..1.6, direction +-25 deg): a uniform
// linear lean of every cell read as a sheared texture, not as weather.
vec3 cloudShear(CloudLayerG L, vec3 p, float alt) {
    vec2 sh = cloudParams.precip2.zw;
    if (dot(sh, sh) <= 0.0) return p;
    float above = max(alt - L.a.x, 0.0);
    float thick = max(L.a.y, 1.0);
    vec2 v = textureLod(cloudCurlNoise, (p.xz - cloudParams.wind.xy) * cloudParams.weather.x * 6.0, 0.0).rg;
    float k = 0.4 + 1.2 * v.x;
    float a = (v.y - 0.5) * 0.9;
    vec2 rsh = vec2(sh.x * cos(a) - sh.y * sin(a), sh.x * sin(a) + sh.y * cos(a)) * k;
    float disp = above * min(above / thick, 1.0) * 1.6;
    return p - vec3(rsh.x, 0.0, rsh.y) * disp;
}

vec4 cloudWeatherAt(CloudLayerG L, vec3 p) {
    vec2 uv = (p.xz - cloudParams.wind.xy) * cloudParams.weather.x + L.c.zw;
    return textureLod(cloudWeatherMap, uv, 0.0);
}

// Extinction (1/m) of one layer at world point p. `alt` = cloudAltitude(p),
// passed in so a caller summing layers computes it once. `detail` = false
// gives the cheap base shape (majorant, light-march, empty-space skip).
// Bound kept for the RT majorant: result <= local coverage * extinction.
float cloudLayerDensity(int li, vec3 p, float alt, bool detail) {
    CloudLayerG L = cloudParams.layers[li];
    if (L.b.w < 0.5 || L.b.x <= 0.0) return 0.0;
    float h = (alt - L.a.x) / max(L.a.y, 1.0);
    if (h <= 0.0 || h >= 1.0) return 0.0;
    p = cloudShear(L, p, alt);

    vec4 w = cloudWeatherAt(L, p);
    // Per-cell development (weather A): shallow cells sit a little higher
    // (no perfectly flat base across the field) and stop lower.
    float dev = w.a;
    h = (h - (1.0 - dev) * 0.04) / (1.0 - (1.0 - dev) * 0.04);
    if (h <= 0.0) return 0.0;
    // Local coverage: the weather channel is ~uniform (CDF-flattened in the
    // generator), so thresholding it at 1 - coverage covers ~coverage of sky.
    float cov = clamp(cloudRemap(w.r, 1.0 - L.a.z, 1.0, 0.0, 1.0), 0.0, 1.0);
    if (cov <= 0.0) return 0.0;
    float type = clamp(L.a.w + (w.g - 0.5) * cloudParams.weather.y, 0.0, 1.0);
    // Cloud HEIGHT follows local coverage (Nubis): a cluster's core towers,
    // its edges stay low -> domes of heaped cells, not a flat-topped slab.
    // Stratiform genera keep their sheet.
    // Height factors used to MULTIPLY (sqrt(cov) x development x profile x
    // narrowing): measured, the tallest cell of a 1.8 km cumulus layer reached
    // 46% of it and the median 25% -- flat 2 km-wide pancakes. Each factor now
    // has a floor; cell aspect stays ~1 (fair-weather cumulus are as tall as wide).
    float convective = smoothstep(0.25, 0.5, type);
    float topFrac = mix(1.0, mix(0.45, 1.0, sqrt(cov)) * mix(0.55, 1.0, dev), convective);
    float hh = h / topFrac;
    if (hh >= 1.0) return 0.0;
    float prof = cloudProfile(type, hh);
    if (prof <= 0.0) return 0.0;

    // Coverage with height: convective towers narrow toward the top (rounded
    // crowns); cumulonimbus spreads it again into the anvil.
    float anvil = smoothstep(0.72, 0.92, hh) * clamp((type - 0.8) * 5.0, 0.0, 1.0);
    float covH = cov * mix(1.0, mix(1.0, 0.75, hh * hh), convective);
    covH = mix(covH, min(1.0, cov * 1.5), anvil);
    covH = min(covH, cov * 1.0 + anvil * (1.0 - cov));   // majorant: <= cov unless anvil

    vec3 drift = vec3(cloudParams.wind.x, 0.0, cloudParams.wind.y);
    float evo = cloudParams.weather.z;
    // Convective cells are taller than wide: stretch the noise vertically.
    vec3 q = (p - drift) * L.c.x * vec3(1.0, mix(1.0, 0.6, convective), 1.0) +
             vec3(evo * 0.37, evo * 0.11, evo * 0.53);
    vec4 n = textureLod(cloudBaseNoise, q, 0.0);
    // Two scales of billows: the base cell (Perlin-Worley) modulated by a
    // cell three times larger -- turrets stacked on turrets, the layered
    // look of a real cumulus instead of one uniform lumpiness.
    vec4 nL = textureLod(cloudBaseNoise, q * 0.33 + vec3(0.17, 0.41, 0.73), 0.0);
    float lowFbm = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
    float big = mix(0.55, 1.0, nL.r);
    float shape = clamp(cloudRemap(n.r * big, -(1.0 - lowFbm), 1.0, 0.0, 1.0) * prof, 0.0, 1.0);
    // Denser, crisper cores up high; softer bases.
    shape *= mix(0.8, 1.0, smoothstep(0.0, 0.3, hh));
    // Shape in 0..1 (core ~1, edge ~0) BEFORE the coverage scale: erosion
    // must see it. Eroding the coverage-scaled value (<= cov, often ~0.4)
    // put every interior near the threshold, so detail ate each cell
    // everywhere instead of only the outer edge of the cloud.
    float shapeN = clamp(cloudRemap(shape, 1.0 - covH, 1.0, 0.0, 1.0), 0.0, 1.0);
    if (shapeN <= 0.0) return 0.0;
    float covScale = min(covH, cov);
    if (!detail) return shapeN * covScale * L.b.x;

    // Curl-distorted detail erosion: wispy, torn at the base (curl strongest
    // there), billowy cauliflower at the top (detail inverted with height).
    vec2 curl = textureLod(cloudCurlNoise, (p.xz - drift.xz) * L.c.x * 0.5, 0.0).rg * 2.0 - 1.0;
    vec3 dq = (p - drift) * L.c.y + vec3(curl.x, 0.0, curl.y) * (1.0 - hh) * 0.6 + vec3(evo * 1.3);
    vec4 d = textureLod(cloudDetailNoise, dq, 0.0);
    float dfbm = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
    float dmod = mix(dfbm, 1.0 - dfbm, clamp(hh * 4.0, 0.0, 1.0));
    float dens = clamp(cloudRemap(shapeN, dmod * L.b.z * 0.45, 1.0, 0.0, 1.0), 0.0, 1.0);
    return dens * covScale * L.b.x;
}

// ── Precipitation (docs/dev/ATMOSPHERE_WEATHER.md §3.1) ─────────────────────
// Local rate (mm/h) at p: layer 0's weather-map B channel (cluster cores)
// read where the drops LEFT the base -- the wind carries them sideways while
// they fall, so shafts lean downwind. Only under cloud (local coverage > 0);
// above the base nothing, and with ground_fraction < 1 the shaft thins out
// and ends before the ground (virga).
float cloudPrecipAt(vec3 p) {
    float rate = cloudParams.precip.x;
    CloudLayerG L = cloudParams.layers[0];
    if (rate <= 0.0 || L.b.w < 0.5) return 0.0;
    float below = L.a.x - cloudAltitude(p);
    if (below <= 0.0) return 0.0;
    float column = max(L.a.x - cloudParams.wind.w, 1.0);   // base above the scene ground
    float f = below / column;
    float gf = cloudParams.precip2.y;
    float fade = gf >= 0.999 ? 1.0 : 1.0 - smoothstep(gf * 0.5, gf, f);
    if (fade <= 0.0) return 0.0;
    vec3 src = p - vec3(cloudParams.precip.z, 0.0, cloudParams.precip.w) * below;
    vec4 w = cloudWeatherAt(L, src);
    float cov = clamp(cloudRemap(w.r, 1.0 - L.a.z, 1.0, 0.0, 1.0), 0.0, 1.0);
    // Soft top: the shaft grows out of the base over its first ~150 m.
    float top = smoothstep(0.0, 150.0, below);
    return rate * w.b * smoothstep(0.0, 0.25, cov) * fade * top;
}

// Extinction (1/m) of the falling drops at p, with vertical streaks (the
// base noise stretched along the fall) so a shaft is not a smooth slab.
float cloudPrecipDensity(vec3 p) {
    float r = cloudPrecipAt(p);
    if (r <= 0.0) return 0.0;
    CloudLayerG L = cloudParams.layers[0];
    vec3 drift = vec3(cloudParams.wind.x, 0.0, cloudParams.wind.y);
    vec3 q = (p - drift) * L.c.x * vec3(1.5, 0.08, 1.5);
    float streak = mix(0.55, 1.0, textureLod(cloudBaseNoise, q, 0.0).g);
    float exponent = cloudParams.precip2.x > 0.5 ? 0.7 : 0.63;
    return cloudParams.precip.y * pow(r / cloudParams.precip.x, exponent) * streak;
}

float cloudDensity(vec3 p, bool detail) {
    float alt = cloudAltitude(p);
    float s = 0.0;
    for (int i = 0; i < 3; ++i) s += cloudLayerDensity(i, p, alt, detail);
    return s;
}

// ── Phase: Jendersie & d'Eon 2023, HG + Draine blend from droplet diameter ──
float cloudPhaseHG(float g, float mu) {
    float d = 1.0 + g * g - 2.0 * g * mu;
    return (1.0 - g * g) / (4.0 * CLOUD_PI * pow(max(d, 1e-5), 1.5));
}

float cloudPhaseDraine(float g, float alpha, float mu) {
    float d = 1.0 + g * g - 2.0 * g * mu;
    return ((1.0 - g * g) * (1.0 + alpha * mu * mu)) /
           (4.0 * CLOUD_PI * (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0) * pow(max(d, 1e-5), 1.5));
}

struct CloudPhaseFit { float gHG; float gD; float aD; float wD; };

// Fit parameters from the droplet diameter (valid 5-50 um). Shared by the
// evaluation below and the importance sampler in cloud_rt.glsl.
CloudPhaseFit cloudPhaseFit(float diameterUm) {
    float d = clamp(diameterUm, 5.0, 50.0);
    CloudPhaseFit f;
    f.gHG = exp(-0.0990567 / (d - 1.67154));
    f.gD  = exp(-2.20679 / (d + 3.91029) - 0.428934);
    f.aD  = exp(3.62489 - 8.29288 / (d + 5.52825));
    f.wD  = exp(-0.599085 / (d - 0.641583) - 0.665888);
    return f;
}

// mu = cos(angle between the propagation direction and the scattered one).
float cloudPhase(float diameterUm, float mu) {
    CloudPhaseFit f = cloudPhaseFit(diameterUm);
    return (1.0 - f.wD) * cloudPhaseHG(f.gHG, mu) + f.wD * cloudPhaseDraine(f.gD, f.aD, mu);
}

#endif // CLOUD_COMMON_GLSL
