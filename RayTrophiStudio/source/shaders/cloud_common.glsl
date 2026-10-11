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
    vec4 d;   // inverse yaw cos, sin; formation role, group
    vec4 e;   // effective RGB albedo, phase override (2 = droplet fit)
    vec4 f;   // effective emission RGB, reserved
    vec4 g;   // group base, height, inverse footprint period, reserved
};

struct CloudLightningG {
    vec4 start;
    vec4 end;
    vec4 color;
    vec4 stage;
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
    vec4 precip;    // rate_mm_h, extinction_per_m at that rate, trail_x, trail_z (upwind lag per m of fall at the ground)
    vec4 precip2;   // snow (0/1), ground_fraction, shear_x, shear_z (m per m of height)
    vec4 scatterAlbedo; // RGB albedo, HG g override (2 = droplet fit)
    vec4 emissionSource; // RGB source radiance, reserved
    vec4 cirrusDirection; // along-wind x,z; camera weather time, reserved
    CloudLightningG lightning[4];
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

bool cloudHasLiquidLayers() {
    if ((int(cloudParams.misc.w + 0.5) & 1) == 0) return false;
    for (int i = 0; i < 3; ++i) {
        if (cloudParams.layers[i].b.w >= 0.5 && cloudParams.layers[i].b.x > 0.0 &&
            cloudParams.layers[i].a.z > 0.0) return true;
    }
    return false;
}

// The norm is convex along a straight segment, so a segment whose endpoints
// are both below every cloud shell cannot enter one. Rain still needs its
// below-cloud integration. This avoids marching ordinary ground fragments.
bool cloudSegmentMayContain(vec3 origin, vec3 direction, float distance) {
    // Rain/snow extinction is bounded by precip.y x the streak peak (1.85):
    // a segment too short to reach an optical depth of 1e-3 cannot show it
    // (indoor bounces, contact segments) and skips the whole march.
    if (cloudParams.precip.x > 0.0 && distance * cloudParams.precip.y * 2.0 > 1e-3) return true;
    float lowest = 1e30;
    for (int i = 0; i < 3; ++i) {
        CloudLayerG L = cloudParams.layers[i];
        if (L.b.w >= 0.5 && L.b.x > 0.0 && L.a.z > 0.0) {
            lowest = min(lowest, L.a.x);
        }
    }
    if (cloudParams.cirrus.x >= 0.5 && cloudParams.cirrus.w > 0.0) {
        lowest = min(lowest, cloudParams.cirrus.y);
    }
    return max(cloudAltitude(origin), cloudAltitude(origin + direction * distance)) >= lowest;
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

// Authored yaw rotates weather and its majorant identically. Additional
// intrinsic layer orientations/phases affect only shape, with no extra reads.
vec2 cloudRotateXZ(CloudLayerG L, vec2 position) {
    return vec2(L.d.x * position.x - L.d.y * position.y,
                L.d.y * position.x + L.d.x * position.y);
}

vec3 cloudShapeCoordinates(int li, vec3 relativePosition) {
    vec3 q = relativePosition;
    q.xz = cloudRotateXZ(cloudParams.layers[li], q.xz);
    int group = int(cloudParams.layers[li].d.w + 0.5);
    int fieldIndex = group > 0 ? group - 1 : li;
    if (fieldIndex == 1) {
        q.xz = vec2(0.8 * q.x - 0.6 * q.z, 0.6 * q.x + 0.8 * q.z);
    } else if (fieldIndex == 2) {
        q.xz = vec2(0.6 * q.x + 0.8 * q.z, -0.8 * q.x + 0.6 * q.z);
    }
    return q;
}

vec3 cloudShapePhase(int li) {
    // Weather already carries the full seed. A bounded seed phase also
    // changes the small formations without large floating-point offsets.
    float seedPhase = fract(cloudParams.misc.y * 0.00390625);
    return fract(vec3(0.173, 0.419, 0.731) * float(li) +
                 vec3(0.237, 0.613, 0.357) * seedPhase);
}

// Linked storm regions share column centres and vertical development. The
// support is computed from integer hashes, not another texture or a tiled
// density box. Neighbour support overlaps continuously at lattice boundaries.
float cloudStormEnvelope(CloudLayerG L, vec3 localPosition, float altitude, vec3 billow) {
    vec2 q = localPosition.xz * L.g.z;
    ivec2 cell = ivec2(floor(q));
    float h = (altitude - L.g.x) / max(L.g.y, 1.0);
    if (h <= 0.0 || h >= 1.0) return 0.0;
    float envelope = 0.0;
    for (int z = -1; z <= 1; ++z) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 id = cell + ivec2(x, z);
            uint hash = uint(id.x) * 73856093u ^ uint(id.y) * 19349663u ^
                        uint(cloudParams.misc.y) * 83492791u;
            hash ^= hash >> 16u;
            hash *= 0x7feb352du;
            hash ^= hash >> 15u;
            hash *= 0x846ca68bu;
            hash ^= hash >> 16u;
            vec3 r = vec3(float(hash & 65535u), float(hash >> 16u),
                          float((hash * 2654435761u) >> 16u)) / 65535.0;
            vec2 center = vec2(id) + vec2(0.25) + r.xy * 0.5;
            float top = mix(0.68, 1.0, r.z);
            float ch = h / top;
            if (ch <= 0.0 || ch >= 1.0) continue;
            // A union of broad rising lobes, not a cylinder with a pointed cap.
            // Even with edge noise and ellipse stretch, support stays below
            // 1.25 cells, where omitted second neighbours begin.
            for (int lobe = 0; lobe < 4; ++lobe) {
                float k = float(lobe);
                float lobeHeight = lobe == 3 ? 0.77 : 0.14 + k * 0.25;
                float verticalRadius = lobe == 3 ? 0.16 : 0.22 + k * 0.025;
                float verticalDistance = (ch - lobeHeight) / verticalRadius;
                // Billow displacement is bounded by +/-0.26. Outside this
                // expanded support the lobe cannot contribute any density.
                if (abs(verticalDistance) >= 1.26) continue;
                float radius = lobe == 3 ? mix(0.58, 0.72, r.y) :
                                           mix(0.34, 0.48, r.x) + k * 0.025;
                vec2 offset = (r.xy - vec2(0.5)) * k * 0.10;
                vec2 delta = (q - center - offset) / radius;
                delta.x *= mix(0.85, 1.15, r.y);
                vec3 v = vec3(delta.x, verticalDistance, delta.y);
                float distanceSquared = dot(v, v);
                if (distanceSquared >= 1.26 * 1.26) continue;
                // Existing 3D formation noise breaks each lobe into smaller
                // cauliflower bulges without adding a texture lookup.
                float edge = sqrt(distanceSquared) + (billow.x - 0.5) * 0.36 +
                             (billow.y - 0.5) * 0.16;
                float mass = 1.0 - smoothstep(0.68, 1.0, edge);
                envelope = max(envelope, mass * smoothstep(0.0, 0.035, ch));
                // The union is already full: no remaining neighbour or lobe
                // can increase it. Keep solving the rim, base and crown.
                if (envelope >= 1.0) return 1.0;
            }
        }
    }
    return envelope;
}

// Linked regions describe one field. Interpolate authored density/shape
// controls between region midpoints so crossing an altitude boundary does
// not restart the noise or slice a tower into differently sized cylinders.
CloudLayerG cloudStormLayerAt(int li, float altitude) {
    CloudLayerG L = cloudParams.layers[li];
    if (L.d.z < 0.5 || L.d.w < 0.5) return L;
    CloudLayerG lower = L;
    CloudLayerG upper = L;
    float lowerHeight = -1e30;
    float upperHeight = 1e30;
    for (int i = 0; i < 3; ++i) {
        CloudLayerG member = cloudParams.layers[i];
        if (member.b.w < 0.5 || member.d.z < 0.5 || member.d.w != L.d.w) continue;
        float height = member.a.x + member.a.y * 0.5;
        if (height <= altitude && height > lowerHeight) {
            lower = member;
            lowerHeight = height;
        }
        if (height >= altitude && height < upperHeight) {
            upper = member;
            upperHeight = height;
        }
    }
    if (lowerHeight == -1e30) {
        lower = upper;
        lowerHeight = upperHeight;
    }
    if (upperHeight == 1e30) {
        upper = lower;
        upperHeight = lowerHeight;
    }
    float t = clamp((altitude - lowerHeight) / max(upperHeight - lowerHeight, 1.0), 0.0, 1.0);
    t = t * t * (3.0 - 2.0 * t);
    L.a.z = mix(lower.a.z, upper.a.z, t);
    L.b.x = mix(lower.b.x, upper.b.x, t);
    L.b.z = mix(lower.b.z, upper.b.z, t);
    L.c.xy = mix(lower.c.xy, upper.c.xy, t);
    return L;
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
    float base = L.d.z > 0.5 ? L.g.x : L.a.x;
    float above = max(alt - base, 0.0);
    float thick = max(L.d.z > 0.5 ? L.g.y : L.a.y, 1.0);
    vec2 v = textureLod(cloudCurlNoise, (p.xz - cloudParams.wind.xy) * cloudParams.weather.x * 6.0, 0.0).rg;
    float k = 0.4 + 1.2 * v.x;
    float a = (v.y - 0.5) * 0.9;
    vec2 rsh = vec2(sh.x * cos(a) - sh.y * sin(a), sh.x * sin(a) + sh.y * cos(a)) * k;
    float disp = above * min(above / thick, 1.0) * 1.6;
    return p - vec3(rsh.x, 0.0, rsh.y) * disp;
}

vec2 cloudWeatherUV(CloudLayerG L, vec3 p) {
    return cloudRotateXZ(L, p.xz - cloudParams.wind.xy) *
           cloudParams.weather.x + L.c.zw;
}

vec4 cloudWeatherAt(CloudLayerG L, vec3 p) {
    return textureLod(cloudWeatherMap, cloudWeatherUV(L, p), 0.0);
}

// Extinction (1/m) of one layer at world point p. `alt` = cloudAltitude(p),
// passed in so a caller summing layers computes it once. `detail` = false
// gives the cheap base shape (majorant, light-march, empty-space skip).
// Bound kept for the RT majorant: result <= local coverage * extinction.
float cloudLayerDensityResolved(int li, vec3 p, float alt, bool detail,
                               out float baseDensity) {
    baseDensity = 0.0;
    if (cloudParams.layers[li].b.w < 0.5 || cloudParams.layers[li].b.x <= 0.0) return 0.0;
    CloudLayerG L = cloudStormLayerAt(li, alt);
    if (L.b.w < 0.5 || L.b.x <= 0.0) return 0.0;
    bool stormRegion = L.d.z > 0.5;
    float h = (alt - L.a.x) / max(L.a.y, 1.0);
    if (h < 0.0 || h >= 1.0 || (!stormRegion && h <= 0.0)) return 0.0;
    p = cloudShear(L, p, alt);

    vec4 w = cloudWeatherAt(L, p);
    // Per-cell development (weather A): shallow cells sit a little higher
    // (no perfectly flat base across the field) and stop lower.
    float dev = w.a;
    if (!stormRegion) {
        h = (h - (1.0 - dev) * 0.04) / (1.0 - (1.0 - dev) * 0.04);
    }
    if (h <= 0.0 && !stormRegion) return 0.0;
    // Local coverage: the weather channel is ~uniform (CDF-flattened in the
    // generator), so thresholding it at 1 - coverage covers ~coverage of sky.
    float cov = clamp(cloudRemap(w.r, 1.0 - L.a.z, 1.0, 0.0, 1.0), 0.0, 1.0);
    if (cov <= 0.0) return 0.0;
    float type = clamp(L.a.w + (w.g - 0.5) * cloudParams.weather.y, 0.0, 1.0);
    vec3 drift = vec3(cloudParams.wind.x, 0.0, cloudParams.wind.y);
    float evo = cloudParams.weather.z;
    vec3 relativePosition = cloudShapeCoordinates(li, p - drift);
    int group = int(L.d.w + 0.5);
    vec3 phase = cloudShapePhase(group > 0 ? group - 1 : li);
    float stormEnvelope = 1.0;
    float convective = smoothstep(0.25, 0.5, type);
    float towering = smoothstep(0.62, 0.95, type);
    int role = int(L.d.z + 0.5);
    if (role == 1) {
        convective = 0.55;
        towering = 0.0;
    } else if (role == 2) {
        convective = 1.0;
        towering = 1.0;
    } else if (role == 3) {
        convective = 0.20;
        towering = 0.0;
    }
    if (stormRegion) {
        float groupHeight = (alt - L.g.x) / max(L.g.y, 1.0);
        convective = 1.0 - 0.7 * smoothstep(0.72, 0.94, groupHeight);
        towering = 1.0 - smoothstep(0.72, 0.94, groupHeight);
    }
    // Development belongs to a horizontal cell, not to successive slices of
    // the 3D texture. Weather A organizes fields, while higher-frequency
    // channels of the existing formation read vary neighbouring towers.
    vec3 formationQ = vec3(relativePosition.x * L.c.x * 0.271,
                           0.37,
                           relativePosition.z * L.c.x * 0.271) + phase;
    formationQ.xz += vec2(evo * 0.10, evo * 0.14);
    vec4 nL = textureLod(cloudBaseNoise, formationQ, 0.0);
    float cellGrowth = smoothstep(0.18, 0.65, nL.b);
    float development = mix(dev, cellGrowth, 0.65 * convective);
    // Cloud HEIGHT follows local coverage (Nubis): a cluster's core towers,
    // its edges stay low -> domes of heaped cells, not a flat-topped slab.
    // Stratiform genera keep their sheet.
    // Height factors used to MULTIPLY (sqrt(cov) x development x profile x
    // narrowing): measured, the tallest cell of a 1.8 km cumulus layer reached
    // 46% of it and the median 25% -- flat 2 km-wide pancakes. Each factor now
    // has a floor; cell aspect stays ~1 (fair-weather cumulus are as tall as wide).
    float topFrac = mix(1.0, mix(0.45, 1.0, sqrt(cov)) *
                        mix(0.55, 1.0, development), convective);
    if (stormRegion) topFrac = 1.0;
    float hh = h / topFrac;
    if (stormRegion) hh = (alt - L.g.x) / max(L.g.y, 1.0);
    if (hh >= 1.0) return 0.0;
    float prof = cloudProfile(type, hh);
    if (stormRegion) prof = 1.0;
    if (prof <= 0.0) return 0.0;

    // Coverage with height: convective towers narrow toward the top (rounded
    // crowns); cumulonimbus spreads it again into the anvil.
    float anvil = smoothstep(0.72, 0.92, hh) * clamp((type - 0.8) * 5.0, 0.0, 1.0);
    float covH = cov * mix(1.0, mix(1.0, 0.75, hh * hh), convective);
    covH = mix(covH, min(1.0, cov * 1.5), anvil);
    covH = min(covH, cov * 1.0 + anvil * (1.0 - cov));   // majorant: <= cov unless anvil
    if (stormRegion) covH = cov;

    // Each tower has its own vertical aspect. Anchor the shape to the layer
    // base and lean the crown locally, leaving the base intact.
    float verticalScale = mix(1.0, mix(0.42, 0.85, cellGrowth), convective);
    verticalScale = mix(verticalScale, mix(0.8, 1.4, cellGrowth), towering);
    if (stormRegion) verticalScale = 0.75;
    float shapeBase = stormRegion ? L.g.x : L.a.x;
    vec3 q = vec3(relativePosition.x, alt - shapeBase, relativePosition.z) *
             L.c.x * vec3(1.0, verticalScale, 1.0) + phase +
             vec3(evo * 0.37, evo * 0.11, evo * 0.53);
    q.xz += (nL.ga - vec2(0.5)) * hh * hh * convective * 0.8;
    // A continuous oblique component separates billow rows across a field.
    q.y += dot(q.xz - phase.xz, vec2(0.19, -0.13)) * convective;
    // Reuse the large-formation channels to bend the small billows. This
    // breaks their regular alignment without another noise octave. Warping
    // affects only the shape, never the weather coverage bound.
    vec3 formationWarp = (nL.gba - vec3(0.5)) *
                         mix(0.12, 0.42, convective);
    vec4 n = textureLod(cloudBaseNoise, q + formationWarp, 0.0);
    if (stormRegion) {
        stormEnvelope = cloudStormEnvelope(L, relativePosition, alt, n.gba);
        if (stormEnvelope <= 0.0) return 0.0;
    }
    float lowFbm = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
    // Sheets retain continuous cover; convective fields have stronger
    // formation-scale separation and crowns assembled from smaller cells.
    float big = mix(mix(0.82, 1.0, nL.r), mix(0.55, 1.0, nL.r), convective);
    float shape = clamp(cloudRemap(n.r * big, -(1.0 - lowFbm), 1.0, 0.0, 1.0) * prof, 0.0, 1.0);
    // Denser, crisper cores up high; softer bases.
    shape *= mix(0.8, 1.0, smoothstep(0.0, 0.3, hh));
    // Shape in 0..1 (core ~1, edge ~0) BEFORE the coverage scale: erosion
    // must see it. Eroding the coverage-scaled value (<= cov, often ~0.4)
    // put every interior near the threshold, so detail ate each cell
    // everywhere instead of only the outer edge of the cloud.
    float shapeN = clamp(cloudRemap(shape, 1.0 - covH, 1.0, 0.0, 1.0), 0.0, 1.0);
    if (stormRegion) {
        shapeN = clamp(cloudRemap(shape, 0.25, 0.95, 0.0, 1.0), 0.0, 1.0) * stormEnvelope;
    }
    // Mid-scale cuts on tall sidewalls bridge the gap between kilometre-wide
    // bodies and fine detail. Reuse the base texture's higher-frequency
    // channels; weather coverage and the tracking bound remain unchanged.
    float flank = smoothstep(0.06, 0.22, hh) *
                  (1.0 - smoothstep(0.45, 0.85, shapeN));
    float midCut = (1.0 - n.b) * L.b.z * towering * flank * 0.22;
    shapeN = clamp(cloudRemap(shapeN, midCut, 1.0, 0.0, 1.0), 0.0, 1.0);
    if (shapeN <= 0.0) return 0.0;
    float covScale = min(covH, cov);
    baseDensity = shapeN * covScale * L.b.x;
    if (!detail) return baseDensity;

    // Fine erosion belongs to the boundary, not to an already dense core.
    // Retain detail at the base and crown even when the horizontal core is
    // full. Fade continuously so crossing the shell adds no density seam.
    float boundaryDetail = 1.0 - smoothstep(0.75, 0.95, shapeN);
    float verticalDetail = max(1.0 - smoothstep(0.04, 0.16, hh),
                               smoothstep(0.78, 0.94, hh));
    float detailWeight = stormRegion ? max(boundaryDetail, verticalDetail) : 1.0;
    if (detailWeight <= 0.0) return shapeN * covScale * L.b.x;

    // Curl-distorted erosion: protect the flat convective base, expose
    // cauliflower crowns and keep stratiform edges softer. The cheap shape
    // remains an upper bound: detail only removes density.
    vec2 curl = textureLod(cloudCurlNoise, relativePosition.xz * L.c.x * 0.5, 0.0).rg * 2.0 - 1.0;
    vec3 curlOffset = vec3(curl.x, 0.0, curl.y);
    float baseProtection = mix(1.0, smoothstep(0.03, 0.22, hh), convective);
    vec3 dq = relativePosition * L.c.y + phase +
              curlOffset * (1.0 - hh) * 0.6 * baseProtection +
              formationWarp * 0.35 + vec3(evo * 1.3);
    vec4 d = textureLod(cloudDetailNoise, dq, 0.0);
    float dfbm = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
    float crown = smoothstep(0.15, 0.65, hh) * convective;
    float dmod = mix(dfbm, 1.0 - dfbm, crown);
    float erosionHeight = mix(0.65, mix(0.2, 1.0, baseProtection), convective);
    float erosionThreshold = dmod * L.b.z * mix(0.45, 0.72, towering) *
                             erosionHeight * detailWeight;
    float dens = clamp(cloudRemap(shapeN, erosionThreshold, 1.0, 0.0, 1.0),
                       0.0, 1.0);
    return dens * covScale * L.b.x;
}

float cloudLayerDensity(int li, vec3 p, float alt, bool detail) {
    float baseDensity;
    return cloudLayerDensityResolved(li, p, alt, detail, baseDensity);
}

// ── Precipitation (docs/dev/ATMOSPHERE_WEATHER.md §3.1) ─────────────────────
// Local rate (mm/h) at p, traced back up the fall line to where the drops
// left the base, and `fallLine` = that release point (cloud frame comes from
// the weather lookup, which already subtracts the drift).
//
// Footprint: the base shape of layer 0 just above its base at the release
// point -- rain hangs under the cells that are actually drawn, not under the
// ~12 km weather blobs -- times the weather B channel (cluster cores: which
// cells rain at all).
//
// Lean: the weather map drifts WITH the wind, and so do drops released at the
// base; uniform wind therefore gives a VERTICAL shaft in the cloud frame.
// Only the slower air below the base makes the drops lag behind their cloud
// (precip.zw = lag per metre of fall at the ground, linear wind deficit ->
// quadratic, trailing foot). The old absolute wind/fall-speed slant tilted a
// 34 m/s shaft ~80 degrees and was the main source of "rays from everywhere".
//
// Tassels ("salkim"): each drop bundle has its own release depth and, as
// virga, its own evaporation depth -- one column noise along the fall line
// staggers both, so the base and a virga's foot are fringed, not cut flat.
//
// The density continues down to whatever surface the ray meets (terrain in
// a valley below y = 0 still gets rain). The scene-ground floor for rays that
// meet NOTHING is cloudPrecipMarch's job: a camera under y = 0 used to look
// down 40 km of uninterrupted rain.
float cloudPrecipSample(vec3 p, out vec3 fallLine) {
    fallLine = p;
    float rate = cloudParams.precip.x;
    CloudLayerG L = cloudParams.layers[0];
    if (rate <= 0.0 || L.b.w < 0.5) return 0.0;
    float below = L.a.x - cloudAltitude(p);
    if (below <= 0.0) return 0.0;
    float column = max(L.a.x - cloudParams.wind.w, 1.0);   // base above the scene ground
    float f = below / column;
    float gf = cloudParams.precip2.y;
    // Cheapest reject first: deeper than any tassel of a virga can reach.
    if (gf < 0.999 && f >= gf * 1.3) return 0.0;
    // precip.zw points upwind (where the foot trails), so the release point
    // of the drop at p is back downwind, under the cloud that has moved on.
    float fl = min(f, 1.0);
    vec3 src = p - vec3(cloudParams.precip.z, 0.0, cloudParams.precip.w) * (below * fl * 0.5);
    // Drops fan out on the way down (turbulence, size sorting): wobble the
    // lookup by a cloud-frame curl field, growing with the fall distance, so
    // a shaft widens and frays toward the ground instead of a straight prism.
    vec2 wob = textureLod(cloudCurlNoise, (src.xz - cloudParams.wind.xy) * (1.0 / 2600.0), 0.0).rg - 0.5;
    src.xz += wob * min(below, column) * 0.35;
    fallLine = src;
    vec2 s = src.xz - cloudParams.wind.xy;
    float tassel = textureLod(cloudBaseNoise, vec3(s.x / 260.0, 0.13, s.y / 260.0), 0.0).a;
    float fade = 1.0;
    if (gf < 0.999) {
        float reach = gf * mix(0.55, 1.3, tassel);
        fade = 1.0 - smoothstep(reach * 0.6, reach, f);
        if (fade <= 0.0) return 0.0;
    }
    vec4 w = cloudWeatherAt(L, src);
    if (w.b <= 0.0) return 0.0;
    float baseShape;
    cloudLayerDensityResolved(0, src, L.a.x + min(0.1 * L.a.y, 300.0), false, baseShape);
    float footprint = smoothstep(0.06, 0.45, baseShape / max(L.b.x, 1e-6));
    if (footprint <= 0.0) return 0.0;
    // Fringed top: bundles leave the base over 60..450 m, staggered per column.
    float top = smoothstep(0.0, mix(60.0, 450.0, tassel), below);
    return rate * w.b * footprint * fade * top;
}

float cloudPrecipAt(vec3 p) {
    vec3 fallLine;
    return cloudPrecipSample(p, fallLine);
}

// Extinction (1/m) of the falling drops at p. Fall streaks: drop bundles
// tens to a few hundred metres across and kilometres long, read along the
// fall line so they bend with the trailing foot. Mean ~1, so the Marshall-
// Palmer optical depth over a long path is kept. `detail` 0..1 fades the
// streaks to their mean (far shafts: a veil, not sub-pixel aliasing).
float cloudPrecipDensityLod(vec3 p, float detail) {
    vec3 fallLine;
    float r = cloudPrecipSample(p, fallLine);
    if (r <= 0.0) return 0.0;
    float streak = 1.0;
    if (detail > 0.0) {
        vec2 s = fallLine.xz - cloudParams.wind.xy;
        float alt = cloudAltitude(p);
        float n1 = textureLod(cloudBaseNoise, vec3(s.x / 420.0, alt / 5200.0, s.y / 420.0), 0.0).g;
        float n2 = textureLod(cloudBaseNoise, vec3(s.x / 110.0 + 0.37, alt / 1900.0, s.y / 110.0 + 0.61), 0.0).b;
        float fine = mix(0.15, 1.85, smoothstep(0.25, 0.75, n1 * 0.65 + n2 * 0.35));
        streak = mix(1.0, fine, detail);
    }
    float exponent = cloudParams.precip2.x > 0.5 ? 0.7 : 0.63;
    return cloudParams.precip.y * pow(r / cloudParams.precip.x, exponent) * streak;
}

float cloudPrecipDensity(vec3 p) {
    return cloudPrecipDensityLod(p, 1.0);
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
float cloudPhaseWithOverride(float diameterUm, float mu, float phaseOverride) {
    if (phaseOverride <= 0.99) {
        return cloudPhaseHG(phaseOverride, mu);
    }
    CloudPhaseFit f = cloudPhaseFit(diameterUm);
    return (1.0 - f.wD) * cloudPhaseHG(f.gHG, mu) + f.wD * cloudPhaseDraine(f.gD, f.aD, mu);
}

float cloudPhase(float diameterUm, float mu) {
    return cloudPhaseWithOverride(diameterUm, mu, cloudParams.scatterAlbedo.w);
}

float cloudLayerPhase(int li, float diameterUm, float mu) {
    return cloudPhaseWithOverride(diameterUm, mu, cloudParams.layers[li].e.w);
}

#endif // CLOUD_COMMON_GLSL
