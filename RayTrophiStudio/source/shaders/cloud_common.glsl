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
//   CLOUD_CELL_BINDING  storm cell map (sampler2D, cloud_cells.comp;
//                       docs/dev/ATMOSPHERE_STORMS.md): three 256^2 fields
//                       stacked -- cells, cell structure, storm column
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
#ifndef CLOUD_CELL_BINDING
#error "cloud_common.glsl: define CLOUD_CELL_BINDING"
#endif
#ifndef CLOUD_VOLUME_BINDING
#error "cloud_common.glsl: define CLOUD_VOLUME_BINDING"
#endif

struct CloudLayerG {
    vec4 a;   // base_altitude_m, thickness_m, coverage, type
    vec4 b;   // extinction_per_m, droplet_diameter_um, erosion, enabled
    vec4 c;   // 1/cell_size_m, 1/detail_size_m, weather offset u, v
    vec4 d;   // inverse yaw cos, sin; formation role, group
    vec4 e;   // effective RGB albedo, phase override (2 = droplet fit)
    vec4 f;   // effective emission RGB, reserved
    vec4 g;   // group base, height, inverse footprint period, base raggedness
};

// One storm cell drawn as a volume (Backend::CloudVolumeSlotGPU).
struct CloudVolumeSlotG {
    vec4 a;   // box min x, box bottom altitude, box min z, active
    vec4 b;   // box size x, height, size z, atlas slot index
    vec4 c;   // cell id, tower, radius m, life
    vec4 d;   // wall allowance m, tropopause fraction, coverage, peak
    vec4 e;   // age s, wind direction x, z, anvil spread
    vec4 f;   // turret detail, rotation, shear x, z
};

#include "cloud_lightning_light.glsl"   // CloudLightningG, CloudFlashLightG

layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 0) uniform sampler3D cloudBaseNoise;
layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 1) uniform sampler3D cloudDetailNoise;
layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 2) uniform sampler2D cloudCurlNoise;
layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 3) uniform sampler2D cloudWeatherMap;
layout(set = CLOUD_SET, binding = CLOUD_BINDING_BASE + 4, std430) readonly buffer CloudParamsBlock {
    // 0..2 the authored layers; 3 the storm column shell (f.w = 1 when on,
    // ATMOSPHERE_CELL_CLOUDS.md): troposphere-deep, dense only in cells;
    // 4 the local fog (ATMOSPHERE_LOCAL_WINDS_FOG.md §4, cloudFogDensity).
    CloudLayerG layers[5];
    vec4 weather;   // 1/extent_m, type_variation, evolution, rt_steps
    vec4 wind;      // drift_x_m, drift_z_m, planet_radius_m, altitude_offset_m
    vec4 cirrus;    // enabled, altitude_m, coverage, optical depth
    vec4 misc;      // 1/cirrus_scale_m, seed, rt_max_bounces, rt_flags (cloud_rt.glsl)
    vec4 precip;    // rate_mm_h, extinction bound at that rate (any form), trail_x, trail_z
                    // (upwind lag per m of fall at the ground for a 1 m/s faller)
    vec4 precip2;   // reserved, ground_fraction, shear_x, shear_z (m per m of height)
    vec4 scatterAlbedo; // RGB albedo, HG g override (2 = droplet fit)
    vec4 emissionSource; // RGB source radiance, reserved
    vec4 cirrusDirection; // along-wind x,z; camera weather time, reserved
    vec4 storms;    // cell map centre x, z, 1/size, enabled
    vec4 storms2;   // stratiform rain share, tropopause fraction, wall allowance m, mammatus
    vec4 precip3;   // sea-level air temperature C, lapse K/m, forced form (0 auto), reserved
    vec4 fog;       // local fog map corner x, z, 1/size, active (map = cell-map sheet 3)
    vec4 volume;    // storm cell volumes: active, voxels per axis, skin m, requested slots
    CloudVolumeSlotG volumeSlots[16];
    CloudLightningG lightning[4];
    CloudFlashLightG flash[4];   // lit cloud base per event (CPU-built)
    CloudBoltSegG bolt[320];     // channel network, 80 per event (CPU-built)
} cloudParams;

layout(set = CLOUD_SET, binding = CLOUD_CELL_BINDING) uniform sampler2D cloudCellMap;
// Storm cell volume atlas (cloud_volume_gen.comp; ATMOSPHERE_CLOUD_VOLUMES.md):
// 128^3 slots stacked along z. R = signed distance to the cloud surface (m,
// negative inside). A 1^3 placeholder while volumes are off.
layout(set = CLOUD_SET, binding = CLOUD_VOLUME_BINDING) uniform sampler3D cloudVolumeAtlas;

const float CLOUD_PI = 3.14159265358979;

// The cell map stacks square fields (cloud_cells.comp); `sheet` 0 = cells,
// 1 = structure, 2 = storm column. v is clamped half a texel inside its
// sheet so the linear filter never blends two of them.
vec4 cloudStormMapAt(vec3 p, float sheet) {
    if (cloudParams.storms.w < 0.5) return vec4(0.0);
    vec2 uv = (p.xz - cloudParams.storms.xy) * cloudParams.storms.z + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec4(0.0);
    ivec2 size = textureSize(cloudCellMap, 0);
    float sheets = float(size.y / size.x);
    float texel = 1.0 / float(size.x);
    uv.y = (sheet + clamp(uv.y, 0.5 * texel, 1.0 - 0.5 * texel)) / sheets;
    return textureLod(cloudCellMap, uv, 0.0);
}

// Storm column fields at p: R anvil cover, G wall cloud (lowered base under
// the main updraft), B reserved (gust front, Faz 5b), A dilated anvil majorant.
vec4 cloudStormColumnAt(vec3 p) {
    return cloudStormMapAt(p, 2.0);
}

bool cloudShellActive() {
    return cloudParams.layers[3].b.w > 0.5 && cloudParams.layers[3].f.w > 0.5;
}

// Storm cells at p (world): R cover floor, G tower (genus floor), B convective
// rain potential, A dilated cover majorant. Zero outside the map or with the
// storm system off. One fetch; the cell loop lives in cloud_cells.comp.
vec4 cloudStormCellAt(vec3 p) {
    return cloudStormMapAt(p, 0.0);
}

// Cell structure at p: RG swirl displacement of the shape lookup (m), B local
// top as a share of the layer, A updraft (turret core). ATMOSPHERE_STORMS §3.1.
// B is stored squared (cloud_cells.comp): filtered as a quadratic, then sqrt.
vec4 cloudStormFlowAt(vec3 p) {
    vec4 f = cloudStormMapAt(p, 1.0);
    f.z = sqrt(max(f.z, 0.0));
    return f;
}

float cloudRemap(float v, float lo, float hi, float nlo, float nhi) {
    return nlo + (v - lo) * (nhi - nlo) / max(hi - lo, 1e-5);
}

// ── Local fog (ATMOSPHERE_LOCAL_WINDS_FOG.md §4) ───────────────────────────
// Layer slot 4. The CPU map (Atmosphere/AtmosphereLocalWeather.h) is the
// ENVELOPE; the shape is the cloud noise's. Scene y throughout: the fog lies
// on the scene's flat ground and terrain, not on the planet sphere.
bool cloudFogActive() {
    return cloudParams.fog.w > 0.5 && cloudParams.layers[4].b.w > 0.5;
}

// Envelope at p (cell-map sheet 3): R fog 1/m, G fog top y, B mist 1/m at
// the ground, A ground y. Zero outside the map.
vec4 cloudFogMapAt(vec3 p) {
    vec2 uv = (p.xz - cloudParams.fog.xy) * cloudParams.fog.z;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec4(0.0);
    ivec2 size = textureSize(cloudCellMap, 0);
    float sheets = float(size.y / size.x);
    float texel = 1.0 / float(size.x);
    uv.y = (3.0 + clamp(uv.y, 0.5 * texel, 1.0 - 0.5 * texel)) / sheets;
    return textureLod(cloudCellMap, uv, 0.0);
}

// Fog extinction (1/m) at p. <= the envelope at every point, so the map's
// maximum (layers[4].b.x) is a majorant.
//   radiation fog: fills the hollow up to a lid that undulates +-35% of its
//                  depth; the body breaks into banks.
//   mist:          thins upward from the ground over 250 m (scale height).
// Constants mirror localFogExtinctionAt (kLocalFogTopSoftM 25 m,
// kLocalMistScaleHeightM 250 m) -- the CPU reports the envelope, this draws it.
float cloudFogDensity(vec3 p, bool detail, out float baseDensity) {
    baseDensity = 0.0;
    if (!cloudFogActive()) return 0.0;
    CloudLayerG L = cloudParams.layers[4];
    if (p.y < L.g.x || p.y > L.g.y) return 0.0;
    vec4 m = cloudFogMapAt(p);
    if (m.x + m.z <= 0.0 || p.y < m.w - 5.0) return 0.0;
    vec3 drift = vec3(cloudParams.wind.x, 0.0, cloudParams.wind.y);
    float evo = cloudParams.weather.z;
    // Banks: the base noise squashed 4x vertically -- fog lies in sheets.
    vec3 q = (p - drift) * L.c.x;
    q.y *= 4.0;
    q += vec3(0.311, 0.127, 0.719) + vec3(evo * 0.21, 0.0, evo * 0.17);
    vec4 n = textureLod(cloudBaseNoise, q, 0.0);
    float fbm = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
    float depth = max(m.y - m.w, 10.0);
    float lid = m.y + (n.r - 0.5) * 0.7 * depth;
    float body = m.x * clamp((lid - p.y) / 25.0, 0.0, 1.0) +
                 m.z * exp(-max(p.y - m.w, 0.0) / 250.0);
    float shape = mix(0.15, 1.0, smoothstep(0.25, 0.7, fbm));
    baseDensity = body * shape;
    if (!detail) return baseDensity;
    // Wisps: the detail octaves erode the edges only (cores stay whole).
    vec4 d = textureLod(cloudDetailNoise, (p - drift) * L.c.y + vec3(evo * 0.9), 0.0);
    float wisp = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
    return body * clamp(cloudRemap(shape, wisp * 0.45, 1.0, 0.0, 1.0), 0.0, 1.0);
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
    for (int i = 0; i < 4; ++i) {   // 3 = the storm column (base lowered by the wall cloud)
        CloudLayerG L = cloudParams.layers[i];
        if (L.b.w >= 0.5 && L.b.x > 0.0 && L.a.z > 0.0) {
            lowest = min(lowest, L.a.x);
        }
    }
    if (cloudParams.cirrus.x >= 0.5 && cloudParams.cirrus.w > 0.0) {
        lowest = min(lowest, cloudParams.cirrus.y);
    }
    // Local fog sits on the ground: any segment may cross it.
    if (cloudFogActive()) lowest = min(lowest, cloudParams.layers[4].a.x);
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

// Storm cell volumes (defined with the storm column below).
bool cloudVolumesActive();
bool cloudVolumeNear(vec3 p, float alt, float dilation);
vec2 cloudVolumeSample(vec3 p, float alt);
// Horizontal distance (m) from the last cloudVolumeSample() point to the
// nearest side of the box holding it (the largest over overlapping boxes;
// 0 = in no box). Effects that reach out of the volume fade with it, so they
// never end at a box face.
float cloudVolumeLastEdge = 0.0;

// Extinction (1/m) of one AUTHORED layer (0..2) at world point p, without
// the storm column. `alt` = cloudAltitude(p), passed in so a caller summing
// layers computes it once. `detail` = false gives the cheap base shape
// (majorant, light-march, empty-space skip).
// Bound kept for the RT majorant: result <= local coverage * extinction.
float cloudLayerOwnDensity(int li, vec3 p, float alt, bool detail,
                           out float baseDensity) {
    baseDensity = 0.0;
    if (cloudParams.layers[li].b.w < 0.5 || cloudParams.layers[li].b.x <= 0.0) return 0.0;
    CloudLayerG L = cloudStormLayerAt(li, alt);
    if (L.b.w < 0.5 || L.b.x <= 0.0) return 0.0;
    bool stormRegion = L.d.z > 0.5;
    float h = (alt - L.a.x) / max(L.a.y, 1.0);
    if (h < 0.0 || h >= 1.0 || (!stormRegion && h <= 0.0)) return 0.0;
    vec3 pWorld = p;   // the volumes are placed unsheared
    p = cloudShear(L, p, alt);

    vec4 w = cloudWeatherAt(L, p);
    // Storm cells raise cover, genus and development locally; the weather
    // noise still shapes their edges (ATMOSPHERE_STORMS.md §3).
    vec4 cell = cloudStormCellAt(p);
    // With the storm column on, the cells are drawn by the column; the layer
    // steps aside inside them (no double density) and keeps its own cover.
    float shellMask = 0.0;
    float skirt = 0.0;
    if (cloudShellActive()) {
        shellMask = smoothstep(0.05, 0.4, cell.r);
        // Inside a cell volume's box the layer steps aside from the volume's
        // ACTUAL surface, not from the cell's cover disc: leaning towers and
        // off-centre turrets left the disc, and the layer filled the towers'
        // lower part a second time -- a hard line at the layer top.
        // Around the tower the deck is drawn INTO the storm (inflow): toward
        // the surface it fills in and heaps up against it. Bounded by the
        // cell's dilated cover (cell.a), which the layer majorant adds in
        // volume mode, so the bound holds.
        // ★ Squared fall-off, a broken cover and the full depth only at the
        // wall: a linear 2.5 km fall-off with cover 0.7..1 and full depth
        // drew a flat-topped slab at the layer top under every storm, cut
        // straight at the box faces (2026-10-14) -- hence the edge fade.
        if (cloudVolumesActive() && cloudVolumeNear(pWorld, alt, 0.0)) {
            float sdfV = cloudVolumeSample(pWorld, alt).x;
            shellMask = 1.0 - smoothstep(-100.0, 100.0, sdfV);
            float s = 1.0 - smoothstep(0.0, 2500.0, sdfV);
            skirt = s * s * smoothstep(0.0, 2500.0, cloudVolumeLastEdge) * cell.a;
        }
        cell = vec4(0.0);
    }
    // Cell structure (turrets, rotation): read only inside a cell's reach.
    vec4 flow = cell.a > 0.0 ? cloudStormFlowAt(p) : vec4(0.0);
    float cellW = smoothstep(0.05, 0.35, cell.r);
    float lift = flow.w;
    // Per-cell development (weather A): shallow cells sit a little higher
    // (no perfectly flat base across the field) and stop lower.
    float dev = max(max(w.a, cell.g), skirt);
    if (!stormRegion) {
        h = (h - (1.0 - dev) * 0.04) / (1.0 - (1.0 - dev) * 0.04);
    }
    if (h <= 0.0 && !stormRegion) return 0.0;
    // Local coverage: the weather channel is ~uniform (CDF-flattened in the
    // generator), so thresholding it at 1 - coverage covers ~coverage of sky.
    float cov = clamp(cloudRemap(w.r, 1.0 - L.a.z, 1.0, 0.0, 1.0), 0.0, 1.0);
    cov = max(cov, cell.r * mix(0.7, 1.0, w.r));   // <= cell.a: the majorant holds
    cov = max(cov, skirt * mix(0.3, 1.0, w.r));    // <= cell.a as well; broken
    cov *= 1.0 - shellMask;
    if (cov <= 0.0) return 0.0;
    float type = clamp(L.a.w + (w.g - 0.5) * cloudParams.weather.y, 0.0, 1.0);
    type = max(type, cell.g);
    type = max(type, 0.85 * skirt);   // the deck heaps into cumulus toward the storm
    vec3 drift = vec3(cloudParams.wind.x, 0.0, cloudParams.wind.y);
    float evo = cloudParams.weather.z;
    // The cell's rotation winds the SHAPE lookup (never the weather or the
    // coverage: the majorant holds). Strongest at mid levels.
    vec3 shapeP = p;
    shapeP.xz += flow.xy * mix(0.55, 1.0, smoothstep(0.0, 0.5, h));
    vec3 relativePosition = cloudShapeCoordinates(li, shapeP - drift);
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
    // ★ The base texture tiles every 1/(c.x * 0.271) m (6.6 km at 1.8 km
    // cells). Read at a FIXED slice, the formation field was one 2D tile
    // repeated across the sky: seen from altitude, rows of identical cells
    // ran to the horizon along the tile diagonal (measured: 4.7 km apart =
    // 6.64 km / sqrt 2). The slice now follows weather G (type, ~50 km
    // features, 150 km period): neighbouring tiles read different depths,
    // one cloud sees a near-constant slice. Shape only, no extra fetch.
    float slice = w.g * 1.7;
    vec3 formationQ = vec3(relativePosition.x * L.c.x * 0.271,
                           0.37 + slice,
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
    // Inside a storm cell the top is the cell's own: rising turrets reach
    // their heights, the body between them stays lower (cloud_cells.comp).
    // Without this the cell's full cover made one flat slab of the layer.
    if (flow.z > 0.0) topFrac = mix(topFrac, max(flow.z, 0.25), cellW);
    topFrac = mix(topFrac, 1.0, skirt * skirt);   // full depth only at the wall
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
    // Rising turrets: taller billows.
    verticalScale *= mix(1.0, 0.75, lift);
    float shapeBase = stormRegion ? L.g.x : L.a.x;
    vec3 q = vec3(relativePosition.x, alt - shapeBase, relativePosition.z) *
             L.c.x * vec3(1.0, verticalScale, 1.0) + phase +
             vec3(evo * 0.37, evo * 0.11, evo * 0.53);
    q.xz += (nL.ga - vec2(0.5)) * hh * hh * convective * 0.8;
    // A continuous oblique component separates billow rows across a field.
    // On its own it left the (1,1) diagonal nearly periodic (0.19 - 0.13 =
    // 0.06 of a slice per tile): the weather slice breaks that too.
    q.y += dot(q.xz - phase.xz, vec2(0.19, -0.13)) * convective + slice;
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
    // Ragged base band (CloudLayer::base_raggedness): the bottom 120-600 m.
    float rag = L.g.w;
    float bandM = clamp((stormRegion ? L.g.y : L.a.y) * 0.1, 120.0, 600.0) * mix(0.5, 1.0, rag);
    float hb = (stormRegion ? alt - L.g.x : h * L.a.y) / bandM;
    bool inBand = rag > 0.0 && hb < 1.0;
    if (detailWeight <= 0.0 && !inBand) return shapeN * covScale * L.b.x;

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
    float crown = max(smoothstep(0.15, 0.65, hh) * convective,
                      lift * smoothstep(0.3, 0.8, hh));
    float dmod = mix(dfbm, 1.0 - dfbm, crown);
    float erosionHeight = mix(0.65, mix(0.2, 1.0, baseProtection), convective);
    // An updraft core is hard-edged cauliflower, not wisps: less erosion.
    float erosionThreshold = dmod * L.b.z * mix(0.45, 0.72, towering) *
                             erosionHeight * detailWeight * mix(1.0, 0.5, lift);
    float dens = clamp(cloudRemap(shapeN, erosionThreshold, 1.0, 0.0, 1.0),
                       0.0, 1.0);
    if (inBand) {
        // A flat condensation base cut by one plane read as a sheet. Large
        // hanging lobes stay where the formation field peaks (km scale, the
        // same read that organises the towers); between them the base
        // retreats up to 75% of the band, and the lobe undersides billow
        // with the base texture's mid octave. Detail pass only: it removes
        // density, so the cheap shape stays the majorant, the light-march
        // bound and the rain footprint.
        float lobes = smoothstep(0.38, 0.72, nL.g);
        float retreat = (1.0 - lobes) * 0.75 * rag;
        float underside = (n.a - 0.5) * 0.3 * rag;
        float body = smoothstep(retreat - 0.1, retreat + 0.15, hb + underside);
        // Fractus: small detached pieces in the retreated gap, from the
        // detail octaves (independent of the lobe field, so they float free).
        float pieces = smoothstep(0.6, 0.82, d.g * 0.6 + d.b * 0.4) * (1.0 - body) * rag;
        dens *= max(body, pieces * 0.85);
    }
    return dens * covScale * L.b.x;
}

// ── Storm column (ATMOSPHERE_CELL_CLOUDS.md §1) ────────────────────────────
// A cell is its own column from the cloud base (lowered into a wall cloud
// under the main updraft) to the tropopause, with overshooting turrets above
// it, an anvil spreading downwind at the tropopause and mammatus pouches
// under the anvil. Layer 3 carries the shell: a = base, depth (to the
// overshoot top); the rest is the host layer's look (noise scales, erosion,
// material). storms2 = (stratiform share, tropopause height fraction in the
// shell, wall-cloud allowance m, mammatus strength).
//
// Bound for the majorant: density <= max(cell cover envelope, anvil cover)
// x extinction, i.e. <= max(cell.a, column.a) x extinction.

// Wind shear for the column: the crowns lean like a layer's, but the lean is
// capped at 3 km of height -- a troposphere-deep quadratic lean laid the
// tower flat.
vec3 cloudShellShear(vec3 p, float alt) {
    vec2 sh = cloudParams.precip2.zw;
    if (dot(sh, sh) <= 0.0) return p;
    CloudLayerG L = cloudParams.layers[3];
    float above = clamp(alt - L.a.x, 0.0, 3000.0);
    vec2 v = textureLod(cloudCurlNoise, (p.xz - cloudParams.wind.xy) * cloudParams.weather.x * 6.0, 0.0).rg;
    float k = 0.4 + 1.2 * v.x;
    float a = (v.y - 0.5) * 0.9;
    vec2 rsh = vec2(sh.x * cos(a) - sh.y * sin(a), sh.x * sin(a) + sh.y * cos(a)) * k;
    float disp = above * (above / 3000.0) * 1.6;
    return p - vec3(rsh.x, 0.0, rsh.y) * disp;
}

float cloudShellMajorant(vec3 p) {
    if (!cloudShellActive()) return 0.0;
    p = cloudShellShear(p, cloudAltitude(p));
    return max(cloudStormCellAt(p).a, cloudStormColumnAt(p).a) * cloudParams.layers[3].b.x;
}

float cloudShellTop(float tower, float anvilH) {
    return tower < 0.75 ? tower / 0.75 * anvilH : mix(anvilH, 1.0, (tower - 0.75) / 0.25);
}

// Tower = a SURFACE, not a cover-scaled noise volume. Reference (towering
// cumulus / congestus): near-vertical walls, rounded stepped crowns, every
// part of the surface covered in hard cauliflower billows at several
// scales, a dense interior and a flat base. "cover x noise" (the layer
// model) drew soft cotton cones without a surface; batch 16 tried it again
// as a stretched layer generator and drew mountains (2026-10-12, reverted).
// Here the turret dome field (cloud_cells.comp, sqrt domes) is the top
// height; billow noise pushes that surface UP (crowns) and, through a
// displaced lookup of the dome field, SIDEWAYS (walls); density = the depth
// below the surface over a few tens of metres. The detail pass only dents it
// (smaller billows), so the cheap shape stays an upper bound.
const float kTowerBillowScaleM = 1800.0;   // big lobes ~ 0.5-1 km
const float kTowerBubbleScaleM = 320.0;    // small cauliflower bubbles
// Orthonormal rotation for the bubble octave (Quilez): it no longer lines up
// with the billow octave of the same tiled texture.
const mat3 kCloudOctaveRotation = mat3(0.00, 0.80, 0.60,
                                       -0.80, 0.36, -0.48,
                                       -0.60, -0.48, 0.64);
// ── Storm cell volumes (ATMOSPHERE_CLOUD_VOLUMES.md §3.3-3.4) ──────────────
// A developed cell is a signed-distance volume in its own atlas slot instead
// of the heightfield below: the column's intervals, majorant and the layers'
// stepping aside stay; inside a box the volume replaces the shell (no double
// density), outside every box the shell still draws the remaining cells.
bool cloudVolumesActive() {
    return cloudParams.volume.x > 0.5 && cloudShellActive();
}

// p inside (or within `dilation` metres of) an active box.
bool cloudVolumeNear(vec3 p, float alt, float dilation) {
    for (int i = 0; i < 16; ++i) {
        CloudVolumeSlotG S = cloudParams.volumeSlots[i];
        if (S.a.w < 0.5) break;   // slots are packed from 0
        vec3 lo = vec3(S.a.x, S.a.y, S.a.z) - dilation;
        vec3 hi = vec3(S.a.x + S.b.x, S.a.y + S.b.y, S.a.z + S.b.z) + dilation;
        vec3 q = vec3(p.x, alt, p.z);
        if (all(greaterThanEqual(q, lo)) && all(lessThanEqual(q, hi))) return true;
    }
    return false;
}

// Largest dent the detail pass cuts into a volume surface (m): the marcher's
// empty-space skip keeps this much margin (plus a voxel) from the surface.
const float kCloudVolumeDentMaxM = 150.0;
// Signed distance of the last cloudVolumeDensity() call (1e6 = p was in no
// volume box). The march reads it right after a column sample to step over
// the empty part of a box (sphere tracing, cloud_rt.glsl).
float cloudVolumeLastSdf = 1e6;

// x = signed distance (m) to the nearest volume surface (1e6 outside every
// box), y = detail kind of that surface (0 crisp cauliflower .. 1 fibrous ice).
vec2 cloudVolumeSample(vec3 p, float alt) {
    vec2 best = vec2(1e6, 0.0);
    cloudVolumeLastEdge = 0.0;
    ivec3 sz = textureSize(cloudVolumeAtlas, 0);
    float slots = float(sz.z / max(sz.x, 1));
    float texel = 1.0 / float(max(sz.x, 1));
    for (int i = 0; i < 16; ++i) {
        CloudVolumeSlotG S = cloudParams.volumeSlots[i];
        if (S.a.w < 0.5) break;
        // The atlas is reallocated a frame after the budget grows: a slot it
        // does not hold yet is skipped, never read out of range.
        if (S.b.w + 0.5 > slots) continue;
        vec3 uvw = vec3((p.x - S.a.x) / S.b.x, (alt - S.a.y) / S.b.y, (p.z - S.a.z) / S.b.z);
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) continue;
        vec2 edge = min(uvw.xz, 1.0 - uvw.xz) * S.b.xz;
        cloudVolumeLastEdge = max(cloudVolumeLastEdge, min(edge.x, edge.y));
        // Slots are stacked along w: clamp half a texel inside the slot so the
        // linear filter never blends two of them.
        uvw.z = (S.b.w + clamp(uvw.z, 0.5 * texel, 1.0 - 0.5 * texel)) / slots;
        vec2 v = textureLod(cloudVolumeAtlas, uvw, 0.0).rg;
        if (v.x < best.x) best = v;
    }
    return best;
}

float cloudVolumeSdf(vec3 p, float alt) {
    return cloudVolumeSample(p, alt).x;
}

// Signed distance at a world point, capped at the atlas clamp so a probe that
// leaves every box reads "far", not 1e6.
float cloudVolumeSdfAt(vec3 q) {
    return min(cloudVolumeSdf(q, cloudAltitude(q)), 2000.0);
}

// Outward surface normal of the volumes near p: tetrahedral gradient of the
// distance over `h` metres (Quilez), 4 atlas reads.
vec3 cloudVolumeNormal(vec3 p, float h) {
    const vec2 k = vec2(1.0, -1.0);
    vec3 g = k.xyy * cloudVolumeSdfAt(p + k.xyy * h) +
             k.yyx * cloudVolumeSdfAt(p + k.yyx * h) +
             k.yxy * cloudVolumeSdfAt(p + k.yxy * h) +
             k.xxx * cloudVolumeSdfAt(p + k.xxx * h);
    return dot(g, g) > 1e-6 ? normalize(g) : vec3(0.0, 1.0, 0.0);
}

// Openness of the surface around p (1 = a free bulge, -> 0 deep in a crevice
// between bubbles): along the normal a flat surface's distance grows by the
// probe length; a neighbouring bulge keeps it smaller. 2 atlas reads.
float cloudVolumeOpenness(vec3 p, vec3 n, float sdf) {
    float occ = 0.0;
    for (int i = 0; i < 2; ++i) {
        float h = i == 0 ? 220.0 : 600.0;
        occ += clamp((sdf + h - cloudVolumeSdfAt(p + n * h)) / h, 0.0, 1.0);
    }
    return 1.0 - 0.45 * occ;
}

// Extinction inside a volume: CONSTANT liquid water (the host layer's
// extinction), not scaled by the cell's cover -- a weak cell is a smaller
// cloud, not a more transparent one (ATMOSPHERE_CLOUD_VOLUMES.md §0). The
// surface ramps over `skin` metres of signed distance; detail only dents it
// (<= kCloudVolumeDentMaxM), so the cheap shape stays the majorant.
//   kind 0 (towers): inverted-Worley bubbles, small cauliflower;
//   kind 1 (anvil):  horizontally stretched fbm, fibrous ice.
float cloudVolumeDensity(vec3 p, float alt, bool detail, out float baseDensity) {
    baseDensity = 0.0;
    vec2 v = cloudVolumeSample(p, alt);
    float sdf = v.x;
    cloudVolumeLastSdf = sdf;
    float skin = max(cloudParams.volume.z, 1.0);
    if (sdf >= 0.0) return 0.0;
    float sigma = cloudParams.layers[3].b.x;
    baseDensity = sigma * smoothstep(0.0, skin, -sdf);
    if (!detail || -sdf > skin + kCloudVolumeDentMaxM) return baseDensity;
    vec3 drift = vec3(cloudParams.wind.x, 0.0, cloudParams.wind.y);
    vec3 rel = p - drift;
    float evo = cloudParams.weather.z;
    vec3 phase = cloudShapePhase(0);
    float dent = 0.0;
    if (v.y < 0.99) {
        vec4 d = textureLod(cloudDetailNoise, (kCloudOctaveRotation * rel) / kTowerBubbleScaleM +
                            phase + vec3(evo * 1.3), 0.0);
        float bubbles = d.r * 0.6 + d.g * 0.3 + d.b * 0.1;
        // The generator carries the billows down to ~600 m (and its voxels
        // ~100 m); this octave is the cauliflower below that.
        dent += (1.0 - v.y) * (1.0 - bubbles) * 120.0;
    }
    if (v.y > 0.01) {
        vec4 f = textureLod(cloudDetailNoise, rel * vec3(1.0 / 900.0, 1.0 / 110.0, 1.0 / 900.0) +
                            phase.zxy + vec3(evo * 0.6), 0.0);
        float fbm = f.r * 0.625 + f.g * 0.25 + f.b * 0.125;
        dent += v.y * (1.0 - fbm) * kCloudVolumeDentMaxM;
    }
    return sigma * smoothstep(0.0, skin, -sdf - dent);
}

float cloudShellDensity(vec3 p, float alt, bool detail, out float baseDensity) {
    baseDensity = 0.0;
    if (!cloudShellActive()) return 0.0;
    if (cloudVolumesActive() && cloudVolumeNear(p, alt, 0.0))
        return cloudVolumeDensity(p, alt, detail, baseDensity);
    CloudLayerG L = cloudParams.layers[3];
    float depth = max(L.a.y, 1.0);
    float h = (alt - L.a.x) / depth;
    if (h <= 0.0 || h >= 1.0) return 0.0;
    p = cloudShellShear(p, alt);
    vec4 cell = cloudStormCellAt(p);
    vec4 column = cloudStormColumnAt(p);
    if (cell.r <= 0.0 && column.r <= 0.0) return 0.0;
    vec4 flow = cloudStormFlowAt(p);
    float anvilH = cloudParams.storms2.y;
    // Local base: the host layer's base, lowered by up to the allowance into
    // a wall cloud under the main updraft.
    float baseM = cloudParams.storms2.z * (1.0 - column.g);
    float aboveBase = alt - L.a.x - baseM;
    if (aboveBase <= 0.0) return 0.0;
    float lift = flow.w;

    vec3 drift = vec3(cloudParams.wind.x, 0.0, cloudParams.wind.y);
    float evo = cloudParams.weather.z;
    vec3 sp = p;
    sp.xz += flow.xy * mix(0.55, 1.0, smoothstep(0.0, 0.5, h));
    vec3 rel = cloudShapeCoordinates(3, sp - drift);
    rel.y = alt - L.a.x;
    vec3 phase = cloudShapePhase(0);
    // Break the texture period. One noise tile is 0.3-2 km, a tower ten:
    // read straight, the same billows marched up the tower in rows. A slow
    // 3D warp (~7 km) shifts every tile differently. Shape lookups only --
    // `rel.y` stays the true height.
    vec3 relW = rel + (textureLod(cloudBaseNoise, rel / 7300.0 + phase.zxy, 0.0).gba - 0.5) * 1800.0;

    // ── Tower ──
    float towerD = 0.0;
    float inside = -1.0;
    if (cell.r > 0.0) {
        // Rising turrets boil faster.
        vec3 qb = relW / kTowerBillowScaleM + phase + vec3(evo * 0.37, evo * (0.11 + 0.2 * lift), evo * 0.53);
        vec4 n = textureLod(cloudBaseNoise, qb, 0.0);
        float billow = n.g * 0.55 + n.b * 0.3 + n.a * 0.15;   // Worley octaves: rounded lobes
        // Walls: read the dome field a few hundred metres off, along a
        // noise vector -- the wall bulges in and out.
        vec2 side = (n.gb - 0.5) * 2.0 * 380.0;
        float towerTop = cloudStormFlowAt(p + vec3(side.x, 0.0, side.y)).z;
        float surfaceM = cloudShellTop(towerTop, anvilH) * depth + (billow - 0.5) * 2.0 * 320.0;
        inside = surfaceM - rel.y;
        // Dense below the surface, a flat (slightly soft) condensation base.
        towerD = smoothstep(0.0, 70.0, inside) * smoothstep(0.0, 120.0, aboveBase) * cell.r;
    }

    // ── Anvil: a WEDGE under the tropopause ──
    // Thick (~1.5 km) over the tower it grows out of, thinning toward its
    // downwind edge -- the underside rises with the distance, the top stays
    // flat (domed a little over the core). A constant-thickness band drew a
    // flying-saucer disc with vertical rims. Mammatus pouches hang from the
    // underside. Fibrous ice: squashed, stretched noise, cover-thresholded.
    float anvilD = 0.0;
    if (column.r > 0.0 && h > anvilH - 0.22 && h < anvilH + 0.04) {
        float a = smoothstep(0.0, 0.7, column.r);
        float thick = mix(0.015, 0.14, a);
        float pouch = 0.0;
        if (cloudParams.storms2.w > 0.0 && a > 0.2) {
            vec4 m = textureLod(cloudBaseNoise, vec3(relW.x, 0.0, relW.z) * L.c.x * 2.2 + phase.zxy, 0.0);
            pouch = smoothstep(0.45, 0.8, m.r) * cloudParams.storms2.w * a;
        }
        float bottom = anvilH - thick - 0.04 * pouch;
        float topA = anvilH + 0.02 * smoothstep(0.5, 1.0, a);
        float band = smoothstep(bottom, bottom + 0.3 * thick, h) *
                     (1.0 - smoothstep(topA - 0.015, topA + 0.01, h));
        float cov = column.r * band;
        if (cov > 0.0) {
            vec3 q = relW * L.c.x * vec3(1.0, 2.2, 1.0) + phase + vec3(evo * 0.37, evo * 0.11, evo * 0.53);
            vec4 n = textureLod(cloudBaseNoise, q, 0.0);
            float lowFbm = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
            float shape = clamp(cloudRemap(n.r, -(1.0 - lowFbm), 1.0, 0.0, 1.0), 0.0, 1.0);
            anvilD = clamp(cloudRemap(shape, 1.0 - cov, 1.0, 0.0, 1.0), 0.0, 1.0) * cov;
        }
    }
    // Both terms are <= the cover read at p (cell.r, column.r): the majorant holds.
    float base = max(towerD, anvilD);
    if (base <= 0.0) return 0.0;
    baseDensity = base * L.b.x;
    if (!detail) return baseDensity;

    // Detail: only removes. Tower: dents between small bubbles (the surface
    // moves in by up to ~90 m where the inverted Worley is low). Anvil:
    // fibrous erosion.
    vec3 dq = (kCloudOctaveRotation * relW) / kTowerBubbleScaleM + phase + vec3(evo * 1.3);
    vec4 d = textureLod(cloudDetailNoise, dq, 0.0);
    float bubbles = d.r * 0.6 + d.g * 0.3 + d.b * 0.1;
    float towerDetail = towerD;
    if (towerD > 0.0) {
        float dent = (1.0 - bubbles) * mix(90.0, 60.0, lift);
        towerDetail = smoothstep(0.0, 70.0, inside - dent) * smoothstep(0.0, 120.0, aboveBase) * cell.r;
    }
    float anvilDetail = anvilD;
    if (anvilD > 0.0) {
        vec4 f = textureLod(cloudDetailNoise, (kCloudOctaveRotation * relW) * L.c.y * vec3(1.0, 0.4, 1.0) +
                            phase + vec3(evo * 1.3), 0.0);
        float dfbm = f.r * 0.625 + f.g * 0.25 + f.b * 0.125;
        float erosion = dfbm * L.b.z * 1.2;
        anvilDetail = clamp(cloudRemap(anvilD / max(column.r, 1e-4), erosion, 1.0, 0.0, 1.0), 0.0, 1.0) *
                      column.r;
        anvilDetail = min(anvilDetail, anvilD);
    }
    return max(towerDetail, anvilDetail) * L.b.x;
}

// Density a march interval of layer `li` sees: the authored layer plus the
// storm column inside its altitude range. Layer 3 = the column alone: its
// intervals cover only the altitudes no authored layer covers
// (cloudIntervals), so nothing is counted twice.
float cloudLayerDensityResolved(int li, vec3 p, float alt, bool detail,
                               out float baseDensity) {
    if (li == 4) return cloudFogDensity(p, detail, baseDensity);
    if (li == 3) return cloudShellDensity(p, alt, detail, baseDensity);
    float own = cloudLayerOwnDensity(li, p, alt, detail, baseDensity);
    if (!cloudShellActive()) return own;
    float shellBase;
    float shell = cloudShellDensity(p, alt, detail, shellBase);
    baseDensity += shellBase;
    return own + shell;
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
// ── Precipitation forms (docs/dev/ATMOSPHERE_PRECIP_TYPES.md §1) ──────────
// The form at a point follows the hydrometeor down its column through the
// climate's temperature profile (linear: precip3.x C at sea level, precip3.y
// K/m): ice aloft melts by the degree-metres of above-zero air it has fallen
// through (snow -> sleet -> rain: the snow line and the melting band come
// out of the profile, no terrain read); liquid falling into sub-zero air is
// supercooled (freezing rain) and refreezes into ice pellets after enough
// cold degree-metres (an inversion's cold surface layer); updrafts make
// graupel and hail (hail survives a long fall); stratiform, shallow cloud
// drizzles. Ids: CPU atmosphere::PrecipForm and IPC precipitation_form.
const int PRECIP_RAIN = 1, PRECIP_DRIZZLE = 2, PRECIP_SNOW = 3, PRECIP_SLEET = 4,
          PRECIP_GRAUPEL = 5, PRECIP_HAIL = 6, PRECIP_FREEZING_RAIN = 7, PRECIP_ICE_PELLETS = 8;

struct CloudPrecipForm {
    vec4 a;   // rain, drizzle, snow, sleet
    vec4 b;   // graupel, hail, freezing rain, ice pellets
};

float cloudAirTempC(float altitude) {
    return cloudParams.precip3.x - cloudParams.precip3.y * min(altitude, 11000.0);
}

// Integral over [z1, z2] of max(sign * T(z), 0) dz, in C*m (linear profile).
float cloudDegreeMetres(float z1, float z2, float sign) {
    if (z2 <= z1) return 0.0;
    float fa = sign * cloudAirTempC(z1), fb = sign * cloudAirTempC(z2);
    if (fa <= 0.0 && fb <= 0.0) return 0.0;
    if (fa >= 0.0 && fb >= 0.0) return 0.5 * (fa + fb) * (z2 - z1);
    float zc = z1 + (z2 - z1) * fa / (fa - fb);   // the 0 C crossing
    return fa > 0.0 ? 0.5 * fa * (zc - z1) : 0.5 * fb * (z2 - zc);
}

CloudPrecipForm cloudPrecipFormAt(float alt, float baseAlt, float drizzleShare, float lift) {
    CloudPrecipForm f;
    f.a = vec4(0.0);
    f.b = vec4(0.0);
    int forced = int(cloudParams.precip3.z + 0.5);
    if (forced > 0) {
        if (forced <= 4) f.a[forced - 1] = 1.0; else f.b[min(forced - 5, 3)] = 1.0;
        return f;
    }
    // Ice aloft: a cold cloud base, or any convective updraft (its upper
    // part is far below zero). A warm shallow cloud rains liquid throughout.
    bool iceAloft = cloudAirTempC(baseAlt) <= 0.0 || lift > 0.05;
    float melt = cloudDegreeMetres(alt, baseAlt, 1.0);
    float liquid = iceAloft ? smoothstep(30.0, 400.0, melt) : 1.0;
    // Supercooled liquid in sub-zero air: freezing rain, or ice pellets once
    // the cold layer it has fallen through is deep enough. The warm layer it
    // melted in is above the cold one (an inversion).
    float freezing = 0.0, pellets = 0.0;
    if (cloudAirTempC(alt) < 0.0 && liquid > 0.0) {
        float refreeze = cloudDegreeMetres(alt, baseAlt, -1.0);
        if (iceAloft) {
            // Count only the cold air BELOW the melting layer.
            float zc = cloudParams.precip3.y < 0.0 ? cloudParams.precip3.x / cloudParams.precip3.y : alt;
            refreeze = cloudDegreeMetres(alt, max(zc, alt), -1.0);
        }
        pellets = liquid * smoothstep(1500.0, 3000.0, refreeze);
        freezing = liquid - pellets;
        liquid = 0.0;
    }
    float sleet = 4.0 * liquid * (1.0 - liquid);
    float snow = (1.0 - liquid - freezing - pellets) * (1.0 - sleet);
    float rain = liquid * (1.0 - sleet);
    // A SHARE of the mass, not the whole of it: rain or snow still carries
    // most of a core's precipitation (live 2026-10-12: as exclusive weights
    // graupel won under every mature core, even at -5 C and under an
    // inversion that should give freezing rain).
    float hail = 0.6 * smoothstep(0.55, 0.85, lift);
    float graupel = 0.4 * smoothstep(0.3, 0.6, lift) * (1.0 - hail);
    if (iceAloft) {
        // Graupel melts within ~1 km below the freezing level; hail survives.
        graupel *= 1.0 - smoothstep(200.0, 1500.0, melt);
        hail *= 1.0 - smoothstep(4000.0, 12000.0, melt);
    } else {
        hail = 0.0;
        graupel = 0.0;
    }
    float rest = 1.0 - hail - graupel;
    float drizzle = rain * drizzleShare;
    f.a = vec4(rain - drizzle, drizzle, snow, sleet) * rest;
    f.b = vec4(graupel, hail, freezing * rest, pellets * rest);
    return f;
}

int cloudPrecipDominantForm(CloudPrecipForm f) {
    int best = 0;
    float w = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (f.a[i] > w) { w = f.a[i]; best = i + 1; }
        if (f.b[i] > w) { w = f.b[i]; best = i + 5; }
    }
    return best;
}

// Optics per form (ATMOSPHERE_PRECIP_TYPES.md §2): extinction (1/m) at
// 1 mm/h and its rate exponent (Marshall-Palmer rain sigma ~ R^0.63; snow
// and graupel scatter far more per mm of water; hail far less), terminal
// fall speed m/s (the lean), forward-lobe share and streak contrast.
const vec4 kPrecipSigmaA = vec4(2.5e-4, 1.2e-3, 2.6e-3, 1.4e-3);
const vec4 kPrecipSigmaB = vec4(1.2e-3, 1.5e-4, 2.5e-4, 8.0e-4);
const vec4 kPrecipExpA   = vec4(0.63, 0.6, 0.7, 0.67);
const vec4 kPrecipExpB   = vec4(0.7, 0.6, 0.63, 0.7);
const vec4 kPrecipFallA  = vec4(6.5, 1.5, 1.0, 3.0);
const vec4 kPrecipFallB  = vec4(3.0, 15.0, 6.5, 4.0);
const vec4 kPrecipFwdA   = vec4(0.5, 0.3, 0.0, 0.2);
const vec4 kPrecipFwdB   = vec4(0.1, 0.5, 0.5, 0.1);
const vec4 kPrecipStreakA = vec4(1.0, 0.15, 0.6, 0.8);
const vec4 kPrecipStreakB = vec4(0.8, 1.0, 1.0, 0.8);

// Altitude-only form (no updraft, no drizzle split): the fall speed that
// sets the lean, decided before the release point is known.
float cloudPrecipFallSpeed(float alt, float baseAlt) {
    CloudPrecipForm f = cloudPrecipFormAt(alt, baseAlt, 0.0, 0.0);
    return max(dot(f.a, kPrecipFallA) + dot(f.b, kPrecipFallB), 0.5);
}

float cloudPrecipSampleForm(vec3 p, out vec3 fallLine, out CloudPrecipForm form) {
    fallLine = p;
    form.a = vec4(0.0);
    form.b = vec4(0.0);
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
    // The lag scales with 1 / fall speed: snow trails far behind its cloud,
    // hail falls nearly straight (capped ~56 degrees).
    float alt = cloudAltitude(p);
    vec2 trail = cloudParams.precip.zw / cloudPrecipFallSpeed(alt, L.a.x);
    if (length(trail) > 1.5) trail *= 1.5 / length(trail);
    vec3 src = p - vec3(trail.x, 0.0, trail.y) * (below * fl * 0.5);
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
    // Where it rains at all: with storm cells, under their convective cores
    // (and their wide dissipating rain), plus a stratiform share of the dense
    // cover elsewhere. Without them, every dense cluster core (weather B).
    float potential = w.b;
    float convective = 0.0, lift = 0.0;
    if (cloudParams.storms.w > 0.5) {
        convective = cloudStormCellAt(src).b;
        potential = max(convective, cloudParams.storms2.x * w.b);
        if (convective > 0.0) lift = cloudStormFlowAt(src).w;
    }
    if (potential <= 0.0) return 0.0;
    // Drizzle: the rain that does not come from a convective core, out of a
    // shallow / stratiform layer (genus below cumulus).
    float stratiformShare = 1.0 - clamp(convective / max(potential, 1e-4), 0.0, 1.0);
    float drizzleShare = stratiformShare * (1.0 - smoothstep(0.25, 0.5, L.a.w));
    form = cloudPrecipFormAt(alt, L.a.x, drizzleShare, lift);
    float baseShape;
    cloudLayerDensityResolved(0, src, L.a.x + min(0.1 * L.a.y, 300.0), false, baseShape);
    float footprint = smoothstep(0.06, 0.45, baseShape / max(L.b.x, 1e-6));
    if (footprint <= 0.0) return 0.0;
    // Fringed top: bundles leave the base over 60..450 m, staggered per column.
    float top = smoothstep(0.0, mix(60.0, 450.0, tassel), below);
    return rate * potential * footprint * fade * top;
}

float cloudPrecipSample(vec3 p, out vec3 fallLine) {
    CloudPrecipForm form;
    return cloudPrecipSampleForm(p, fallLine, form);
}

float cloudPrecipAt(vec3 p) {
    vec3 fallLine;
    return cloudPrecipSample(p, fallLine);
}

// Dominant form id at p (0 = no precipitation): IPC precipitation_form.
float cloudPrecipFormIdAt(vec3 p) {
    vec3 fallLine;
    CloudPrecipForm form;
    if (cloudPrecipSampleForm(p, fallLine, form) <= 0.0) return 0.0;
    return float(cloudPrecipDominantForm(form));
}

// Extinction (1/m) of the falling drops at p. Fall streaks: drop bundles
// tens to a few hundred metres across and kilometres long, read along the
// fall line so they bend with the trailing foot. Mean ~1, so the Marshall-
// Palmer optical depth over a long path is kept. `detail` 0..1 fades the
// streaks to their mean (far shafts: a veil, not sub-pixel aliasing).
// `forward` = share of the forward lobe in the form mix (0 isotropic snow).
float cloudPrecipDensityLod(vec3 p, float detail, out float forward) {
    vec3 fallLine;
    CloudPrecipForm form;
    forward = 0.0;
    float r = cloudPrecipSampleForm(p, fallLine, form);
    if (r <= 0.0) return 0.0;
    forward = dot(form.a, kPrecipFwdA) + dot(form.b, kPrecipFwdB);
    // Drizzle is a homogeneous mist, hail sharp streaks.
    detail *= dot(form.a, kPrecipStreakA) + dot(form.b, kPrecipStreakB);
    float streak = 1.0;
    if (detail > 0.0) {
        vec2 s = fallLine.xz - cloudParams.wind.xy;
        float alt = cloudAltitude(p);
        float n1 = textureLod(cloudBaseNoise, vec3(s.x / 420.0, alt / 5200.0, s.y / 420.0), 0.0).g;
        float n2 = textureLod(cloudBaseNoise, vec3(s.x / 110.0 + 0.37, alt / 1900.0, s.y / 110.0 + 0.61), 0.0).b;
        float fine = mix(0.15, 1.85, smoothstep(0.25, 0.75, n1 * 0.65 + n2 * 0.35));
        streak = mix(1.0, fine, detail);
    }
    vec4 sa = kPrecipSigmaA * pow(vec4(r), kPrecipExpA);
    vec4 sb = kPrecipSigmaB * pow(vec4(r), kPrecipExpB);
    return (dot(form.a, sa) + dot(form.b, sb)) * streak;
}

float cloudPrecipDensity(vec3 p) {
    float forward;
    return cloudPrecipDensityLod(p, 1.0, forward);
}

float cloudDensity(vec3 p, bool detail) {
    float alt = cloudAltitude(p);
    float s = 0.0, b;
    for (int i = 0; i < 3; ++i) s += cloudLayerOwnDensity(i, p, alt, detail, b);
    return s + cloudShellDensity(p, alt, detail, b) + cloudFogDensity(p, detail, b);
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
