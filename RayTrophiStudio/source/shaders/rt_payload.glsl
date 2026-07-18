// rt_payload.glsl — THE definition of the location-0 path-tracing payload,
// shared by every RT stage that traces or receives it (raygen, closesthit,
// miss, hair/sphere/volume closesthit, photon raygen). The payload lives in
// registers/scratch across every traceRayEXT, so its size is a direct
// occupancy cost on the whole pipeline — keep it packed.
//
// 20 dwords (was 29). What got packed or dropped:
//   - primaryAlbedo/primaryTransmission → 2x packHalf2x16 (denoiser AOV data,
//     averaged over samples; half is far below what OIDN can resolve)
//   - primaryNormal → snorm16x2 octahedral (same argument)
//   - primaryHit + primaryMaterialId + dispersion hero channel → one meta word
//   - occluded, hitEmissive, primaryMetallic → deleted (written, never read)
//
// ABI RULE (unchanged): every field change here means recompiling ALL shaders
// that include this file — they alias the same payload memory.

struct RayPayload {
    vec3  radiance;
    vec3  attenuation;
    vec3  scatterOrigin;
    vec3  scatterDir;
    uint  seed;
    bool  scattered;
    bool  skipAABBs;      // set by volume_closesthit when a solid surface is found inside
    uint  bounceType;
    // ── Primary-hit AOV block, packed ───────────────────────────────────────
    uint  primaryARG;     // packHalf2x16(primaryAlbedo.r, primaryAlbedo.g)
    uint  primaryABT;     // packHalf2x16(primaryAlbedo.b, primaryTransmission)
    uint  primaryNrm;     // packSnorm2x16(octahedral(primaryNormal))
    uint  primaryMeta;    // bits 0-15  : primary material id (0xFFFF = none/unknown —
                          //              material id 0 is VALID, compare against the mask)
                          // bit  16    : primary-hit-recorded flag
                          // bits 17-18 : spectral dispersion hero channel
                          //              (0 = unset, 1/2/3 = R/G/B; persists across
                          //               bounces — per-bounce resets must preserve it)
};

const uint PL_MATID_MASK   = 0xFFFFu;
const uint PL_PRIMARY_DONE = 1u << 16;
const uint PL_DISP_SHIFT   = 17u;
const uint PL_DISP_MASK    = 3u << 17;

// Octahedral unit-vector packing for the denoiser normal AOV.
vec2 plOctWrap(vec2 v) {
    return (1.0 - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0,
                                    v.y >= 0.0 ? 1.0 : -1.0);
}
uint plPackNormal(vec3 n) {
    float l = abs(n.x) + abs(n.y) + abs(n.z);
    vec2 p = (l > 1e-8) ? n.xy / l : vec2(0.0);
    if (n.z < 0.0) p = plOctWrap(p);
    return packSnorm2x16(p);
}
vec3 plUnpackNormal(uint u) {
    vec2 e = unpackSnorm2x16(u);
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = plOctWrap(n.xy);
    float len = length(n);
    return (len > 1e-8) ? n / len : vec3(0.0, 0.0, 1.0);
}

// Per-bounce reset value for primaryMeta: no material, primary not recorded,
// dispersion channel PRESERVED from the previous bounce (reset once per PATH
// by passing 0u as prevMeta).
uint plMetaReset(uint prevMeta) {
    return (prevMeta & PL_DISP_MASK) | PL_MATID_MASK;
}
