// rt_payload.glsl — THE definition of the location-0 path-tracing payload,
// shared by every RT stage that traces or receives it (raygen, closesthit,
// miss, hair/sphere/volume closesthit, photon raygen). The payload lives in
// registers/scratch across every traceRayEXT, so its size is a direct
// occupancy cost on the whole pipeline — keep it packed.
//
// 21 dwords (was 29). What got packed or dropped:
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
    // Set by volume_closesthit when its solid probe found a surface inside the
    // box: the ray re-traces from the SAME origin, so the box it just marched
    // must not win again. RENAMED from skipAABBs, and the rename is the point —
    // it used to mean gl_RayFlagsSkipAABBEXT, which removed EVERY procedural
    // primitive: splat spheres (0x04), authored spheres, hair. See
    // RT_MASK_VOLUME_HANDOFF for why that was wrong.
    bool  skipVolumeAABBs;
    uint  bounceType;
    bool  skipGasVolumes; // one-shot handoff from a gas segment to a SurfaceSDF
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
const uint PL_PRIMARY_VOLUME = 1u << 19;
// For ray-marched fog/gas volumes, primaryNrm stores floatBitsToUint of the
// optical-depth centroid ray distance instead of an octahedral normal.
const uint PL_PRIMARY_VOLUME_DEPTH = 1u << 20;

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

// ── Bounce classification (payload.bounceType) ───────────────────────────────
// Lives here rather than in a shader because bounceType is a PAYLOAD field, and
// every shader that writes one needs the same numbering. closesthit.rchit owned
// the full set while volume_closesthit.rchit declared its own BOUNCE_TRANSPARENT
// — a partial copy that would have silently disagreed the moment the numbering
// changed. raygen reads these to budget bounce depth per class.
const float RAY_OFFSET  = 1e-3;   // Yüzey offset (self-intersection önleme)
const uint BOUNCE_SPECULAR = 0u;
const uint BOUNCE_DIFFUSE = 1u;
const uint BOUNCE_TRANSMISSION = 2u;
const uint BOUNCE_TRANSPARENT = 3u;
// Resin interactions (glossy coat reflect + absorbing diffuse base) are tagged
// separately so raygen can cap them at a small dedicated budget — an
// energy-preserving resin would otherwise run full-depth GI paths (TDR risk).
const uint BOUNCE_RESIN = 4u;
// Interior-volume anchor: bit 21 of mat.flags (VK_MAT_FLAG_RESIN_OBJ_SPACE).
// Set = the dust/speck fields are evaluated in OBJECT space (the interior
// moves/rotates with the mesh); clear = legacy world anchor (fixed in space).
const uint MAT_FLAG_RESIN_OBJ_SPACE = (1u << 21);
// Thin-shell film: bit 19 of mat.flags (VK_MAT_FLAG_BUBBLE). Lives here rather
// than in closesthit.rchit because the fluid isosurface dispatches the same
// branch (volume_closesthit.rchit) — the moment a second shader needed it, a
// per-shader copy was one edit away from silently disagreeing.
// ★ NOTE the host has TWO different bits under this name: VK_MAT_FLAG_BUBBLE
// (1<<19, vulkan_material_types.h) and GPU_MAT_FLAG_BUBBLE (1<<18,
// material_gpu.h) — different material structs, different backends. This is
// the Vulkan one; do not "fix" it to match the other.
const uint MAT_FLAG_BUBBLE = (1u << 19);
// Glass marble full-volume entry: tagged on the FRONT-face transmit so raygen
// integrates the real interior segment (dust/dirt) before the next surface.
const uint BOUNCE_MARBLE = 5u;
// Glass mirror lobe (Fresnel reflect or TIR at an interface): the ray did NOT
// cross the surface. Kept distinct from BOUNCE_TRANSMISSION so the photon pass
// only counts real refractions as "crossed glass" — tagging reflections as
// transmission made photons bounced off a sphere's OUTER surface splat a
// mirrored ghost caustic on the floor. Camera-side raygen spends the same
// transmission budget on it, so camera behavior is unchanged.
const uint BOUNCE_GLASS_REFLECT = 6u;

// Shared tracing constants. Same reasoning as the bounce codes above: both
// closest-hit shaders and the shared BSDF module need identical values, and a
// per-shader copy of a shadow epsilon is how two paths start showing different
// contact shadows with nothing to point at.
const float INV_PI      = 0.31830988618379067154;
const float SHADOW_TMIN = 1e-3;   // Shadow rays: avoid near-field self/adjacent contact acne
// TLAS visibility: 0x01 authored solids, 0x02 gas/fog AABBs,
// 0x04 transient simulation/splat geometry, 0x08 SurfaceSDF AABBs.
// Direct-light shadows see solids and splats. Volume-internal probes retain
// their narrower masks so particle-rich gas does not cause nested RT work.
const uint RT_MASK_DIRECT_SHADOW = 0x05u;

// ★ Volume→solid handoff cull mask: clear BOTH volume bits (0x02 gas, 0x08
// SurfaceSDF), keep everything else — solids AND splats (0x04).
//
// This replaces gl_RayFlagsSkipAABBEXT, which removed every procedural
// primitive at once. The handoff's justification (volume_closesthit.rchit,
// "the nearest hit from the same origin IS that surface") only holds for
// geometry the solid PROBE could see, and the gas march's probe uses 0xF1 —
// blind to splats. So a splat sitting between the box entry and the solid was
// dropped twice: once by the probe that set solidT, and again by the handoff
// that erased it from the re-trace. Symptom: splat geometry inside a gas
// domain vanishes unless it happens to sit in FRONT of the box, where its own
// closest-hit wins before the volume ever runs.
//
// Clearing both volume bits (rather than only the one this instance carries)
// keeps the no-self-hit guarantee absolute: TLAS volume instances are only ever
// 0x02 or 0x08 (VulkanBackend.cpp), so the re-trace provably cannot land back
// in a volume box and ping-pong through raygen's free-pass budget.
//
// ★ Note this is NOT the same as skipGasVolumes (0xFD, gas→SurfaceSDF handoff):
// that one must KEEP 0x08 alive, because reaching the coincident liquid
// boundary is its entire purpose. The two masks look similar and mean opposite
// things about bit 0x08 — do not merge them.
const uint RT_MASK_VOLUME_HANDOFF = 0xF5u;
// Gas segment → coincident SurfaceSDF: clear only the gas bit.
const uint RT_MASK_GAS_HANDOFF    = 0xFDu;

