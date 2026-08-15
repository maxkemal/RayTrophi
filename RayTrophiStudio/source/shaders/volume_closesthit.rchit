/*
 * RayTrophi Studio — Vulkan Volume Closest-Hit Shader
 * Volumetric Ray Marching with Multi-Lobe Henyey-Greenstein Scattering
 *
 * OptiX uyumluluk:
 *   - HitGroupData volumetric fields ile tam eşleşme
 *   - GpuVDBVolume / GpuGasVolume density/scatter/absorption modeline uyumlu  
 *   - Henyey-Greenstein dual-lobe phase function
 *   - Delta tracking (ratio tracking) ile uyumlu woodcock stepping
 *   - Light march ile self-shadowing
 *
 * SBT offset = 1 (volume hit group index), sbtStride = 2 (triangle + volume)
 * Bu shader VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR ile çalışır.
 */

#version 460
#extension GL_EXT_ray_tracing                          : require
#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_EXT_nonuniform_qualifier                 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// ============================================================
// Constants
// ============================================================
const float PI      = 3.14159265358979323846;
const float TWO_PI  = 6.28318530717958647692;
const float INV_4PI = 0.07957747154594766788;
const float EPSILON = 1e-6;

// ============================================================
// Push Constants — must match raygen/closesthit CameraPushConstants
// ============================================================
layout(push_constant) uniform CameraPC {
    vec4  origin;
    vec4  lowerLeft;
    vec4  horizontal;
    vec4  vertical;
    uint  frameCount;
    uint  minSamples;
    uint  lightCount;
    float varianceThreshold;
    uint  maxSamples;
    float exposure_factor;

    float aperture;
    float focusDistance;
    float distortion;
    uint  bladeCount;

    uint  caEnabled;
    float caAmount;
    float caRScale;
    float caBScale;

    uint  vignetteEnabled;
    float vignetteAmount;
    float vignetteFalloff;
    float pad0;

    uint  shakeEnabled;
    float shakeOffsetX;
    float shakeOffsetY;
    float shakeOffsetZ;

    float shakeRotX;
    float shakeRotY;
    float shakeRotZ;
    float waterTime;
} cam;

// ============================================================
// Payload — raygen/closesthit ile tam eşleşme
// ============================================================
// Payload — shared ABI, single source of truth
#include "rt_payload.glsl"
#include "volume_instrumentation.glsl"


layout(location = 0) rayPayloadInEXT RayPayload payload;
// Shadow payload: rgb = transmissive tint, w = reached-light flag (0 = hit/occluded).
// This shader only uses it as a binary probe (w), with OpaqueEXT so any-hits never run.
layout(location = 1) rayPayloadEXT vec4 shadowPayload;

// ============================================================
// Descriptor Bindings
// ============================================================
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

struct LightData {
    vec4 position;
    vec4 color;
    vec4 params;
    vec4 direction;
    vec4 area_u;
    vec4 area_v;
};

layout(set = 0, binding = 3, scalar) readonly buffer LightBuffer { LightData l[]; } lights;
layout(set = 0, binding = 6) uniform sampler2D materialTextures[];
// Volume programs are live inside a long ray-march loop. Keeping the surface
// VM's 32 vec3 register file here cuts GPU occupancy even for legacy VDBs whose
// runtime branch never evaluates a graph. The host only binds programs whose
// compacted peak-live count fits this volume-specific budget.
#define MP_REGISTER_COUNT 12
#include "material_program.glsl"

// ═══════════════════════════════════════════════════════════════════════════════
// Binding 9: Volume Instances SSBO
// Matches VulkanRT::VkVolumeInstance (256 bytes per instance)
// ═══════════════════════════════════════════════════════════════════════════════
struct VkVolumeInstance {
    // Transform (48 bytes = 12 floats)
    float transform[12];
    
    // Bounds (24 bytes)
    vec3  aabb_min;
    vec3  aabb_max;
    
    // Density (16 bytes)
    float density_multiplier;
    float density_remap_low;
    float density_remap_high;
    float noise_scale;
    
    // Scattering (32 bytes)
    vec3  scatter_color;
    float scatter_coefficient;
    float scatter_anisotropy;
    float scatter_anisotropy_back;
    float scatter_lobe_mix;
    float scatter_multi;
    
    // Absorption (16 bytes)
    vec3  absorption_color;
    float absorption_coefficient;
    
    // Emission (16 bytes)
    vec3  emission_color;
    float emission_intensity;
    
    // Ray march params (16 bytes)
    float step_size;
    int   max_steps;
    int   shadow_steps;
    float shadow_strength;
    
    // Flags (16 bytes)
    int   volume_type;
    int   is_active;
    float voxel_size;
    int   shadow_stride;
    
    // Inverse transform (48 bytes = 12 floats)
    float inv_transform[12];
    
    // Reserved (24 bytes) — matches VkVolumeInstance C++ layout
    uint64_t vdb_grid_address;   // NanoVDB grid device address (or 0)
    uint64_t vdb_temp_address;   // secondary grid (temperature etc.)
    float    _reserved[2];       // [0] density cutoff, [1] reserved

    // Emission extension (256 bytes) — blackbody / color-ramp
    int   emission_mode;         // 0=off, 1=plain color, 2=blackbody/color-ramp
    float temperature_scale;
    float blackbody_intensity;
    float max_temperature;
    int   color_ramp_enabled;
    int   ramp_stop_count;
    int   _ramp_pad[2];
    float ramp_positions[8];
    float ramp_colors_r[8];
    float ramp_colors_g[8];
    float ramp_colors_b[8];
    float pivot_offset[3];
    int   source_type;
    float cloud_coverage;
    float cloud_detail;
    float cloud_erosion;
    float cloud_base_scale;
    float cloud_edge_fade;
    float cloud_offset_x;
    float cloud_offset_z;
    float cloud_seed;
    // [6] is Surface-SDF foam opacity for source_type 4, otherwise authored
    // minimum emission temperature. [7..11] carry density-noise parameters.
    float _ext_reserved[12];
    // Appended acceleration block — MUST match VkVolumeInstance in
    // include/Backend/vulkan_volume_types.h (576 bytes). Every shader that
    // declares this struct carries the same tail: the SSBO stride is
    // per-declaration, so one stale copy shifts every instance after the first.
    uint64_t majorant_address;   // per-block density max for live dense gas (0 = none)
    float    majorant_dim[3];    // block-grid resolution
    float    majorant_block;     // cells per block edge
    uint64_t flame_address;      // combustion reaction field for live dense gas (0 = none)
    uint64_t emissive_list_address; // [0]=count, [1..]=emitting block indices
    float    emissive_capacity;
    float    iso_material_index;  // SDF isosurface material, 1-based (0 = none)
    // FULLY CLAIMED: [0]=pore amount, [1]=world units per pore cell,
    // [2]=pore size variation, [3]=coordinate space (0=Material, 1=Domain,
    // 2=World — see materialAnchor below / in volume_closesthit.rchit).
    float    _accel_reserved[4];
    // Material coordinate (UVW) RESIDUAL grid: dense xyz triples at sim-grid
    // resolution holding (uvw - cell centre), so a texture on a liquid flows
    // WITH the liquid instead of the body sliding through a world-anchored
    // projection. 0 = not published, and the shader MUST then fall back to world
    // anchoring — never read 0 as "the coordinate is the origin", which
    // collapses the surface onto a single texel. See sampleMaterialCoord for why
    // this is a displacement and not the coordinate itself.
    uint64_t uvw_residual_address;
    // Same grid/origin/voxel as the residual field; 0 = not published.
    // Slotted beside the other address so the struct stays 624 bytes.
    uint64_t composition_address;
    float    uvw_dim[3];
    // World placement of that grid: origin of cell (0,0,0) and its cell size.
    // ★ NOT derivable from aabb_min/aabb_max — on a live fluid those are the
    // tight ACTIVE box of the dense/SDF grid and move every frame, while this
    // buffer spans the whole SIM grid. Using the former to index the latter
    // smears the texture along the flow and makes it swim.
    float    uvw_origin[3];
    float    uvw_voxel;
    // Explicit tail padding, mirroring vulkan_volume_types.h exactly. C++ pads
    // the alignas(16) struct to 624; scalar layout would stop at 612. Implicit
    // padding is where the two are free to disagree, and a stride mismatch does
    // not fail loudly — it reads the NEXT volume's fields as this one's.
    float    _uvw_pad[1];
};

layout(set = 0, binding = 9, scalar) readonly buffer VolumeBuffer { VkVolumeInstance v[]; } volumes;

// ════════════════════════════════════════════════════════════════════════════════
// EXTENDED WORLD DATA — for fog/atmosphere access
// ════════════════════════════════════════════════════════════════════════════════
struct VkWorldDataExtended {
    vec3  sunDir;       int   mode;
    vec3  sunColor;     float sunIntensity;
    float sunSize;      float mieAnisotropy;
    float rayleighDensity; float mieDensity;
    float humidity;     float temperature;
    float ozoneAbsorptionScale; float atmosphereIntensity;
    float airDensity;   float dustDensity;
    float ozoneDensity; float altitude;
    float planetRadius; float atmosphereHeight;
    int   multiScatterEnabled; float multiScatterFactor;
    int   cloudsEnabled; float cloudCoverage; float cloudDensity; float cloudScale;
    float cloudHeightMin; float cloudHeightMax; float cloudOffsetX; float cloudOffsetZ;
    float cloudQuality; float cloudDetail; int cloudBaseSteps; int cloudLightSteps;
    float cloudShadowStrength; float cloudAmbientStrength; float cloudSilverIntensity; float cloudAbsorption;
    float cloudAnisotropy; float cloudAnisotropyBack; float cloudLobeMix; float cloudEmissiveIntensity;
    vec3  cloudEmissiveColor; float _pad3;
    int   fogEnabled;   float fogDensity; float fogHeight; float fogFalloff;
    float fogDistance;  float fogSunScatter; vec3 fogColor; float _pad4;
    int   godRaysEnabled; float godRaysIntensity; float godRaysDensity; int godRaysSamples;
    int   aerialEnabled; float aerialMinDistance; float aerialMaxDistance; float aerialDensity;
    int   weatherEnabled; int weatherType; float weatherIntensity; float weatherDensity;
    vec3  weatherWindDirection; float weatherWindSpeed;
    float weatherPrecipitationScale; float weatherVisibility; float weatherSurfaceWetness; float weatherSurfaceAccumulation;
    float weatherSurfaceSettling; float weatherSurfaceHeight;
    int   weatherVisualMode; int weatherSurfaceResponseEnabled;
    int   envTexSlot;   float envIntensity; float envRotation; int _pad5; // nishitaLutReady
    int   envOverlayEnabled; int envOverlayBlendMode; float envOverlayIntensity; float envOverlayRotation;
    uvec2 transmittanceLUT; uvec2 skyviewLUT; uvec2 multiScatterLUT; uvec2 aerialPerspectiveLUT;
};

layout(set = 0, binding = 7, scalar) readonly buffer WorldBuffer { VkWorldDataExtended w; } worldData;
layout(set = 0, binding = 8) uniform sampler2D atmosphereLUTs[4];

// ============================================================
// Hit Attributes from intersection shader
// ============================================================
hitAttributeEXT vec2 volumeHitAttrib; // .x = tNear, .y = tFar

// ============================================================
// PCG RNG
// ============================================================
uint pcgNext(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint seed) {
    return float(pcgNext(seed)) * (1.0 / 4294967296.0);
}

// Volume-shadow query required by the shared BSDF module. The surface shader
// marches every other volume here; this shader is already INSIDE one, and its
// own grids are sampled through sampleDensityAcc rather than that path. Return
// full transmittance for now and say so: it means a liquid surface is not
// shadowed by a SEPARATE volume (a smoke plume over a pool). It is not a
// regression — this path had no volumetric shadowing at all before — but it is
// the one place the two shaders do not yet agree.
float computeVolumeShadowTransmittance(vec3 shadowOrigin, vec3 lightDir, float maxDist) {
    return 1.0;
}

// Water lobes in the shared module need the v3 helpers.
#include "water_v3.glsl"
// ONE BSDF for both shaders: this is what lets the fluid isosurface take an
// ordinary scene material (molten glass, lava, mud, chocolate) instead of the
// hand-written Fresnel + Beer-Lambert dielectric it used to be limited to.
// Scene materials. Bindings 2/24 already exist in set 0 for the surface
// closest-hit; set 0 is shared across the pipeline, so declaring them here
// needs no host-side change at all.
#include "material_struct.glsl"
layout(set = 0, binding = 2, scalar) readonly buffer VolMaterialBuffer  { Material    m[]; } materials;
layout(set = 0, binding = 24, scalar) readonly buffer VolMaterialExtBuffer { MaterialExt m[]; } materialsExt;

// Packed-ORM channel policy (which channel roughness/metallic live in, per
// material flags). Shared with the surface closest-hit rather than copied: a
// second copy of that policy is exactly how the isosurface would end up
// reading a different channel than the mesh for the very same texture.
#include "pbr_texture_policy.glsl"

#include "bsdf_scatter.glsl"

// ═══════════════════════════════════════════════════════════════════════════
// TRI-PLANAR TEXTURING for the SDF isosurface
// ═══════════════════════════════════════════════════════════════════════════
// A raymarched isosurface has no UVs, and cannot have them in the usual sense:
// there is no mesh to unwrap, and the surface is rebuilt from the field every
// frame, so there is nothing stable for a UV to be attached to. Until now that
// meant only the material's SCALAR values reached the liquid — assigning a
// texture looked like it did nothing.
//
// Tri-planar projection is the standard answer: sample the texture once per
// world axis plane and blend the three by the (sharpened) squared normal.
//
// ★ Tiling reuses the material's OWN uv_scale / uv_offset rather than adding
// new fields. On a mesh those multiply the mesh UV; here they are world units
// per tile. Same control, same meaning — "how does this texture tile" — with
// no ABI change and nothing new for an author to learn.
//
// ★ ANCHORED IN MATERIAL SPACE, not world space. The projection formula below
// is unchanged; what changed is the coordinate fed into it. Each particle
// remembers where it was born, that value is gathered onto a grid, and the
// shading point samples it — so a FLOWING liquid carries its texture instead of
// sliding through a stationary pattern. See sampleMaterialCoord.
//
// Because the coordinate is seeded with the birth position in world units, a
// body that has not moved yields uvw == worldPos and renders EXACTLY as the old
// world-anchored path did. Nothing to re-author, and the still case is a
// bit-level regression test rather than a judgement call.
//
// ★★ REMAINING LIMIT, and it is inherent to material coordinates rather than to
// this implementation: the coordinate STRETCHES with the flow. A long pour
// smears the texture along the stretch direction and, after enough deformation,
// the mapping becomes chaotic. The cure is blending two coordinate sets seeded
// at different times; that is deliberately not built here, because a single set
// is what makes the still case identical and therefore verifiable.
// ═══════════════════════════════════════════════════════════════════════════
// Re-seating a scattered ray outside the level-set band, WITHOUT stepping over
// whatever sits inside the push distance.
// ═══════════════════════════════════════════════════════════════════════════
// Every lobe leaves this surface via offset_ray, whose offset is a few float
// ULPs. That is right for a triangle — a plane of zero thickness. It is wrong
// here: the isosurface lives inside a proxy band about half a voxel thick, so a
// ULP-offset ray restarts INSIDE the band and immediately re-hits the same
// surface. The cure is to push along the direction of travel by roughly a
// voxel.
//
// ★ But that push is CENTIMETRES at production voxel sizes, so for a direction
// that continues INTO the surface it can clear the liquid AND whatever is
// immediately behind it in one step. Symptom: a pooled liquid looks right from
// above, where the body is thick, and fails at the thin SIDES, where one voxel
// spans the whole wall — the surface behind is never handed over.
//
// So probe exactly the push segment first. It is a few centimetres long and
// terminates on first hit, so it costs far less than the march that produced
// this hit. Reflections are exempt: they leave the surface outward and have
// nothing to skip.
//
// ★ ONE rule, ONE place, used by every inward-going lobe. It used to be inlined
// in the thin-shell branch with the justification "only the pass-through goes
// straight in" — then the glass lobe was added, whose refracted direction also
// goes in, and that justification silently stopped covering it.
vec3 seatOutsideBand(vec3 hitPos, vec3 dir, vec3 N, float push, out bool handOff) {
    handOff = false;
    // Outward-going (reflection): nothing between here and open space.
    if (dot(dir, N) > 0.0) return hitPos + dir * push;

    const uint PUSH_FLAGS = gl_RayFlagsTerminateOnFirstHitEXT
                          | gl_RayFlagsSkipClosestHitShaderEXT
                          | gl_RayFlagsNoOpaqueEXT;
    // Exclude gas/fog AABBs (0x02) and SurfaceSDF AABBs (0x08), but KEEP
    // transient simulation particles (0x04) — same set as the entry-side probe.
    const uint PUSH_MASK  = 0xF5;
    const uint PUSH_PROBE = 0xC17D5EEDu;
    shadowPayload = vec4(0.0, 0.0, 0.0, uintBitsToFloat(PUSH_PROBE));
    traceRayEXT(topLevelAS, PUSH_FLAGS, PUSH_MASK, 0, 1, 1,
                hitPos, 1e-4, dir, push, 1);
    handOff = (shadowPayload.w < 0.5);
    // On a hand-off, continue from the hit point itself with the volume AABBs
    // skipped (caller sets skipAABBs) so the triangle closest-hit fires on that
    // surface at its true distance and true facing — the same contract the
    // entry-side solid probe uses, and the reason it needs no epsilon.
    return handOff ? hitPos : (hitPos + dir * push);
}

// ═══════════════════════════════════════════════════════════════════════════
// TRI-PLANAR NORMAL MAPPING — whiteout blend
// ═══════════════════════════════════════════════════════════════════════════
// This is the piece that could not be built before material coordinates
// existed, and the reason is worth stating: a bump pattern nailed to the world
// while the liquid flows through it does not read as detail, it reads as a
// lighting error. The anchor had to come first.
//
// ★ Why not the ordinary tangent-frame path: an isosurface has no UVs and no
// tangents. Tri-planar gives three sets of UVs, so it needs THREE tangent
// frames and a rule for recombining them.
//
// ★★ WHITEOUT BLEND, not a weighted average of the three world-space normals.
// Averaging pulls every sample toward the geometric normal — the detail
// survives on the axis-aligned faces and quietly flattens everywhere the three
// projections meet, which is exactly where a viewer looks for shape. Whiteout
// (Golus) adds each tangent normal's XY into the geometric normal's other two
// components before blending, so overlapping detail REINFORCES instead of
// cancelling.
//
// ★★★ The axis SIGN matters and is the classic omission. Without flipping the
// tangent basis by sign(N), the two opposite faces of every axis get mirrored
// derivatives: one side of a droplet shows bumps and the other shows dents,
// from the same texture. It looks like the map is wrong rather than the frame.
// ═══════════════════════════════════════════════════════════════════════════
// ★★★ ONE PLANE PER SAMPLE, CHOSEN STOCHASTICALLY — not three blended.
// ═══════════════════════════════════════════════════════════════════════════
// Blending the three projections hides the SEAM but not the DECOMPOSITION: on
// an oblique surface the three stretch in different directions, so the pattern
// visibly splits into three sheared copies of itself. No blend weight fixes
// that, because all three copies are genuinely present.
//
// Picking ONE plane per sample with probability equal to its weight is
// unbiased — the expected value is exactly the blend — but the error becomes
// NOISE instead of a persistent ghost, and a path tracer already averages
// noise away. The seam converges out; the decomposition never would.
//
// ★★ It is also CHEAPER, which is unusual for a quality fix: one texture fetch
// instead of three. That matters most for the normal map, which was the
// heaviest consumer here at three fetches per hit.
//
// ★ THE SAME xi MUST BE USED FOR EVERY CHANNEL AT ONE HIT. Draw it once per
// hit and pass it down. Independent draws per channel would let albedo come
// from one projection while roughness came from another, at the same point on
// the surface — the channels would stop describing the same texel and the
// result would converge to a plausible-looking average of two materials, which
// is far harder to recognise as wrong than an obvious seam.
int triplanarPlane(vec3 N, float xi) {
    // Sharpened squared normal. The exponent sets how narrow the transition
    // is: 4 is the usual compromise — lower spreads the choice across a wider
    // band, higher makes the three regions nearly hard-edged.
    vec3 w = pow(abs(N), vec3(4.0));
    w /= max(w.x + w.y + w.z, 1e-6);
    if (xi < w.x) return 0;
    if (xi < w.x + w.y) return 1;
    return 2;
}

vec2 triplanarUV(int plane, vec3 p, vec2 scale, vec2 offset) {
    if (plane == 0) return p.zy * scale + offset;
    if (plane == 1) return p.xz * scale + offset;
    return p.xy * scale + offset;
}

vec3 triplanarNormal(uint texId, vec3 anchor, vec3 Ng, vec2 scale, vec2 offset,
                     float strength, float xi) {
    int plane = triplanarPlane(Ng, xi);

    vec3 t = texture(materialTextures[nonuniformEXT(texId)],
                     triplanarUV(plane, anchor, scale, offset)).rgb * 2.0 - 1.0;

    // Strength scales the tangent tilt, never the Z: scaling all three would
    // renormalise straight back to the same direction and the slider would do
    // nothing — a control that moves and changes nothing gets reported as a bug.
    t.xy *= strength;

    // ★★★ The axis SIGN matters and is the classic omission. Without flipping
    // the tangent basis by sign(N), the two opposite faces of every axis get
    // mirrored derivatives: one side of a droplet shows bumps and the other
    // shows dents, from the same texture. It looks like the map is wrong rather
    // than the frame.
    vec3 axisSign = sign(Ng);

    // Whiteout (Golus): fold the geometric normal in on the two axes this plane
    // does not own, so detail REINFORCES the shape instead of pulling every
    // sample back toward Ng — which is what a plain tangent add would do, and
    // it flattens exactly where a viewer looks for form.
    vec3 n;
    if (plane == 0) {
        t.z *= axisSign.x;
        n = vec3(t.xy + Ng.zy, abs(t.z) * Ng.x).zyx;
    } else if (plane == 1) {
        t.z *= axisSign.y;
        n = vec3(t.xy + Ng.xz, abs(t.z) * Ng.y).xzy;
    } else {
        t.z *= axisSign.z;
        n = vec3(t.xy + Ng.xy, abs(t.z) * Ng.z).xyz;
    }

    float len2 = dot(n, n);
    // Degenerate only if the sample cancels the geometric normal, which a valid
    // map cannot do — but a BLANK or wrongly-typed texture (an albedo bound into
    // the normal slot) can. Fall back to the geometric normal rather than
    // emitting a NaN that turns the surface black and looks like a geometry bug.
    return (len2 > 1e-12) ? (n * inversesqrt(len2)) : Ng;
}

vec4 triplanarTexel(uint texId, vec3 worldPos, vec3 N, vec2 scale, vec2 offset,
                    float xi) {
    return texture(materialTextures[nonuniformEXT(texId)],
                   triplanarUV(triplanarPlane(N, xi), worldPos, scale, offset));
}


vec3 sampleTransmittanceLUT(vec3 worldPos, vec3 sunDir) {
    if (worldData.w.mode != 2) return vec3(1.0);
    if (worldData.w.atmosphereHeight <= 0.0) return vec3(1.0);
    if (worldData.w._pad5 == 0) return vec3(1.0);
    float Rg = max(worldData.w.planetRadius, 1.0);
    vec3 p = worldPos + vec3(0.0, Rg, 0.0);
    float altitude = max(length(p) - Rg, 0.0);
    vec3 up = normalize(p);
    float cosTheta = dot(up, normalize(sunDir));
    float u = clamp((cosTheta + 0.2) / 1.2, 0.0, 1.0);
    float v = clamp(altitude / worldData.w.atmosphereHeight, 0.0, 1.0);
    return textureLod(atmosphereLUTs[0], vec2(u, v), 0.0).rgb;
}

vec3 sampleSkyAmbient(vec3 viewDir) {
    // Directional sky ambient: blend up-direction with view direction
    // so horizon/sun tint mixes into the medium more naturally.
    vec3 ambDir = normalize(mix(vec3(0.0, 1.0, 0.0), normalize(viewDir), 0.45));
    ambDir = normalize(ambDir + normalize(worldData.w.sunDir) * 0.15);

    if (worldData.w.mode == 2 && worldData.w._pad5 != 0) {
        float az = atan(ambDir.z, ambDir.x);
        if (az < 0.0) az += TWO_PI;
        float u = az / TWO_PI;
        float v = (1.0 - clamp(ambDir.y, -1.0, 1.0)) * 0.5;
        return textureLod(atmosphereLUTs[1], vec2(u, v), 0.0).rgb
             * (0.15 * worldData.w.atmosphereIntensity);
    }
    return worldData.w.sunColor * (0.15 * worldData.w.atmosphereIntensity);
}

// ============================================================
// Powder Effect for volumetric clouds (OptiX Parity)
// ============================================================
float gpu_powder_effect(float density, float cos_theta) {
    float powder = 1.0 - exp(-density * 2.0);
    float forward_bias = 0.5 + 0.5 * max(0.0, cos_theta);
    return powder * forward_bias;
}

// ============================================================
// Scene light sampling for volumes
// ============================================================
// The volume march used to treat every non-directional light as a bare point
// light: no cone, no shape. A spot in smoke therefore lit a sphere instead of a
// shaft — the one shot gas simulation exists for. These mirror the surface
// shader's sample_light_direction_gl / spot_light_falloff_gl so a light behaves
// the same whether it hits a wall or the smoke in front of it.
//
// LightData::position.w = type: 0 point, 1 directional, 2 area, 3 spot.

float volSpotFalloff(LightData light, vec3 wi) {
    // wi points FROM the shading position TOWARD the light, matching the
    // surface version's convention (it negates before the cone test).
    float cosTheta = dot(-wi, normalize(light.direction.xyz));
    float inner = light.params.z;
    float outer = light.direction.w;
    if (cosTheta < outer) return 0.0;
    if (cosTheta > inner) return 1.0;
    float t = (cosTheta - outer) / (inner - outer + 1e-6);
    return t * t;
}

// Direction, distance and geometric attenuation from a march sample toward a
// light. `attenuation` EXCLUDES the 1/r^2 term for positional lights: that term
// is integrated analytically over the march segment by volSegmentInvSqAverage
// below, because point-sampling it once per step is what makes light shafts
// depend on where the step boundary happened to land.
// Returns false when the light cannot contribute (behind the cone, degenerate).
bool volSampleLight(LightData light, vec3 pos, float ru, float rv,
                    out vec3 wi, out float dist, out float attenuation,
                    out bool inverseSquare) {
    int type = int(light.position.w + 0.5);
    attenuation = 1.0;
    inverseSquare = true;
    if (type == 1) {
        // Directional: no falloff, no closest approach.
        wi = normalize(light.direction.xyz);
        dist = 1e6;
        inverseSquare = false;
        return true;
    }
    if (type == 2) {
        // Area: stratified-enough single sample on the rectangle. Sampling the
        // shape (rather than its centre) is what gives soft shadow edges inside
        // the medium instead of a hard point-light core.
        float uOff = (ru - 0.5) * light.params.y;
        float vOff = (rv - 0.5) * light.params.z;
        vec3 lightSample = light.position.xyz +
                           light.area_u.xyz * uOff + light.area_v.xyz * vOff;
        vec3 L = lightSample - pos;
        dist = length(L);
        if (dist < 1e-4) return false;
        wi = L / dist;
        vec3 lightNormal = normalize(cross(light.area_u.xyz, light.area_v.xyz));
        // Emission is one-sided on the surface path; keep that here or a quad
        // light would glow into the medium behind it.
        float cosLight = max(dot(-wi, lightNormal), 0.0);
        if (cosLight <= 0.0) return false;
        attenuation = cosLight;
        return true;
    }
    // Point (0) and spot (3).
    vec3 L = light.position.xyz - pos;
    dist = length(L);
    if (dist < 1e-4) return false;
    wi = L / dist;
    if (type == 3) {
        float falloff = volSpotFalloff(light, wi);
        if (falloff < 1e-4) return false;
        attenuation = falloff;
    }
    return true;
}

// Average of 1/r^2 over the march segment [t0, t1] for a light at `lightPos`,
// i.e. (1/dt) * integral of dt' / |rayOrigin + rayDir*t' - lightPos|^2.
//
// This is the equiangular integral in closed form:
//   integral dt / (h^2 + (t - tc)^2) = (1/h) * atan((t - tc)/h)
// with tc the closest-approach parameter and h the perpendicular distance from
// the light to the ray line. A ray marcher cannot use equiangular DISTANCE
// sampling (its sample positions are fixed), but it can integrate the term that
// makes those samples inadequate — and 1/r^2 is that term. A light passing
// close to the ray no longer produces a result that depends on step size:
// point-sampling the falloff at the segment start either misses the peak
// entirely or lands on it and blows the segment up, which is why coarse steps
// gave banded, flickering shafts.
//
// The phase function and shadow transmittance are still evaluated once per
// segment. They vary smoothly; the singularity lives in 1/r^2.
float volSegmentInvSqAverage(vec3 rayOrigin, vec3 rayDir, vec3 lightPos,
                             float t0, float t1) {
    float dt = t1 - t0;
    if (dt <= 1e-9) return 0.0;
    vec3 toLight = lightPos - rayOrigin;
    float tc = dot(toLight, rayDir);              // closest approach along the ray
    float h2 = max(dot(toLight, toLight) - tc * tc, 0.0);
    float h = sqrt(h2);
    if (h < 1e-4) {
        // Ray passes (numerically) through the light: the integral is
        // 1/(t0-tc) - 1/(t1-tc), which still diverges if the segment contains
        // tc. Clamp with a small radius so a light sitting exactly on the ray
        // produces a bright but finite sample instead of an inf that poisons
        // the whole accumulation buffer.
        h = 1e-4;
        h2 = h * h;
    }
    float a0 = atan((t0 - tc) / h);
    float a1 = atan((t1 - tc) / h);
    return (a1 - a0) / (h * dt);
}

// ============================================================
// Henyey-Greenstein Phase Function
// ============================================================
float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return INV_4PI * (1.0 - g2) / (denom * sqrt(denom) + EPSILON);
}

// Dual-lobe HG phase function (matches OptiX implementation)
float dualLobeHG(float cosTheta, float g_forward, float g_back, float lobeMix) {
    float phaseForward = henyeyGreenstein(cosTheta, g_forward);
    float phaseBack    = henyeyGreenstein(cosTheta, g_back);
    return mix(phaseBack, phaseForward, lobeMix);
}


// ============================================================
// 3D Noise (procedural density for volume_type=1)
// ============================================================
float hash3D(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

vec3 hash3Gradient(vec3 p) {
    p = vec3(
        dot(p, vec3(127.1, 311.7, 74.7)),
        dot(p, vec3(269.5, 183.3, 246.1)),
        dot(p, vec3(113.5, 271.9, 124.6))
    );
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453);
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * f * (f * (f * 6.0 - vec3(15.0)) + vec3(10.0));

    float n000 = dot(hash3Gradient(i + vec3(0, 0, 0)), f - vec3(0, 0, 0));
    float n100 = dot(hash3Gradient(i + vec3(1, 0, 0)), f - vec3(1, 0, 0));
    float n010 = dot(hash3Gradient(i + vec3(0, 1, 0)), f - vec3(0, 1, 0));
    float n110 = dot(hash3Gradient(i + vec3(1, 1, 0)), f - vec3(1, 1, 0));
    float n001 = dot(hash3Gradient(i + vec3(0, 0, 1)), f - vec3(0, 0, 1));
    float n101 = dot(hash3Gradient(i + vec3(1, 0, 1)), f - vec3(1, 0, 1));
    float n011 = dot(hash3Gradient(i + vec3(0, 1, 1)), f - vec3(0, 1, 1));
    float n111 = dot(hash3Gradient(i + vec3(1, 1, 1)), f - vec3(1, 1, 1));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z) * 0.5 + 0.5;
}

float fbmNoise(vec3 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise3D(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

float proceduralCloudDensity(VkVolumeInstance vol, vec3 localPos) {
    vec3 span = max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
    vec3 normPos = clamp((localPos - vol.aabb_min) / span, vec3(0.0), vec3(1.0));
    float baseScale = max(vol.cloud_base_scale, 1.0);
    vec3 cloudCoord = vec3(
        normPos.x * baseScale + vol.cloud_offset_x,
        normPos.y * 1.35,
        normPos.z * baseScale + vol.cloud_offset_z);
    vec3 seedOffset = vec3(vol.cloud_seed * 0.137, vol.cloud_seed * 0.317, vol.cloud_seed * 0.719);
    cloudCoord += seedOffset;

    float coverage = clamp(vol.cloud_coverage, 0.0, 1.0);
    float detail = clamp(vol.cloud_detail, 0.0, 1.0);
    float erosion = clamp(vol.cloud_erosion, 0.0, 1.0);

    float warpX = fbmNoise(vec3(cloudCoord.x * 0.38, cloudCoord.y * 0.16, cloudCoord.z * 0.38) + vec3(11.0, 0.0, 7.0), 2) - 0.5;
    float warpZ = fbmNoise(vec3(cloudCoord.x * 0.38, cloudCoord.y * 0.16, cloudCoord.z * 0.38) + vec3(41.0, 3.0, 23.0), 2) - 0.5;
    vec3 warped = cloudCoord + vec3(warpX * 1.35, 0.0, warpZ * 1.35);

    float base = fbmNoise(vec3(warped.x * 0.52, warped.y * 0.28, warped.z * 0.52), 4);
    float billow = 1.0 - abs(fbmNoise(vec3(warped.x * 1.15, warped.y * 0.5, warped.z * 1.15) + vec3(17.0, 3.0, 11.0), 4) * 2.0 - 1.0);
    float detailNoise = fbmNoise(warped * mix(2.8, 7.0, detail) + vec3(31.0, 7.0, 19.0), 2);

    float puffy = smoothstep(0.32, 0.88, billow);
    float cumulus = clamp((vol.density_multiplier - 0.38) / 0.85, 0.0, 1.0);
    float shape = mix(base, base * 0.45 + puffy * 0.75, mix(0.55, 0.9, cumulus));
    shape -= detailNoise * mix(0.06, 0.28, erosion);

    float threshold = mix(0.80, 0.26, coverage) - cumulus * 0.08;
    float density = max((shape - threshold) / max(1.0 - threshold, 1e-4), 0.0);

    float bottom = smoothstep(mix(0.12, 0.04, cumulus), mix(0.42, 0.24, cumulus), normPos.y);
    float top = 1.0 - smoothstep(mix(0.72, 0.82, cumulus), 1.02, normPos.y);
    float dome = mix(1.0, smoothstep(0.08, 0.58, normPos.y) * (1.0 - smoothstep(0.88, 1.04, normPos.y)) + 0.25, cumulus);
    float heightProfile = bottom * top * dome;

    vec3 edge = vec3(0.5) - abs(normPos - vec3(0.5));
    float edgeFalloff = smoothstep(0.0, max(vol.cloud_edge_fade, 0.02), min(edge.x, edge.z));
    return density * mix(density, sqrt(max(density, 0.0)), cumulus * 0.55) * heightProfile * edgeFalloff * mix(4.6, 3.4, cumulus);
}

// ============================================================
// NanoVDB GLSL Setup
// ============================================================
#define PNANOVDB_GLSL
#define PNANOVDB_BUF_CUSTOM

struct pnanovdb_buf_t {
    uint64_t address;
};

// We define a buffer block matching NanoVDB scalar layout
layout(buffer_reference, std430, buffer_reference_align=4) buffer NanoVDBBlock {
    uint data[];
};

uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byte_offset) {
    NanoVDBBlock blk = NanoVDBBlock(buf.address);
    return blk.data[byte_offset >> 2];
}

uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byte_offset) {
    NanoVDBBlock blk = NanoVDBBlock(buf.address);
    uint idx = byte_offset >> 2;
    return uvec2(blk.data[idx], blk.data[idx + 1]);
}

void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint byte_offset, uint value) {}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint byte_offset, uvec2 value) {}

#include "PNanoVDB.h"

// ── Persistent-accessor trilinear sampler ─────────────────────────────────
// Caller must have pre-fetched buf/mapH/rootH and called pnanovdb_readaccessor_init
// once. The accessor caches the last walked root→internal→leaf path; reusing it
// across spatially-coherent march steps (typical step ≪ leaf size = 8 voxels)
// skips the tree walk on cache hits. This collapses NanoVDB sampling cost in the
// hot loop from O(tree-depth) to O(1) for the common case.
float sampleNanoVDBFloatTrilinearAcc(
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    vec3 worldPos)
{
    pnanovdb_vec3_t wPos = pnanovdb_vec3_uniform(0.0);
    wPos.x = worldPos.x; wPos.y = worldPos.y; wPos.z = worldPos.z;
    pnanovdb_vec3_t iPos = pnanovdb_map_apply_inverse(buf, mapH, wPos);

    vec3 idxPos = vec3(iPos.x, iPos.y, iPos.z);
    vec3 p0 = floor(idxPos);
    vec3 frac = fract(idxPos);

    float d[8];
    for (int i = 0; i < 8; ++i) {
        pnanovdb_coord_t coord;
        coord.x = int(p0.x) + ((i & 1) != 0 ? 1 : 0);
        coord.y = int(p0.y) + ((i & 2) != 0 ? 1 : 0);
        coord.z = int(p0.z) + ((i & 4) != 0 ? 1 : 0);
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, coord);
        d[i] = pnanovdb_read_float(buf, addr);
    }

    float dx00 = mix(d[0], d[1], frac.x);
    float dx10 = mix(d[2], d[3], frac.x);
    float dx01 = mix(d[4], d[5], frac.x);
    float dx11 = mix(d[6], d[7], frac.x);
    float dxy0 = mix(dx00, dx10, frac.y);
    float dxy1 = mix(dx01, dx11, frac.y);
    return mix(dxy0, dxy1, frac.z);
}

// Live gas domains expose their Vulkan compute fields directly to RT. The
// dense grid stores cell-centred floats in x-major order.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer DenseGasFloatGrid {
    float values[];
};

float sampleDenseGasFloat(uint64_t address, VkVolumeInstance vol, vec3 localPos) {
    if (address == 0) return 0.0;

    ivec3 resolution = ivec3(
        int(vol._ext_reserved[0] + 0.5),
        int(vol._ext_reserved[1] + 0.5),
        int(vol._ext_reserved[2] + 0.5));
    if (any(lessThanEqual(resolution, ivec3(0)))) return 0.0;

    float voxelSize = max(vol.voxel_size, 1e-6);
    vec3 gridOrigin = vec3(
        vol._ext_reserved[3],
        vol._ext_reserved[4],
        vol._ext_reserved[5]);
    vec3 gridPos = (localPos - gridOrigin) / voxelSize - vec3(0.5);
    if (any(lessThan(gridPos, vec3(-0.5))) ||
        any(greaterThan(gridPos, vec3(resolution) - vec3(0.5)))) {
        return 0.0;
    }

    ivec3 p0 = clamp(ivec3(floor(gridPos)), ivec3(0), resolution - ivec3(1));
    ivec3 p1 = min(p0 + ivec3(1), resolution - ivec3(1));
    vec3 f = clamp(gridPos - vec3(p0), vec3(0.0), vec3(1.0));
    DenseGasFloatGrid grid = DenseGasFloatGrid(address);
    int xyStride = resolution.x * resolution.y;

    int i000 = p0.x + p0.y * resolution.x + p0.z * xyStride;
    int i100 = p1.x + p0.y * resolution.x + p0.z * xyStride;
    int i010 = p0.x + p1.y * resolution.x + p0.z * xyStride;
    int i110 = p1.x + p1.y * resolution.x + p0.z * xyStride;
    int i001 = p0.x + p0.y * resolution.x + p1.z * xyStride;
    int i101 = p1.x + p0.y * resolution.x + p1.z * xyStride;
    int i011 = p0.x + p1.y * resolution.x + p1.z * xyStride;
    int i111 = p1.x + p1.y * resolution.x + p1.z * xyStride;

    float z0 = mix(
        mix(grid.values[i000], grid.values[i100], f.x),
        mix(grid.values[i010], grid.values[i110], f.x),
        f.y);
    float z1 = mix(
        mix(grid.values[i001], grid.values[i101], f.x),
        mix(grid.values[i011], grid.values[i111], f.x),
        f.y);
    return mix(z0, z1, f.z);
}

// ═══════════════════════════════════════════════════════════════════════════
// MATERIAL COORDINATE (UVW) SAMPLING
// ═══════════════════════════════════════════════════════════════════════════
// The isosurface has no UVs and cannot have them: there is no mesh to unwrap,
// and the surface is rebuilt from the field every frame, so there is nothing
// stable for a UV to be attached to. What CAN be stable is a coordinate carried
// by the liquid itself — each particle remembers where it was born, and that
// value is gathered onto a grid the shader can sample.
//
// So the projection below is still tri-planar (no tangent frame exists here
// either), but it is anchored in MATERIAL space instead of world space. The
// difference only shows once the liquid moves: a pour now carries its texture
// instead of sliding through a stationary pattern.
//
// ★ Identity at rest. The coordinate is seeded with the birth POSITION in world
// units, so for a body that has not moved, uvw == worldPos and this returns
// exactly what the old world-anchored path returned. That is deliberate: it
// makes "did this change anything it should not have?" answerable by rendering
// a still tank and diffing, rather than by judgement.
// ★★★ THE BUFFER HOLDS A DISPLACEMENT, NOT A COORDINATE. Cell c stores
// (uvw - centre(c)); the coordinate is rebuilt here as worldPos + d.
//
// Storing the sum instead would cap the WHOLE coordinate at the sim voxel size,
// because the position term — which is full-resolution and free right here in
// the shader — would have had to survive a trip through the grid. That is
// exactly what "material mode looks like one pixel per cell" was. Only the
// deformation is grid-limited now, and deformation is smooth.
//
// ★ Consequence worth knowing before debugging anything here: in still liquid
// d is identically zero, so this returns worldPos EXACTLY. A resting tank
// therefore has to match World mode pixel for pixel.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer MaterialResidualGrid {
    float values[];      // interleaved xyz displacement per cell
};

// Returns false when this volume publishes no coordinate field. Callers MUST
// then fall back to worldPos. Returning a zero vector instead would map the
// entire surface to one texel — a flat-tinted liquid, which looks like a
// material authoring mistake rather than like missing data.
bool sampleMaterialCoord(VkVolumeInstance vol, vec3 worldPos, out vec3 coord) {
    coord = worldPos;
    if (vol.uvw_residual_address == 0) return false;

    ivec3 res = ivec3(int(vol.uvw_dim[0] + 0.5),
                      int(vol.uvw_dim[1] + 0.5),
                      int(vol.uvw_dim[2] + 0.5));
    if (any(lessThanEqual(res, ivec3(0)))) return false;
    if (vol.uvw_voxel <= 0.0) return false;

    // ★★★ INDEXED IN WORLD SPACE, THROUGH THE GRID'S OWN ORIGIN AND CELL SIZE.
    //
    // This used to go through inv_transform and normalise by aabb_min/aabb_max,
    // by analogy with the density sampler. That analogy is false. On a live
    // fluid, aabb is the tight ACTIVE box of the dense/SDF grid — padded by a
    // cell and recomputed as the liquid moves — while this buffer spans the
    // whole SIM grid at the sim's own resolution. Mapping the first onto the
    // second scaled the field by the ratio of their extents and offset it by the
    // difference of their origins, and since the active box follows the liquid,
    // both errors changed every frame.
    //
    // ★ The symptom was NOT "wrong texture position", which somebody would have
    // reported immediately. A slowly varying scale error reads as the pattern
    // smearing along the flow and swimming over the surface — i.e. it looks like
    // a quality problem in the coordinate, and it survived a whole round of
    // genuine quality work sitting underneath it.
    //
    // The producer walks the sim grid in world space, so the inverse of that
    // walk is world space too. No transform: if the domain moves, its grid
    // origin moves with it and arrives here already updated.
    vec3 uvwOrigin = vec3(vol.uvw_origin[0], vol.uvw_origin[1], vol.uvw_origin[2]);
    vec3 gridPos = (worldPos - uvwOrigin) / vol.uvw_voxel - vec3(0.5);

    // CLAMP rather than reject out-of-range. The producer extrapolated the field
    // a few voxels past the supported region precisely so the surface — which
    // sits at the edge of it — reads valid data; failing out here at the domain
    // wall would undo that and re-introduce a world-anchored ring exactly where
    // liquid touches the boundary.
    ivec3 p0 = clamp(ivec3(floor(gridPos)), ivec3(0), res - ivec3(1));
    ivec3 p1 = min(p0 + ivec3(1), res - ivec3(1));
    vec3  f  = clamp(gridPos - vec3(p0), vec3(0.0), vec3(1.0));

    // ★ Quintic ease on the interpolation fraction (Perlin's 6t^5-15t^4+10t^3).
    // Plain trilinear is only C0: the value is continuous across a cell boundary
    // but its DERIVATIVE jumps, and the derivative is what a texture lookup
    // actually rides. Easing makes it vanish at the boundaries instead.
    //
    // ★★★ THIS IS ONLY LEGAL ON A RESIDUAL, and getting that backwards is what
    // produced the reported blockiness. Applied to an ABSOLUTE coordinate the
    // ease modulates the identity gradient itself: d(coord)/d(world) becomes
    // s'(f), which runs from 0.0 at every cell face to 1.875 at every cell
    // centre. The texture is then frozen on the cell boundaries and compressed
    // in the middles — a rectangular quilt of plateaus with pinched seams,
    // at exactly one tile per cell. The seam-removal tool was the seam.
    //
    // On a residual it does what it was meant to do: the identity gradient is
    // added outside this function and is untouched, so easing only softens the
    // small deformation term.
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    MaterialResidualGrid g = MaterialResidualGrid(vol.uvw_residual_address);
    int xy = res.x * res.y;
    int i000 = (p0.x + p0.y * res.x + p0.z * xy) * 3;
    int i100 = (p1.x + p0.y * res.x + p0.z * xy) * 3;
    int i010 = (p0.x + p1.y * res.x + p0.z * xy) * 3;
    int i110 = (p1.x + p1.y * res.x + p0.z * xy) * 3;
    int i001 = (p0.x + p0.y * res.x + p1.z * xy) * 3;
    int i101 = (p1.x + p0.y * res.x + p1.z * xy) * 3;
    int i011 = (p0.x + p1.y * res.x + p1.z * xy) * 3;
    int i111 = (p1.x + p1.y * res.x + p1.z * xy) * 3;

    vec3 c000 = vec3(g.values[i000], g.values[i000 + 1], g.values[i000 + 2]);
    vec3 c100 = vec3(g.values[i100], g.values[i100 + 1], g.values[i100 + 2]);
    vec3 c010 = vec3(g.values[i010], g.values[i010 + 1], g.values[i010 + 2]);
    vec3 c110 = vec3(g.values[i110], g.values[i110 + 1], g.values[i110 + 2]);
    vec3 c001 = vec3(g.values[i001], g.values[i001 + 1], g.values[i001 + 2]);
    vec3 c101 = vec3(g.values[i101], g.values[i101 + 1], g.values[i101 + 2]);
    vec3 c011 = vec3(g.values[i011], g.values[i011 + 1], g.values[i011 + 2]);
    vec3 c111 = vec3(g.values[i111], g.values[i111 + 1], g.values[i111 + 2]);

    vec3 z0 = mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y);
    vec3 z1 = mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y);

    // worldPos carries the full-resolution identity; the grid only bends it.
    coord = worldPos + mix(z0, z1, f.z);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPOSITION — which materials this point of the liquid is made of.
// ═══════════════════════════════════════════════════════════════════════════
// Per cell the producer stores the two dominant material slots and the weight
// of the second. Both slots are 1-based ids; 0 means "the built-in dielectric",
// which is a real choice for a substance and not an absence.
//
// ★★★ ONLY THE WEIGHT IS INTERPOLATED. An index is a NAME: the value halfway
// between material 2 and material 4 is material 3, an unrelated material that
// would appear as a band along every boundary. So the slots come from the
// NEAREST cell and only the blend fraction is filtered. In a binary mixture —
// the ordinary case — the pair is the same everywhere the mixture exists, so
// nothing is lost; with three or more, the pair switches at a cell boundary
// while the fraction stays smooth.
//
// ★ Falls back to the domain material when nothing is published, which is what
// makes a single-material domain cost exactly what it did before.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer CompositionGrid {
    float values[];      // interleaved (slotA, slotB, weightB) per cell
};

void sampleComposition(VkVolumeInstance vol, vec3 worldPos,
                       inout float slotA, out float slotB, out float weightB) {
    slotB = slotA;
    weightB = 0.0;
    if (vol.composition_address == 0) return;
    if (vol.uvw_voxel <= 0.0) return;

    ivec3 res = ivec3(int(vol.uvw_dim[0] + 0.5),
                      int(vol.uvw_dim[1] + 0.5),
                      int(vol.uvw_dim[2] + 0.5));
    if (any(lessThanEqual(res, ivec3(0)))) return;

    // Same world-space indexing as the residual field — same grid, same origin,
    // same cell size. Sharing the placement is what guarantees the mixture and
    // the coordinate describe the same point.
    vec3 uvwOrigin = vec3(vol.uvw_origin[0], vol.uvw_origin[1], vol.uvw_origin[2]);
    vec3 gridPos = (worldPos - uvwOrigin) / vol.uvw_voxel - vec3(0.5);

    ivec3 p0 = clamp(ivec3(floor(gridPos)), ivec3(0), res - ivec3(1));
    ivec3 p1 = min(p0 + ivec3(1), res - ivec3(1));
    vec3  f  = clamp(gridPos - vec3(p0), vec3(0.0), vec3(1.0));

    CompositionGrid g = CompositionGrid(vol.composition_address);
    int xy = res.x * res.y;

    // Nearest cell decides WHICH two materials; see the note above.
    ivec3 pn = ivec3(greaterThan(f, vec3(0.5))) * (p1 - p0) + p0;
    int   ni = (pn.x + pn.y * res.x + pn.z * xy) * 3;
    slotA = g.values[ni + 0];
    slotB = g.values[ni + 1];

    // The weight is a quantity and IS interpolated, so the transition between
    // two substances is smooth rather than stepping at cell boundaries.
    int i000 = (p0.x + p0.y * res.x + p0.z * xy) * 3 + 2;
    int i100 = (p1.x + p0.y * res.x + p0.z * xy) * 3 + 2;
    int i010 = (p0.x + p1.y * res.x + p0.z * xy) * 3 + 2;
    int i110 = (p1.x + p1.y * res.x + p0.z * xy) * 3 + 2;
    int i001 = (p0.x + p0.y * res.x + p1.z * xy) * 3 + 2;
    int i101 = (p1.x + p0.y * res.x + p1.z * xy) * 3 + 2;
    int i011 = (p0.x + p1.y * res.x + p1.z * xy) * 3 + 2;
    int i111 = (p1.x + p1.y * res.x + p1.z * xy) * 3 + 2;

    float z0 = mix(mix(g.values[i000], g.values[i100], f.x),
                   mix(g.values[i010], g.values[i110], f.x), f.y);
    float z1 = mix(mix(g.values[i001], g.values[i101], f.x),
                   mix(g.values[i011], g.values[i111], f.x), f.y);
    weightB = clamp(mix(z0, z1, f.z), 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// THE ONE ANCHOR. Every isosurface pattern goes through here — tri-planar
// textures, the resin interior march, the porosity lattice, and the opacity
// mask. Three spaces, ONE place they diverge.
// ═══════════════════════════════════════════════════════════════════════════
// ★ That single divergence point is the whole design. The consumers do not know
// which space they are in and must not: the moment two of them resolve the
// space separately, they are free to disagree, and a resin interior anchored
// differently from the albedo painted over it reads as "the shading is a bit
// off" rather than as a coordinate bug.
//
//   COORD_MATERIAL - the parcel's own coordinate. Carried BY the liquid, so a
//                    pour takes its pattern with it. Identity for anything that
//                    has not moved, so it is also the safe default.
//   COORD_DOMAIN   - the container's local space. Travels with a carried vessel
//                    while the liquid flows THROUGH the pattern: the right
//                    answer for something painted on the tank, and the wrong
//                    one for something that belongs to the substance.
//   COORD_WORLD    - nailed to the room. The behaviour that shipped before
//                    material coordinates existed. Kept because it is a real
//                    look (a projected pattern), and because it is the escape
//                    hatch from the material coordinate's one true weakness —
//                    it STRETCHES with the flow, so a violently deformed splash
//                    eventually maps chaotically and an artist needs a way out
//                    that is not "animate less".
//
// The bool form above stays separate rather than folded in here because "there
// is no coordinate field" and "the coordinate happens to equal the world
// position" are different facts, and a caller that ever needs to tell them
// apart must not have to re-derive it from a value that looks identical.
const uint COORD_MATERIAL = 0u;
const uint COORD_DOMAIN   = 1u;
const uint COORD_WORLD    = 2u;

vec3 materialAnchor(VkVolumeInstance vol, vec3 worldPos) {
    uint space = uint(clamp(vol._accel_reserved[3], 0.0, 2.0) + 0.5);

    if (space == COORD_WORLD) return worldPos;

    if (space == COORD_DOMAIN) {
        // Volume-local, through the same inverse transform the density sampler
        // uses. NOT normalised to [0,1]: keeping it in world-sized units means
        // uv_scale/pore_scale stay "world units per tile" in every space, so
        // switching space re-anchors the pattern without also resizing it.
        vec3 localPos;
        localPos.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y
                   + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
        localPos.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y
                   + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
        localPos.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y
                   + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];
        return localPos;
    }

    // COORD_MATERIAL. Falls back to worldPos when no field is published, which
    // is what makes an unbuilt or unavailable coordinate degrade to the old
    // behaviour instead of collapsing the surface onto the origin.
    vec3 c;
    sampleMaterialCoord(vol, worldPos, c);
    return c;
}

layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer DenseGasMajorantGrid {
    float blocks[];
};

// [0] = emitter count (may exceed capacity; the producer keeps counting so the
// overflow is visible), [1..] = block indices into the majorant grid.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer DenseGasEmissiveList {
    uint entries[];
};

// Empty-space skip for LIVE DENSE gas (volume_type 4 / source_type 5), which
// has no NanoVDB hierarchy to walk: a dense domain is a solid box of cells and
// the march used to pay for every step of it even when the smoke occupied a
// corner. sim_gas_majorant.comp reduces the density field to one maximum per
// 8^3 block; a block whose maximum is at or below the authored cutoff cannot
// contribute, so the ray jumps straight to that block's exit.
//
// Returns the distance to advance along `worldDir`, or 0 when the current block
// has content (or no majorant is published — then this must return 0 so the
// caller marches normally; treating a missing majorant as empty would delete
// the smoke rather than merely slow the render down).
//
// The reduction already covers one cell past each far face, so trilinear
// reconstruction at the boundary of a skipped block still reads zero.
float denseGasEmptyBlockStep(VkVolumeInstance vol, vec3 worldPos, vec3 worldDir,
                             float cutoff) {
    if (vol.majorant_address == 0 || vol.majorant_block < 1.0) return 0.0;

    ivec3 bdim = ivec3(vol.majorant_dim[0] + 0.5,
                       vol.majorant_dim[1] + 0.5,
                       vol.majorant_dim[2] + 0.5);
    if (any(lessThanEqual(bdim, ivec3(0)))) return 0.0;

    // Same object-space mapping the dense sampler uses, so a block index here
    // addresses exactly the cells sampleDenseGasFloat would read.
    vec3 localPos;
    localPos.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11];
    vec3 localDir;
    localDir.x = vol.inv_transform[0]*worldDir.x + vol.inv_transform[1]*worldDir.y + vol.inv_transform[2]*worldDir.z;
    localDir.y = vol.inv_transform[4]*worldDir.x + vol.inv_transform[5]*worldDir.y + vol.inv_transform[6]*worldDir.z;
    localDir.z = vol.inv_transform[8]*worldDir.x + vol.inv_transform[9]*worldDir.y + vol.inv_transform[10]*worldDir.z;

    float voxelSize = max(vol.voxel_size, 1e-6);
    vec3 gridOrigin = vec3(vol._ext_reserved[3], vol._ext_reserved[4], vol._ext_reserved[5]);
    float blockWorld = vol.majorant_block * voxelSize;   // block edge in object units
    vec3 blockPos = (localPos - gridOrigin) / blockWorld;
    ivec3 b = ivec3(floor(blockPos));
    if (any(lessThan(b, ivec3(0))) || any(greaterThanEqual(b, bdim))) return 0.0;

    DenseGasMajorantGrid mg = DenseGasMajorantGrid(vol.majorant_address);
    int bi = b.x + b.y * bdim.x + b.z * bdim.x * bdim.y;
    // The stored maximum is RAW grid density. sampleDensityAcc rejects on the
    // REMAPPED value before applying density_multiplier, so remap the block
    // maximum the same way and compare the same quantity — multiplying here
    // would reject blocks the sampler would have kept (or worse, skip blocks it
    // would have shaded) whenever the multiplier is not 1.
    float blockMax = mg.blocks[bi];
    float remapped = max((blockMax - vol.density_remap_low) /
                         max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0);
    if (remapped > cutoff) return 0.0;   // block has content — march it

    // Slab exit of this block, in object units (== world units for a rigid
    // transform; a scaled volume errs on the short side, which only costs a
    // redundant sample).
    vec3 bmin = gridOrigin + vec3(b) * blockWorld;
    vec3 bmax = bmin + vec3(blockWorld);
    float tExit = 1e30;
    for (int axis = 0; axis < 3; ++axis) {
        float d = localDir[axis];
        if (abs(d) < 1e-9) continue;
        float bound = d > 0.0 ? bmax[axis] : bmin[axis];
        tExit = min(tExit, (bound - localPos[axis]) / d);
    }
    if (tExit >= 1e29 || tExit <= 0.0) return 0.0;
    // Land just inside the next block so the following iteration cannot pick
    // this same block again and stall the march.
    return tExit + blockWorld * 0.01;
}

// Conservative NanoVDB hierarchy skip. Leaf voxels cannot be skipped at this
// hierarchy level, so avoid the second is_active lookup when dim <= 1.
// Inactive coarse tiles advance to one voxel before their exit so trilinear
// reconstruction still sees density across the active boundary.
float nanoEmptyTileStep(
    VkVolumeInstance vol,
    vec3 worldPos,
    vec3 worldDir,
    float baseStep,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    out uint skipKind)
{
    skipKind = 0u;
    if (buf.address == 0) return baseStep;

    vec3 localP;
    localP.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y
             + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3] - vol.pivot_offset[0];
    localP.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y
             + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7] - vol.pivot_offset[1];
    localP.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y
             + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11] - vol.pivot_offset[2];

    vec3 worldP1 = worldPos + worldDir;
    vec3 localP1;
    localP1.x = vol.inv_transform[0]*worldP1.x + vol.inv_transform[1]*worldP1.y
              + vol.inv_transform[2]*worldP1.z + vol.inv_transform[3] - vol.pivot_offset[0];
    localP1.y = vol.inv_transform[4]*worldP1.x + vol.inv_transform[5]*worldP1.y
              + vol.inv_transform[6]*worldP1.z + vol.inv_transform[7] - vol.pivot_offset[1];
    localP1.z = vol.inv_transform[8]*worldP1.x + vol.inv_transform[9]*worldP1.y
              + vol.inv_transform[10]*worldP1.z + vol.inv_transform[11] - vol.pivot_offset[2];

    pnanovdb_vec3_t p0 = pnanovdb_vec3_uniform(0.0);
    p0.x = localP.x; p0.y = localP.y; p0.z = localP.z;
    pnanovdb_vec3_t p1 = pnanovdb_vec3_uniform(0.0);
    p1.x = localP1.x; p1.y = localP1.y; p1.z = localP1.z;
    pnanovdb_vec3_t ip0 = pnanovdb_map_apply_inverse(buf, mapH, p0);
    pnanovdb_vec3_t ip1 = pnanovdb_map_apply_inverse(buf, mapH, p1);
    vec3 idx = vec3(ip0.x, ip0.y, ip0.z);
    vec3 idxDir = vec3(ip1.x - ip0.x, ip1.y - ip0.y, ip1.z - ip0.z);

    pnanovdb_coord_t ijk;
    ijk.x = int(floor(idx.x)); ijk.y = int(floor(idx.y)); ijk.z = int(floor(idx.z));
    uint dim = pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk);
    if (dim <= 1u) return baseStep;
    bool regionIsActive = pnanovdb_readaccessor_is_active(
        PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk);
    if (regionIsActive) return baseStep;

    vec3 tileMin = floor(idx / float(dim)) * float(dim);
    vec3 tileMax = tileMin + vec3(float(dim));
    vec3 boundary = mix(tileMin, tileMax, greaterThan(idxDir, vec3(0.0)));
    vec3 axisT = vec3(1e30);
    if (abs(idxDir.x) > 1e-8) axisT.x = (boundary.x - idx.x) / idxDir.x;
    if (abs(idxDir.y) > 1e-8) axisT.y = (boundary.y - idx.y) / idxDir.y;
    if (abs(idxDir.z) > 1e-8) axisT.z = (boundary.z - idx.z) / idxDir.z;
    float tileExit = min(axisT.x, min(axisT.y, axisT.z));
    if (!(tileExit > baseStep)) return baseStep;
    skipKind = 1u;
    return max(baseStep, tileExit - max(vol.voxel_size, baseStep));
}

// Trilinear interpolation of a FloatGrid
float sampleNanoVDBFloatTrilinear(uint64_t gridAddr, vec3 worldPos) {
    if (gridAddr == 0) return 0.0;
    
    pnanovdb_buf_t buf;
    buf.address = gridAddr;
    
    // Get Handles
    pnanovdb_grid_handle_t gridH; gridH.address.byte_offset = 0u;
    pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(buf, gridH);
    pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(buf, treeH);
    pnanovdb_map_handle_t  mapH  = pnanovdb_grid_get_map(buf, gridH);
    
    // Init Accessor
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, rootH);
    
    // World to Index mapped
    pnanovdb_vec3_t wPos = pnanovdb_vec3_uniform(0.0);
    wPos.x = worldPos.x; wPos.y = worldPos.y; wPos.z = worldPos.z;
    pnanovdb_vec3_t iPos = pnanovdb_map_apply_inverse(buf, mapH, wPos);
    
    vec3 idxPos = vec3(iPos.x, iPos.y, iPos.z);
    
    vec3 p0 = floor(idxPos);
    vec3 frac = fract(idxPos);
    
    float d[8];
    for (int i = 0; i < 8; ++i) {
        pnanovdb_coord_t coord;
        coord.x = int(p0.x) + ((i & 1) != 0 ? 1 : 0);
        coord.y = int(p0.y) + ((i & 2) != 0 ? 1 : 0);
        coord.z = int(p0.z) + ((i & 4) != 0 ? 1 : 0);
        
        // Fast leaf-level read
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, coord);
        d[i] = pnanovdb_read_float(buf, addr);
    }
    
    float dx00 = mix(d[0], d[1], frac.x);
    float dx10 = mix(d[2], d[3], frac.x);
    float dx01 = mix(d[4], d[5], frac.x);
    float dx11 = mix(d[6], d[7], frac.x);
    
    float dxy0 = mix(dx00, dx10, frac.y);
    float dxy1 = mix(dx01, dx11, frac.y);
    
    return mix(dxy0, dxy1, frac.z);
}

// ============================================================
// Blackbody RGB — Kim et al. approximation (matches OptiX blackbody_to_rgb)
// ============================================================
vec3 blackbodyToRGB(float kelvin) {
    kelvin = clamp(kelvin, 1000.0, 40000.0);
    float temp = kelvin / 100.0;
    float r, g, b;
    // Red
    if (temp <= 66.0) { r = 1.0; }
    else { r = clamp(329.698727446 * pow(temp - 60.0, -0.1332047592) / 255.0, 0.0, 1.0); }
    // Green
    if (temp <= 66.0) { g = clamp((99.4708025861 * log(temp) - 161.1195681661) / 255.0, 0.0, 1.0); }
    else              { g = clamp(288.1221695283 * pow(temp - 60.0, -0.0755148492) / 255.0, 0.0, 1.0); }
    // Blue
    if (temp >= 66.0)      { b = 1.0; }
    else if (temp <= 19.0) { b = 0.0; }
    else { b = clamp((138.5177312231 * log(temp - 10.0) - 305.0447927307) / 255.0, 0.0, 1.0); }
    return vec3(r, g, b);
}

// OptiX parity: preserve hue while limiting tiny, extremely hot cells that
// otherwise dominate rare indirect paths and show up as volume fireflies.
vec3 clampVolumeRadiance(vec3 c, float maxLuma) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return (luma > maxLuma && luma > 1e-6) ? c * (maxLuma / luma) : c;
}

// ============================================================
// Color Ramp — linear interpolation over stop list
// ============================================================
vec3 sampleColorRamp(VkVolumeInstance vol, float t) {
    if (vol.ramp_stop_count == 0) return vec3(1.0);
    vec3 c0 = vec3(vol.ramp_colors_r[0], vol.ramp_colors_g[0], vol.ramp_colors_b[0]);
    if (t <= vol.ramp_positions[0]) return c0;
    int last = vol.ramp_stop_count - 1;
    vec3 cN = vec3(vol.ramp_colors_r[last], vol.ramp_colors_g[last], vol.ramp_colors_b[last]);
    if (t >= vol.ramp_positions[last]) return cN;
    for (int i = 1; i < vol.ramp_stop_count; ++i) {
        if (t < vol.ramp_positions[i]) {
            float f = (t - vol.ramp_positions[i-1]) / max(vol.ramp_positions[i] - vol.ramp_positions[i-1], 1e-6);
            vec3 a = vec3(vol.ramp_colors_r[i-1], vol.ramp_colors_g[i-1], vol.ramp_colors_b[i-1]);
            vec3 b = vec3(vol.ramp_colors_r[i],   vol.ramp_colors_g[i],   vol.ramp_colors_b[i]);
            return mix(a, b, f);
        }
    }
    return cN;
}

// ============================================================
// Temperature Sampling — secondary NanoVDB grid via vdb_temp_address
// ============================================================
float sampleTemperature(VkVolumeInstance vol, vec3 worldPos) {
    if (vol.vdb_temp_address == 0) return 0.0;
    vec3 localPos;
    localPos.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y
               + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y
               + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y
               + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11];
    
    // Safety check bound box instead of 0.5 cube
    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) return 0.0;
    
    // Pivot offset correction (OptiX parity)
    localPos.x -= vol.pivot_offset[0];
    localPos.y -= vol.pivot_offset[1];
    localPos.z -= vol.pivot_offset[2];

    if (vol.volume_type == 4 && vol.source_type == 5) {
        // Simulation heat is normalized; retain the old live-VDB Kelvin scale.
        return sampleDenseGasFloat(vol.vdb_temp_address, vol, localPos) * 3000.0;
    }

    return sampleNanoVDBFloatTrilinear(vol.vdb_temp_address, localPos);
}

// Combustion reaction rate (GridFluid's bounded `interaction` field) at a world
// position. Live dense gas only. Uses exactly the transform + pivot chain
// sampleTemperature applies, so the reaction lines up with the temperature it
// modulates instead of being offset by the pivot on a moved domain.
float sampleFlame(VkVolumeInstance vol, vec3 worldPos) {
    if (vol.flame_address == 0) return 0.0;
    if (!(vol.volume_type == 4 && vol.source_type == 5)) return 0.0;
    vec3 localPos;
    localPos.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y
               + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y
               + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y
               + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11];
    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) return 0.0;
    localPos.x -= vol.pivot_offset[0];
    localPos.y -= vol.pivot_offset[1];
    localPos.z -= vol.pivot_offset[2];
    return clamp(sampleDenseGasFloat(vol.flame_address, vol, localPos), 0.0, 1.0);
}

float applyMaterialDensityNoise(VkVolumeInstance vol, vec3 localPos, float density) {
    if (vol._ext_reserved[7] < 0.5 || density <= 0.0) return density;
    vec3 extent = max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
    vec3 p = (localPos - vol.aabb_min) / extent;
    float scale = max(vol._ext_reserved[8], 0.001);
    float strength = clamp(vol._ext_reserved[9], 0.0, 1.0);
    int detail = clamp(int(vol._ext_reserved[10] + 0.5), 1, 8);
    float seed = vol._ext_reserved[11];
    vec3 seedOffset = vec3(seed * 0.1031, seed * 0.11369, seed * 0.13787);
    float field = fbmNoise(p * scale + seedOffset, detail);
    return density * mix(1.0, field, strength);
}

// ── Persistent-accessor density sampler ───────────────────────────────────
// Same dispatch as sampleDensity but routes volume_type==2 NanoVDB reads through
// the caller's pre-initialized accessor. Other volume types (homogeneous, procedural,
// cloud) ignore the accessor params. Caller must guarantee buf/mapH are valid when
// vol.volume_type == 2 && vol.vdb_grid_address != 0; otherwise the accessor args are
// untouched.
float sampleDensityAccMode(
    VkVolumeInstance vol,
    vec3 worldPos,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    bool applyOpticalControls)
{
    vec3 localPos;
    localPos.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y
               + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y
               + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y
               + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];

    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) {
        return 0.0;
    }

    float density = 1.0;

    if (vol.volume_type == 0) {
        density = 1.0;
    } else if (vol.volume_type == 1) {
        vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
        vec3 noiseCoord = normPos * vol.noise_scale;
        density = fbmNoise(noiseCoord, 4);
        vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
        float edgeFalloff = min(min(edgeDist.x, edgeDist.y), edgeDist.z);
        density *= smoothstep(0.0, 0.1, edgeFalloff);
    } else if (vol.volume_type == 2) {
        if (vol.vdb_grid_address != 0) {
            vec3 vdbWorldPos = localPos;
            vdbWorldPos.x -= vol.pivot_offset[0];
            vdbWorldPos.y -= vol.pivot_offset[1];
            vdbWorldPos.z -= vol.pivot_offset[2];
            density = sampleNanoVDBFloatTrilinearAcc(buf, mapH, acc, vdbWorldPos);
        } else {
            vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
            vec3 noiseCoord = normPos * max(vol.noise_scale, 1.0);
            density = fbmNoise(noiseCoord, 4);
            vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
            density *= smoothstep(0.0, 0.1, min(min(edgeDist.x, edgeDist.y), edgeDist.z));
        }
    } else if (vol.volume_type == 3 || vol.source_type == 3) {
        density = proceduralCloudDensity(vol, localPos);
    } else if (vol.volume_type == 4 && vol.source_type == 5) {
        density = sampleDenseGasFloat(vol.vdb_grid_address, vol, localPos);
    }

    // SurfaceSDF consumes the producer's surface-centred 0..1 proxy directly.
    // Density/Remap/Cutoff/volume-noise are optical fog controls: applying them
    // before the iso=0.5 test moves the geometry and grows a false skirt along
    // domain walls. Explicit surface modifiers (porosity and opacity mask) are
    // applied later by sampleIsoField and remain geometric by design.
    if (!applyOpticalControls) {
        return (isnan(density) || isinf(density)) ? 0.0 : max(density, 0.0);
    }

    density = applyMaterialDensityNoise(vol, localPos, density);
    float remappedDensity = max((density - vol.density_remap_low) / max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0);
    float densityCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    if (remappedDensity <= densityCutoff) {
        return 0.0;
    }
    // A binary cutoff changes from empty to full extinction in one sample.
    // Against aerial perspective that discontinuity reads as a dark contour at
    // the sparse topology boundary. Preserve the authored rejection threshold,
    // then feather only the narrow band immediately above it.
    float cutoffFade = densityCutoff > 0.0
        ? smoothstep(densityCutoff, densityCutoff * 2.0, remappedDensity)
        : 1.0;
    return remappedDensity * vol.density_multiplier * cutoffFade;
}

float sampleDensityAcc(
    VkVolumeInstance vol,
    vec3 worldPos,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc)
{
    return sampleDensityAccMode(vol, worldPos, buf, mapH, acc, true);
}

// ═══════════════════════════════════════════════════════════════════════════
// PROCEDURAL POROSITY — fermented dough, aerated batter, pumice, set foam
// ═══════════════════════════════════════════════════════════════════════════
// Returns the amount to SUBTRACT from the density before the ISO test.
//
// ★ Why subtract from the FIELD instead of cutting alpha at the surface.
// A pore made by alpha has no rim: the hole is punched in an otherwise flat
// shading point, so its edge carries the parent surface's normal, refracts
// wrongly and casts no self-shadow. Displacing the field makes the pore real
// geometry — the surface normal here IS the field gradient, so the rim picks
// up the noise derivative for FREE, from the six samples already being taken.
// That is the whole reason this is a field term and not a shading term.
//
// ★★ MUST be a pure function of world position and the domain's own
// parameters — nothing per-ray, nothing per-material, no state. It is called
// from TWO places: the shading march AND nearestSurfaceSDFCrossing, the
// arbiter that decides where gas hands the ray over to liquid. If those two
// disagree by so much as a term, the gas is clipped against a surface the
// shader never draws, and nothing reports it. That constraint is also why the
// parameters live on the domain instance: the arbiter runs for OTHER volumes
// and has no access to this domain's surface material.
//
// Worley rather than fbm, deliberately: a bubble is a CELL. The distance to
// the nearest feature point of a Worley lattice is already a packed-sphere
// field, so 1-d gives voids at the cell centres. fbm would give clouds.
float isoPoreOffset(VkVolumeInstance vol, vec3 worldPos) {
    float amount = vol._accel_reserved[0];
    if (amount <= 1e-4) return 0.0;                       // off: exact old path
    float scale  = max(vol._accel_reserved[1], 1e-4);     // world units per cell
    float detail = clamp(vol._accel_reserved[2], 0.0, 1.0);

    // MATERIAL space, not world space, and not voxel space.
    //
    // Not voxel space, because the crumb must keep its real size when the domain
    // resolution changes, or every resolution edit re-bakes the bread. That is
    // why `scale` is in world units and why the coordinate below is too.
    //
    // Not world space, because a world-anchored pore lattice is nailed to the
    // room: a rising dough would slide THROUGH its own bubbles, gaining and
    // losing pores as it moved. The bubbles belong to the material, so they are
    // addressed in the material's coordinate — which also means they now stretch
    // with it, the way real crumb does. Falls back to worldPos when the domain
    // publishes no coordinate field, i.e. exactly the previous behaviour.
    //
    // ★ This stays inside sampleIsoField's ONE-field contract: the arbiter
    // (nearestSurfaceSDFCrossing) reaches this through the same call and reads
    // the same vol, so it sees the identical displaced field. Anchoring here
    // would only break that if it depended on something per-ray or per-material,
    // which the coordinate grid is not — it is per-domain data, like `scale`.
    vec3 p = materialAnchor(vol, worldPos) / scale;

    float pore = 1.0 - clamp(rh_worley(p) * 2.0, 0.0, 1.0);
    if (detail > 1e-3) {
        // A second, finer cell size mixed in by MAX (union of voids, not an
        // average): this is what separates bread crumb, which has mixed bubble
        // sizes, from packing foam, which has one.
        float fine = 1.0 - clamp(rh_worley(p * 2.7 + vec3(19.3, 7.1, 31.7)) * 2.0, 0.0, 1.0);
        pore = max(pore, fine * detail);
    }
    return amount * pore;
}

// ═══════════════════════════════════════════════════════════════════════════
// OPACITY MASK — holes in the liquid, cut in the FIELD
// ═══════════════════════════════════════════════════════════════════════════
// Returns the amount to SUBTRACT from the density before the ISO test, so a
// masked-out region is a real hole in the level set.
//
// ★ Why not an any-hit shader, which is the obvious answer. The volume
// procedural hit group binds VK_SHADER_UNUSED_KHR for any-hit, and the AABB
// BLAS is already built non-opaque — so the acceleration structure was never
// the obstacle, and adding an any-hit here would have "worked". It would also
// have produced RIMLESS holes: a cutout at the shading point punches through an
// otherwise flat surface, so the edge carries the parent surface's normal,
// refracts as if the hole were not there, and casts no self-shadow. On a liquid
// that is glaring, because the edge is exactly where refraction is read. Cut in
// the field instead and the rim normal falls out of the gradient that six
// samples are already computing — the same reason porosity is a field term.
//
// ★★ MEANING, matched to the mesh path deliberately. On a triangle,
// `mat.opacity` as a SCALAR becomes transmission (closesthit.rchit: `opacity <
// 0.99 -> transmission = 1 - opacity`), while the opacity TEXTURE is a cutout
// mask. This keeps that split exactly: only the texture cuts geometry here, and
// the scalar is left to the transmission path further down. Making the scalar
// erode the body instead would mean one material reads as semi-transparent
// glass on a mesh and as a thinner object on a liquid — the same number meaning
// two things, which is how a "the material looks wrong on water" report becomes
// unfindable.
//
// ★★★ THRESHOLDED, NOT STOCHASTIC, and this is forced rather than chosen. The
// mesh dithers intermediate mask values across samples (shadow_anyhit does
// exactly that, and it is right there). This field CANNOT: it is read by
// nearestSurfaceSDFCrossing, the gas/liquid handover arbiter, which must agree
// with the shading march sample for sample. A randomised threshold would make
// gas clip against a surface that differs every sample — shimmering smoke edges
// near liquid, with nothing pointing at opacity as the cause. The field must be
// a pure function of position, so the mask is hard.
float isoAlphaOffset(VkVolumeInstance vol, vec3 worldPos) {
    if (vol.iso_material_index <= 0.5) return 0.0;          // no bound material
    Material am = materials.m[uint(vol.iso_material_index - 1.0)];
    if (am.opacity_tex == 0u) return 0.0;                   // off: not one fetch

    // Tri-planar with EQUAL weights, not the sharpened-normal blend the albedo
    // path uses. There is no normal available here and there cannot be: this
    // function runs inside the gradient taps that COMPUTE the normal, so asking
    // for one is circular. Equal thirds of a hard mask make the test a majority
    // vote across the three projections, which is deterministic and stable.
    //
    // ★ Consequence worth knowing before it is mistaken for a bug: the hole edge
    // and an albedo edge cut from the SAME image will not land in exactly the
    // same place on steep faces, because the two use different blends.
    vec3 p = materialAnchor(vol, worldPos);
    vec2 sc = vec2(abs(am.uv_scale_x) > 1e-6 ? am.uv_scale_x : 1.0,
                   abs(am.uv_scale_y) > 1e-6 ? am.uv_scale_y : 1.0);
    vec2 of = vec2(am.uv_offset_x, am.uv_offset_y);
    vec4 tx = textureLod(materialTextures[nonuniformEXT(am.opacity_tex)], p.zy * sc + of, 0.0)
            + textureLod(materialTextures[nonuniformEXT(am.opacity_tex)], p.xz * sc + of, 0.0)
            + textureLod(materialTextures[nonuniformEXT(am.opacity_tex)], p.xy * sc + of, 0.0);
    tx *= (1.0 / 3.0);

    float mask = ((am.flags & MAT_FLAG_OPACITY_RGBA) != 0u) ? tx.a : tx.r;
    if (mask >= 0.5) return 0.0;                            // solid here

    // 1.0 is enough to carry even a fully-interior cell (density 1.0) below the
    // 0.5 ISO threshold, so a masked region is gone THROUGH the body rather than
    // dimpled on its surface. A smaller push would erode the silhouette and
    // leave the interior intact — a hole that closes as the liquid thickens,
    // which reads as the mask "not working in deep water".
    return 1.0;
}

// The ONE field every isosurface consumer must read. Both ISO threshold sites
// and all six gradient samples go through here, so the surface that gas is
// clipped against, the surface that is shaded, and the normal used to shade it
// can never come from different fields.
// `acc` is inout for the same reason sampleDensityAcc takes it that way: it is
// a persistent NanoVDB read accessor whose node cache must survive the call.
float sampleIsoField(VkVolumeInstance vol, vec3 worldPos, pnanovdb_buf_t buf,
                     pnanovdb_map_handle_t mapH, inout pnanovdb_readaccessor_t acc) {
    return sampleDensityAccMode(vol, worldPos, buf, mapH, acc, false)
         - isoPoreOffset(vol, worldPos)
         - isoAlphaOffset(vol, worldPos);
}

// ============================================================
// Density Sampling — supports homogeneous and procedural noise
// ============================================================
float sampleDensity(VkVolumeInstance vol, vec3 worldPos) {
    // Transform world pos → object space
    vec3 localPos;
    localPos.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y 
               + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y 
               + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y 
               + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];
    
    // Check against real bounding box instead of [-0.5, 0.5]^3
    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) {
        return 0.0;
    }
    
    float density = 1.0;
    
    if (vol.volume_type == 0) {
        // Homogeneous: constant density
        density = 1.0;
    } else if (vol.volume_type == 1) {
        // Procedural noise: convert localPos from world scale back to standard normalized coords
        // The procedural noise historically mapped [-0.5, 0.5] to fit bounds.
        // We remap the precise boundary.
        vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
        vec3 noiseCoord = normPos * vol.noise_scale;
        density = fbmNoise(noiseCoord, 4);
        
        // Smooth falloff near edges
        vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
        float edgeFalloff = min(min(edgeDist.x, edgeDist.y), edgeDist.z);
        density *= smoothstep(0.0, 0.1, edgeFalloff);
        
    } else if (vol.volume_type == 2) {
        // NanoVDB grid sampling.
        if (vol.vdb_grid_address != 0) {
            // Apply OptiX pivot parity since NanoVDB indexing assumes raw bounding spatial coordinates
            vec3 vdbWorldPos = localPos;
            vdbWorldPos.x -= vol.pivot_offset[0];
            vdbWorldPos.y -= vol.pivot_offset[1];
            vdbWorldPos.z -= vol.pivot_offset[2];
            
            density = sampleNanoVDBFloatTrilinear(vol.vdb_grid_address, vdbWorldPos);
        } else {
            // Fallback: procedural noise
            vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
            vec3 noiseCoord = normPos * max(vol.noise_scale, 1.0);
            density = fbmNoise(noiseCoord, 4);
            vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
            density *= smoothstep(0.0, 0.1, min(min(edgeDist.x, edgeDist.y), edgeDist.z));
        }
    } else if (vol.volume_type == 3 || vol.source_type == 3) {
        density = proceduralCloudDensity(vol, localPos);
    } else if (vol.volume_type == 4 && vol.source_type == 5) {
        density = sampleDenseGasFloat(vol.vdb_grid_address, vol, localPos);
    }
    
    density = applyMaterialDensityNoise(vol, localPos, density);
    // Apply density remap (No upper clamp, matches OptiX fmaxf)
    float remappedDensity = max((density - vol.density_remap_low) / max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0);
    float densityCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    if (remappedDensity <= densityCutoff) {
        return 0.0;
    }

    float cutoffFade = densityCutoff > 0.0
        ? smoothstep(densityCutoff, densityCutoff * 2.0, remappedDensity)
        : 1.0;
    return remappedDensity * vol.density_multiplier * cutoffFade;
}

// Accessor-aware lightMarch. Reuses the caller's density-grid accessor so the
// shadow-march inner loop also benefits from leaf-level cache hits.
float lightMarchAcc(
    VkVolumeInstance vol,
    vec3 pos,
    vec3 lightDir,
    float maxDist,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    float shadowStrengthOverride)
{
    if (vol.shadow_steps <= 0) return 1.0;
    if (maxDist <= 1e-4) return 1.0;

    vec3 shadowAbsColor = max(vol.absorption_color, vec3(0.0));
    float shadowAbsPeak = max(
        shadowAbsColor.r, max(shadowAbsColor.g, shadowAbsColor.b));
    vec3 shadowAbsWeights = shadowAbsPeak > 1e-5
        ? shadowAbsColor / shadowAbsPeak : vec3(1.0);
    float sigma_t = vol.scatter_coefficient +
        vol.absorption_coefficient *
        dot(shadowAbsWeights, vec3(0.2126, 0.7152, 0.0722));
    if (sigma_t <= EPSILON) return 1.0;

    int reqSteps = clamp(vol.shadow_steps, 1, 64);
    uint shadowMatIndex = vol._reserved[1] > 0.5 ? uint(vol._reserved[1] - 1.0) : 0u;
    bool allowSparseTraversal = buf.address != 0 &&
        (vol._reserved[1] <= 0.5 || matProgramOffset(shadowMatIndex) == MATPROG_NONE);
    // Same Volume-Graph rule as the sparse path: a program may synthesize
    // density where the source field is empty, so the block skip is only valid
    // for unmodified density.
    bool allowDenseBlockSkip = vol.majorant_address != 0 &&
        (vol._reserved[1] <= 0.5 || matProgramOffset(shadowMatIndex) == MATPROG_NONE);
    float denseBlockCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    // maxDist is the chord from this scatter point to the volume boundary.
    // Integrate the whole chord; half-length marching over-lights the far side
    // of dense gas and erases the rolling self-shadow detail.
    float marchLength = maxDist;
    float midpointDensity = sampleDensityAcc(
        vol, pos + lightDir * (0.5 * marchLength), buf, mapH, acc);
    float tauHint = max(midpointDensity, 0.0) * sigma_t * marchLength;
    float stepScale = clamp(sqrt(max(tauHint, 0.0)), 0.20, 1.0);
    // Dense gas needs eight buffer loads per trilinear density lookup. Twelve
    // jittered samples over the exact exit chord retain soft self-shadow detail
    // while avoiding the 16-sample worst case on every cached shadow update.
    int shadowSampleCap = (vol.volume_type == 4 && vol.source_type == 5) ? 12 : 16;
    int steps = clamp(
        int(ceil(float(min(reqSteps, shadowSampleCap)) * stepScale)),
        min(3, reqSteps),
        min(reqSteps, shadowSampleCap));
    float stepSize = marchLength / float(max(steps, 1));
    stepSize = max(stepSize, 1e-5);
    float jitter = fract(sin(dot(pos, vec3(12.9898, 78.233, 37.719)) +
                             dot(lightDir, vec3(39.346, 11.135, 83.155))) * 43758.5453);

    float s_trans = 0.0;
    float distanceAlongRay = min((jitter + 0.5) * stepSize, marchLength);
    int densitySamples = 1; // includes midpoint density used for adaptive step count
    int traversalIters = 0;
    int maxTraversalIters = steps * 4 + 8;
    while (distanceAlongRay < marchLength &&
           densitySamples < steps &&
           traversalIters < maxTraversalIters) {
        vec3 samplePos = pos + lightDir * distanceAlongRay;
        // Live dense gas: same block-majorant skip the primary march uses. The
        // shadow chord crosses the domain just like a camera ray does, so an
        // empty region cost a full sample budget here too — and this march runs
        // once per light per (strided) step, so it is the more expensive of the
        // two places to walk through nothing.
        if (vol.volume_type == 4 && vol.source_type == 5 && allowDenseBlockSkip) {
            float denseSkip = denseGasEmptyBlockStep(
                vol, samplePos, lightDir, denseBlockCutoff);
            if (denseSkip > stepSize * 1.01) {
                distanceAlongRay += min(denseSkip, marchLength - distanceAlongRay);
                traversalIters++;
                continue;
            }
        }
        if (allowSparseTraversal) {
            uint shadowSkipKind = 0u;
            float sparseStep = nanoEmptyTileStep(
                vol, samplePos, lightDir, stepSize, buf, mapH, acc,
                shadowSkipKind);
            if (sparseStep > stepSize * 1.01) {
                distanceAlongRay += min(sparseStep, marchLength - distanceAlongRay);
                traversalIters++;
                continue;
            }
        }

        float d = sampleDensityAcc(vol, samplePos, buf, mapH, acc);
        s_trans += d * sigma_t * stepSize;
        if (s_trans > 10.0) break;
        distanceAlongRay += stepSize;
        densitySamples++;
        traversalIters++;
    }
    volumeRecordShadowSamples(uint(densitySamples));

    float phys_trans = exp(-s_trans);

    float shadowStrength = (shadowStrengthOverride >= 0.0)
        ? clamp(shadowStrengthOverride, 0.0, 1.0)
        : clamp(vol.shadow_strength * 0.92, 0.0, 1.0);
    return 1.0 - shadowStrength * (1.0 - phys_trans);
}

// ============================================================
// Light March — estimate transmittance from scatter point toward light
// (OptiX vol_light_steps eşdeğeri)
// ============================================================
float lightMarch(VkVolumeInstance vol, vec3 pos, vec3 lightDir, float maxDist) {
    if (vol.shadow_steps <= 0) return 1.0;
    if (maxDist <= 1e-4) return 1.0;
    
    // [SHADOW FIX] Match OptiX: s_step_world = world_vol_extent / (shadow_steps * 2)
    // This covers half of maxDist with shadow_steps samples.
    // OLD code used  min(maxDist, step_size*2) / shadow_steps  which for a large cloud
    // (maxDist=10, step_size=0.1) would only march 0.2 world units — missing 98% of the
    // volume. That left dense cores fully lit → solid white.
    vec3 shadowAbsColor = max(vol.absorption_color, vec3(0.0));
    float shadowAbsPeak = max(
        shadowAbsColor.r, max(shadowAbsColor.g, shadowAbsColor.b));
    vec3 shadowAbsWeights = shadowAbsPeak > 1e-5
        ? shadowAbsColor / shadowAbsPeak : vec3(1.0);
    float sigma_t = vol.scatter_coefficient +
        vol.absorption_coefficient *
        dot(shadowAbsWeights, vec3(0.2126, 0.7152, 0.0722));
    if (sigma_t <= EPSILON) return 1.0;

    int reqSteps = clamp(vol.shadow_steps, 1, 64);
    float dMid = sampleDensity(vol, pos + lightDir * (0.5 * maxDist));
    float tauHint = max(0.0, dMid) * sigma_t * maxDist;
    if (tauHint <= 0.02) return 1.0;
    float stepScale = clamp(sqrt(tauHint), 0.25, 1.0);
    int steps = int(ceil(float(reqSteps) * stepScale));
    steps = clamp(steps, 3, min(reqSteps, 16));

    float stepSize = maxDist / (float(steps) * 2.0);
    // Clamp to avoid NaN/zero-step if shadow_steps was set very high
    stepSize = max(stepSize, 1e-5);
    float jitter = fract(sin(dot(pos, vec3(12.9898, 78.233, 37.719)) +
                             dot(lightDir, vec3(39.346, 11.135, 83.155))) * 43758.5453);
    
    // Accumulate optical depth (matches OptiX: density_sum += sigma_t * step)
    float s_trans = 0.0;
    uint measuredShadowSamples = 1u; // includes dMid used for adaptive step count
    for (int i = 0; i < steps; i++) {
        vec3 samplePos = pos + lightDir * (float(i) + jitter + 0.5) * stepSize;
        float d = sampleDensity(vol, samplePos);
        measuredShadowSamples++;
        s_trans += d * sigma_t * stepSize;
        if (s_trans > 10.0) break; // fully occluded
    }
    volumeRecordShadowSamples(measuredShadowSamples);
    
    // Multi-scatter shadow blend — matches OptiX:
    // Beer-Lambert shadow transmission. Shadow strength remains an artistic
    // blend between the physical result and an unshadowed result.
    float phys_trans = exp(-s_trans);
    
    float shadowStrength = clamp(vol.shadow_strength * 0.92, 0.0, 1.0);
    return 1.0 - shadowStrength * (1.0 - phys_trans);
}

// ============================================================
// Main — Volume Ray March Entry Point
// ============================================================
// Distance, in world-ray parameter units, from a point in the volume to the
// local AABB exit along the light direction.
float volumeExitDistance(VkVolumeInstance vol, vec3 worldPos, vec3 worldDir) {
    vec3 lp;
    lp.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y
         + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
    lp.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y
         + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
    lp.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y
         + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];
    vec3 ld;
    ld.x = vol.inv_transform[0] * worldDir.x + vol.inv_transform[1] * worldDir.y
         + vol.inv_transform[2] * worldDir.z;
    ld.y = vol.inv_transform[4] * worldDir.x + vol.inv_transform[5] * worldDir.y
         + vol.inv_transform[6] * worldDir.z;
    ld.z = vol.inv_transform[8] * worldDir.x + vol.inv_transform[9] * worldDir.y
         + vol.inv_transform[10] * worldDir.z;
    const float DIR_EPS = 1e-8;
    vec3 invDir = vec3(
        abs(ld.x) > DIR_EPS ? 1.0 / ld.x : (ld.x >= 0.0 ? 1e20 : -1e20),
        abs(ld.y) > DIR_EPS ? 1.0 / ld.y : (ld.y >= 0.0 ? 1e20 : -1e20),
        abs(ld.z) > DIR_EPS ? 1.0 / ld.z : (ld.z >= 0.0 ? 1e20 : -1e20));
    vec3 t0 = (vol.aabb_min - lp) * invDir;
    vec3 t1 = (vol.aabb_max - lp) * invDir;
    vec3 tFar = max(t0, t1);
    return max(min(min(tFar.x, tFar.y), tFar.z), 0.0);
}

bool volumeRayInterval(VkVolumeInstance vol,
                       vec3 worldOrigin,
                       vec3 worldDir,
                       out float intervalNear,
                       out float intervalFar) {
    vec3 lp;
    lp.x = vol.inv_transform[0] * worldOrigin.x + vol.inv_transform[1] * worldOrigin.y
         + vol.inv_transform[2] * worldOrigin.z + vol.inv_transform[3];
    lp.y = vol.inv_transform[4] * worldOrigin.x + vol.inv_transform[5] * worldOrigin.y
         + vol.inv_transform[6] * worldOrigin.z + vol.inv_transform[7];
    lp.z = vol.inv_transform[8] * worldOrigin.x + vol.inv_transform[9] * worldOrigin.y
         + vol.inv_transform[10] * worldOrigin.z + vol.inv_transform[11];
    vec3 ld;
    ld.x = vol.inv_transform[0] * worldDir.x + vol.inv_transform[1] * worldDir.y
         + vol.inv_transform[2] * worldDir.z;
    ld.y = vol.inv_transform[4] * worldDir.x + vol.inv_transform[5] * worldDir.y
         + vol.inv_transform[6] * worldDir.z;
    ld.z = vol.inv_transform[8] * worldDir.x + vol.inv_transform[9] * worldDir.y
         + vol.inv_transform[10] * worldDir.z;
    const float DIR_EPS = 1e-8;
    vec3 invDir = vec3(
        abs(ld.x) > DIR_EPS ? 1.0 / ld.x : (ld.x >= 0.0 ? 1e20 : -1e20),
        abs(ld.y) > DIR_EPS ? 1.0 / ld.y : (ld.y >= 0.0 ? 1e20 : -1e20),
        abs(ld.z) > DIR_EPS ? 1.0 / ld.z : (ld.z >= 0.0 ? 1e20 : -1e20));
    vec3 t0 = (vol.aabb_min - lp) * invDir;
    vec3 t1 = (vol.aabb_max - lp) * invDir;
    vec3 lo = min(t0, t1);
    vec3 hi = max(t0, t1);
    intervalNear = max(max(lo.x, lo.y), lo.z);
    intervalFar = min(min(hi.x, hi.y), hi.z);
    return intervalFar > max(intervalNear, 0.0);
}

// Find the nearest real liquid iso crossing, not merely its procedural AABB.
// This is the Vulkan equivalent of OptiX's per-ray ordered VDB loop: gas is
// integrated only up to the liquid boundary, then the path is handed to the
// SurfaceSDF closest-hit instead of marching two complete overlapping boxes.
float nearestSurfaceSDFCrossing(vec3 rayOrigin,
                                vec3 rayDir,
                                float rangeNear,
                                float rangeFar,
                                uint currentVolume,
                                uint volumeCount) {
    float nearestHit = rangeFar + 1.0;
    uint count = min(volumeCount, 16u);
    for (uint candidateIndex = 0u; candidateIndex < count; ++candidateIndex) {
        if (candidateIndex == currentVolume) continue;
        VkVolumeInstance surface = volumes.v[candidateIndex];
        if (surface.is_active == 0 || surface.source_type != 4 ||
            surface.volume_type != 2 || surface.vdb_grid_address == 0) {
            continue;
        }

        float boxNear, boxFar;
        if (!volumeRayInterval(surface, rayOrigin, rayDir, boxNear, boxFar)) continue;
        float beginT = max(rangeNear, max(boxNear, 0.001));
        float endT = min(rangeFar, boxFar);
        if (endT <= beginT || beginT >= nearestHit) continue;

        pnanovdb_buf_t surfaceBuf;
        surfaceBuf.address = surface.vdb_grid_address;
        pnanovdb_grid_handle_t gridH;
        gridH.address.byte_offset = 0u;
        pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(surfaceBuf, gridH);
        pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(surfaceBuf, treeH);
        pnanovdb_map_handle_t mapH = pnanovdb_grid_get_map(surfaceBuf, gridH);
        pnanovdb_readaccessor_t acc;
        pnanovdb_readaccessor_init(acc, rootH);

        const float ISO = 0.5;
        float span = endT - beginT;
        int cap = clamp(surface.max_steps, 32, 512);
        float fineStep = clamp(surface.step_size,
                               surface.voxel_size * 0.1,
                               surface.voxel_size * 0.5);
        float step = max(0.001, max(fineStep, span / float(cap)));
        int steps = min(int(ceil(span / step)) + 1, cap + 1);
        float t0s = beginT;
        float d0 = sampleIsoField(
            surface, rayOrigin + rayDir * t0s, surfaceBuf, mapH, acc);
        bool startedInside = d0 > ISO;
        for (int s = 0; s < steps; ++s) {
            float t1s = min(t0s + step, endT);
            float d1 = sampleIsoField(
                surface, rayOrigin + rayDir * t1s, surfaceBuf, mapH, acc);
            bool crossed = startedInside
                ? (d0 >= ISO && d1 < ISO)
                : (d0 < ISO && d1 >= ISO);
            if (crossed) {
                float a = t0s;
                float b = t1s;
                for (int refine = 0; refine < 4; ++refine) {
                    float mid = 0.5 * (a + b);
                    float dm = sampleIsoField(
                        surface, rayOrigin + rayDir * mid, surfaceBuf, mapH, acc);
                    if ((dm > ISO) == startedInside) a = mid;
                    else b = mid;
                }
                nearestHit = min(nearestHit, 0.5 * (a + b));
                break;
            }
            t0s = t1s;
            d0 = d1;
            if (t0s >= endT) break;
        }
    }
    return nearestHit <= rangeFar ? nearestHit : -1.0;
}

void main() {
    // Volume instance index from gl_InstanceCustomIndexEXT
    // (Set via TLASInstance::customIndex when building TLAS for volume objects)
    uint volIdx = gl_InstanceCustomIndexEXT;
    payload.skipGasVolumes = false;
    uint volCount = uint(max(int(cam.pad0), 0));
    // ★BLACK BOX ROOT. volIdx is the TLAS customIndex, BAKED when the TLAS was
    // built; volCount comes from the volume packet, refreshed every frame. They
    // desync whenever the packet shrinks without a TLAS rebuild — and it does not
    // merely shrink, it can go to ZERO: when every domain is dropped for a frame
    // syncVDBVolumesToGPU takes the `vols.empty()` path and publishes count 0
    // (VulkanBackend.cpp ~20077) while the TLAS still holds every volume instance.
    // Then EVERY volume AABB fails this test at once. That is the cached-playback
    // trigger: on a restored frame with no particles and no active density cells,
    // the packet is genuinely empty. Live playback never empties it, which is
    // exactly why the black box only shows up once a cache exists.
    // Terminating here is what paints it black: `scattered = false` is TERMINAL in
    // raygen — no sky, no geometry behind, zero radiance. Measured signature of
    // this failure: transmission 1.0 everywhere (nothing absorbs), no first-hit
    // normal, extinction_terminations 0, step_budget_exhausted ~0.
    // Same rule as the is_active guard below, and as the comment in
    // volume_intersection.rint: a volume with no content passes the ray THROUGH.
    if (volIdx >= volCount) {
        vec3 passDir = normalize(gl_WorldRayDirectionEXT);
        payload.radiance = vec3(0.0);
        payload.attenuation = vec3(1.0);
        payload.scatterOrigin =
            gl_WorldRayOriginEXT + passDir * (max(volumeHitAttrib.y, gl_HitTEXT) + 0.002);
        payload.scatterDir = passDir;
        payload.scattered = true;
        payload.skipAABBs = false;
        payload.bounceType = BOUNCE_TRANSPARENT;
        return;
    }
    VkVolumeInstance vol = volumes.v[volIdx];
    
    // Defensive fallback: inactive AABBs are rejected by the intersection
    // shader and should never reach closest-hit. `scattered=false` is terminal
    // in raygen, so it must not be used as a transparent-skip mechanism.
    if (vol.is_active == 0) {
        vec3 inactiveDir = normalize(gl_WorldRayDirectionEXT);
        payload.radiance = vec3(0.0);
        payload.attenuation = vec3(1.0);
        payload.scatterOrigin =
            gl_WorldRayOriginEXT + inactiveDir * (max(volumeHitAttrib.y, gl_HitTEXT) + 0.002);
        payload.scatterDir = inactiveDir;
        payload.scattered = true;
        payload.skipAABBs = false;
        payload.bounceType = BOUNCE_TRANSPARENT;
        return;
    }
    
    vec3 rayOrigin = gl_WorldRayOriginEXT;
    vec3 rayDir    = normalize(gl_WorldRayDirectionEXT);
    
    // Get intersection range from the intersection shader
    float rawTNear = volumeHitAttrib.x;
    float tNear = rawTNear;
    float tFar  = volumeHitAttrib.y;
    bool cameraInsideVolume = (rawTNear <= 0.0);
    
    // Ensure valid march range
    tNear = max(tNear, 0.001);
    if (tFar <= tNear) {
        // Degenerate interval (grazing hit / camera on the exit face). Same rule:
        // never terminate the path on a volume box — pass through instead, or a
        // sliver of black pixels traces the AABB silhouette.
        payload.radiance = vec3(0.0);
        payload.attenuation = vec3(1.0);
        payload.scatterOrigin = rayOrigin + rayDir * (max(tFar, gl_HitTEXT) + 0.002);
        payload.scatterDir = rayDir;
        payload.scattered = true;
        payload.skipAABBs = false;
        payload.bounceType = BOUNCE_TRANSPARENT;
        return;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Persistent NanoVDB accessor — initialized ONCE per ray, reused across all
    // density samples in the main march, solid probe, and lightMarch inner loops.
    // For typical step_size ≪ 8 voxels (leaf size), most consecutive samples land
    // in the same leaf and the accessor's cached path skips the tree walk entirely.
    // Non-NanoVDB volumes (homogeneous, procedural, cloud) ignore these handles.
    // ══════════════════════════════════════════════════════════════════════════
    pnanovdb_buf_t        vdbBuf;
    pnanovdb_map_handle_t vdbMapH;
    pnanovdb_readaccessor_t vdbAcc;
    {
        vdbBuf.address = (vol.volume_type == 2 && vol.vdb_grid_address != 0) ? vol.vdb_grid_address : uint64_t(0);
        if (vdbBuf.address != 0) {
            pnanovdb_grid_handle_t gridH; gridH.address.byte_offset = 0u;
            pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(vdbBuf, gridH);
            pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(vdbBuf, treeH);
            vdbMapH = pnanovdb_grid_get_map(vdbBuf, gridH);
            pnanovdb_readaccessor_init(vdbAcc, rootH);
        } else {
            // Dummy zero-init so unrelated reads through Acc variants are well-defined.
            vdbMapH.address.byte_offset = 0u;
            pnanovdb_root_handle_t dummyRoot; dummyRoot.address.byte_offset = 0u;
            pnanovdb_readaccessor_init(vdbAcc, dummyRoot);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // FLUID SURFACE (source_type == 4): isosurface raymarch + Snell refraction
    // ══════════════════════════════════════════════════════════════════════════
    // The volume's density channel is a SDF-derived proxy band: 0 outside the
    // fluid, 1 inside, with a smooth ~surface_band_voxels-wide transition at
    // the surface. We walk the ray, find the first iso=0.5 crossing, compute
    // the surface normal from the gradient, and refract through with the
    // water IOR. No volumetric scatter — the surface is treated as a
    // dielectric boundary. Subsequent entries (second hit = exit) attenuate
    // by Beer-Lambert through the absorption channel for distance traversed.
    if (vol.source_type == 4) {
        const float ISO_THRESH = 0.5;
        // IOR comes from the bound fluid material (_ext_reserved[0]); fall back
        // to water if unset. Drives both refraction bending and Fresnel.
        float IOR_WATER = (vol._ext_reserved[0] > 1.0) ? vol._ext_reserved[0] : 1.33;

        // Walk quality. The step must satisfy TWO constraints:
        //   1. fine enough for surface accuracy (voxel-relative, step_size),
        //   2. coarse enough that `max_steps` actually reach tFar — otherwise
        //      the walk stops partway and the far side of the fluid is skipped
        //      (back cells "pass"). We take the MAX of the fine step and the
        //      cover step (marchLen / cap), so a large domain coarsens
        //      gracefully instead of being truncated. Raise max_steps for both
        //      fine AND complete on big domains.
        float marchLen = max(0.0, tFar - tNear);
        int   isoCap = clamp(vol.max_steps, 32, 2048);
        float fineStep = clamp(vol.step_size, vol.voxel_size * 0.1, vol.voxel_size * 0.5);
        float coverStep = marchLen / float(isoCap);
        float step = max(0.001, max(fineStep, coverStep));
        int   maxSteps = min(int(marchLen / step) + 2, isoCap + 2);

        float t      = tNear;
        float startD = sampleIsoField(vol, rayOrigin + rayDir * t, vdbBuf, vdbMapH, vdbAcc);
        // ★Side test with hysteresis — the black-band candidate.
        // A bare `startD > ISO_THRESH` is a coin flip whenever the walk BEGINS on
        // the boundary, and on Vulkan that happens by construction: the gas
        // closest-hit hands the ray over by restarting it a hair before the
        // crossing it just found by bisection, so the first sample sits within
        // rounding distance of 0.5. Land one ulp on the wrong side and the surface
        // is shaded as an EXIT — the entry event never happens and Beer-Lambert is
        // applied over the whole remaining depth, i.e. a correctly placed but
        // pitch-black surface, flickering per pixel.
        // OptiX uses the same naive test and stays clean precisely because it never
        // restarts a ray at a boundary: one call walks the ordered volume list from
        // the camera's t_enter. So this ambiguity is structurally Vulkan-only.
        // "Inside" now means robustly inside (interior reads 1.0, ≫ 0.55); a start
        // sitting ON the boundary is treated as just-outside-about-to-enter, which
        // is exactly what a handoff means. prevD is pulled below the threshold too,
        // otherwise the first step can't register the entering crossing and the
        // surface would silently vanish instead.
        const float ISO_HYST = 0.05;
        bool  startInside = startD > ISO_THRESH + ISO_HYST;
        float prevD  = startInside ? startD : min(startD, ISO_THRESH - 1e-4);
        float hitT   = -1.0;

        for (int s = 0; s < maxSteps; ++s) {
            float nextT = min(t + step, tFar);
            float curD = sampleIsoField(vol, rayOrigin + rayDir * nextT, vdbBuf, vdbMapH, vdbAcc);
            bool crossed = startInside
                ? (prevD >= ISO_THRESH && curD < ISO_THRESH)   // exit
                : (prevD <  ISO_THRESH && curD >= ISO_THRESH); // enter
            if (crossed) {
                // Precise binary search (bisection) refinement to find the exact isosurface intersection.
                // Eliminates sawtooth / staircasing / wood-grain aliasing artifacts completely!
                float t0 = t;
                float t1 = nextT;
                float d0 = prevD;
                float d1 = curD;
                for (int it = 0; it < 4; ++it) {
                    float tMid = 0.5 * (t0 + t1);
                    float dMid = sampleIsoField(vol, rayOrigin + rayDir * tMid, vdbBuf, vdbMapH, vdbAcc);
                    bool midInside = dMid > ISO_THRESH;
                    if (startInside == midInside) {
                        t0 = tMid;
                        d0 = dMid;
                    } else {
                        t1 = tMid;
                        d1 = dMid;
                    }
                }
                float denom = d1 - d0;
                float frac = (ISO_THRESH - d0) / ((abs(denom) > EPSILON) ? denom : EPSILON);
                frac = clamp(frac, 0.0, 1.0);
                hitT = t0 + frac * (t1 - t0);
                break;
            }
            prevD = curD;
            t = nextT;
            if (t >= tFar) break;
        }

        // ── Whitewater foam composited in FRONT of the surface ──────────────
        // Foam rides this same volume's TEMPERATURE channel (scaled by
        // FOAM_TEMP_SCALE = 10000 on upload; see
        // SceneData::syncSimulationRenderVolumes). One volume on the domain AABB
        // means no coincident-volume case for the integrator to drop — this is
        // the fix for the SDF+foam black cube. March it as bright white single-
        // scatter in front of the iso surface; everything behind is dimmed by
        // the foam transmittance. NO self-shadow: self-shadowing is what blacked
        // out the old separate fog volume.
        vec3  foam_inscatter = vec3(0.0);
        float foam_T = 1.0;
        if (vol.vdb_temp_address != 0) {
            const float FOAM_TEMP_SCALE = 10000.0;
            // Opacity rides _ext_reserved[6] (foam_shader density × scatter,
            // packed at sync); fall back to a sane default if unset.
            float foamOptical = vol._ext_reserved[6] > 1e-3 ? vol._ext_reserved[6] : 8.0;
            // Foam tint rides _ext_reserved[3..5] (foam_shader scattering colour);
            // default to a faint cool white.
            vec3 foamAlbedo = vec3(vol._ext_reserved[3], vol._ext_reserved[4], vol._ext_reserved[5]);
            if (dot(foamAlbedo, vec3(1.0)) < 1e-3) foamAlbedo = vec3(0.95, 0.97, 1.0);

            float foamEnd = (hitT < 0.0) ? tFar : hitT;
            float foamLen = foamEnd - tNear;
            if (foamLen > 1e-5) {
                int fsteps = int(foamLen / max(vol.voxel_size, foamLen / 48.0)) + 1;
                fsteps = min(fsteps, 64);
                float fstep = foamLen / float(fsteps);
                vec3  skyAmb  = sampleSkyAmbient(rayDir);
                vec3  sunDirF = normalize(worldData.w.sunDir);
                float jitf    = rnd(payload.seed);
                // ── Ray-constant shading terms (HOISTED out of the per-step loop) ──
                // view·sun, the HG phase and the thin-film cos() are identical for
                // every foam sample on this ray → evaluate them ONCE, not per step.
                float cosVS      = dot(rayDir, sunDirF);                      // view·sun
                float foamPhase  = henyeyGreenstein(cosVS, 0.35);            // mild forward lobe
                vec3  silver     = mix(vec3(1.0), 0.5 + 0.5 * cos(cosVS * 9.0 + vec3(0.0, 2.0, 4.0)), 0.15);
                // Foam shares the volume's Edge Cutoff (_reserved[0]) so the faint
                // low-density foam fringe is clipped like the water density — otherwise
                // it leaves a grey haze the density cutoff can't reach.
                float foamCutoff = max(vol._reserved[0], 1e-4);
                for (int fs = 0; fs < fsteps && foam_T > 0.01; ++fs) {
                    float ft   = tNear + (float(fs) + jitf) * fstep;
                    vec3  fpos = rayOrigin + rayDir * ft;
                    float fd   = sampleTemperature(vol, fpos) * (1.0 / FOAM_TEMP_SCALE);
                    if (fd <= foamCutoff) continue;
                    float a    = 1.0 - exp(-fd * foamOptical * fstep);
                    // Soft edge falloff over [cutoff, 2·cutoff] → clean fade, not a ring.
                    a *= smoothstep(foamCutoff, foamCutoff * 2.0, fd);

                    // ── Production whitewater shading ──────────────────────────
                    // Foam is a bright, high-albedo single-scatter medium. Two cues
                    // sell the look: a FORWARD-SCATTER silver lining (HG phase toward
                    // the light, so backlit crests glow) and a POWDER term (gentle
                    // edge-dark / core-bright self-occlusion). Still scene-lit (lights
                    // + nishita sun + sky ambient) and NO self-shadow, so it never
                    // goes black in scene-light-lit setups.
                    float powder    = mix(0.75, 1.0, gpu_powder_effect(fd, cosVS)); // gentle edge-dark (cosVS/foamPhase hoisted)
                    vec3 Lf = vec3(0.0);
                    if (cam.lightCount > 0u) {
                        int li = clamp(int(floor(rnd(payload.seed) * float(cam.lightCount))),
                                       0, int(cam.lightCount) - 1);
                        LightData lt = lights.l[li];
                        int   lty = int(lt.position.w + 0.5);
                        vec3  ldir; float latten = 1.0;
                        if (lty == 1) {
                            ldir = normalize(lt.direction.xyz);
                        } else {
                            vec3 toL = lt.position.xyz - fpos;
                            float dL = length(toL);
                            ldir = (dL > EPSILON) ? (toL / dL) : vec3(0.0, 1.0, 0.0);
                            if (dL > EPSILON) latten = 1.0 / (dL * dL);
                        }
                        float lph = henyeyGreenstein(dot(rayDir, ldir), 0.35);  // per-light silver lining
                        Lf += float(cam.lightCount) * lt.color.rgb * lt.color.a * latten
                            * (1.0 + lph * 3.0) * powder;
                    }
                    // Bubble-aggregate upgrade: thin-film silver tint (hoisted above).
                    if (worldData.w.mode == 2) {
                        Lf += sampleTransmittanceLUT(fpos, sunDirF)
                            * worldData.w.sunColor * worldData.w.sunIntensity
                            * (1.0 + foamPhase * 4.0) * powder * silver;        // backlit crest glow
                    } else {
                        Lf += worldData.w.sunColor * worldData.w.sunIntensity
                            * (foamPhase * 2.0) * powder * silver;              // sun glow w/o atmosphere LUT
                    }
                    Lf += skyAmb;                       // ambient fill (never fully dark)
                    // MULTIPLE SCATTERING: foam is a packed mass of air bubbles; its
                    // white comes from light diffusing across many interfaces, so thick
                    // foam reads as a bright FILLED white, not a thin wisp. Near-
                    // isotropic boost that saturates with local density.
                    float ms = 1.0 - exp(-fd * 6.0);
                    Lf += 1.2 * ms * (skyAmb + 0.4 * worldData.w.sunColor * worldData.w.sunIntensity);

                    foam_inscatter += foam_T * a * (foamAlbedo * Lf);
                    foam_T *= (1.0 - a);
                }
            }
        }

        // ── Solid geometry inside the domain ────────────────────────────────
        // The volume AABB is hit immediately when the ray is inside it, so the
        // iso branch would refract straight to the water surface and SKIP any
        // solid triangle sitting between the ray origin and that surface (or,
        // on a miss, anywhere in the domain). Probe for a solid in the relevant
        // span; if one is closer, stop and let it render (skipAABBs) instead of
        // refracting past it. Tints by the water depth in front of it.
        {
            float probeFar = (hitT < 0.0) ? tFar : hitT;
            const uint SOLID_FLAGS = gl_RayFlagsTerminateOnFirstHitEXT
                                   | gl_RayFlagsSkipClosestHitShaderEXT
                                   | gl_RayFlagsNoOpaqueEXT;
            // Exclude gas/fog AABBs (0x02) and SurfaceSDF AABBs (0x08), and
            // KEEP transient simulation particles (0x04).
            //
            // ★★★ 0x04 USED TO BE EXCLUDED HERE AND THAT MADE SPLAT SPHERES
            // DISAPPEAR. The exclusion was written when a fluid domain drew
            // EITHER spheres OR an isosurface, so no sphere could ever be inside
            // a liquid volume's AABB. Per-substance representation broke that
            // assumption: one domain now draws some substances as spheres and
            // reconstructs the rest into the isosurface, so the spheres live
            // INSIDE this volume's box. Invisible to this probe, they were
            // handled by whichever branch came next:
            //   • a real surface behind them  -> hand-off fires for THAT surface,
            //     the re-trace skips the AABBs, and the spheres draw correctly.
            //     ("renders fine in front of a solid")
            //   • nothing behind them          -> the miss branch below teleports
            //     the ray to tFar, straight PAST the spheres. ("invisible in
            //     empty space")
            //   • liquid pooled behind them    -> the iso branch shades the water
            //     and the spheres never get a chance. ("shows the dielectric")
            // One excluded bit, three different-looking symptoms, and the only
            // configuration that looked healthy was the one with no spheres at
            // all. ★★ The comment claiming "particles remain visible to primary
            // rays" stayed true and stopped being the whole story: a primary ray
            // that enters a volume never gets back to the particles on its own.
            //
            // ★ The cost argument still holds where it was made: the GAS march's
            // 1+6 nested solid-location probes still exclude 0x04. This is the
            // fluid-surface branch (vol.source_type == 4) and this is ONE
            // terminate-on-first-hit probe per volume hit.
            const uint SOLID_MASK  = 0xF5;
            const uint VOLUME_SOLID_PROBE = 0xC17D5EEDu;
            shadowPayload = vec4(0.0, 0.0, 0.0, uintBitsToFloat(VOLUME_SOLID_PROBE));
            traceRayEXT(topLevelAS, SOLID_FLAGS, SOLID_MASK, 0, 1, 1,
                        rayOrigin, max(1e-4, tNear - 0.002), rayDir, probeFar + 0.002, 1);
            if (shadowPayload.w < 0.5) {
                float solidT = shadowPayload.x;
                // Absorption for the water the ray crossed before the solid.
                if (startInside) {
                    float depth = max(0.0, solidT - tNear);
                    payload.attenuation *= exp(-vol.absorption_color * vol.absorption_coefficient * depth);
                    payload.attenuation *= vol.scatter_color;
                }
                // Foam floating in front of the solid still shows; dim the solid
                // behind it by the foam transmittance.
                payload.radiance    += foam_inscatter;
                payload.attenuation *= foam_T;
                // Continue straight to the solid; skip volume AABBs so the
                // triangle closesthit fires on the next trace.
                // ★ Re-trace from the ORIGINAL origin, with no step-back epsilon.
                // A fixed 0.01 offset is an absolute length applied to a distance
                // that scales with the scene: where it fails to land in FRONT of
                // the surface, the re-traced ray strikes the BACK face, closesthit
                // face-forwards the normal, and the wall shades as if lit from
                // inside — flipped normals in the debug view, black in beauty, and
                // only ever inside an active volume. No epsilon is needed: the ray
                // reached this volume because nothing was closer than the box, and
                // skipAABBs now removes the box, so the nearest hit from the same
                // origin IS that surface, at its true distance and true facing.
                payload.scatterOrigin = rayOrigin;
                payload.scatterDir    = rayDir;
                payload.skipAABBs     = true;
                payload.scattered     = true;
                payload.bounceType    = 0u;
                return;
            }
        }

        if (hitT < 0.0) {
            // Ray missed the iso-surface AND no solid — pass through, but any
            // airborne foam/spray in this segment still adds light + occlusion.
            payload.radiance    += foam_inscatter;
            payload.attenuation *= foam_T;
            payload.scatterOrigin = rayOrigin + rayDir * (tFar + 0.002);
            payload.scatterDir    = rayDir;
            payload.scattered     = true;
            payload.bounceType    = BOUNCE_TRANSPARENT;
            return;
        }

        vec3 hitPos = rayOrigin + rayDir * hitT;

        // Central-difference gradient on the density field for the surface
        // normal. h = voxel_size keeps the stencil aligned with the proxy
        // band thickness (band ≈ 0.5*voxel by default), so the gradient is
        // numerically well-conditioned.
        float h = max(0.001, vol.voxel_size);
        float sxp = sampleIsoField(vol, hitPos + vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float sxm = sampleIsoField(vol, hitPos - vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float syp = sampleIsoField(vol, hitPos + vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float sym = sampleIsoField(vol, hitPos - vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float szp = sampleIsoField(vol, hitPos + vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
        float szm = sampleIsoField(vol, hitPos - vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
        vec3 grad = vec3(sxp - sxm, syp - sym, szp - szm);
        float gradLen = length(grad);

        // Foam / whitewater: SDF Laplacian = surface curvature. High |curvature|
        // = wave crest / breaking edge / splash -> whiten. Reuses the 6 gradient
        // samples + the centre (≈iso), so it's nearly free.
        float foam_strength = clamp(vol._ext_reserved[2], 0.0, 1.0);
        if (foam_strength > 1e-3) {
            float dc = sampleIsoField(vol, hitPos, vdbBuf, vdbMapH, vdbAcc);
            float lap = abs((sxp + sxm + syp + sym + szp + szm) - 6.0 * dc);
            float foam = foam_strength * smoothstep(0.15, 0.7, lap);
            // Bright white whitewater, lit by the current throughput.
            payload.radiance += payload.attenuation * foam * vec3(0.9);
        }
        // ★ A degenerate COARSE gradient is not proof that the crossing is fake.
        // The ray march already measured a threshold sign change in this interval;
        // retry the local derivative before choosing a conservative fallback.
        // Density intentionally moves the 0.5 crossing through the proxy band
        // and makes the rendered body fuller. Near a domain-side wall or a thin
        // accumulated sheet, the one-voxel stencil can put both taps on the same
        // plateau even though the march above measured a real crossing. Retry
        // only that failed normal at quarter- and sixteenth-voxel scales. Normal
        // hits pay nothing extra; a difficult side wall pays at most 12 samples.
        for (int retry = 0; retry < 2 && gradLen <= 1e-4; ++retry) {
            h *= 0.25;
            sxp = sampleIsoField(vol, hitPos + vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
            sxm = sampleIsoField(vol, hitPos - vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
            syp = sampleIsoField(vol, hitPos + vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
            sym = sampleIsoField(vol, hitPos - vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
            szp = sampleIsoField(vol, hitPos + vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
            szm = sampleIsoField(vol, hitPos - vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
            grad = vec3(sxp - sxm, syp - sym, szp - szm);
            gradLen = length(grad);
        }
        // If the sparse-grid quantisation defeats both finer stencils, the
        // crossing direction still tells us the density derivative's sign:
        // outside->inside rises along the ray, inside->outside falls. Preserve
        // that measured entry/exit orientation instead of deleting the BSDF
        // surface with a transparent pass-through.
        if (gradLen <= 1e-4) {
            grad = startInside ? -rayDir : rayDir;
            gradLen = 1.0;
        }
        // Density increases TOWARD fluid interior, so -gradient points OUT of
        // the surface (toward the less-dense / air side).
        vec3 N = normalize(-grad);

        // ── Rough dielectric event (Fresnel importance-sampled). ────────────
        // Orient the geometric normal against the incoming ray.
        if (dot(rayDir, N) > 0.0) N = -N;

        // GGX roughness: jitter the normal inside a microfacet lobe so both the
        // reflection AND the refraction blur with surface_roughness
        // (_ext_reserved[1]). 0 = mirror-smooth still water.
        float roughness = clamp(vol._ext_reserved[1], 0.0, 1.0);

        // -- Scene material on the liquid surface -------------------------
        // vol.iso_material_index is the 1-based scene material the domain binds
        // for its surface. Deliberately NOT _reserved[1]: that slot means "run
        // the volume material VM with this program" and is gated on the graph
        // driving volume closure slots, so a plain Principled material resolves
        // to none there and would never have reached this branch.
        // When one is bound, the isosurface is shaded by the SAME lobe
        // selection a triangle gets: clearcoat, metal, transmission and the
        // random-walk SSS. That is what makes molten glass, lava, mud and
        // chocolate ordinary material settings instead of shader special
        // cases -- this branch used to be a bare Fresnel + Beer-Lambert
        // dielectric, which is why a dense material could only ever read as
        // dark glass.
        // No material bound -> fall through to that original path, so scenes
        // authored before this change render exactly as they did.
        // ── Which material(s) THIS point is made of ──────────────────────────
        // With a composition field published, the domain's single material is
        // only the fallback: each point resolves to the substances actually
        // present there. Without one, isoMatSlot is exactly vol.iso_material_index
        // and everything below is unchanged.
        float isoMatSlot = vol.iso_material_index;
        float isoMatSlotB = isoMatSlot;
        float isoMixB = 0.0;
        sampleComposition(vol, hitPos, isoMatSlot, isoMatSlotB, isoMixB);

        if (isoMatSlot > 0.5 || isoMatSlotB > 0.5) {
            // ★ A slot of 0 means "the built-in dielectric" for THAT substance,
            // which is a real authoring choice — one substance a full Principled
            // BSDF, another plain refractive liquid, in the same body. Clamped
            // to a valid index here and given zero weight below, so the mixture
            // degrades toward the material that IS bound instead of indexing
            // material -1 and reading whatever precedes the table.
            uint isoMatIdx  = uint(max(isoMatSlot,  1.0) - 1.0);
            uint isoMatIdxB = uint(max(isoMatSlotB, 1.0) - 1.0);
            float wB = (isoMatSlotB > 0.5) ? isoMixB : 0.0;
            if (isoMatSlot <= 0.5) wB = 1.0;   // A is the dielectric; B carries it

            Material    im  = materials.m[isoMatIdx];
            MaterialExt imx = materialsExt.m[isoMatIdx];

            // ═══════════════════════════════════════════════════════════════
            // BLEND THE PARAMETERS, SHADE ONCE.
            // ═══════════════════════════════════════════════════════════════
            // ★★★ Not "shade twice and mix the results". Principled's parameter
            // space is built to be interpolated, and one shading call keeps the
            // lobe selection, the SSS walk and the refraction event SINGLE — a
            // point in a mixture refracts once, through one interface, with an
            // intermediate IOR. Mixing two shaded results would give two
            // refractions and two entry events at one surface point, which reads
            // as a ghosted double image exactly where the substances meet.
            //
            // ★ Only the fields this branch actually consumes are blended.
            // Blending texture IDs would be meaningless (an index is a name, not
            // a quantity), so the dominant material's textures win: at wB > 0.5
            // the pair is swapped below instead.
            if (wB > 0.001) {
                if (wB > 0.5) {
                    // Dominant side owns the texture bindings and the flags.
                    // No temporary needed: the old A is re-fetched by index on
                    // the line below, not read out of `im`.
                    im  = materials.m[isoMatIdxB];
                    imx = materialsExt.m[isoMatIdxB];
                    isoMatIdx = isoMatIdxB;
                    wB = 1.0 - wB;
                    isoMatIdxB = uint(max(isoMatSlot, 1.0) - 1.0);
                }
                Material other = materials.m[isoMatIdxB];
                im.albedo_r     = mix(im.albedo_r,     other.albedo_r,     wB);
                im.albedo_g     = mix(im.albedo_g,     other.albedo_g,     wB);
                im.albedo_b     = mix(im.albedo_b,     other.albedo_b,     wB);
                im.roughness    = mix(im.roughness,    other.roughness,    wB);
                im.metallic     = mix(im.metallic,     other.metallic,     wB);
                im.transmission = mix(im.transmission, other.transmission, wB);
                im.ior          = mix(im.ior,          other.ior,          wB);
                im.opacity      = mix(im.opacity,      other.opacity,      wB);
                im.emission_r   = mix(im.emission_r,   other.emission_r,   wB);
                im.emission_g   = mix(im.emission_g,   other.emission_g,   wB);
                im.emission_b   = mix(im.emission_b,   other.emission_b,   wB);
                im.emission_strength =
                    mix(im.emission_strength, other.emission_strength, wB);
                im.clearcoat  = mix(im.clearcoat,  other.clearcoat,  wB);
                im.clearcoat_roughness =
                    mix(im.clearcoat_roughness, other.clearcoat_roughness, wB);
                im.subsurface_amount =
                    mix(im.subsurface_amount, other.subsurface_amount, wB);
                im.translucent = mix(im.translucent, other.translucent, wB);
                im.specular    = mix(im.specular,    other.specular,    wB);
                im.normal_strength =
                    mix(im.normal_strength, other.normal_strength, wB);
            }

            // ★ ONE exit push for EVERY lobe below — see the long note at the
            // bottom of this block. Hoisted here because the thin-shell and
            // resin branches return early and need it just as much as the
            // Principled path does; a ULP offset re-hits the same surface.
            float exitPush = max(0.003, vol.voxel_size);

            // ── GEOMETRIC vs SHADING normal ──────────────────────────────────
            // Ng is the level-set gradient: the real orientation of the surface.
            // N becomes the SHADING normal once a normal map perturbs it.
            //
            // ★★ THE SPLIT IS NOT COSMETIC. Everything that reasons about WHERE
            // the surface is must keep using Ng — above all seatOutsideBand,
            // which decides from dot(dir, N) whether a scattered ray is heading
            // out of the body or into it. Feed it a perturbed normal and that
            // test flips near grazing angles: an outgoing ray gets classified as
            // inward, takes the probe path, and either self-occludes or steps
            // through a thin wall. The result is light leaking through liquid
            // exactly where the bump detail is strongest, which reads as a
            // normal-map artefact and is really a band-logic one.
            //
            // Shading uses N; band, refraction geometry and ray re-seating use Ng.
            vec3 Ng = N;
            // ★ ONE projection draw for this hit, shared by the normal map and
            // every texture channel below. Drawn here, before the first use, so
            // there is exactly one place it can be got wrong. Independent draws
            // per channel would sample albedo from one plane and roughness from
            // another at the same surface point — which converges to a blend of
            // two different materials and looks like a plausible material rather
            // than like a bug.
            float triXi = rnd(payload.seed);
            if (im.normal_tex > 0u) {
                vec2 nmScale = vec2(abs(im.uv_scale_x) > 1e-6 ? im.uv_scale_x : 1.0,
                                    abs(im.uv_scale_y) > 1e-6 ? im.uv_scale_y : 1.0);
                // Perturbed HERE, before SurfaceSample is filled and before the
                // direct-light block reads N. Doing it later would light the
                // surface with the flat normal and bounce off the bumped one —
                // "the normal map only affects reflections", which is a report
                // that sends you looking at the map instead of at the ordering.
                N = triplanarNormal(im.normal_tex,
                                    materialAnchor(vol, hitPos), Ng,
                                    nmScale, vec2(im.uv_offset_x, im.uv_offset_y),
                                    max(im.normal_strength, 0.0), triXi);
            }

            // Depth absorption and whitewater are properties of the LIQUID
            // BODY, not of the surface lobe, so they apply whichever branch is
            // taken below. Hoisted out of the Principled path for that reason.
            if (startInside) {
                float depth = max(0.0, hitT - tNear);
                payload.attenuation *= exp(-(vol.absorption_color * vol.absorption_coefficient) * depth);
            }
            payload.radiance    += foam_inscatter;
            payload.attenuation *= foam_T;

            // ── THIN-SHELL FILM (bubble) ─────────────────────────────────────
            // Soap foam / champagne head sitting ON the liquid. A thin shell is
            // entered and exited parallel, so there is no net refraction: the
            // ray either Fresnel-reflects off the film (bright silver rim,
            // strongest at grazing) or passes STRAIGHT through. Mirrors the
            // triangle branch in closesthit.rchit — same fields, same look, so
            // a bubble material reads identically on a mesh and on a liquid.
            //
            // Dispatched BEFORE the Principled fill because on the triangle
            // side it is likewise a separate type branch, not a lobe: there is
            // no roughness/metallic/SSS to build for a film.
            if ((im.flags & MAT_FLAG_BUBBLE) != 0u) {
                float cosTb = min(abs(dot(rayDir, N)), 1.0);
                float bio   = (imx.bubble_ior > 1.0001) ? imx.bubble_ior : 1.33;
                float r0b   = (1.0 - bio) / (1.0 + bio); r0b = r0b * r0b;
                float fresb = r0b + (1.0 - r0b) * pow(1.0 - cosTb, 5.0);
                vec3 bDir, bAtt;
                bool bPassThrough = false;
                if (rnd(payload.seed) < fresb) {
                    bDir = reflect(rayDir, N);              // bright Fresnel rim
                    if (imx.bubble_film > 1e-3) {
                        float opd = imx.bubble_film * (1.0 / max(cosTb, 0.15));
                        bAtt = vec3(0.55 + 0.45 * cos(opd * TWO_PI),
                                    0.55 + 0.45 * cos(opd * TWO_PI + 2.0944),
                                    0.55 + 0.45 * cos(opd * TWO_PI + 4.1888));
                    } else {
                        bAtt = vec3(1.0);
                    }
                } else {
                    bDir = rayDir;                          // straight through
                    bAtt = vec3(0.85) + 0.15 * vec3(im.albedo_r, im.albedo_g, im.albedo_b);
                    bPassThrough = true;
                }
                // ★ Both lobes leave along bDir, pushed clear of the level-set
                // band. The triangle version offsets a few ULPs off the plane;
                // here that restarts INSIDE the band — and for the pass-through
                // lobe (bDir == rayDir) that means the ray immediately re-hits
                // the same film and never leaves the surface at all.
                // ★ The push must not STEP OVER real geometry.
                //
                // exitPush is scaled to the level-set band, i.e. to voxel_size —
                // centimetres at production settings. The pass-through lobe
                // travels straight INTO whatever sits behind the film, so any
                // solid closer than one voxel is simply jumped past: the ray
                // restarts beyond it and the surface behind never renders. That
                // is the reported "thin shell skips the surface behind", and it
                // is specific to this lobe — the reflection lobe leaves the
                // surface outward, and the Principled path's own scatter lobes
                // do too.
                //
                // So probe just the push segment first. It is a few centimetres
                // long and terminates on first hit, so it costs far less than
                // the march that produced this hit. If a solid is inside it,
                // hand the ray over exactly the way the entry-side solid probe
                // above does: continue from the hit point with the volume AABBs
                // skipped, so the triangle closest-hit fires on that surface at
                // its true distance and true facing.
                bool bHandOff = false;
                payload.scatterDir    = bDir;
                payload.scatterOrigin = seatOutsideBand(hitPos, bDir, Ng, exitPush, bHandOff);
                payload.skipAABBs     = bHandOff;
                payload.attenuation  *= bAtt;
                payload.scattered     = true;
                // ★ The two lobes must be tagged DIFFERENTLY, and this is what
                // produced the black patches on the shell.
                //
                // A straight-through crossing is not a scattering event — the
                // ray leaves along the direction it arrived on. raygen only
                // treats a pass as free when the direction is unchanged AND
                // either the attenuation is exactly 1 or the tag is
                // BOUNCE_TRANSPARENT (isTransparentPass, raygen.rgen). The film
                // tints by 0.85..1.0, so the attenuation test can never pass —
                // tagged SPECULAR, every crossing spent a full GI bounce.
                //
                // A mesh bubble survives that: two crossings. A foam shell on an
                // isosurface does not — the level-set band is crossed many times
                // over, the budget runs out before the ray reaches the ground
                // behind, and raygen breaks the loop. Those paths return only
                // what they had accumulated, which is why the holes appeared
                // exactly where the shell is thickest and why the surface behind
                // was never reached.
                //
                // The reflection lobe DOES redirect the ray, so it stays a real
                // specular bounce and keeps costing one.
                payload.bounceType    = bPassThrough ? BOUNCE_TRANSPARENT
                                                     : BOUNCE_SPECULAR;
                // NOTE: the triangle branch clears payload.radiance here. Do NOT
                // copy that — this shader ACCUMULATES radiance (+=), and the
                // foam in-scatter for this hit was already added above.
                payload.primaryARG = packHalf2x16(vec2(im.albedo_r, im.albedo_g));
                payload.primaryABT = packHalf2x16(vec2(0.0, 1.0));   // aerial parity
                payload.primaryNrm = plPackNormal(N);
                return;
            }

            SurfaceSample ss = defaultSurfaceSample();
            ss.P      = hitPos;
            ss.N      = N;
            ss.rayDir = rayDir;
            ss.albedo    = vec3(im.albedo_r, im.albedo_g, im.albedo_b);
            // The MATERIAL's roughness wins. Binding a material means the
            // liquid is shaded by that material, full stop — having one of its
            // parameters silently overridden by a panel elsewhere is the kind
            // of split authority that makes a control look broken. The domain's
            // Surface Roughness governs the built-in dielectric below, and the
            // panel greys it out while a material is bound so the split is
            // visible rather than inferred.
            // This is the material's SCALAR roughness; the tri-planar block
            // below OVERRIDES it when a roughness texture is bound. (This note
            // used to claim a roughness texture could not reach an isosurface
            // at all. That stopped being true one screen further down, and a
            // comment asserting a capability does not exist is worse than no
            // comment — nobody goes looking for a feature they have been told
            // is impossible.)
            ss.roughness = clamp(im.roughness, 0.0, 1.0);
            ss.metallic  = clamp(im.metallic, 0.0, 1.0);
            ss.specular  = im.specular;
            ss.clearcoat          = clamp(im.clearcoat, 0.0, 1.0);
            ss.clearcoatRoughness = clamp(im.clearcoat_roughness, 0.001, 1.0);
            ss.clearcoatIridescence   = imx.clearcoat_iridescence;
            ss.clearcoatFilmThickness = imx.clearcoat_film_thickness;
            ss.translucent        = clamp(im.translucent, 0.0, 1.0);
            ss.subsurfaceAmount   = clamp(im.subsurface_amount, 0.0, 1.0);
            ss.subsurfaceColor    = max(vec3(imx.subsurface_r, imx.subsurface_g, imx.subsurface_b), vec3(0.001));
            ss.subsurfaceRadius   = max(vec3(imx.subsurface_radius_r, imx.subsurface_radius_g, imx.subsurface_radius_b), vec3(0.001));
            ss.subsurfaceScale    = max(imx.subsurface_scale, 0.001);
            ss.subsurfaceAnisotropy = clamp(imx.subsurface_anisotropy, -0.99, 0.99);

            // ── Tri-planar textures ──────────────────────────────────────────
            // This is what lets a texture reach the liquid at all. Before it,
            // only scalar material values did, and assigning an image to a
            // fluid surface silently did nothing.
            //
            // Roughness/metallic go through the SHARED packed-ORM policy, so a
            // texture authored for a mesh reads the same channel here. Writing
            // a second channel rule for the isosurface would eventually pick a
            // different one and shade the liquid as metal for no visible reason.
            {
                // uv_scale doubles as world units per tile here; 0 would
                // collapse every sample onto one texel, so treat it as unset.
                vec2 texScale = vec2(abs(im.uv_scale_x) > 1e-6 ? im.uv_scale_x : 1.0,
                                     abs(im.uv_scale_y) > 1e-6 ? im.uv_scale_y : 1.0);
                vec2 texOffset = vec2(im.uv_offset_x, im.uv_offset_y);

                // ★ THE ANCHOR. Everything below projects from `texAnchor`
                // rather than from the world hit position, so the pattern is
                // attached to the liquid instead of to the room. Sampled once
                // and shared by all four maps: two maps resolving the anchor
                // separately would be two chances for them to disagree, and a
                // normal map registered against a different coordinate than its
                // albedo is a class of wrongness that reads as "the lighting is
                // off" rather than as a coordinate bug.
                vec3 texAnchor = materialAnchor(vol, hitPos);
                // Blend weights stay on the WORLD normal. They only decide which
                // of the three projections dominates, and deriving a material-
                // space normal would need the coordinate field's Jacobian for a
                // difference that is invisible while the mapping is near
                // identity — which is the regime this is useful in. It IS the
                // reason a violently stretched region can show a seam shift; see
                // the stretch limit noted on FluidParticles::uvw.

                if (im.albedo_tex > 0u) {
                    // Replaces, matching the surface path (the texture is the
                    // base colour there, not a tint on it).
                    ss.albedo = triplanarTexel(im.albedo_tex, texAnchor, N, texScale, texOffset, triXi).rgb;
                }
                if (im.roughness_tex > 0u) {
                    ss.roughness = clamp(samplePackedRoughness(
                        triplanarTexel(im.roughness_tex, texAnchor, N, texScale, texOffset, triXi),
                        0.0, im.flags), 0.0, 1.0);
                }
                if (im.metallic_tex > 0u) {
                    ss.metallic = clamp(samplePackedMetallic(
                        triplanarTexel(im.metallic_tex, texAnchor, N, texScale, texOffset, triXi),
                        im.flags), 0.0, 1.0);
                }
                // Normal maps are handled EARLIER, where Ng/N are split — not
                // here with the other maps. They have to be: this block runs
                // after SurfaceSample is filled and after the direct-light
                // block, both of which read the shading normal, so perturbing
                // it here would light the surface flat and bounce it bumped.
            }

            // ── EMISSION ─────────────────────────────────────────────────────
            // SurfaceSample is a REFLECTION parameter set: it has no slot for
            // emission, transmission or opacity. So a bound material's Emission
            // never reached the liquid — the branch simply never read the field.
            // Reported as "emission worked yesterday and stopped": binding a
            // material moves shading into this branch, and this branch had no
            // emission, so the glow disappeared the moment a material was bound.
            //
            // Added as LOCAL radiance, matching the foam in-scatter convention
            // right above: raygen multiplies payload.radiance by the throughput
            // it already carries, so pre-multiplying here would square it.
            //
            // The thin-shell branch above deliberately returns before this, the
            // same way the triangle branch does — an emissive bubble material
            // would wash the film out.
            {
                vec3 isoEmission = vec3(im.emission_r, im.emission_g, im.emission_b)
                                 * max(im.emission_strength, 0.0);
                if (im.emission_tex > 0u) {
                    // Emission texture is authoritative for colour, exactly as on
                    // the surface path; strength stays the material's.
                    vec2 emScale = vec2(abs(im.uv_scale_x) > 1e-6 ? im.uv_scale_x : 1.0,
                                        abs(im.uv_scale_y) > 1e-6 ? im.uv_scale_y : 1.0);
                    // Same anchor as the maps above — resolved again here
                    // because this block sits outside their scope, NOT because
                    // it is a different coordinate.
                    isoEmission = triplanarTexel(im.emission_tex,
                                                 materialAnchor(vol, hitPos), N,
                                                 emScale, vec2(im.uv_offset_x, im.uv_offset_y),
                                                 triXi).rgb
                                * max(im.emission_strength, 0.0);
                }
                payload.radiance += isoEmission;
            }

            // ── Glass / resin split, drawn ONCE ──────────────────────────────
            // Declared here because BOTH branches below read it and the draw
            // must be the same one: rolling separately would let a material
            // take the glass lobe and the resin coat in the same hit.
            //
            // The rule is the triangle's, kept verbatim: a material carrying
            // BOTH transmission and interior depth is an amber/jade body, not a
            // coated one. Picking the coat for it would silently repaint every
            // existing scene with such a material bound to a liquid.
            // Resin coat absorption, hoisted so the NEE block below can apply it
            // to DIRECT light too. The triangle path does exactly this
            // (`resinActive/resinDensity/resinExt` carried into its NEE); without
            // it a thick tinted coat would dim the indirect bounce and leave the
            // direct term at full strength, i.e. the coat would look thinner the
            // brighter the key light is.
            bool  neeResinActive  = false;
            vec3  neeResinExt     = vec3(0.0);
            float neeResinDensity = 0.0;

            // ★★★ TWO DIFFERENT TRANSMISSION VALUES, AND THEY MUST STAY APART.
            //
            // `isoTrans` SELECTS A LOBE here. On the triangle path the identically
            // named local does NOT — SurfaceSample carries no transmission field,
            // so there it only ever reaches evaluate_brdf_gl. That asymmetry is
            // invisible when reading the two files side by side and it is exactly
            // what a line copied from one into the other walks into.
            //
            // Copying the mesh's `opacity < 0.99 -> transmission = 1 - opacity`
            // into the value below did precisely that: every material authored
            // with opacity even slightly under 1 — which is most SKIN, WAX, JADE,
            // MARBLE and MILK, i.e. exactly the subsurface materials — started
            // taking the GLASS lobe and returning through scatterGlass before
            // scatterPrincipled (where the subsurface lobe lives) ever ran.
            //
            // Two symptoms, one cause, and neither of them names it:
            //   - "SSS broke"          — the subsurface lobe is never reached.
            //   - "textures stopped
            //      having any effect"  — scatterGlass ignores albedo/roughness
            //                           detail, so the tri-planar maps that DID
            //                           land stop showing.
            // Both come back together when the lobe selector is restored.
            float isoTransAuthored = clamp(im.transmission, 0.0, 1.0);

            // LOBE SELECTION: authored transmission ONLY. An opaque material must
            // never be routed into refraction because someone dialled its opacity
            // to 0.95 — that is a coverage statement, not "this is glass".
            bool  takeGlassLobe = (isoTransAuthored > 0.01) &&
                                  (rnd(payload.seed) < isoTransAuthored);

            // SHADING ONLY: scalar opacity thins the diffuse lobe in the BRDF
            // evaluation, which is all the mesh's copy of this rule actually
            // does. The isosurface read im.opacity nowhere before, so a
            // semi-transparent material still looked fully solid on a liquid —
            // that part of the fix stands, it just may not touch lobe choice.
            //
            // The TEXTURE half of opacity is elsewhere entirely: it cut geometry
            // back in isoAlphaOffset.
            float isoTrans = isoTransAuthored;
            if (im.opacity < 0.99 && im.metallic < 0.1 && isoTransAuthored < 0.01) {
                isoTrans = clamp(1.0 - im.opacity, 0.0, 1.0);
            }

            // ── TRANSMISSION (real refraction) ───────────────────────────────
            // ★ This is the lobe whose absence made "transmission does nothing"
            // on the liquid. scatterPrincipled is reflection-only, so binding a
            // material actually made the surface LESS capable than leaving it
            // unbound: the built-in dielectric below refracts, the material path
            // did not. Nothing about the data was wrong.
            //
            // Uses the SAME scatterGlass the mesh uses — it already lives in the
            // shared bsdf_scatter module — so IOR, roughness, dispersion and the
            // resin-tinted interior behave identically on a mesh and on a liquid.
            //
            // ★ frontFace comes from startInside, which is SDF-INHERITED data:
            // the march already established which side of the level set the ray
            // began on. A mesh has to read a winding order for this; here it is
            // a measurement, and it is the reason the lobe can be trusted on a
            // surface that has no consistent orientation of its own.
            if (takeGlassLobe) {
                float isoIor = (im.ior > 1.0001) ? im.ior : 1.33;
                scatterGlass(hitPos,
                             /*macroNormal  */ Ng,  // level-set gradient: the real surface
                             /*shadingNormal*/ N,   // normal-mapped, if a map is bound
                             /*frontFace    */ !startInside,
                             rayDir, ss.albedo, isoIor, ss.roughness,
                             imx.transmission_density,
                             clamp(vec3(imx.resin_color_r, imx.resin_color_g, imx.resin_color_b),
                                   vec3(0.0), vec3(1.0)),
                             im.dispersion, payload.seed);
                // ★ Same band re-seat every other lobe here needs — and this one
                // NEEDS the probe: the refracted direction continues INTO the
                // body, so on a thin wall (the side of a pool) a blind voxel
                // push clears the liquid and the surface behind it together.
                bool gHandOff = false;
                payload.scatterOrigin =
                    seatOutsideBand(hitPos, payload.scatterDir, Ng, exitPush, gHandOff);
                payload.skipAABBs = gHandOff;
                payload.primaryARG = packHalf2x16(ss.albedo.rg);
                payload.primaryABT = packHalf2x16(vec2(ss.albedo.b, 1.0));
                payload.primaryNrm = plPackNormal(N);
                return;
            }

            // ── RESIN COAT ───────────────────────────────────────────────────
            // A refractive ABSORBING layer over an opaque base: lacquered/candied
            // surfaces, honey and syrup skins, anything with suspended interior
            // structure. Fresnel-splits the surface — the reflection lobe is the
            // glossy coat, everything else reaches the base, which is tinted by
            // the coat absorption and then shaded as an ordinary opaque surface.
            //
            // ★ The coat thickness is AUTHORED (transmission_density), NOT
            // measured from the field. This is a skin ON the surface, not the
            // body of the liquid — the body's own depth absorption already ran
            // above, and folding the two together would double-count. That is
            // also why this matches the triangle path's thickness model exactly
            // rather than using the (better) real hitT - tNear distance: same
            // parameter, same look, on a mesh and on a liquid alike.
            //
            // The glass/resin split is drawn once, above — on the draws where
            // the glass lobe won we already returned through scatterGlass, so
            // reaching here means this hit is the coat.
            if (imx.transmission_density > 1e-4 && !takeGlassLobe) {
                float effIor = max((im.ior > 0.01) ? im.ior : 1.45, 1.45);
                float cosTr  = clamp(dot(-rayDir, N), 0.0, 1.0);
                float fresr  = schlickFresnel(cosTr, effIor);
                // Coat gloss is the resin LAYER's own roughness, independent of
                // the base (which is forced rough below).
                float resinRough = clamp(imx.resin_roughness, 0.0, 1.0);
                if (rnd(payload.seed) < fresr) {
                    vec3 Vr = -rayDir;
                    vec3 refl;
                    if (resinRough < 0.02) {
                        refl = reflect(rayDir, N);
                    } else {
                        // ggxSampleVNDF returns the REFLECTED direction directly.
                        float alphaR = max(resinRough * resinRough, 1e-4);
                        refl = ggxSampleVNDF(N, Vr, alphaR, rnd(payload.seed), rnd(payload.seed));
                        if (dot(refl, N) <= 0.0) refl = reflect(rayDir, N);
                    }
                    refl = normalize(refl);
                    payload.scatterDir    = refl;
                    // Outward-going, so the helper's probe is skipped and this
                    // is just the band push — routed through it anyway so every
                    // lobe in this branch obeys ONE rule. Divergence between
                    // these sites is what produced the thin-wall bug.
                    bool rHandOff = false;
                    payload.scatterOrigin =
                        seatOutsideBand(hitPos, refl, Ng, exitPush, rHandOff);
                    payload.skipAABBs = rHandOff;
                    payload.scattered     = true;
                    payload.bounceType    = BOUNCE_RESIN;   // raygen's resin budget
                    payload.primaryARG = packHalf2x16(ss.albedo.rg);
                    payload.primaryABT = packHalf2x16(vec2(ss.albedo.b, 1.0));
                    payload.primaryNrm = plPackNormal(N);
                    return;
                }
                // Reached the base. Per-channel absorption ∝ (1 - tint), so a
                // warm tint passes its own hue and swallows the complement
                // instead of darkening everything.
                vec3  ct    = clamp(vec3(imx.resin_color_r, imx.resin_color_g, imx.resin_color_b),
                                    vec3(0.0), vec3(1.0));
                float ctMax = max(ct.r, max(ct.g, ct.b));
                vec3  ext   = (vec3(1.0) - ct) * 1.35 + vec3(0.22 * (1.0 - ctMax));
                neeResinActive  = true;
                neeResinExt     = ext;
                neeResinDensity = imx.transmission_density;
                float cosVr = max(abs(cosTr), 0.25);

                vec3 Tdir = refract(rayDir, N, 1.0 / effIor);
                if (dot(Tdir, Tdir) < 1e-6) Tdir = rayDir;   // TIR fallback
                Tdir = normalize(Tdir);

                // NOTE: the triangle path also parallax-offsets the base ALBEDO
                // TEXTURE lookup along the refracted lateral travel. There is no
                // texture to offset here — the isosurface has no UVs — so the
                // base albedo is the material's scalar Base Color.
                bool resinHasInclusions = (imx.resin_inclusion > 0.001 ||
                                           imx.resin_dirt > 0.001 ||
                                           imx.resin_shard > 0.001);
                if (resinHasInclusions) {
                    // One light direction to shade the interior specks by (no
                    // shadow rays; the march is purely procedural). This uses
                    // volSampleLight — the surface shader's pick_smart_light_gl
                    // does NOT exist in this stage.
                    vec3 resinLightDir = N;
                    if (cam.lightCount > 0u) {
                        uint li = min(uint(rnd(payload.seed) * float(cam.lightCount)),
                                      cam.lightCount - 1u);
                        vec3 wi_; float d_; float a_; bool invSq_;
                        if (volSampleLight(lights.l[li], hitPos,
                                           rnd(payload.seed), rnd(payload.seed),
                                           wi_, d_, a_, invSq_) && dot(wi_, wi_) > 1e-8) {
                            resinLightDir = normalize(wi_);
                        }
                    }
                    // ANCHOR. The triangle path offers object space so the
                    // interior travels with the mesh. There is no object here —
                    // the analogue is the MATERIAL coordinate, which is what the
                    // liquid has instead of an object: the inclusions are now
                    // carried by the fluid rather than left standing in the tank
                    // while it pours past them.
                    //
                    // ★ The ORIGIN goes through the shared anchor, so the resin
                    // interior sits in whatever space the domain selected —
                    // material, domain-local or world — and cannot end up in a
                    // different one from the albedo painted over it.
                    //
                    // ★★ MAT_FLAG_RESIN_OBJ_SPACE IS DELIBERATELY NOT APPLIED TO
                    // THE ORIGIN HERE, and that is a behaviour change worth being
                    // explicit about. That flag means "object space" — on a mesh,
                    // the object is the mesh. On a liquid there is no object, and
                    // the analogue is the volume's own transform, which is
                    // exactly what COORD_DOMAIN already does inside the anchor.
                    // Applying both would run the inverse transform TWICE and
                    // push the inclusions into a space that is nobody's: still
                    // stable, still smooth, just wrong by a whole transform —
                    // and it would look like a scale bug in the inclusion size
                    // rather than like a double transform. On the isosurface the
                    // domain's Coordinate Space is the single authority; the flag
                    // keeps its full meaning on the triangle path, where it is
                    // the only thing that can express it.
                    //
                    // ★★★ The DIRECTIONS still take the flag's rotation. They
                    // cannot go through the anchor — transforming a direction
                    // properly needs the field's Jacobian — and a march direction
                    // that disagrees with its origin's space does not fail
                    // visibly: it makes the inclusions subtly elongate along the
                    // flow, which reads as "that is what resin looks like".
                    // Origin in the anchored space with directions in the
                    // volume's rotation is the honest approximation: exact
                    // wherever the mapping is near-rigid, and it degrades by
                    // SHEARING the pattern rather than tearing it.
                    vec3 mOrg = materialAnchor(vol, hitPos);
                    vec3 mDir = Tdir, mLit = resinLightDir;
                    if ((im.flags & MAT_FLAG_RESIN_OBJ_SPACE) != 0u) {
                        // Directions drop the translation column.
                        mat3 volRot = mat3(vol.inv_transform[0], vol.inv_transform[4], vol.inv_transform[8],
                                           vol.inv_transform[1], vol.inv_transform[5], vol.inv_transform[9],
                                           vol.inv_transform[2], vol.inv_transform[6], vol.inv_transform[10]);
                        mDir = normalize(volRot * Tdir);
                        mLit = normalize(volRot * resinLightDir);
                    }
                    ResinMarch rm = resinMarchInterior(
                        mOrg, mDir, imx.transmission_density, ext,
                        imx.resin_inclusion, imx.resin_dirt,
                        vec3(imx.resin_dirt_color_r, imx.resin_dirt_color_g, imx.resin_dirt_color_b),
                        imx.resin_shard, imx.resin_shard_hue,
                        clamp(ct * 0.5 + vec3(0.45), 0.0, 1.0),   // dust tint from coat colour
                        mLit,
                        max(imx.resin_inclusion_scale, 0.01),
                        uint(imx.dust_style + 0.5),
                        vec3(imx.dust_color_a_r, imx.dust_color_a_g, imx.dust_color_a_b),
                        vec3(imx.dust_color_b_r, imx.dust_color_b_g, imx.dust_color_b_b),
                        uint(imx.shard_shape + 0.5), payload.seed);
                    if (rm.dirtHit) {
                        ss.albedo = rm.dirtAlbedo;   // terminate on the speck
                    } else {
                        ss.albedo = mix(ss.albedo * rm.absorb, rm.dustTint * rm.absorb, rm.dustCover);
                        ss.albedo = clamp(ss.albedo + rm.shardGlow + rm.dustGlow + vec3(rm.sparkle),
                                          0.0, 1.0);
                    }
                } else {
                    float pathLen = 2.0 * imx.transmission_density / cosVr;
                    ss.albedo *= exp(-pathLen * ext);
                }
                // The coat already took the specular, so the base is rough and
                // non-metallic — same as the triangle path.
                //
                // The base is rough and non-metallic, same as the triangle path.
                // It DOES get direct lighting now — see the NEE block below,
                // which applies the coat absorption to it as well.
                ss.roughness = 1.0;
                ss.metallic  = 0.0;
            }

            // ═══════════════════════════════════════════════════════════════
            // DIRECT LIGHTING (NEE) — the reason a liquid used to render darker
            // and far noisier than a mesh carrying the SAME material
            // ═══════════════════════════════════════════════════════════════
            // Until now this branch sampled a bounce direction and returned, so
            // the surface received light ONLY when a random BSDF sample happened
            // to land on a light. For anything but a huge, close source that is a
            // tiny solid angle, so most samples contributed nothing (dark) and
            // the rare hits carried enormous energy (noisy). The mesh path never
            // had that problem because it samples the light explicitly.
            //
            // ★ The give-away in a bug report is the pair: "dimmer AND grainier,
            // same material, same distance". Energy loss alone would be dim and
            // clean; a bad lobe would be bright and wrong. Dim + noisy together
            // is the signature of a missing direct-light estimator.
            //
            // ★★ Uses the SHARED evaluators — evaluate_brdf_gl, pdf_brdf_gl,
            // compute_light_pdf_gl, the same light picker — not a private copy.
            // That is the point: the liquid and the mesh are now lit by one
            // estimator, so they cannot drift apart under future edits the way
            // two look-alike implementations would.
            //
            // ★★★ SHADOW RAYS USE MASK 0x01 (triangles only), matching the mesh.
            // The liquid's own AABB is on 0x08, so it is invisible to the shadow
            // ray and A LIQUID DOES NOT SHADOW ITSELF in the direct term. This is
            // deliberate for now, not an oversight: a shadow ray leaving this
            // surface starts inside the half-voxel level-set band and would
            // self-occlude immediately, which needs the same seatOutsideBand
            // treatment the scatter lobes get plus a march for the thickness. The
            // visible consequence is that a deep pool of an OPAQUE liquid (molten
            // metal, chocolate) is lit slightly too evenly on its underside.
            // Clear water, which is what this path mostly renders, is unaffected.
            //
            // Skipped entirely when the glass lobe was taken — that path returned
            // long before here — and skipped for the thin-shell film, which also
            // returns early. Both are specular: NEE contributes nothing to a
            // delta lobe and would just cost a shadow ray.
            {
                float pdf_select = 0.0;
                int lightIdx = pick_smart_light_gl(uvec2(0), hitPos, pdf_select);
                if (lightIdx >= 0) {
                    vec3 wi; float dist; float lightAtten;
                    if (sample_light_direction_gl(lights.l[lightIdx], hitPos,
                                                  rnd(payload.seed), rnd(payload.seed),
                                                  wi, dist, lightAtten) &&
                        dot(wi, wi) > 1e-8) {
                        wi = normalize(wi);
                        float NdotL = max(dot(N, wi), 0.0);
                        if (NdotL > 1e-6) {
                            // Outward-going by construction (NdotL > 0), so the
                            // band probe inside the helper returns immediately —
                            // this costs nothing but keeps the ONE exit-push rule
                            // rather than inventing a second epsilon here.
                            bool sHandOff = false;
                            vec3 shadowOrigin =
                                seatOutsideBand(hitPos, wi, Ng, exitPush, sHandOff);

                            shadowPayload = vec4(1.0, 1.0, 1.0, 0.0);
                            float tmax = max(dist - 1e-3, 1e-3);
                            traceRayEXT(topLevelAS,
                                        gl_RayFlagsTerminateOnFirstHitEXT
                                      | gl_RayFlagsSkipClosestHitShaderEXT,
                                        RT_MASK_DIRECT_SHADOW, 0, 1, 1,
                                        shadowOrigin, 1e-4, wi, tmax, 1);
                            vec3 vis = (shadowPayload.w > 0.5) ? shadowPayload.rgb : vec3(0.0);

                            if (any(greaterThan(vis, vec3(1e-4)))) {
                                vec3 V    = normalize(-rayDir);
                                vec3 brdf = evaluate_brdf_gl(N, V, wi, ss.albedo,
                                                             ss.roughness, ss.metallic,
                                                             ss.specular, isoTrans);
                                vec3 Li = lights.l[lightIdx].color.rgb
                                        * lights.l[lightIdx].color.a * lightAtten;

                                int  ltype  = int(lights.l[lightIdx].position.w + 0.5);
                                bool isDelta = (ltype == 0 || ltype == 1);

                                vec3 contrib;
                                if (isDelta) {
                                    // A delta light has no solid angle for the BSDF
                                    // to have sampled, so there is nothing to
                                    // balance against — MIS weight is 1 by
                                    // definition. Applying a heuristic here would
                                    // halve point and sun lighting for no reason.
                                    contrib = brdf * Li * NdotL / max(pdf_select, 1e-6);
                                } else {
                                    float pdfL = compute_light_pdf_gl(lights.l[lightIdx], dist, 1.0)
                                               * pdf_select;
                                    float pdfB = pdf_brdf_gl(N, V, wi, ss.roughness);
                                    contrib = brdf * Li * NdotL
                                            * power_heuristic(pdfL, pdfB)
                                            / max(pdfL, 1e-6);
                                }

                                if (neeResinActive) {
                                    // Slanted ENTRY path through the coat, exactly
                                    // as the triangle path does it.
                                    float cosL = max(NdotL, 0.05);
                                    contrib *= exp(-(neeResinDensity / cosL) * neeResinExt);
                                }

                                // Firefly clamp, same shape as the mesh's. A single
                                // sample landing on a tiny bright light with a
                                // near-zero pdf otherwise writes a permanent white
                                // dot into the accumulator.
                                contrib = max(contrib, vec3(0.0));
                                contrib = min(contrib, vec3(1e4));
                                if (any(isnan(contrib)) || any(isinf(contrib))) contrib = vec3(0.0);

                                payload.radiance += contrib * vis;
                            }
                        }
                    }
                }
            }

            scatterPrincipled(ss, payload.seed);

            // ★ Re-seat the scattered ray OUTSIDE the level-set band.
            //
            // Every lobe leaves via offset_ray(), whose offset is a few float
            // ULPs — correct for a triangle, where the surface is a plane of
            // zero thickness. This surface is not: it is an isosurface inside a
            // proxy band about half a voxel thick, i.e. ~25 mm at the 5 cm
            // voxels the fluid actually runs at. A ULP-offset ray restarts
            // INSIDE that band, immediately re-hits the same surface, and
            // scatters again — and again.
            //
            // That single fact explains three separate symptoms:
            //   - metal turns hard BLACK: each re-hit multiplies throughput by
            //     the metal albedo, and a metal has no diffuse escape;
            //   - clearcoat piles up glare: each re-hit is another draw of the
            //     stochastic layer lottery, with its 1/p compensation;
            //   - it is WORST on overlapping droplets and layered sheets, where
            //     re-entry finds yet another crossing instead of open air.
            // The fall-through dielectric below never showed it because it
            // pushes 3 mm along the outgoing direction and mostly transmits.
            //
            // Push along the direction of travel, so it works for a reflection
            // and a transmission alike, and scale it to the field the band is
            // built from rather than to a hand-tuned millimetre.
            // (exitPush is declared at the top of this block — the thin-shell
            // and resin branches above return early and need the same push.)
            //
            // Goes through the shared seat helper as well: scatterPrincipled's
            // translucent and subsurface lobes can also leave INTO the body, and
            // there the blind push has the same thin-wall problem. Reflections
            // cost nothing extra — the helper returns immediately for them.
            bool pHandOff = false;
            payload.scatterOrigin =
                seatOutsideBand(hitPos, payload.scatterDir, Ng, exitPush, pHandOff);
            payload.skipAABBs = pHandOff;

            payload.primaryARG = packHalf2x16(ss.albedo.rg);
            payload.primaryABT = packHalf2x16(vec2(ss.albedo.b, 1.0));
            payload.primaryNrm = plPackNormal(N);
            return;
        }

        if (roughness > 1e-3) {
            float a = roughness * roughness;
            float u1 = rnd(payload.seed);
            float u2 = rnd(payload.seed);
            float phi = TWO_PI * u1;
            float cosT = sqrt(max(0.0, (1.0 - u2) / (1.0 + (a * a - 1.0) * u2)));
            float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
            vec3 hT = vec3(sinT * cos(phi), sinT * sin(phi), cosT);
            vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            vec3 T = normalize(cross(up, N));
            vec3 B = cross(N, T);
            vec3 Np = normalize(hT.x * T + hT.y * B + hT.z * N);
            if (dot(rayDir, Np) < 0.0) N = Np;  // keep facing the ray
        }

        // Beer-Lambert over the segment the ray just traversed INSIDE the fluid
        // (only when it started inside — i.e. this is an exit / internal event).
        // Pure water is clear; the blue comes from this depth absorption.
        if (startInside) {
            float depth = max(0.0, hitT - tNear);
            vec3 sigmaA = vol.absorption_color * vol.absorption_coefficient;
            payload.attenuation *= exp(-sigmaA * depth);
        }

        // Fresnel, importance-sampled: reflect (scene/sky) vs refract (through).
        // Picking the branch by Fresnel probability means the per-branch weight
        // is 1 — no (1-fresnel)/fresnel multiply, far less noise than splitting.
        float eta = startInside ? IOR_WATER : (1.0 / IOR_WATER);
        float cosTheta = clamp(abs(dot(rayDir, N)), 0.0, 1.0);
        float r0 = (1.0 - IOR_WATER) / (1.0 + IOR_WATER);
        r0 = r0 * r0;
        float fresnel = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);

        vec3 refrDir = refract(rayDir, N, eta);
        bool tir = dot(refrDir, refrDir) < 1e-6;
        vec3 outDir;
        if (tir || rnd(payload.seed) < fresnel) {
            // Reflection — traces the scene/sky next bounce. No water cast.
            outDir = reflect(rayDir, N);
        } else {
            // Refraction — through the water; mild surface cast.
            outDir = normalize(refrDir);
            payload.attenuation *= vol.scatter_color;
        }

        // Whitewater foam in front of the surface: add its in-scatter now and
        // dim the redirected lobe (water/sky behind) by the foam transmittance.
        payload.radiance    += foam_inscatter;
        payload.attenuation *= foam_T;

        payload.scatterOrigin       = hitPos + outDir * 0.003;
        payload.scatterDir          = outDir;
        payload.scattered           = true;
        payload.primaryARG          = packHalf2x16(vol.scatter_color.rg);
        payload.primaryABT          = packHalf2x16(vec2(vol.scatter_color.b, 1.0));
        payload.primaryNrm          = plPackNormal(N);
        payload.bounceType          = 0u;
        return;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Ordered gas -> liquid handoff for overlapping burning-fluid domains.
    // Ordinary VDB/cloud volumes keep the original fast path.
    float layeredSurfaceT = -1.0;
    // ★Gate on "there is something in front of me", NOT on how the gas happens to be
    // stored. This used to require source_type == 5 (live dense GPU gas), which silently
    // excluded the SAME gas domain whenever it was replayed from a cache: restoring a
    // frame calls setGridDomainStates(), which clears gpu_resident_fields_valid, so the
    // domain publishes as a HOST NanoVDB volume instead. With the handoff gated off, the
    // gas marched its whole AABB and terminated the segment, and the SurfaceSDF volume
    // overlapping it was never reached — "the gas renders, the water surface is gone",
    // reproducible on cached playback only.
    // ★★An earlier attempt widened this to `source_type == 2` and changed NOTHING, because
    // 2 is not a value source_type can hold. The producer is VolumetricRenderer.cpp:505-508
    // — isosurface → 4, procedural → 3, everything else → 0 — and VulkanBackend.cpp:20204
    // then promotes live dense gas to 5. `2` is the VOLUME_TYPE of a NanoVDB volume, a
    // different field; that mixup is why the cached-playback bug survived the "fix".
    // ★OptiX is the oracle here and it settles the question: use_live_dense_gpu requires
    // !use_optix (scene_data.h:2356), so under OptiX the gas is ALWAYS host NanoVDB — and
    // OptiX renders this exact scene correctly, live and cached. Host-NanoVDB gas over a
    // liquid surface is therefore a configuration that MUST work.
    // So state the gate as what it means: hand off unless I AM the liquid (4) or a
    // procedural cloud (3). nearestSurfaceSDFCrossing skips every candidate that is not a
    // SurfaceSDF, so with no liquid in the scene this is a ≤16-slot early-out loop. An
    // ordinary VDB genuinely overlapping a water surface WANTS this ordering too.
    if (vol.source_type != 4 && vol.source_type != 3 && volCount > 1u) {
        layeredSurfaceT = nearestSurfaceSDFCrossing(
            rayOrigin, rayDir, tNear, tFar, volIdx, volCount);
        if (layeredSurfaceT > 0.0) {
            if (layeredSurfaceT <= tNear + 0.003) {
                payload.radiance = vec3(0.0);
                payload.attenuation = vec3(1.0);
                payload.scatterOrigin =
                    rayOrigin + rayDir * max(tNear, layeredSurfaceT - 0.001);
                payload.scatterDir = rayDir;
                payload.scattered = true;
                payload.skipGasVolumes = true;
                payload.bounceType = BOUNCE_TRANSPARENT;
                return;
            }
            tFar = min(tFar, layeredSurfaceT - 0.001);
        }
    }

    // SOLID SURFACE DETECTION inside the volume AABB
    // If a solid triangle exists between tNear and tFar, we must stop the march
    // just before it and signal raygen to fire the next bounce with
    // gl_RayFlagsSkipAABBEXT so the triangle closesthit fires correctly.
    //
    // Strategy: one any-hit distance probe. The dedicated payload sentinel makes
    // shadow_anyhit return gl_HitTEXT directly; the former 1 + 5/6 binary-search
    // traces multiplied catastrophically when a gas domain touched the ground.
    // ══════════════════════════════════════════════════════════════════════════
    float solidT = -1.0;  // -1 = no solid found
    {
        // Performance gate:
        // In optically thick segments, inner solid surfaces are effectively invisible.
        // Skip costly triangle probes in those cases.
        bool needSolidProbe = true;
        float marchDistProbe = max(tFar - tNear, 0.0);
        float sigmaTCoeff = max(vol.scatter_coefficient + vol.absorption_coefficient, 0.0);
        // ★ Never gate the CAMERA segment. When the gate is wrong here the cost is
        // not a dimmer surface, it is a MISSING one: solidT stays -1, the march is
        // never clamped, the triangle's closesthit never fires, and the viewer
        // looks straight through the wall to whatever stands behind it. The
        // estimate below cannot be made safe for this case either — it multiplies
        // the sampled density by the FULL box traversal, so a large domain drives
        // tauEst up regardless of how close to the entry the surface actually sits,
        // and a surface a few centimetres inside a big smoke box is declared
        // invisible. One probe per primary ray is a bounded, predictable cost; on
        // secondary bounces a genuinely thick medium really does hide what is
        // behind it, so the gate keeps earning its keep there.
        bool primarySegment = (payload.primaryMeta & PL_PRIMARY_DONE) == 0u;
        if (!primarySegment && sigmaTCoeff > EPSILON && marchDistProbe > 1e-4) {
            float tA = tNear + min(0.05 * marchDistProbe, 0.25);
            float tB = tNear + 0.5 * marchDistProbe;
            float dA = sampleDensityAcc(vol, rayOrigin + rayDir * tA, vdbBuf, vdbMapH, vdbAcc);
            float dB = sampleDensityAcc(vol, rayOrigin + rayDir * tB, vdbBuf, vdbMapH, vdbAcc);
            // ★★ CONSERVATIVE, not representative.
            //
            // Skipping this probe does not merely omit a cheap detail: solidT
            // stays -1, so tFar is never clamped to the surface, the march runs
            // the full AABB INCLUDING the part behind the solid, and the
            // scatter continuation that lets the triangle's closesthit fire
            // never happens. The surface is not dimmed — it is never shaded at
            // all, and the volume closes over it as an opaque blob.
            //
            // So the estimate must err toward PROBING. Averaging two taps let a
            // flame's dense core speak for the whole ray: raise scatter or
            // absorption a little, tauEst crosses the gate, and every surface
            // behind that part of the fire vanishes at once — the reported
            // "past a threshold it goes fully opaque", and only in the areas
            // that happen to have geometry behind them. Take the MINIMUM
            // instead: a ray is only declared hopeless when it is thick
            // everywhere it was sampled, not merely thick somewhere.
            float dMin = max(0.0, min(dA, dB));
            float tauEst = dMin * sigmaTCoeff * marchDistProbe;
            float transEst = exp(-tauEst);
            // And the gate itself was far too loose. At 3% transmittance a lit
            // floor behind the fire is plainly visible, so dropping it was a
            // visible error, not an invisible optimisation. Only skip once the
            // surface could contribute less than a few thousandths.
            float probeThreshold = cameraInsideVolume ? 0.006 : 0.003;
            needSolidProbe = (transEst > probeThreshold);
        }

        if (needSolidProbe) {
            const uint PROBE_FLAGS = gl_RayFlagsTerminateOnFirstHitEXT
                                   | gl_RayFlagsSkipClosestHitShaderEXT
                                   | gl_RayFlagsNoOpaqueEXT;
            // Real scene solids use bit 0x01. Exclude gas/fog AABBs (0x02),
            // transient simulation particles (0x04), and SurfaceSDF AABBs
            // (0x08); otherwise a particle-rich
            // gas preset turns each volume hit into up to six nested RT probes.
            const uint PROBE_MASK  = 0xF1;

            // Initial check: any solid in [tNear, tFar]?
            const uint VOLUME_SOLID_PROBE = 0xC17D5EEDu;
            shadowPayload = vec4(0.0, 0.0, 0.0, uintBitsToFloat(VOLUME_SOLID_PROBE));
            traceRayEXT(topLevelAS, PROBE_FLAGS, PROBE_MASK, 0, 1, 1,
                        rayOrigin, max(1e-4, tNear - 0.002), rayDir, tFar + 0.002, 1);
            volumeRecordSolidProbe(shadowPayload.w < 0.5);
            if (shadowPayload.w < 0.5) {
                solidT = shadowPayload.x;
                // Clamp march to just before the solid
                tFar = solidT - 0.01;
                // If the solid is essentially at or before the entry point, skip AABBs on next bounce
                if (tFar <= tNear) {
                    payload.radiance = vec3(0.0);
                    // No step-back epsilon — see the note at the first solid handoff.
                    payload.scatterOrigin = rayOrigin;
                    payload.scatterDir = rayDir;
                    payload.scattered = true;
                    payload.skipAABBs = true;
                    return;
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // RAY MARCH through volume (Regular stepping with jitter)
    // Matches OptiX volumetric ray march approach
    // ══════════════════════════════════════════════════════════════════════════
    float stepSize = max(vol.step_size, 1e-4);
    float marchDist = tFar - tNear;
    const int hardStepCeiling = 2048;
    int authoredStepBudget = max(vol.max_steps, 1);
    int effectiveStepBudget = min(hardStepCeiling, authoredStepBudget);
    int requiredSteps = max(1, int(ceil(marchDist / stepSize)));
    int maxSteps = min(requiredSteps, effectiveStepBudget);
    // Canonical Vulkan baseline: divide the complete interval into a fixed,
    // authored number of segments. Optical integration must never shorten the
    // geometric advance and strand the ray before tFar.
    float baseStep = marchDist / float(max(maxSteps, 1));
    
    float sigma_s_coeff = vol.scatter_coefficient;
    float sigma_a_coeff = vol.absorption_coefficient;
    uint volumeMatIndex = vol._reserved[1] > 0.5 ? uint(vol._reserved[1] - 1.0) : 0u;
    uint volumeProgram = vol._reserved[1] > 0.5 ? matProgramOffset(volumeMatIndex) : MATPROG_NONE;
    // Preserve the legacy fast path: a temperature NanoVDB lookup is eight
    // trilinear tree samples and must not run for an ordinary non-emissive cloud.
    bool needsTemperature = (vol.emission_mode >= 2) || (volumeProgram != MATPROG_NONE);
    
    vec3  accumulated_radiance = vec3(0.0);
    vec3  transmittance = vec3(1.0);
    float opticalDepthWeightedT = 0.0;
    float opticalDepthWeight = 0.0;
    // Deterministic analytical single-scattering is accumulated below. Keep
    // the legacy continuation branch structurally present but disabled: a
    // second stochastic HG estimator would double-count the same event.
    bool didScatter = false;
    float scatterT = tFar;
    vec3 ambientSky = sampleSkyAmbient(rayDir);
    int shadowStride = clamp(vol.shadow_stride, 1, 16);
    // Keep this scalarized. Some NVIDIA Vulkan RT compiler versions crash
    // during pipeline creation on dynamically indexed local bool arrays in a
    // closest-hit shader.
    float cachedSceneShadow0 = 1.0;
    float cachedSceneShadow1 = 1.0;
    bool cachedSceneShadow0Valid = false;
    bool cachedSceneShadow1Valid = false;
    float cachedSunShadow = 1.0;
    bool cachedSunShadowValid = false;
    float cachedAmbientShadow = 1.0;
    bool cachedAmbientShadowValid = false;

    // ── Scene-light subset for this ray ───────────────────────────────────────
    // At most two lights are evaluated per volume sample. The old code took
    // lights 0 and 1 by index while still weighting by lightCount/2, so from the
    // third light onward the scene was lit by the WRONG lights at inflated
    // energy — a five-light rig scaled lights 0-1 by 2.5x and never sampled the
    // other three at all.
    //
    // Two distinct lights are drawn uniformly WITHOUT replacement, which gives
    // each light probability 2/N and therefore weight N/2 — the same constant as
    // before, now attached to an estimator that is actually unbiased.
    //
    // The draw is RAY-CONSTANT, not per step, and that is deliberate: the
    // shadow-transmittance cache below is keyed on the slot (0/1) and reused for
    // up to `shadow_stride` steps. Re-drawing per step would silently pair one
    // light's radiance with another light's cached visibility.
    int   sampledLightCount = int(min(cam.lightCount, 2u));
    int   sampledLight0 = 0;
    int   sampledLight1 = 0;
    float lightWeight = 1.0;
    if (cam.lightCount > 0u) {
        int nLights = int(cam.lightCount);
        sampledLight0 = clamp(int(floor(rnd(payload.seed) * float(nLights))), 0, nLights - 1);
        if (nLights > 1) {
            // Offset draw over the remaining N-1 lights keeps the pair distinct
            // without a rejection loop.
            int offset = clamp(int(floor(rnd(payload.seed) * float(nLights - 1))), 0, nLights - 2);
            sampledLight1 = (sampledLight0 + 1 + offset) % nLights;
        }
        lightWeight = float(nLights) / float(sampledLightCount);
    }
    // Must match sampleSkyAmbient(): this is the representative direction of
    // the sky hemisphere whose radiance is used as the ambient source.
    vec3 ambientLightDir = normalize(vec3(0.0, 1.0, 0.0) * 0.55 + rayDir * 0.45);

    // Exactly the rejection threshold sampleDensityAcc applies to the remapped
    // density. The block skip must use the SAME test, or it discards space the
    // sampler would have shaded.
    float denseSkipCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;

    // ── Volume emission NEE: pick one emitting block for this ray ─────────────
    // Fire already lights the smoke, but only when a scattered ray happens to
    // land inside it — that is the noise. Aiming a sample straight at a burning
    // block turns the same energy into a converged image instead of fireflies.
    //
    // The emitter is chosen ONCE PER RAY and its radiance evaluated once, for the
    // same reason the scene-light pair is: the transmittance toward it is cached
    // across strided steps, so a per-step re-pick would pair one emitter's
    // radiance with another's visibility. Variance is resolved across samples.
    vec3  emitterPos = vec3(0.0);
    vec3  emitterRadiance = vec3(0.0);
    bool  hasEmitter = false;
    float cachedEmitterShadow = 1.0;
    bool  cachedEmitterShadowValid = false;
    if (vol.emissive_list_address != 0 && vol.emissive_capacity >= 1.0 &&
        vol.majorant_address != 0 && vol.emission_mode >= 2) {
        DenseGasEmissiveList elist = DenseGasEmissiveList(vol.emissive_list_address);
        uint emitterCount = elist.entries[0];
        // The producer keeps counting past capacity so the overflow is visible;
        // only the stored prefix may be indexed.
        uint usable = min(emitterCount, uint(vol.emissive_capacity));
        if (usable > 0u) {
            uint pick = uint(floor(rnd(payload.seed) * float(usable)));
            pick = min(pick, usable - 1u);
            uint blockIndex = elist.entries[1u + pick];

            ivec3 bdim = ivec3(vol.majorant_dim[0] + 0.5,
                               vol.majorant_dim[1] + 0.5,
                               vol.majorant_dim[2] + 0.5);
            if (bdim.x > 0 && bdim.y > 0 && bdim.z > 0) {
                ivec3 b = ivec3(int(blockIndex) % bdim.x,
                                (int(blockIndex) / bdim.x) % bdim.y,
                                int(blockIndex) / (bdim.x * bdim.y));
                float blockWorld = vol.majorant_block * max(vol.voxel_size, 1e-6);
                vec3 gridOrigin = vec3(vol._ext_reserved[3], vol._ext_reserved[4],
                                       vol._ext_reserved[5]);
                // Jitter inside the block so a coarse block grid does not read as
                // a lattice of point lights.
                vec3 jitterUVW = vec3(rnd(payload.seed), rnd(payload.seed), rnd(payload.seed));
                vec3 localEmitter = gridOrigin + (vec3(b) + jitterUVW) * blockWorld;
                emitterPos.x = vol.transform[0]*localEmitter.x + vol.transform[1]*localEmitter.y
                             + vol.transform[2]*localEmitter.z + vol.transform[3];
                emitterPos.y = vol.transform[4]*localEmitter.x + vol.transform[5]*localEmitter.y
                             + vol.transform[6]*localEmitter.z + vol.transform[7];
                emitterPos.z = vol.transform[8]*localEmitter.x + vol.transform[9]*localEmitter.y
                             + vol.transform[10]*localEmitter.z + vol.transform[11];

                float ed = sampleDensityAcc(vol, emitterPos, vdbBuf, vdbMapH, vdbAcc);
                float et = sampleTemperature(vol, emitterPos);
                if (ed > 0.0 && et > 0.0) {
                    float rangeMin = max(vol._ext_reserved[6], 0.0);
                    float rangeMax = (vol.max_temperature > rangeMin + 1.0)
                        ? vol.max_temperature : (rangeMin + 1500.0);
                    float eKelvin = (et > 20.0)
                        ? clamp(et, rangeMin, rangeMax)
                        : mix(rangeMin, rangeMax, clamp(et, 0.0, 1.0));
                    vec3 eColor;
                    if (vol.color_ramp_enabled != 0 && vol.ramp_stop_count > 0) {
                        float tr = clamp((eKelvin - rangeMin) / max(rangeMax - rangeMin, 1.0), 0.0, 1.0);
                        eColor = sampleColorRamp(vol, clamp(tr * vol.temperature_scale, 0.0, 1.0));
                    } else {
                        eColor = blackbodyToRGB(eKelvin * vol.temperature_scale);
                    }
                    vec3 eEmis = eColor * ed * vol.blackbody_intensity;
                    float eReaction = sampleFlame(vol, emitterPos);
                    eEmis *= (1.0 + eReaction);   // matches the march's reaction boost
                    // Radiant intensity of the block: emission per unit length
                    // (sigma_t * emis, the march's source term) times the block
                    // volume it stands for. Uniform pick over `usable` emitters
                    // makes the estimator weight `usable`.
                    float eSigmaT = ed * (vol.scatter_coefficient + vol.absorption_coefficient);
                    float blockVolume = blockWorld * blockWorld * blockWorld;
                    emitterRadiance = eEmis * eSigmaT * blockVolume * float(usable);
                    hasEmitter = any(greaterThan(emitterRadiance, vec3(1e-6)));
                }
            }
        }
    }

    // Jitter first sample to reduce banding. Mix in ray state so horizon rays do
    // not share coherent step planes across neighboring pixels.
    float rayJitter = fract(sin(dot(rayOrigin + rayDir * 17.0, vec3(12.9898, 78.233, 37.719)) + float(payload.seed) * 0.000173) * 43758.5453);
    float t = tNear + rayJitter * baseStep;
    int step = 0;
    uint measuredDensitySamples = 0u;
    uint measuredEmptySegments = 0u;
    uint measuredTopologySegments = 0u;
    uint measuredDensityLeafSegments = 0u;
    while (t < tFar && step < maxSteps) {
        vec3  samplePos = rayOrigin + rayDir * t;

        // Live dense gas: block-majorant skip. Same safety rule as the NanoVDB
        // branch below — a Volume Graph may synthesize density where the source
        // field has none, so skipping is only valid for unmodified density.
        if (vol.volume_type == 4 && vol.source_type == 5 &&
            volumeProgram == MATPROG_NONE) {
            float denseSkip = denseGasEmptyBlockStep(
                vol, samplePos, rayDir, denseSkipCutoff);
            if (denseSkip > baseStep * 1.01) {
                int skippedSegments = max(1, int(floor(denseSkip / baseStep)));
                measuredEmptySegments += uint(skippedSegments);
                measuredTopologySegments += uint(skippedSegments);
                step += skippedSegments;
                t = tNear + (float(step) + rayJitter) * baseStep;
                continue;
            }
        }

        // A Volume Graph may synthesize density in inactive source voxels.
        // Hierarchy skipping is therefore safe only for legacy/baked density.
        if (vdbBuf.address != 0 && volumeProgram == MATPROG_NONE) {
            uint skipKind = 0u;
            float sparseStep = nanoEmptyTileStep(
                vol, samplePos, rayDir, baseStep, vdbBuf, vdbMapH, vdbAcc,
                skipKind);
            if (sparseStep > baseStep * 1.01) {
                int skippedSegments = max(1, int(floor(sparseStep / baseStep)));
                measuredEmptySegments += uint(skippedSegments);
                if (skipKind == 2u)
                    measuredDensityLeafSegments += uint(skippedSegments);
                else
                    measuredTopologySegments += uint(skippedSegments);
                step += skippedSegments;
                t = tNear + (float(step) + rayJitter) * baseStep;
                continue;
            }
        }
        
        float density = sampleDensityAcc(vol, samplePos, vdbBuf, vdbMapH, vdbAcc);
        measuredDensitySamples++;
        float temperature = needsTemperature ? sampleTemperature(vol, samplePos) : 0.0;
        vec3 stepScatterColor = vol.scatter_color;
        vec3 stepAbsorptionColor = vol.absorption_color;
        vec3 stepEmissionColor = vol.emission_color;
        float stepEmissionStrength = vol.emission_intensity;
        float stepAnisotropy = vol.scatter_anisotropy;
        float stepMultiScatter = vol.scatter_multi;
        if (volumeProgram != MATPROG_NONE) {
            float emptyAttribs[MP_ATTRIB_SLOTS];
            for (int ai = 0; ai < MP_ATTRIB_SLOTS; ++ai) emptyAttribs[ai] = 0.0;
            vec3 localPos;
            localPos.x = vol.inv_transform[0]*samplePos.x + vol.inv_transform[1]*samplePos.y + vol.inv_transform[2]*samplePos.z + vol.inv_transform[3];
            localPos.y = vol.inv_transform[4]*samplePos.x + vol.inv_transform[5]*samplePos.y + vol.inv_transform[6]*samplePos.z + vol.inv_transform[7];
            localPos.z = vol.inv_transform[8]*samplePos.x + vol.inv_transform[9]*samplePos.y + vol.inv_transform[10]*samplePos.z + vol.inv_transform[11];
            vec3 normalizedLocalPos = clamp(
                (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-6)),
                vec3(0.0), vec3(1.0));
            MatProgOut vp = evalMaterialProgram(
                volumeProgram, vec2(0.0), samplePos, -rayDir, 0.5, vec3(0.0),
                emptyAttribs, localPos, -rayDir,
                density, temperature,
                (temperature > 0.0) ? clamp(temperature / max(vol.max_temperature, 1.0), 0.0, 1.0) : 0.0,
                0.0, vec3(0.0), samplePos,
                vol.scatter_color, vol.emission_color, max(vol.voxel_size, 1e-6), normalizedLocalPos,
                cam.waterTime);
            if ((vp.volumeWritten & (1u << 0)) != 0u) density = max(vp.volumeDensity, 0.0);
            if ((vp.volumeWritten & (1u << 1)) != 0u) stepScatterColor = max(vp.volumeScatterColor, vec3(0.0));
            if ((vp.volumeWritten & (1u << 2)) != 0u) sigma_s_coeff = max(vp.volumeScatterStrength, 0.0);
            if ((vp.volumeWritten & (1u << 3)) != 0u) stepAbsorptionColor = max(vp.volumeAbsorptionColor, vec3(0.0));
            if ((vp.volumeWritten & (1u << 4)) != 0u) sigma_a_coeff = max(vp.volumeAbsorptionStrength, 0.0);
            if ((vp.volumeWritten & (1u << 5)) != 0u) stepEmissionColor = max(vp.volumeEmissionColor, vec3(0.0));
            if ((vp.volumeWritten & (1u << 6)) != 0u) stepEmissionStrength = max(vp.volumeEmissionStrength, 0.0);
            if ((vp.volumeWritten & (1u << 7)) != 0u) stepAnisotropy = clamp(vp.volumeAnisotropy, -0.99, 0.99);
            if ((vp.volumeWritten & (1u << 8)) != 0u) stepMultiScatter = clamp(vp.volumeMultiScatter, 0.0, 1.0);
        }
        float sigma_s_local = density * sigma_s_coeff;
        // RGB absorption. Black historically meant "neutral absorption" in
        // existing assets, so preserve it as an achromatic fallback. Otherwise
        // Absorption Color identifies the wavelengths removed by the medium.
        vec3 authoredAbsorption = max(stepAbsorptionColor, vec3(0.0));
        float absorptionPeak = max(
            authoredAbsorption.r,
            max(authoredAbsorption.g, authoredAbsorption.b));
        vec3 absorptionWeights = absorptionPeak > 1e-5
            ? authoredAbsorption / absorptionPeak
            : vec3(1.0);
        vec3 sigma_a_local = density * sigma_a_coeff * absorptionWeights;
        vec3 sigma_t_rgb = vec3(sigma_s_local) + sigma_a_local;
        float sigma_t_local = dot(
            sigma_t_rgb, vec3(0.2126, 0.7152, 0.0722));
        // Let low-density regions survive based on scattering optical depth, not
        // raw density alone. This means increasing scatter can actually keep thin
        // edges visible instead of them being discarded by a fixed density gate.
        float sparseCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.04;
        float scatter_keep = clamp((sigma_s_local * baseStep) / sparseCutoff, 0.0, 1.0);
        if (sigma_t_local <= EPSILON) {
            step++;
            t = tNear + (float(step) + rayJitter) * baseStep;
            continue;
        }
        
        // Fixed coverage segment. Beer-Lambert remains analytic over the full
        // segment; adaptive optical substeps will return only with a separate
        // substep budget.
        float dt = min(baseStep, tFar - t);
        if (dt <= 1e-6) break;

        // Current extinction
        vec3 extinction = sigma_t_rgb * dt;
        vec3 sampleTransmittance = exp(-extinction);
        
        // ── Multi-scatter transmittance blend (matches OptiX) ──
        // Blends single-scatter (Beer's law) with a softer 0.25x extinction approximation
        // to model multiple scattering. When scatter_multi > 0, volumes appear brighter
        // and more translucent — matching the OptiX renderer output.
        // Extinction remains Beer-Lambert; multiple scattering redistributes
        // direction instead of making the same medium transmit extra energy.
        vec3 one_minus_sampleT = vec3(1.0) - sampleTransmittance;
        // Representative volume distance for aerial perspective. Weight each
        // sample by the camera-path opacity it actually contributes, not by the
        // procedural AABB entry/exit. Sparse fringe therefore cannot move the
        // apparent atmospheric endpoint to the domain wall.
        float stepOpacityWeight = dot(
            transmittance * one_minus_sampleT,
            vec3(0.2126, 0.7152, 0.0722));
        opticalDepthWeightedT +=
            (t + 0.5 * dt) * max(stepOpacityWeight, 0.0);
        opticalDepthWeight += max(stepOpacityWeight, 0.0);
        
        // ── Volume Emission ──
        // Mode 0 = none, 1 = plain color, 2 = blackbody/color-ramp via temperature grid.
        // Energy-stable integration: multiply by one_minus_sampleT (bounded by 1) instead of dt.
        vec3 emis = vec3(0.0);
        if (vol.emission_mode >= 1) {
            if (vol.emission_mode == 1) {
                // Plain constant-color emission
                emis = stepEmissionColor * stepEmissionStrength * density;
            } else if (vol.emission_mode >= 2) {
                // Blackbody / color-ramp — temperature grid first, density fallback for parity.
                if (temperature <= 0.0) temperature = density;

                // _ext_reserved[6] is foam opacity for Surface SDF (type 4);
                // all other volumes receive authored minimum temperature here.
                float rangeMin = (vol.source_type == 4)
                    ? 0.0 : max(vol._ext_reserved[6], 0.0);
                float rangeMax = (vol.max_temperature > rangeMin + 1.0)
                    ? vol.max_temperature : (rangeMin + 1500.0);
                // Simulation grids arrive as Kelvin-scaled heat. Normalized
                // fallbacks map into the same authored temperature interval.
                float authoredKelvin = (temperature > 20.0)
                    ? clamp(temperature, rangeMin, rangeMax)
                    : mix(rangeMin, rangeMax, clamp(temperature, 0.0, 1.0));
                float t_ramp = clamp(
                    (authoredKelvin - rangeMin) / max(rangeMax - rangeMin, 1.0),
                    0.0, 1.0);

                vec3 e_color;
                if (vol.color_ramp_enabled != 0 && vol.ramp_stop_count > 0) {
                    e_color = sampleColorRamp(vol, clamp(t_ramp * vol.temperature_scale, 0.0, 1.0));
                } else {
                    // The authored interval constrains blackbody color; Temp
                    // Scale then provides the intentional artistic offset.
                    e_color = blackbodyToRGB(authoredKelvin * vol.temperature_scale);
                }
                // ── Combustion reaction boost ───────────────────────────────
                // Temperature alone cannot tell a flame from the hot smoke
                // sitting above it: both are hot, so the whole plume glows and
                // the fire reads as luminous smoke rather than as a flame with
                // a core. The gas solver already knows the difference — its
                // bounded `interaction` field is the burn RATE, nonzero only
                // where fuel is being consumed THIS step.
                //
                // Applied as an ADDITION on top of the existing temperature
                // emission, not as a mask over it. Masking would be the more
                // physical answer (smoke that has stopped burning should go
                // dark) but it would also silently re-light every existing
                // scene; the reaction zone becoming a distinct, brighter source
                // is the signal that was missing, and it cannot dim anything.
                // Kept density-coupled, so empty cells never emit.
                vec3 baseEmission = e_color * density * vol.blackbody_intensity;
                vec3 reactionEmission = vec3(0.0);
                if (vol.flame_address != 0) {
                    float reaction = sampleFlame(vol, samplePos);
                    if (reaction > 0.0) {
                        reactionEmission = baseEmission * reaction;
                    }
                }
                emis = clampVolumeRadiance(baseEmission + reactionEmission, 64.0);
            }
        }
        
        // ── In-Scattering (Direct Lighting) ──
        if (sigma_s_local > 0.0) {
            vec3 inscatter = vec3(0.0);
            
            // Sample scene lights for object volumes. For the procedural sky cloud,
            // Nishita already provides the sun/sky block below; sampling the default
            // directional light here doubles the light march cost and shifts parity
            // away from the OptiX sky-cloud path.
            bool useSceneLights = !(worldData.w.mode == 2 && (vol.volume_type == 3 || vol.source_type == 3));
            if (useSceneLights && cam.lightCount > 0u) {
                for (int ls = 0; ls < sampledLightCount; ls++) {
                    int li = (ls == 0) ? sampledLight0 : sampledLight1;
                    LightData light = lights.l[li];
                    int lightType = int(light.position.w + 0.5);

                    vec3  lightDir;
                    float lightDist;
                    float lightAtten = 1.0;
                    bool  lightInverseSquare = true;
                    // Only an area light consumes randoms: point/spot/directional
                    // are delta sources, and drawing for them would churn the RNG
                    // stream every step for nothing.
                    //
                    // The area sample varies per step while the shadow-march
                    // result is cached across `shadow_stride` steps, so a cached
                    // transmittance can belong to a slightly different point on
                    // the light. The offset is bounded by the light's own extent
                    // and decorrelates the noise; pinning the sample per ray
                    // instead would trade that for visible banding.
                    float ru = 0.5, rv = 0.5;
                    if (lightType == 2) {
                        ru = rnd(payload.seed);
                        rv = rnd(payload.seed);
                    }
                    if (!volSampleLight(light, samplePos, ru, rv,
                                        lightDir, lightDist, lightAtten,
                                        lightInverseSquare)) {
                        continue;   // outside the cone / behind an area light
                    }
                    if (lightInverseSquare) {
                        // Analytic 1/r^2 across the whole segment instead of a
                        // point sample at its start. Without this the width and
                        // brightness of a light shaft track the step size rather
                        // than the light.
                        //
                        // Area lights sample a point on the shape, so their
                        // closest-approach geometry is that sample's, not the
                        // light origin's — reconstruct the sampled position from
                        // the direction and distance already returned.
                        vec3 sampledLightPos = samplePos + lightDir * lightDist;
                        lightAtten *= volSegmentInvSqAverage(
                            rayOrigin, rayDir, sampledLightPos, t, t + dt);
                    }

                    // Phase function evaluation
                    float cosTheta = dot(rayDir, lightDir);
                    float phase = dualLobeHG(cosTheta, stepAnisotropy,
                                             vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                    vec3 scatteringAlbedoRGB =
                        vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
                    float scatteringAlbedo = dot(
                        scatteringAlbedoRGB, vec3(0.2126, 0.7152, 0.0722));
                    phase = mix(
                        phase, 1.0 / (4.0 * PI),
                        clamp(stepMultiScatter * scatteringAlbedo, 0.0, 1.0));
                    
                    // Light march transmittance through volume toward light.
                    // Geometry occlusion is NOT checked per-step (matching OptiX raymarch_volumetric_object
                    // behavior): solid objects are handled by TLAS traversal order, not per-sample rays.
                    // This prevents hard black shadows from solids inside the volume.
                    float shadowTr = 1.0;
                    if (sigma_t_local * dt > 0.02) {
                        bool shadowValid = (ls == 0)
                            ? cachedSceneShadow0Valid : cachedSceneShadow1Valid;
                        bool updateShadow = !shadowValid || (step % shadowStride) == 0;
                        if (updateShadow) {
                            float shadowMaxDist = min(
                                lightDist,
                                max(baseStep, volumeExitDistance(vol, samplePos, lightDir)));
                            float evaluatedShadow = lightMarchAcc(
                                vol, samplePos, lightDir, shadowMaxDist,
                                vdbBuf, vdbMapH, vdbAcc, -1.0);
                            if (ls == 0) {
                                cachedSceneShadow0 = evaluatedShadow;
                                cachedSceneShadow0Valid = true;
                            } else {
                                cachedSceneShadow1 = evaluatedShadow;
                                cachedSceneShadow1Valid = true;
                            }
                        }
                        shadowTr = (ls == 0) ? cachedSceneShadow0 : cachedSceneShadow1;
                    }
                    
                    vec3 lightColor = light.color.rgb * light.color.a;
                    inscatter += lightWeight * lightColor * lightAtten * phase * shadowTr
                               * stepScatterColor * scatteringAlbedoRGB;
                }
            }
            
            // ── Fire (volume emission) NEE ──────────────────────────────
            // Same estimator shape as a scene light: analytic 1/r^2 across the
            // segment, phase toward the emitter, and the medium's own
            // transmittance in between. Without this the fire only reaches the
            // surrounding smoke through random bounces, which is exactly the
            // noise the user sees.
            if (hasEmitter) {
                vec3 toEmitter = emitterPos - samplePos;
                float emitterDist = length(toEmitter);
                if (emitterDist > 1e-3) {
                    vec3 emitterDir = toEmitter / emitterDist;
                    float ePhase = dualLobeHG(dot(rayDir, emitterDir), stepAnisotropy,
                                              vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                    vec3 eAlbedoRGB = vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
                    ePhase = mix(ePhase, 1.0 / (4.0 * PI),
                                 clamp(stepMultiScatter *
                                       dot(eAlbedoRGB, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0));
                    // Strided like the other shadow terms; the emitter is
                    // ray-constant so the cached value stays valid for it.
                    if (!cachedEmitterShadowValid || (step % shadowStride) == 0) {
                        cachedEmitterShadow = lightMarchAcc(
                            vol, samplePos, emitterDir,
                            min(emitterDist, max(baseStep,
                                volumeExitDistance(vol, samplePos, emitterDir))),
                            vdbBuf, vdbMapH, vdbAcc, -1.0);
                        cachedEmitterShadowValid = true;
                    }
                    float eFalloff = volSegmentInvSqAverage(
                        rayOrigin, rayDir, emitterPos, t, t + dt);
                    inscatter += emitterRadiance * eFalloff * ePhase
                               * cachedEmitterShadow * stepScatterColor * eAlbedoRGB;
                }
            }

            // Sun/sky light contribution (if Nishita sky active)
            if (worldData.w.mode == 2) {
                vec3 sunDir = normalize(worldData.w.sunDir);
                float cosSun = dot(rayDir, sunDir);
                float sunPhase = dualLobeHG(cosSun, stepAnisotropy,
                                            vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                vec3 scatteringAlbedoRGB =
                    vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
                float scatteringAlbedo = dot(
                    scatteringAlbedoRGB, vec3(0.2126, 0.7152, 0.0722));
                sunPhase = mix(
                    sunPhase, 1.0 / (4.0 * PI),
                    clamp(stepMultiScatter * scatteringAlbedo, 0.0, 1.0));
                
                float sunShadowTr = 1.0;
                if (sigma_t_local * dt > 0.02) {
                    bool updateSunShadow = !cachedSunShadowValid || (step % shadowStride) == 0;
                    if (updateSunShadow) {
                        float sunShadowMaxDist = max(
                            baseStep, volumeExitDistance(vol, samplePos, sunDir));
                        cachedSunShadow = lightMarchAcc(
                            vol, samplePos, sunDir, sunShadowMaxDist,
                            vdbBuf, vdbMapH, vdbAcc, -1.0);
                        cachedSunShadowValid = true;
                    }
                    sunShadowTr = cachedSunShadow;
                }
                
                vec3 sunLi = sampleTransmittanceLUT(samplePos, sunDir) * worldData.w.sunColor * worldData.w.sunIntensity;
                inscatter += sunLi * sunPhase * sunShadowTr
                           * stepScatterColor * scatteringAlbedoRGB;
            }
            
            // 3. Sky/Ambient lighting (closer to CPU world.evaluate(up) behavior)
            float thin_scatter = scatter_keep * scatter_keep;
            vec3 ambientAlbedo =
                vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
            float ambientAlbedoLuma = dot(
                ambientAlbedo, vec3(0.2126, 0.7152, 0.0722));
            // Ambient is incident radiance too: letting it reach every dense
            // sample unattenuated fills the entire cloud with a flat white
            // source and erases density/depth cues. Use a cached representative
            // sky-hemisphere march so cores darken while sky-facing lobes remain
            // lit. A minimum stride of four bounds the Vulkan RT cost.
            // Hemispherical ambient changes slowly along a primary ray. Eight
            // samples of reuse removes a major nested NanoVDB-march cost without
            // altering its Beer-Lambert visibility.
            int ambientShadowStride = max(shadowStride, 8);
            bool updateAmbientShadow =
                !cachedAmbientShadowValid || (step % ambientShadowStride) == 0;
            if (updateAmbientShadow && sigma_t_local * dt > 0.01) {
                float ambientShadowMaxDist = max(
                    baseStep,
                    volumeExitDistance(vol, samplePos, ambientLightDir));
                cachedAmbientShadow = lightMarchAcc(
                    vol, samplePos, ambientLightDir, ambientShadowMaxDist,
                    vdbBuf, vdbMapH, vdbAcc, 1.0);
                cachedAmbientShadowValid = true;
            }
            inscatter += ambientSky * stepScatterColor * ambientAlbedo
                       * thin_scatter * cachedAmbientShadow;

            // Bounded higher-order sky scattering approximation. Recycle only
            // a fraction of the ambient energy blocked on the representative
            // sky path; unlike emission this vanishes when external sky light
            // vanishes, remains density/depth dependent, and never exceeds the
            // original unoccluded ambient budget:
            //   visibility = T_sky + multi * albedo * P(2+) * (1-T_sky) <= 1.
            float trappedSky = 1.0 - cachedAmbientShadow;
            float localOpticalDepth = sigma_t_local * dt;
            float higherOrderProbability = 1.0 - exp(-localOpticalDepth);
            float multiSkyVisibility =
                stepMultiScatter * ambientAlbedoLuma * trappedSky
                * higherOrderProbability;
            inscatter += ambientSky * stepScatterColor * ambientAlbedo
                       * thin_scatter * multiSkyVisibility;
            
            // CPU parity integration:
            // step_color = source * (1 - step_transmittance)
            // accumulated += step_color * current_transparency
            accumulated_radiance += transmittance * (inscatter + emis) * one_minus_sampleT;
        } else if (any(greaterThan(emis, vec3(0.0)))) {
            // Emission-only medium segment
            accumulated_radiance += transmittance * emis * one_minus_sampleT;
        }
        
        // Update transmittance
        transmittance *= sampleTransmittance;
        
        // ── Stochastic Scatter Event (Delta Tracking) ──
        // Probability of scattering at this step
        // No stochastic volume continuation: the direct single-scattering
        // estimator above already accounts for redirected radiance.
        
        // Early termination if transmittance is negligible
        if (max(transmittance.r, max(transmittance.g, transmittance.b)) < 0.001) {
            break;
        }
        step++;
        t = tNear + (float(step) + rayJitter) * baseStep;
    }
    
    // ══════════════════════════════════════════════════════════════════════════
    // OUTPUT — Set payload for path tracer integration
    // ══════════════════════════════════════════════════════════════════════════
    
    // Sanitize against NaN/Inf. A single non-finite sample (e.g. a degenerate
    // ray when two coincident volumes — foam fog + fluid iso surface — share the
    // domain AABB) otherwise poisons the accumulation buffer permanently → the
    // whole domain goes black until accumulation resets. Drop the bad sample.
    if (any(isnan(accumulated_radiance)) || any(isinf(accumulated_radiance)))
        accumulated_radiance = vec3(0.0);
    if (any(isnan(transmittance)) || any(isinf(transmittance)) ||
        any(lessThan(transmittance, vec3(0.0))))
        transmittance = vec3(1.0);
    float transmittanceLuma = dot(
        transmittance, vec3(0.2126, 0.7152, 0.0722));

    // Accumulated in-scattered radiance
    payload.radiance  = accumulated_radiance;
    payload.skipAABBs = false; // default; overridden below if solid found
    // ── Solid surface found inside the volume: hand off to closesthit ──
    // March was already clamped to solidT. Now position the scatter ray
    // just before the solid surface and tell raygen to skip AABBs next
    // bounce (gl_RayFlagsSkipAABBEXT) so the triangle closesthit fires.
    if (solidT >= 0.0) {
        volumeRecordRay(
            measuredDensitySamples,
            measuredEmptySegments,
            measuredTopologySegments,
            measuredDensityLeafSegments,
            VOLUME_MARCH_COMPLETED);
        payload.attenuation  *= transmittance;
        // No step-back epsilon — see the note at the first solid handoff. This was
        // the site the camera ray actually goes through for a gas domain.
        payload.scatterOrigin = rayOrigin;
        payload.scatterDir    = rayDir;
        payload.scattered     = true;
        payload.skipAABBs     = true;
        return;
    }

    float volumeContribution = max(max(accumulated_radiance.r, accumulated_radiance.g), accumulated_radiance.b);
    float volumeOpacity = 1.0 - transmittanceLuma;
    // Keep ultra-thin fog tails out of the primary auxiliary buffers. They were
    // being treated as first-hit geometry, which overemphasized weak density
    // regions in Vulkan RT denoiser output compared to OptiX.
    bool primaryVolumeInteraction =
        volumeOpacity > 0.04 || volumeContribution > 5e-4;
    if ((payload.primaryMeta & PL_PRIMARY_DONE) == 0u && primaryVolumeInteraction) {
        float representativeVolumeT = opticalDepthWeight > 1e-8
            ? opticalDepthWeightedT / opticalDepthWeight
            : 0.5 * (tNear + tFar);
        payload.primaryARG  = packHalf2x16(vol.scatter_color.rg);
        payload.primaryABT  = packHalf2x16(vec2(vol.scatter_color.b, transmittanceLuma));
        // PL_PRIMARY_VOLUME reinterprets primaryNrm as float bits containing
        // the optical-depth centroid distance. Raygen reconstructs the volume
        // AOV normal as -primaryRayDir without increasing payload size.
        payload.primaryNrm  = floatBitsToUint(representativeVolumeT);
        // Material id stays 0xFFFF: volumes have no scene material index.
        payload.primaryMeta = (payload.primaryMeta & PL_DISP_MASK)
                            | PL_PRIMARY_DONE | PL_PRIMARY_VOLUME
                            | PL_PRIMARY_VOLUME_DEPTH | PL_MATID_MASK;
    }

    if (didScatter && transmittanceLuma < 0.99) {
        // Scatter event — continue path with new direction
        vec3 scatterPos = rayOrigin + rayDir * scatterT;
        
        // Choose lobe for direction sampling
        float g = (rnd(payload.seed) < vol.scatter_lobe_mix) 
                  ? vol.scatter_anisotropy 
                  : vol.scatter_anisotropy_back;
        vec3 newDir = sampleHG(rayDir, g, payload.seed);
        
        payload.scatterOrigin = scatterPos;
        payload.scatterDir    = newDir;
        payload.attenuation  *= vol.scatter_color * transmittance;
        payload.scattered     = true;
    } else {
        // No scatter — attenuate throughput by volume transmittance
        payload.attenuation *= transmittance;

        // Set scattered = true with original direction to let the ray continue through
        // Ensure forward progress to avoid re-hitting the same boundary when camera is inside.
        payload.scatterOrigin = layeredSurfaceT > 0.0
            ? rayOrigin + rayDir * max(tNear, layeredSurfaceT - 0.0005)
            : rayOrigin + rayDir * (tFar + 0.002);
        payload.scatterDir    = rayDir;
        payload.scattered     = (transmittanceLuma > 0.01); // Stop if fully absorbed
        payload.skipGasVolumes =
            layeredSurfaceT > 0.0 && payload.scattered;
        if (volumeOpacity <= 0.04 && volumeContribution <= 5e-4) {
            payload.bounceType = BOUNCE_TRANSPARENT;
        }
    }
    uint marchOutcome = transmittanceLuma <= 0.01
        ? VOLUME_MARCH_EXTINCTION
        : ((step >= maxSteps && t < tFar)
            ? VOLUME_MARCH_STEP_BUDGET
            : VOLUME_MARCH_COMPLETED);
    volumeRecordRay(
        measuredDensitySamples,
        measuredEmptySegments,
        measuredTopologySegments,
        measuredDensityLeafSegments,
        marchOutcome);
}
