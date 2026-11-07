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
#extension GL_EXT_ray_query                            : require
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
#include "surface_sdf_march_policy.glsl"


layout(location = 0) rayPayloadInEXT RayPayload payload;
#include "volume_foreground_dispatch.glsl"
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
    float random_walk_enabled;    // 1 = path-traced multiple scattering (volume_closesthit)
    float random_walk_max_events; // 8..512 scattering events per walk
    float random_walk_exact_events; // 1..512 true-phase events before the similarity switch
    float _retired_cloud[5];   // retired Faz 3b: sky clouds are no longer a volume (ATMOSPHERE_CLOUDS.md)
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
    // Appended at byte 624; total record stride is now 672 bytes.
    float density_curve[12];
    float emission_mask[4];
};
#include "volume_density_curve.glsl"


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
    int   fogEnabled;   float fogDensity; float fogHeight; float fogFalloff;
    float fogDistance;  float fogAnisotropy; vec3 fogAlbedo; float _pad4;
    int   godRaysEnabled; float godRaysIntensity; float godRaysDensity; int godRaysSamples;
    int   aerialFroxelReady; uint volumeMarchLod; uint volumeLightCacheLo; uint volumeLightCacheHi;
    int   weatherEnabled; int weatherType; float weatherIntensity; float weatherDensity;
    vec3  weatherWindDirection; float weatherWindSpeed;
    float weatherPrecipitationScale; float _weatherPad0; float weatherSurfaceWetness; float weatherSurfaceAccumulation;
    float weatherSurfaceSettling; float weatherSurfaceHeight;
    int   _weatherPad1; int weatherSurfaceResponseEnabled;
    int   envTexSlot;   float envIntensity; float envRotation; int _pad5; // nishitaLutReady
    int   envOverlayEnabled; int envOverlayBlendMode; float envOverlayIntensity; float envOverlayRotation;
    uvec2 transmittanceLUT; uvec2 skyviewLUT; uvec2 multiScatterLUT; uvec2 aerialPerspectiveLUT;
};

layout(set = 0, binding = 7, scalar) readonly buffer WorldBuffer { VkWorldDataExtended w; } worldData;
layout(set = 0, binding = 8) uniform sampler2D atmosphereLUTs[4];
#include "cloud_hit_segment.glsl"

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

// Surface occlusion is independent of fog attenuation. Separate participating
// media remain handled by the existing volume path.
float rtSurfaceSdfShadowTransmittance(vec3 origin, vec3 direction, float maxDist);
float computeVolumeShadowTransmittance(vec3 shadowOrigin, vec3 lightDir, float maxDist,
                                       int lightIndex) {
    return rtSurfaceSdfShadowTransmittance(shadowOrigin, lightDir, maxDist);
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

// The isosurface is an SSS boundary the triangle probe cannot see; see
// sssCustomExit below and the hook comment in bsdf_scatter.glsl.
#define SSS_CUSTOM_EXIT
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
    // skipped (caller sets skipVolumeAABBs) so the triangle closest-hit fires on that
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
float volSegmentInvSqAverageExt(vec3 rayOrigin, vec3 rayDir, vec3 lightPos,
                                float t0, float t1, float minH) {
    float dt = t1 - t0;
    if (dt <= 1e-9) return 0.0;
    vec3 toLight = lightPos - rayOrigin;
    float tc = dot(toLight, rayDir);              // closest approach along the ray
    float h2 = max(dot(toLight, toLight) - tc * tc, 0.0);
    float h = max(sqrt(h2), max(minH, 1e-4));
    float a0 = atan((t0 - tc) / h);
    float a1 = atan((t1 - tc) / h);
    return (a1 - a0) / (h * dt);
}

float volSegmentInvSqAverage(vec3 rayOrigin, vec3 rayDir, vec3 lightPos,
                             float t0, float t1) {
    return volSegmentInvSqAverageExt(rayOrigin, rayDir, lightPos, t0, t1, 1e-4);
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

// ============================================================
// NanoVDB GLSL Setup
// ============================================================
#include "volume_hit_sampling.glsl"
#define RT_SDF_SHADOW_SAMPLE sampleNanoVDBFloatTrilinearAcc
#include "surface_sdf_rt_shadow.glsl"

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
        // A liquid isosurface is source_type == 4 whatever storage it currently
        // uses. Everything else is not our business.
        if (surface.is_active == 0 || surface.source_type != 4) continue;
        volumeRecordArbiterCandidate();

        float boxNear, boxFar;
        if (!volumeRayInterval(surface, rayOrigin, rayDir, boxNear, boxFar)) {
            volumeRecordArbiterNoBox();
            continue;
        }
        float beginT = max(rangeNear, max(boxNear, 0.001));
        float endT = min(rangeFar, boxFar);
        if (endT <= beginT || beginT >= nearestHit) {
            volumeRecordArbiterEmptyRange();
            continue;
        }

        // ★★★ "Cannot evaluate it here" is NOT "it is not there".
        //
        // This test used to also demand volume_type == 2 and a non-zero
        // vdb_grid_address, i.e. a HOST NanoVDB grid. A liquid surface published
        // as a live dense GPU field (volume_type 4) failed it silently, the
        // arbiter returned -1, and the gas march then ran its whole AABB and
        // teleported the ray to tFar — straight PAST the liquid. Symptom: a black
        // band at the fluid boundary whenever gas and liquid share a domain.
        //
        // ★ And it is why putting a BOX inside the domain "fixed" it: a triangle
        // is found by the solid probe, which knows nothing about NanoVDB, so the
        // handoff fired and the re-trace reached the liquid through the TLAS.
        // Delete the box and this arbiter is the only remaining path.
        //
        // The conservative action for a surface we cannot sample is to hand the
        // ray over at the box entry and let the liquid's OWN closest-hit decide.
        // Erring the other way erases geometry that is plainly there, which is
        // the strictly worse failure — the same reasoning as the solid probe's
        // "err toward PROBING" rule.
        // A live GPU surface (isLiveDenseSurface) is sampleable too: its proxy is
        // a dense device grid, read by sampleIsoField's dense branch. Only the
        // NanoVDB header walk below must not run on it — those bytes are floats.
        const bool surfaceDense = isLiveDenseSurface(surface);
        if (!surfaceDense &&
            (surface.volume_type != 2 || surface.vdb_grid_address == 0)) {
            volumeRecordArbiterReject();
            nearestHit = min(nearestHit, beginT);
            continue;
        }

        pnanovdb_buf_t surfaceBuf;
        surfaceBuf.address = surfaceDense ? uint64_t(0) : surface.vdb_grid_address;
        pnanovdb_map_handle_t mapH;
        mapH.address.byte_offset = 0u;
        pnanovdb_readaccessor_t acc;
        if (!surfaceDense) {
            pnanovdb_grid_handle_t gridH;
            gridH.address.byte_offset = 0u;
            pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(surfaceBuf, gridH);
            pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(surfaceBuf, treeH);
            mapH = pnanovdb_grid_get_map(surfaceBuf, gridH);
            pnanovdb_readaccessor_init(acc, rootH);
        }

        const float ISO = 0.5;
        float span = endT - beginT;
        int cap = clamp(surface.max_steps, 32, 512);
        int steps;
        float step = surfaceSdfMarchStep(span, surface.voxel_size,
                                        surface.step_size, cap, steps);
        float t0s = beginT;
        float d0 = sampleIsoField(
            surface, rayOrigin + rayDir * t0s, surfaceBuf, mapH, acc);
        bool startedInside = d0 > ISO;
        bool foundCrossing = false;
        #include "surface_sdf_arbiter_search.glsl"

        if (!foundCrossing) volumeRecordArbiterNoCrossing();
        if (startedInside) volumeRecordArbiterStartedInside(foundCrossing);
    }
    return nearestHit <= rangeFar ? nearestHit : -1.0;
}

// ── SSS exit through the liquid isosurface ──────────────────────────────────
// Set by the isosurface block right around its scatterPrincipled call; every
// other scatter in this shader leaves it off, so a walk started from a
// triangle is never clipped by an unrelated field.
bool g_sssIsoEnabled = false;
uint g_sssIsoVolume = 0u;

bool sssCustomExit(vec3 origin, vec3 dir, float tMax, out float tHit, out vec3 exitN) {
    tHit = 0.0;
    exitN = vec3(0.0, 1.0, 0.0);
    if (!g_sssIsoEnabled || !(tMax > 0.0)) return false;
    VkVolumeInstance sv = volumes.v[g_sssIsoVolume];

    pnanovdb_buf_t sBuf;
    sBuf.address = sv.vdb_grid_address;
    pnanovdb_grid_handle_t gridH;
    gridH.address.byte_offset = 0u;
    pnanovdb_map_handle_t mapH;
    pnanovdb_readaccessor_t acc;
    if (sv.volume_type == 2 && sv.vdb_grid_address != 0) {
        pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(sBuf, gridH);
        pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(sBuf, treeH);
        mapH = pnanovdb_grid_get_map(sBuf, gridH);
        pnanovdb_readaccessor_init(acc, rootH);
    }

    // Same iso, step and bisection as nearestSurfaceSDFCrossing, but only the
    // INSIDE -> OUTSIDE transition counts, and only after the segment has
    // actually been inside: the walk's first origin sits a few ULPs under the
    // entry hit, possibly still in the band's outer half, and treating that as
    // an instant exit would hand the ray straight back where it came in.
    const float ISO = 0.5;
    const int SSS_ISO_CAP = 128;
    float fineStep = clamp(sv.step_size, sv.voxel_size * 0.1, sv.voxel_size * 0.5);
    float stepLen = max(0.0005, max(fineStep, tMax / float(SSS_ISO_CAP)));
    int steps = min(int(ceil(tMax / stepLen)), SSS_ISO_CAP);
    float t0 = 0.0;
    float d0 = sampleIsoField(sv, origin, sBuf, mapH, acc);
    bool wasInside = d0 >= ISO;
    for (int s = 0; s < steps; ++s) {
        float t1 = min(t0 + stepLen, tMax);
        float d1 = sampleIsoField(sv, origin + dir * t1, sBuf, mapH, acc);
        if (wasInside && d1 < ISO) {
            float a = t0;
            float b = t1;
            for (int r = 0; r < 5; ++r) {
                float m = 0.5 * (a + b);
                if (sampleIsoField(sv, origin + dir * m, sBuf, mapH, acc) >= ISO) a = m;
                else b = m;
            }
            tHit = 0.5 * (a + b);
            vec3 p = origin + dir * tHit;
            float h = max(0.001, sv.voxel_size);
            vec3 grad = vec3(
                sampleIsoField(sv, p + vec3(h, 0.0, 0.0), sBuf, mapH, acc) - sampleIsoField(sv, p - vec3(h, 0.0, 0.0), sBuf, mapH, acc),
                sampleIsoField(sv, p + vec3(0.0, h, 0.0), sBuf, mapH, acc) - sampleIsoField(sv, p - vec3(0.0, h, 0.0), sBuf, mapH, acc),
                sampleIsoField(sv, p + vec3(0.0, 0.0, h), sBuf, mapH, acc) - sampleIsoField(sv, p - vec3(0.0, 0.0, h), sBuf, mapH, acc));
            // The field grows INTO the liquid, so outward is -grad.
            exitN = length(grad) > 1e-8 ? -normalize(grad) : dir;
            return true;
        }
        wasInside = wasInside || d1 >= ISO;
        t0 = t1;
        if (t0 >= tMax) break;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// PATH-TRACED MULTIPLE SCATTERING — random walk inside one volume
// ═══════════════════════════════════════════════════════════════════════════
// ★ Why this exists. The march below integrates SINGLE scattering analytically
// and stands in for the rest with multi_scatter (phase isotropization plus a
// softened self-shadow). That is fine for smoke, whose albedo is moderate. A
// high-albedo medium (snow, cloud, foam: albedo 0.95..0.999) gets most of its
// brightness from tens to hundreds of bounces, and no single-scatter estimate
// can produce it: measured 2026-10-05, a snow fog under a sun rendered GREY
// against the lit ground it was lying on.
//
// The walk samples real scattering events, does next-event estimation (scene
// light, sun = its Directional light) at each, and picks the next direction from the phase
// function. The whole walk runs HERE and returns to raygen as ONE diffuse
// bounce: hundreds of internal events must not drain the path's bounce budget,
// and the sky arrives the honest way, through the ray that finally escapes.
//
// ★ Free flights use REGULAR tracking (accumulate optical depth with the
// density sampled at each segment midpoint, solve the exact hit inside the
// segment), not delta tracking. Delta tracking needs a majorant, and every
// density transform in sampleDensityAcc (remap without upper clamp, material
// noise, fog erosion already baked into the grid) would have to be bounded for
// it. Regular tracking needs no bound, and the same empty-space skips as the
// march keep thin regions cheap.
//
// Known approximation: flights are sampled on the LARGEST channel extinction
// (sigma_s + sigma_a) and every real collision scatters with weight
// scatter_color * sigma_s / that. Chromatic absorption is therefore slightly
// over-darkened in its less-absorbing channels; for a grey medium (snow) it is
// exact.

#if !defined(RT_VOLUME_SURFACE_STAGE)
#include "volume_hit_random_walk.glsl"
#endif

#include "volume_overlap_selection.glsl"
// Incoming PL_CAMERA_SEGMENT flag, captured before the payload is rewritten.
// Camera-only fire scattering and the secondary march LOD share this policy.
bool volumeCameraSegment = true;
#if !defined(RT_VOLUME_SURFACE_STAGE)
#include "volume_sun_shadow_cache.glsl"
#include "volume_light_cache.glsl"
#include "volume_light_cache_fill.glsl"

#include "volume_internal_rt.glsl"
#include "volume_radiance_cache.glsl"
#include "volume_phase_cache.glsl"
#include "volume_mixture.glsl"
#endif

#if !defined(RT_VOLUME_SURFACE_STAGE)
#include "volume_cached_segment.glsl"
#endif

// A cached shadow refreshes once per `stride` samples. With an in-scatter
// stride k > 1 lighting runs only on every k-th sample, and `step % stride == 0`
// could fall between evaluations for the whole ray. Refresh on the evaluated
// sample inside each window instead; k = 1 is exactly `step % stride == 0`.
bool volumeShadowDue(int step, int stride, int lightStride) {
    int window = max(stride, lightStride);
    return (step % window) < lightStride;
}

void shadeVolumeHit() {
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
        payload.skipVolumeAABBs = false;
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
        payload.skipVolumeAABBs = false;
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
        payload.skipVolumeAABBs = false;
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
    selectForegroundGas(rayOrigin, rayDir, volCount, payload.volumeTraversalMask,
                        volIdx, vol, tNear, tFar, cameraInsideVolume);
#if defined(RT_VOLUME_SURFACE_STAGE)
    if (vol.source_type != 4) {
        payload.volumeTraversalMask = VOLUME_FOREGROUND_GAS_DISPATCH;
        return;
    }
#endif
    float selectedVolumeSpan = tFar - tNear;
    // Coexisting gas/fog: march only up to the next membership change, and fold
    // a second overlapping gas volume into this march as a companion medium.
    // See volume_overlap_selection.glsl for why sequential marching is wrong.
    int companionIdx = resolveGasOverlapSegment(
        rayOrigin, rayDir, volCount, volIdx, vol, tNear, tFar, selectedVolumeSpan);
    bool hasCompanion = companionIdx >= 0;
    VkVolumeInstance comp = hasCompanion ? volumes.v[uint(companionIdx)] : vol;
    pnanovdb_buf_t        compBuf;
    pnanovdb_map_handle_t compMapH;
    pnanovdb_readaccessor_t compAcc;
    {
        compBuf.address = (hasCompanion && comp.volume_type == 2 && comp.vdb_grid_address != 0)
            ? comp.vdb_grid_address : uint64_t(0);
        if (compBuf.address != 0) {
            pnanovdb_grid_handle_t gridH; gridH.address.byte_offset = 0u;
            pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(compBuf, gridH);
            pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(compBuf, treeH);
            compMapH = pnanovdb_grid_get_map(compBuf, gridH);
            pnanovdb_readaccessor_init(compAcc, rootH);
        } else {
            compMapH.address.byte_offset = 0u;
            pnanovdb_root_handle_t dummyRoot; dummyRoot.address.byte_offset = 0u;
            pnanovdb_readaccessor_init(compAcc, dummyRoot);
        }
    }
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
#if !defined(RT_VOLUME_FOG_STAGE)
    if (vol.source_type == 4) {
#include "volume_surface_shading.glsl"
    }
#endif

#if !defined(RT_VOLUME_SURFACE_STAGE)
#include "volume_fog_shading.glsl"
#endif
}

void main() {
    uint pathFlags = payload.primaryMeta & (PL_CAMERA_SEGMENT | PL_NO_SKY_RADIANCE);
    volumeInstrumentationSetSegment((pathFlags & PL_CAMERA_SEGMENT) != 0u);
    volumeCameraSegment = (pathFlags & PL_CAMERA_SEGMENT) != 0u;
    shadeVolumeHit();
    if (payload.volumeTraversalMask == VOLUME_FOREGROUND_GAS_DISPATCH) {
        return;
    }
    payload.primaryMeta |= pathFlags;
    cloudApplyVolumeSegment();
}
