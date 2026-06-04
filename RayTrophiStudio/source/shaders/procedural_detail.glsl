// procedural_detail.glsl
// Shared procedural surface detail utilities for RayTrophi shaders.
// Compatible with GLSL 4.5+ (raster) and 4.6 (ray tracing).
//
// API
//   vec2  pd_tileBreak   (vec2 uv, vec3 worldPos, float strength)  → perturbed UV
//   float pd_dirt        (vec3 worldPos, float scale)               → [0, 1]
//   float pd_roughnessVar(vec3 worldPos, float scale)               → [-0.5, +0.5]
//
// Usage pattern (material_preview / closesthit):
//   if (mat.micro_detail_strength > 0.0) {
//       float sc  = max(mat.micro_detail_scale, 0.5);
//       float str = mat.micro_detail_strength;
//       uv = pd_tileBreak(uv, worldPos, 0.25 * str);
//       // ... sample textures with perturbed uv ...
//       float dirtFactor = pd_dirt(worldPos, sc) * str;
//       vec3 dirtColor   = vec3(0.14, 0.10, 0.08);
//       albedo = mix(albedo, albedo * dirtColor, dirtFactor);
//       roughness = clamp(roughness + pd_roughnessVar(worldPos, sc) * str * 0.5, 0.04, 1.0);
//   }

// ─── Hash ─────────────────────────────────────────────────────────────────────

float pd_hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float pd_hash3(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

// ─── Value Noise ──────────────────────────────────────────────────────────────

float pd_vnoise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // quintic — eliminates block-edge artifacts
    return mix(
        mix(pd_hash2(i),           pd_hash2(i + vec2(1.0, 0.0)), f.x),
        mix(pd_hash2(i + vec2(0.0, 1.0)), pd_hash2(i + vec2(1.0, 1.0)), f.x),
        f.y);
}

float pd_vnoise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // quintic
    return mix(
        mix(mix(pd_hash3(i),                 pd_hash3(i + vec3(1,0,0)), f.x),
            mix(pd_hash3(i + vec3(0,1,0)),   pd_hash3(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(pd_hash3(i + vec3(0,0,1)),   pd_hash3(i + vec3(1,0,1)), f.x),
            mix(pd_hash3(i + vec3(0,1,1)),   pd_hash3(i + vec3(1,1,1)), f.x), f.y),
        f.z);
}

// ─── fBm (4-octave, domain-rotated from original p) ──────────────────────────
// All rotations applied to the ORIGINAL p — not cascaded through scaled/offset
// intermediates (which would produce asymmetric block artifacts).
// Two rotation angles for maximum de-correlation between octaves.
// cos(0.5)=0.8776 sin(0.5)=0.4794  |  cos(1.0)=0.5403 sin(1.0)=0.8415

float pd_fbm(vec3 p) {
    float rx1 = p.x*0.8776 - p.z*0.4794;
    float rz1 = p.x*0.4794 + p.z*0.8776;
    float rx2 = p.x*0.5403 - p.z*0.8415;
    float rz2 = p.x*0.8415 + p.z*0.5403;
    return pd_vnoise3(p)                                                * 0.5000
         + pd_vnoise3(vec3(rx1*2.0+1.7,  p.y*2.0+9.2,  rz1*2.0+3.5)) * 0.2500
         + pd_vnoise3(vec3(rx2*4.0+8.3,  p.y*4.0+2.8,  rz2*4.0+5.1)) * 0.1250
         + pd_vnoise3(vec3(rx1*8.0+4.1,  p.y*8.0+6.7,  rz2*8.0+2.3)) * 0.0625;
}

// ─── Tile-Break ───────────────────────────────────────────────────────────────
// Perturbs UV coordinates using world-position-seeded noise to break the
// visible grid repetition of tiling textures.
// strength: 0.0–0.35 recommended (0 = disabled, 0.15 = subtle, 0.3 = strong)

vec2 pd_tileBreak(vec2 uv, vec3 worldPos, float strength) {
    float n0 = pd_vnoise2(uv * 4.17 + worldPos.xz * 0.13);
    float n1 = pd_vnoise2(uv * 4.17 + worldPos.yz * 0.13 + vec2(3.3, 7.1));
    return uv + (vec2(n0, n1) - 0.5) * strength;
}

// ─── Procedural Dirt ──────────────────────────────────────────────────────────
// Simulates surface contamination that settles in geometric low-frequency
// hollows.  Multiply albedo with mix(1, dirtColor, pd_dirt(...) * strength).
// scale: world-space frequency (1–5 typical — larger = finer detail patches)
// Returns [0, 1] where 1 = maximum dirt accumulation.

float pd_dirt(vec3 worldPos, float scale) {
    float n = pd_fbm(worldPos * scale);
    // Dirt gathers in "quiet" fBm valleys — smoothstep pulls out the darks
    return smoothstep(0.62, 0.28, n);
}

// ─── Roughness Variation ──────────────────────────────────────────────────────
// Breaks uniform-roughness surfaces with micro-scale variation that simulates
// uneven wear, microscratches, and material inconsistency.
// Returns a signed delta in [-0.5, +0.5]. Add to base roughness and clamp.

float pd_roughnessVar(vec3 worldPos, float scale) {
    return (pd_vnoise3(worldPos * scale * 2.5 + vec3(5.5, 3.1, 8.9)) - 0.5);
}
