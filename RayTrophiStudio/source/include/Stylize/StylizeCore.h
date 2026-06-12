#pragma once
//
// StylizeCore.h — single-source stylize math, compiled BOTH as host C++ (CPU
// stylize path) and as CUDA __device__ code (GPU stylize kernel). There is
// exactly ONE implementation here, so the CPU and GPU outputs cannot drift
// ("bire bir" guarantee). StylizePostProcess.cpp converts Vec3 -> SV3 and
// calls into this header; the .cu kernel builds SV3 AOVs from device buffers
// and calls the same functions.
//
// Self-contained on purpose: it pulls in NO project headers (Vec3.h drags
// <random>/<iostream>; StylizeModeState.h drags <vector>/<string>). It uses a
// tiny POD SV3 vector and POD profile/AOV mirrors so it stays clean inside an
// nvcc translation unit.
//
// IMPORTANT — keep this byte-for-byte equivalent to the original CPU math:
//   * lerp3(a,b,t) reproduces Vec3::lerp  -> a + t*(b-a)
//   * lerpf(a,b,t) reproduces Vec3::lerpf -> a*(1-t)+b*t
//   * normalize3 reproduces Vec3::normalize -> zeroes vectors with len_sq <= 1e-6f
// Use only the float math intrinsics (floorf/powf/atan2f/...) — they exist on
// both host (MSVC <cmath>) and device (CUDA builtins) and match the float
// overloads std::* resolved to on the CPU.
//

#if defined(__CUDACC__)
    #define STYLIZE_HD __host__ __device__
#else
    #define STYLIZE_HD
    #include <cmath>
#endif

namespace StylizeCore {

// ---------------------------------------------------------------------------
// Enum mirrors (values MUST match Stylize::* in StylizeModeState.h).
// Stored as plain ints in the POD profile so the struct is device-copyable.
// ---------------------------------------------------------------------------
enum StrokeDir { SD_SurfaceNormal = 0, SD_Vertical, SD_Horizontal, SD_Diagonal, SD_CrossHatch };
enum LineType  { LT_Ink = 0, LT_OilPaint, LT_Pencil, LT_DryBrush, LT_Pressure };
enum ColorMode { CM_PaletteShadow = 0, CM_CustomColor, CM_MaterialTint, CM_WarmPaint, CM_CoolPencil };
enum SkyStyle  { SK_PainterlyClouds = 0, SK_CartoonCel, SK_SunsetBands, SK_InkWash, SK_ClearGradient };

// ---------------------------------------------------------------------------
// Tiny vector + math helpers (host/device).
// ---------------------------------------------------------------------------
struct SV3 { float x, y, z; };

STYLIZE_HD inline SV3 mk(float x, float y, float z) { SV3 v; v.x = x; v.y = y; v.z = z; return v; }
STYLIZE_HD inline SV3 mk(float s) { return mk(s, s, s); }

STYLIZE_HD inline SV3 operator+(const SV3& a, const SV3& b) { return mk(a.x + b.x, a.y + b.y, a.z + b.z); }
STYLIZE_HD inline SV3 operator-(const SV3& a, const SV3& b) { return mk(a.x - b.x, a.y - b.y, a.z - b.z); }
STYLIZE_HD inline SV3 operator*(const SV3& a, const SV3& b) { return mk(a.x * b.x, a.y * b.y, a.z * b.z); }
STYLIZE_HD inline SV3 operator*(const SV3& a, float t)      { return mk(a.x * t, a.y * t, a.z * t); }
STYLIZE_HD inline SV3 operator*(float t, const SV3& a)      { return a * t; }
STYLIZE_HD inline SV3 operator+(const SV3& a, float s)      { return mk(a.x + s, a.y + s, a.z + s); }

STYLIZE_HD inline float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
STYLIZE_HD inline float saturate(float v) { return clampf(v, 0.0f, 1.0f); }
STYLIZE_HD inline SV3 clamp01(const SV3& v) { return mk(saturate(v.x), saturate(v.y), saturate(v.z)); }
STYLIZE_HD inline float luminance(const SV3& c) { return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f; }

STYLIZE_HD inline float dot3(const SV3& a, const SV3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
STYLIZE_HD inline SV3 cross3(const SV3& a, const SV3& b) {
    return mk(a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x);
}
STYLIZE_HD inline float lensq3(const SV3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
STYLIZE_HD inline float length3(const SV3& v) { return sqrtf(lensq3(v)); }
STYLIZE_HD inline SV3 normalize3(const SV3& v) {            // reproduces Vec3::normalize (1e-6f gate)
    float len_sq = lensq3(v);
    if (len_sq > 1e-6f) {
        float inv_len = 1.0f / sqrtf(len_sq);
        return mk(v.x * inv_len, v.y * inv_len, v.z * inv_len);
    }
    return mk(0.0f, 0.0f, 0.0f);
}

STYLIZE_HD inline SV3 lerp3(const SV3& a, const SV3& b, float t) {   // reproduces Vec3::lerp
    return mk(a.x + t * (b.x - a.x),
              a.y + t * (b.y - a.y),
              a.z + t * (b.z - a.z));
}
STYLIZE_HD inline float lerpf(float a, float b, float t) { return a * (1.0f - t) + b * t; }  // Vec3::lerpf

// ---------------------------------------------------------------------------
// POD mirrors of the stylize state (no Vec3 / std containers).
// ---------------------------------------------------------------------------
struct SkyCore {
    int   enabled;
    int   style;            // SkyStyle
    SV3   horizon_color;
    SV3   zenith_color;
    SV3   sun_glow_color;
    float gradient_strength;
    float cloud_brush_scale;
    float cloud_brush_strength;
    float wind_smear;
    float horizon_haze;
    float sun_disc_scale;
    float cloud_roundness;
};

struct MaterialCore {
    int   enabled;
    int   stroke_direction; // StrokeDir
    float brush_strength;
    float brush_scale;
    float pigment_thickness;
    float dry_brush;
    int   wet_oil_model;
    float oil_body;
    float paint_load;
    float pickup_rate;
    float deposit_rate;
    float bristle_buildup;
    float surface_adherence;
    float depth_scale_response;
    float edge_respect;
    float palette_influence;
    float material_color_preservation;
    float color_simplification;
    float roughness_bias;
    float normal_softening;
};

struct OutlineCore {
    int   enabled;
    int   line_type;        // LineType
    int   color_mode;       // ColorMode
    SV3   custom_color;
    float strength;
    float width;
    float depth_sensitivity;
    float normal_sensitivity;
    float taper;
    float break_up;
    float color_bleed;
    float distance_thinning;
    float detail_protection;
};

struct StyleProfileCore {
    SV3   palette_shadow;
    SV3   palette_mid;
    SV3   palette_highlight;
    float global_strength;
    float temporal_coherence;
    SkyCore      sky;
    MaterialCore material;
    OutlineCore  outline;
};

struct StylizeAOVCore {
    SV3   albedo;
    SV3   normal;
    SV3   world_position;
    SV3   view_dir;
    SV3   sun_dir;
    float screen_u;
    float screen_v;
    float sun_size_degrees;
    float sun_elevation_degrees;
    int   nishita_clouds_enabled;
    float nishita_cloud_coverage;
    float nishita_cloud_density;
    float nishita_cloud_scale;
    float nishita_cloud_offset_x;
    float nishita_cloud_offset_z;
    int   nishita_cloud_seed;
    float depth;
    float edge;
    float pixel_scale;   // world units per pixel at the hit point (0 = unknown)
    unsigned int material_id;
    int   valid;
    int   hit;
};

// Defaults matching Stylize::StylizeAOVSample (used by the no-AOV overload).
STYLIZE_HD inline StylizeAOVCore defaultAOV() {
    StylizeAOVCore a;
    a.albedo = mk(0.0f); a.normal = mk(0.0f); a.world_position = mk(0.0f);
    a.view_dir = mk(0.0f, 0.0f, -1.0f); a.sun_dir = mk(0.32f, 0.82f, 0.46f);
    a.screen_u = 0.0f; a.screen_v = 0.0f;
    a.sun_size_degrees = 0.545f; a.sun_elevation_degrees = 35.0f;
    a.nishita_clouds_enabled = 0; a.nishita_cloud_coverage = 0.45f;
    a.nishita_cloud_density = 0.65f; a.nishita_cloud_scale = 1.0f;
    a.nishita_cloud_offset_x = 0.0f; a.nishita_cloud_offset_z = 0.0f;
    a.nishita_cloud_seed = 0; a.depth = 0.0f; a.edge = 0.0f; a.pixel_scale = 0.0f;
    a.material_id = 0xFFFFFFFFu; a.valid = 0; a.hit = 0;
    return a;
}

// ---------------------------------------------------------------------------
// Noise (integer-hash based -> bit-identical host/device).
// ---------------------------------------------------------------------------
STYLIZE_HD inline float hashNoise(int x, int y, int seed) {
    unsigned int n = static_cast<unsigned int>(x) * 1973u
                   ^ static_cast<unsigned int>(y) * 9277u
                   ^ static_cast<unsigned int>(seed) * 26699u
                   ^ 0x68bc21ebu;
    n = (n ^ (n >> 15u)) * 2246822519u;
    n = (n ^ (n >> 13u)) * 3266489917u;
    n = n ^ (n >> 16u);
    return static_cast<float>(n & 0x00ffffffu) / static_cast<float>(0x00ffffffu);
}

STYLIZE_HD inline float smoothstepf(float edge0, float edge1, float x) {
    float t = saturate((x - edge0) / fmaxf(1e-5f, edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

STYLIZE_HD inline float valueNoise(float x, float y, int seed) {
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float sx = tx * tx * (3.0f - 2.0f * tx);
    const float sy = ty * ty * (3.0f - 2.0f * ty);

    const float n00 = hashNoise(x0, y0, seed);
    const float n10 = hashNoise(x0 + 1, y0, seed);
    const float n01 = hashNoise(x0, y0 + 1, seed);
    const float n11 = hashNoise(x0 + 1, y0 + 1, seed);
    const float nx0 = n00 + (n10 - n00) * sx;
    const float nx1 = n01 + (n11 - n01) * sx;
    return nx0 + (nx1 - nx0) * sy;
}

STYLIZE_HD inline float fbmNoise(float x, float y, int seed) {
    float amp = 0.55f;
    float freq = 1.0f;
    float sum = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += valueNoise(x * freq, y * freq, seed + i * 17) * amp;
        norm += amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }
    return norm > 0.0f ? sum / norm : 0.0f;
}

STYLIZE_HD inline float fract01(float v) { return v - floorf(v); }

// ---------------------------------------------------------------------------
// Stylize building blocks.
// ---------------------------------------------------------------------------
STYLIZE_HD inline SV3 paletteRamp(float luma, const StyleProfileCore& profile) {
    luma = saturate(luma);
    if (luma < 0.5f) {
        return lerp3(profile.palette_shadow, profile.palette_mid, luma * 2.0f);
    }
    return lerp3(profile.palette_mid, profile.palette_highlight, (luma - 0.5f) * 2.0f);
}

STYLIZE_HD inline SV3 preserveMaterialHue(const SV3& style_color, const SV3& material_color, float amount) {
    amount = saturate(amount);
    const float mat_luma = fmaxf(0.035f, luminance(material_color));
    const float style_luma = fmaxf(0.035f, luminance(style_color));
    SV3 hue = mk(
        clampf(material_color.x / mat_luma, 0.20f, 2.80f),
        clampf(material_color.y / mat_luma, 0.20f, 2.80f),
        clampf(material_color.z / mat_luma, 0.20f, 2.80f)
    );
    SV3 hue_preserved = clamp01(hue * style_luma);
    return lerp3(style_color, hue_preserved, amount);
}

STYLIZE_HD inline SV3 dominantPigmentTint(const SV3& color, float amount) {
    amount = saturate(amount);
    const float mx = fmaxf(color.x, fmaxf(color.y, color.z));
    const float mn = fminf(color.x, fminf(color.y, color.z));
    const float chroma = saturate((mx - mn) / fmaxf(0.08f, mx));
    if (amount <= 0.001f || chroma <= 0.001f) {
        return color;
    }

    SV3 target = color;
    const float lift = 0.025f * amount;
    const float dom_boost = 1.0f + 0.16f * amount;
    const float sub_pull = 1.0f - 0.18f * amount;
    if (color.x >= color.y && color.x >= color.z) {
        target.x = fminf(1.0f, color.x * dom_boost + lift);
        target.y = color.y * sub_pull;
        target.z = color.z * sub_pull;
    } else if (color.y >= color.x && color.y >= color.z) {
        target.x = color.x * sub_pull;
        target.y = fminf(1.0f, color.y * dom_boost + lift);
        target.z = color.z * sub_pull;
    } else {
        target.x = color.x * sub_pull;
        target.y = color.y * sub_pull;
        target.z = fminf(1.0f, color.z * dom_boost + lift);
    }

    const float src_luma = fmaxf(0.025f, luminance(color));
    const float dst_luma = fmaxf(0.025f, luminance(target));
    target = clamp01(target * clampf(src_luma / dst_luma, 0.72f, 1.28f));
    return lerp3(color, target, amount * (0.35f + chroma * 0.65f));
}

STYLIZE_HD inline SV3 simplifyColor(const SV3& color, float amount) {
    amount = saturate(amount);
    const float levels = fmaxf(2.0f, 16.0f - amount * 12.0f);
    SV3 stepped = mk(
        floorf(saturate(color.x) * levels + 0.5f) / levels,
        floorf(saturate(color.y) * levels + 0.5f) / levels,
        floorf(saturate(color.z) * levels + 0.5f) / levels
    );
    return lerp3(color, stepped, amount);
}

STYLIZE_HD inline float surfacePixelScale(const StylizeAOVCore& aov, const StyleProfileCore& profile, float base_scale) {
    if (!aov.hit) {
        return base_scale;
    }
    const float depth = fmaxf(0.0f, aov.depth);
    const float depth_response = saturate(profile.material.depth_scale_response);
    const float surface_lock = saturate(profile.material.surface_adherence);
    const float depth_factor = clampf(1.0f + log1pf(depth) * 0.22f * depth_response, 0.75f, 2.35f);
    return base_scale * (1.0f + (depth_factor - 1.0f) * surface_lock);
}

STYLIZE_HD inline void surfaceLockedPixel(float& fx, float& fy, const StylizeAOVCore& aov, const StyleProfileCore& profile, float pixel_scale) {
    if (!aov.hit) {
        return;
    }
    const float surface_lock = saturate(profile.material.surface_adherence);
    if (surface_lock <= 0.001f) {
        return;
    }

    const int mat_seed = static_cast<int>(aov.material_id & 0xFFFFu);
    const float mat_offset_x = (hashNoise(mat_seed, 19, 0) - 0.5f) * pixel_scale * 2.2f;
    const float mat_offset_y = (hashNoise(mat_seed, 37, 1) - 0.5f) * pixel_scale * 2.2f;
    const float depth_offset = log1pf(fmaxf(0.0f, aov.depth)) * pixel_scale * 0.18f;

    fx += (mat_offset_x + aov.normal.x * depth_offset) * surface_lock;
    fy += (mat_offset_y - aov.normal.y * depth_offset + aov.normal.z * depth_offset * 0.35f) * surface_lock;
}

STYLIZE_HD inline void surfaceBasis(const SV3& normal, SV3& tangent, SV3& bitangent) {
    SV3 n = length3(normal) > 1e-5f ? normalize3(normal) : mk(0.0f, 1.0f, 0.0f);
    SV3 up = fabsf(n.y) < 0.82f ? mk(0.0f, 1.0f, 0.0f) : mk(1.0f, 0.0f, 0.0f);
    tangent = normalize3(cross3(up, n));
    if (length3(tangent) <= 1e-5f) {
        tangent = mk(1.0f, 0.0f, 0.0f);
    }
    bitangent = normalize3(cross3(n, tangent));
}

STYLIZE_HD inline bool surfaceStrokeCoordinates(float screen_u,
                                                float screen_v,
                                                const StylizeAOVCore& aov,
                                                const StyleProfileCore& profile,
                                                float v_scale,
                                                float& u,
                                                float& v) {
    u = screen_u;
    v = screen_v;
    if (!aov.hit) {
        return false;
    }

    const float surface_lock = saturate(profile.material.surface_adherence);
    if (surface_lock <= 0.001f) {
        return false;
    }

    SV3 tangent;
    SV3 bitangent;
    surfaceBasis(aov.normal, tangent, bitangent);

    const float brush_scale = fmaxf(0.05f, profile.material.brush_scale);
    const float world_scale = fmaxf(0.015f, 0.10f + brush_scale * 0.28f);
    const int mat_seed = static_cast<int>(aov.material_id & 0xFFFFu);
    const float mat_u = (hashNoise(mat_seed, 71, 2) - 0.5f) * 4.0f;
    const float mat_v = (hashNoise(mat_seed, 97, 3) - 0.5f) * 4.0f;

    (void)screen_u;
    (void)screen_v;
    const float world_u = dot3(aov.world_position, tangent) / world_scale + mat_u;
    const float world_v = dot3(aov.world_position, bitangent) / (world_scale * v_scale) + mat_v;
    u = world_u;
    v = world_v;
    return true;
}

STYLIZE_HD inline float strokeField(int x, int y, const StylizeAOVCore& aov, const StyleProfileCore& profile, int frame_index) {
    const float brush_scale = fmaxf(0.05f, profile.material.brush_scale);
    const float pixel_scale = surfacePixelScale(aov, profile, 14.0f + brush_scale * 28.0f);
    const float coherent_frame = static_cast<float>(frame_index) * (1.0f - profile.temporal_coherence);

    float nx = aov.hit ? aov.normal.x : 0.0f;
    float ny = aov.hit ? aov.normal.y : 1.0f;
    float nz = aov.hit ? aov.normal.z : 0.0f;
    float angle = atan2f(nz + ny * 0.35f, nx + 0.001f);
    switch (profile.material.stroke_direction) {
        case SD_Vertical: angle = 1.5707963f; break;
        case SD_Horizontal: angle = 0.0f; break;
        case SD_Diagonal: angle = 0.7853982f; break;
        case SD_CrossHatch:
            angle = ((x / 48 + y / 48) & 1) ? 0.7853982f : -0.7853982f;
            break;
        case SD_SurfaceNormal:
        default:
            break;
    }
    float ca = cosf(angle);
    float sa = sinf(angle);
    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    surfaceLockedPixel(fx, fy, aov, profile, pixel_scale);

    float screen_u = (fx * ca + fy * sa) / pixel_scale;
    float screen_v = (-fx * sa + fy * ca) / (pixel_scale * 0.32f);
    if (aov.hit) {
        screen_u += aov.depth * 0.018f + nx * 1.7f;
        screen_v += ny * 1.1f + nz * 0.7f;
    }
    float u = screen_u;
    float v = screen_v;
    surfaceStrokeCoordinates(screen_u, screen_v, aov, profile, 0.32f, u, v);

    const int seed = static_cast<int>(coherent_frame) + static_cast<int>(aov.material_id & 0xFFu) * 23;
    const float long_stroke = fbmNoise(u, v, seed);
    const float fiber = fbmNoise(u * 2.8f + 19.0f, v * 0.55f - 7.0f, seed + 91);
    float stroke = (long_stroke - 0.5f) * 0.75f + (fiber - 0.5f) * 0.25f;
    return stroke;
}

// ---------------------------------------------------------------------------
// Brush daub field — discrete, overlapping, oriented paint stamps. Each pixel
// finds the topmost elliptical daub on a jittered grid in stroke space; the
// winner paints a flat, tonally-offset coat with bristle streaks and a signed
// impasto rim. Per-daub random priority gives painter ordering, so overlapping
// daubs keep crisp boundaries instead of averaging into mush.
// ---------------------------------------------------------------------------
struct BrushStrokeSample {
    float coverage;   // 0..1 paint deposited by the winning daub
    float tone;       // -1..1 per-daub tonal offset
    float bristle;    // -1..1 bristle streaks along the daub axis
    float rim;        // -1..1 signed impasto rim (+ lit top edge, - shadow edge)
    float gap;        // 0..1 bare canvas between daubs
};

STYLIZE_HD inline void evalBrushDaubLayer(float u, float v, int seed,
                                          float aspect, float breakup, float edge_rag,
                                          BrushStrokeSample& s) {
    s.coverage = 0.0f; s.tone = 0.0f; s.bristle = 0.0f; s.rim = 0.0f; s.gap = 1.0f;
    const float spacing = aspect * 0.74f;
    const int row0 = static_cast<int>(floorf(v));
    float best_priority = -1.0f;
    float best_cov = 0.0f;
    float best_tone = 0.0f;
    float best_rim = 0.0f;
    float best_bris = 0.0f;
    float total_cov = 0.0f;
    for (int rj = -1; rj <= 1; ++rj) {
        const int row = row0 + rj;
        const float row_shift = hashNoise(row, seed, 11) * spacing;
        const int slot0 = static_cast<int>(floorf((u + row_shift) / spacing));
        for (int sj = -1; sj <= 1; ++sj) {
            const int slot = slot0 + sj;
            const int cell = seed + row * 131 + slot * 977;
            if (hashNoise(slot, row, cell + 1) < breakup * 0.55f) {
                continue;                                   // dry brush: missing daub
            }
            const float h_cu = hashNoise(slot, row, cell + 2);
            const float h_cv = hashNoise(slot, row, cell + 3);
            const float cu = (static_cast<float>(slot) + 0.5f + (h_cu - 0.5f) * 0.55f) * spacing - row_shift;
            const float cv = static_cast<float>(row) + 0.5f + (h_cv - 0.5f) * 0.40f;
            const float du = u - cu;
            const float dv = v - cv;
            if (fabsf(du) > aspect * 0.80f || fabsf(dv) > 1.45f) {
                continue;                                   // cheap reject before trig/noise
            }
            const float ja = (hashNoise(slot, row, cell + 4) - 0.5f) * 0.50f;
            const float cj = cosf(ja);
            const float sn = sinf(ja);
            const float ru = du * cj + dv * sn;
            const float rv = -du * sn + dv * cj;
            const float half_len = aspect * (0.40f + hashNoise(slot, row, cell + 5) * 0.24f);
            const float half_wid = 0.36f + hashNoise(slot, row, cell + 6) * 0.22f;
            const float t = ru / fmaxf(1e-4f, half_len);    // -1..1 along the stroke
            const float at = fabsf(t);
            if (at > 1.15f) {
                continue;
            }
            // wavy spine — real strokes curve a little
            const float wob = (valueNoise(ru * 1.6f + h_cu * 9.0f, h_cv * 7.0f, cell + 12) - 0.5f) * half_wid * 0.55f;
            const float rvw = rv - wob;
            // rounded head, thinning tail
            float w_local = half_wid * (0.30f + 0.70f * sqrtf(fmaxf(0.0f, 1.0f - t * t)));
            w_local *= 1.0f - 0.35f * smoothstepf(0.10f, 1.0f, t);
            const float dn = fabsf(rvw) / fmaxf(1e-4f, w_local);
            const float lon = 1.0f - smoothstepf(0.82f, 1.06f, at);
            const float lat = 1.0f - smoothstepf(0.58f, 1.0f, dn);
            float cov = lat * lon;
            if (cov <= 0.02f) {
                continue;
            }
            // bristle pattern across the width — streaks AND torn ragged ends;
            // edge_rag tears the stroke wherever the AOV edge runs through it
            const float bris = valueNoise(rvw * 3.1f / fmaxf(0.05f, half_wid), t * 2.2f, cell + 13);
            const float rag_zone = saturate(smoothstepf(0.35f, 0.95f, at) + edge_rag);
            cov *= 1.0f - rag_zone * smoothstepf(0.30f, 0.78f, bris) * (0.55f + breakup * 0.35f);
            if (cov <= 0.02f) {
                continue;
            }
            total_cov = fmaxf(total_cov, cov);
            const float priority = hashNoise(slot, row, cell + 7) + cov * 0.35f;
            if (priority > best_priority) {
                best_priority = priority;
                best_cov = cov;
                best_tone = (hashNoise(slot, row, cell + 8) - 0.5f) * 2.0f;
                best_bris = (bris - 0.5f) * 2.0f;
                const float band = smoothstepf(0.55f, 0.95f, dn) * lat * lon;
                best_rim = band * clampf(-rvw / fmaxf(1e-4f, w_local) * 1.7f, -1.0f, 1.0f);
            }
        }
    }
    if (best_priority < 0.0f) {
        return;
    }
    s.coverage = best_cov;
    s.tone = best_tone;
    s.rim = best_rim;
    s.bristle = best_bris;
    s.gap = saturate(1.0f - total_cov);
}

STYLIZE_HD inline BrushStrokeSample brushStrokeField(int x, int y,
                                                     const StylizeAOVCore& aov,
                                                     const StyleProfileCore& profile,
                                                     int frame_index) {
    const float brush_scale = fmaxf(0.05f, profile.material.brush_scale);
    // daub width in pixels — SCREEN-space size, so strokes never blow up or
    // shrink with surface orientation/distance (depth growth stays knob-gated
    // through surfacePixelScale only).
    const float daub_px = surfacePixelScale(aov, profile, 6.0f + brush_scale * 10.0f);
    const float coherent_frame = static_cast<float>(frame_index) * (1.0f - profile.temporal_coherence);

    const float nx = aov.hit ? aov.normal.x : 0.0f;
    const float ny = aov.hit ? aov.normal.y : 1.0f;
    const float nz = aov.hit ? aov.normal.z : 0.0f;
    float angle = atan2f(nz + ny * 0.35f, nx + 0.001f);
    switch (profile.material.stroke_direction) {
        case SD_Vertical: angle = 1.5707963f; break;
        case SD_Horizontal: angle = 0.0f; break;
        case SD_Diagonal: angle = 0.7853982f; break;
        case SD_CrossHatch:
            angle = ((x / 48 + y / 48) & 1) ? 0.7853982f : -0.7853982f;
            break;
        case SD_SurfaceNormal:
        default:
            break;
    }
    const float ca = cosf(angle);
    const float sa = sinf(angle);
    const float fx = static_cast<float>(x);
    const float fy = static_cast<float>(y);
    // Stroke-space coordinates. With Surface Lock the daubs are parameterized
    // on the SURFACE (world tangent plane, rotated by the stroke direction) so
    // they stay glued to geometry under camera motion — no screen-door sliding.
    // The scale is depth-normalized and split across two log2-quantized levels
    // that crossfade, so the on-screen daub size stays ~constant (within ~2x)
    // without the pattern swimming or popping.
    float pu;
    float pv;
    float world_per_daub;
    const float surface_lock = aov.hit ? saturate(profile.material.surface_adherence) : 0.0f;
    if (surface_lock > 0.001f) {
        SV3 tangent;
        SV3 bitangent;
        surfaceBasis(aov.normal, tangent, bitangent);
        const float tu = dot3(aov.world_position, tangent);
        const float tv = dot3(aov.world_position, bitangent);
        pu = tu * ca + tv * sa;
        pv = -tu * sa + tv * ca;
        // exact world-per-pixel from the AOV when the builder filled it; the
        // 0.0007*depth term is only a legacy fov-guess fallback
        const float wpp = aov.pixel_scale > 1e-7f
            ? aov.pixel_scale
            : fmaxf(1e-5f, aov.depth) * 0.0007f;
        world_per_daub = wpp * daub_px;
    } else {
        pu = fx * ca + fy * sa;
        pv = -fx * sa + fy * ca;
        world_per_daub = daub_px;
    }
    const int mat_seed = static_cast<int>(aov.material_id & 0xFFFFu);
    const float mat_u = (hashNoise(mat_seed, 71, 2) - 0.5f) * 4.0f + nx * 1.7f;
    const float mat_v = (hashNoise(mat_seed, 97, 3) - 0.5f) * 4.0f + ny * 1.1f + nz * 0.7f;

    const float l = log2f(fmaxf(1e-6f, world_per_daub));
    const int lvl = static_cast<int>(floorf(l));
    const float lf = l - static_cast<float>(lvl);
    const float s0 = exp2f(static_cast<float>(lvl));
    const float s1 = s0 * 2.0f;

    const int seed = static_cast<int>(coherent_frame) * 13
                   + static_cast<int>(aov.material_id & 0xFFu) * 53 + 977;
    const float aspect = 3.4f;                              // daub length / width
    const float breakup = saturate(profile.material.dry_brush);
    // AOV edges tear the stroke tips instead of clipping the paint — silhouettes
    // end in ragged brush starts/stops, not a hard geometric line.
    const float edge_rag = aov.hit
        ? saturate(aov.edge) * (0.35f + saturate(profile.material.edge_respect) * 0.65f)
        : 0.0f;
    // Seeds depend on the LOD level, so when the camera dollies and the level
    // increments, the incoming fine layer is EXACTLY the previous coarse layer
    // (no reshuffle at the boundary).
    BrushStrokeSample lodA;
    evalBrushDaubLayer(pu / s0 + mat_u, pv / s0 + mat_v, seed + lvl * 101,
                       aspect, breakup, edge_rag, lodA);
    BrushStrokeSample lodB;
    evalBrushDaubLayer(pu / s1 + mat_u, pv / s1 + mat_v, seed + (lvl + 1) * 101,
                       aspect, breakup, edge_rag, lodB);

    BrushStrokeSample s;
    const float wA = lodA.coverage * (1.0f - lf);
    const float wB = lodB.coverage * lf;
    const float wsum = wA + wB;
    s.coverage = lerpf(lodA.coverage, lodB.coverage, lf);
    s.gap = lerpf(lodA.gap, lodB.gap, lf);
    if (wsum > 1e-4f) {
        s.tone = (lodA.tone * wA + lodB.tone * wB) / wsum;
        s.bristle = (lodA.bristle * wA + lodB.bristle * wB) / wsum;
        s.rim = (lodA.rim * wA + lodB.rim * wB) / wsum;
    } else {
        s.tone = 0.0f;
        s.bristle = 0.0f;
        s.rim = 0.0f;
    }
    return s;
}

struct WetOilStroke {
    float bristle;
    float ridge;
    float drag;
    float body;
};

STYLIZE_HD inline WetOilStroke wetOilStrokeModel(int x,
                                                 int y,
                                                 const StylizeAOVCore& aov,
                                                 const StyleProfileCore& profile,
                                                 int frame_index,
                                                 float stroke) {
    const float brush_scale = fmaxf(0.05f, profile.material.brush_scale);
    const float oil_body = saturate(profile.material.oil_body);
    const float buildup = saturate(profile.material.bristle_buildup);
    const float pixel_scale = surfacePixelScale(aov, profile, 7.0f + brush_scale * 18.0f);
    const float coherent_frame = static_cast<float>(frame_index) * (1.0f - profile.temporal_coherence);
    const int seed = static_cast<int>(coherent_frame) + 311 + static_cast<int>(aov.material_id & 0xFFu) * 29;

    float nx = aov.hit ? aov.normal.x : 0.0f;
    float ny = aov.hit ? aov.normal.y : 1.0f;
    float angle = atan2f(ny, nx + 0.001f);
    if (profile.material.stroke_direction == SD_Vertical) {
        angle = 1.5707963f;
    } else if (profile.material.stroke_direction == SD_Horizontal) {
        angle = 0.0f;
    } else if (profile.material.stroke_direction == SD_Diagonal) {
        angle = 0.7853982f;
    }

    const float ca = cosf(angle);
    const float sa = sinf(angle);
    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    surfaceLockedPixel(fx, fy, aov, profile, pixel_scale);
    const float screen_u = (fx * ca + fy * sa) / pixel_scale;
    const float screen_v = (-fx * sa + fy * ca) / (pixel_scale * 0.22f);
    float u = screen_u;
    float v = screen_v;
    surfaceStrokeCoordinates(screen_u, screen_v, aov, profile, 0.22f, u, v);

    const float bristle_noise = fbmNoise(u * 3.6f + oil_body * 2.0f, v * 0.42f - 11.0f, seed);
    const float body_noise = fbmNoise(u * 0.85f - 5.0f, v * 0.55f + 13.0f, seed + 73);
    const float bristle = (bristle_noise - 0.5f) * (0.45f + buildup * 0.55f);
    WetOilStroke oil;
    oil.bristle = bristle;
    oil.ridge = smoothstepf(0.42f, 0.96f, fabsf(stroke) + body_noise * 0.28f + buildup * 0.24f);
    oil.drag = smoothstepf(0.18f, 0.88f, body_noise * (0.55f + oil_body * 0.45f) + fabsf(bristle));
    oil.body = body_noise;
    return oil;
}

STYLIZE_HD inline SV3 applySkyLayer(const SV3& input_color,
                                    const StylizeAOVCore& aov,
                                    int x,
                                    int y,
                                    int frame_index,
                                    const StyleProfileCore& profile) {
    const SkyCore& sky = profile.sky;
    if (!sky.enabled) {
        return input_color;
    }

    SV3 dir = lensq3(aov.view_dir) > 1e-8f ? normalize3(aov.view_dir) : mk(0.0f, 0.0f, -1.0f);
    SV3 sun_dir = lensq3(aov.sun_dir) > 1e-8f ? normalize3(aov.sun_dir) : normalize3(mk(0.32f, 0.82f, 0.46f));
    const float azimuth = atan2f(dir.z, dir.x);
    const float sphere_u = azimuth / (2.0f * 3.14159265f) + 0.5f;
    const float sphere_v = saturate(dir.y * 0.5f + 0.5f);
    const float horizon = 1.0f - smoothstepf(-0.08f, 0.34f, dir.y);
    const float sky_t = smoothstepf(-0.16f, 0.88f, dir.y);
    const float coherent_frame = static_cast<float>(frame_index) * (1.0f - profile.temporal_coherence);
    const bool cartoon_sky = sky.style == SK_CartoonCel;
    const bool ink_sky = sky.style == SK_InkWash;
    const bool sunset_sky = sky.style == SK_SunsetBands;
    const bool clear_sky = sky.style == SK_ClearGradient;

    SV3 gradient = lerp3(sky.horizon_color, sky.zenith_color, sky_t);
    if (cartoon_sky) {
        const float band_noise = (fbmNoise(sphere_u * 3.0f, sky_t * 1.7f, 911) - 0.5f) * 0.045f;
        const float band = floorf(saturate(sky_t + band_noise) * 5.0f + 0.5f) / 5.0f;
        gradient = lerp3(sky.horizon_color, sky.zenith_color, band);
    } else if (sunset_sky) {
        const float wave = (fbmNoise(sphere_u * 2.4f, sky_t * 1.2f, 937) - 0.5f) * 0.075f;
        const float lower_band = smoothstepf(0.03f, 0.30f, sky_t + wave) * (1.0f - smoothstepf(0.36f, 0.58f, sky_t + wave));
        const float upper_band = smoothstepf(0.30f, 0.58f, sky_t - wave * 0.45f) * (1.0f - smoothstepf(0.68f, 0.88f, sky_t));
        const SV3 peach = lerp3(sky.horizon_color, sky.sun_glow_color, 0.72f);
        const SV3 violet = lerp3(sky.zenith_color, profile.palette_shadow, 0.20f);
        gradient = lerp3(gradient, peach, lower_band * 0.48f);
        gradient = lerp3(gradient, violet, upper_band * 0.22f);
    } else if (!ink_sky) {
        const float input_luma = fmaxf(0.025f, luminance(input_color));
        const float gradient_luma = fmaxf(0.025f, luminance(gradient));
        gradient = clamp01(gradient * clampf(input_luma / gradient_luma, 0.65f, 1.45f));
    }

    const float gradient_mix = cartoon_sky || sunset_sky || ink_sky
        ? saturate(0.72f + sky.gradient_strength * 0.28f)
        : saturate(sky.gradient_strength);
    SV3 color = lerp3(input_color, gradient, gradient_mix);

    const float haze_amount = saturate(sky.horizon_haze) * horizon;
    const SV3 haze_color = clamp01(lerp3(sky.horizon_color, sky.sun_glow_color, 0.18f));
    color = lerp3(color, haze_color, haze_amount * (sunset_sky ? 0.78f : 0.55f));

    const float brush_scale = fmaxf(0.10f, sky.cloud_brush_scale);
    const float nishita_scale = aov.nishita_clouds_enabled ? aov.nishita_cloud_scale : 1.0f;
    const float cloud_scale = fmaxf(0.08f, brush_scale * nishita_scale);
    const float cloud_coverage = aov.nishita_clouds_enabled ? aov.nishita_cloud_coverage : 0.42f;
    const float cloud_density = aov.nishita_clouds_enabled ? aov.nishita_cloud_density : 0.72f;
    const float wind = coherent_frame * 0.018f * saturate(sky.wind_smear);
    const float world_u = fract01(sphere_u + aov.nishita_cloud_offset_x * 0.0025f + wind);
    const float world_v = saturate(sphere_v + aov.nishita_cloud_offset_z * 0.0025f + wind * 0.10f);
    const int cloud_seed = 1207 + aov.nishita_cloud_seed * 37;
    const float cloud_band = smoothstepf(0.18f, 0.74f, world_v) * (1.0f - smoothstepf(0.88f, 1.0f, world_v));
    float cloud_shape = 0.0f;
    float cloud_detail = 0.0f;
    float cloud_shadow = 0.0f;
    float cloud_highlight = 0.0f;
    float cloud_rim = 0.0f;
    if (!clear_sky) {
        const float banks = sunset_sky ? 3.0f : (cartoon_sky ? 3.5f : 4.0f);
        const float bank_pos = world_v * banks;
        const int base_bank = static_cast<int>(floorf(bank_pos));
        const float roundness = clampf(sky.cloud_roundness, 0.05f, 1.0f);
        const float groups_per_bank = clampf(1.8f + cloud_scale * 0.42f + cloud_coverage * 2.4f, 2.0f, 6.5f);
        for (int bank_offset = -1; bank_offset <= 1; ++bank_offset) {
            const int bank = base_bank + bank_offset;
            const float bank_center = (static_cast<float>(bank) + 0.5f) / banks;
            const float bank_wave = (fbmNoise(world_u * 2.2f + static_cast<float>(bank) * 1.7f, 0.37f, cloud_seed + 41) - 0.5f) * 0.055f;
            const float bank_v = bank_center + bank_wave + horizon * 0.06f;
            const float bank_keep = 1.0f - smoothstepf(0.105f, 0.205f, fabsf(world_v - bank_v));
            if (bank_keep <= 0.001f) {
                continue;
            }

            const float group_pos = world_u * groups_per_bank + hashNoise(bank, cloud_seed, 9) * 0.65f;
            const int base_group = static_cast<int>(floorf(group_pos));
            for (int group_offset = -1; group_offset <= 1; ++group_offset) {
                const int group = base_group + group_offset;
                const float h0 = hashNoise(group, bank + cloud_seed, 0);
                const float h1 = hashNoise(group, bank + cloud_seed, 1);
                const float group_center_u = fract01((static_cast<float>(group) + 0.20f + h0 * 0.60f) / groups_per_bank);
                const float group_center_v = bank_v + (h1 - 0.5f) * 0.045f;
                const float group_width = 0.070f + cloud_coverage * 0.095f + (cartoon_sky ? 0.030f : 0.0f);
                for (int lobe = 0; lobe < 6; ++lobe) {
                    const float lh0 = hashNoise(group, cloud_seed + bank * 17 + lobe * 31, 5);
                    const float lh1 = hashNoise(group, cloud_seed + bank * 17 + lobe * 31, 6);
                    const float lh2 = hashNoise(group, cloud_seed + bank * 17 + lobe * 31, 7);
                    const float lobe_offset = (static_cast<float>(lobe) - 2.5f) * group_width * (0.34f + lh0 * 0.20f);
                    const float lobe_u = fract01(group_center_u + lobe_offset);
                    const float arch = 1.0f - fabsf(static_cast<float>(lobe) - 2.5f) / 2.5f;
                    const float lobe_v = group_center_v + arch * (0.035f + roundness * 0.035f) + (lh1 - 0.5f) * 0.026f;
                    float du = fabsf(world_u - lobe_u);
                    du = fminf(du, 1.0f - du);
                    const float dv = (world_v - lobe_v) * (2.55f - roundness * 0.80f);
                    const float rx = group_width * (0.58f + lh2 * 0.38f);
                    const float ry = rx * (0.40f + roundness * 0.24f);
                    const float d = sqrtf((du * du) / fmaxf(1e-5f, rx * rx) + (dv * dv) / fmaxf(1e-5f, ry * ry));
                    const float puff = (1.0f - smoothstepf(0.78f, 1.0f, d)) * bank_keep;
                    const float interior = (1.0f - smoothstepf(0.38f, 0.90f, d)) * bank_keep;
                    const float lower = smoothstepf(-ry * 0.20f, ry * 1.20f, -dv);
                    const float upper = smoothstepf(-ry * 1.15f, ry * 0.05f, dv);
                    const float rim = smoothstepf(0.72f, 0.98f, d) * puff;
                    cloud_shape = fmaxf(cloud_shape, puff);
                    cloud_detail += interior * (0.25f + lh2 * 0.18f);
                    cloud_shadow = fmaxf(cloud_shadow, puff * lower * (0.35f + lh0 * 0.22f));
                    cloud_highlight = fmaxf(cloud_highlight, interior * upper * (0.42f + lh1 * 0.32f));
                    cloud_rim = fmaxf(cloud_rim, rim);
                }
            }
        }
        const float coverage_gate = lerpf(0.72f, 0.12f, saturate(cloud_coverage));
        const float keep = smoothstepf(coverage_gate, 1.0f, cloud_shape + cloud_detail * 0.18f);
        cloud_shape *= keep;
        cloud_shadow *= keep;
        cloud_highlight *= keep;
        cloud_rim *= keep;
    }

    const float cloud_alpha = saturate(sky.cloud_brush_strength) * saturate(cloud_density) * cloud_band * saturate(cloud_shape);
    SV3 cloud_tint = cartoon_sky
        ? mk(1.0f, 0.96f, 0.78f)
        : clamp01(lerp3(mk(0.96f, 0.92f, 0.82f), profile.palette_highlight, ink_sky ? 0.08f : 0.28f));
    if (ink_sky) {
        cloud_tint = lerp3(mk(0.88f, 0.88f, 0.80f), profile.palette_shadow, 0.16f);
    }
    const SV3 shadow_tint = cartoon_sky
        ? mk(0.58f, 0.72f, 0.92f)
        : clamp01(lerp3(profile.palette_shadow, sky.zenith_color, ink_sky ? 0.42f : 0.28f));
    const SV3 highlight_tint = cartoon_sky
        ? mk(1.0f, 0.98f, 0.86f)
        : clamp01(lerp3(mk(1.0f, 0.96f, 0.84f), sky.sun_glow_color, sunset_sky ? 0.45f : 0.18f));
    color = lerp3(color, shadow_tint, cloud_alpha * cloud_shadow * (cartoon_sky ? 0.48f : 0.34f));
    color = lerp3(color, cloud_tint, cloud_alpha * (cartoon_sky ? 0.92f : 0.66f));
    color = lerp3(color, highlight_tint, cloud_alpha * cloud_highlight * (cartoon_sky ? 0.64f : 0.46f));
    if (cartoon_sky) {
        color = lerp3(color, mk(0.24f, 0.42f, 0.76f), cloud_alpha * cloud_rim * 0.16f);
    }
    if (!cartoon_sky) {
        const float fiber = fbmNoise(world_u * 18.0f, world_v * 5.0f, cloud_seed + 82) - 0.5f;
        color = color * (1.0f + fiber * cloud_alpha * (ink_sky ? 0.08f : 0.12f));
    }

    const float sun_dot = saturate(dot3(dir, sun_dir));
    const float sun_size = fmaxf(0.01f, aov.sun_size_degrees);
    const float low_sun_factor = aov.sun_elevation_degrees < 15.0f
        ? 1.0f + (15.0f - fmaxf(aov.sun_elevation_degrees, -10.0f)) * 0.04f
        : 1.0f;
    const float style_sun_scale = fmaxf(0.5f, sky.sun_disc_scale) * (cartoon_sky ? 1.35f : 1.0f);
    const float sun_radius = sun_size * low_sun_factor * style_sun_scale * (3.14159265f / 180.0f) * 0.5f;
    const float sun_threshold = cosf(sun_radius);
    const float sun_disk = cartoon_sky
        ? (sun_dot > sun_threshold ? 1.0f : 0.0f)
        : smoothstepf(sun_threshold, 1.0f, sun_dot);
    const float sun_glow = powf(sun_dot, sunset_sky ? 8.0f : 14.0f) * (0.24f + saturate(sky.gradient_strength) * 0.64f);
    color = lerp3(color, clamp01(color + sky.sun_glow_color * (sunset_sky ? 0.72f : 0.45f)), saturate(sun_glow) * (cartoon_sky ? 0.16f : 0.38f));
    const SV3 sun_color = cartoon_sky
        ? clamp01(sky.sun_glow_color * 1.55f + mk(0.08f, 0.04f, 0.0f))
        : clamp01(sky.sun_glow_color * 1.35f);
    color = lerp3(color, sun_color, sun_disk * (cartoon_sky ? 1.0f : 0.86f));

    if (!cartoon_sky && !clear_sky) {
        const float stroke = fbmNoise(sphere_u * 10.0f + wind * 4.0f, sphere_v * 3.0f, 1423) - 0.5f;
        color = color * (1.0f + stroke * saturate(sky.cloud_brush_strength) * (ink_sky ? 0.06f : 0.08f));
    }
    (void)x;
    (void)y;

    return clamp01(color);
}

STYLIZE_HD inline SV3 outlineColor(const StyleProfileCore& profile, const StylizeAOVCore& aov, const SV3& base_albedo) {
    const OutlineCore& outline = profile.outline;
    switch (outline.color_mode) {
        case CM_CustomColor:
            return clamp01(outline.custom_color);
        case CM_MaterialTint: {
            SV3 material_ink = base_albedo * (0.34f + outline.color_bleed * 0.22f);
            return clamp01(lerp3(profile.palette_shadow * 0.55f, material_ink, outline.color_bleed));
        }
        case CM_WarmPaint:
            return clamp01(lerp3(profile.palette_shadow * 0.55f, mk(0.34f, 0.18f, 0.08f), 0.35f + outline.color_bleed * 0.35f));
        case CM_CoolPencil:
            return clamp01(lerp3(profile.palette_shadow * 0.45f, mk(0.12f, 0.14f, 0.17f), 0.55f));
        case CM_PaletteShadow:
        default:
            return clamp01(profile.palette_shadow * 0.55f);
    }
}

STYLIZE_HD inline float outlineTexture(int x, int y, const StylizeAOVCore& aov, const StyleProfileCore& profile, int frame_index) {
    const OutlineCore& outline = profile.outline;
    const float edge = saturate(aov.edge);
    const float distance = aov.hit ? log1pf(fmaxf(0.0f, aov.depth)) : 0.0f;
    const float distance_factor = aov.hit
        ? 1.0f / (1.0f + distance * (0.20f + outline.distance_thinning * 0.85f))
        : 1.0f;
    const float adaptive_width = lerpf(outline.width, outline.width * distance_factor, outline.distance_thinning);
    const float width = fmaxf(0.1f, adaptive_width);
    const int mat_seed = static_cast<int>(aov.material_id & 0xFFu) * 41 + static_cast<int>(frame_index * (1.0f - profile.temporal_coherence));
    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    if (aov.hit) {
        const float push = distance * 3.0f;
        fx += aov.normal.x * push;
        fy += (aov.normal.y + aov.normal.z * 0.35f) * push;
    }

    const float grain = fbmNoise(fx / (12.0f + width * 8.0f), fy / (7.0f + width * 5.0f), mat_seed + 503);
    const float fiber = fbmNoise(fx / (4.0f + width * 3.0f) + 17.0f, fy / (18.0f + width * 9.0f) - 5.0f, mat_seed + 719);
    const float taper = smoothstepf(0.02f, 0.90f, edge) * (1.0f - outline.taper * smoothstepf(0.55f, 1.0f, edge));
    float coverage = saturate(edge * (0.75f + width * 0.45f));
    const float fine_detail = saturate((1.0f - distance_factor) * (1.0f - smoothstepf(0.45f, 1.0f, edge)));
    coverage *= 1.0f - fine_detail * outline.detail_protection * 0.82f;

    switch (outline.line_type) {
        case LT_OilPaint:
            coverage *= 0.80f + grain * 0.42f;
            coverage += smoothstepf(0.62f, 0.95f, fiber) * outline.color_bleed * 0.18f;
            break;
        case LT_Pencil:
            coverage *= 0.52f + fiber * 0.58f;
            coverage *= 0.85f + grain * 0.18f;
            break;
        case LT_DryBrush: {
            const float dry_cut = smoothstepf(0.18f, 0.86f, grain + fiber * 0.35f);
            coverage *= 0.42f + dry_cut * 0.78f;
            break;
        }
        case LT_Pressure:
            coverage *= 0.68f + smoothstepf(0.18f, 0.95f, edge + grain * 0.18f) * 0.55f;
            coverage *= 1.0f - outline.taper * 0.35f * (1.0f - edge);
            break;
        case LT_Ink:
        default:
            coverage *= 0.92f + grain * 0.16f;
            break;
    }

    const float breakup = saturate(outline.break_up);
    if (breakup > 0.001f) {
        const float keep = smoothstepf(breakup * 0.45f, 0.96f, grain + edge * 0.28f);
        coverage *= lerpf(1.0f, keep, breakup);
    }

    return saturate(coverage * (0.55f + taper * 0.45f));
}

// ---------------------------------------------------------------------------
// Main entry — mirrors Stylize::applyPostProcess(input, aov, x, y, frame, state).
// Caller guarantees state.enabled; this checks global_strength like the CPU code.
// ---------------------------------------------------------------------------
STYLIZE_HD inline SV3 applyPostProcess(const SV3& input_color,
                                       const StylizeAOVCore& aov,
                                       int x,
                                       int y,
                                       int frame_index,
                                       const StyleProfileCore& profile) {
    const float strength = saturate(profile.global_strength);
    if (strength <= 0.0001f) {
        return input_color;
    }

    SV3 color = clamp01(input_color);
    const SV3 base_albedo = aov.hit ? clamp01(aov.albedo) : color;
    const float albedo_luma = luminance(base_albedo);
    const bool material_domain = aov.hit != 0;

    if (aov.valid && !aov.hit) {
        color = applySkyLayer(color, aov, x, y, frame_index, profile);
    }

    const float palette_influence = saturate(profile.material.palette_influence);
    const float edge_guard = aov.hit
        ? (1.0f - saturate(aov.edge * (0.35f + profile.material.edge_respect * 0.65f)))
        : 1.0f;

    if (material_domain) {
        SV3 palette_color = paletteRamp(albedo_luma, profile);
        if (aov.hit) {
            palette_color = preserveMaterialHue(
                palette_color,
                base_albedo,
                profile.material.material_color_preservation);
        }
        const float palette_mix = palette_influence * strength * (0.16f + profile.material.color_simplification * 0.46f);
        color = lerp3(color, palette_color, palette_mix);

        const float material_guard = aov.hit
            ? (0.35f + 0.65f * (1.0f - saturate(profile.material.material_color_preservation)))
            : 1.0f;
        color = simplifyColor(color, strength * profile.material.color_simplification * 0.48f * edge_guard * material_guard);

        const bool brush_on = profile.material.enabled && profile.material.brush_strength > 0.001f;
        const bool outline_on = profile.outline.enabled && aov.edge > 0.001f;
        BrushStrokeSample daub;
        daub.coverage = 0.0f; daub.tone = 0.0f; daub.bristle = 0.0f; daub.rim = 0.0f; daub.gap = 1.0f;
        if (brush_on || outline_on) {
            daub = brushStrokeField(x, y, aov, profile, frame_index);
        }

        if (brush_on) {
            const float stroke = strokeField(x, y, aov, profile, frame_index);
            const float lit_luma = luminance(color);
            const float shadow_boost = 1.0f + (1.0f - smoothstepf(0.08f, 0.48f, lit_luma)) * 0.22f;
            const float highlight_thin = 1.0f - smoothstepf(0.62f, 0.96f, lit_luma) * 0.42f;
            // No edge_guard on the daub layer: edges tear the strokes inside the
            // field (edge_rag) instead of fading the paint into a clean cutout.
            const float brush_amount = profile.material.brush_strength * strength * shadow_boost * highlight_thin;
            const float guarded_amount = brush_amount * edge_guard;
            const float daub_opacity = saturate(brush_amount * 2.2f);
            const SV3 paint_color = dominantPigmentTint(color, 0.42f + palette_influence * 0.18f);
            SV3 palette_tint = lerp3(profile.palette_shadow, profile.palette_highlight, saturate(albedo_luma + stroke * 0.28f));
            if (aov.hit) {
                palette_tint = preserveMaterialHue(
                    palette_tint,
                    paint_color,
                    profile.material.material_color_preservation);
            }
            const SV3 stroke_tint = lerp3(paint_color, palette_tint, palette_influence * 0.18f);
            if (profile.material.wet_oil_model) {
                const WetOilStroke oil = wetOilStrokeModel(x, y, aov, profile, frame_index, stroke);
                const float oil_body = saturate(profile.material.oil_body);
                const float paint_load = saturate(profile.material.paint_load);
                const float pickup = saturate(profile.material.pickup_rate);
                const float deposit = saturate(profile.material.deposit_rate);
                const float wet_visibility = 0.55f + oil_body * 0.62f + paint_load * 0.20f;
                const float wet_mask = saturate(guarded_amount * wet_visibility * (0.08f + oil_body * 0.12f + oil.drag * 0.18f));
                const float deposit_mask = saturate(guarded_amount * paint_load * deposit * wet_visibility * (0.05f + oil.ridge * 0.10f));
                const float bristle_mask = saturate(guarded_amount * (0.10f + oil_body * 0.12f + profile.material.bristle_buildup * 0.10f));
                const SV3 picked_color = lerp3(color, paint_color, 0.28f + pickup * 0.22f);
                const SV3 carried_color = lerp3(picked_color, stroke_tint, saturate(deposit * paint_load * palette_influence * 0.22f));
                const float body_luma = (oil.body - 0.5f) * (0.045f + oil_body * 0.055f);
                const float bristle_luma = (stroke + oil.bristle) * (0.055f + oil_body * 0.065f);
                SV3 material_stroke = clamp01(paint_color * (1.0f + body_luma + bristle_luma));
                material_stroke = lerp3(material_stroke, carried_color, saturate(palette_influence * 0.18f + deposit * 0.08f));
                color = lerp3(color, material_stroke, wet_mask);
                color = lerp3(color, carried_color, deposit_mask);
                color = color * (1.0f + (stroke + oil.bristle) * bristle_mask * 0.42f);
                color = lerp3(color, color * (1.0f + 0.045f * oil.ridge), oil.ridge * guarded_amount * oil_body * 0.35f);
                color = lerp3(color, color * (0.985f - oil.ridge * 0.015f), profile.material.dry_brush * guarded_amount * (1.0f - oil.drag) * 0.32f);
                SV3 wet_daub = clamp01(paint_color * (1.0f + daub.tone * (0.16f + oil_body * 0.10f)
                                                          + (daub.bristle * 0.6f + oil.bristle) * 0.12f));
                wet_daub = lerp3(wet_daub, carried_color, saturate(palette_influence * 0.20f + deposit * 0.10f));
                color = lerp3(color, wet_daub, daub_opacity * daub.coverage * (0.30f + paint_load * 0.35f));
                color = clamp01(color * (1.0f + daub.rim * daub_opacity * (0.10f + oil_body * 0.14f)));
            } else {
                SV3 daub_color = clamp01(paint_color * (1.0f + daub.tone * 0.20f + daub.bristle * 0.12f + stroke * 0.06f));
                daub_color = lerp3(daub_color, stroke_tint, palette_influence * 0.22f);
                color = lerp3(color, daub_color, daub_opacity * daub.coverage * 0.85f);
                color = clamp01(color * (1.0f + daub.rim * daub_opacity
                                                    * (0.10f + saturate(profile.material.pigment_thickness) * 0.16f)));
                const float gap_amount = daub.gap * daub_opacity * profile.material.dry_brush * 0.35f;
                color = lerp3(color, lerp3(color, profile.palette_shadow, 0.45f), gap_amount);
                color = color * (1.0f + stroke * guarded_amount * 0.10f);
            }
        }

        if (profile.material.pigment_thickness > 0.001f) {
            const float pigment = profile.material.pigment_thickness * strength;
            color = mk(
                powf(saturate(color.x), 1.0f + pigment * 0.18f),
                powf(saturate(color.y), 1.0f + pigment * 0.14f),
                powf(saturate(color.z), 1.0f + pigment * 0.10f)
            );
            const float edge_pigment = aov.hit ? saturate(aov.edge * 0.8f) : 0.0f;
            color = lerp3(color, color + profile.palette_highlight * 0.035f * palette_influence, pigment * (1.0f - edge_pigment));
            color = lerp3(color, lerp3(base_albedo, profile.palette_shadow * 0.7f, palette_influence), pigment * edge_pigment * 0.24f);
        }

        if (outline_on) {
            float line = outlineTexture(x, y, aov, profile, frame_index);
            // contour drawn with the same brush: it breaks at daub gaps and
            // tapers at stroke tips instead of tracing a solid geometric line
            line *= 0.40f + 0.60f * smoothstepf(0.10f, 0.70f, daub.coverage + fabsf(daub.bristle) * 0.25f);
            const float edge = saturate(line * profile.outline.strength * strength);
            const SV3 ink = outlineColor(profile, aov, base_albedo);
            color = lerp3(color, ink, edge);
        }
    }

    return clamp01(lerp3(input_color, color, strength));
}

} // namespace StylizeCore
