#include "Stylize/StylizePostProcess.h"

#include <algorithm>
#include <cmath>

namespace Stylize {
namespace {

float saturate(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

Vec3 clamp01(const Vec3& v) {
    return Vec3(saturate(v.x), saturate(v.y), saturate(v.z));
}

float luminance(const Vec3& c) {
    return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;
}

float hashNoise(int x, int y, int seed) {
    unsigned int n = static_cast<unsigned int>(x) * 1973u
                   ^ static_cast<unsigned int>(y) * 9277u
                   ^ static_cast<unsigned int>(seed) * 26699u
                   ^ 0x68bc21ebu;
    n = (n ^ (n >> 15u)) * 2246822519u;
    n = (n ^ (n >> 13u)) * 3266489917u;
    n = n ^ (n >> 16u);
    return static_cast<float>(n & 0x00ffffffu) / static_cast<float>(0x00ffffffu);
}

float smoothstep(float edge0, float edge1, float x) {
    float t = saturate((x - edge0) / std::max(1e-5f, edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

float valueNoise(float x, float y, int seed) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
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

float fbmNoise(float x, float y, int seed) {
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

float fract01(float v) {
    return v - std::floor(v);
}

Vec3 paletteRamp(float luma, const StyleProfile& profile) {
    luma = saturate(luma);
    if (luma < 0.5f) {
        return Vec3::lerp(profile.palette_shadow, profile.palette_mid, luma * 2.0f);
    }
    return Vec3::lerp(profile.palette_mid, profile.palette_highlight, (luma - 0.5f) * 2.0f);
}

Vec3 preserveMaterialHue(const Vec3& style_color, const Vec3& material_color, float amount) {
    amount = saturate(amount);
    const float mat_luma = std::max(0.035f, luminance(material_color));
    const float style_luma = std::max(0.035f, luminance(style_color));
    Vec3 hue(
        std::clamp(material_color.x / mat_luma, 0.20f, 2.80f),
        std::clamp(material_color.y / mat_luma, 0.20f, 2.80f),
        std::clamp(material_color.z / mat_luma, 0.20f, 2.80f)
    );
    Vec3 hue_preserved = clamp01(hue * style_luma);
    return Vec3::lerp(style_color, hue_preserved, amount);
}

Vec3 simplifyColor(const Vec3& color, float amount) {
    amount = saturate(amount);
    const float levels = std::max(2.0f, 16.0f - amount * 12.0f);
    Vec3 stepped(
        std::floor(saturate(color.x) * levels + 0.5f) / levels,
        std::floor(saturate(color.y) * levels + 0.5f) / levels,
        std::floor(saturate(color.z) * levels + 0.5f) / levels
    );
    return Vec3::lerp(color, stepped, amount);
}

float surfacePixelScale(const StylizeAOVSample& aov, const StyleProfile& profile, float base_scale) {
    if (!aov.hit) {
        return base_scale;
    }
    const float depth = std::max(0.0f, aov.depth);
    const float depth_response = saturate(profile.material.depth_scale_response);
    const float surface_lock = saturate(profile.material.surface_adherence);
    const float depth_factor = std::clamp(1.0f + std::log1p(depth) * 0.22f * depth_response, 0.75f, 2.35f);
    return base_scale * (1.0f + (depth_factor - 1.0f) * surface_lock);
}

void surfaceLockedPixel(float& fx, float& fy, const StylizeAOVSample& aov, const StyleProfile& profile, float pixel_scale) {
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
    const float depth_offset = std::log1p(std::max(0.0f, aov.depth)) * pixel_scale * 0.18f;

    fx += (mat_offset_x + aov.normal.x * depth_offset) * surface_lock;
    fy += (mat_offset_y - aov.normal.y * depth_offset + aov.normal.z * depth_offset * 0.35f) * surface_lock;
}

void surfaceBasis(const Vec3& normal, Vec3& tangent, Vec3& bitangent) {
    Vec3 n = normal.length() > 1e-5f ? normal.normalize() : Vec3(0.0f, 1.0f, 0.0f);
    Vec3 up = std::abs(n.y) < 0.82f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    tangent = Vec3::cross(up, n).normalize();
    if (tangent.length() <= 1e-5f) {
        tangent = Vec3(1.0f, 0.0f, 0.0f);
    }
    bitangent = Vec3::cross(n, tangent).normalize();
}

bool surfaceStrokeCoordinates(float screen_u,
                              float screen_v,
                              const StylizeAOVSample& aov,
                              const StyleProfile& profile,
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

    Vec3 tangent;
    Vec3 bitangent;
    surfaceBasis(aov.normal, tangent, bitangent);

    const float brush_scale = std::max(0.05f, profile.material.brush_scale);
    const float world_scale = std::max(0.015f, 0.10f + brush_scale * 0.28f);
    const int mat_seed = static_cast<int>(aov.material_id & 0xFFFFu);
    const float mat_u = (hashNoise(mat_seed, 71, 2) - 0.5f) * 4.0f;
    const float mat_v = (hashNoise(mat_seed, 97, 3) - 0.5f) * 4.0f;

    const float world_u = Vec3::dot(aov.world_position, tangent) / world_scale + mat_u;
    const float world_v = Vec3::dot(aov.world_position, bitangent) / (world_scale * v_scale) + mat_v;
    u = screen_u * (1.0f - surface_lock) + world_u * surface_lock;
    v = screen_v * (1.0f - surface_lock) + world_v * surface_lock;
    return true;
}

float strokeField(int x, int y, const StylizeAOVSample& aov, const StyleProfile& profile, int frame_index) {
    const float brush_scale = std::max(0.05f, profile.material.brush_scale);
    const float pixel_scale = surfacePixelScale(aov, profile, 14.0f + brush_scale * 28.0f);
    const float coherent_frame = static_cast<float>(frame_index) * (1.0f - profile.temporal_coherence);

    float nx = aov.hit ? aov.normal.x : 0.0f;
    float ny = aov.hit ? aov.normal.y : 1.0f;
    float nz = aov.hit ? aov.normal.z : 0.0f;
    float angle = std::atan2(nz + ny * 0.35f, nx + 0.001f);
    switch (profile.material.stroke_direction) {
        case StrokeDirectionMode::Vertical: angle = 1.5707963f; break;
        case StrokeDirectionMode::Horizontal: angle = 0.0f; break;
        case StrokeDirectionMode::Diagonal: angle = 0.7853982f; break;
        case StrokeDirectionMode::CrossHatch:
            angle = ((x / 48 + y / 48) & 1) ? 0.7853982f : -0.7853982f;
            break;
        case StrokeDirectionMode::SurfaceNormal:
        default:
            break;
    }
    float ca = std::cos(angle);
    float sa = std::sin(angle);
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

struct WetOilStroke {
    float bristle = 0.0f;
    float ridge = 0.0f;
    float drag = 0.0f;
    float body = 0.0f;
};

WetOilStroke wetOilStrokeModel(int x,
                               int y,
                               const StylizeAOVSample& aov,
                               const StyleProfile& profile,
                               int frame_index,
                               float stroke) {
    const float brush_scale = std::max(0.05f, profile.material.brush_scale);
    const float oil_body = saturate(profile.material.oil_body);
    const float buildup = saturate(profile.material.bristle_buildup);
    const float pixel_scale = surfacePixelScale(aov, profile, 7.0f + brush_scale * 18.0f);
    const float coherent_frame = static_cast<float>(frame_index) * (1.0f - profile.temporal_coherence);
    const int seed = static_cast<int>(coherent_frame) + 311 + static_cast<int>(aov.material_id & 0xFFu) * 29;

    float nx = aov.hit ? aov.normal.x : 0.0f;
    float ny = aov.hit ? aov.normal.y : 1.0f;
    float angle = std::atan2(ny, nx + 0.001f);
    if (profile.material.stroke_direction == StrokeDirectionMode::Vertical) {
        angle = 1.5707963f;
    } else if (profile.material.stroke_direction == StrokeDirectionMode::Horizontal) {
        angle = 0.0f;
    } else if (profile.material.stroke_direction == StrokeDirectionMode::Diagonal) {
        angle = 0.7853982f;
    }

    const float ca = std::cos(angle);
    const float sa = std::sin(angle);
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
    oil.ridge = smoothstep(0.42f, 0.96f, std::abs(stroke) + body_noise * 0.28f + buildup * 0.24f);
    oil.drag = smoothstep(0.18f, 0.88f, body_noise * (0.55f + oil_body * 0.45f) + std::abs(bristle));
    oil.body = body_noise;
    return oil;
}

Vec3 applySkyLayer(const Vec3& input_color,
                   const StylizeAOVSample& aov,
                   int x,
                   int y,
                   int frame_index,
                   const StyleProfile& profile) {
    const StylizedSkySettings& sky = profile.sky;
    if (!sky.enabled) {
        return input_color;
    }

    Vec3 dir = aov.view_dir.length_squared() > 1e-8f ? aov.view_dir.normalize() : Vec3(0.0f, 0.0f, -1.0f);
    Vec3 sun_dir = aov.sun_dir.length_squared() > 1e-8f ? aov.sun_dir.normalize() : Vec3(0.32f, 0.82f, 0.46f).normalize();
    const float azimuth = std::atan2(dir.z, dir.x);
    const float sphere_u = azimuth / (2.0f * 3.14159265f) + 0.5f;
    const float sphere_v = saturate(dir.y * 0.5f + 0.5f);
    const float horizon = 1.0f - smoothstep(-0.08f, 0.34f, dir.y);
    const float sky_t = smoothstep(-0.16f, 0.88f, dir.y);
    const float coherent_frame = static_cast<float>(frame_index) * (1.0f - profile.temporal_coherence);
    const bool cartoon_sky = sky.style == SkyStylePreset::CartoonCel;
    const bool ink_sky = sky.style == SkyStylePreset::InkWash;
    const bool sunset_sky = sky.style == SkyStylePreset::SunsetBands;
    const bool clear_sky = sky.style == SkyStylePreset::ClearGradient;

    Vec3 gradient = Vec3::lerp(sky.horizon_color, sky.zenith_color, sky_t);
    if (cartoon_sky) {
        const float band_noise = (fbmNoise(sphere_u * 3.0f, sky_t * 1.7f, 911) - 0.5f) * 0.045f;
        const float band = std::floor(saturate(sky_t + band_noise) * 5.0f + 0.5f) / 5.0f;
        gradient = Vec3::lerp(sky.horizon_color, sky.zenith_color, band);
    } else if (sunset_sky) {
        const float wave = (fbmNoise(sphere_u * 2.4f, sky_t * 1.2f, 937) - 0.5f) * 0.075f;
        const float lower_band = smoothstep(0.03f, 0.30f, sky_t + wave) * (1.0f - smoothstep(0.36f, 0.58f, sky_t + wave));
        const float upper_band = smoothstep(0.30f, 0.58f, sky_t - wave * 0.45f) * (1.0f - smoothstep(0.68f, 0.88f, sky_t));
        const Vec3 peach = Vec3::lerp(sky.horizon_color, sky.sun_glow_color, 0.72f);
        const Vec3 violet = Vec3::lerp(sky.zenith_color, profile.palette_shadow, 0.20f);
        gradient = Vec3::lerp(gradient, peach, lower_band * 0.48f);
        gradient = Vec3::lerp(gradient, violet, upper_band * 0.22f);
    } else if (!ink_sky) {
        const float input_luma = std::max(0.025f, luminance(input_color));
        const float gradient_luma = std::max(0.025f, luminance(gradient));
        gradient = clamp01(gradient * std::clamp(input_luma / gradient_luma, 0.65f, 1.45f));
    }

    const float gradient_mix = cartoon_sky || sunset_sky || ink_sky
        ? saturate(0.72f + sky.gradient_strength * 0.28f)
        : saturate(sky.gradient_strength);
    Vec3 color = Vec3::lerp(input_color, gradient, gradient_mix);

    const float haze_amount = saturate(sky.horizon_haze) * horizon;
    const Vec3 haze_color = clamp01(Vec3::lerp(sky.horizon_color, sky.sun_glow_color, 0.18f));
    color = Vec3::lerp(color, haze_color, haze_amount * (sunset_sky ? 0.78f : 0.55f));

    const float brush_scale = std::max(0.10f, sky.cloud_brush_scale);
    const float nishita_scale = aov.nishita_clouds_enabled ? aov.nishita_cloud_scale : 1.0f;
    const float cloud_scale = std::max(0.08f, brush_scale * nishita_scale);
    const float cloud_coverage = aov.nishita_clouds_enabled ? aov.nishita_cloud_coverage : 0.42f;
    const float cloud_density = aov.nishita_clouds_enabled ? aov.nishita_cloud_density : 0.72f;
    const float wind = coherent_frame * 0.018f * saturate(sky.wind_smear);
    const float world_u = fract01(sphere_u + aov.nishita_cloud_offset_x * 0.0025f + wind);
    const float world_v = saturate(sphere_v + aov.nishita_cloud_offset_z * 0.0025f + wind * 0.10f);
    const int cloud_seed = 1207 + aov.nishita_cloud_seed * 37;
    const float cloud_band = smoothstep(0.18f, 0.74f, world_v) * (1.0f - smoothstep(0.88f, 1.0f, world_v));
    float cloud_shape = 0.0f;
    float cloud_detail = 0.0f;
    float cloud_shadow = 0.0f;
    float cloud_highlight = 0.0f;
    float cloud_rim = 0.0f;
    if (!clear_sky) {
        const float banks = sunset_sky ? 3.0f : (cartoon_sky ? 3.5f : 4.0f);
        const float bank_pos = world_v * banks;
        const int base_bank = static_cast<int>(std::floor(bank_pos));
        const float roundness = std::clamp(sky.cloud_roundness, 0.05f, 1.0f);
        const float groups_per_bank = std::clamp(1.8f + cloud_scale * 0.42f + cloud_coverage * 2.4f, 2.0f, 6.5f);
        for (int bank_offset = -1; bank_offset <= 1; ++bank_offset) {
            const int bank = base_bank + bank_offset;
            const float bank_center = (static_cast<float>(bank) + 0.5f) / banks;
            const float bank_wave = (fbmNoise(world_u * 2.2f + static_cast<float>(bank) * 1.7f, 0.37f, cloud_seed + 41) - 0.5f) * 0.055f;
            const float bank_v = bank_center + bank_wave + horizon * 0.06f;
            const float bank_keep = 1.0f - smoothstep(0.105f, 0.205f, std::abs(world_v - bank_v));
            if (bank_keep <= 0.001f) {
                continue;
            }

            const float group_pos = world_u * groups_per_bank + hashNoise(bank, cloud_seed, 9) * 0.65f;
            const int base_group = static_cast<int>(std::floor(group_pos));
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
                    const float arch = 1.0f - std::abs(static_cast<float>(lobe) - 2.5f) / 2.5f;
                    const float lobe_v = group_center_v + arch * (0.035f + roundness * 0.035f) + (lh1 - 0.5f) * 0.026f;
                    float du = std::abs(world_u - lobe_u);
                    du = std::min(du, 1.0f - du);
                    const float dv = (world_v - lobe_v) * (2.55f - roundness * 0.80f);
                    const float rx = group_width * (0.58f + lh2 * 0.38f);
                    const float ry = rx * (0.40f + roundness * 0.24f);
                    const float d = std::sqrt((du * du) / std::max(1e-5f, rx * rx) + (dv * dv) / std::max(1e-5f, ry * ry));
                    const float puff = (1.0f - smoothstep(0.78f, 1.0f, d)) * bank_keep;
                    const float interior = (1.0f - smoothstep(0.38f, 0.90f, d)) * bank_keep;
                    const float lower = smoothstep(-ry * 0.20f, ry * 1.20f, -dv);
                    const float upper = smoothstep(-ry * 1.15f, ry * 0.05f, dv);
                    const float rim = smoothstep(0.72f, 0.98f, d) * puff;
                    cloud_shape = std::max(cloud_shape, puff);
                    cloud_detail += interior * (0.25f + lh2 * 0.18f);
                    cloud_shadow = std::max(cloud_shadow, puff * lower * (0.35f + lh0 * 0.22f));
                    cloud_highlight = std::max(cloud_highlight, interior * upper * (0.42f + lh1 * 0.32f));
                    cloud_rim = std::max(cloud_rim, rim);
                }
            }
        }
        const float coverage_gate = Vec3::lerpf(0.72f, 0.12f, saturate(cloud_coverage));
        const float keep = smoothstep(coverage_gate, 1.0f, cloud_shape + cloud_detail * 0.18f);
        cloud_shape *= keep;
        cloud_shadow *= keep;
        cloud_highlight *= keep;
        cloud_rim *= keep;
    }

    const float cloud_alpha = saturate(sky.cloud_brush_strength) * saturate(cloud_density) * cloud_band * saturate(cloud_shape);
    Vec3 cloud_tint = cartoon_sky
        ? Vec3(1.0f, 0.96f, 0.78f)
        : clamp01(Vec3::lerp(Vec3(0.96f, 0.92f, 0.82f), profile.palette_highlight, ink_sky ? 0.08f : 0.28f));
    if (ink_sky) {
        cloud_tint = Vec3::lerp(Vec3(0.88f, 0.88f, 0.80f), profile.palette_shadow, 0.16f);
    }
    const Vec3 shadow_tint = cartoon_sky
        ? Vec3(0.58f, 0.72f, 0.92f)
        : clamp01(Vec3::lerp(profile.palette_shadow, sky.zenith_color, ink_sky ? 0.42f : 0.28f));
    const Vec3 highlight_tint = cartoon_sky
        ? Vec3(1.0f, 0.98f, 0.86f)
        : clamp01(Vec3::lerp(Vec3(1.0f, 0.96f, 0.84f), sky.sun_glow_color, sunset_sky ? 0.45f : 0.18f));
    color = Vec3::lerp(color, shadow_tint, cloud_alpha * cloud_shadow * (cartoon_sky ? 0.48f : 0.34f));
    color = Vec3::lerp(color, cloud_tint, cloud_alpha * (cartoon_sky ? 0.92f : 0.66f));
    color = Vec3::lerp(color, highlight_tint, cloud_alpha * cloud_highlight * (cartoon_sky ? 0.64f : 0.46f));
    if (cartoon_sky) {
        color = Vec3::lerp(color, Vec3(0.24f, 0.42f, 0.76f), cloud_alpha * cloud_rim * 0.16f);
    }
    if (!cartoon_sky) {
        const float fiber = fbmNoise(world_u * 18.0f, world_v * 5.0f, cloud_seed + 82) - 0.5f;
        color = color * (1.0f + fiber * cloud_alpha * (ink_sky ? 0.08f : 0.12f));
    }

    const float sun_dot = saturate(Vec3::dot(dir, sun_dir));
    const float sun_size = std::max(0.01f, aov.sun_size_degrees);
    const float low_sun_factor = aov.sun_elevation_degrees < 15.0f
        ? 1.0f + (15.0f - std::max(aov.sun_elevation_degrees, -10.0f)) * 0.04f
        : 1.0f;
    const float style_sun_scale = std::max(0.5f, sky.sun_disc_scale) * (cartoon_sky ? 1.35f : 1.0f);
    const float sun_radius = sun_size * low_sun_factor * style_sun_scale * (3.14159265f / 180.0f) * 0.5f;
    const float sun_threshold = std::cos(sun_radius);
    const float sun_disk = cartoon_sky
        ? (sun_dot > sun_threshold ? 1.0f : 0.0f)
        : smoothstep(sun_threshold, 1.0f, sun_dot);
    const float sun_glow = std::pow(sun_dot, sunset_sky ? 8.0f : 14.0f) * (0.24f + saturate(sky.gradient_strength) * 0.64f);
    color = Vec3::lerp(color, clamp01(color + sky.sun_glow_color * (sunset_sky ? 0.72f : 0.45f)), saturate(sun_glow) * (cartoon_sky ? 0.16f : 0.38f));
    const Vec3 sun_color = cartoon_sky
        ? clamp01(sky.sun_glow_color * 1.55f + Vec3(0.08f, 0.04f, 0.0f))
        : clamp01(sky.sun_glow_color * 1.35f);
    color = Vec3::lerp(color, sun_color, sun_disk * (cartoon_sky ? 1.0f : 0.86f));

    if (!cartoon_sky && !clear_sky) {
        const float stroke = fbmNoise(sphere_u * 10.0f + wind * 4.0f, sphere_v * 3.0f, 1423) - 0.5f;
        color = color * (1.0f + stroke * saturate(sky.cloud_brush_strength) * (ink_sky ? 0.06f : 0.08f));
    }
    (void)x;
    (void)y;

    return clamp01(color);
}

Vec3 outlineColor(const StyleProfile& profile, const StylizeAOVSample& aov, const Vec3& base_albedo) {
    const StylizedOutlineSettings& outline = profile.outline;
    switch (outline.color_mode) {
        case OutlineColorMode::CustomColor:
            return clamp01(outline.custom_color);
        case OutlineColorMode::MaterialTint: {
            Vec3 material_ink = base_albedo * (0.34f + outline.color_bleed * 0.22f);
            return clamp01(Vec3::lerp(profile.palette_shadow * 0.55f, material_ink, outline.color_bleed));
        }
        case OutlineColorMode::WarmPaint:
            return clamp01(Vec3::lerp(profile.palette_shadow * 0.55f, Vec3(0.34f, 0.18f, 0.08f), 0.35f + outline.color_bleed * 0.35f));
        case OutlineColorMode::CoolPencil:
            return clamp01(Vec3::lerp(profile.palette_shadow * 0.45f, Vec3(0.12f, 0.14f, 0.17f), 0.55f));
        case OutlineColorMode::PaletteShadow:
        default:
            return clamp01(profile.palette_shadow * 0.55f);
    }
}

float outlineTexture(int x, int y, const StylizeAOVSample& aov, const StyleProfile& profile, int frame_index) {
    const StylizedOutlineSettings& outline = profile.outline;
    const float edge = saturate(aov.edge);
    const float distance = aov.hit ? std::log1p(std::max(0.0f, aov.depth)) : 0.0f;
    const float distance_factor = aov.hit
        ? 1.0f / (1.0f + distance * (0.20f + outline.distance_thinning * 0.85f))
        : 1.0f;
    const float adaptive_width = Vec3::lerpf(outline.width, outline.width * distance_factor, outline.distance_thinning);
    const float width = std::max(0.1f, adaptive_width);
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
    const float taper = smoothstep(0.02f, 0.90f, edge) * (1.0f - outline.taper * smoothstep(0.55f, 1.0f, edge));
    float coverage = saturate(edge * (0.75f + width * 0.45f));
    const float fine_detail = saturate((1.0f - distance_factor) * (1.0f - smoothstep(0.45f, 1.0f, edge)));
    coverage *= 1.0f - fine_detail * outline.detail_protection * 0.82f;

    switch (outline.line_type) {
        case OutlineLineType::OilPaint:
            coverage *= 0.80f + grain * 0.42f;
            coverage += smoothstep(0.62f, 0.95f, fiber) * outline.color_bleed * 0.18f;
            break;
        case OutlineLineType::Pencil:
            coverage *= 0.52f + fiber * 0.58f;
            coverage *= 0.85f + grain * 0.18f;
            break;
        case OutlineLineType::DryBrush: {
            const float dry_cut = smoothstep(0.18f, 0.86f, grain + fiber * 0.35f);
            coverage *= 0.42f + dry_cut * 0.78f;
            break;
        }
        case OutlineLineType::Pressure:
            coverage *= 0.68f + smoothstep(0.18f, 0.95f, edge + grain * 0.18f) * 0.55f;
            coverage *= 1.0f - outline.taper * 0.35f * (1.0f - edge);
            break;
        case OutlineLineType::Ink:
        default:
            coverage *= 0.92f + grain * 0.16f;
            break;
    }

    const float breakup = saturate(outline.break_up);
    if (breakup > 0.001f) {
        const float keep = smoothstep(breakup * 0.45f, 0.96f, grain + edge * 0.28f);
        coverage *= Vec3::lerpf(1.0f, keep, breakup);
    }

    return saturate(coverage * (0.55f + taper * 0.45f));
}

} // namespace

Vec3 applyPostProcess(const Vec3& input_color,
                      int x,
                      int y,
                      int frame_index,
                      const StylizeModeState& state) {
    return applyPostProcess(input_color, StylizeAOVSample{}, x, y, frame_index, state);
}

Vec3 applyPostProcess(const Vec3& input_color,
                      const StylizeAOVSample& aov,
                      int x,
                      int y,
                      int frame_index,
                      const StylizeModeState& state) {
    if (!state.enabled) {
        return input_color;
    }

    const StyleProfile& profile = state.profile;
    const float strength = saturate(profile.global_strength);
    if (strength <= 0.0001f) {
        return input_color;
    }

    Vec3 color = clamp01(input_color);
    const Vec3 base_albedo = aov.hit ? clamp01(aov.albedo) : color;
    const float albedo_luma = luminance(base_albedo);
    const bool material_domain = aov.hit;

    if (aov.valid && !aov.hit) {
        color = applySkyLayer(color, aov, x, y, frame_index, profile);
    }

    const float palette_influence = saturate(profile.material.palette_influence);
    const float edge_guard = aov.hit
        ? (1.0f - saturate(aov.edge * (0.35f + profile.material.edge_respect * 0.65f)))
        : 1.0f;

    if (material_domain) {
        Vec3 palette_color = paletteRamp(albedo_luma, profile);
        if (aov.hit) {
            palette_color = preserveMaterialHue(
                palette_color,
                base_albedo,
                profile.material.material_color_preservation);
        }
        const float palette_mix = palette_influence * strength * (0.16f + profile.material.color_simplification * 0.46f);
        color = Vec3::lerp(color, palette_color, palette_mix);

        const float material_guard = aov.hit
            ? (0.35f + 0.65f * (1.0f - saturate(profile.material.material_color_preservation)))
            : 1.0f;
        color = simplifyColor(color, strength * profile.material.color_simplification * 0.48f * edge_guard * material_guard);

        if (profile.material.enabled && profile.material.brush_strength > 0.001f) {
            const float stroke = strokeField(x, y, aov, profile, frame_index);
            const float brush_amount = profile.material.brush_strength * strength * edge_guard;
            const float dry_mask = smoothstep(0.12f, 0.62f, std::abs(stroke) + profile.material.dry_brush * 0.35f);
            Vec3 palette_tint = Vec3::lerp(profile.palette_shadow, profile.palette_highlight, saturate(albedo_luma + stroke * 0.28f));
            if (aov.hit) {
                palette_tint = preserveMaterialHue(
                    palette_tint,
                    base_albedo,
                    profile.material.material_color_preservation);
            }
            const Vec3 stroke_tint = Vec3::lerp(base_albedo, palette_tint, palette_influence);
            if (profile.material.wet_oil_model) {
                const WetOilStroke oil = wetOilStrokeModel(x, y, aov, profile, frame_index, stroke);
                const float oil_body = saturate(profile.material.oil_body);
                const float paint_load = saturate(profile.material.paint_load);
                const float pickup = saturate(profile.material.pickup_rate);
                const float deposit = saturate(profile.material.deposit_rate);
                const float wet_visibility = 0.85f + oil_body * 1.25f + paint_load * 0.45f;
                const float wet_mask = saturate(brush_amount * wet_visibility * (0.14f + oil_body * 0.24f + oil.drag * 0.34f));
                const float deposit_mask = saturate(brush_amount * paint_load * deposit * wet_visibility * (0.18f + oil.ridge * 0.34f));
                const float bristle_mask = saturate(brush_amount * (0.22f + oil_body * 0.34f + profile.material.bristle_buildup * 0.22f));
                const Vec3 picked_color = Vec3::lerp(base_albedo, color, pickup * (0.35f + oil_body * 0.65f));
                const Vec3 carried_color = Vec3::lerp(picked_color, stroke_tint, saturate(deposit * paint_load));
                const float body_luma = (oil.body - 0.5f) * (0.12f + oil_body * 0.18f);
                const float bristle_luma = (stroke + oil.bristle) * (0.18f + oil_body * 0.24f);
                Vec3 material_stroke = clamp01(base_albedo * (1.0f + body_luma + bristle_luma));
                material_stroke = Vec3::lerp(material_stroke, carried_color, saturate(palette_influence * 0.55f + deposit * 0.25f));
                color = Vec3::lerp(color, material_stroke, wet_mask);
                color = Vec3::lerp(color, carried_color, deposit_mask);
                color = color * (1.0f + (stroke + oil.bristle) * bristle_mask);
                color = Vec3::lerp(color, color + profile.palette_highlight * (0.035f + 0.035f * palette_influence), oil.ridge * brush_amount * oil_body);
                color = Vec3::lerp(color, color * (0.92f - oil.ridge * 0.08f), profile.material.dry_brush * brush_amount * (1.0f - oil.drag));
            } else {
                color = color * (1.0f + stroke * brush_amount * 0.16f);
                color = Vec3::lerp(color, stroke_tint, brush_amount * dry_mask * profile.material.dry_brush * 0.18f);
            }
        }

        if (profile.material.pigment_thickness > 0.001f) {
            const float pigment = profile.material.pigment_thickness * strength;
            color = Vec3(
                std::pow(saturate(color.x), 1.0f + pigment * 0.18f),
                std::pow(saturate(color.y), 1.0f + pigment * 0.14f),
                std::pow(saturate(color.z), 1.0f + pigment * 0.10f)
            );
            const float edge_pigment = aov.hit ? saturate(aov.edge * 0.8f) : 0.0f;
            color = Vec3::lerp(color, color + profile.palette_highlight * 0.035f * palette_influence, pigment * (1.0f - edge_pigment));
            color = Vec3::lerp(color, Vec3::lerp(base_albedo, profile.palette_shadow * 0.7f, palette_influence), pigment * edge_pigment * 0.24f);
        }

        if (profile.outline.enabled && aov.edge > 0.001f) {
            const float line = outlineTexture(x, y, aov, profile, frame_index);
            const float edge = saturate(line * profile.outline.strength * strength);
            const Vec3 ink = outlineColor(profile, aov, base_albedo);
            color = Vec3::lerp(color, ink, edge);
        }
    }

    return clamp01(Vec3::lerp(input_color, color, strength));
}

} // namespace Stylize
