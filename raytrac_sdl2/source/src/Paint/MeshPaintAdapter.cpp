#include "Paint/MeshPaintAdapter.h"

#include "scene_data.h"
#include "Triangle.h"
#include "Material.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include <algorithm>
#include <cmath>
#include <tuple>

namespace Paint {

namespace {

TextureType toTextureType(PaintChannel channel) {
    switch (channel) {
        case PaintChannel::BaseColor: return TextureType::Albedo;
        case PaintChannel::Normal: return TextureType::Normal;
        case PaintChannel::Roughness: return TextureType::Roughness;
        case PaintChannel::Metallic: return TextureType::Metallic;
        case PaintChannel::Emission: return TextureType::Emission;
        case PaintChannel::Mask: return TextureType::Unknown;
        case PaintChannel::Transmission: return TextureType::Transmission;
        case PaintChannel::Opacity: return TextureType::Opacity;
    }
    return TextureType::Unknown;
}

const char* channelName(PaintChannel channel) {
    switch (channel) {
        case PaintChannel::BaseColor: return "Base Color";
        case PaintChannel::Normal: return "Normal";
        case PaintChannel::Roughness: return "Roughness";
        case PaintChannel::Metallic: return "Metallic";
        case PaintChannel::Emission: return "Emission";
        case PaintChannel::Mask: return "Mask";
        case PaintChannel::Transmission: return "Transmission";
        case PaintChannel::Opacity: return "Opacity";
    }
    return "Channel";
}

CompactVec4 defaultChannelPixel(PaintChannel channel) {
    switch (channel) {
        case PaintChannel::BaseColor:
        case PaintChannel::Emission:
            return CompactVec4(255, 255, 255, 255);
        case PaintChannel::Normal:
            return CompactVec4(128, 128, 255, 255);
        case PaintChannel::Roughness:
        case PaintChannel::Metallic:
        case PaintChannel::Mask:
        case PaintChannel::Transmission:
            return CompactVec4(0, 0, 0, 255);
        case PaintChannel::Opacity:
            // Default = fully visible: a freshly-created opacity canvas should
            // not silently hide the surface. Erase reverts texels to this.
            return CompactVec4(255, 255, 255, 255);
    }
    return CompactVec4(255, 255, 255, 255);
}

std::shared_ptr<Texture> createBlankTexture(const std::string& name, int resolution, TextureType type) {
    auto texture = std::make_shared<Texture>(name, resolution, resolution, type);
    texture->width = resolution;
    texture->height = resolution;
    texture->pixels.assign(static_cast<size_t>(resolution) * static_cast<size_t>(resolution), CompactVec4(255, 255, 255, 255));
    texture->m_is_loaded = true;
    return texture;
}

void fillTextureDefault(Texture& texture, PaintChannel channel) {
    const CompactVec4 fill = defaultChannelPixel(channel);
    for (auto& pixel : texture.pixels) {
        pixel = fill;
    }
}

CompactVec4 makeBrushPixel(PaintChannel channel, const BrushSettings& brush) {
    const auto toByte = [](float value) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    const auto linearToSrgbByte = [&](float value) -> uint8_t {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        const float srgb = (clamped <= 0.0031308f)
            ? (clamped * 12.92f)
            : (1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f);
        return toByte(srgb);
    };

    switch (channel) {
        case PaintChannel::BaseColor:
            // Albedo textures are sampled as sRGB->linear later, so painted values must be
            // encoded back to sRGB bytes before storing or repeated mix/smudge will darken.
            return CompactVec4(
                linearToSrgbByte(brush.color.x),
                linearToSrgbByte(brush.color.y),
                linearToSrgbByte(brush.color.z),
                255);
        case PaintChannel::Emission:
            return CompactVec4(toByte(brush.color.x), toByte(brush.color.y), toByte(brush.color.z), 255);
        case PaintChannel::Normal: {
            // Tangent-space normals must point out of the surface (n.z >= 0).
            // brush.color is a free RGB picker, so a user-chosen red (1,0,0)
            // would otherwise decode to (1,-1,-1) → n.z negative → BSDFs
            // see a normal pointing into the surface, producing flipped
            // shading and NaN reflection terms. Clamp z to non-negative
            // before re-normalising.
            Vec3 n(
                brush.color.x * 2.0f - 1.0f,
                brush.color.y * 2.0f - 1.0f,
                brush.color.z * 2.0f - 1.0f);
            if (n.z < 0.0f) n.z = 0.0f;
            n = n.normalize();
            // Edge case: clamping z to 0 with x=y=0 leaves a zero vector.
            // Snap to the flat normal in that pathological case.
            if (!std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z) ||
                (n.x == 0.0f && n.y == 0.0f && n.z == 0.0f)) {
                n = Vec3(0.0f, 0.0f, 1.0f);
            }
            return CompactVec4(
                toByte(n.x * 0.5f + 0.5f),
                toByte(n.y * 0.5f + 0.5f),
                toByte(n.z * 0.5f + 0.5f),
                255);
        }
        case PaintChannel::Roughness:
        case PaintChannel::Metallic:
        case PaintChannel::Mask:
        case PaintChannel::Transmission: {
            const float grayscale = std::clamp((brush.color.x + brush.color.y + brush.color.z) / 3.0f, 0.0f, 1.0f);
            const uint8_t value = toByte(grayscale);
            return CompactVec4(value, value, value, 255);
        }
        case PaintChannel::Opacity: {
            // Write the same value into all four channels (incl. alpha) so the
            // material_preview_frag opacity sampler returns the painted mask
            // regardless of which path it picks: RGBA-alpha mode reads .a,
            // grayscale-mask mode reads .r. If we left .a hard-coded at 255,
            // any material whose flags bit 256 (or shared albedo+opacity slot)
            // selected the alpha branch would see "always opaque" and the dab
            // would have no visible effect.
            const float grayscale = std::clamp((brush.color.x + brush.color.y + brush.color.z) / 3.0f, 0.0f, 1.0f);
            const uint8_t value = toByte(grayscale);
            return CompactVec4(value, value, value, value);
        }
    }
    return CompactVec4(255, 255, 255, 255);
}
void rotateBrushCoords(float& x, float& y, float degrees) {
    if (std::abs(degrees) <= 0.001f) {
        return;
    }

    const float radians = degrees * 3.14159265f / 180.0f;
    const float cs = std::cos(radians);
    const float sn = std::sin(radians);
    const float rx = x * cs - y * sn;
    const float ry = x * sn + y * cs;
    x = rx;
    y = ry;
}

Vec3 applyTintToSample(const Vec3& sampled, const Vec3& tint, float tint_strength, PaintTextureTintMode tint_mode) {
    const float strength = std::clamp(tint_strength, 0.0f, 1.0f);
    Vec3 tinted = sampled;
    switch (tint_mode) {
        case PaintTextureTintMode::Multiply:
            tinted = Vec3(
                std::clamp(sampled.x * tint.x, 0.0f, 1.0f),
                std::clamp(sampled.y * tint.y, 0.0f, 1.0f),
                std::clamp(sampled.z * tint.z, 0.0f, 1.0f));
            break;
        case PaintTextureTintMode::Recolor: {
            const float luminance = std::clamp(sampled.x * 0.299f + sampled.y * 0.587f + sampled.z * 0.114f, 0.0f, 1.0f);
            tinted = Vec3(
                std::clamp(tint.x * luminance, 0.0f, 1.0f),
                std::clamp(tint.y * luminance, 0.0f, 1.0f),
                std::clamp(tint.z * luminance, 0.0f, 1.0f));
            break;
        }
        case PaintTextureTintMode::Overlay: {
            const auto overlay = [](float base, float blend) -> float {
                return base < 0.5f
                    ? (2.0f * base * blend)
                    : (1.0f - 2.0f * (1.0f - base) * (1.0f - blend));
            };
            tinted = Vec3(
                std::clamp(overlay(sampled.x, tint.x), 0.0f, 1.0f),
                std::clamp(overlay(sampled.y, tint.y), 0.0f, 1.0f),
                std::clamp(overlay(sampled.z, tint.z), 0.0f, 1.0f));
            break;
        }
    }
    return Vec3(
        sampled.x + (tinted.x - sampled.x) * strength,
        sampled.y + (tinted.y - sampled.y) * strength,
        sampled.z + (tinted.z - sampled.z) * strength);
}

CompactVec4 makeBrushTexturePixel(PaintChannel channel, const BrushSettings& brush, float nx, float ny) {
    if (!brush.use_paint_texture || !brush.paint_texture || !brush.paint_texture->is_loaded()) {
        return makeBrushPixel(channel, brush);
    }

    float sx = nx * std::max(0.01f, brush.alpha_scale);
    float sy = ny * std::max(0.01f, brush.alpha_scale);
    rotateBrushCoords(sx, sy, brush.alpha_rotation_degrees);
    const float u = std::clamp(sx * 0.5f + 0.5f, 0.0f, 1.0f);
    const float v = std::clamp(sy * 0.5f + 0.5f, 0.0f, 1.0f);

    const auto toByte = [](float value) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    const auto linearToSrgbByte = [&](float value) -> uint8_t {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        const float srgb = (clamped <= 0.0031308f)
            ? (clamped * 12.92f)
            : (1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f);
        return toByte(srgb);
    };

    switch (channel) {
        case PaintChannel::BaseColor: {
            Vec3 sampled = brush.paint_texture->get_color_bilinear(u, v);
            sampled = applyTintToSample(sampled, brush.color, brush.paint_texture_tint_strength, brush.paint_texture_tint_mode);
            return CompactVec4(
                linearToSrgbByte(sampled.x),
                linearToSrgbByte(sampled.y),
                linearToSrgbByte(sampled.z),
                255);
        }
        case PaintChannel::Emission: {
            Vec3 sampled = brush.paint_texture->get_color_bilinear(u, v);
            sampled = applyTintToSample(sampled, brush.color, brush.paint_texture_tint_strength, brush.paint_texture_tint_mode);
            return CompactVec4(toByte(sampled.x), toByte(sampled.y), toByte(sampled.z), 255);
        }
        case PaintChannel::Roughness:
        case PaintChannel::Metallic:
        case PaintChannel::Mask:
        case PaintChannel::Transmission: {
            const float texture_value = brush.paint_texture->sampleIntensity(u, v);
            const float tint_value = std::clamp((brush.color.x + brush.color.y + brush.color.z) / 3.0f, 0.0f, 1.0f);
            const float grayscale = texture_value + (texture_value * tint_value - texture_value) *
                std::clamp(brush.paint_texture_tint_strength, 0.0f, 1.0f);
            const uint8_t value = toByte(grayscale);
            return CompactVec4(value, value, value, 255);
        }
        case PaintChannel::Opacity: {
            // Mirror the same all-channel write that makeBrushPixel does for
            // Opacity: shader reads .a or .r depending on material flags;
            // putting the value in both leaves the mask working in either mode.
            const float texture_value = brush.paint_texture->sampleIntensity(u, v);
            const float tint_value = std::clamp((brush.color.x + brush.color.y + brush.color.z) / 3.0f, 0.0f, 1.0f);
            const float grayscale = texture_value + (texture_value * tint_value - texture_value) *
                std::clamp(brush.paint_texture_tint_strength, 0.0f, 1.0f);
            const uint8_t value = toByte(grayscale);
            return CompactVec4(value, value, value, value);
        }
        case PaintChannel::Normal: {
            Vec3 sampled = brush.paint_texture->get_color_bilinear(u, v);
            Vec3 n(
                sampled.x * 2.0f - 1.0f,
                sampled.y * 2.0f - 1.0f,
                sampled.z * 2.0f - 1.0f);
            // Defensive: a paint_texture wired in by the user that isn't a
            // real tangent-space normal map (random photo, height map, etc.)
            // can yield n.z < 0. Clamp to keep the normal on the
            // outward-facing hemisphere.
            if (n.z < 0.0f) n.z = 0.0f;
            n = n.normalize();
            if (!std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z) ||
                (n.x == 0.0f && n.y == 0.0f && n.z == 0.0f)) {
                n = Vec3(0.0f, 0.0f, 1.0f);
            }
            return CompactVec4(
                toByte(n.x * 0.5f + 0.5f),
                toByte(n.y * 0.5f + 0.5f),
                toByte(n.z * 0.5f + 0.5f),
                255);
        }
    }

    return makeBrushPixel(channel, brush);
}

float computeBrushWeight(float distance_px, float radius_px, float falloff) {
    if (radius_px <= 0.0f) {
        return 0.0f;
    }

    // Allow sub-pixel brush sizes for low texel-density meshes.
    if (radius_px <= 1.0f) {
        const float effective_radius = radius_px + 0.75f;
        return std::clamp(1.0f - (distance_px / effective_radius), 0.0f, 1.0f);
    }

    const float normalized = std::clamp(distance_px / radius_px, 0.0f, 1.0f);
    const float inner = std::clamp(1.0f - falloff, 0.0f, 1.0f);
    if (normalized <= inner) {
        return 1.0f;
    }

    const float t = std::clamp((normalized - inner) / std::max(0.001f, 1.0f - inner), 0.0f, 1.0f);
    return 1.0f - (t * t * (3.0f - 2.0f * t));
}

// Same falloff curve as computeBrushWeight but the distance is already
// normalised (0 = brush centre, 1 = brush edge). The elliptical footprint
// path uses this so the unit-circle metric stays exact regardless of axis
// scaling.
float computeBrushWeightNormalized(float dist_norm, float radius_px, float falloff) {
    if (radius_px <= 0.0f) {
        return 0.0f;
    }
    if (radius_px <= 1.0f) {
        const float scale = radius_px / (radius_px + 0.75f);
        return std::clamp(1.0f - dist_norm * scale, 0.0f, 1.0f);
    }
    const float normalized = std::clamp(dist_norm, 0.0f, 1.0f);
    const float inner = std::clamp(1.0f - falloff, 0.0f, 1.0f);
    if (normalized <= inner) {
        return 1.0f;
    }
    const float t = std::clamp((normalized - inner) / std::max(0.001f, 1.0f - inner), 0.0f, 1.0f);
    return 1.0f - (t * t * (3.0f - 2.0f * t));
}

// World-space lengths of ∂P/∂U and ∂P/∂V across the active UV set on `tri`.
// Tells the caller "1 unit of U covers `out_world_per_u` world units, 1 unit
// of V covers `out_world_per_v`". Falls back to (1,1) on degenerate input
// so the caller transparently drops to the isotropic pixel-space footprint.
void computeTriangleUvJacobianLengths(const Triangle& tri, int uv_set,
                                       float& out_world_per_u, float& out_world_per_v) {
    out_world_per_u = 1.0f;
    out_world_per_v = 1.0f;

    Vec2 uv0, uv1, uv2;
    if (uv_set > 0 && static_cast<size_t>(uv_set) < tri.getUVSetCount()) {
        std::tie(uv0, uv1, uv2) = tri.getUVSetCoordinates(static_cast<size_t>(uv_set));
    } else {
        std::tie(uv0, uv1, uv2) = tri.getUVCoordinates();
    }

    const Vec3 p0 = tri.getVertexPosition(0);
    const Vec3 e1 = tri.getVertexPosition(1) - p0;
    const Vec3 e2 = tri.getVertexPosition(2) - p0;

    const float du1 = uv1.u - uv0.u;
    const float dv1 = uv1.v - uv0.v;
    const float du2 = uv2.u - uv0.u;
    const float dv2 = uv2.v - uv0.v;

    const float det = du1 * dv2 - du2 * dv1;
    if (std::abs(det) < 1e-10f) return;

    const float inv = 1.0f / det;
    const Vec3 dPdU = (e1 * dv2 - e2 * dv1) * inv;
    const Vec3 dPdV = (e2 * du1 - e1 * du2) * inv;

    const float lu = dPdU.length();
    const float lv = dPdV.length();
    if (lu > 1e-12f) out_world_per_u = lu;
    if (lv > 1e-12f) out_world_per_v = lv;
}

// Per-axis pixel-radius correction so a brush that is circular in world
// space lands as an axis-aligned ellipse in texel space, while preserving
// the geometric mean radius (`kx*ky == 1`). A unit-aspect mesh + square
// texture yields (1,1); stretched UVs / non-uniform mesh scale produce the
// elongation needed to cancel the distortion.
//
// Anisotropy is clamped to 6:1 — past that point the painted ellipse
// becomes uncomfortably elongated and the better answer is for the user
// to fix their UVs.
void computeBrushAxisCorrection(float world_per_u, float world_per_v,
                                int width, int height,
                                float& out_kx, float& out_ky) {
    out_kx = 1.0f;
    out_ky = 1.0f;
    if (world_per_u <= 0.0f || world_per_v <= 0.0f || width <= 0 || height <= 0) return;

    const float tex_per_world_u = static_cast<float>(width)  / world_per_u;
    const float tex_per_world_v = static_cast<float>(height) / world_per_v;
    if (tex_per_world_u <= 0.0f || tex_per_world_v <= 0.0f) return;

    const float mean = std::sqrt(tex_per_world_u * tex_per_world_v);
    if (mean <= 1e-12f) return;

    constexpr float kMaxAnisotropy = 6.0f;
    constexpr float kMinAnisotropy = 1.0f / kMaxAnisotropy;
    float kx = tex_per_world_u / mean;
    float ky = tex_per_world_v / mean;
    if (kx > kMaxAnisotropy) kx = kMaxAnisotropy;
    if (ky > kMaxAnisotropy) ky = kMaxAnisotropy;
    if (kx < kMinAnisotropy) kx = kMinAnisotropy;
    if (ky < kMinAnisotropy) ky = kMinAnisotropy;
    out_kx = kx;
    out_ky = ky;
}

void blendPixelChannel(uint8_t& dst, uint8_t target, float alpha) {
    const float current = static_cast<float>(dst);
    const float blended = current + (static_cast<float>(target) - current) * std::clamp(alpha, 0.0f, 1.0f);
    dst = static_cast<uint8_t>(std::clamp(blended, 0.0f, 255.0f));
}

float hashNoise(float x, float y) {
    const float value = std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return value - std::floor(value);
}

float smoothNoise(float x, float y) {
    const float ix = std::floor(x);
    const float iy = std::floor(y);
    const float tx = x - ix;
    const float ty = y - iy;

    const float n00 = hashNoise(ix, iy);
    const float n10 = hashNoise(ix + 1.0f, iy);
    const float n01 = hashNoise(ix, iy + 1.0f);
    const float n11 = hashNoise(ix + 1.0f, iy + 1.0f);

    const float sx = tx * tx * (3.0f - 2.0f * tx);
    const float sy = ty * ty * (3.0f - 2.0f * ty);
    const float nx0 = n00 + (n10 - n00) * sx;
    const float nx1 = n01 + (n11 - n01) * sx;
    return nx0 + (nx1 - nx0) * sy;
}


float sampleImportedBrushAlpha(const std::shared_ptr<Texture>& texture, float nx, float ny, float scale, float rotation_degrees) {
    if (!texture || !texture->is_loaded()) {
        return 1.0f;
    }

    float scaled_x = nx * std::max(0.01f, scale);
    float scaled_y = ny * std::max(0.01f, scale);
    rotateBrushCoords(scaled_x, scaled_y, rotation_degrees);
    const float u = std::clamp(scaled_x * 0.5f + 0.5f, 0.0f, 1.0f);
    const float v = std::clamp(scaled_y * 0.5f + 0.5f, 0.0f, 1.0f);
    return std::clamp(texture->sampleIntensity(u, v), 0.0f, 1.0f);
}

float sampleBrushAlpha(BrushAlphaPreset preset, float nx, float ny, float scale, float rotation_degrees) {
    float sx = nx * std::max(0.01f, scale);
    float sy = ny * std::max(0.01f, scale);
    rotateBrushCoords(sx, sy, rotation_degrees);
    const float radial = std::clamp(1.0f - std::sqrt(nx * nx + ny * ny), 0.0f, 1.0f);

    switch (preset) {
        case BrushAlphaPreset::SoftRound:
            return radial;
        case BrushAlphaPreset::HardRound:
            return radial > 0.2f ? 1.0f : 0.0f;
        case BrushAlphaPreset::Noise: {
            const float noise = hashNoise((sx + 1.0f) * 17.0f, (sy + 1.0f) * 19.0f);
            return radial * (0.45f + noise * 0.55f);
        }
        case BrushAlphaPreset::Scratch: {
            const float streaks = std::abs(std::sin((sx * 18.0f) + (sy * 2.5f)));
            const float breakup = hashNoise((sx + 2.0f) * 11.0f, (sy + 2.0f) * 23.0f);
            const float scratch = std::pow(1.0f - streaks, 3.0f) * (0.35f + breakup * 0.65f);
            return radial * std::clamp(scratch * 2.2f, 0.0f, 1.0f);
        }
        case BrushAlphaPreset::Cloud: {
            const float n1 = smoothNoise((sx + 3.0f) * 3.0f, (sy + 5.0f) * 3.0f);
            const float n2 = smoothNoise((sx + 11.0f) * 6.0f, (sy + 7.0f) * 6.0f) * 0.5f;
            const float n3 = smoothNoise((sx + 19.0f) * 12.0f, (sy + 13.0f) * 12.0f) * 0.25f;
            const float cloud = std::clamp((n1 + n2 + n3) / 1.75f, 0.0f, 1.0f);
            return radial * (0.25f + cloud * 0.75f);
        }
    }

    return radial;
}

float sampleBrushMask(const BrushSettings& brush, float nx, float ny) {
    if (brush.use_imported_alpha && brush.alpha_texture && brush.alpha_texture->is_loaded()) {
        return sampleImportedBrushAlpha(brush.alpha_texture, nx, ny, brush.alpha_scale, brush.alpha_rotation_degrees);
    }

    return sampleBrushAlpha(brush.alpha_preset, nx, ny, brush.alpha_scale, brush.alpha_rotation_degrees);
}

// Alpha gate from the user-loaded paint texture itself. PNG stamps with a
// transparent background usually store RGB=(0,0,0) on alpha=0 pixels, so
// without this the brush deposits visible black rings around the actual
// stamp. Mirrors makeBrushTexturePixel's UV transform so the alpha lines up
// with the colour sample.
float sampleBrushPaintTextureAlpha(const BrushSettings& brush, float nx, float ny) {
    if (!brush.use_paint_texture || !brush.paint_texture || !brush.paint_texture->is_loaded()) {
        return 1.0f;
    }
    if (!brush.paint_texture->has_alpha) {
        return 1.0f;
    }

    float sx = nx * std::max(0.01f, brush.alpha_scale);
    float sy = ny * std::max(0.01f, brush.alpha_scale);
    rotateBrushCoords(sx, sy, brush.alpha_rotation_degrees);
    const float u = std::clamp(sx * 0.5f + 0.5f, 0.0f, 1.0f);
    const float v = std::clamp(sy * 0.5f + 0.5f, 0.0f, 1.0f);
    return std::clamp(brush.paint_texture->get_alpha_bilinear(u, v), 0.0f, 1.0f);
}

CompactVec4 sampleTexturePixelBilinear(const std::vector<CompactVec4>& pixels, int width, int height, float u, float v);

std::shared_ptr<Texture> cloneTextureForPaint(const std::shared_ptr<Texture>& source,
                                              const std::string& name,
                                              int resolution,
                                              TextureType type,
                                              PaintChannel channel,
                                              bool& out_seeded) {
    out_seeded = false;

    if (source && source->is_loaded() && !source->pixels.empty()) {
        const int fallback_resolution = resolution > 0 ? resolution : 1024;
        const int source_w = source->width > 0 ? source->width : fallback_resolution;
        const int source_h = source->height > 0 ? source->height : fallback_resolution;
        const int target_resolution = resolution > 0 ? resolution : std::max(source_w, source_h);
        auto texture = std::make_shared<Texture>(name, target_resolution, target_resolution, type);
        texture->width = target_resolution;
        texture->height = target_resolution;
        if (source_w == target_resolution && source_h == target_resolution) {
            texture->pixels = source->pixels;
        } else {
            texture->pixels.resize(static_cast<size_t>(target_resolution) * static_cast<size_t>(target_resolution));
            for (int y = 0; y < target_resolution; ++y) {
                for (int x = 0; x < target_resolution; ++x) {
                    const float u = target_resolution > 1
                        ? static_cast<float>(x) / static_cast<float>(target_resolution - 1)
                        : 0.0f;
                    const float v = target_resolution > 1
                        ? 1.0f - (static_cast<float>(y) / static_cast<float>(target_resolution - 1))
                        : 0.0f;
                    texture->pixels[static_cast<size_t>(y) * static_cast<size_t>(target_resolution) + static_cast<size_t>(x)] =
                        sampleTexturePixelBilinear(source->pixels, source_w, source_h, u, v);
                }
            }
        }
        texture->has_alpha = source->has_alpha;
        texture->is_gray_scale = source->is_gray_scale;
        texture->m_is_loaded = true;
        // Force opaque alpha on the paint canvas itself for every channel
        // *except* Opacity. Opacity stores the mask in alpha, so clobbering
        // it would silently delete the asset's opacity information on the
        // first paint binding. For all other channels alpha=0 leaks would
        // propagate into the bound albedo/opacity slot, producing the
        // black-square symptom we hardened seedFromTextureSet against.
        if (channel != PaintChannel::Opacity) {
            for (auto& p : texture->pixels) p.a = 255;
        }
        out_seeded = true;
        return texture;
    }

    return createBlankTexture(name, resolution > 0 ? resolution : 1024, type);
}

CompactVec4 sampleTexturePixelBilinear(const std::vector<CompactVec4>& pixels, int width, int height, float u, float v) {
    if (width <= 0 || height <= 0 || pixels.empty()) {
        return CompactVec4(255, 255, 255, 255);
    }

    const float x = std::clamp(u, 0.0f, 1.0f) * static_cast<float>(width - 1);
    const float y = (1.0f - std::clamp(v, 0.0f, 1.0f)) * static_cast<float>(height - 1);
    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, height - 1);
    const int x1 = std::clamp(x0 + 1, 0, width - 1);
    const int y1 = std::clamp(y0 + 1, 0, height - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const auto& c00 = pixels[static_cast<size_t>(y0) * static_cast<size_t>(width) + static_cast<size_t>(x0)];
    const auto& c10 = pixels[static_cast<size_t>(y0) * static_cast<size_t>(width) + static_cast<size_t>(x1)];
    const auto& c01 = pixels[static_cast<size_t>(y1) * static_cast<size_t>(width) + static_cast<size_t>(x0)];
    const auto& c11 = pixels[static_cast<size_t>(y1) * static_cast<size_t>(width) + static_cast<size_t>(x1)];
    const auto lerp_u8 = [&](uint8_t a, uint8_t b, float t) -> float {
        return static_cast<float>(a) + (static_cast<float>(b) - static_cast<float>(a)) * t;
    };

    const float r0 = lerp_u8(c00.r, c10.r, tx);
    const float g0 = lerp_u8(c00.g, c10.g, tx);
    const float b0 = lerp_u8(c00.b, c10.b, tx);
    const float a0 = lerp_u8(c00.a, c10.a, tx);
    const float r1 = lerp_u8(c01.r, c11.r, tx);
    const float g1 = lerp_u8(c01.g, c11.g, tx);
    const float b1 = lerp_u8(c01.b, c11.b, tx);
    const float a1 = lerp_u8(c01.a, c11.a, tx);

    return CompactVec4(
        static_cast<uint8_t>(std::clamp(r0 + (r1 - r0) * ty, 0.0f, 255.0f)),
        static_cast<uint8_t>(std::clamp(g0 + (g1 - g0) * ty, 0.0f, 255.0f)),
        static_cast<uint8_t>(std::clamp(b0 + (b1 - b0) * ty, 0.0f, 255.0f)),
        static_cast<uint8_t>(std::clamp(a0 + (a1 - a0) * ty, 0.0f, 255.0f)));
}

Vec3 decodeNormalPixel(const CompactVec4& pixel) {
    return Vec3(
        (static_cast<float>(pixel.r) / 255.0f) * 2.0f - 1.0f,
        (static_cast<float>(pixel.g) / 255.0f) * 2.0f - 1.0f,
        (static_cast<float>(pixel.b) / 255.0f) * 2.0f - 1.0f).normalize();
}

Vec3 buildNormalFromHeight(const std::shared_ptr<Texture>& height_texture, int px, int py, float strength) {
    const int width = height_texture->width;
    const int height = height_texture->height;
    const auto sample_height = [&](int sx, int sy) -> float {
        const int clamped_x = std::clamp(sx, 0, width - 1);
        const int clamped_y = std::clamp(sy, 0, height - 1);
        const CompactVec4& pixel =
            height_texture->pixels[static_cast<size_t>(clamped_y) * static_cast<size_t>(width) + static_cast<size_t>(clamped_x)];
        return static_cast<float>(pixel.r) / 255.0f;
    };
    const float h00 = sample_height(px - 1, py - 1);
    const float h10 = sample_height(px, py - 1);
    const float h20 = sample_height(px + 1, py - 1);
    const float h01 = sample_height(px - 1, py);
    const float h21 = sample_height(px + 1, py);
    const float h02 = sample_height(px - 1, py + 1);
    const float h12 = sample_height(px, py + 1);
    const float h22 = sample_height(px + 1, py + 1);

    // Sobel 3x3 horizontal/vertical
    const float dx_sobel = ((h20 + 2.0f * h21 + h22) - (h00 + 2.0f * h01 + h02)) * 0.25f;
    const float dy_sobel = ((h02 + 2.0f * h12 + h22) - (h00 + 2.0f * h10 + h20)) * 0.25f;

    // Apply strength and a normalization factor
    // 0.25f here is to prevent over-intensity at default settings
    const float dx = dx_sobel * strength * 0.25f;
    const float dy = dy_sobel * strength * 0.25f;

    return Vec3(-dx, -dy, 1.0f).normalize();
}

CompactVec4 encodeNormalPixel(const Vec3& normal) {
    const Vec3 n = normal.normalize();
    const auto to_byte = [](float value) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    return CompactVec4(
        to_byte(n.x * 0.5f + 0.5f),
        to_byte(n.y * 0.5f + 0.5f),
        to_byte(n.z * 0.5f + 0.5f),
        255);
}

Vec3 buildNormalFromHeightGeometryAware(const std::shared_ptr<Texture>& height_texture,
                                         const Triangle& tri, int uv_set,
                                         int px, int py, float strength) {
    const int width = height_texture->width;
    const int height = height_texture->height;

    auto get_h = [&](int x, int y) {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return static_cast<float>(height_texture->pixels[y * width + x].r) / 255.0f;
    };

    // 5-tap Cross Sampling for smoother gradients (especially at high res like 4096)
    // Helps reducing the 8-bit quantization "steps/layers" artifacts.
    const float hR1 = get_h(px + 1, py);
    const float hL1 = get_h(px - 1, py);
    const float hU1 = get_h(px, py - 1);
    const float hD1 = get_h(px, py + 1);

    const float hR2 = get_h(px + 2, py);
    const float hL2 = get_h(px - 2, py);
    const float hU2 = get_h(px, py - 2);
    const float hD2 = get_h(px, py + 2);

    // Weighted average for smoothing discrete steps
    const float gH = ((hR2 - hL2) * 0.5f + (hR1 - hL1)) * 0.5f;
    const float gV = ((hD2 - hU2) * 0.5f + (hD1 - hU1)) * 0.5f;

    // Use the triangle's UV→world Jacobian (constant across the triangle) to
    // convert "2 texel offset" into a world distance. Calling
    // getWorldPositionAtUV per neighbour barycentric-extrapolated outside the
    // UV bbox at triangle edges, blowing distH/distV up to random values and
    // turning the painted bump noisy. The Jacobian uses fixed edge vectors
    // so the world distance is stable for every texel inside or outside the
    // triangle UV rect alike.
    // Use the same UV set the paint texture is sampled through. Falling
    // back to set 0 here on a multi-set mesh would derive the Jacobian
    // from a different parametrisation than the one the brush dab landed
    // on, producing visibly tilted bumps along seams of secondary UV sets.
    Vec2 uv0, uv1, uv2;
    if (uv_set > 0 && static_cast<size_t>(uv_set) < tri.getUVSetCount()) {
        std::tie(uv0, uv1, uv2) = tri.getUVSetCoordinates(static_cast<size_t>(uv_set));
    } else {
        std::tie(uv0, uv1, uv2) = tri.getUVCoordinates();
    }
    const Vec3 p0 = tri.getVertexPosition(0);
    const Vec3 e1 = tri.getVertexPosition(1) - p0;
    const Vec3 e2 = tri.getVertexPosition(2) - p0;
    const float du1 = uv1.u - uv0.u;
    const float dv1 = uv1.v - uv0.v;
    const float du2 = uv2.u - uv0.u;
    const float dv2 = uv2.v - uv0.v;
    const float det = du1 * dv2 - du2 * dv1;

    // Default to "1 texel = 1 world unit" if the UV→world map is degenerate;
    // matches the pre-Jacobian fallback in buildNormalFromHeight.
    float distH = 1.0f;
    float distV = 1.0f;
    if (std::abs(det) >= 1e-10f) {
        const float inv = 1.0f / det;
        const Vec3 dPdU = (e1 * dv2 - e2 * dv1) * inv;
        const Vec3 dPdV = (e2 * du1 - e1 * du2) * inv;
        // 2-texel offset in pixel space → UV delta = 2 / (texDim - 1).
        // World distance for that span is |dPdU| * uv_step (note sign flip
        // for V because pixel y axis runs opposite to UV v axis — only the
        // magnitude matters here).
        const float du_step = (width  > 1) ? (2.0f / static_cast<float>(width  - 1)) : 0.0f;
        const float dv_step = (height > 1) ? (2.0f / static_cast<float>(height - 1)) : 0.0f;
        distH = dPdU.length() * du_step;
        distV = dPdV.length() * dv_step;
    }

    // Anti-divergence floor: clamp the world distance to a small fraction of
    // the triangle's average edge length. This stops scaled-down meshes (or
    // tiny UV islands) from making `gH / distH` explode.
    const float edge_avg = (e1.length() + e2.length() + (e1 - e2).length()) * (1.0f / 3.0f);
    const float min_dist = std::max(1e-5f, edge_avg * 1e-4f);
    if (distH < min_dist) distH = min_dist;
    if (distV < min_dist) distV = min_dist;

    // Sensitivity Dampening: Geometry aware gradients can be extreme on high-poly meshes.
    // 0.15 scaler so that 1.0 strength still feels usable on default meshes.
    const float sensitivity = 0.15f;
    float dx = (gH / distH) * strength * sensitivity;
    float dy = (gV / distV) * strength * sensitivity;

    return Vec3(-dx, -dy, 1.0f).normalize();
}

Vec3 combineNormalsPD(const Vec3& base_normal, const Vec3& detail_normal) {
    const Vec3 base = base_normal.normalize();
    const Vec3 detail = detail_normal.normalize();
    const float base_z = std::max(0.15f, base.z);
    const float detail_z = std::max(0.15f, detail.z);
    return Vec3(
        (base.x / base_z) + (detail.x / detail_z),
        (base.y / base_z) + (detail.y / detail_z),
        1.0f).normalize();
}

void resizeTexturePixels(Texture& texture, int resolution) {
    if (resolution <= 0 || (texture.width == resolution && texture.height == resolution)) {
        return;
    }

    const int source_width = texture.width;
    const int source_height = texture.height;
    const std::vector<CompactVec4> source_pixels = texture.pixels;
    texture.width = resolution;
    texture.height = resolution;
    texture.pixels.assign(static_cast<size_t>(resolution) * static_cast<size_t>(resolution), CompactVec4(255, 255, 255, 255));
    for (int y = 0; y < resolution; ++y) {
        for (int x = 0; x < resolution; ++x) {
            const float u = resolution > 1 ? static_cast<float>(x) / static_cast<float>(resolution - 1) : 0.0f;
            const float v = resolution > 1 ? 1.0f - (static_cast<float>(y) / static_cast<float>(resolution - 1)) : 0.0f;
            texture.pixels[static_cast<size_t>(y) * static_cast<size_t>(resolution) + static_cast<size_t>(x)] =
                sampleTexturePixelBilinear(source_pixels, source_width, source_height, u, v);
        }
    }
}

} // namespace

MeshPaintAdapter::MeshPaintAdapter(SceneData* scene, const std::shared_ptr<Triangle>& triangle)
    : scene_(scene), triangle_(triangle) {}

SurfaceType MeshPaintAdapter::getSurfaceType() const {
    return SurfaceType::Mesh;
}

PaintSurfaceTarget MeshPaintAdapter::getTarget() const {
    PaintSurfaceTarget target;
    target.type = SurfaceType::Mesh;
    target.material_id = getMaterialID();
    target.display_name = getNodeName();
    if (Material* material = MaterialManager::getInstance().getMaterial(target.material_id)) {
        if (auto* pbsdf = dynamic_cast<PrincipledBSDF*>(material)) {
            target.uv_set = std::max(0, pbsdf->selected_uv_set);
        }
    }
    return target;
}

PaintAdapterCapabilities MeshPaintAdapter::getCapabilities() const {
    PaintAdapterCapabilities caps;
    caps.supports_layers = true;
    caps.supports_texture_set = true;
    caps.supports_material_channels = true;
    return caps;
}

bool MeshPaintAdapter::isValid() const {
    return scene_ != nullptr && triangle_ != nullptr && triangle_->terrain_id == -1;
}

int MeshPaintAdapter::getLayerCount() const {
    const PaintLayerStack* stack = getLayerStack();
    return stack ? stack->layerCount() : 1;
}

std::string MeshPaintAdapter::getLayerName(int index) const {
    const PaintLayerStack* stack = getLayerStack();
    if (stack) {
        const PaintLayerData* ld = stack->layerAt(index);
        return ld ? ld->meta.name : std::string();
    }
    return index == 0 ? "Background" : std::string();
}

bool MeshPaintAdapter::beginStroke(const PaintStrokeContext& ctx) {
    (void)ctx;
    return isValid();
}

bool MeshPaintAdapter::applyDab(const Vec3& world_hit_point, const BrushSettings& brush, const PaintStrokeContext& ctx) {
    (void)world_hit_point;
    (void)brush;
    (void)ctx;
    return false;
}

void MeshPaintAdapter::endStroke() {}

uint16_t MeshPaintAdapter::getMaterialID() const {
    return triangle_ ? triangle_->getMaterialID() : MaterialManager::INVALID_MATERIAL_ID;
}

std::string MeshPaintAdapter::getNodeName() const {
    return triangle_ ? triangle_->getNodeName() : std::string();
}

std::string MeshPaintAdapter::getMaterialName() const {
    return MaterialManager::getInstance().getMaterialName(getMaterialID());
}

PaintTextureSet* MeshPaintAdapter::getTextureSet() const {
    if (!isValid()) {
        return nullptr;
    }

    const std::string key = getNodeName() + "#" + std::to_string(getMaterialID());
    auto it = scene_->mesh_paint_texture_sets.find(key);
    return (it != scene_->mesh_paint_texture_sets.end()) ? &it->second : nullptr;
}

PaintTextureSet& MeshPaintAdapter::ensureTextureSet(int resolution) {
    PaintTextureSet set;
    set.target_node_name = getNodeName();
    set.material_id = getMaterialID();
    set.resolution = resolution > 0 ? resolution : 1024;

    const std::string key = set.makeKey();
    auto [it, inserted] = scene_->mesh_paint_texture_sets.emplace(key, set);
    if (!inserted && resolution > 0) {
        it->second.resolution = resolution;
    }
    return it->second;
}

std::shared_ptr<Texture> MeshPaintAdapter::getChannelSourceTexture(PaintChannel channel) const {
    std::shared_ptr<Material> material = triangle_ ? triangle_->getMaterial() : nullptr;
    if (!material) {
        return nullptr;
    }

    switch (channel) {
        case PaintChannel::BaseColor: return material->albedoProperty.texture;
        case PaintChannel::Normal: return material->normalProperty.texture;
        case PaintChannel::Roughness: return material->roughnessProperty.texture;
        case PaintChannel::Metallic: return material->metallicProperty.texture;
        case PaintChannel::Emission: return material->emissionProperty.texture;
        case PaintChannel::Mask: return material->heightProperty.texture;
        case PaintChannel::Transmission: return material->transmissionProperty.texture;
        case PaintChannel::Opacity: return material->opacityProperty.texture;
    }
    return nullptr;
}

bool MeshPaintAdapter::assignTextureToChannel(PaintChannel channel) {
    if (!isValid()) {
        return false;
    }

    PaintTextureSet& set = ensureTextureSet(0);
    std::shared_ptr<Texture>& target_texture = set.getTextureRef(channel);
    if (!target_texture) {
        std::string baseName = getNodeName() + "_" + getMaterialName() + "_" + std::string(channelName(channel));
        for (char &c : baseName) {
            if (c == ' ' || c == ':' || c == '\\' || c == '/' || c == '%' || c == '\\n' || c == '\\r') c = '_';
        }
        const std::string texture_name = std::string("generated/") + baseName + ".png";
        bool seeded = false;
        std::shared_ptr<Texture> source_texture = getChannelSourceTexture(channel);
        target_texture = cloneTextureForPaint(source_texture, texture_name, set.resolution, toTextureType(channel), channel, seeded);
        if (!seeded && target_texture) {
            fillTextureDefault(*target_texture, channel);
        }
        set.setSourceInfo(channel, seeded, (source_texture && !source_texture->name.empty()) ? source_texture->name : "");
        set.setSourceTexture(channel, source_texture);
    }

    set.initialized = true;
    bindTextureSetToMaterial();
    return true;
}

bool MeshPaintAdapter::createTextureSet() {
    if (!isValid()) {
        return false;
    }

    ensureTextureSet(0);
    bool ok = true;
    ok = assignTextureToChannel(PaintChannel::BaseColor) && ok;
    ok = assignTextureToChannel(PaintChannel::Normal) && ok;
    ok = assignTextureToChannel(PaintChannel::Roughness) && ok;
    ok = assignTextureToChannel(PaintChannel::Metallic) && ok;
    ok = assignTextureToChannel(PaintChannel::Emission) && ok;
    ok = assignTextureToChannel(PaintChannel::Mask) && ok;
    ok = assignTextureToChannel(PaintChannel::Transmission) && ok;
    bindTextureSetToMaterial();
    return ok;
}

void MeshPaintAdapter::bindTextureSetToMaterial() {
    if (!triangle_) {
        return;
    }

    std::shared_ptr<Material> material = triangle_->getMaterial();
    PaintTextureSet* set = getTextureSet();
    if (!material || !set) {
        return;
    }

    if (set->base_color) {
        material->albedoProperty.texture = set->base_color;
    }
    if (set->normal) {
        material->normalProperty.texture = set->normal;
    }
    if (set->roughness) {
        material->roughnessProperty.texture = set->roughness;
    }
    if (set->metallic) {
        material->metallicProperty.texture = set->metallic;
    }
    if (set->emission) {
        material->emissionProperty.texture = set->emission;
    }
    if (set->mask) {
        material->heightProperty.texture = set->mask;
    }
    if (set->transmission) {
        material->transmissionProperty.texture = set->transmission;
    }
    if (set->opacity) {
        material->opacityProperty.texture = set->opacity;
    }
}

bool MeshPaintAdapter::restoreOriginalMaterialTextures() {
    if (!triangle_ || !scene_) {
        return false;
    }

    std::shared_ptr<Material> material = triangle_->getMaterial();
    PaintTextureSet* set = getTextureSet();
    if (!material || !set) {
        return false;
    }

    material->albedoProperty.texture = set->getSourceTexture(PaintChannel::BaseColor);
    material->normalProperty.texture = set->getSourceTexture(PaintChannel::Normal);
    material->roughnessProperty.texture = set->getSourceTexture(PaintChannel::Roughness);
    material->metallicProperty.texture = set->getSourceTexture(PaintChannel::Metallic);
    material->emissionProperty.texture = set->getSourceTexture(PaintChannel::Emission);
    material->heightProperty.texture = set->getSourceTexture(PaintChannel::Mask);
    material->transmissionProperty.texture = set->getSourceTexture(PaintChannel::Transmission);
    material->opacityProperty.texture = set->getSourceTexture(PaintChannel::Opacity);

    set->base_color.reset();
    set->normal.reset();
    set->roughness.reset();
    set->metallic.reset();
    set->emission.reset();
    set->mask.reset();
    set->transmission.reset();
    set->opacity.reset();
    set->initialized = false;
    set->seeded_from_existing.fill(false);
    set->source_texture_names.fill(std::string{});
    set->source_textures.fill(nullptr);
    return true;
}

bool MeshPaintAdapter::resizeTextureSet(int resolution) {
    PaintTextureSet* set = getTextureSet();
    if (!set || !set->initialized || resolution <= 0) {
        return false;
    }

    auto resize_channel = [&](PaintChannel channel) {
        std::shared_ptr<Texture> texture = set->getTexture(channel);
        if (!texture) {
            return;
        }
        const bool was_uploaded = texture->isUploaded();
        resizeTexturePixels(*texture, resolution);
        if (was_uploaded) {
            texture->cleanup_gpu();
        }
        texture->upload_to_gpu();
    };

    resize_channel(PaintChannel::BaseColor);
    resize_channel(PaintChannel::Normal);
    resize_channel(PaintChannel::Roughness);
    resize_channel(PaintChannel::Metallic);
    resize_channel(PaintChannel::Emission);
    resize_channel(PaintChannel::Mask);
    resize_channel(PaintChannel::Transmission);
    resize_channel(PaintChannel::Opacity);
    set->resolution = resolution;

    PaintLayerStack* stack = getLayerStack();
    if (stack && stack->layerCount() > 0) {
        stack->setResolution(resolution, resolution);
        stack->flattenInto(*set);
    }

    bindTextureSetToMaterial();
    return true;
}

bool MeshPaintAdapter::paintAtUV(PaintChannel channel, const Vec2& uv, const BrushSettings& brush, float dt) {
    PaintTextureSet* set = getTextureSet();
    if (!set) {
        return false;
    }

    std::shared_ptr<Texture> texture = set->getTexture(channel);
    if (!texture || texture->width <= 0 || texture->height <= 0 || texture->pixels.empty()) {
        return false;
    }

    const float clamped_u = std::clamp(uv.u, 0.0f, 1.0f);
    const float clamped_v = std::clamp(uv.v, 0.0f, 1.0f);
    const int width = texture->width;
    const int height = texture->height;
    const float center_x = clamped_u * static_cast<float>(width - 1);
    const float center_y = (1.0f - clamped_v) * static_cast<float>(height - 1);
    const float reference_res = 1024.0f;
    const float res_scale = static_cast<float>(width) / reference_res;
    const float radius_px = std::max(0.1f, brush.radius * res_scale);

    // Anisotropy correction so a circular brush in world space lands as an
    // axis-aligned ellipse in texel space (preserves world-area).
    float kx = 1.0f, ky = 1.0f;
    if (triangle_) {
        float wpu = 1.0f, wpv = 1.0f;
        const int uv_set = getTarget().uv_set;
        computeTriangleUvJacobianLengths(*triangle_, uv_set, wpu, wpv);
        computeBrushAxisCorrection(wpu, wpv, width, height, kx, ky);
    }
    const float radius_x = std::max(0.1f, radius_px * kx);
    const float radius_y = std::max(0.1f, radius_px * ky);

    const int min_x = std::max(0, static_cast<int>(std::floor(center_x - radius_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + radius_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(center_y - radius_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + radius_y)));
    const float strength = std::clamp(brush.strength * brush.flow * dt * 60.0f, 0.0f, 1.0f);
    const CompactVec4 erase_pixel = defaultChannelPixel(channel);
    std::vector<CompactVec4> source_pixels;
    if (brush.tool == BrushTool::Soften) {
        source_pixels = texture->pixels;
    }

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - center_x;
            const float dy = (static_cast<float>(py) + 0.5f) - center_y;
            const float nx = dx / radius_x;
            const float ny = dy / radius_y;
            const float dist_norm = std::sqrt(nx * nx + ny * ny);
            if (dist_norm > 1.0f) {
                continue;
            }

            const float alpha_mask = sampleBrushMask(brush, nx, ny);
            const float texture_alpha = sampleBrushPaintTextureAlpha(brush, nx, ny);
            const CompactVec4 brush_pixel = makeBrushTexturePixel(channel, brush, nx, ny);
            const float weight = computeBrushWeightNormalized(dist_norm, radius_px,
                                    std::clamp(brush.falloff, 0.0f, 1.0f)) * alpha_mask * texture_alpha * strength;
            if (weight <= 0.001f) {
                continue;
            }

            CompactVec4& pixel = texture->pixels[static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px)];
            if (brush.tool == BrushTool::Erase) {
                blendPixelChannel(pixel.r, erase_pixel.r, weight);
                blendPixelChannel(pixel.g, erase_pixel.g, weight);
                blendPixelChannel(pixel.b, erase_pixel.b, weight);
            } else if (brush.tool == BrushTool::Soften) {
                Vec3 sum(0.0f, 0.0f, 0.0f);
                int sample_count = 0;
                for (int oy = -1; oy <= 1; ++oy) {
                    const int sy = std::clamp(py + oy, 0, height - 1);
                    for (int ox = -1; ox <= 1; ++ox) {
                        const int sx = std::clamp(px + ox, 0, width - 1);
                        const CompactVec4& src = source_pixels[static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx)];
                        sum.x += static_cast<float>(src.r);
                        sum.y += static_cast<float>(src.g);
                        sum.z += static_cast<float>(src.b);
                        ++sample_count;
                    }
                }

                const uint8_t avg_r = static_cast<uint8_t>(sum.x / static_cast<float>(sample_count));
                const uint8_t avg_g = static_cast<uint8_t>(sum.y / static_cast<float>(sample_count));
                const uint8_t avg_b = static_cast<uint8_t>(sum.z / static_cast<float>(sample_count));
                blendPixelChannel(pixel.r, avg_r, weight);
                blendPixelChannel(pixel.g, avg_g, weight);
                blendPixelChannel(pixel.b, avg_b, weight);
            } else {
                blendPixelChannel(pixel.r, brush_pixel.r, weight);
                blendPixelChannel(pixel.g, brush_pixel.g, weight);
                blendPixelChannel(pixel.b, brush_pixel.b, weight);
                if (channel == PaintChannel::Normal) {
                    Vec3 n = Vec3(
                        (pixel.r / 255.0f) * 2.0f - 1.0f,
                        (pixel.g / 255.0f) * 2.0f - 1.0f,
                        (pixel.b / 255.0f) * 2.0f - 1.0f).normalize();
                    pixel.r = static_cast<uint8_t>(std::clamp((n.x * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                    pixel.g = static_cast<uint8_t>(std::clamp((n.y * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                    pixel.b = static_cast<uint8_t>(std::clamp((n.z * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                }
            }
            changed = true;
        }
    }

    if (!changed) {
        return false;
    }

    bindTextureSetToMaterial();
    if (texture->isUploaded()) {
        texture->upload_region_to_gpu(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
    } else {
        texture->upload_to_gpu();
    }
    return true;
}

bool MeshPaintAdapter::fillChannel(PaintChannel channel, const BrushSettings& brush, int layer_index) {
    PaintTextureSet* set = getTextureSet();
    if (!set) {
        return false;
    }

    std::shared_ptr<Texture> texture = set->getTexture(channel);
    if (!texture || texture->width <= 0 || texture->height <= 0 || texture->pixels.empty()) {
        return false;
    }

    const int width = texture->width;
    const int height = texture->height;

    // Build the filled pixel buffer once so we can apply it both to the
    // destination layer and fall back onto the raw texture.
    std::vector<CompactVec4> filled(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            const float u = width > 1 ? static_cast<float>(px) / static_cast<float>(width - 1) : 0.0f;
            const float v = height > 1 ? 1.0f - (static_cast<float>(py) / static_cast<float>(height - 1)) : 0.0f;
            const float nx = u * 2.0f - 1.0f;
            const float ny = v * 2.0f - 1.0f;
            CompactVec4 pixel = makeBrushTexturePixel(channel, brush, nx, ny);
            pixel.a = 255;  // Fill is fully opaque so composite fully replaces lower layers.
            filled[static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px)] = pixel;
        }
    }

    // Route the fill through the layer stack so that subsequent brush strokes
    // and Add-Layer operations (which trigger a full flatten via composite)
    // see the fill as canonical pixel data instead of pulling stale pixels
    // from a pre-fill base layer.
    PaintLayerStack& stack = ensureLayerStack();
    if (stack.width() != width || stack.height() != height) {
        stack.setResolution(width, height);
    }

    // Pick the target layer: caller-supplied index, or base layer as fallback.
    int target = (layer_index >= 0 && layer_index < stack.layerCount()) ? layer_index : 0;
    if (stack.layerCount() == 0) {
        stack.ensureBaseLayer();
        target = 0;
    }

    PaintLayerData* dst = stack.layerAt(target);
    if (dst) {
        auto& buf = dst->ensurePixels(channel);
        buf = filled;
    }

    // Re-composite the channel into the texture and upload. This keeps the
    // texture and the stack consistent so later dirty-region composites do
    // not reintroduce the pre-fill pixels.
    stack.flattenChannelInto(channel, *set);
    bindTextureSetToMaterial();
    return true;
}

void MeshPaintAdapter::releaseLayerStackFromScene() {
    if (!scene_ || !isValid()) return;
    const std::string key = getNodeName() + "#" + std::to_string(getMaterialID());
    scene_->mesh_paint_layer_stacks.erase(key);
}

bool MeshPaintAdapter::generateNormalFromHeight(float strength) {
    PaintTextureSet* set = getTextureSet();
    if (!set) {
        return false;
    }

    std::shared_ptr<Texture> height_texture = set->getTexture(PaintChannel::Mask);
    if (!height_texture || height_texture->width <= 0 || height_texture->height <= 0 || height_texture->pixels.empty()) {
        return false;
    }

    if (!assignTextureToChannel(PaintChannel::Normal)) {
        return false;
    }

    std::shared_ptr<Texture> normal_texture = set->getTexture(PaintChannel::Normal);
    if (!normal_texture || normal_texture->width <= 0 || normal_texture->height <= 0 || normal_texture->pixels.empty()) {
        return false;
    }

    if (normal_texture->width != height_texture->width || normal_texture->height != height_texture->height) {
        // Deallocate the old CUDA array so that the upload at the end
        // allocates a correctly-sized one.  Without this, updateGPU()
        // would memcpy new-resolution data into the old-resolution array.
        if (normal_texture->isUploaded()) {
            normal_texture->cleanup_gpu();
        }
        normal_texture->width = height_texture->width;
        normal_texture->height = height_texture->height;
        normal_texture->pixels.assign(
            static_cast<size_t>(normal_texture->width) * static_cast<size_t>(normal_texture->height),
            defaultChannelPixel(PaintChannel::Normal));
    }

    const int width = height_texture->width;
    const int height = height_texture->height;
    const float normal_strength = std::max(0.01f, strength);
    std::shared_ptr<Texture> source_normal = set->getSourceTexture(PaintChannel::Normal);
    const bool has_source_normal =
        source_normal &&
        source_normal->is_loaded() &&
        source_normal->width > 0 &&
        source_normal->height > 0 &&
        !source_normal->pixels.empty();
    const Vec3 flat_normal(0.0f, 0.0f, 1.0f);

    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            const Vec3 generated = buildNormalFromHeight(height_texture, px, py, normal_strength);
            Vec3 base_normal = flat_normal;
            if (has_source_normal) {
                const float u = width > 1 ? static_cast<float>(px) / static_cast<float>(width - 1) : 0.0f;
                const float v = height > 1 ? 1.0f - (static_cast<float>(py) / static_cast<float>(height - 1)) : 0.0f;
                base_normal = decodeNormalPixel(sampleTexturePixelBilinear(
                    source_normal->pixels,
                    source_normal->width,
                    source_normal->height,
                    u,
                    v));
            }
            const Vec3 n = combineNormalsPD(base_normal, generated);
            CompactVec4& dst = normal_texture->pixels[static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px)];
            dst = encodeNormalPixel(n);
        }
    }

    bindTextureSetToMaterial();
    if (normal_texture->isUploaded()) {
        normal_texture->updateGPU();
    } else {
        normal_texture->upload_to_gpu();
    }
    return true;
}

bool MeshPaintAdapter::bakeHeightIntoNormal(float strength, bool clear_height_mask) {
    PaintTextureSet* set = getTextureSet();
    if (!set) {
        return false;
    }

    std::shared_ptr<Texture> height_texture = set->getTexture(PaintChannel::Mask);
    if (!height_texture || height_texture->width <= 0 || height_texture->height <= 0 || height_texture->pixels.empty()) {
        return false;
    }

    if (!assignTextureToChannel(PaintChannel::Normal)) {
        return false;
    }

    std::shared_ptr<Texture> normal_texture = set->getTexture(PaintChannel::Normal);
    if (!normal_texture || normal_texture->width <= 0 || normal_texture->height <= 0 || normal_texture->pixels.empty()) {
        return false;
    }

    if (normal_texture->width != height_texture->width || normal_texture->height != height_texture->height) {
        // Deallocate the old CUDA array so that the upload at the end
        // allocates a correctly-sized one.  Without this, updateGPU()
        // would memcpy new-resolution data into the old-resolution array.
        if (normal_texture->isUploaded()) {
            normal_texture->cleanup_gpu();
        }
        normal_texture->width = height_texture->width;
        normal_texture->height = height_texture->height;
        normal_texture->pixels.assign(
            static_cast<size_t>(normal_texture->width) * static_cast<size_t>(normal_texture->height),
            defaultChannelPixel(PaintChannel::Normal));
    }

    const int width = height_texture->width;
    const int height = height_texture->height;
    const float normal_strength = std::max(0.01f, strength);
    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            const size_t index = static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px);
            
            // Generate baked surface bump from the height mask
            Vec3 baked_bump = buildNormalFromHeight(height_texture, px, py, normal_strength);
            
            // Decode existing texture normal
            const Vec3 base_normal = decodeNormalPixel(normal_texture->pixels[index]);
            
            // Blend them using Partial Derivative (PD) blending
            const Vec3 combined = combineNormalsPD(base_normal, baked_bump);
            
            normal_texture->pixels[index] = encodeNormalPixel(combined);
        }
    }

    if (clear_height_mask) {
        fillTextureDefault(*height_texture, PaintChannel::Mask);
    }

    bindTextureSetToMaterial();
    if (normal_texture->isUploaded()) {
        normal_texture->updateGPU();
    } else {
        normal_texture->upload_to_gpu();
    }
    if (clear_height_mask) {
        if (height_texture->isUploaded()) {
            height_texture->updateGPU();
        } else {
            height_texture->upload_to_gpu();
        }
    }
    return true;
}

bool MeshPaintAdapter::cloneAtUV(PaintChannel channel, const Vec2& dst_uv, const Vec2& src_uv, const BrushSettings& brush, float dt) {
    PaintTextureSet* set = getTextureSet();
    if (!set) {
        return false;
    }

    std::shared_ptr<Texture> texture = set->getTexture(channel);
    if (!texture || texture->width <= 0 || texture->height <= 0 || texture->pixels.empty()) {
        return false;
    }

    const int width = texture->width;
    const int height = texture->height;
    const float center_x = std::clamp(dst_uv.u, 0.0f, 1.0f) * static_cast<float>(width - 1);
    const float center_y = (1.0f - std::clamp(dst_uv.v, 0.0f, 1.0f)) * static_cast<float>(height - 1);
    const float src_center_x = std::clamp(src_uv.u, 0.0f, 1.0f) * static_cast<float>(width - 1);
    const float src_center_y = (1.0f - std::clamp(src_uv.v, 0.0f, 1.0f)) * static_cast<float>(height - 1);
    const float reference_res = 1024.0f;
    const float res_scale = static_cast<float>(width) / reference_res;
    const float radius_px = std::max(0.1f, brush.radius * res_scale);

    float kx = 1.0f, ky = 1.0f;
    if (triangle_) {
        float wpu = 1.0f, wpv = 1.0f;
        const int uv_set = getTarget().uv_set;
        computeTriangleUvJacobianLengths(*triangle_, uv_set, wpu, wpv);
        computeBrushAxisCorrection(wpu, wpv, width, height, kx, ky);
    }
    const float radius_x = std::max(0.1f, radius_px * kx);
    const float radius_y = std::max(0.1f, radius_px * ky);

    const int min_x = std::max(0, static_cast<int>(std::floor(center_x - radius_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + radius_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(center_y - radius_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + radius_y)));
    const float strength = std::clamp(brush.strength * brush.flow * dt * 60.0f, 0.0f, 1.0f);
    const std::vector<CompactVec4> source_pixels = texture->pixels;

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - center_x;
            const float dy = (static_cast<float>(py) + 0.5f) - center_y;
            const float nx = dx / radius_x;
            const float ny = dy / radius_y;
            const float dist_norm = std::sqrt(nx * nx + ny * ny);
            if (dist_norm > 1.0f) {
                continue;
            }

            const float alpha_mask = sampleBrushMask(brush, nx, ny);
            const float weight = computeBrushWeightNormalized(dist_norm, radius_px,
                                    std::clamp(brush.falloff, 0.0f, 1.0f)) * alpha_mask * strength;
            if (weight <= 0.001f) {
                continue;
            }

            const float src_px_f = std::clamp(src_center_x + dx, 0.0f, static_cast<float>(width - 1));
            const float src_py_f = std::clamp(src_center_y + dy, 0.0f, static_cast<float>(height - 1));
            // Bilinear sample for sub-pixel source offsets — matches
            // cloneLayerAtUV's behaviour. Nearest-neighbour truncation here
            // produced visible stair-step artefacts on the legacy (non-layer)
            // clone path when the user dragged the brush diagonally.
            const float src_u = (width  > 1) ? (src_px_f / static_cast<float>(width  - 1)) : 0.0f;
            const float src_v = (height > 1) ? (1.0f - src_py_f / static_cast<float>(height - 1)) : 0.0f;
            const CompactVec4 src = sampleTexturePixelBilinear(source_pixels, width, height, src_u, src_v);
            CompactVec4& dst = texture->pixels[static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px)];
            blendPixelChannel(dst.r, src.r, weight);
            blendPixelChannel(dst.g, src.g, weight);
            blendPixelChannel(dst.b, src.b, weight);
            changed = true;
        }
    }

    if (!changed) {
        return false;
    }

    bindTextureSetToMaterial();
    if (texture->isUploaded()) {
        texture->updateGPU();
    } else {
        texture->upload_to_gpu();
    }
    return true;
}

bool MeshPaintAdapter::updateNormalFromHeightArea(const Vec2& center_uv, float radius_px, float strength) {
    if (!isValid()) return false;
    PaintTextureSet* set = getTextureSet();
    if (!set) return false;

    std::shared_ptr<Texture> height_tex = set->getTexture(PaintChannel::Mask);
    std::shared_ptr<Texture> normal_tex = set->getTexture(PaintChannel::Normal);
    if (!height_tex || !normal_tex) return false;
    if (height_tex->pixels.empty() || normal_tex->pixels.empty()) return false;

    const int width = height_tex->width;
    const int height = height_tex->height;

    // Normal texture must match height texture dimensions for indexed access.
    if (normal_tex->width != width || normal_tex->height != height) return false;
    const size_t expected_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (normal_tex->pixels.size() < expected_pixels || height_tex->pixels.size() < expected_pixels) return false;

    const float reference_res = 1024.0f;
    const float res_scale = static_cast<float>(width) / reference_res;
    const float effective_radius_px = radius_px * res_scale;

    // Match the elliptical footprint used by paintLayerAtUV / paintAtUV so
    // the height-driven normal update covers the same texels the brush dab
    // touched. Without this, anisotropic UVs leave a circular Sobel band
    // surrounded by the elliptical paint area, producing visible seams.
    const int active_uv_set = getTarget().uv_set;
    float kx = 1.0f, ky = 1.0f;
    if (triangle_) {
        float wpu = 1.0f, wpv = 1.0f;
        computeTriangleUvJacobianLengths(*triangle_, active_uv_set, wpu, wpv);
        computeBrushAxisCorrection(wpu, wpv, width, height, kx, ky);
    }
    const float radius_x = std::max(0.1f, effective_radius_px * kx);
    const float radius_y = std::max(0.1f, effective_radius_px * ky);

    // Convert UV to pixel space
    const float cx = center_uv.u * static_cast<float>(width - 1);
    const float cy = (1.0f - center_uv.v) * static_cast<float>(height - 1);

    // Sobel needs 2 extra pixels for the 5-tap kernel; widen each axis
    // independently so the elliptical footprint stays covered on both axes.
    const float range_x = radius_x + 2.0f;
    const float range_y = radius_y + 2.0f;
    const int min_x = std::max(0, static_cast<int>(std::floor(cx - range_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(cx + range_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(cy - range_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(cy + range_y)));

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - cx;
            const float dy = (static_cast<float>(py) + 0.5f) - cy;
            const float ex = dx / radius_x;
            const float ey = dy / radius_y;
            if (ex * ex + ey * ey > 1.0f) {
                continue;
            }

            const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px);
            if (idx >= normal_tex->pixels.size()) continue;

            // Re-calculate normal for this specific pixel from current height map state
            // Use triangle info for geometry-aware scaling, sampling the same UV
            // set the paint surface is bound to.
            Vec3 baked_bump = triangle_ ? buildNormalFromHeightGeometryAware(height_tex, *triangle_, active_uv_set, px, py, strength)
                                        : buildNormalFromHeight(height_tex, px, py, strength);

            Vec3 base_normal = decodeNormalPixel(normal_tex->pixels[idx]);
            Vec3 combined = combineNormalsPD(base_normal, baked_bump);

            normal_tex->pixels[idx] = encodeNormalPixel(combined);
            changed = true;
        }
    }

    if (changed) {
        if (normal_tex->isUploaded()) normal_tex->upload_region_to_gpu(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
        else normal_tex->upload_to_gpu();
    }
    return changed;
}

// ======================== Layer Stack ========================

PaintLayerStack* MeshPaintAdapter::getLayerStack() {
    if (!isValid() || !scene_) return nullptr;
    const std::string key = getNodeName() + "#" + std::to_string(getMaterialID());
    auto it = scene_->mesh_paint_layer_stacks.find(key);
    return (it != scene_->mesh_paint_layer_stacks.end()) ? &it->second : nullptr;
}

const PaintLayerStack* MeshPaintAdapter::getLayerStack() const {
    if (!isValid() || !scene_) return nullptr;
    const std::string key = getNodeName() + "#" + std::to_string(getMaterialID());
    auto it = scene_->mesh_paint_layer_stacks.find(key);
    return (it != scene_->mesh_paint_layer_stacks.end()) ? &it->second : nullptr;
}

PaintLayerStack& MeshPaintAdapter::ensureLayerStack() {
    const std::string key = getNodeName() + "#" + std::to_string(getMaterialID());
    auto it = scene_->mesh_paint_layer_stacks.find(key);
    if (it != scene_->mesh_paint_layer_stacks.end()) {
        // Defensive: if the stack was created before the texture set was ready
        // (e.g. base layer has no pixels for any channel), try to seed now.
        PaintLayerStack& existing = it->second;
        if (existing.layerCount() == 1) {
            PaintLayerData* base = existing.layerAt(0);
            if (base) {
                bool has_any = false;
                for (int ch = 0; ch < static_cast<int>(kPaintChannelCount) && !has_any; ++ch)
                    has_any = base->hasPixels(static_cast<PaintChannel>(ch));
                if (!has_any) {
                    PaintTextureSet* tex_set = getTextureSet();
                    if (tex_set && tex_set->initialized) {
                        existing.seedFromTextureSet(*tex_set);
                    }
                }
            }
        }
        return existing;
    }

    // Create a new layer stack and seed it from the existing texture set if available.
    PaintLayerStack stack;
    PaintTextureSet* tex_set = getTextureSet();
    if (tex_set && tex_set->initialized) {
        stack.seedFromTextureSet(*tex_set);
    } else {
        // Use the texture set resolution or the default.
        const int res = tex_set ? tex_set->resolution : 1024;
        stack.setResolution(res, res);
        stack.ensureBaseLayer();
    }

    auto [inserted_it, _] = scene_->mesh_paint_layer_stacks.emplace(key, std::move(stack));
    return inserted_it->second;
}

PaintDirtyRect MeshPaintAdapter::paintLayerAtUV(int layer_index, PaintChannel channel,
                                                const Vec2& uv, const BrushSettings& brush, float dt,
                                                float aspect_u, float aspect_v)
{
    PaintDirtyRect dirty;
    PaintLayerStack* stack = getLayerStack();
    if (!stack) return dirty;

    if (PaintTextureSet* tex_set = getTextureSet()) {
        std::shared_ptr<Texture> texture = tex_set->getTexture(channel);
        if (texture && texture->width > 0 && texture->height > 0 &&
            (stack->width() != texture->width || stack->height() != texture->height)) {
            stack->setResolution(texture->width, texture->height);
        }
    }

    PaintLayerData* layer = stack->layerAt(layer_index);
    if (!layer || layer->meta.locked || !layer->meta.visible) return dirty;

    auto& pixels = layer->ensurePixels(channel);
    const int width  = layer->width;
    const int height = layer->height;
    if (width <= 0 || height <= 0 || pixels.empty()) return dirty;

    const float clamped_u = std::clamp(uv.u, 0.0f, 1.0f);
    const float clamped_v = std::clamp(uv.v, 0.0f, 1.0f);
    const float center_x = clamped_u * static_cast<float>(width - 1);
    const float center_y = (1.0f - clamped_v) * static_cast<float>(height - 1);
    const float reference_res = 1024.0f;
    const float res_scale = static_cast<float>(width) / reference_res;
    const float radius_px = std::max(0.1f, brush.radius * res_scale);

    // Per-axis pixel radius. Caller may pass overrides (e.g. for tools that
    // already compute their own footprint); otherwise we derive one from the
    // hit triangle's UV→world Jacobian so a circular brush in world space
    // lands as an ellipse in texel space, preserving area.
    float kx = aspect_u, ky = aspect_v;
    if (triangle_ && std::abs(aspect_u - 1.0f) < 1e-6f && std::abs(aspect_v - 1.0f) < 1e-6f) {
        float wpu = 1.0f, wpv = 1.0f;
        const int uv_set = getTarget().uv_set;
        computeTriangleUvJacobianLengths(*triangle_, uv_set, wpu, wpv);
        computeBrushAxisCorrection(wpu, wpv, width, height, kx, ky);
    }
    const float radius_x = std::max(0.1f, radius_px * kx);
    const float radius_y = std::max(0.1f, radius_px * ky);

    const int min_x = std::max(0, static_cast<int>(std::floor(center_x - radius_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + radius_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(center_y - radius_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + radius_y)));
    const float strength = std::clamp(brush.strength * brush.flow * dt * 60.0f, 0.0f, 1.0f);

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - center_x;
            const float dy = (static_cast<float>(py) + 0.5f) - center_y;
            const float nx = dx / radius_x;
            const float ny = dy / radius_y;
            const float dist_norm = std::sqrt(nx * nx + ny * ny);
            if (dist_norm > 1.0f) continue;

            const float alpha_mask = sampleBrushMask(brush, nx, ny);
            const float texture_alpha = sampleBrushPaintTextureAlpha(brush, nx, ny);
            const CompactVec4 brush_pixel = makeBrushTexturePixel(channel, brush, nx, ny);
            const float weight = computeBrushWeightNormalized(dist_norm, radius_px,
                                    std::clamp(brush.falloff, 0.0f, 1.0f)) * alpha_mask * texture_alpha * strength;
            if (weight <= 0.001f) continue;

            CompactVec4& pixel = pixels[static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px)];

            if (brush.tool == BrushTool::Erase) {
                // Erase on a layer = reduce alpha towards 0.
                const float new_alpha = static_cast<float>(pixel.a) * (1.0f - weight);
                pixel.a = static_cast<uint8_t>(std::clamp(new_alpha, 0.0f, 255.0f));
            } else if (brush.tool == BrushTool::Soften) {
                // 3x3 box blur from layer's own pixels.
                Vec3 sum(0.0f, 0.0f, 0.0f);
                float asum = 0.0f;
                int sample_count = 0;
                for (int oy = -1; oy <= 1; ++oy) {
                    const int sy = std::clamp(py + oy, 0, height - 1);
                    for (int ox = -1; ox <= 1; ++ox) {
                        const int sx = std::clamp(px + ox, 0, width - 1);
                        const CompactVec4& src = pixels[static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx)];
                        sum.x += static_cast<float>(src.r);
                        sum.y += static_cast<float>(src.g);
                        sum.z += static_cast<float>(src.b);
                        asum  += static_cast<float>(src.a);
                        ++sample_count;
                    }
                }
                const float inv = 1.0f / static_cast<float>(sample_count);
                blendPixelChannel(pixel.r, static_cast<uint8_t>(sum.x * inv), weight);
                blendPixelChannel(pixel.g, static_cast<uint8_t>(sum.y * inv), weight);
                blendPixelChannel(pixel.b, static_cast<uint8_t>(sum.z * inv), weight);
                pixel.a = static_cast<uint8_t>(std::clamp(
                    static_cast<float>(pixel.a) + (asum * inv - static_cast<float>(pixel.a)) * weight,
                    0.0f, 255.0f));
            } else {
                // Porter-Duff source-over with straight (un-premultiplied) alpha.
                // Blending RGB with `weight` directly would store premultiplied
                // colour while the compositor treats layer RGB as straight,
                // producing a `weight^2` darkening at the falloff edges (the
                // black-ring artefact on fresh transparent layers).
                const float src_a = std::clamp(weight, 0.0f, 1.0f);
                const float dst_a = static_cast<float>(pixel.a) / 255.0f;
                const float out_a = src_a + dst_a * (1.0f - src_a);
                if (out_a > 1e-6f) {
                    const float inv_out_a = 1.0f / out_a;
                    const float dr = static_cast<float>(pixel.r);
                    const float dg = static_cast<float>(pixel.g);
                    const float db = static_cast<float>(pixel.b);
                    const float sr = static_cast<float>(brush_pixel.r);
                    const float sg = static_cast<float>(brush_pixel.g);
                    const float sb = static_cast<float>(brush_pixel.b);
                    const float keep = dst_a * (1.0f - src_a);
                    pixel.r = static_cast<uint8_t>(std::clamp((sr * src_a + dr * keep) * inv_out_a, 0.0f, 255.0f));
                    pixel.g = static_cast<uint8_t>(std::clamp((sg * src_a + dg * keep) * inv_out_a, 0.0f, 255.0f));
                    pixel.b = static_cast<uint8_t>(std::clamp((sb * src_a + db * keep) * inv_out_a, 0.0f, 255.0f));
                }
                pixel.a = static_cast<uint8_t>(std::clamp(out_a * 255.0f, 0.0f, 255.0f));
            }
            changed = true;
        }
    }

    if (changed) dirty.expand(min_x, min_y, max_x, max_y);
    return dirty;
}

PaintDirtyRect MeshPaintAdapter::cloneLayerAtUV(int layer_index, PaintChannel channel,
                                                const Vec2& dst_uv, const Vec2& src_uv,
                                                const BrushSettings& brush, float dt)
{
    PaintDirtyRect dirty;
    PaintLayerStack* stack = getLayerStack();
    if (!stack) return dirty;

    if (PaintTextureSet* tex_set = getTextureSet()) {
        std::shared_ptr<Texture> texture = tex_set->getTexture(channel);
        if (texture && texture->width > 0 && texture->height > 0 &&
            (stack->width() != texture->width || stack->height() != texture->height)) {
            stack->setResolution(texture->width, texture->height);
        }
    }

    PaintLayerData* layer = stack->layerAt(layer_index);
    if (!layer || layer->meta.locked || !layer->meta.visible) return dirty;

    auto& pixels = layer->ensurePixels(channel);
    const int width  = layer->width;
    const int height = layer->height;
    if (width <= 0 || height <= 0 || pixels.empty()) return dirty;

    const float center_x = std::clamp(dst_uv.u, 0.0f, 1.0f) * static_cast<float>(width - 1);
    const float center_y = (1.0f - std::clamp(dst_uv.v, 0.0f, 1.0f)) * static_cast<float>(height - 1);
    const float src_center_x = std::clamp(src_uv.u, 0.0f, 1.0f) * static_cast<float>(width - 1);
    const float src_center_y = (1.0f - std::clamp(src_uv.v, 0.0f, 1.0f)) * static_cast<float>(height - 1);
    const float reference_res = 1024.0f;
    const float res_scale = static_cast<float>(width) / reference_res;
    const float radius_px = std::max(0.1f, brush.radius * res_scale);

    float kx = 1.0f, ky = 1.0f;
    if (triangle_) {
        float wpu = 1.0f, wpv = 1.0f;
        const int uv_set = getTarget().uv_set;
        computeTriangleUvJacobianLengths(*triangle_, uv_set, wpu, wpv);
        computeBrushAxisCorrection(wpu, wpv, width, height, kx, ky);
    }
    const float radius_x = std::max(0.1f, radius_px * kx);
    const float radius_y = std::max(0.1f, radius_px * ky);

    const int min_x = std::max(0, static_cast<int>(std::floor(center_x - radius_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + radius_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(center_y - radius_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + radius_y)));
    const float strength = std::clamp(brush.strength * brush.flow * dt * 60.0f, 0.0f, 1.0f);

    // Source data: read from the layer's own pixels (clone within layer).
    const std::vector<CompactVec4> source_pixels = pixels;

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - center_x;
            const float dy = (static_cast<float>(py) + 0.5f) - center_y;
            const float nx = dx / radius_x;
            const float ny = dy / radius_y;
            const float dist_norm = std::sqrt(nx * nx + ny * ny);
            if (dist_norm > 1.0f) continue;

            const float alpha_mask = sampleBrushMask(brush, nx, ny);
            const float weight = computeBrushWeightNormalized(dist_norm, radius_px,
                                    std::clamp(brush.falloff, 0.0f, 1.0f)) * alpha_mask * strength;
            if (weight <= 0.001f) continue;

            const float src_px_f = std::clamp(src_center_x + dx, 0.0f, static_cast<float>(width - 1));
            const float src_py_f = std::clamp(src_center_y + dy, 0.0f, static_cast<float>(height - 1));
            const CompactVec4 sampled = sampleTexturePixelBilinear(
                source_pixels, width, height,
                src_px_f / static_cast<float>(width - 1),
                1.0f - src_py_f / static_cast<float>(height - 1));

            CompactVec4& pixel = pixels[static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px)];
            // Same Porter-Duff straight-alpha source-over as paintLayerAtUV;
            // see comment there for the black-ring rationale.
            const float src_a = std::clamp(weight, 0.0f, 1.0f);
            const float dst_a = static_cast<float>(pixel.a) / 255.0f;
            const float out_a = src_a + dst_a * (1.0f - src_a);
            if (out_a > 1e-6f) {
                const float inv_out_a = 1.0f / out_a;
                const float dr = static_cast<float>(pixel.r);
                const float dg = static_cast<float>(pixel.g);
                const float db = static_cast<float>(pixel.b);
                const float sr = static_cast<float>(sampled.r);
                const float sg = static_cast<float>(sampled.g);
                const float sb = static_cast<float>(sampled.b);
                const float keep = dst_a * (1.0f - src_a);
                pixel.r = static_cast<uint8_t>(std::clamp((sr * src_a + dr * keep) * inv_out_a, 0.0f, 255.0f));
                pixel.g = static_cast<uint8_t>(std::clamp((sg * src_a + dg * keep) * inv_out_a, 0.0f, 255.0f));
                pixel.b = static_cast<uint8_t>(std::clamp((sb * src_a + db * keep) * inv_out_a, 0.0f, 255.0f));
            }
            pixel.a = static_cast<uint8_t>(std::clamp(out_a * 255.0f, 0.0f, 255.0f));
            changed = true;
        }
    }

    if (changed) dirty.expand(min_x, min_y, max_x, max_y);
    return dirty;
}

void MeshPaintAdapter::compositeAndUpload() {
    PaintLayerStack* stack = getLayerStack();
    PaintTextureSet* tex_set = getTextureSet();
    if (!stack || !tex_set || !tex_set->initialized) return;

    stack->flattenInto(*tex_set);
    bindTextureSetToMaterial();
}

void MeshPaintAdapter::compositeAndUploadChannels(const PaintChannel* channels, int count) {
    PaintLayerStack* stack = getLayerStack();
    PaintTextureSet* tex_set = getTextureSet();
    if (!stack || !tex_set || !tex_set->initialized) return;

    for (int i = 0; i < count; ++i) {
        stack->flattenChannelInto(channels[i], *tex_set);
    }
    bindTextureSetToMaterial();
}

void MeshPaintAdapter::compositeAndUploadRegion(const PaintChannel* channels, int count,
                                                const PaintDirtyRect& dirty) {
    if (dirty.empty()) return;

    PaintLayerStack* stack = getLayerStack();
    PaintTextureSet* tex_set = getTextureSet();
    if (!stack || !tex_set || !tex_set->initialized) return;

    for (int i = 0; i < count; ++i) {
        stack->flattenChannelRegionInto(channels[i], *tex_set, dirty);
    }
    bindTextureSetToMaterial();
}

} // namespace Paint
