#include "Paint/MeshPaintAdapter.h"

#include "scene_data.h"
#include "Triangle.h"
#include "Material.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "SurfaceFlowField.h"
#include "SimulationWorld.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <tuple>

namespace Paint {

namespace {

constexpr std::array<PaintChannel, 3> kWetAuxScalarChannels = {
    PaintChannel::Roughness,
    PaintChannel::Metallic,
    PaintChannel::Transmission
};

struct WetAuxChannelBinding {
    PaintChannel channel = PaintChannel::Roughness;
    std::vector<CompactVec4>* pixels = nullptr;
    std::shared_ptr<Texture> texture;
    std::vector<CompactVec4> source_pixels;
    bool changed = false;
};

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
        case PaintChannel::Transmission:
            return CompactVec4(0, 0, 0, 255);
        case PaintChannel::Mask:
            // Height masks use mid-gray as neutral displacement.
            return CompactVec4(128, 128, 128, 255);
        case PaintChannel::Opacity:
            // Default = fully visible: a freshly-created opacity canvas should
            // not silently hide the surface. Erase reverts texels to this.
            return CompactVec4(255, 255, 255, 255);
    }
    return CompactVec4(255, 255, 255, 255);
}

Vec3 triangleCentroid(const Triangle& tri) {
    return (tri.getVertexPosition(0) + tri.getVertexPosition(1) + tri.getVertexPosition(2)) / 3.0f;
}

bool isFiniteVec3(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
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

const BrushChannelInput* getBrushChannelInput(const BrushSettings& brush, PaintChannel channel) {
    const size_t index = static_cast<size_t>(channel);
    if (index >= brush.channel_inputs.size()) {
        return nullptr;
    }
    const BrushChannelInput& input = brush.channel_inputs[index];
    return input.enabled ? &input : nullptr;
}

Vec3 getBrushChannelColor(PaintChannel channel, const BrushSettings& brush) {
    if (const BrushChannelInput* input = getBrushChannelInput(brush, channel)) {
        return input->color;
    }
    return brush.color;
}

std::shared_ptr<Texture> getBrushChannelTexture(PaintChannel channel, const BrushSettings& brush) {
    if (const BrushChannelInput* input = getBrushChannelInput(brush, channel)) {
        if (input->use_paint_texture && input->paint_texture && input->paint_texture->is_loaded()) {
            return input->paint_texture;
        }
        return nullptr;
    }
    if (brush.use_paint_texture && brush.paint_texture && brush.paint_texture->is_loaded()) {
        return brush.paint_texture;
    }
    return nullptr;
}

float getBrushChannelTintStrength(PaintChannel channel, const BrushSettings& brush) {
    if (const BrushChannelInput* input = getBrushChannelInput(brush, channel)) {
        return input->tint_strength;
    }
    return brush.paint_texture_tint_strength;
}

PaintTextureTintMode getBrushChannelTintMode(PaintChannel channel, const BrushSettings& brush) {
    if (const BrushChannelInput* input = getBrushChannelInput(brush, channel)) {
        return input->tint_mode;
    }
    return brush.paint_texture_tint_mode;
}

float srgbToLinear01(float value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    return clamped <= 0.04045f
        ? (clamped / 12.92f)
        : std::pow((clamped + 0.055f) / 1.055f, 2.4f);
}

float linearToSrgb01(float value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    return clamped <= 0.0031308f
        ? (clamped * 12.92f)
        : (1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f);
}

float weatherSpawnHash(float x, float y) {
    const float v = std::sinf(x * 127.1f + y * 311.7f) * 43758.5453f;
    return v - std::floor(v);
}

uint8_t linearToSrgbByte01(float value) {
    return static_cast<uint8_t>(std::clamp(linearToSrgb01(value) * 255.0f + 0.5f, 0.0f, 255.0f));
}

float srgbByteToLinear01(uint8_t value) {
    return srgbToLinear01(static_cast<float>(value) / 255.0f);
}

void blendBaseColorChannelLinear(uint8_t& dst, uint8_t target, float alpha) {
    const float t = std::clamp(alpha, 0.0f, 1.0f);
    const float current_linear = srgbByteToLinear01(dst);
    const float target_linear = srgbByteToLinear01(target);
    dst = linearToSrgbByte01(current_linear + (target_linear - current_linear) * t);
}
void blendHeightMaskPixelChannels(CompactVec4& dst, const CompactVec4& src, float opacity) {
    const float sa = std::clamp(opacity, 0.0f, 1.0f);
    const float neutral = 128.0f;
    const float dst_delta = static_cast<float>(dst.r) - neutral;
    const float src_delta = static_cast<float>(src.r) - neutral;
    const float out = neutral + dst_delta + src_delta * sa;
    const uint8_t value = static_cast<uint8_t>(std::clamp(out, 0.0f, 255.0f) + 0.5f);
    dst.r = value;
    dst.g = value;
    dst.b = value;
    dst.a = 255;
}
}

CompactVec4 makeBrushPixel(PaintChannel channel, const BrushSettings& brush) {
    const Vec3 channel_color = getBrushChannelColor(channel, brush);
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
                linearToSrgbByte(channel_color.x),
                linearToSrgbByte(channel_color.y),
                linearToSrgbByte(channel_color.z),
                255);
        case PaintChannel::Emission:
            return CompactVec4(toByte(channel_color.x), toByte(channel_color.y), toByte(channel_color.z), 255);
        case PaintChannel::Normal: {
            // Tangent-space normals must point out of the surface (n.z >= 0).
            // brush.color is a free RGB picker, so a user-chosen red (1,0,0)
            // would otherwise decode to (1,-1,-1) → n.z negative → BSDFs
            // see a normal pointing into the surface, producing flipped
            // shading and NaN reflection terms. Clamp z to non-negative
            // before re-normalising.
            Vec3 n(
                channel_color.x * 2.0f - 1.0f,
                channel_color.y * 2.0f - 1.0f,
                channel_color.z * 2.0f - 1.0f);
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
            const float grayscale = std::clamp((channel_color.x + channel_color.y + channel_color.z) / 3.0f, 0.0f, 1.0f);
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
            const float grayscale = std::clamp((channel_color.x + channel_color.y + channel_color.z) / 3.0f, 0.0f, 1.0f);
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

float brushShapeAspectScale(const BrushSettings& brush) {
    return std::sqrt(std::clamp(brush.shape_aspect, 0.1f, 8.0f));
}

float brushShapeDistance(BrushShape shape, float x, float y, float roundness) {
    const float ax = std::abs(x);
    const float ay = std::abs(y);
    switch (shape) {
        case BrushShape::Circle:
            return std::sqrt(x * x + y * y);
        case BrushShape::Rectangle: {
            const float p = 8.0f + std::clamp(roundness, 0.0f, 1.0f) * 16.0f;
            return std::pow(std::pow(ax, p) + std::pow(ay, p), 1.0f / p);
        }
        case BrushShape::Capsule: {
            const float half_segment = 0.55f;
            const float qx = std::max(ax - half_segment, 0.0f);
            return std::sqrt(qx * qx + ay * ay);
        }
        case BrushShape::Flat: {
            const float p = 10.0f + std::clamp(roundness, 0.0f, 1.0f) * 18.0f;
            return std::pow(std::pow(ax, p) + std::pow(ay, p), 1.0f / p);
        }
    }
    return std::sqrt(x * x + y * y);
}

struct BrushFootprintSample {
    float x = 0.0f;
    float y = 0.0f;
    float dist_norm = 0.0f;
};

BrushFootprintSample sampleBrushFootprint(const BrushSettings& brush, float dx, float dy, float radius_x, float radius_y) {
    float x = dx / std::max(0.001f, radius_x);
    float y = dy / std::max(0.001f, radius_y);
    rotateBrushCoords(x, y, -brush.alpha_rotation_degrees);

    const float aspect_scale = brushShapeAspectScale(brush);
    x /= aspect_scale;
    y *= aspect_scale;

    BrushFootprintSample sample;
    sample.x = x;
    sample.y = y;
    sample.dist_norm = brushShapeDistance(brush.shape, x, y, brush.shape_roundness);
    return sample;
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
    std::shared_ptr<Texture> paint_texture = getBrushChannelTexture(channel, brush);
    if (!paint_texture) {
        return makeBrushPixel(channel, brush);
    }
    const Vec3 channel_color = getBrushChannelColor(channel, brush);
    const float tint_strength = getBrushChannelTintStrength(channel, brush);
    const PaintTextureTintMode tint_mode = getBrushChannelTintMode(channel, brush);

    float sx = nx * std::max(0.01f, brush.alpha_scale);
    float sy = ny * std::max(0.01f, brush.alpha_scale);
    if (brush.flip_alpha_x) sx = -sx;
    if (brush.flip_alpha_y) sy = -sy;
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
            Vec3 sampled = paint_texture->get_color_bilinear(u, v);
            sampled = applyTintToSample(sampled, channel_color, tint_strength, tint_mode);
            return CompactVec4(
                linearToSrgbByte(sampled.x),
                linearToSrgbByte(sampled.y),
                linearToSrgbByte(sampled.z),
                255);
        }
        case PaintChannel::Emission: {
            Vec3 sampled = paint_texture->get_color_bilinear(u, v);
            sampled = applyTintToSample(sampled, channel_color, tint_strength, tint_mode);
            return CompactVec4(toByte(sampled.x), toByte(sampled.y), toByte(sampled.z), 255);
        }
        case PaintChannel::Roughness:
        case PaintChannel::Metallic:
        case PaintChannel::Mask:
        case PaintChannel::Transmission: {
            const float texture_value = paint_texture->sampleIntensity(u, v);
            const float tint_value = std::clamp((channel_color.x + channel_color.y + channel_color.z) / 3.0f, 0.0f, 1.0f);
            const float grayscale = texture_value + (texture_value * tint_value - texture_value) *
                std::clamp(tint_strength, 0.0f, 1.0f);
            const uint8_t value = toByte(grayscale);
            return CompactVec4(value, value, value, 255);
        }
        case PaintChannel::Opacity: {
            // Mirror the same all-channel write that makeBrushPixel does for
            // Opacity: shader reads .a or .r depending on material flags;
            // putting the value in both leaves the mask working in either mode.
            const float texture_value = paint_texture->sampleIntensity(u, v);
            const float tint_value = std::clamp((channel_color.x + channel_color.y + channel_color.z) / 3.0f, 0.0f, 1.0f);
            const float grayscale = texture_value + (texture_value * tint_value - texture_value) *
                std::clamp(tint_strength, 0.0f, 1.0f);
            const uint8_t value = toByte(grayscale);
            return CompactVec4(value, value, value, value);
        }
        case PaintChannel::Normal: {
            Vec3 sampled = paint_texture->get_color_bilinear(u, v);
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

std::array<Vec2, 3> getTriangleUvArray(const Triangle& tri, int uv_set) {
    Vec2 uv0, uv1, uv2;
    if (uv_set > 0 && static_cast<size_t>(uv_set) < tri.getUVSetCount()) {
        std::tie(uv0, uv1, uv2) = tri.getUVSetCoordinates(static_cast<size_t>(uv_set));
    } else {
        std::tie(uv0, uv1, uv2) = tri.getUVCoordinates();
    }
    return { uv0, uv1, uv2 };
}

bool computeUvBarycentric(const Vec2& p, const std::array<Vec2, 3>& tri_uvs,
                          float& w0, float& w1, float& w2) {
    const float x0 = tri_uvs[0].u;
    const float y0 = tri_uvs[0].v;
    const float x1 = tri_uvs[1].u;
    const float y1 = tri_uvs[1].v;
    const float x2 = tri_uvs[2].u;
    const float y2 = tri_uvs[2].v;
    const float det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (std::abs(det) < 1e-10f) {
        return false;
    }

    w0 = ((y1 - y2) * (p.u - x2) + (x2 - x1) * (p.v - y2)) / det;
    w1 = ((y2 - y0) * (p.u - x2) + (x0 - x2) * (p.v - y2)) / det;
    w2 = 1.0f - w0 - w1;
    return std::isfinite(w0) && std::isfinite(w1) && std::isfinite(w2);
}

Vec2 uvFromBarycentric(const std::array<Vec2, 3>& tri_uvs, float w0, float w1, float w2) {
    return Vec2(
        tri_uvs[0].u * w0 + tri_uvs[1].u * w1 + tri_uvs[2].u * w2,
        tri_uvs[0].v * w0 + tri_uvs[1].v * w1 + tri_uvs[2].v * w2);
}

void blendPixelChannel(uint8_t& dst, uint8_t target, float alpha) {
    const float current = static_cast<float>(dst);
    const float blended = current + (static_cast<float>(target) - current) * std::clamp(alpha, 0.0f, 1.0f);
    dst = static_cast<uint8_t>(std::clamp(blended, 0.0f, 255.0f));
}

void blendScalarPixel(CompactVec4& dst, uint8_t target, float alpha) {
    blendPixelChannel(dst.r, target, alpha);
    dst.g = dst.r;
    dst.b = dst.r;
    dst.a = 255;
}

void blendScalarPixel(CompactVec4& dst, const CompactVec4& src, float alpha) {
    blendScalarPixel(dst, src.r, alpha);
}

bool isScalarMaterialChannel(PaintChannel channel) {
    return channel == PaintChannel::Roughness ||
           channel == PaintChannel::Metallic ||
           channel == PaintChannel::Transmission;
}

float applyScalarLayerBlendMode(float dst, float src, LayerBlendMode mode) {
    switch (mode) {
        default:
        case LayerBlendMode::Normal:
            return src;
        case LayerBlendMode::Add:
            return std::min(dst + src, 1.0f);
        case LayerBlendMode::Multiply:
            return dst * src;
        case LayerBlendMode::Screen:
            return 1.0f - ((1.0f - dst) * (1.0f - src));
        case LayerBlendMode::Overlay:
            return dst < 0.5f
                ? (2.0f * dst * src)
                : (1.0f - 2.0f * (1.0f - dst) * (1.0f - src));
    }
}

float compositeScalarLayerValue(float dst, const CompactVec4& src, float opacity, LayerBlendMode mode) {
    const float sa = (static_cast<float>(src.a) / 255.0f) * std::clamp(opacity, 0.0f, 1.0f);
    if (sa <= 0.0f) {
        return dst;
    }
    const float src_value = static_cast<float>(src.r) / 255.0f;
    const float blended = applyScalarLayerBlendMode(dst, src_value, mode);
    return std::clamp(dst + (blended - dst) * sa, 0.0f, 1.0f);
}

float compositeScalarBelowLayer(const PaintLayerStack* stack, int layer_index, PaintChannel channel, size_t pixel_index) {
    if (!stack) {
        return 0.0f;
    }

    float result = static_cast<float>(defaultChannelPixel(channel).r) / 255.0f;
    for (int index = 0; index < layer_index; ++index) {
        const PaintLayerData* below = stack->layerAt(index);
        if (!below || !below->meta.visible || !below->hasPixels(channel)) {
            continue;
        }
        const auto& src = below->getPixels(channel);
        if (pixel_index >= src.size()) {
            continue;
        }
        result = compositeScalarLayerValue(result, src[pixel_index], below->meta.opacity, below->meta.blend_mode);
    }
    return result;
}

float sampleScalarBilinear(const std::vector<float>& values, int width, int height, float x, float y) {
    if (width <= 0 || height <= 0 || values.empty()) {
        return 0.0f;
    }

    const float sx = std::clamp(x, 0.0f, static_cast<float>(width - 1));
    const float sy = std::clamp(y, 0.0f, static_cast<float>(height - 1));
    const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, height - 1);
    const int x1 = std::clamp(x0 + 1, 0, width - 1);
    const int y1 = std::clamp(y0 + 1, 0, height - 1);
    const float tx = sx - static_cast<float>(x0);
    const float ty = sy - static_cast<float>(y0);

    const float v00 = values[static_cast<size_t>(y0) * static_cast<size_t>(width) + static_cast<size_t>(x0)];
    const float v10 = values[static_cast<size_t>(y0) * static_cast<size_t>(width) + static_cast<size_t>(x1)];
    const float v01 = values[static_cast<size_t>(y1) * static_cast<size_t>(width) + static_cast<size_t>(x0)];
    const float v11 = values[static_cast<size_t>(y1) * static_cast<size_t>(width) + static_cast<size_t>(x1)];
    const float a0 = v00 + (v10 - v00) * tx;
    const float a1 = v01 + (v11 - v01) * tx;
    return a0 + (a1 - a0) * ty;
}

float decodeRaisedThickness(uint8_t value) {
    return std::clamp((static_cast<float>(value) - 128.0f) / 127.0f, 0.0f, 1.0f);
}

uint8_t encodeRaisedThickness(float value) {
    return static_cast<uint8_t>(std::clamp(128.0f + std::clamp(value, 0.0f, 1.0f) * 127.0f, 128.0f, 255.0f) + 0.5f);
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

float sampleBrushAlpha(BrushAlphaPreset preset, float nx, float ny, float scale, float rotation_degrees, bool radial_gate) {
    float sx = nx * std::max(0.01f, scale);
    float sy = ny * std::max(0.01f, scale);
    rotateBrushCoords(sx, sy, rotation_degrees);
    const float radial = radial_gate ? std::clamp(1.0f - std::sqrt(nx * nx + ny * ny), 0.0f, 1.0f) : 1.0f;

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
    if (brush.flip_alpha_x) nx = -nx;
    if (brush.flip_alpha_y) ny = -ny;
    if (brush.use_imported_alpha && brush.alpha_texture && brush.alpha_texture->is_loaded()) {
        return sampleImportedBrushAlpha(brush.alpha_texture, nx, ny, brush.alpha_scale, brush.alpha_rotation_degrees);
    }

    const bool radial_gate = brush.shape == BrushShape::Circle;
    return sampleBrushAlpha(brush.alpha_preset, nx, ny, brush.alpha_scale, brush.alpha_rotation_degrees, radial_gate);
}

// Alpha gate from the user-loaded paint texture itself. PNG stamps with a
// transparent background usually store RGB=(0,0,0) on alpha=0 pixels, so
// without this the brush deposits visible black rings around the actual
// stamp. Mirrors makeBrushTexturePixel's UV transform so the alpha lines up
// with the colour sample.
float sampleBrushPaintTextureAlpha(PaintChannel channel, const BrushSettings& brush, float nx, float ny) {
    std::shared_ptr<Texture> paint_texture = getBrushChannelTexture(channel, brush);
    if (!paint_texture) {
        return 1.0f;
    }
    if (!paint_texture->has_alpha) {
        return 1.0f;
    }

    float sx = nx * std::max(0.01f, brush.alpha_scale);
    float sy = ny * std::max(0.01f, brush.alpha_scale);
    if (brush.flip_alpha_x) sx = -sx;
    if (brush.flip_alpha_y) sy = -sy;
    rotateBrushCoords(sx, sy, brush.alpha_rotation_degrees);
    const float u = std::clamp(sx * 0.5f + 0.5f, 0.0f, 1.0f);
    const float v = std::clamp(sy * 0.5f + 0.5f, 0.0f, 1.0f);
    return std::clamp(paint_texture->get_alpha_bilinear(u, v), 0.0f, 1.0f);
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

 // namespace

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

void MeshPaintAdapter::clearWetSimulation() {
    wet_basecolor_state_ = WetSimulationState{};
    invalidateWetFlowField();
}

std::vector<std::shared_ptr<Triangle>> MeshPaintAdapter::gatherNodeFacadesForPaint(
    const std::string& nodeName, uint16_t materialId) const {
    std::vector<std::shared_ptr<Triangle>> out;
    if (!scene_) return out;

    // Flat (SoA) mesh first: a TriangleMesh in world.objects carries the whole node — materialize a
    // facade per face so wet-flow / seam passes see EVERY triangle, not just the representative.
    for (const auto& obj : scene_->world.objects) {
        auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj);
        if (tm && tm->nodeName == nodeName && tm->geometry) {
            const size_t nT = tm->num_triangles();
            out.reserve(nT);
            for (size_t t = 0; t < nT; ++t) {
                auto facade = std::make_shared<Triangle>(tm, static_cast<uint32_t>(t));
                if (facade->getMaterialID() == materialId) out.push_back(std::move(facade));
            }
            return out;
        }
    }

    // Facade meshes: the edit cache holds the live face set; else scan world.objects.
    auto cache_it = scene_->base_mesh_cache.find(nodeName);
    if (cache_it != scene_->base_mesh_cache.end() && !cache_it->second.empty()) {
        return cache_it->second;
    }
    out.reserve(scene_->world.objects.size());
    for (const auto& object : scene_->world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(object);
        if (tri && tri->getNodeName() == nodeName && tri->getMaterialID() == materialId) {
            out.push_back(tri);
        }
    }
    return out;
}

void MeshPaintAdapter::invalidateWetFlowField() {
    wet_flow_field_cache_ = WetFlowFieldCache{};
}

void MeshPaintAdapter::rebuildWetFlowField(int width, int height) {
    WetFlowFieldCache cache;
    cache.target_key = getNodeName() + "#" + std::to_string(getMaterialID());
    cache.width = width;
    cache.height = height;
    cache.uv_set = getTarget().uv_set;

    std::vector<std::shared_ptr<Triangle>> candidates;
    if (scene_) {
        candidates = gatherNodeFacadesForPaint(getNodeName(), getMaterialID());
    }

    const RayTrophiSim::SimulationForceFieldSnapshot* force_snapshot = nullptr;
    if (scene_) {
        force_snapshot = &scene_->getSimulationWorld().getForceFieldSnapshot();
        if (force_snapshot->empty()) {
            force_snapshot = nullptr;
        }
    }
    cache.force_snapshot_version = force_snapshot ? force_snapshot->version() : 0;

    cache.infos.reserve(candidates.size() + 1);
    auto add_flow_info = [&](const std::shared_ptr<Triangle>& tri) {
        if (!tri || tri->getNodeName() != getNodeName() || tri->getMaterialID() != getMaterialID()) {
            return;
        }

        Vec3 driving_acceleration(0.0f, -1.0f, 0.0f);
        if (force_snapshot) {
            const Vec3 field_force = force_snapshot->evaluateAt(
                triangleCentroid(*tri),
                0.0f,
                Vec3(0.0f, 0.0f, 0.0f),
                RayTrophiSim::SimulationSystemKind::WetSurface);
            if (isFiniteVec3(field_force)) {
                driving_acceleration = driving_acceleration + field_force * (1.0f / 9.81f);
            }
        }

        WetFlowTriangleInfo info;
        if (!RayTrophiSim::SurfaceFlowField::computeTrianglePixelFlow(
                *tri,
                cache.uv_set,
                width,
                height,
                info.flow,
                driving_acceleration)) {
            return;
        }
        info.uvs = getTriangleUvArray(*tri, cache.uv_set);
        info.min_u = std::min({ info.uvs[0].u, info.uvs[1].u, info.uvs[2].u });
        info.max_u = std::max({ info.uvs[0].u, info.uvs[1].u, info.uvs[2].u });
        info.min_v = std::min({ info.uvs[0].v, info.uvs[1].v, info.uvs[2].v });
        info.max_v = std::max({ info.uvs[0].v, info.uvs[1].v, info.uvs[2].v });
        cache.max_slope = std::max(cache.max_slope, info.flow.slope);
        cache.max_flow_length = std::max(cache.max_flow_length, info.flow.flow_length);
        cache.infos.push_back(info);
    };

    add_flow_info(triangle_);
    for (const auto& tri : candidates) {
        if (!triangle_ || tri.get() != triangle_.get()) {
            add_flow_info(tri);
        }
    }

    cache.lookup_resolution = std::clamp(std::min(width, height) / 8, 16, 64);
    cache.lookup_indices.assign(static_cast<size_t>(cache.lookup_resolution) * static_cast<size_t>(cache.lookup_resolution), -1);
    for (int gy = 0; gy < cache.lookup_resolution; ++gy) {
        for (int gx = 0; gx < cache.lookup_resolution; ++gx) {
            const Vec2 uv(
                cache.lookup_resolution > 1 ? static_cast<float>(gx) / static_cast<float>(cache.lookup_resolution - 1) : 0.0f,
                cache.lookup_resolution > 1 ? 1.0f - static_cast<float>(gy) / static_cast<float>(cache.lookup_resolution - 1) : 0.0f);
            int best_index = -1;
            float best_margin = -1e9f;
            for (int info_index = 0; info_index < static_cast<int>(cache.infos.size()); ++info_index) {
                const WetFlowTriangleInfo& info = cache.infos[info_index];
                const float uv_pad = 0.0025f;
                if (uv.u < info.min_u - uv_pad || uv.u > info.max_u + uv_pad ||
                    uv.v < info.min_v - uv_pad || uv.v > info.max_v + uv_pad) {
                    continue;
                }
                float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f;
                if (!computeUvBarycentric(uv, info.uvs, w0, w1, w2)) {
                    continue;
                }
                const float min_w = std::min({ w0, w1, w2 });
                if (min_w < -0.003f) {
                    continue;
                }
                if (min_w > best_margin) {
                    best_margin = min_w;
                    best_index = info_index;
                }
            }
            cache.lookup_indices[static_cast<size_t>(gy) * static_cast<size_t>(cache.lookup_resolution) + static_cast<size_t>(gx)] = best_index;
        }
    }

    wet_flow_field_cache_ = std::move(cache);
}

int MeshPaintAdapter::findWetFlowTriangleIndex(const Vec2& uv, int hint_index) const {
    const WetFlowFieldCache& cache = wet_flow_field_cache_;
    if (cache.infos.empty()) {
        return -1;
    }

    const auto try_match = [&](int info_index) -> bool {
        if (info_index < 0 || info_index >= static_cast<int>(cache.infos.size())) {
            return false;
        }
        const WetFlowTriangleInfo& info = cache.infos[info_index];
        const float uv_pad = 0.0025f;
        if (uv.u < info.min_u - uv_pad || uv.u > info.max_u + uv_pad ||
            uv.v < info.min_v - uv_pad || uv.v > info.max_v + uv_pad) {
            return false;
        }
        float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f;
        if (!computeUvBarycentric(uv, info.uvs, w0, w1, w2)) {
            return false;
        }
        return w0 >= -0.003f && w1 >= -0.003f && w2 >= -0.003f;
    };

    if (try_match(hint_index)) {
        return hint_index;
    }

    if (cache.lookup_resolution > 0 && !cache.lookup_indices.empty()) {
        const int gx = std::clamp(static_cast<int>(std::round(uv.u * static_cast<float>(cache.lookup_resolution - 1))), 0, cache.lookup_resolution - 1);
        const int gy = std::clamp(static_cast<int>(std::round((1.0f - uv.v) * static_cast<float>(cache.lookup_resolution - 1))), 0, cache.lookup_resolution - 1);
        const int lookup_index = cache.lookup_indices[static_cast<size_t>(gy) * static_cast<size_t>(cache.lookup_resolution) + static_cast<size_t>(gx)];
        if (try_match(lookup_index)) {
            return lookup_index;
        }
    }

    for (int info_index = 0; info_index < static_cast<int>(cache.infos.size()); ++info_index) {
        if (try_match(info_index)) {
            return info_index;
        }
    }

    return -1;
}

void MeshPaintAdapter::rebuildWetSeamLinks() {
    wet_seam_links_.clear();
    wet_seam_key_.clear();
    if (!scene_ || !triangle_) {
        return;
    }

    const std::string target_key = getNodeName() + "#" + std::to_string(getMaterialID());
    wet_seam_key_ = target_key;

    const auto& source_indices = triangle_->getAssimpVertexIndices();
    std::array<Vec2, 3> source_uvs{};
    if (getTarget().uv_set > 0 && static_cast<size_t>(getTarget().uv_set) < triangle_->getUVSetCount()) {
        auto [u0, u1, u2] = triangle_->getUVSetCoordinates(static_cast<size_t>(getTarget().uv_set));
        source_uvs = { u0, u1, u2 };
    } else {
        auto [u0, u1, u2] = triangle_->getUVCoordinates();
        source_uvs = { u0, u1, u2 };
    }

    std::vector<std::shared_ptr<Triangle>> candidates = gatherNodeFacadesForPaint(getNodeName(), getMaterialID());

    for (const auto& candidate : candidates) {
        if (!candidate || candidate.get() == triangle_.get() || candidate->getMaterialID() != getMaterialID()) {
            continue;
        }

        const auto& other_indices = candidate->getAssimpVertexIndices();
        std::array<int, 2> source_shared = { -1, -1 };
        std::array<int, 2> target_shared = { -1, -1 };
        int shared_count = 0;
        for (int si = 0; si < 3; ++si) {
            for (int ti = 0; ti < 3; ++ti) {
                if (source_indices[si] == other_indices[ti]) {
                    if (shared_count < 2) {
                        source_shared[shared_count] = si;
                        target_shared[shared_count] = ti;
                    }
                    ++shared_count;
                }
            }
        }
        if (shared_count != 2) {
            continue;
        }

        int source_opposite = 0;
        while (source_opposite == source_shared[0] || source_opposite == source_shared[1]) {
            ++source_opposite;
        }
        int target_opposite = 0;
        while (target_opposite == target_shared[0] || target_opposite == target_shared[1]) {
            ++target_opposite;
        }

        std::array<Vec2, 3> target_uvs{};
        if (getTarget().uv_set > 0 && static_cast<size_t>(getTarget().uv_set) < candidate->getUVSetCount()) {
            auto [u0, u1, u2] = candidate->getUVSetCoordinates(static_cast<size_t>(getTarget().uv_set));
            target_uvs = { u0, u1, u2 };
        } else {
            auto [u0, u1, u2] = candidate->getUVCoordinates();
            target_uvs = { u0, u1, u2 };
        }

        WetSeamLink link;
        link.source_uvs = source_uvs;
        link.target_uvs = target_uvs;
        link.source_edge_a = source_shared[0];
        link.source_edge_b = source_shared[1];
        link.source_opposite = source_opposite;
        link.target_edge_a = target_shared[0];
        link.target_edge_b = target_shared[1];
        link.target_opposite = target_opposite;
        wet_seam_links_.push_back(link);
    }
}

void MeshPaintAdapter::mirrorWetRegionAcrossSeams(std::vector<CompactVec4>& pixels, int width, int height,
                                                  PaintDirtyRect region, float blend_strength) {
    if (region.empty() || wet_basecolor_state_.wetness.empty() || width <= 1 || height <= 1) {
        return;
    }
    const std::string target_key = getNodeName() + "#" + std::to_string(getMaterialID());
    if (wet_seam_key_ != target_key) {
        rebuildWetSeamLinks();
    }
    if (wet_seam_links_.empty()) {
        return;
    }

    const float seam_band = 0.08f;
    const float blend = std::clamp(blend_strength, 0.0f, 1.0f);
    for (int py = std::max(0, region.min_y); py <= std::min(height - 1, region.max_y); ++py) {
        for (int px = std::max(0, region.min_x); px <= std::min(width - 1, region.max_x); ++px) {
            const size_t src_idx = static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px);
            const float wet = wet_basecolor_state_.wetness[src_idx];
            if (wet <= 0.001f) {
                continue;
            }

            const Vec2 source_uv(
                width > 1 ? static_cast<float>(px) / static_cast<float>(width - 1) : 0.0f,
                height > 1 ? 1.0f - static_cast<float>(py) / static_cast<float>(height - 1) : 0.0f);
            const CompactVec4 src_pixel = pixels[src_idx];

            for (const WetSeamLink& link : wet_seam_links_) {
                float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f;
                if (!computeUvBarycentric(source_uv, link.source_uvs, w0, w1, w2)) {
                    continue;
                }
                const float weights[3] = { w0, w1, w2 };
                const float edge_distance = weights[link.source_opposite];
                if (edge_distance < -0.005f || edge_distance > seam_band) {
                    continue;
                }

                float mapped[3] = { 0.0f, 0.0f, 0.0f };
                mapped[link.target_edge_a] = std::max(0.0f, weights[link.source_edge_a]);
                mapped[link.target_edge_b] = std::max(0.0f, weights[link.source_edge_b]);
                mapped[link.target_opposite] = std::max(0.0f, edge_distance);
                const float sum = mapped[0] + mapped[1] + mapped[2];
                if (sum <= 1e-6f) {
                    continue;
                }
                mapped[0] /= sum;
                mapped[1] /= sum;
                mapped[2] /= sum;

                const Vec2 target_uv = uvFromBarycentric(link.target_uvs, mapped[0], mapped[1], mapped[2]);
                const int dst_x = std::clamp(static_cast<int>(std::round(target_uv.u * static_cast<float>(width - 1))), 0, width - 1);
                const int dst_y = std::clamp(static_cast<int>(std::round((1.0f - target_uv.v) * static_cast<float>(height - 1))), 0, height - 1);
                const size_t dst_idx = static_cast<size_t>(dst_y) * static_cast<size_t>(width) + static_cast<size_t>(dst_x);
                const float proximity = 1.0f - std::clamp(edge_distance / seam_band, 0.0f, 1.0f);
                const float alpha = std::clamp(blend * wet * proximity, 0.0f, 1.0f);
                if (alpha <= 0.001f) {
                    continue;
                }

                blendBaseColorChannelLinear(pixels[dst_idx].r, src_pixel.r, alpha);
                blendBaseColorChannelLinear(pixels[dst_idx].g, src_pixel.g, alpha);
                blendBaseColorChannelLinear(pixels[dst_idx].b, src_pixel.b, alpha);
                wet_basecolor_state_.wetness[dst_idx] = std::max(wet_basecolor_state_.wetness[dst_idx], wet * proximity);
                if (dst_idx < wet_basecolor_state_.thickness.size() && src_idx < wet_basecolor_state_.thickness.size()) {
                    wet_basecolor_state_.thickness[dst_idx] = std::max(
                        wet_basecolor_state_.thickness[dst_idx],
                        wet_basecolor_state_.thickness[src_idx] * proximity);
                }
                wet_basecolor_state_.active_region.expand(dst_x, dst_y, dst_x, dst_y);
            }
        }
    }
}

void MeshPaintAdapter::noteWetDab(int layer_index, const Vec2& uv, const BrushSettings& brush,
                                  float dt, float deposit_ratio) {
    if ((brush.paint_mode != BrushPaintMode::Wet && brush.paint_mode != BrushPaintMode::Oil) || !isValid()) {
        return;
    }

    PaintLayerData* layer = nullptr;
    PaintTextureSet* texture_set = nullptr;
    std::shared_ptr<Texture> texture;
    std::vector<CompactVec4>* mask_pixels = nullptr;
    std::vector<WetAuxChannelBinding> aux_channels;
    bool uses_layers = false;
    uint32_t layer_id = 0;
    int width = 0;
    int height = 0;

    PaintLayerStack* stack = getLayerStack();
    if (stack && stack->layerCount() > 0) {
        layer = stack->layerAt(layer_index);
        if (!layer || layer->meta.locked || !layer->meta.visible) {
            return;
        }
        layer->ensurePixels(PaintChannel::BaseColor);
        if (brush.write_height_mask) {
            mask_pixels = &layer->ensurePixels(PaintChannel::Mask);
        }
        for (PaintChannel aux_channel : kWetAuxScalarChannels) {
            if (!getBrushChannelInput(brush, aux_channel)) {
                continue;
            }
            aux_channels.push_back(WetAuxChannelBinding{
                aux_channel,
                &layer->ensurePixels(aux_channel),
                nullptr
            });
        }
        uses_layers = true;
        layer_id = layer->id;
        width = layer->width;
        height = layer->height;
    } else {
        texture_set = getTextureSet();
        if (!texture_set) {
            return;
        }
        texture = texture_set->getTexture(PaintChannel::BaseColor);
        if (!texture || texture->width <= 0 || texture->height <= 0 || texture->pixels.empty()) {
            return;
        }
        if (brush.write_height_mask) {
            std::shared_ptr<Texture> mask_texture = texture_set->getTexture(PaintChannel::Mask);
            if (mask_texture && mask_texture->width == texture->width && mask_texture->height == texture->height && !mask_texture->pixels.empty()) {
                mask_pixels = &mask_texture->pixels;
            }
        }
        for (PaintChannel aux_channel : kWetAuxScalarChannels) {
            if (!getBrushChannelInput(brush, aux_channel)) {
                continue;
            }
            if (!assignTextureToChannel(aux_channel)) {
                continue;
            }
            std::shared_ptr<Texture> aux_texture = texture_set->getTexture(aux_channel);
            if (!aux_texture || aux_texture->width != texture->width || aux_texture->height != texture->height || aux_texture->pixels.empty()) {
                continue;
            }
            aux_channels.push_back(WetAuxChannelBinding{ aux_channel, &aux_texture->pixels, aux_texture });
        }
        width = texture->width;
        height = texture->height;
    }

    if (width <= 0 || height <= 0) {
        return;
    }

    const std::string target_key = getNodeName() + "#" + std::to_string(getMaterialID());
    WetSimulationState& state = wet_basecolor_state_;
    if (state.target_key != target_key ||
        state.width != width ||
        state.height != height ||
        state.uses_layers != uses_layers ||
        state.layer_id != layer_id) {
        state = WetSimulationState{};
        state.target_key = target_key;
        state.width = width;
        state.height = height;
        state.uses_layers = uses_layers;
        state.layer_id = layer_id;
        state.wetness.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
        state.pigment.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
        state.thickness.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
    } else if (state.wetness.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
        state.wetness.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
        state.pigment.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
        state.thickness.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
        state.active_region = PaintDirtyRect{};
    } else if (state.pigment.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
        state.pigment.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
    } else if (state.thickness.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
        state.thickness.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
    }

    if (brush.write_height_mask && mask_pixels && !state.thickness.empty()) {
        for (size_t idx = 0; idx < state.thickness.size() && idx < mask_pixels->size(); ++idx) {
            state.thickness[idx] = std::max(state.thickness[idx], decodeRaisedThickness((*mask_pixels)[idx].r));
        }
    }

    const float clamped_u = std::clamp(uv.u, 0.0f, 1.0f);
    const float clamped_v = std::clamp(uv.v, 0.0f, 1.0f);
    const float center_x = clamped_u * static_cast<float>(width - 1);
    const float center_y = (1.0f - clamped_v) * static_cast<float>(height - 1);
    const float reference_res = 1024.0f;
    const float res_scale = static_cast<float>(width) / reference_res;
    const float radius_px = std::max(0.1f, brush.radius * res_scale);

    float kx = 1.0f;
    float ky = 1.0f;
    if (triangle_) {
        float wpu = 1.0f;
        float wpv = 1.0f;
        const int uv_set = getTarget().uv_set;
        computeTriangleUvJacobianLengths(*triangle_, uv_set, wpu, wpv);
        computeBrushAxisCorrection(wpu, wpv, width, height, kx, ky);
    }
    const float extent_scale = std::max(brushShapeAspectScale(brush), 1.0f / brushShapeAspectScale(brush));
    const float radius_x = std::max(0.1f, radius_px * kx);
    const float radius_y = std::max(0.1f, radius_px * ky);
    const float bound_radius_x = radius_x * extent_scale;
    const float bound_radius_y = radius_y * extent_scale;

    const int min_x = std::max(0, static_cast<int>(std::floor(center_x - bound_radius_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + bound_radius_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(center_y - bound_radius_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + bound_radius_y)));
    const float strength = std::clamp(brush.strength * brush.flow * dt * 60.0f, 0.0f, 1.0f);
    const float carried_paint = std::clamp(brush.paint_load, 0.0f, 1.0f);
    const float deposited_pigment = std::clamp(deposit_ratio, 0.0f, 1.0f) * (0.25f + carried_paint * 0.75f);
    const float activation = std::clamp(
        strength * std::clamp(brush.wetness, 0.0f, 1.0f) * (0.04f + deposited_pigment * 0.96f),
        0.0f,
        1.0f);
    if (activation <= 0.001f) {
        return;
    }

    std::vector<CompactVec4>* pixels = nullptr;
    if (uses_layers) {
        pixels = &layer->ensurePixels(PaintChannel::BaseColor);
    } else if (texture) {
        pixels = &texture->pixels;
    }
    PaintDirtyRect dirty;

    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - center_x;
            const float dy = (static_cast<float>(py) + 0.5f) - center_y;
            const BrushFootprintSample fp = sampleBrushFootprint(brush, dx, dy, radius_x, radius_y);
            if (fp.dist_norm > 1.0f) {
                continue;
            }

            const float nx = dx / radius_x;
            const float ny = dy / radius_y;
            const float alpha_mask = sampleBrushMask(brush, nx, ny);
            const float texture_alpha = sampleBrushPaintTextureAlpha(PaintChannel::BaseColor, brush, nx, ny);
        const CompactVec4 brush_pixel = makeBrushTexturePixel(PaintChannel::BaseColor, brush, nx, ny);
            const float wet_add = computeBrushWeightNormalized(
                    fp.dist_norm,
                    radius_px,
                    std::clamp(brush.falloff, 0.0f, 1.0f)) * alpha_mask * texture_alpha * activation;
            if (wet_add <= 0.001f) {
                continue;
            }

            const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px);
            const float pigment_add = wet_add * deposited_pigment;
            state.wetness[idx] = std::clamp(state.wetness[idx] + wet_add * (0.18f + deposited_pigment * 0.82f), 0.0f, 1.0f);
            if (idx < state.pigment.size()) {
                state.pigment[idx] = std::clamp(state.pigment[idx] + pigment_add, 0.0f, 1.0f);
            }
            if (brush.write_height_mask && idx < state.thickness.size()) {
                const float thickness_add = pigment_add * std::clamp(brush.height_contribution * 1.6f, 0.0f, 1.0f);
                state.thickness[idx] = std::clamp(state.thickness[idx] + thickness_add, 0.0f, 1.0f);
            }

            if (pixels && idx < pixels->size() && pigment_add > 0.0f) {
                CompactVec4& pixel = (*pixels)[idx];
                const float dst_alpha = static_cast<float>(pixel.a) / 255.0f;
                const float seed_alpha = std::clamp(
                    wet_add * (0.22f + deposited_pigment * 0.78f),
                    0.0f,
                    1.0f);
                const float seed_visibility = std::clamp((0.2f - dst_alpha) / 0.2f, 0.0f, 1.0f);
                const float effective_seed = seed_alpha * seed_visibility;
                if (effective_seed > 0.001f) {
                    const float out_alpha = effective_seed + dst_alpha * (1.0f - effective_seed);
                    if (out_alpha > 1e-6f) {
                        const float inv_out_alpha = 1.0f / out_alpha;
                        const float keep = dst_alpha * (1.0f - effective_seed);
                        pixel.r = static_cast<uint8_t>(std::clamp((static_cast<float>(brush_pixel.r) * effective_seed +
                                                                  static_cast<float>(pixel.r) * keep) * inv_out_alpha,
                                                                 0.0f, 255.0f));
                        pixel.g = static_cast<uint8_t>(std::clamp((static_cast<float>(brush_pixel.g) * effective_seed +
                                                                  static_cast<float>(pixel.g) * keep) * inv_out_alpha,
                                                                 0.0f, 255.0f));
                        pixel.b = static_cast<uint8_t>(std::clamp((static_cast<float>(brush_pixel.b) * effective_seed +
                                                                  static_cast<float>(pixel.b) * keep) * inv_out_alpha,
                                                                 0.0f, 255.0f));
                        pixel.a = static_cast<uint8_t>(std::clamp(out_alpha * 255.0f, 0.0f, 255.0f));
                    }
                }
            }

            for (WetAuxChannelBinding& aux : aux_channels) {
                if (!aux.pixels || idx >= aux.pixels->size()) {
                    continue;
                }
                const float aux_texture_alpha = sampleBrushPaintTextureAlpha(aux.channel, brush, nx, ny);
                const float aux_wet_add = computeBrushWeightNormalized(
                    fp.dist_norm,
                    radius_px,
                    std::clamp(brush.falloff, 0.0f, 1.0f)) * alpha_mask * aux_texture_alpha * activation;
                if (aux_wet_add <= 0.001f) {
                    continue;
                }

                CompactVec4& aux_pixel = (*aux.pixels)[idx];
                const CompactVec4 aux_brush_pixel = makeBrushTexturePixel(aux.channel, brush, nx, ny);
                const float aux_seed_alpha = std::clamp(
                    aux_wet_add * (0.22f + deposited_pigment * 0.78f),
                    0.0f,
                    1.0f);
                blendScalarPixel(aux_pixel, aux_brush_pixel, aux_seed_alpha);
                dirty.expand(px, py, px, py);
            }

            state.active_region.expand(px, py, px, py);
            dirty.expand(px, py, px, py);
        }
    }

    if (pixels && !dirty.empty()) {
        mirrorWetRegionAcrossSeams(*pixels, width, height, dirty, 0.55f);
        for (WetAuxChannelBinding& aux : aux_channels) {
            if (aux.pixels) {
                mirrorWetRegionAcrossSeams(*aux.pixels, width, height, dirty, 0.55f);
            }
        }
    }
}

bool MeshPaintAdapter::noteWeatherExposure(int layer_index, const WeatherParams& weather, float dt) {
    if (!isValid() || dt <= 0.0f) {
        return false;
    }
    if (weather.enabled == 0 ||
        weather.surface_response_enabled == 0 ||
        weather.type != WEATHER_RAIN ||
        weather.intensity <= 0.0f ||
        weather.density <= 0.0f) {
        return false;
    }

    PaintLayerData* layer = nullptr;
    PaintTextureSet* texture_set = nullptr;
    std::shared_ptr<Texture> texture;
    bool uses_layers = false;
    uint32_t layer_id = 0;
    int width = 0;
    int height = 0;

    PaintLayerStack* stack = getLayerStack();
    if (stack && stack->layerCount() > 0) {
        const int clamped_layer_index = std::clamp(layer_index, 0, stack->layerCount() - 1);
        layer = stack->layerAt(clamped_layer_index);
        if (!layer || layer->meta.locked || !layer->meta.visible) {
            return false;
        }
        layer->ensurePixels(PaintChannel::BaseColor);
        uses_layers = true;
        layer_id = layer->id;
        width = layer->width;
        height = layer->height;
    } else {
        texture_set = getTextureSet();
        if (!texture_set) {
            return false;
        }
        texture = texture_set->getTexture(PaintChannel::BaseColor);
        if (!texture || texture->width <= 0 || texture->height <= 0 || texture->pixels.empty()) {
            return false;
        }
        width = texture->width;
        height = texture->height;
    }

    if (width <= 0 || height <= 0) {
        return false;
    }

    const std::string target_key = getNodeName() + "#" + std::to_string(getMaterialID());
    WetSimulationState& state = wet_basecolor_state_;
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (state.target_key != target_key ||
        state.width != width ||
        state.height != height ||
        state.uses_layers != uses_layers ||
        state.layer_id != layer_id) {
        state = WetSimulationState{};
        state.target_key = target_key;
        state.width = width;
        state.height = height;
        state.uses_layers = uses_layers;
        state.layer_id = layer_id;
        state.wetness.assign(pixel_count, 0.0f);
        state.pigment.assign(pixel_count, 0.0f);
        state.thickness.assign(pixel_count, 0.0f);
    } else if (state.wetness.size() != pixel_count ||
               state.pigment.size() != pixel_count ||
               state.thickness.size() != pixel_count) {
        state.wetness.assign(pixel_count, 0.0f);
        state.pigment.assign(pixel_count, 0.0f);
        state.thickness.assign(pixel_count, 0.0f);
        state.active_region = PaintDirtyRect{};
    }

    const float rain_amount = std::clamp(weather.surface_wetness_output, 0.0f, 1.0f) *
                              (0.30f + std::clamp(weather.density, 0.0f, 1.0f) * 0.70f);
    if (rain_amount <= 0.001f) {
        return false;
    }

    const float resolution_scale = std::sqrt(static_cast<float>(pixel_count)) / 1024.0f;
    const float spawn_rate = (0.20f + rain_amount * 1.75f) * std::max(0.30f, resolution_scale);
    state.weather_spawn_accumulator += spawn_rate * dt * 60.0f;
    const int spawn_count = std::clamp(static_cast<int>(std::floor(state.weather_spawn_accumulator)), 0, 8);
    if (spawn_count <= 0) {
        return false;
    }
    state.weather_spawn_accumulator -= static_cast<float>(spawn_count);

    state.weather_cluster_refresh_accumulator += dt * (0.25f + rain_amount * 0.55f);
    if (state.weather_spawn_cursor == 0 || state.weather_cluster_refresh_accumulator >= 1.0f) {
        const float cluster_seed = static_cast<float>(state.weather_spawn_cursor) + rain_amount * 37.0f + static_cast<float>(width + height);
        state.weather_cluster_u = weatherSpawnHash(cluster_seed, 17.3f);
        state.weather_cluster_v = weatherSpawnHash(cluster_seed, 59.9f);
        state.weather_cluster_refresh_accumulator -= std::floor(state.weather_cluster_refresh_accumulator);
    }

    const float cluster_spread_u = std::clamp(0.010f + rain_amount * 0.040f, 0.010f, 0.060f);
    const float cluster_spread_v = std::clamp(0.014f + rain_amount * 0.055f, 0.014f, 0.080f);

    bool changed = false;
    for (int spawn_index = 0; spawn_index < spawn_count; ++spawn_index) {
        const uint32_t cursor = ++state.weather_spawn_cursor;
        const float seed = static_cast<float>(cursor) + rain_amount * 19.0f;
        const float u = std::clamp(
            state.weather_cluster_u + (weatherSpawnHash(seed, 11.7f) - 0.5f) * cluster_spread_u,
            0.0f,
            1.0f);
        const float v = std::clamp(
            state.weather_cluster_v + (weatherSpawnHash(seed, 47.3f) - 0.5f) * cluster_spread_v,
            0.0f,
            1.0f);
        const float radius = 0.55f + weatherSpawnHash(seed, 83.1f) * (0.65f + rain_amount * 1.05f);
        const float stretch_y = 1.0f + rain_amount * 1.8f;
        const float wet_add = 0.028f + rain_amount * (0.045f + weatherSpawnHash(seed, 131.9f) * 0.060f);
        const float thickness_add = rain_amount * (0.003f + weatherSpawnHash(seed, 191.4f) * 0.012f);
        const float center_x = u * static_cast<float>(width - 1);
        const float center_y = v * static_cast<float>(height - 1);

        const int min_x = std::max(0, static_cast<int>(std::floor(center_x - radius - 1.0f)));
        const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + radius + 1.0f)));
        const int min_y = std::max(0, static_cast<int>(std::floor(center_y - radius * stretch_y - 1.0f)));
        const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + radius * stretch_y + 1.0f)));

        for (int py = min_y; py <= max_y; ++py) {
            for (int px = min_x; px <= max_x; ++px) {
                const float dx = (static_cast<float>(px) + 0.5f) - center_x;
                const float dy = ((static_cast<float>(py) + 0.5f) - center_y) / stretch_y;
                const float dist = std::sqrt(dx * dx + dy * dy) / std::max(radius, 0.001f);
                if (dist > 1.0f) {
                    continue;
                }

                const float falloff = 1.0f - dist * dist * (3.0f - 2.0f * dist);
                const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px);
                state.wetness[idx] = std::clamp(state.wetness[idx] + wet_add * falloff, 0.0f, 1.0f);
                state.thickness[idx] = std::clamp(state.thickness[idx] + thickness_add * falloff, 0.0f, 1.0f);
                state.active_region.expand(px, py, px, py);
                changed = true;
            }
        }
    }

    return changed;
}

float MeshPaintAdapter::sampleWetPickupReservoir(const Vec2& uv) const {
    const WetSimulationState& state = wet_basecolor_state_;
    if (state.width <= 0 || state.height <= 0 || state.wetness.empty()) {
        return 0.0f;
    }

    const float clamped_u = std::clamp(uv.u, 0.0f, 1.0f);
    const float clamped_v = std::clamp(uv.v, 0.0f, 1.0f);
    const float sample_x = clamped_u * static_cast<float>(std::max(1, state.width - 1));
    const float sample_y = (1.0f - clamped_v) * static_cast<float>(std::max(1, state.height - 1));
    const float wet = sampleScalarBilinear(state.wetness, state.width, state.height, sample_x, sample_y);
    const float thickness = state.thickness.empty()
        ? 0.0f
        : sampleScalarBilinear(state.thickness, state.width, state.height, sample_x, sample_y);

    const float wet_gate = std::clamp((wet - 0.04f) / 0.28f, 0.0f, 1.0f);
    const float thickness_gate = std::clamp((thickness - 0.02f) / 0.20f, 0.0f, 1.0f);
    return std::clamp(wet_gate * (0.30f + thickness_gate * 0.70f), 0.0f, 1.0f);
}

bool MeshPaintAdapter::tickWetPaint(const BrushSettings& brush, float dt,
                                    bool auto_normal_from_height_enabled,
                                    float normal_strength) {
    WetSimulationState& state = wet_basecolor_state_;
    if (state.wetness.empty() || state.active_region.empty()) {
        return false;
    }

    dt = dt > 0.0f ? dt : (1.0f / 60.0f);

    const int max_dimension = std::max(state.width, state.height);
    auto targetSimulationHz = [&]() -> float {
        switch (brush.wet_simulation_quality) {
            case WetSimulationQuality::Balanced:
                return 18.0f;
            case WetSimulationQuality::High:
                return 30.0f;
            case WetSimulationQuality::Ultra:
                return 60.0f;
            case WetSimulationQuality::Auto:
            default:
                if (max_dimension <= 1024) return 60.0f;
                if (max_dimension <= 2048) return 30.0f;
                if (max_dimension <= 4096) return 15.0f;
                return 10.0f;
        }
    };

    state.simulation_time_accumulator += dt;
    const float target_hz = std::max(1.0f, targetSimulationHz());
    const float target_step = 1.0f / target_hz;
    if (state.simulation_time_accumulator + 1e-6f < target_step) {
        return false;
    }
    dt = std::min(state.simulation_time_accumulator, target_step * 2.5f);
    state.simulation_time_accumulator = 0.0f;

    std::vector<CompactVec4>* pixels = nullptr;
    std::vector<CompactVec4>* mask_pixels = nullptr;
    std::shared_ptr<Texture> texture;
    std::shared_ptr<Texture> mask_texture;
    std::vector<WetAuxChannelBinding> aux_channels;
    if (state.uses_layers) {
        PaintLayerStack* stack = getLayerStack();
        PaintLayerData* layer = stack ? stack->layerById(state.layer_id) : nullptr;
        if (!layer) {
            clearWetSimulation();
            return false;
        }
        pixels = &layer->ensurePixels(PaintChannel::BaseColor);
        if (brush.write_height_mask || auto_normal_from_height_enabled) {
            mask_pixels = &layer->ensurePixels(PaintChannel::Mask);
        }
        for (PaintChannel aux_channel : kWetAuxScalarChannels) {
            if (!getBrushChannelInput(brush, aux_channel)) {
                continue;
            }
            aux_channels.push_back(WetAuxChannelBinding{
                aux_channel,
                &layer->ensurePixels(aux_channel),
                nullptr
            });
        }
        if (layer->width != state.width || layer->height != state.height) {
            clearWetSimulation();
            return false;
        }
    } else {
        PaintTextureSet* set = getTextureSet();
        texture = set ? set->getTexture(PaintChannel::BaseColor) : nullptr;
        if (!texture || texture->width != state.width || texture->height != state.height || texture->pixels.empty()) {
            clearWetSimulation();
            return false;
        }
        pixels = &texture->pixels;
        if (brush.write_height_mask || auto_normal_from_height_enabled) {
            PaintTextureSet* set = getTextureSet();
            mask_texture = set ? set->getTexture(PaintChannel::Mask) : nullptr;
            if (mask_texture && mask_texture->width == state.width && mask_texture->height == state.height && !mask_texture->pixels.empty()) {
                mask_pixels = &mask_texture->pixels;
            }
        }
        for (PaintChannel aux_channel : kWetAuxScalarChannels) {
            if (!getBrushChannelInput(brush, aux_channel)) {
                continue;
            }
            if (!assignTextureToChannel(aux_channel)) {
                continue;
            }
            PaintTextureSet* set = getTextureSet();
            std::shared_ptr<Texture> aux_texture = set ? set->getTexture(aux_channel) : nullptr;
            if (!aux_texture || aux_texture->width != state.width || aux_texture->height != state.height || aux_texture->pixels.empty()) {
                continue;
            }
            aux_channels.push_back(WetAuxChannelBinding{ aux_channel, &aux_texture->pixels, aux_texture });
        }
    }

    if (!pixels || pixels->size() != static_cast<size_t>(state.width) * static_cast<size_t>(state.height)) {
        clearWetSimulation();
        return false;
    }
    for (const WetAuxChannelBinding& aux : aux_channels) {
        if (!aux.pixels || aux.pixels->size() != static_cast<size_t>(state.width) * static_cast<size_t>(state.height)) {
            clearWetSimulation();
            return false;
        }
    }

    const bool oil_mode = brush.paint_mode == BrushPaintMode::Oil;
    const float lifetime = std::max(0.05f, brush.wet_lifetime_seconds);
    const float lifetime_runoff_factor = std::clamp((lifetime - 0.35f) / 1.5f, 0.0f, 1.0f);
    const float diffusion = std::clamp(brush.wet_diffusion, 0.0f, 2.0f) * (oil_mode ? 0.58f : 1.0f);
    const float runoff = std::clamp(brush.wet_runoff, 0.0f, 2.0f) * lifetime_runoff_factor * (oil_mode ? 0.18f : 1.0f);
    const float absorption = std::clamp(brush.wet_absorption, 0.0f, 1.0f) * (oil_mode ? 0.55f : 1.0f);
    const float drip_head = std::clamp(brush.wet_drip_head, 0.0f, 1.5f) * lifetime_runoff_factor * (oil_mode ? 0.24f : 1.0f);
    const float terminal_buildup = std::clamp(brush.wet_terminal_buildup, 0.0f, 1.5f) * (oil_mode ? 1.35f : 1.0f);
    const float terminal_softness = std::clamp(brush.wet_terminal_softness, 0.1f, 1.0f);

    const std::string target_key = getNodeName() + "#" + std::to_string(getMaterialID());
    uint64_t force_snapshot_version = 0;
    if (scene_) {
        scene_->refreshSimulationForceFieldSnapshot();
        const auto& force_snapshot = scene_->getSimulationWorld().getForceFieldSnapshot();
        if (!force_snapshot.empty()) {
            force_snapshot_version = force_snapshot.version();
        }
    }

    if (wet_flow_field_cache_.target_key != target_key ||
        wet_flow_field_cache_.width != state.width ||
        wet_flow_field_cache_.height != state.height ||
        wet_flow_field_cache_.uv_set != getTarget().uv_set ||
        wet_flow_field_cache_.force_snapshot_version != force_snapshot_version) {
        rebuildWetFlowField(state.width, state.height);
    }

    const WetFlowFieldCache& flow_cache = wet_flow_field_cache_;
    const bool has_downhill = !flow_cache.infos.empty();
    const float reference_res = 1024.0f;
    const float brush_radius_px = std::max(0.1f, brush.radius * (static_cast<float>(state.width) / reference_res));
    const float max_flow_metric = has_downhill
        ? std::clamp(1.0f + std::log1p(std::max(0.0f, flow_cache.max_flow_length)) * 0.85f, 1.0f, 4.0f)
        : 0.0f;
    const float max_shift = has_downhill
        ? std::min(28.0f, brush_radius_px * 0.35f + max_flow_metric * runoff * std::max(0.18f, flow_cache.max_slope) * dt * 60.0f * 6.0f)
        : 0.0f;
    const int pad = std::max(2, static_cast<int>(std::ceil(max_shift)) + 2);

    const int min_x = std::max(0, state.active_region.min_x - pad);
    const int min_y = std::max(0, state.active_region.min_y - pad);
    const int max_x = std::min(state.width - 1, state.active_region.max_x + pad);
    const int max_y = std::min(state.height - 1, state.active_region.max_y + pad);

    const int source_min_x = std::max(0, min_x - pad);
    const int source_min_y = std::max(0, min_y - pad);
    const int source_max_x = std::min(state.width - 1, max_x + pad);
    const int source_max_y = std::min(state.height - 1, max_y + pad);
    const int source_width = source_max_x - source_min_x + 1;
    const int source_height = source_max_y - source_min_y + 1;

    auto copyPixelWindow = [&](const std::vector<CompactVec4>& src) {
        std::vector<CompactVec4> window(static_cast<size_t>(source_width) * static_cast<size_t>(source_height));
        for (int y = 0; y < source_height; ++y) {
            const size_t src_offset = static_cast<size_t>(source_min_y + y) * static_cast<size_t>(state.width) + static_cast<size_t>(source_min_x);
            const size_t dst_offset = static_cast<size_t>(y) * static_cast<size_t>(source_width);
            std::copy_n(src.begin() + static_cast<std::ptrdiff_t>(src_offset), source_width, window.begin() + static_cast<std::ptrdiff_t>(dst_offset));
        }
        return window;
    };

    auto copyScalarWindow = [&](const std::vector<float>& src) {
        std::vector<float> window(static_cast<size_t>(source_width) * static_cast<size_t>(source_height));
        for (int y = 0; y < source_height; ++y) {
            const size_t src_offset = static_cast<size_t>(source_min_y + y) * static_cast<size_t>(state.width) + static_cast<size_t>(source_min_x);
            const size_t dst_offset = static_cast<size_t>(y) * static_cast<size_t>(source_width);
            std::copy_n(src.begin() + static_cast<std::ptrdiff_t>(src_offset), source_width, window.begin() + static_cast<std::ptrdiff_t>(dst_offset));
        }
        return window;
    };

    const std::vector<CompactVec4> source_pixels = copyPixelWindow(*pixels);
    const std::vector<CompactVec4> source_mask_pixels = mask_pixels ? copyPixelWindow(*mask_pixels) : std::vector<CompactVec4>{};
    const std::vector<float> source_wetness = copyScalarWindow(state.wetness);
    const std::vector<float> source_pigment = copyScalarWindow(state.pigment);
    const std::vector<float> source_thickness = copyScalarWindow(state.thickness);
    for (WetAuxChannelBinding& aux : aux_channels) {
        aux.source_pixels = copyPixelWindow(*aux.pixels);
    }

    auto sourcePixelIndex = [&](int gx, int gy) -> size_t {
        const int local_x = std::clamp(gx - source_min_x, 0, source_width - 1);
        const int local_y = std::clamp(gy - source_min_y, 0, source_height - 1);
        return static_cast<size_t>(local_y) * static_cast<size_t>(source_width) + static_cast<size_t>(local_x);
    };

    auto sampleSourcePixel = [&](const std::vector<CompactVec4>& src, float gx, float gy) {
        const float local_x = gx - static_cast<float>(source_min_x);
        const float local_y = gy - static_cast<float>(source_min_y);
        const float u = source_width > 1 ? (local_x / static_cast<float>(source_width - 1)) : 0.0f;
        const float v = source_height > 1 ? (1.0f - local_y / static_cast<float>(source_height - 1)) : 0.0f;
        return sampleTexturePixelBilinear(src, source_width, source_height, u, v);
    };

    auto sampleSourceScalar = [&](const std::vector<float>& src, float gx, float gy) {
        return sampleScalarBilinear(src, source_width, source_height,
                                    gx - static_cast<float>(source_min_x),
                                    gy - static_cast<float>(source_min_y));
    };

    PaintDirtyRect dirty;
    PaintDirtyRect next_active;
    bool changed = false;
    bool mask_changed = false;
    auto liftLayerAlpha = [&](CompactVec4& pixel, float pigment_amount, float wet_amount,
                              float incoming_alpha, float blend_alpha) {
        if (!state.uses_layers) {
            return;
        }

        const float current_alpha = static_cast<float>(pixel.a) / 255.0f;
        const float pigment_alpha = std::clamp(
            pigment_amount * (0.24f + std::clamp(wet_amount, 0.0f, 1.0f) * 0.60f),
            0.0f,
            1.0f);
        const float carried_alpha = std::clamp(incoming_alpha * blend_alpha, 0.0f, 1.0f);
        const float target_alpha = std::max(current_alpha, std::max(pigment_alpha, carried_alpha));
        if (target_alpha > current_alpha) {
            pixel.a = static_cast<uint8_t>(std::clamp(target_alpha * 255.0f, 0.0f, 255.0f));
        }
    };
    int last_flow_info_index = flow_cache.infos.empty() ? -1 : 0;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(state.width) + static_cast<size_t>(px);
            const float wet = state.wetness[idx];
            const float pigment = idx < state.pigment.size() ? state.pigment[idx] : 0.0f;
            const float thickness = source_thickness[sourcePixelIndex(px, py)];
            float advected_wet = 0.0f;
            float advected_pigment = 0.0f;
            float advected_thickness = 0.0f;
            const CompactVec4 current_source_pixel = source_pixels[sourcePixelIndex(px, py)];
            CompactVec4 advected_pixel = current_source_pixel;
            CompactVec4 advected_mask = mask_pixels
                ? source_mask_pixels[sourcePixelIndex(px, py)]
                : CompactVec4(128, 128, 128, 255);
            float runoff_alpha = 0.0f;
            float drip_head_alpha = 0.0f;
            float flow_scale = 0.0f;
            float downhill_px_x = 0.0f;
            float downhill_px_y = 0.0f;
            float local_slope = 0.0f;
            float advected_source_x = static_cast<float>(px);
            float advected_source_y = static_cast<float>(py);
            bool pixel_has_downhill = false;
            if (!flow_cache.infos.empty()) {
                const Vec2 pixel_uv(
                    state.width > 1 ? static_cast<float>(px) / static_cast<float>(state.width - 1) : 0.0f,
                    state.height > 1 ? 1.0f - static_cast<float>(py) / static_cast<float>(state.height - 1) : 0.0f);
                const int matched_index = findWetFlowTriangleIndex(pixel_uv, last_flow_info_index);
                if (matched_index >= 0) {
                    const WetFlowTriangleInfo& info = flow_cache.infos[matched_index];
                    downhill_px_x = info.flow.flow_x;
                    downhill_px_y = info.flow.flow_y;
                    local_slope = info.flow.slope;
                    pixel_has_downhill = true;
                    last_flow_info_index = matched_index;
                }
            }
            if (pixel_has_downhill) {
                const float effective_slope = std::max(0.18f, local_slope);
                const float local_flow_length = std::sqrt(downhill_px_x * downhill_px_x + downhill_px_y * downhill_px_y);
                const float local_flow_metric = std::clamp(1.0f + std::log1p(std::max(0.0f, local_flow_length)) * 0.85f, 1.0f, 4.0f);
                const float dir_x = local_flow_length > 1e-6f ? (downhill_px_x / local_flow_length) : 0.0f;
                const float dir_y = local_flow_length > 1e-6f ? (downhill_px_y / local_flow_length) : 0.0f;
                flow_scale = std::min(6.0f, runoff * effective_slope * local_flow_metric * std::max(wet, 0.05f) * dt * 60.0f * 1.65f);
                if (flow_scale > 0.001f) {
                    const float pixel_shift_x = dir_x * flow_scale;
                    const float pixel_shift_y = dir_y * flow_scale;
                    advected_source_x = static_cast<float>(px) - pixel_shift_x;
                    advected_source_y = static_cast<float>(py) - pixel_shift_y;
                    advected_pixel = sampleSourcePixel(source_pixels, advected_source_x, advected_source_y);
                    if (mask_pixels) {
                        advected_mask = sampleSourcePixel(source_mask_pixels, advected_source_x, advected_source_y);
                    }
                    advected_wet = sampleSourceScalar(source_wetness, advected_source_x, advected_source_y);
                    advected_pigment = sampleSourceScalar(source_pigment, advected_source_x, advected_source_y);
                    advected_thickness = sampleSourceScalar(source_thickness, advected_source_x, advected_source_y);
                    runoff_alpha = std::clamp(flow_scale * (0.26f + (local_flow_metric - 1.0f) * 0.12f), 0.0f, 1.0f);
                    const float drip_bias = std::clamp(advected_wet - wet * 0.25f, 0.0f, 1.0f);
                    drip_head_alpha = std::clamp(runoff_alpha * drip_head * (0.35f + drip_bias), 0.0f, 1.0f);
                }
            }

            if (wet <= 0.001f && advected_wet <= 0.001f) {
                continue;
            }

            const float pigment_presence = std::max(pigment, advected_pigment);
            const float runoff_wet_presence = std::max(wet, advected_wet);
            const float pigment_gate = oil_mode
                ? std::clamp((pigment_presence - 0.03f) / 0.18f, 0.0f, 1.0f)
                : std::clamp((pigment_presence - 0.008f) / 0.10f, 0.0f, 1.0f);
            const float carrier_gate = oil_mode
                ? pigment_gate
                : std::clamp(pigment_gate * 0.7f + runoff_wet_presence * 0.45f, 0.0f, 1.0f);
            const float color_runoff_alpha = runoff_alpha * carrier_gate;
            const float color_drip_head_alpha = drip_head_alpha * carrier_gate;
            const float mix_alpha = std::clamp(
                diffusion * wet * (oil_mode ? (0.10f + pigment_gate * 0.90f) : (0.22f + carrier_gate * 0.78f)) * dt * 60.0f,
                0.0f,
                1.0f);
            float sum_r = 0.0f;
            float sum_g = 0.0f;
            float sum_b = 0.0f;
            float sum_w = 0.0f;
            if (mix_alpha > 0.015f) {
                for (int oy = -1; oy <= 1; ++oy) {
                    const int sy = std::clamp(py + oy, 0, state.height - 1);
                    for (int ox = -1; ox <= 1; ++ox) {
                        const int sx = std::clamp(px + ox, 0, state.width - 1);
                        const size_t sidx = sourcePixelIndex(sx, sy);
                        const CompactVec4& sample = source_pixels[sidx];
                        const float sample_pigment = source_pigment[sidx];
                        const float sample_weight = sample_pigment * (0.10f + source_wetness[sidx]);
                        if (sample_weight <= 0.0005f) {
                            continue;
                        }
                        sum_r += srgbByteToLinear01(sample.r) * sample_weight;
                        sum_g += srgbByteToLinear01(sample.g) * sample_weight;
                        sum_b += srgbByteToLinear01(sample.b) * sample_weight;
                        sum_w += sample_weight;
                    }
                }
            }

            CompactVec4& dst = (*pixels)[idx];
            const uint8_t before_r = dst.r;
            const uint8_t before_g = dst.g;
            const uint8_t before_b = dst.b;
            const uint8_t before_a = dst.a;
            CompactVec4* dst_mask = mask_pixels ? &(*mask_pixels)[idx] : nullptr;
            const uint8_t before_mask = dst_mask ? dst_mask->r : 0;
            float next_thickness = thickness;
            float next_pigment = pigment;
            if (color_runoff_alpha > 0.001f) {
                blendBaseColorChannelLinear(dst.r, advected_pixel.r, color_runoff_alpha);
                blendBaseColorChannelLinear(dst.g, advected_pixel.g, color_runoff_alpha);
                blendBaseColorChannelLinear(dst.b, advected_pixel.b, color_runoff_alpha);
                if (color_drip_head_alpha > 0.001f) {
                    blendBaseColorChannelLinear(dst.r, advected_pixel.r, color_drip_head_alpha);
                    blendBaseColorChannelLinear(dst.g, advected_pixel.g, color_drip_head_alpha);
                    blendBaseColorChannelLinear(dst.b, advected_pixel.b, color_drip_head_alpha);
                }
                liftLayerAlpha(dst,
                               std::max(pigment, advected_pigment),
                               std::max(wet, advected_wet),
                               static_cast<float>(advected_pixel.a) / 255.0f,
                               std::max(color_runoff_alpha, color_drip_head_alpha));
            }
            next_thickness = std::clamp(
                next_thickness * (1.0f - color_runoff_alpha * (oil_mode ? 0.10f : 0.28f)) +
                    advected_thickness * color_runoff_alpha * (oil_mode ? 0.55f : 1.0f),
                0.0f,
                1.0f);
            next_pigment = std::clamp(
                std::max(
                    pigment * (oil_mode ? 0.985f : 0.992f),
                    advected_pigment * (oil_mode ? color_runoff_alpha : (0.28f + color_runoff_alpha * 0.72f))) -
                    dt * (oil_mode ? (0.10f + absorption * 0.20f) : (0.06f + absorption * 0.12f)),
                0.0f,
                1.0f);

            if (sum_w > 0.0f) {
                const uint8_t avg_r = linearToSrgbByte01(sum_r / sum_w);
                const uint8_t avg_g = linearToSrgbByte01(sum_g / sum_w);
                const uint8_t avg_b = linearToSrgbByte01(sum_b / sum_w);
                blendBaseColorChannelLinear(dst.r, avg_r, mix_alpha);
                blendBaseColorChannelLinear(dst.g, avg_g, mix_alpha);
                blendBaseColorChannelLinear(dst.b, avg_b, mix_alpha);
                liftLayerAlpha(dst,
                               std::max(next_pigment, pigment_presence),
                               std::max(wet, advected_wet),
                               1.0f,
                               mix_alpha);
                if (idx < state.thickness.size()) {
                    const float thickness_mix_alpha = std::clamp(mix_alpha * 0.32f, 0.0f, 1.0f);
                    const float local_thickness_target = std::clamp(
                        std::max(thickness, advected_thickness * (0.55f + color_runoff_alpha * 0.25f)),
                        0.0f,
                        1.0f);
                    next_thickness = std::clamp(
                        next_thickness + (local_thickness_target - next_thickness) * thickness_mix_alpha,
                        0.0f,
                        1.0f);
                }
                if (dst.r != before_r || dst.g != before_g || dst.b != before_b) {
                    dirty.expand(px, py, px, py);
                    changed = true;
                }
                if (dst.a != before_a) {
                    dirty.expand(px, py, px, py);
                    changed = true;
                }
                if (dst_mask && dst_mask->r != before_mask) {
                    dirty.expand(px, py, px, py);
                    mask_changed = true;
                }
            } else if (dst.a != before_a) {
                dirty.expand(px, py, px, py);
                changed = true;
            } else if (dst_mask && dst_mask->r != before_mask) {
                dirty.expand(px, py, px, py);
                mask_changed = true;
            }

            for (WetAuxChannelBinding& aux : aux_channels) {
                CompactVec4& aux_dst = (*aux.pixels)[idx];
                const CompactVec4 current_aux_source = aux.source_pixels[sourcePixelIndex(px, py)];
                CompactVec4 advected_aux_pixel = current_aux_source;
                if (flow_scale > 0.001f) {
                    advected_aux_pixel = sampleSourcePixel(aux.source_pixels, advected_source_x, advected_source_y);
                }

                float sum_scalar = 0.0f;
                float sum_scalar_weight = 0.0f;
                if (mix_alpha > 0.015f) {
                    for (int oy = -1; oy <= 1; ++oy) {
                        const int sy = std::clamp(py + oy, 0, state.height - 1);
                        for (int ox = -1; ox <= 1; ++ox) {
                            const int sx = std::clamp(px + ox, 0, state.width - 1);
                            const size_t sidx = sourcePixelIndex(sx, sy);
                            const float sample_pigment = source_pigment[sidx];
                            const float sample_weight = sample_pigment * (0.10f + source_wetness[sidx]);
                            if (sample_weight <= 0.0005f) {
                                continue;
                            }
                            sum_scalar += static_cast<float>(aux.source_pixels[sidx].r) * sample_weight;
                            sum_scalar_weight += sample_weight;
                        }
                    }
                }

                const uint8_t before_aux = aux_dst.r;
                if (color_runoff_alpha > 0.001f) {
                    blendScalarPixel(aux_dst, advected_aux_pixel, color_runoff_alpha);
                    if (color_drip_head_alpha > 0.001f) {
                        blendScalarPixel(aux_dst, advected_aux_pixel, color_drip_head_alpha);
                    }
                }
                if (sum_scalar_weight > 0.0f) {
                    const uint8_t avg_scalar = static_cast<uint8_t>(std::clamp(sum_scalar / sum_scalar_weight, 0.0f, 255.0f));
                    blendScalarPixel(aux_dst, avg_scalar, mix_alpha);
                }
                if (aux_dst.r != before_aux) {
                    dirty.expand(px, py, px, py);
                    aux.changed = true;
                }
            }

            const float absorbed = absorption * std::max(wet, advected_wet) * dt * 0.7f;
            next_thickness = std::max(0.0f, next_thickness - absorbed * 0.35f);

            if (idx < state.pigment.size()) {
                state.pigment[idx] = next_pigment;
            }
            if (idx < state.thickness.size()) {
                state.thickness[idx] = next_thickness;
            }

            if (dst_mask) {
                const uint8_t mask_value = encodeRaisedThickness(next_thickness);
                dst_mask->r = mask_value;
                dst_mask->g = mask_value;
                dst_mask->b = mask_value;
                dst_mask->a = 255;
                if (dst_mask->r != before_mask) {
                    dirty.expand(px, py, px, py);
                    mask_changed = true;
                }
            }

            state.wetness[idx] = std::max(0.0f, std::max(wet, advected_wet * runoff_alpha) - (dt / lifetime) - absorbed);
            if (state.wetness[idx] > 0.001f) {
                next_active.expand(px, py, px, py);
            }
            if (next_thickness > 0.001f) {
                next_active.expand(px, py, px, py);
            }

            if (pixel_has_downhill && flow_scale > 0.02f && runoff_alpha > 0.01f) {
                const float local_flow_length = std::sqrt(downhill_px_x * downhill_px_x + downhill_px_y * downhill_px_y);
                const float dir_x = local_flow_length > 1e-6f ? (downhill_px_x / local_flow_length) : 0.0f;
                const float dir_y = local_flow_length > 1e-6f ? (downhill_px_y / local_flow_length) : 0.0f;
                const float trail_distance = std::min(brush_radius_px * 0.6f + 2.0f, std::max(flow_scale * 0.95f, brush_radius_px * std::clamp(wet, 0.15f, 1.0f) * 0.22f));
                const float target_xf = static_cast<float>(px) + dir_x * trail_distance;
                const float target_yf = static_cast<float>(py) + dir_y * trail_distance;
                const int target_x = std::clamp(static_cast<int>(std::round(target_xf)), 0, state.width - 1);
                const int target_y = std::clamp(static_cast<int>(std::round(target_yf)), 0, state.height - 1);
                const size_t target_idx = static_cast<size_t>(target_y) * static_cast<size_t>(state.width) + static_cast<size_t>(target_x);
                if (target_idx != idx) {
                    CompactVec4& downstream = (*pixels)[target_idx];
                    const float transport_alpha = std::clamp(
                        runoff_alpha * (oil_mode ? 0.24f : 0.55f) + drip_head_alpha * (oil_mode ? 0.12f : 0.35f),
                        0.0f,
                        1.0f);
                    const float transported_wet = std::max(
                        wet * (oil_mode ? (0.18f + transport_alpha * 0.08f) : (0.55f + transport_alpha * 0.25f)),
                        advected_wet * runoff_alpha * (oil_mode ? 0.32f : 1.0f));
                    const float transported_pigment = std::clamp(
                        std::max(
                            pigment * (oil_mode ? (0.58f + transport_alpha * 0.22f) : (0.44f + transport_alpha * 0.28f)),
                            advected_pigment * (oil_mode ? color_runoff_alpha : (0.24f + color_runoff_alpha * 0.76f))),
                        0.0f,
                        1.0f);
                    const float pigment_transport_alpha = std::clamp(
                        oil_mode
                            ? (transport_alpha * (0.24f + transported_pigment * 0.76f) + color_runoff_alpha * 0.08f) *
                                std::clamp((transported_pigment - 0.015f) / 0.22f, 0.0f, 1.0f)
                            : transport_alpha * (0.18f + transported_pigment * 0.82f),
                        0.0f,
                        1.0f);
                    if (pigment_transport_alpha > 0.001f) {
                        blendBaseColorChannelLinear(downstream.r, current_source_pixel.r, pigment_transport_alpha);
                        blendBaseColorChannelLinear(downstream.g, current_source_pixel.g, pigment_transport_alpha);
                        blendBaseColorChannelLinear(downstream.b, current_source_pixel.b, pigment_transport_alpha);
                        liftLayerAlpha(downstream,
                                       transported_pigment,
                                       transported_wet,
                                       static_cast<float>(current_source_pixel.a) / 255.0f,
                                       pigment_transport_alpha);
                        dirty.expand(target_x, target_y, target_x, target_y);
                        changed = true;
                    }

                    if (mask_pixels && transported_pigment > (oil_mode ? 0.08f : 0.02f)) {
                        const float transported_thickness = std::clamp(
                            std::max(thickness, next_thickness) * transported_pigment *
                                (oil_mode ? (0.52f + pigment_transport_alpha * 0.30f) : (0.28f + pigment_transport_alpha * 0.45f)),
                            0.0f,
                            1.0f);
                        if (target_idx < state.thickness.size()) {
                            state.thickness[target_idx] = std::max(state.thickness[target_idx], transported_thickness);
                        }

                        const float pigment_mass = std::clamp(
                            transported_thickness * (0.45f + transported_wet * 0.35f) *
                            (0.35f + std::clamp(brush.paint_load, 0.0f, 1.0f) * 0.65f) *
                            (0.25f + std::clamp(brush.deposit_rate, 0.0f, 1.0f) * 0.75f),
                            0.0f,
                            1.0f);
                        const float pigment_gate = std::clamp((pigment_mass - 0.18f) / 0.42f, 0.0f, 1.0f);
                        const float thickness_gate = std::clamp((transported_thickness - 0.10f) / 0.35f, 0.0f, 1.0f);
                        const float wet_gate = std::clamp((transported_wet - 0.08f) / 0.30f, 0.0f, 1.0f);
                        const float dry_phase = std::clamp((0.32f - transported_wet) / 0.24f, 0.0f, 1.0f);
                        const float terminal_flow = std::clamp(1.0f - flow_scale / std::max(0.8f, brush_radius_px * 0.12f + 0.65f), 0.0f, 1.0f);
                        const float bead_strength = terminal_buildup * pigment_gate * thickness_gate * wet_gate * dry_phase * terminal_flow;
                        if (bead_strength > 0.001f) {
                            const int tail_back_step = std::max(1, static_cast<int>(std::round(1.0f + terminal_softness * 1.5f)));
                            const int side_step = terminal_softness > 0.45f ? 1 : 0;
                            auto applyBeadSample = [&](int bead_x, int bead_y, float weight_scale, float color_scale) {
                                if (bead_x < 0 || bead_x >= state.width || bead_y < 0 || bead_y >= state.height) {
                                    return;
                                }
                                const size_t bead_idx = static_cast<size_t>(bead_y) * static_cast<size_t>(state.width) + static_cast<size_t>(bead_x);
                                if (bead_idx >= state.thickness.size()) {
                                    return;
                                }
                                const float bead_add = bead_strength * weight_scale * 0.28f;
                                if (bead_add <= 0.0005f) {
                                    return;
                                }
                                CompactVec4& bead_pixel = (*pixels)[bead_idx];
                                const float bead_color_alpha = std::clamp(
                                    bead_strength * color_scale * (0.16f + pigment_transport_alpha * 0.36f),
                                    0.0f,
                                    1.0f);
                                if (bead_color_alpha > 0.0005f) {
                                    blendBaseColorChannelLinear(bead_pixel.r, downstream.r, bead_color_alpha);
                                    blendBaseColorChannelLinear(bead_pixel.g, downstream.g, bead_color_alpha);
                                    blendBaseColorChannelLinear(bead_pixel.b, downstream.b, bead_color_alpha);
                                    liftLayerAlpha(bead_pixel,
                                                   transported_pigment,
                                                   transported_wet,
                                                   static_cast<float>(downstream.a) / 255.0f,
                                                   bead_color_alpha);
                                    dirty.expand(bead_x, bead_y, bead_x, bead_y);
                                    changed = true;
                                }
                                state.thickness[bead_idx] = std::clamp(state.thickness[bead_idx] + bead_add, 0.0f, 1.0f);
                                next_active.expand(bead_x, bead_y, bead_x, bead_y);
                            };

                            applyBeadSample(target_x, target_y, 1.0f, 1.0f);
                            applyBeadSample(
                                std::clamp(target_x - static_cast<int>(std::round(dir_x * tail_back_step)), 0, state.width - 1),
                                std::clamp(target_y - static_cast<int>(std::round(dir_y * tail_back_step)), 0, state.height - 1),
                                0.38f + terminal_softness * 0.12f,
                                0.55f);
                            if (side_step > 0) {
                                const int side_x = static_cast<int>(std::round(-dir_y * side_step));
                                const int side_y = static_cast<int>(std::round(dir_x * side_step));
                                applyBeadSample(std::clamp(target_x + side_x, 0, state.width - 1), std::clamp(target_y + side_y, 0, state.height - 1), 0.12f, 0.22f);
                                applyBeadSample(std::clamp(target_x - side_x, 0, state.width - 1), std::clamp(target_y - side_y, 0, state.height - 1), 0.12f, 0.22f);
                            }
                        }
                        mask_changed = true;
                    }

                    state.wetness[target_idx] = std::max(
                        state.wetness[target_idx],
                        transported_wet * (0.16f + transported_pigment * 0.84f));
                    if (target_idx < state.pigment.size()) {
                        state.pigment[target_idx] = std::max(
                            state.pigment[target_idx],
                            transported_pigment * (0.70f + pigment_transport_alpha * 0.20f));
                    }
                    if (target_idx < state.thickness.size()) {
                        next_active.expand(target_x, target_y, target_x, target_y);
                    }
                    next_active.expand(target_x, target_y, target_x, target_y);

                    for (WetAuxChannelBinding& aux : aux_channels) {
                        CompactVec4& downstream_aux = (*aux.pixels)[target_idx];
                        const uint8_t before_aux = downstream_aux.r;
                        if (pigment_transport_alpha > 0.001f) {
                            const CompactVec4 current_aux_source = aux.source_pixels[sourcePixelIndex(px, py)];
                            blendScalarPixel(downstream_aux, current_aux_source, pigment_transport_alpha);
                        }
                        if (downstream_aux.r != before_aux) {
                            dirty.expand(target_x, target_y, target_x, target_y);
                            aux.changed = true;
                        }
                    }
                }
            }
        }
    }

    state.active_region = next_active;
    const bool aux_changed = std::any_of(aux_channels.begin(), aux_channels.end(), [](const WetAuxChannelBinding& aux) {
        return aux.changed;
    });
    if ((!changed && !mask_changed && !aux_changed) || dirty.empty()) {
        return false;
    }

    mirrorWetRegionAcrossSeams(*pixels, state.width, state.height, dirty, 0.35f);
    if (mask_pixels) {
        mirrorWetRegionAcrossSeams(*mask_pixels, state.width, state.height, dirty, 0.45f);
        mask_changed = true;
    }
    for (WetAuxChannelBinding& aux : aux_channels) {
        if (aux.changed && aux.pixels) {
            mirrorWetRegionAcrossSeams(*aux.pixels, state.width, state.height, dirty, 0.35f);
        }
    }

    if (state.uses_layers) {
        std::vector<PaintChannel> channels;
        channels.reserve(2 + aux_channels.size());
        channels.push_back(PaintChannel::BaseColor);
        if (mask_changed) {
            channels.push_back(PaintChannel::Mask);
        }
        for (const WetAuxChannelBinding& aux : aux_channels) {
            if (aux.changed) {
                channels.push_back(aux.channel);
            }
        }
        compositeAndUploadRegion(channels.data(), static_cast<int>(channels.size()), dirty);
    } else if (texture) {
        bindTextureSetToMaterial();
        if (texture->isUploaded()) {
            texture->upload_region_to_gpu(dirty.min_x, dirty.min_y,
                                          dirty.max_x - dirty.min_x + 1,
                                          dirty.max_y - dirty.min_y + 1);
        } else {
            texture->upload_to_gpu();
        }
        if (mask_changed && mask_texture) {
            if (mask_texture->isUploaded()) {
                mask_texture->upload_region_to_gpu(dirty.min_x, dirty.min_y,
                                                   dirty.max_x - dirty.min_x + 1,
                                                   dirty.max_y - dirty.min_y + 1);
            } else {
                mask_texture->upload_to_gpu();
            }
        }
        for (WetAuxChannelBinding& aux : aux_channels) {
            if (!aux.changed || !aux.texture) {
                continue;
            }
            if (aux.texture->isUploaded()) {
                aux.texture->upload_region_to_gpu(dirty.min_x, dirty.min_y,
                                                  dirty.max_x - dirty.min_x + 1,
                                                  dirty.max_y - dirty.min_y + 1);
            } else {
                aux.texture->upload_to_gpu();
            }
        }
    }

    if (mask_changed && auto_normal_from_height_enabled) {
        updateNormalFromHeightRegion(dirty, normal_strength);
    }
    return true;
}

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
    
    // Ensure the layer stack exists and bakes this channel's UVs BEFORE we zero the material UVs
    ensureLayerStack();
    
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

    // Since the PaintLayerStack baked the original texture's UV transformations into the pixel data,
    // we must reset the material's UV transforms to 1.0/0.0 to prevent the GPU from scaling the baked strokes.
    if (auto pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(material)) {
        pbsdf->textureTransform.scale = Vec2(1.0f, 1.0f);
        pbsdf->textureTransform.translation = Vec2(0.0f, 0.0f);
        pbsdf->textureTransform.tilingFactor = Vec2(1.0f, 1.0f);
        pbsdf->textureTransform.rotation_degrees = 0.0f;
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
        const int old_width = texture->width;
        const int old_height = texture->height;
        const bool was_uploaded = texture->isUploaded();
        resizeTexturePixels(*texture, resolution);
        if (texture->width != old_width || texture->height != old_height) {
            texture->markVulkanDirtyFull();
        }
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
    const float extent_scale = std::max(brushShapeAspectScale(brush), 1.0f / brushShapeAspectScale(brush));
    const float radius_x = std::max(0.1f, radius_px * kx);
    const float radius_y = std::max(0.1f, radius_px * ky);
    const float bound_radius_x = radius_x * extent_scale;
    const float bound_radius_y = radius_y * extent_scale;

    const int min_x = std::max(0, static_cast<int>(std::floor(center_x - bound_radius_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + bound_radius_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(center_y - bound_radius_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + bound_radius_y)));
    const float strength = std::clamp(brush.strength * brush.flow * dt * 60.0f, 0.0f, 1.0f);
    const CompactVec4 erase_pixel = defaultChannelPixel(channel);
    std::vector<CompactVec4> source_pixels;
    if (brush.tool == BrushTool::Soften || brush.tool == BrushTool::Sharpen) {
        source_pixels = texture->pixels;
    }

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - center_x;
            const float dy = (static_cast<float>(py) + 0.5f) - center_y;
            const BrushFootprintSample fp = sampleBrushFootprint(brush, dx, dy, radius_x, radius_y);
            const float nx = dx / radius_x;
            const float ny = dy / radius_y;
            const float dist_norm = fp.dist_norm;
            if (dist_norm > 1.0f) {
                continue;
            }

            const float alpha_mask = sampleBrushMask(brush, nx, ny);
            const float texture_alpha = sampleBrushPaintTextureAlpha(channel, brush, nx, ny);
            const CompactVec4 brush_pixel = makeBrushTexturePixel(channel, brush, nx, ny);
            const float weight = computeBrushWeightNormalized(dist_norm, radius_px,
                                     std::clamp(brush.falloff, 0.0f, 1.0f)) * alpha_mask * texture_alpha * strength;
            if (weight <= 0.001f) {
                continue;
            }

            CompactVec4& pixel = texture->pixels[static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px)];
            if (brush.tool == BrushTool::Erase) {
                if (channel == PaintChannel::Mask) {
                    blendHeightMaskPixelChannels(pixel, erase_pixel, weight);
                } else {
                    blendPixelChannel(pixel.r, erase_pixel.r, weight);
                    blendPixelChannel(pixel.g, erase_pixel.g, weight);
                    blendPixelChannel(pixel.b, erase_pixel.b, weight);
                }
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
                if (channel == PaintChannel::Mask) {
                    blendHeightMaskPixelChannels(pixel, CompactVec4(avg_r, avg_g, avg_b, 255), weight);
                } else {
                    blendPixelChannel(pixel.r, avg_r, weight);
                    blendPixelChannel(pixel.g, avg_g, weight);
                    blendPixelChannel(pixel.b, avg_b, weight);
                }
            } else if (brush.tool == BrushTool::Sharpen) {
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
                const float avg_r = sum.x / static_cast<float>(sample_count);
                const float avg_g = sum.y / static_cast<float>(sample_count);
                const float avg_b = sum.z / static_cast<float>(sample_count);

                const float factor = 1.5f;
                const float center_r = static_cast<float>(pixel.r);
                const float center_g = static_cast<float>(pixel.g);
                const float center_b = static_cast<float>(pixel.b);

                const uint8_t sharp_r = static_cast<uint8_t>(std::clamp(center_r + (center_r - avg_r) * factor, 0.0f, 255.0f));
                const uint8_t sharp_g = static_cast<uint8_t>(std::clamp(center_g + (center_g - avg_g) * factor, 0.0f, 255.0f));
                const uint8_t sharp_b = static_cast<uint8_t>(std::clamp(center_b + (center_b - avg_b) * factor, 0.0f, 255.0f));

                if (channel == PaintChannel::Mask) {
                    blendHeightMaskPixelChannels(pixel, CompactVec4(sharp_r, sharp_g, sharp_b, 255), weight);
                } else {
                    blendPixelChannel(pixel.r, sharp_r, weight);
                    blendPixelChannel(pixel.g, sharp_g, weight);
                    blendPixelChannel(pixel.b, sharp_b, weight);
                }
            } else if (brush.tool == BrushTool::Dodge) {
                if (channel == PaintChannel::Mask) {
                    float val = static_cast<float>(pixel.r) / 255.0f;
                    val = std::min(val + weight * 0.5f, 1.0f);
                    const uint8_t byte_val = static_cast<uint8_t>(val * 255.0f + 0.5f);
                    pixel.r = byte_val; pixel.g = byte_val; pixel.b = byte_val;
                } else {
                    float r_f = static_cast<float>(pixel.r) / 255.0f;
                    float g_f = static_cast<float>(pixel.g) / 255.0f;
                    float b_f = static_cast<float>(pixel.b) / 255.0f;
                    r_f = r_f + (1.0f - r_f) * weight * 0.5f;
                    g_f = g_f + (1.0f - g_f) * weight * 0.5f;
                    b_f = b_f + (1.0f - b_f) * weight * 0.5f;
                    pixel.r = static_cast<uint8_t>(std::clamp(r_f * 255.0f + 0.5f, 0.0f, 255.0f));
                    pixel.g = static_cast<uint8_t>(std::clamp(g_f * 255.0f + 0.5f, 0.0f, 255.0f));
                    pixel.b = static_cast<uint8_t>(std::clamp(b_f * 255.0f + 0.5f, 0.0f, 255.0f));
                }
            } else if (brush.tool == BrushTool::Burn) {
                if (channel == PaintChannel::Mask) {
                    float val = static_cast<float>(pixel.r) / 255.0f;
                    val = std::max(val * (1.0f - weight * 0.5f), 0.0f);
                    const uint8_t byte_val = static_cast<uint8_t>(val * 255.0f + 0.5f);
                    pixel.r = byte_val; pixel.g = byte_val; pixel.b = byte_val;
                } else {
                    const float factor = 1.0f - weight * 0.5f;
                    pixel.r = static_cast<uint8_t>(std::clamp(static_cast<float>(pixel.r) * factor, 0.0f, 255.0f));
                    pixel.g = static_cast<uint8_t>(std::clamp(static_cast<float>(pixel.g) * factor, 0.0f, 255.0f));
                    pixel.b = static_cast<uint8_t>(std::clamp(static_cast<float>(pixel.b) * factor, 0.0f, 255.0f));
                }
            } else {
                if (channel == PaintChannel::Mask) {
                    blendHeightMaskPixelChannels(pixel, brush_pixel, weight);
                } else {
                    blendPixelChannel(pixel.r, brush_pixel.r, weight);
                    blendPixelChannel(pixel.g, brush_pixel.g, weight);
                    blendPixelChannel(pixel.b, brush_pixel.b, weight);
                }
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
    const float extent_scale = std::max(brushShapeAspectScale(brush), 1.0f / brushShapeAspectScale(brush));
    const float radius_x = std::max(0.1f, radius_px * kx);
    const float radius_y = std::max(0.1f, radius_px * ky);
    const float bound_radius_x = radius_x * extent_scale;
    const float bound_radius_y = radius_y * extent_scale;

    const int min_x = std::max(0, static_cast<int>(std::floor(center_x - bound_radius_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + bound_radius_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(center_y - bound_radius_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + bound_radius_y)));
    const float strength = std::clamp(brush.strength * brush.flow * dt * 60.0f, 0.0f, 1.0f);
    const std::vector<CompactVec4> source_pixels = texture->pixels;

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - center_x;
            const float dy = (static_cast<float>(py) + 0.5f) - center_y;
            const BrushFootprintSample fp = sampleBrushFootprint(brush, dx, dy, radius_x, radius_y);
            const float nx = dx / radius_x;
            const float ny = dy / radius_y;
            const float dist_norm = fp.dist_norm;
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
            if (channel == PaintChannel::Mask) {
                blendHeightMaskPixelChannels(dst, src, weight);
            } else {
                blendPixelChannel(dst.r, src.r, weight);
                blendPixelChannel(dst.g, src.g, weight);
                blendPixelChannel(dst.b, src.b, weight);
            }
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

    std::shared_ptr<Texture> source_normal = set->getSourceTexture(PaintChannel::Normal);
    const bool has_source_normal =
        source_normal &&
        source_normal->is_loaded() &&
        source_normal->width > 0 &&
        source_normal->height > 0 &&
        !source_normal->pixels.empty();
    const Vec3 flat_normal(0.0f, 0.0f, 1.0f);

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

bool MeshPaintAdapter::updateNormalFromHeightRegion(const PaintDirtyRect& dirty, float strength) {
    if (!isValid() || dirty.empty()) return false;
    PaintTextureSet* set = getTextureSet();
    if (!set) return false;

    std::shared_ptr<Texture> height_tex = set->getTexture(PaintChannel::Mask);
    std::shared_ptr<Texture> normal_tex = set->getTexture(PaintChannel::Normal);
    if (!height_tex || !normal_tex) return false;
    if (height_tex->pixels.empty() || normal_tex->pixels.empty()) return false;

    const int width = height_tex->width;
    const int height = height_tex->height;
    if (normal_tex->width != width || normal_tex->height != height) return false;
    const size_t expected_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (normal_tex->pixels.size() < expected_pixels || height_tex->pixels.size() < expected_pixels) return false;

    std::shared_ptr<Texture> source_normal = set->getSourceTexture(PaintChannel::Normal);
    const bool has_source_normal =
        source_normal &&
        source_normal->is_loaded() &&
        source_normal->width > 0 &&
        source_normal->height > 0 &&
        !source_normal->pixels.empty();
    const Vec3 flat_normal(0.0f, 0.0f, 1.0f);
    const int active_uv_set = getTarget().uv_set;

    const int min_x = std::max(0, dirty.min_x - 1);
    const int max_x = std::min(width - 1, dirty.max_x + 1);
    const int min_y = std::max(0, dirty.min_y - 1);
    const int max_y = std::min(height - 1, dirty.max_y + 1);

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px);
            if (idx >= normal_tex->pixels.size()) continue;

            Vec3 baked_bump = triangle_ ? buildNormalFromHeightGeometryAware(height_tex, *triangle_, active_uv_set, px, py, strength)
                                        : buildNormalFromHeight(height_tex, px, py, strength);

            Vec3 base_normal = flat_normal;
            if (has_source_normal) {
                const float u = width > 1 ? static_cast<float>(px) / static_cast<float>(width - 1) : 0.0f;
                const float v = height > 1 ? 1.0f - static_cast<float>(py) / static_cast<float>(height - 1) : 0.0f;
                base_normal = decodeNormalPixel(sampleTexturePixelBilinear(
                    source_normal->pixels,
                    source_normal->width,
                    source_normal->height,
                    u,
                    v));
            }

            normal_tex->pixels[idx] = encodeNormalPixel(combineNormalsPD(base_normal, baked_bump));
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
    
    PaintLayerStack::PaintUVParams uv_params;
    if (triangle_ && triangle_->getMaterial()) {
        std::shared_ptr<Material> mat = triangle_->getMaterial();
        if (auto pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(mat)) {
            uv_params.scale_x = pbsdf->textureTransform.scale.u;
            uv_params.scale_y = pbsdf->textureTransform.scale.v;
            uv_params.offset_x = pbsdf->textureTransform.translation.u;
            uv_params.offset_y = pbsdf->textureTransform.translation.v;
            uv_params.tiling_x = pbsdf->textureTransform.tilingFactor.u;
            uv_params.tiling_y = pbsdf->textureTransform.tilingFactor.v;
            uv_params.rotation_degrees = pbsdf->textureTransform.rotation_degrees;
            uv_params.wrap_mode = static_cast<int>(pbsdf->textureTransform.wrapMode);
        }
    }

    if (it != scene_->mesh_paint_layer_stacks.end()) {
        PaintLayerStack& existing = it->second;
        PaintTextureSet* tex_set = getTextureSet();

        // Ensure subsequent channels use the UV params from the original bake session
        // to prevent later channels from ignoring the UV transform after bindTextureSetToMaterial zeroes it out.
        PaintLayerStack::PaintUVParams use_uv = existing.has_baked_uv_params ? existing.baked_uv_params : uv_params;

        if (tex_set && tex_set->initialized && existing.layerCount() > 0) {
            PaintLayerData* base = existing.layerAt(0);
            if (base) {
                for (int ch = 0; ch < static_cast<int>(kPaintChannelCount); ++ch) {
                    const auto channel = static_cast<PaintChannel>(ch);
                    if (base->hasPixels(channel)) {
                        continue;
                    }

                    std::shared_ptr<Texture> texture = tex_set->getTexture(channel);
                    if (!texture || !texture->is_loaded() || texture->pixels.empty()) {
                        continue;
                    }
                    
                    // We just call seedFromTextureSet to properly handle UV baking for any newly added channels
                    existing.seedFromTextureSet(*tex_set, use_uv);
                    for (int c = 0; c < static_cast<int>(kPaintChannelCount); ++c) {
                        existing.flattenChannelInto(static_cast<PaintChannel>(c), *tex_set);
                    }
                    break;
                }
            }
        }

        // Defensive: if the stack was created before the texture set was ready
        // (e.g. base layer has no pixels for any channel), try to seed now.
        if (existing.layerCount() == 1) {
            PaintLayerData* base = existing.layerAt(0);
            if (base) {
                bool has_any = false;
                for (int ch = 0; ch < static_cast<int>(kPaintChannelCount) && !has_any; ++ch)
                    has_any = base->hasPixels(static_cast<PaintChannel>(ch));
                if (!has_any) {
                    PaintTextureSet* tex_set = getTextureSet();
                    if (tex_set && tex_set->initialized) {
                        existing.seedFromTextureSet(*tex_set, use_uv);
                        for (int c = 0; c < static_cast<int>(kPaintChannelCount); ++c) {
                            existing.flattenChannelInto(static_cast<PaintChannel>(c), *tex_set);
                        }
                    }
                }
            }
        }
        return existing;
    }

    // Create a new layer stack and seed it from the existing texture set if available.
    PaintLayerStack stack;
    stack.baked_uv_params = uv_params;
    stack.has_baked_uv_params = true;
    
    PaintTextureSet* tex_set = getTextureSet();
    if (tex_set && tex_set->initialized) {
        stack.seedFromTextureSet(*tex_set, uv_params);
        for (int c = 0; c < static_cast<int>(kPaintChannelCount); ++c) {
            stack.flattenChannelInto(static_cast<PaintChannel>(c), *tex_set);
        }
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
    const float extent_scale = std::max(brushShapeAspectScale(brush), 1.0f / brushShapeAspectScale(brush));
    const float radius_x = std::max(0.1f, radius_px * kx);
    const float radius_y = std::max(0.1f, radius_px * ky);
    const float bound_radius_x = radius_x * extent_scale;
    const float bound_radius_y = radius_y * extent_scale;

    const int min_x = std::max(0, static_cast<int>(std::floor(center_x - bound_radius_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + bound_radius_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(center_y - bound_radius_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + bound_radius_y)));
    const float strength = std::clamp(brush.strength * brush.flow * dt * 60.0f, 0.0f, 1.0f);
    const bool scalar_channel = isScalarMaterialChannel(channel);

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - center_x;
            const float dy = (static_cast<float>(py) + 0.5f) - center_y;
            const BrushFootprintSample fp = sampleBrushFootprint(brush, dx, dy, radius_x, radius_y);
            const float nx = dx / radius_x;
            const float ny = dy / radius_y;
            const float dist_norm = fp.dist_norm;
            if (dist_norm > 1.0f) continue;

            const float alpha_mask = sampleBrushMask(brush, nx, ny);
            const float texture_alpha = sampleBrushPaintTextureAlpha(channel, brush, nx, ny);
            const CompactVec4 brush_pixel = makeBrushTexturePixel(channel, brush, nx, ny);
            const float weight = computeBrushWeightNormalized(dist_norm, radius_px,
                                    std::clamp(brush.falloff, 0.0f, 1.0f)) * alpha_mask * texture_alpha * strength;
            if (weight <= 0.001f) continue;

            const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px);
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
            } else if (brush.tool == BrushTool::Sharpen) {
                Vec3 sum(0.0f, 0.0f, 0.0f);
                int sample_count = 0;
                for (int oy = -1; oy <= 1; ++oy) {
                    const int sy = std::clamp(py + oy, 0, height - 1);
                    for (int ox = -1; ox <= 1; ++ox) {
                        const int sx = std::clamp(px + ox, 0, width - 1);
                        const CompactVec4& src = pixels[static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx)];
                        sum.x += static_cast<float>(src.r);
                        sum.y += static_cast<float>(src.g);
                        sum.z += static_cast<float>(src.b);
                        ++sample_count;
                    }
                }
                const float avg_r = sum.x / static_cast<float>(sample_count);
                const float avg_g = sum.y / static_cast<float>(sample_count);
                const float avg_b = sum.z / static_cast<float>(sample_count);

                const float factor = 1.5f;
                const float center_r = static_cast<float>(pixel.r);
                const float center_g = static_cast<float>(pixel.g);
                const float center_b = static_cast<float>(pixel.b);

                const uint8_t sharp_r = static_cast<uint8_t>(std::clamp(center_r + (center_r - avg_r) * factor, 0.0f, 255.0f));
                const uint8_t sharp_g = static_cast<uint8_t>(std::clamp(center_g + (center_g - avg_g) * factor, 0.0f, 255.0f));
                const uint8_t sharp_b = static_cast<uint8_t>(std::clamp(center_b + (center_b - avg_b) * factor, 0.0f, 255.0f));

                blendPixelChannel(pixel.r, sharp_r, weight);
                blendPixelChannel(pixel.g, sharp_g, weight);
                blendPixelChannel(pixel.b, sharp_b, weight);
            } else if (brush.tool == BrushTool::Dodge) {
                float r_f = static_cast<float>(pixel.r) / 255.0f;
                float g_f = static_cast<float>(pixel.g) / 255.0f;
                float b_f = static_cast<float>(pixel.b) / 255.0f;
                r_f = r_f + (1.0f - r_f) * weight * 0.5f;
                g_f = g_f + (1.0f - g_f) * weight * 0.5f;
                b_f = b_f + (1.0f - b_f) * weight * 0.5f;
                pixel.r = static_cast<uint8_t>(std::clamp(r_f * 255.0f + 0.5f, 0.0f, 255.0f));
                pixel.g = static_cast<uint8_t>(std::clamp(g_f * 255.0f + 0.5f, 0.0f, 255.0f));
                pixel.b = static_cast<uint8_t>(std::clamp(b_f * 255.0f + 0.5f, 0.0f, 255.0f));
                
                const float src_a = std::clamp(weight, 0.0f, 1.0f);
                const float dst_a = static_cast<float>(pixel.a) / 255.0f;
                const float out_a = dst_a + (1.0f - dst_a) * src_a * 0.5f;
                pixel.a = static_cast<uint8_t>(std::clamp(out_a * 255.0f, 0.0f, 255.0f));
            } else if (brush.tool == BrushTool::Burn) {
                const float factor = 1.0f - weight * 0.5f;
                pixel.r = static_cast<uint8_t>(std::clamp(static_cast<float>(pixel.r) * factor, 0.0f, 255.0f));
                pixel.g = static_cast<uint8_t>(std::clamp(static_cast<float>(pixel.g) * factor, 0.0f, 255.0f));
                pixel.b = static_cast<uint8_t>(std::clamp(static_cast<float>(pixel.b) * factor, 0.0f, 255.0f));
                
                const float src_a = std::clamp(weight, 0.0f, 1.0f);
                const float dst_a = static_cast<float>(pixel.a) / 255.0f;
                const float out_a = dst_a + (1.0f - dst_a) * src_a * 0.5f;
                pixel.a = static_cast<uint8_t>(std::clamp(out_a * 255.0f, 0.0f, 255.0f));
            } else if (scalar_channel && layer->meta.blend_mode == LayerBlendMode::Normal) {
                const float layer_opacity = std::clamp(layer->meta.opacity, 0.0f, 1.0f);
                const float target_value = static_cast<float>(brush_pixel.r) / 255.0f;
                const float under_value = compositeScalarBelowLayer(stack, layer_index, channel, idx);
                const float current_value = compositeScalarLayerValue(under_value, pixel, layer_opacity, LayerBlendMode::Normal);
                const float next_value = std::clamp(current_value + (target_value - current_value) * weight, 0.0f, 1.0f);

                float desired_alpha = static_cast<float>(pixel.a) / 255.0f;
                const float denom = target_value - under_value;
                if (layer_opacity > 1e-6f) {
                    if (std::abs(denom) > 1e-6f) {
                        const float desired_sa = std::clamp((next_value - under_value) / denom, 0.0f, layer_opacity);
                        desired_alpha = std::clamp(desired_sa / layer_opacity, 0.0f, 1.0f);
                    } else {
                        desired_alpha = std::clamp(desired_alpha + (1.0f - desired_alpha) * weight, 0.0f, 1.0f);
                    }
                }

                pixel.r = brush_pixel.r;
                pixel.g = brush_pixel.g;
                pixel.b = brush_pixel.b;
                pixel.a = static_cast<uint8_t>(std::clamp(desired_alpha * 255.0f, 0.0f, 255.0f));
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
    const float extent_scale = std::max(brushShapeAspectScale(brush), 1.0f / brushShapeAspectScale(brush));
    const float radius_x = std::max(0.1f, radius_px * kx);
    const float radius_y = std::max(0.1f, radius_px * ky);
    const float bound_radius_x = radius_x * extent_scale;
    const float bound_radius_y = radius_y * extent_scale;

    const int min_x = std::max(0, static_cast<int>(std::floor(center_x - bound_radius_x)));
    const int max_x = std::min(width - 1, static_cast<int>(std::ceil(center_x + bound_radius_x)));
    const int min_y = std::max(0, static_cast<int>(std::floor(center_y - bound_radius_y)));
    const int max_y = std::min(height - 1, static_cast<int>(std::ceil(center_y + bound_radius_y)));
    const float strength = std::clamp(brush.strength * brush.flow * dt * 60.0f, 0.0f, 1.0f);

    // Source data: read from the layer's own pixels (clone within layer).
    const std::vector<CompactVec4> source_pixels = pixels;

    bool changed = false;
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            const float dx = (static_cast<float>(px) + 0.5f) - center_x;
            const float dy = (static_cast<float>(py) + 0.5f) - center_y;
            const BrushFootprintSample fp = sampleBrushFootprint(brush, dx, dy, radius_x, radius_y);
            const float nx = dx / radius_x;
            const float ny = dy / radius_y;
            const float dist_norm = fp.dist_norm;
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
