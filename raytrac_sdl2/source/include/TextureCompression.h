#pragma once

#include "Texture.h"

enum class TextureSemantic : uint8_t {
    Unknown = 0,
    Albedo,
    Normal,
    Scalar,
    Emission,
    HDR
};

enum class TextureCompressionTarget : uint8_t {
    None = 0,
    BC4,
    BC5,
    BC7
};

struct TextureCompressionPlan {
    TextureSemantic semantic = TextureSemantic::Unknown;
    TextureCompressionTarget preferredTarget = TextureCompressionTarget::None;
    bool preferSingleChannelFallback = false;
    // Which RGBA channel to extract for BC4 (0=R, 1=G, 2=B, 3=A).
    // Matches the channel the shader reads in pbr_texture_policy.glsl:
    //   roughness → .g (ORM packed) or .r (grayscale)
    //   metallic  → .b (ORM packed) or .r (grayscale)
    //   opacity   → .a (RGBA source) or .r (grayscale mask)
    //   AO/transmission → .r
    uint8_t sourceChannel = 0;
};

inline TextureCompressionPlan buildTextureCompressionPlan(const Texture* tex, TextureType type) {
    TextureCompressionPlan plan{};
    if (!tex) return plan;

    switch (type) {
        case TextureType::Albedo:
            plan.semantic = TextureSemantic::Albedo;
            plan.preferredTarget = TextureCompressionTarget::BC7;
            break;
        case TextureType::Emission:
            plan.semantic = tex->is_hdr ? TextureSemantic::HDR : TextureSemantic::Emission;
            // HDR emission stays uncompressed (float data); LDR emission → BC7
            plan.preferredTarget = tex->is_hdr ? TextureCompressionTarget::None : TextureCompressionTarget::BC7;
            break;
        case TextureType::Normal:
            plan.semantic = TextureSemantic::Normal;
            plan.preferredTarget = TextureCompressionTarget::BC5;
            break;
        case TextureType::Roughness:
            plan.semantic = TextureSemantic::Scalar;
            // Always BC4 (4 bpp single-channel) regardless of source layout:
            //   grayscale source  → extract .r (sourceChannel=0)
            //   ORM-packed source → extract .g (sourceChannel=1)
            // BC4 stores the chosen channel exactly in R. The shader sets bit 9
            // (VK_MAT_FLAG_ROUGHNESS_IN_R) and reads .r. This eliminates the BC7
            // multi-channel approximation that produced wrong values on metallic-
            // looking inputs, and avoids "gray-but-not-quite-gray" detection
            // failures (JPEG/codec artifacts where R≈G≈B but not equal) silently
            // routing to the BC7 path.
            plan.preferredTarget = TextureCompressionTarget::BC4;
            plan.preferSingleChannelFallback = !tex->has_alpha;
            plan.sourceChannel = tex->is_gray_scale ? 0 : 1;
            break;
        case TextureType::Metallic:
            plan.semantic = TextureSemantic::Scalar;
            // Same as Roughness; ORM puts metallic in .b (sourceChannel=2).
            plan.preferredTarget = TextureCompressionTarget::BC4;
            plan.preferSingleChannelFallback = !tex->has_alpha;
            plan.sourceChannel = tex->is_gray_scale ? 0 : 2;
            break;
        case TextureType::Opacity:
            plan.semantic = TextureSemantic::Scalar;
            plan.preferredTarget = TextureCompressionTarget::BC4;
            plan.preferSingleChannelFallback = tex->is_gray_scale && !tex->has_alpha;
            // RGBA texture: opacity lives in the alpha channel.
            plan.sourceChannel = tex->has_alpha ? 3 : 0;
            break;
        case TextureType::AO:
        case TextureType::Transmission:
        case TextureType::Specular:
            plan.semantic = TextureSemantic::Scalar;
            plan.preferredTarget = TextureCompressionTarget::BC4;
            plan.preferSingleChannelFallback = tex->is_gray_scale && !tex->has_alpha;
            plan.sourceChannel = 0; // AO, transmission and specular are always in R
            break;
        default:
            break;
    }

    return plan;
}
