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
};

inline TextureCompressionPlan buildTextureCompressionPlan(const Texture* tex, TextureType type) {
    TextureCompressionPlan plan{};
    if (!tex) return plan;

    switch (type) {
        case TextureType::Albedo:
            plan.semantic = TextureSemantic::Albedo;
            // Keep color textures uncompressed for now.
            // Existing BC7 cache can shift color/gamma in Vulkan, so only scalar maps stay compressed.
            plan.preferredTarget = TextureCompressionTarget::None;
            break;
        case TextureType::Emission:
            plan.semantic = tex->is_hdr ? TextureSemantic::HDR : TextureSemantic::Emission;
            plan.preferredTarget = TextureCompressionTarget::None;
            break;
        case TextureType::Normal:
            plan.semantic = TextureSemantic::Normal;
            plan.preferredTarget = TextureCompressionTarget::BC5;
            break;
        case TextureType::Roughness:
        case TextureType::Metallic:
        case TextureType::AO:
        case TextureType::Transmission:
        case TextureType::Opacity:
            plan.semantic = TextureSemantic::Scalar;
            plan.preferredTarget = TextureCompressionTarget::BC4;
            plan.preferSingleChannelFallback = tex->is_gray_scale && !tex->has_alpha;
            break;
        default:
            break;
    }

    return plan;
}
