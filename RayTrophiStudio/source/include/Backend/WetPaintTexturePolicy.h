#pragma once

#include "TextureCompression.h"

namespace Backend {

// GPU-authored Paint channel pages need a writable RGBA8 image, not a BC4/BC5
// cache or an R8/RG8 upload. Other imported/generated textures keep their policy.
inline bool isWetPaintChannelTexture(const Texture* texture) {
    return texture && texture->name.rfind("generated/wet/", 0) == 0;
}

inline bool allowCompactPaintTexture(const Texture* texture) {
    return !isWetPaintChannelTexture(texture);
}

inline TextureCompressionPlan paintTextureCompressionPlan(const Texture* texture,
                                                          TextureType type) {
    return isWetPaintChannelTexture(texture) ? TextureCompressionPlan{}
                                            : buildTextureCompressionPlan(texture, type);
}

template <class Map>
inline auto findPaintTextureImage(Map& images, const Texture* texture) {
    // The material resolver keys scalar/normal views with forceLinear and
    // preferSingleChannel flags. Base-colour-only lookup misses these images.
    for (uint64_t flags : {3ull, 1ull, 0ull, 2ull}) {
        auto found = images.find((static_cast<uint64_t>(texture->m_uid) << 2u) | flags);
        if (found != images.end() && found->second > 0) return found;
    }
    return images.end();
}

} // namespace Backend
