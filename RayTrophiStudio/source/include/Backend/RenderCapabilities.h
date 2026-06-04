#ifndef BACKEND_RENDER_CAPABILITIES_H
#define BACKEND_RENDER_CAPABILITIES_H

#include <cstdint>

namespace Backend {

struct RenderBackendCapabilities {
    bool hasVulkan = false;
    bool hasMaterialPreview = false;
    bool hasVulkanRT = false;
    bool hasOptix = false;
    bool hasCUDA = false;
};

enum class TextureConsumer : uint32_t {
    None = 0,
    RasterPreview = 1u << 0,
    VulkanRT = 1u << 1,
    Optix = 1u << 2
};

inline TextureConsumer operator|(TextureConsumer lhs, TextureConsumer rhs) {
    return static_cast<TextureConsumer>(
        static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

inline TextureConsumer operator&(TextureConsumer lhs, TextureConsumer rhs) {
    return static_cast<TextureConsumer>(
        static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

inline TextureConsumer& operator|=(TextureConsumer& lhs, TextureConsumer rhs) {
    lhs = lhs | rhs;
    return lhs;
}

inline bool hasTextureConsumer(TextureConsumer mask, TextureConsumer bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

} // namespace Backend

#endif // BACKEND_RENDER_CAPABILITIES_H
