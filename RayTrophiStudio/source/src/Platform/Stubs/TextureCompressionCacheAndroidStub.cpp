// Android stub for the BC7/BC5/BC4 texture cache (replaces Utils/TextureCompressionCache.cpp,
// which is built on DirectXTex and has no Android port). Textures are used uncompressed.
// Nothing fakes success: builders return false with an explicit reason, lookups return "no cache".
#include "TextureCompressionCache.h"

#include <cstdio>
#include <mutex>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace {
const char* kReason = "texture compression cache is not available on Android";

void logOnce() {
    static std::once_flag flag;
    std::call_once(flag, [] {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_WARN, "RayTrophi", "[TextureCache] %s", kReason);
#else
        std::fprintf(stderr, "[TextureCache] %s\n", kReason);
#endif
    });
}
} // namespace

std::optional<std::filesystem::path> getProjectTextureCacheDirectory() {
    return std::nullopt;
}

bool clearProjectTextureCache(std::string* outReason) {
    logOnce();
    if (outReason) *outReason = kReason;
    return false;
}

std::optional<TextureCompressedCacheCandidate> findCompressedTextureCacheCandidate(
    const Texture&, TextureType, bool) {
    return std::nullopt;
}

bool tryBuildCompressedTextureCache(const Texture&, TextureType, bool,
                                    TextureCompressedCacheCandidate&, std::string* outReason) {
    logOnce();
    if (outReason) *outReason = kReason;
    return false;
}

std::string queryManagedCacheTag(const Texture&, TextureType) {
    return std::string();
}

size_t invalidateManagedTextureCacheForTexture(const Texture&) {
    return 0;
}
