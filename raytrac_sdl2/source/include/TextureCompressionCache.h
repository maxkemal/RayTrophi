#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "Texture.h"
#include "TextureCompression.h"

struct TextureCompressedCacheCandidate {
    std::filesystem::path ddsPath;
    TextureCompressionTarget target = TextureCompressionTarget::None;
    bool srgb = false;
};

std::optional<std::filesystem::path> getProjectTextureCacheDirectory();

bool clearProjectTextureCache(std::string* outReason = nullptr);

std::optional<TextureCompressedCacheCandidate> findCompressedTextureCacheCandidate(
    const Texture& tex,
    TextureType type,
    bool srgb);

bool tryBuildCompressedTextureCache(
    const Texture& tex,
    TextureType type,
    bool srgb,
    TextureCompressedCacheCandidate& outCandidate,
    std::string* outReason = nullptr);

// Returns "BC7", "BC5", or "BC4" if a managed cache DDS exists for this texture+type,
// otherwise empty. Checks managed cache only — suitable for per-frame UI queries
// when results are memoised by the caller.
std::string queryManagedCacheTag(const Texture& tex, TextureType type);
