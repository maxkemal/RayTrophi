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

// Removes any managed DDS cache entries that belong to this texture (matched by
// filename stem). Call this when the texture's pixels are mutated in memory
// (mesh paint, autoMask, etc.) so the next session re-bakes from the new pixel
// data instead of resurrecting the pre-edit compressed bytes from disk.
// Adjacent (user-supplied) DDS files next to the source asset are NOT touched.
// Returns the number of files removed (0 when nothing matched).
size_t invalidateManagedTextureCacheForTexture(const Texture& tex);
