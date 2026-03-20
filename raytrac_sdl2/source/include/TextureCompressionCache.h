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
