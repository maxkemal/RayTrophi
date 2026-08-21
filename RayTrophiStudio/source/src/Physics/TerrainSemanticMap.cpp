#include "TerrainSemanticMap.h"

#include "TerrainSystem.h"
#include "Texture.h"
#include "globals.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace TerrainSemanticMap {

bool publish(TerrainObject& terrain,
             const NodeSystem::Image2DData& image,
             std::string& error) {
    if (!image.isValid() || image.channels != 4) {
        error = "Semantic map input must be a valid RGBA image";
        return false;
    }
    const int width = terrain.paintGridWidth();
    const int height = terrain.paintGridHeight();
    if (width < 2 || height < 2) {
        error = "Terrain paint resolution is invalid";
        return false;
    }
    if (!terrain.surfaceSemanticMap) {
        terrain.surfaceSemanticMap = std::make_shared<Texture>(
            nullptr, TextureType::Unknown, "TerrainSemanticMap");
    }
    Texture& texture = *terrain.surfaceSemanticMap;
    texture.width = width;
    texture.height = height;
    texture.pixels.resize(static_cast<size_t>(width) * height);

    for (int y = 0; y < height; ++y) {
        const int terrainRow = (height - 1) - y;
        const float sourceY = std::clamp(
            ((static_cast<float>(terrainRow) + 0.5f) * image.height / height) - 0.5f,
            0.0f, static_cast<float>(image.height - 1));
        const int y0 = static_cast<int>(std::floor(sourceY));
        const int y1 = (std::min)(y0 + 1, image.height - 1);
        const float ty = sourceY - y0;
        for (int x = 0; x < width; ++x) {
            const float sourceX = std::clamp(
                ((static_cast<float>(x) + 0.5f) * image.width / width) - 0.5f,
                0.0f, static_cast<float>(image.width - 1));
            const int x0 = static_cast<int>(std::floor(sourceX));
            const int x1 = (std::min)(x0 + 1, image.width - 1);
            const float tx = sourceX - x0;
            CompactVec4& pixel = texture.pixels[static_cast<size_t>(y) * width + x];
            uint8_t* destination[4] = {&pixel.r, &pixel.g, &pixel.b, &pixel.a};
            for (int channel = 0; channel < 4; ++channel) {
                const auto sample = [&](int sx, int sy) {
                    return (*image.data)[
                        (static_cast<size_t>(sy) * image.width + sx) * 4 + channel];
                };
                const float top = sample(x0, y0) + (sample(x1, y0) - sample(x0, y0)) * tx;
                const float bottom = sample(x0, y1) + (sample(x1, y1) - sample(x0, y1)) * tx;
                const float value = std::clamp(top + (bottom - top) * ty, 0.0f, 1.0f);
                *destination[channel] = static_cast<uint8_t>(value * 255.0f + 0.5f);
            }
        }
    }
    texture.m_is_loaded = true;
    texture.m_uid = Texture::nextUid();
    texture.markVulkanDirtyFull();
    if (g_hasOptix) g_optix_rebuild_pending = true;
    return true;
}

void resizeToPaintGrid(TerrainObject& terrain) {
    if (!terrain.surfaceSemanticMap || !terrain.surfaceSemanticMap->is_loaded()) return;
    Texture& texture = *terrain.surfaceSemanticMap;
    const int width = terrain.paintGridWidth();
    const int height = terrain.paintGridHeight();
    if (width < 2 || height < 2 ||
        (texture.width == width && texture.height == height)) return;
    const int sourceWidth = texture.width;
    const int sourceHeight = texture.height;
    const std::vector<CompactVec4> source = texture.pixels;
    if (sourceWidth < 1 || sourceHeight < 1 ||
        source.size() != static_cast<size_t>(sourceWidth) * sourceHeight) return;
    texture.pixels.resize(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        const float py = static_cast<float>(y) * (sourceHeight - 1) / (height - 1);
        const int y0 = static_cast<int>(py);
        const int y1 = (std::min)(y0 + 1, sourceHeight - 1);
        const float ty = py - y0;
        for (int x = 0; x < width; ++x) {
            const float px = static_cast<float>(x) * (sourceWidth - 1) / (width - 1);
            const int x0 = static_cast<int>(px);
            const int x1 = (std::min)(x0 + 1, sourceWidth - 1);
            const float tx = px - x0;
            CompactVec4& destination = texture.pixels[static_cast<size_t>(y) * width + x];
            uint8_t* channels[4] = {&destination.r, &destination.g, &destination.b, &destination.a};
            for (int channel = 0; channel < 4; ++channel) {
                const auto value = [channel](const CompactVec4& pixel) {
                    const uint8_t* channels[4] = {&pixel.r, &pixel.g, &pixel.b, &pixel.a};
                    return static_cast<float>(*channels[channel]);
                };
                const CompactVec4& p00 = source[static_cast<size_t>(y0) * sourceWidth + x0];
                const CompactVec4& p10 = source[static_cast<size_t>(y0) * sourceWidth + x1];
                const CompactVec4& p01 = source[static_cast<size_t>(y1) * sourceWidth + x0];
                const CompactVec4& p11 = source[static_cast<size_t>(y1) * sourceWidth + x1];
                const float top = value(p00) + (value(p10) - value(p00)) * tx;
                const float bottom = value(p01) + (value(p11) - value(p01)) * tx;
                *channels[channel] = static_cast<uint8_t>(
                    std::clamp(top + (bottom - top) * ty, 0.0f, 255.0f) + 0.5f);
            }
        }
    }
    texture.width = width;
    texture.height = height;
    texture.m_uid = Texture::nextUid();
    texture.markVulkanDirtyFull();
    if (g_hasOptix) g_optix_rebuild_pending = true;
}

bool savePng(const TerrainObject& terrain, const std::string& path, std::string& error) {
    if (!terrain.surfaceSemanticMap || !terrain.surfaceSemanticMap->is_loaded()) {
        error = "Terrain semantic map is unavailable";
        return false;
    }
    const Texture& texture = *terrain.surfaceSemanticMap;
    if (texture.width < 1 || texture.height < 1 ||
        texture.pixels.size() != static_cast<size_t>(texture.width) * texture.height) {
        error = "Terrain semantic map storage is invalid";
        return false;
    }
    std::vector<uint8_t> rgba(static_cast<size_t>(texture.width) * texture.height * 4);
    for (int y = 0; y < texture.height; ++y) {
        for (int x = 0; x < texture.width; ++x) {
            const CompactVec4& pixel = texture.pixels[static_cast<size_t>(y) * texture.width + x];
            const size_t destination = (static_cast<size_t>(texture.height - 1 - y) * texture.width + x) * 4;
            rgba[destination + 0] = pixel.r;
            rgba[destination + 1] = pixel.g;
            rgba[destination + 2] = pixel.b;
            rgba[destination + 3] = pixel.a;
        }
    }
    if (!stbi_write_png(path.c_str(), texture.width, texture.height, 4,
                        rgba.data(), texture.width * 4)) {
        error = "Failed to write terrain semantic PNG";
        return false;
    }
    return true;
}

bool loadPng(TerrainObject& terrain, const std::string& path, std::string& error) {
    int width = 0, height = 0, channels = 0;
    unsigned char* rgba = stbi_load(path.c_str(), &width, &height, &channels, 4);
    if (!rgba || width < 1 || height < 1) {
        if (rgba) stbi_image_free(rgba);
        error = "Failed to load terrain semantic PNG";
        return false;
    }
    if (!terrain.surfaceSemanticMap) {
        terrain.surfaceSemanticMap = std::make_shared<Texture>(
            nullptr, TextureType::Unknown, "TerrainSemanticMap");
    }
    Texture& texture = *terrain.surfaceSemanticMap;
    texture.width = width;
    texture.height = height;
    texture.pixels.resize(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t source = (static_cast<size_t>(y) * width + x) * 4;
            CompactVec4& pixel = texture.pixels[static_cast<size_t>(height - 1 - y) * width + x];
            pixel.r = rgba[source + 0];
            pixel.g = rgba[source + 1];
            pixel.b = rgba[source + 2];
            pixel.a = rgba[source + 3];
        }
    }
    stbi_image_free(rgba);
    texture.m_is_loaded = true;
    texture.m_uid = Texture::nextUid();
    texture.markVulkanDirtyFull();
    if (g_hasOptix) g_optix_rebuild_pending = true;
    return true;
}

std::vector<uint8_t> rgbaBytes(const Texture& texture) {
    if (texture.width < 1 || texture.height < 1 ||
        texture.pixels.size() != static_cast<size_t>(texture.width) * texture.height) return {};
    std::vector<uint8_t> rgba(texture.pixels.size() * 4);
    for (size_t i = 0; i < texture.pixels.size(); ++i) {
        rgba[i * 4 + 0] = texture.pixels[i].r;
        rgba[i * 4 + 1] = texture.pixels[i].g;
        rgba[i * 4 + 2] = texture.pixels[i].b;
        rgba[i * 4 + 3] = texture.pixels[i].a;
    }
    return rgba;
}

} // namespace TerrainSemanticMap
