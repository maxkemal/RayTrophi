#pragma once

#include <array>
#include <memory>
#include <string>
#include "Texture.h"

namespace Paint {

enum class PaintChannel : unsigned char {
    BaseColor = 0,
    Normal,
    Roughness,
    Metallic,
    Emission,
    Mask,
    Transmission,
    Opacity
};

// Number of values in PaintChannel. Centralised so layer/snapshot arrays,
// composite loops and serialisation paths all stay aligned when channels
// are added or removed.
inline constexpr size_t kPaintChannelCount = 8;

struct PaintTextureSet {
    std::string target_node_name;
    uint16_t material_id = 0xFFFF;
    int resolution = 1024;
    bool initialized = false;
    std::array<bool, kPaintChannelCount> seeded_from_existing{};
    std::array<std::string, kPaintChannelCount> source_texture_names{};
    std::array<std::shared_ptr<Texture>, kPaintChannelCount> source_textures{};

    std::shared_ptr<Texture> base_color;
    std::shared_ptr<Texture> normal;
    std::shared_ptr<Texture> roughness;
    std::shared_ptr<Texture> metallic;
    std::shared_ptr<Texture> emission;
    std::shared_ptr<Texture> mask;
    std::shared_ptr<Texture> transmission;
    std::shared_ptr<Texture> opacity;

    std::string makeKey() const;
    std::shared_ptr<Texture> getTexture(PaintChannel channel) const;
    std::shared_ptr<Texture>& getTextureRef(PaintChannel channel);
    bool wasSeededFromExisting(PaintChannel channel) const;
    const std::string& getSourceTextureName(PaintChannel channel) const;
    std::shared_ptr<Texture> getSourceTexture(PaintChannel channel) const;
    void setSourceInfo(PaintChannel channel, bool seeded, const std::string& source_name);
    void setSourceTexture(PaintChannel channel, const std::shared_ptr<Texture>& texture);
};

} // namespace Paint
