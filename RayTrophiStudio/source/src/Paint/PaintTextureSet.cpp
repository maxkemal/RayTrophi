#include "Paint/PaintTextureSet.h"

namespace Paint {

namespace {

size_t toIndex(PaintChannel channel) {
    return static_cast<size_t>(channel);
}

} // namespace

std::string PaintTextureSet::makeKey() const {
    return target_node_name + "#" + std::to_string(material_id);
}

std::shared_ptr<Texture> PaintTextureSet::getTexture(PaintChannel channel) const {
    switch (channel) {
        case PaintChannel::BaseColor: return base_color;
        case PaintChannel::Normal: return normal;
        case PaintChannel::Roughness: return roughness;
        case PaintChannel::Metallic: return metallic;
        case PaintChannel::Emission: return emission;
        case PaintChannel::Mask: return mask;
        case PaintChannel::Transmission: return transmission;
        case PaintChannel::Opacity: return opacity;
    }
    return nullptr;
}

std::shared_ptr<Texture>& PaintTextureSet::getTextureRef(PaintChannel channel) {
    switch (channel) {
        case PaintChannel::BaseColor: return base_color;
        case PaintChannel::Normal: return normal;
        case PaintChannel::Roughness: return roughness;
        case PaintChannel::Metallic: return metallic;
        case PaintChannel::Emission: return emission;
        case PaintChannel::Mask: return mask;
        case PaintChannel::Transmission: return transmission;
        case PaintChannel::Opacity: return opacity;
    }
    return base_color;
}

bool PaintTextureSet::wasSeededFromExisting(PaintChannel channel) const {
    return seeded_from_existing[toIndex(channel)];
}

const std::string& PaintTextureSet::getSourceTextureName(PaintChannel channel) const {
    return source_texture_names[toIndex(channel)];
}

std::shared_ptr<Texture> PaintTextureSet::getSourceTexture(PaintChannel channel) const {
    return source_textures[toIndex(channel)];
}

void PaintTextureSet::setSourceInfo(PaintChannel channel, bool seeded, const std::string& source_name) {
    seeded_from_existing[toIndex(channel)] = seeded;
    source_texture_names[toIndex(channel)] = source_name;
}

void PaintTextureSet::setSourceTexture(PaintChannel channel, const std::shared_ptr<Texture>& texture) {
    source_textures[toIndex(channel)] = texture;
}

} // namespace Paint
