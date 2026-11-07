#pragma once

#include <cstring>

namespace MaterialInspectorSchema {

// Presentation metadata only. Validation and material storage remain in the
// existing service; animation flags describe this editor's existing key actions.
enum class Group { Surface, Transparency, Interior, Subsurface, Coat, Emission, Detail };
enum class Kind { Scalar, Integer, Color, Vector, Toggle, Choice };
struct Field {
    const char* id;
    const char* label;
    Group group;
    Kind kind;
    bool keyAction;
};
inline const Field fields[] = {
    {"Alpha Cutout", "Alpha Cutout", Group::Transparency, Kind::Toggle, false},
    {"spec", "Specular", Group::Surface, Kind::Scalar, true},
    {"Dispersion##disp", "Dispersion", Group::Transparency, Kind::Scalar, false},
    {"Bubble (thin shell)", "Bubble (thin shell)", Group::Transparency, Kind::Toggle, false},
    {"Rim IOR##bubble", "Rim IOR", Group::Transparency, Kind::Scalar, false},
    {"Thin-Film##bubble", "Thin-Film", Group::Transparency, Kind::Scalar, false},
    {"opac", "Opacity", Group::Transparency, Kind::Scalar, true},
    {"trns", "Translucency", Group::Transparency, Kind::Scalar, false},
    {"Interior Depth##transdens", "Interior Depth", Group::Interior, Kind::Scalar, false},
    {"Interior Tint##resincol", "Interior Tint", Group::Interior, Kind::Color, false},
    {"Coat Gloss##resinrough", "Coat Roughness", Group::Interior, Kind::Scalar, false},
    {"Interior Preset##resinpreset", "Interior Preset", Group::Interior, Kind::Choice, false},
    {"Dust##resindust", "Dust", Group::Interior, Kind::Scalar, false},
    {"Dust Style##duststyle", "Dust Style", Group::Interior, Kind::Choice, false},
    {"Dust Color A##dustcola", "Dust Color A", Group::Interior, Kind::Color, false},
    {"Dust Color B##dustcolb", "Dust Color B", Group::Interior, Kind::Color, false},
    {"Dirt Specks##resindirt", "Dirt Specks", Group::Interior, Kind::Scalar, false},
    {"Dirt Color##resindirtcol", "Dirt Color", Group::Interior, Kind::Color, false},
    {"Glass Shards##resinshard", "Glass Shards", Group::Interior, Kind::Scalar, false},
    {"Shard Shape##shardshape", "Shard Shape", Group::Interior, Kind::Choice, false},
    {"Rainbow##resinshardrainbow", "Rainbow", Group::Interior, Kind::Toggle, false},
    {"Shard Hue##resinshardhue", "Shard Hue", Group::Interior, Kind::Scalar, false},
    {"Inclusion Scale##resininclscale", "Inclusion Scale", Group::Interior, Kind::Scalar, false},
    {"Anchor##resinanchor", "Anchor", Group::Interior, Kind::Choice, false},
    {"SSS Color", "Scattering Color", Group::Subsurface, Kind::Color, false},
    {"Radius (RGB)", "Scattering Radius", Group::Subsurface, Kind::Vector, false},
    {"SSS Method", "Method", Group::Subsurface, Kind::Choice, false},
    {"Walk Max Steps", "Walk Limit", Group::Subsurface, Kind::Integer, false},
    {"sss", "Amount", Group::Subsurface, Kind::Scalar, false},
    {"pd_sc", "Scale", Group::Detail, Kind::Scalar, false},
    {"sscl", "Scale", Group::Subsurface, Kind::Scalar, false},
    {"ssani", "Anisotropy", Group::Subsurface, Kind::Scalar, false},
    {"ssior", "IOR", Group::Subsurface, Kind::Scalar, false},
    {"cc", "Amount", Group::Coat, Kind::Scalar, false},
    {"ccr", "Roughness", Group::Coat, Kind::Scalar, false},
    {"cci", "Iridescence", Group::Coat, Kind::Scalar, false},
    {"ccft", "Film Thickness", Group::Coat, Kind::Scalar, false},
    {"Emission", "Emission", Group::Emission, Kind::Color, true},
    {"mems", "Intensity", Group::Emission, Kind::Scalar, false},
    {"pd_str", "Strength", Group::Detail, Kind::Scalar, false},
    {"pd_tb", "Tile Break", Group::Detail, Kind::Scalar, false},
    {"Base Color", "Base Color", Group::Surface, Kind::Color, true},
    {"Metallic", "Metallic", Group::Surface, Kind::Scalar, true},
    {"Roughness", "Roughness", Group::Surface, Kind::Scalar, true},
    {"IOR", "IOR", Group::Transparency, Kind::Scalar, true},
    {"Transmission", "Transmission", Group::Transparency, Kind::Scalar, true},
};

inline const Field* find(const char* id) {
    for (const auto& field : fields) {
        if (std::strcmp(field.id, id) == 0) {
            return &field;
        }
    }
    return nullptr;
}

} // namespace MaterialInspectorSchema
