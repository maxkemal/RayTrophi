#pragma once

#include "NodeSystem/NodeCore.h"

#include <cstdint>
#include <string>
#include <vector>

struct TerrainObject;
class Texture;

namespace TerrainSemanticMap {

    // Publishes non-normalized RGBA controls at terrain paint resolution.
    // Channel contract: R=Flow, G=Wetness, B=Ice, A=Hardness.
    bool publish(TerrainObject& terrain,
                 const NodeSystem::Image2DData& image,
                 std::string& error);

    void resizeToPaintGrid(TerrainObject& terrain);
    bool savePng(const TerrainObject& terrain, const std::string& path, std::string& error);
    bool loadPng(TerrainObject& terrain, const std::string& path, std::string& error);
    std::vector<uint8_t> rgbaBytes(const Texture& texture);

} // namespace TerrainSemanticMap
