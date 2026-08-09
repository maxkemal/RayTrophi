#pragma once

#include "FractureGenerator.h"
#include "Hittable.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace RayTrophiSim {
bool isFractureSourceObject(const std::shared_ptr<Hittable>& object,
                            const std::string& node);
bool gatherFractureSource(
    const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::string& node, std::vector<FractureInputTri>& out,
    uint16_t& out_material);
}
