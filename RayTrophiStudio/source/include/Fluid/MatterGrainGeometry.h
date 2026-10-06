#pragma once

#include <memory>
#include <string>
#include <vector>

class Hittable;
namespace RayTrophiSim {
struct SurfaceMeshTriangle;
namespace Fluid {
bool collectFlatGrainCollider(const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::string& name, std::vector<SurfaceMeshTriangle>& triangles);
}
}
