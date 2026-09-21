#pragma once
#include "Animation/AnimationData.h"
#include "Animation/NodeHierarchy.h"
#include "json.hpp"
namespace RigAuthoring {
nlohmann::json serializeRigHierarchy(const RayTrophi::NodeHierarchy&);
bool deserializeRigHierarchy(const nlohmann::json&, RayTrophi::NodeHierarchy&, std::string& error);
// Compatibility for projects written before nodeHierarchy persistence existed.
bool rebuildLegacyRigHierarchy(const BoneData&, const std::string& character,
                               RayTrophi::NodeHierarchy&, std::string& error);
}
