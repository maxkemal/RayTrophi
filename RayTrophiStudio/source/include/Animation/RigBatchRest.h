#pragma once
#include "Animation/NodeHierarchy.h"
namespace RigAuthoring {
// Pure baseline -> target conversion; selected parent+child get one world delta.
bool transformRestHierarchy(const RayTrophi::NodeHierarchy&,const Matrix4x4& placement,const std::vector<std::string>& bones,const Matrix4x4& worldDelta,RayTrophi::NodeHierarchy& output,std::string& error);
}
