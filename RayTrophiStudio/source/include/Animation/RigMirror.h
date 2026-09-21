#pragma once
#include "Animation/RigAnatomy.h"
#include "Animation/NodeHierarchy.h"
#include "Matrix4x4.h"
namespace RigAuthoring {
struct MirrorPlane {std::string axis="x";float offset=0;};
bool mirrorReflection(const MirrorPlane&,Matrix4x4& pointReflection,Matrix4x4& basisReflection,std::string& error);
bool mirrorPairs(const RayTrophi::NodeHierarchy&,const RigAnatomy&,const std::vector<std::string>& sources,const std::string& direction,std::vector<RigSymmetry>& copies,std::string& error);
bool mirrorRest(const RayTrophi::NodeHierarchy&,const RigAnatomy&,const std::vector<std::string>& sources,const std::string& direction,const MirrorPlane&,RayTrophi::NodeHierarchy& output,std::vector<std::string>& targets,std::string& error);
// Copy rest-relative local motion using joint rest bases in actor coordinates.
bool mirrorPose(const RayTrophi::NodeHierarchy& rest,const RayTrophi::NodeHierarchy& pose,const RigAnatomy&,const std::vector<std::string>& sources,const std::string& direction,const std::string& axis,RayTrophi::NodeHierarchy& output,std::string& error);
bool mirrorLandmarks(const RayTrophi::NodeHierarchy&,const RigAnatomy&,const Matrix4x4& placement,const nlohmann::json& marks,const std::vector<std::string>& sources,const std::string& direction,const MirrorPlane&,nlohmann::json& output,std::string& error);
bool createMirrorBone(const RayTrophi::NodeHierarchy&,const RigAnatomy&,const std::string& source,const std::string& key,const std::string& label,const std::string& sourceSide,const MirrorPlane&,RayTrophi::NodeHierarchy& output,RigAnatomy& anatomy,std::string& error);
}
