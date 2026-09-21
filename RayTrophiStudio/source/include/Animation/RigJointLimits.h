#pragma once
#include "Animation/RigJointRules.h"
#include <cstdint>
struct SceneData;
namespace RigAuthoring {
struct JointLimitView {
 std::string character,bone,poseSource;JointRule rule;
 Matrix4x4 neutralWorld=Matrix4x4::identity(),jointWorld=Matrix4x4::identity();
 bool owned=false,hasRule=false,ik=false,outside=false;uint64_t revision=0;
 float twist=0,swing=0;
};
bool getJointLimitView(const SceneData&,const std::string& character,const std::string& bone,JointLimitView&,std::string& error);
bool replaceJointLimits(const std::vector<JointRule>&,const RayTrophi::NodeHierarchy&,const std::string& bone,float minimum,float maximum,float swing,std::vector<JointRule>& output,std::string& error);
Vec3 jointLimitAxis(const JointLimitView&);
Vec3 jointLimitReference(const JointLimitView&);
Vec3 jointLimitArcPoint(const JointLimitView&,float radius,float degrees);
Vec3 jointLimitConePoint(const JointLimitView&,float radius,float swingDegrees,float azimuthDegrees);
bool jointOutsideLimits(const JointRule&,float twistDegrees,float swingDegrees);
nlohmann::json jointLimitViewJson(const JointLimitView&);
}
