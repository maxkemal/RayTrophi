#pragma once
#include "Animation/NodeHierarchy.h"
#include "Vec3.h"
#include "json.hpp"
#include <string>
#include <vector>
namespace RigAuthoring {
struct JointRule {
 std::string bone,type="free";
 bool enabled=false,lockTranslation=true;
 Vec3 axis=Vec3(1,0,0); // Unit axis in the joint's rest-relative local frame.
 float minimum=-180,maximum=180,swing=180; // Degrees; neutral must remain valid.
};
bool validateJointRules(const std::vector<JointRule>&,const RayTrophi::NodeHierarchy&,std::string& error);
nlohmann::json serializeJointRules(const std::vector<JointRule>&);
bool deserializeJointRules(const nlohmann::json&,const RayTrophi::NodeHierarchy&,std::vector<JointRule>&,std::string& error);
bool constrainJointPose(const RayTrophi::NodeHierarchy& rest,const std::vector<JointRule>&,const RayTrophi::NodeHierarchy& input,RayTrophi::NodeHierarchy& output,std::vector<std::string>& hits,std::string& error);
}
