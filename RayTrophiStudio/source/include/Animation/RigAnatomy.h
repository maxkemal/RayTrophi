#pragma once
#include <string>
#include <vector>
#include "json.hpp"
#include "Animation/RigJointRules.h"
#include "Animation/RigIK.h"
namespace RayTrophi { class NodeHierarchy; }
namespace RigAuthoring {
struct RigRole { std::string role,bone; };
struct RigSymmetry { std::string left,right; };
struct RigChain { std::string name; std::vector<std::string> bones; };
struct RigFitRule {
    std::string bone;
    std::string start;
    std::string end;
    float position = .5f;
};
struct RigControlDriver {
    std::string bone;
    Vec3 axis = Vec3(0, 0, 1);
    float degrees = 0.f;
};
struct RigDrivenControl {
    std::string id;
    std::string label;
    std::string group;
    std::string anchor;
    std::string side = "center";
    std::string shape = "ring";
    float minimum = -1.f;
    float maximum = 1.f;
    float defaultValue = 0.f;
    std::vector<RigControlDriver> drivers;
};
struct RigAnatomy {
    std::string family="custom";
    std::vector<RigRole> roles;
    std::vector<RigSymmetry> symmetry;
    std::vector<RigChain> chains;
    std::vector<JointRule> joints;
    std::vector<IKControl> controls;
    std::vector<RigFitRule> fitRules;
    std::vector<RigDrivenControl> drivenControls;
};
bool validateRigAnatomy(const RigAnatomy&,const RayTrophi::NodeHierarchy&,std::string& error);
nlohmann::json serializeRigAnatomy(const RigAnatomy&);
bool deserializeRigAnatomy(const nlohmann::json&,const RayTrophi::NodeHierarchy&,RigAnatomy&,std::string& error);
void renameAnatomyBone(RigAnatomy&,const std::string& oldKey,const std::string& newKey);
bool anatomyReferencesBone(const RigAnatomy&,const std::string& bone);
bool buildLimbIKControls(const RigAnatomy&,const RayTrophi::NodeHierarchy&,std::vector<IKControl>&,std::string& error);
}
