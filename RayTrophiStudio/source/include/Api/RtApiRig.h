#pragma once
#include "Animation/RigView.h"
#include "Api/RtApiRigPoseView.h"
namespace rtapi {
struct Result;
Result setRigMode(const std::string& mode, const std::string& character = "");
Result getRigMode(std::string& mode, std::string& character);
Result listRigCharacters(std::vector<std::string>&);
Result listRigBones(const std::string&, std::vector<RigAuthoring::BoneView>&);
Result selectRigBone(const std::string&, const std::string&);
Result clearRigSelection();
Result getSelectedRigBone(RigAuthoring::BoneView&, bool& hasSelection);
Result getRigOverlayVisible(bool&);
Result setRigOverlayVisible(bool);
}
