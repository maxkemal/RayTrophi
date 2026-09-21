#pragma once
#include "json.hpp"
#include "Matrix4x4.h"
#include <vector>
#include <string>
#include <cstdint>
namespace rtapi {
struct Result;
Result getRigPoseCoverage(const std::string& character,nlohmann::json& output);
Result getRigPoseState(const std::string& character,nlohmann::json& output);
Result createRigPoseClip(const std::string& character,const std::string& name,float fps);
Result selectRigPoseClip(const std::string& character,const std::string& clip);
Result setRigPoseAutoKey(bool enabled);
Result setRigPoseFrame(int frame);
Result previewRigPoseTransform(const std::string& character,const std::vector<std::string>& bones,const Matrix4x4& worldDelta,uint64_t revision);
Result previewRigPoseLocals(const std::string& character,const nlohmann::json& locals,uint64_t revision);
Result getRigDrivenControls(const std::string& character, nlohmann::json& output);
Result previewRigControlValues(const std::string& character, const nlohmann::json& values,
                               uint64_t revision);
Result mirrorRigPose(const std::string& character,const std::vector<std::string>& bones,uint64_t revision,const std::string& direction="selected",const std::string& axis="x");
Result applyRigPosePreview(const std::string& character);
Result cancelRigPosePreview(const std::string& character);
Result insertRigPoseKeys(const std::string& character,const std::vector<std::string>& bones);
Result removeRigPoseKeys(const std::string& character,const std::vector<std::string>& bones);
Result editRigPoseKey(const std::string& character, const std::string& bone,
                      const std::string& channel, int sourceFrame, int targetFrame,
                      const nlohmann::json& value = nlohmann::json());
}
