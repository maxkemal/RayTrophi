#pragma once
#include "json.hpp"
#include "Vec3.h"
#include "Quaternion.h"
#include <string>
#include <vector>
#include <cstdint>
namespace rtapi {
struct Result;
Result getRigControls(const std::string& character, nlohmann::json& output);
Result selectRigControl(const std::string& character, const std::string& control,
                        const std::string& handle = "target");
Result createRigControls(const std::string& character, const nlohmann::json& controls,
                         uint64_t revision);
Result createRigChainControl(const std::string& character, const std::string& chain,
                             uint64_t revision);
Result createRigAimControl(const std::string& character, const std::string& role,
                           uint64_t revision);
Result setRigIKTarget(const std::string& character, const std::string& control,
                      const Vec3& targetWorld, const Vec3& poleWorld, uint64_t revision);
Result setRigIKOrientation(const std::string& character, const std::string& control,
                           const Quaternion& orientationWorld, bool enabled, uint64_t revision);
Result getRigIKChannels(const std::string& character, nlohmann::json& output);
Result insertRigIKKey(const std::string& character, const std::string& control, uint64_t revision);
Result removeRigIKKey(const std::string& character, const std::string& control, uint64_t revision);
Result setRigIKContactInterval(const std::string& character, const std::string& control,
                               int startFrame, int endFrame, uint64_t revision);
Result clearRigIKChannels(const std::string& character, const std::string& control,
                          uint64_t revision);
Result bakeRigIKChannels(const std::string& character, const std::string& name, int startFrame,
                         int endFrame, uint64_t revision);
Result setRigIKSpline(const std::string& character, const std::string& control,
                      const std::vector<Vec3>& pointsWorld, bool enabled, uint64_t revision);
Result setRigIKFK(const std::string& character, const std::string& control, float blend,
                  uint64_t revision);
Result matchRigIKToFK(const std::string& character, const std::string& control,
                      uint64_t revision);
Result matchRigFKToIK(const std::string& character, const std::string& control,
                      uint64_t revision);
Result setRigIKContact(const std::string& character, const std::string& control, bool enabled,
                       uint64_t revision);
}
