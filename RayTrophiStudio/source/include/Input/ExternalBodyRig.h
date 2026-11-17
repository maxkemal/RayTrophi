#pragma once

#include "Matrix4x4.h"
#include "json.hpp"
#include <string>
#include <vector>

struct SceneData;
namespace Input {
bool bindExternalBodyRig(SceneData &scene, const std::string &character, int person_id,
                         const nlohmann::json &config, std::string &error);
bool clearExternalBodyRig(SceneData &scene, const std::string &character, std::string &error);
nlohmann::json externalBodyRigs(const SceneData &scene);
void tickExternalBodyRigs(SceneData &scene);
void resetExternalBodyRigs();
bool hasExternalBodyRigs(const SceneData &scene);
bool externalBodyRigOwns(const SceneData &scene, const std::string &character);
bool copyExternalBodyRigPose(SceneData &scene, const std::string &character,
                             std::vector<Matrix4x4> &skin, bool &changed);
bool copyExternalBodyRigFrame(SceneData &scene, std::vector<Matrix4x4> &skin);

// Existing kinematic proxy sets remain the sole domain producer. A missing or
// unreliable person/bone is unresolved, never a frozen/rest ghost collider.
bool externalBodyRigColliderAvailable(const SceneData &scene, const std::string &character,
                                      std::string &error);
bool externalBodyRigColliderBoneAvailable(const SceneData &scene, const std::string &character,
                                          const std::string &bone);
} // namespace Input
