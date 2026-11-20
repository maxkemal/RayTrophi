#pragma once
#include "Vec3.h"
#include "json.hpp"
#include <string>

struct SceneData;
namespace Input { struct BodyPerson; }

namespace Paint {
void clearHiddenGameObjects();
nlohmann::json hiddenGameObjectsDesign();
bool validateHiddenGameObjects(SceneData& scene, const nlohmann::json& config,
                              std::string& error);
bool configureHiddenGameObjects(SceneData& scene, const nlohmann::json& config,
                               std::string& error);
bool resetHiddenGameObjects(SceneData& scene, std::string& error);
bool hiddenGameObjectsCommand(SceneData& scene, const nlohmann::json& input,
                              nlohmann::json& output, std::string& error);
// Unmanaged scene geometry retains the old blocking behavior for carried balloons.
bool hiddenObjectBlocksBalloon(const std::string& object);
// Managed static meshes may move through instance handles without rebaking CPU P.
bool hiddenObjectUsesInstancePose(const std::string& object);
// Same policy for tracked avatars, camera card and IPC test anchors.
Vec3 constrainHiddenPlayerAnchor(const std::string& owner, const Vec3& desired,
                                float radius, float height);
void constrainHiddenBodyPerson(Input::BodyPerson& person);
void clearHiddenPlayerHistory();
} // namespace Paint
