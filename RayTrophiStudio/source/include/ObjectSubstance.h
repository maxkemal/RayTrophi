#pragma once

#include "json.hpp"
#include <string>
#include <memory>
#include <cstdint>

struct SceneData;
struct UIContext;
namespace RayTrophiSim {
struct ParticleColliderDesc;
}

namespace matter {
nlohmann::json validateObjectSubstance(const nlohmann::json& binding);
nlohmann::json objectSubstanceState(const SceneData& scene, const std::string& object,
                                  bool knownFlatNode = false);
nlohmann::json invokeObjectSubstance(UIContext& ctx, const std::string& operation,
                                    const nlohmann::json& params);
void syncObjectSubstances(SceneData& scene);
void applyObjectSubstance(const SceneData& scene, RayTrophiSim::ParticleColliderDesc& collider);
bool colliderMatchesObjectSubstance(const SceneData& scene,
                                   const RayTrophiSim::ParticleColliderDesc& collider);
nlohmann::json saveObjectSubstances(const SceneData& scene);
void restoreObjectSubstances(SceneData& scene, const nlohmann::json& data);
nlohmann::json takeObjectSubstance(SceneData& scene, const std::string& object);
void putObjectSubstance(SceneData& scene, const std::string& object,
                        const nlohmann::json& binding);
void copyObjectSubstance(SceneData& scene, const std::string& source, const std::string& target);
std::shared_ptr<void> retainObjectSubstanceHistory(const nlohmann::json& state);
bool objectSubstanceHistoryUses(const std::string& substance);
std::uint64_t objectSubstanceFingerprint(const SceneData& scene);
void drawObjectSubstance(UIContext& ctx, const std::string& object);
} // namespace matter
