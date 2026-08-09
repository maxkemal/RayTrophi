#pragma once
#include "json.hpp"
namespace RayTrophiSim {
class AshDebrisSystem;
nlohmann::json serializeAshDebris(const AshDebrisSystem& system);
void deserializeAshDebris(const nlohmann::json& root, AshDebrisSystem& system);
}
