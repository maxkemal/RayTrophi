// Legacy Water/River system: Player link alternative ("no legacy water").
//
// PLAYER BUILD ONLY. Studio links WaterManager.cpp / RiverManager.cpp (FFT
// ocean, river meshing, wave tick); those stay out of the Player by the
// 2026-11-13 scope decision (realtime water will be designed separately). The
// two implementations must never be linked into the same target; Studio lists
// this file as None.
//
// Semantics: the Player never owns a legacy water surface or river. Baked
// geometry still loads as ordinary flat meshes with their materials; only the
// water/river manager records are ignored. Lookups return nullptr, creation
// refuses (nullptr), serialization writes an empty list.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md (Faz 3c).

#include "WaterSystem.h"
#include "RiverSpline.h"
#include "globals.h"

// ---------------------------------------------------------------- WaterManager

WaterSurface* WaterManager::getWaterSurface(int /*id*/) {
    return nullptr;
}

WaterSurface* WaterManager::getWaterSurfaceByNodeName(const std::string& /*nodeName*/) {
    return nullptr;
}

WaterSurface* WaterManager::bindExistingWaterMesh(const std::shared_ptr<TriangleMesh>& /*mesh*/,
                                                  WaterSurface::Type /*type*/) {
    return nullptr;
}

WaterSurface* WaterManager::createWaterFromIndexedMesh(
    SceneData& /*scene*/,
    const std::string& /*name*/,
    const std::vector<Vec3>& /*positions*/,
    const std::vector<Vec2>& /*uvs*/,
    const std::vector<uint32_t>& /*indices*/,
    const std::shared_ptr<Transform>& /*transform*/,
    WaterSurface::Type /*type*/,
    const WaterWaveParams& /*params*/,
    const std::vector<float>& /*waterDepth*/,
    const std::vector<float>& /*shoreFactor*/) {
    return nullptr;
}

void WaterManager::removeWaterSurface(SceneData& /*scene*/, int /*id*/) {
}

void WaterManager::clear() {
    water_surfaces.clear();
    next_id = 1;
}

void WaterManager::syncSurfaceMaterial(WaterSurface* /*surf*/) {
}

bool WaterManager::syncVulkanFFTTexturesForMaterial(uint16_t /*material_id*/, Backend::IBackend* /*backend*/,
                                                    int64_t& /*outHeightTexture*/, int64_t& /*outNormalTexture*/) {
    return false;
}

nlohmann::json WaterManager::serialize() const {
    return nlohmann::json::array();
}

void WaterManager::deserialize(const nlohmann::json& j, SceneData& /*scene*/) {
    if (j.is_array() && !j.empty()) {
        SCENE_LOG_WARN("[Player] Legacy water surfaces are not supported; " +
                       std::to_string(j.size()) + " record(s) ignored (baked meshes still load).");
    }
}

// ---------------------------------------------------------------- RiverManager

void RiverManager::clear(SceneData* /*scene*/) {
    rivers.clear();
    next_id = 1;
}

void RiverManager::generateMesh(RiverSpline* /*river*/, SceneData& /*scene*/) {
}

void RiverManager::updateAllRivers(SceneData& /*scene*/) {
}

void RiverManager::removeRiver(SceneData& /*scene*/, int /*id*/) {
}
