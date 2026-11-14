#include "OfflineRenderBridge.h"

OfflineRenderBridge::OfflineRenderBridge(Backend::IBackend* backend)
    : backend_(backend), available_(false) {
}

void OfflineRenderBridge::clearScene() const {
}

void OfflineRenderBridge::clearHairGeometry() const {
}

void OfflineRenderBridge::resetAccumulation() const {
}

void OfflineRenderBridge::updateHairMaterialsOnly(const Hair::HairSystem& hairSystem) const {
}

void OfflineRenderBridge::validateMaterialIndices(const OptixGeometryData& data) const {
}

void OfflineRenderBridge::buildFromDataTLAS(
    const OptixGeometryData& data, const std::vector<std::shared_ptr<Hittable>>& objects) const {
}

void OfflineRenderBridge::updateMaterialBuffer(const std::vector<GpuMaterial>& materials) const {
}

void OfflineRenderBridge::updateSBTVolumetricData(
    const std::vector<OptixGeometryData::VolumetricInfo>& info) const {
}

void OfflineRenderBridge::syncSBTMaterialData(const std::vector<GpuMaterial>& materials,
                                              bool syncTerrain) const {
}

void OfflineRenderBridge::buildHairGeometry(
    const float4* vertices, const unsigned int* indices, const uint32_t* strandIDs,
    const float3* tangents, const float2* rootUVs, const float* strandV, size_t vertexCount,
    size_t segmentCount, const GpuHairMaterial& material, const std::string& groomName,
    int materialID, int meshMaterialID, bool useBSpline, bool clearPrevious) const {
}
