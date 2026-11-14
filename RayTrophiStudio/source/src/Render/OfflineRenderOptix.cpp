#include "OfflineRenderBridge.h"
#include "Backend/OptixBackend.h"
#include "OptixWrapper.h"

namespace {
OptixWrapper* wrapper(Backend::IBackend* backend) {
    auto* optix = dynamic_cast<Backend::OptixBackend*>(backend);
    return optix ? optix->getOptixWrapper() : nullptr;
}
} // namespace

OfflineRenderBridge::OfflineRenderBridge(Backend::IBackend* backend)
    : backend_(backend), available_(wrapper(backend) != nullptr) {
}

void OfflineRenderBridge::clearScene() const {
    if (auto* optix = wrapper(backend_)) {
        optix->clearScene();
    }
}

void OfflineRenderBridge::clearHairGeometry() const {
    if (auto* optix = wrapper(backend_)) {
        optix->clearHairGeometry();
    }
}

void OfflineRenderBridge::resetAccumulation() const {
    if (auto* optix = wrapper(backend_)) {
        optix->resetAccumulation();
    }
}

void OfflineRenderBridge::updateHairMaterialsOnly(const Hair::HairSystem& hairSystem) const {
    if (auto* optix = wrapper(backend_)) {
        optix->updateHairMaterialsOnly(hairSystem);
    }
}

void OfflineRenderBridge::validateMaterialIndices(const OptixGeometryData& data) const {
    if (auto* optix = wrapper(backend_)) {
        optix->validateMaterialIndices(data);
    }
}

void OfflineRenderBridge::buildFromDataTLAS(
    const OptixGeometryData& data, const std::vector<std::shared_ptr<Hittable>>& objects) const {
    if (auto* optix = wrapper(backend_)) {
        optix->buildFromDataTLAS(data, objects);
    }
}

void OfflineRenderBridge::updateMaterialBuffer(const std::vector<GpuMaterial>& materials) const {
    if (auto* optix = wrapper(backend_)) {
        optix->updateMaterialBuffer(materials);
    }
}

void OfflineRenderBridge::updateSBTVolumetricData(
    const std::vector<OptixGeometryData::VolumetricInfo>& info) const {
    if (auto* optix = wrapper(backend_)) {
        optix->updateSBTVolumetricData(info);
    }
}

void OfflineRenderBridge::syncSBTMaterialData(const std::vector<GpuMaterial>& materials,
                                              bool syncTerrain) const {
    if (auto* optix = wrapper(backend_)) {
        optix->syncSBTMaterialData(materials, syncTerrain);
    }
}

void OfflineRenderBridge::buildHairGeometry(
    const float4* vertices, const unsigned int* indices, const uint32_t* strandIDs,
    const float3* tangents, const float2* rootUVs, const float* strandV, size_t vertexCount,
    size_t segmentCount, const GpuHairMaterial& material, const std::string& groomName,
    int materialID, int meshMaterialID, bool useBSpline, bool clearPrevious) const {
    if (auto* optix = wrapper(backend_)) {
        optix->buildHairGeometry(vertices, indices, strandIDs, tangents, rootUVs, strandV,
                                 vertexCount, segmentCount, material, groomName, materialID,
                                 meshMaterialID, useBSpline, clearPrevious);
    }
}
