#pragma once
#include "OptixTypes.h"

class Hittable;
namespace Hair {
class HairSystem;
}
namespace Backend {
class IBackend;
}

// Internal Studio/Player link adapter. All business logic remains in the existing
// backend; the Player implementation reports unavailable and never creates OptiX.
class OfflineRenderBridge {
public:
    explicit OfflineRenderBridge(Backend::IBackend* backend);
    explicit operator bool() const {
        return available_;
    }
    const OfflineRenderBridge* operator->() const {
        return this;
    }
    void clearScene() const;
    void clearHairGeometry() const;
    void resetAccumulation() const;
    void updateHairMaterialsOnly(const Hair::HairSystem& hairSystem) const;
    void validateMaterialIndices(const OptixGeometryData& data) const;
    void buildFromDataTLAS(const OptixGeometryData& data,
                           const std::vector<std::shared_ptr<Hittable>>& objects) const;
    void updateMaterialBuffer(const std::vector<GpuMaterial>& materials) const;
    void updateSBTVolumetricData(const std::vector<OptixGeometryData::VolumetricInfo>& info) const;
    void syncSBTMaterialData(const std::vector<GpuMaterial>& materials, bool syncTerrain) const;
    void buildHairGeometry(const float4* vertices, const unsigned int* indices,
                           const uint32_t* strandIDs, const float3* tangents, const float2* rootUVs,
                           const float* strandV, size_t vertexCount, size_t segmentCount,
                           const GpuHairMaterial& material, const std::string& groomName,
                           int materialID, int meshMaterialID, bool useBSpline,
                           bool clearPrevious) const;

private:
    Backend::IBackend* backend_ = nullptr;
    bool available_ = false;
};
