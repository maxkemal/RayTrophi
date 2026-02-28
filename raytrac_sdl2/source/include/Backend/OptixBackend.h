/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          OptixBackend.h
 * Description:   NVIDIA OptiX implementation details
 * =========================================================================
 */
#ifndef OPTIX_BACKEND_H
#define OPTIX_BACKEND_H

#include "Backend/IBackend.h"
#include <memory>
#include <vector>

class OptixWrapper;
struct OptixGeometryData;

namespace Backend {

class OptixBackend : public IBackend {
public:
    OptixBackend();
    OptixBackend(OptixWrapper* existingWrapper); // Wrap existing (non-owning)
    virtual ~OptixBackend();

    bool initialize() override;
    void shutdown() override;
    void loadShaders(const ShaderProgramData& data) override;
    BackendInfo getInfo() const override;

    uint32_t uploadTriangles(const std::vector<TriangleData>& triangles, const std::string& meshName) override;
    uint32_t uploadHairStrands(const std::vector<HairStrandData>& strands, const std::string& groomName) override;
    void updateMeshTransform(uint32_t meshHandle, const Matrix4x4& transform) override;
    void rebuildAccelerationStructure() override;
    void showAllInstances() override;
    void updateSceneGeometry(const std::vector<std::shared_ptr<Hittable>>& objects, const std::vector<Matrix4x4>& boneMatrices) override;
    void updateInstanceMaterialBinding(const std::string& nodeName, int oldMatID, int newMatID) override;
    void setVisibilityByNodeName(const std::string& nodeName, bool visible) override;
    void updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) override;

    void uploadMaterials(const std::vector<MaterialData>& materials) override;
    void uploadHairMaterials(const std::vector<HairMaterialData>& materials) override;
    
    int64_t uploadTexture2D(const void* data, uint32_t width, uint32_t height, uint32_t channels, bool sRGB, bool isFloat = false) override;
    void destroyTexture(int64_t textureHandle) override;

    void setRenderParams(const RenderParams& params) override;
    void setCamera(const CameraParams& camera) override;
    void setTime(float time, float deltaTime) override;
    void updateInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) override;
    void updateInstanceTransform(int instance_id, const float transform[12]) override;
    void setStatusCallback(std::function<void(const std::string&, int)> callback) override;
    void* getNativeCommandQueue() override;
    void setWindParams(const Vec3& direction, float strength, float speed, float time) override;
    void renderPass(bool accumulate = true) override;
    void renderProgressive(void* outSurface, void* outWindow, void* outRenderer, 
                           int width, int height, void* outFramebuffer, void* outTexture) override;
    void downloadImage(void* outPixels) override;
    int getCurrentSampleCount() const override;
    bool isAccumulationComplete() const override;

    void setEnvironmentMap(int64_t hdrTextureHandle) override;
    void setSkyParams() override;
    void setLights(const std::vector<std::shared_ptr<Light>>& lights) override;
    void setWorldData(const void* worldData) override;
    
    void updateVDBVolumes(const std::vector<GpuVDBVolume>& volumes) override;
    void updateGasVolumes(const std::vector<GpuGasVolume>& volumes) override;

    void waitForCompletion() override;
    void resetAccumulation() override;
    float getMillisecondsPerSample() const override;

    // Extended IBackend methods (delegated to OptixWrapper)
    bool isUsingTLAS() const override;
    std::vector<int> getInstancesByNodeName(const std::string& nodeName) const override;
    void updateObjectTransform(const std::string& nodeName, const Matrix4x4& transform) override;
    void syncCamera(const Camera& cam) override;
    void hideInstancesByNodeName(const std::string& nodeName) override;
    int getPickedObjectId(int x, int y, int viewport_width, int viewport_height) override;
    std::string getPickedObjectName(int x, int y, int viewport_width, int viewport_height) override;

    // Legacy OptiX support (for transitional refactor)
    OptixWrapper* getOptixWrapper() { return m_optix; }

private:
    std::unique_ptr<OptixWrapper> m_optix_owned;  // Owning pointer (when we create our own)
    OptixWrapper* m_optix = nullptr;               // Always valid, may point to owned or external
};

} // namespace Backend

#endif // OPTIX_BACKEND_H
