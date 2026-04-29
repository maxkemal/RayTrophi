#pragma once

#include "Backend/VulkanBackend.h"

namespace Backend {

// Dedicated viewport-facing backend identity for Solid/Matcap/Preview paths.
// It intentionally reuses the proven Vulkan raster implementation today so we
// can keep existing capabilities while separating viewport concerns from the
// rendered-path backend lifecycle.
class VulkanViewportBackend final : public VulkanBackendAdapter {
public:
    VulkanViewportBackend() = default;
    ~VulkanViewportBackend() override = default;
    const char* sceneTextureOwnerScope() const override { return "VulkanViewportBackend"; }

    // Set external material buffer from the render backend for MaterialPreview mode.
    // This avoids duplicating material uploads — viewport backend borrows the buffer.
    void setExternalMaterialBuffer(VkBuffer buffer, VkDeviceSize size);

private:
    VkBuffer m_externalMaterialBuffer = VK_NULL_HANDLE;
    VkDeviceSize m_externalMaterialBufferSize = 0;
    uint32_t m_externalMaterialCount = 0;
public:

    void renderProgressive(void* outSurface, void* outWindow, void* outRenderer,
                           int width, int height, void* outFramebuffer, void* outTexture) override;

    void setInteractiveViewportMatcap(int64_t textureID) override;
    void setInteractiveViewportMatcapPreset(int preset) override;
    void setInteractiveViewportMatcapImpl(int64_t textureID) override;
    void setInteractiveViewportMatcapPresetImpl(int preset) override;

    void uploadTerrainLayerMaterials(const std::vector<TerrainLayerData>& layers) override;

    bool updateRasterMeshFromTriangles(
        const std::string& nodeName,
        const std::vector<std::shared_ptr<Triangle>>& triangles) override;

    bool patchRasterMeshTriangles(
        const std::string& nodeName,
        const std::vector<size_t>& dirtyIndices,
        const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries) override;
    bool cloneRasterObjectByNodeName(
        const std::string& sourceNodeName,
        const std::string& newNodeName,
        const Matrix4x4& transform) override;

    void buildRasterGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) override;
    void syncRasterInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) override;
    void syncRasterSkinnedVertices(const std::vector<std::shared_ptr<Hittable>>& objects,
                                   const std::vector<Matrix4x4>& boneMatrices) override;

protected:
    bool ensureInteractiveViewportResourcesImpl(const std::string& shaderDir, int width, int height) override;
    void destroyInteractiveViewportResourcesImpl(bool keepPipeline = false) override;
    void renderInteractiveViewportImpl(void* outSurface, int width, int height,
                                       void* outFramebuffer, void* outTexture) override;
};

} // namespace Backend
