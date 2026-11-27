#pragma once

#include "Backend/VulkanBackend.h"
#include "Viewport/RasterViewportFrameRing.h"
#include "Viewport/ParticleBillboardFrameBuffer.h"
#include "Viewport/RayFusionViewportPasses.h"
#ifdef RT_PLAYER
#include "Viewport/PlayerDepthCard.h"
#endif

namespace Backend {

// Dedicated viewport-facing backend identity for Solid/Matcap/Preview paths.
// It intentionally reuses the proven Vulkan raster implementation today so we
// can keep existing capabilities while separating viewport concerns from the
// rendered-path backend lifecycle.
class VulkanViewportBackend final : public VulkanBackendAdapter {
public:
    VulkanViewportBackend() = default;
    ~VulkanViewportBackend() override;
    const char* sceneTextureOwnerScope() const override { return "VulkanViewportBackend"; }

    // Set external material buffer from the render backend for MaterialPreview mode.
    // This avoids duplicating material uploads — viewport backend borrows the buffer.
    void setExternalMaterialBuffer(VkBuffer buffer, VkDeviceSize size);

    void setInteractiveViewportSynchronousPresent(bool enabled) override;
    bool getInteractiveViewportFrameTelemetry(RasterFrameTelemetry& out) const override;

private:
#ifdef RT_PLAYER
    PlayerDepthCard m_playerDepthCard;
#endif
    bool serviceAcousticGpuRasterIdle();
    std::shared_ptr<RayFusion::ProbeOverlay> m_probeOverlay;
    void recordRayFusionProbeOverlay(VkCommandBuffer cmd, const Matrix4x4& viewProj);
    // Particle billboards (VulkanViewportParticles.cpp).
    void ensureParticleBillboardPipelines(const std::string& shaderDir);
    bool recordParticleBillboardCompaction(VkCommandBuffer cmd);
    void makeParticleBillboardCompactionVisible(VkCommandBuffer cmd);
    void recordParticleBillboards(VkCommandBuffer cmd, const float viewProjGL[16],
                                  const float viewGL[16], bool usingFrameRing);
    void destroyParticleBillboardResources(bool keepPipeline);
    ParticleBillboardFrameBuffer m_syncBillboardVertices;
    std::unique_ptr<RasterViewportFrameRing> m_rasterFrameRing;
    // A failed ring uses synchronous presentation only for a bounded recovery
    // window. A transient queue/fence collision must not penalize the session
    // forever; repeated failures back off to avoid create/destroy every frame.
    bool m_rasterFrameRingUnavailable = false;
    std::uint64_t m_rasterFrameSequence = 0;
    std::uint64_t m_rasterFrameRingRetryFrame = 0;
    std::uint32_t m_rasterFrameRingFailureCount = 0;
    bool m_hasRasterPresentedFrame = false;
    // Device-lost tek satir raporlanir; kurtarma Main'in backend_changed
    // yolunda. Bkz. renderInteractiveViewportImpl'deki VK_ERROR_DEVICE_LOST dali.
    bool m_loggedRasterDeviceLost = false;
    // ★ Doku kusagi tripwire'i oturumda BIR KEZ raporlanir; her karede basmak
    //   arizayi degil yeniden deneme dongusunu gosterirdi (ayni hatayi
    //   device-lost yolunda bir kez yedik).
    bool m_loggedStaleMaterialPreviewDescSet = false;
    // Log bir kez yazilir ama SAYAC her olayda artar: tekrar eden bir purge ile
    // tek seferlik bir yarisi ancak bu sayi ayirt eder.
    std::uint64_t m_staleMaterialPreviewDescSetEvents = 0;
    // Instances in the last recorded sphere-impostor draw (0 = not drawn).
    std::uint64_t m_sphereImpostorsDrawn = 0;
    bool m_rasterSynchronousPresent = false;
    RasterFrameTelemetry m_rasterTelemetry;
    VkBuffer m_externalMaterialBuffer = VK_NULL_HANDLE;
    VkDeviceSize m_externalMaterialBufferSize = 0;
    uint32_t m_externalMaterialCount = 0;

    // ── Material preview scene-feature specialization (O3 / O3c) ────────────
    // material_preview_frag carries spec constants (ids 0..7, default true).
    // The main + covered pipelines are created with only the features the scene
    // can reach, and rebuilt when a new one appears.
    // Implementation: Viewport/MaterialPreviewCoverage.cpp.
public:
    static constexpr uint32_t kMpSpecTerrain       = 1u << 0;
    static constexpr uint32_t kMpSpecGraph         = 1u << 1;
    static constexpr uint32_t kMpSpecWeather       = 1u << 2;
    static constexpr uint32_t kMpSpecGlass         = 1u << 3;
    static constexpr uint32_t kMpSpecExtLobes      = 1u << 4;
    static constexpr uint32_t kMpSpecAltLighting   = 1u << 5;
    static constexpr uint32_t kMpSpecFallbackSky   = 1u << 6;
    static constexpr uint32_t kMpSpecRayFusionGi   = 1u << 7;
    static constexpr uint32_t kMpSpecOptTex        = 1u << 8;
    static constexpr uint32_t kMpSpecClearcoat     = 1u << 9;
    static constexpr uint32_t kMpSpecPaintCanvas   = 1u << 10;
    static constexpr uint32_t kMpSpecClearcoatIrid = 1u << 11;
    static constexpr uint32_t kMpSpecAll           = kMpSpecTerrain | kMpSpecGraph | kMpSpecWeather |
                                                     kMpSpecGlass | kMpSpecExtLobes |
                                                     kMpSpecAltLighting | kMpSpecFallbackSky |
                                                     kMpSpecRayFusionGi | kMpSpecOptTex |
                                                     kMpSpecClearcoat | kMpSpecPaintCanvas |
                                                     kMpSpecClearcoatIrid;
    static constexpr uint32_t kMpSpecCount         = 13; // 0..9 bool, 10 uint texPages, 11..12 bool
    struct MaterialPreviewSpecStorage {
        VkSpecializationMapEntry entries[kMpSpecCount]{};
        uint32_t values[kMpSpecCount]{};
        VkSpecializationInfo info{};
    };
    static void fillMaterialPreviewSpec(uint32_t mask, MaterialPreviewSpecStorage& out,
                                        uint32_t texPages = 32u);
private:
    struct MaterialPreviewSpecRecipe;
    std::shared_ptr<MaterialPreviewSpecRecipe> m_mpSpecRecipe;
    VkPipeline m_mpGlassOpaquePipeline = VK_NULL_HANDLE;
    VkPipeline m_mpGlassTransmissionPipeline = VK_NULL_HANDLE;
    VkRenderPass m_mpGlassTransmissionRenderPass = VK_NULL_HANDLE;
    uint32_t m_mpGlassMask = 0;
    uint32_t m_mpGlassTexPages = 0;
    uint32_t m_mpSpecMask = kMpSpecAll;      // what the live pipelines were built with
    uint32_t m_mpSpecTexPages = 32u;         // active material texture array pages (1..32)
    uint32_t m_mpSpecShrinkFrames = 0;       // debounce before dropping a feature
    uint32_t m_mpSpecRebuilds = 0;
    bool m_mpSpecFailed = false;
    VkPipeline m_postPipelineFast = VK_NULL_HANDLE;
    VkRenderPass m_hdrRenderPassNoMotion = VK_NULL_HANDLE;
    RasterPostFastResources m_postFast{};
    uint32_t materialPreviewSceneFeatureMask() const;
    uint32_t materialPreviewActiveTexPages() const;
    void captureMaterialPreviewSpecRecipe(const VkGraphicsPipelineCreateInfo& info,
                                          const std::string& shaderDir, uint32_t mask,
                                          uint32_t texPages = 32u);
    void ensureMaterialPreviewGlassPipelines();
    void destroyMaterialPreviewGlassPipelines();
    void serviceMaterialPreviewFeatureSpec();
protected:
    RasterAppliedState sampleRasterAppliedState() const override;
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
    bool updateRasterMeshFromMeshSoA(const std::string& nodeName, const TriangleMesh* mesh,
                                     const std::vector<uint32_t>* dirtyVertices = nullptr) override;
    // Counts hinted refits so every Nth one re-runs the full diff as an audit.
    uint64_t m_soaRefitHintedCalls = 0;
protected:
    bool ensureInteractiveViewportResourcesImpl(const std::string& shaderDir, int width, int height) override;
    void destroyInteractiveViewportResourcesImpl(bool keepPipeline = false) override;
    void renderInteractiveViewportImpl(void* outSurface, int width, int height,
                                       void* outFramebuffer, void* outTexture) override;
    void drainInteractiveViewportInFlight() override;

    // ── Realtime post (alan derinligi + goruntuleme donusumu) ───────────────
    // ★★★ Bu iki fonksiyon TEK bir seyin iki yarisidir: kaynak (descriptor) ve
    //   kayit (dispatch). Ayri tutuluyorlar cunku descriptor'lar YENIDEN
    //   BOYUTLANDIRMADA, dispatch ise HER KAREDE kosar.
    bool updateRasterPostDescriptors();
    bool recordRasterPostPass(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                              const VkViewport& viewport, const VkRect2D& scissor);

    // ── TAA (raster_taa.comp) ───────────────────────────────────────────────
    // Ayni ikilik: kaynak kurulumu yeniden boyutlandirmada, kayit her karede.
    // `ensure` pipeline'i bir kez kurar; basarisizlik OLUMCUL DEGIL -- TAA
    // kapanir ve viewport eskisi gibi (jittersiz, gecmissiz) cizer.
    bool ensureRasterTaaResources(uint32_t width, uint32_t height);
    void destroyRasterTaaResources(bool keepPipeline);
    // `blend` ve gecerlilik cagiranda hesaplanir; bu fonksiyon yalnizca kaydeder.
    void recordRasterTaaPass(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                             const Matrix4x4& invViewProjUnjittered);

    std::shared_ptr<struct RasterSkinningBatchState> m_rasterSkinningBatch;
};

} // namespace Backend
