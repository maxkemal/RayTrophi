#include "Viewport/RasterGpuCull.h"

#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace Backend {

namespace {

// VulkanViewportBackend.cpp'deki loadViewportSPV ile ayni; o dosya-statik
// oldugu icin buradan gorunmuyor.
std::vector<uint32_t> loadCullSPV(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) return {};
    std::vector<uint32_t> buffer(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) return {};
    return buffer;
}

// Globals blogu, raster_cull.comp icindeki std430 duzeniyle BIREBIR.
//   planes[6]              offset   0, 96 bayt
//   cameraPos              offset  96, 16 bayt
//   scatterTriangleTarget  offset 112
//   lodMinDistanceSq       offset 116
//   lodMaxDistanceSq       offset 120
//   meshCount              offset 124
//   lodBandWidth           offset 128
struct GlobalsGPU {
    float    planes[24];
    float    cameraPos[4];
    uint32_t scatterTriangleTarget;
    float    lodMinDistanceSq;
    float    lodMaxDistanceSq;
    uint32_t meshCount;
    float    lodBandWidth;
};
static_assert(sizeof(GlobalsGPU) == 132, "GlobalsGPU must match raster_cull.comp std430 layout");

struct MeshStateGPU {
    float lodDistanceSq;
    float lastFull;
    float lastProxy;
    float lastVisible;
};
static_assert(sizeof(MeshStateGPU) == 16, "MeshStateGPU must match raster_cull.comp state[]");

constexpr uint32_t kLocalSize = 256;

} // namespace

static_assert(sizeof(RasterGpuCull::MeshBinding) == 48,
              "MeshBinding must match the MeshParam struct in raster_cull.comp");

struct RasterGpuCull::Impl {
    VulkanRT::VulkanDevice* device = nullptr;

    VkPipeline            pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout     = VK_NULL_HANDLE;
    VkDescriptorPool      descPool       = VK_NULL_HANDLE;
    VkDescriptorSet       descSet        = VK_NULL_HANDLE;

    VulkanRT::BufferHandle bounds;      // CPU_TO_GPU
    VulkanRT::BufferHandle meshParams;  // CPU_TO_GPU
    VulkanRT::BufferHandle dstMatrices; // GPU_ONLY  (STORAGE | VERTEX)
    VulkanRT::BufferHandle counters;    // GPU_ONLY  (STORAGE | TRANSFER_DST)
    VulkanRT::BufferHandle cmds;        // GPU_ONLY  (STORAGE | INDIRECT)
    VulkanRT::BufferHandle globals;     // CPU_TO_GPU
    VulkanRT::BufferHandle state;       // CPU_TO_GPU (shader yazar, CPU okur)

    void* boundsMapped     = nullptr;
    void* meshParamsMapped = nullptr;
    void* globalsMapped    = nullptr;
    void* stateMapped      = nullptr;

    uint32_t instanceCapacity = 0;   // kaynak (bounds + src matris)
    uint32_t outCapacity      = 0;   // sikistirilmis cikti yuvasi
    uint32_t meshCapacity     = 0;
    uint32_t drawSlotCapacity = 0;

    // Descriptor'i yalnizca kaynak buffer degistiginde yeniden yaz.
    VkBuffer boundSrcInstanceBuffer = VK_NULL_HANDLE;

    std::vector<MeshBinding> bindings;
    uint32_t activeMeshCount = 0;

    void destroyBuffers() {
        if (!device) return;
        auto drop = [&](VulkanRT::BufferHandle& b, void*& mapped) {
            if (b.buffer) {
                if (mapped) { device->unmapBuffer(b); mapped = nullptr; }
                device->destroyBuffer(b);
                b = VulkanRT::BufferHandle{};
            }
        };
        void* none = nullptr;
        drop(bounds, boundsMapped);
        drop(meshParams, meshParamsMapped);
        drop(globals, globalsMapped);
        drop(state, stateMapped);
        drop(dstMatrices, none);
        drop(counters, none);
        drop(cmds, none);
        boundSrcInstanceBuffer = VK_NULL_HANDLE;
    }
};

RasterGpuCull::RasterGpuCull() : impl_(std::make_unique<Impl>()) {}
RasterGpuCull::~RasterGpuCull() { destroy(); }

bool RasterGpuCull::isReady() const {
    return impl_->pipeline != VK_NULL_HANDLE &&
           impl_->dstMatrices.buffer != VK_NULL_HANDLE &&
           impl_->cmds.buffer != VK_NULL_HANDLE;
}

bool RasterGpuCull::ensure(VulkanRT::VulkanDevice& device,
                           const std::string& shaderDir,
                           uint32_t instanceCapacity,
                           uint32_t outCapacity,
                           uint32_t meshCount,
                           uint32_t drawSlotCount) {
    if (instanceCapacity == 0 || outCapacity == 0 ||
        meshCount == 0 || drawSlotCount == 0) return false;

    if (impl_->device && impl_->device != &device) {
        destroy();
    }
    impl_->device = &device;
    VkDevice vk = device.getDevice();
    if (vk == VK_NULL_HANDLE) return false;

    // ---- pipeline (bir kez) ------------------------------------------------
    if (impl_->pipeline == VK_NULL_HANDLE) {
        const std::string spvPath = shaderDir + "/raster_cull.spv";
        std::vector<uint32_t> spv = loadCullSPV(spvPath);
        if (spv.empty()) {
            SCENE_LOG_WARN("[RasterCull] raster_cull.spv okunamadi: " + spvPath +
                           " -- GPU culling kapali kalacak.");
            return false;
        }

        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 };
        VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pi.poolSizeCount = 1; pi.pPoolSizes = &ps; pi.maxSets = 1;
        if (vkCreateDescriptorPool(vk, &pi, nullptr, &impl_->descPool) != VK_SUCCESS) {
            SCENE_LOG_WARN("[RasterCull] descriptor pool olusturulamadi.");
            return false;
        }

        VkDescriptorSetLayoutBinding b[8]{};
        for (uint32_t i = 0; i < 8; ++i) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        li.bindingCount = 8; li.pBindings = b;
        if (vkCreateDescriptorSetLayout(vk, &li, nullptr, &impl_->descLayout) != VK_SUCCESS) {
            destroy(); return false;
        }

        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2 };
        VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pli.setLayoutCount = 1; pli.pSetLayouts = &impl_->descLayout;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(vk, &pli, nullptr, &impl_->pipelineLayout) != VK_SUCCESS) {
            destroy(); return false;
        }

        VkShaderModuleCreateInfo si{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        si.codeSize = spv.size() * 4; si.pCode = spv.data();
        VkShaderModule sm = VK_NULL_HANDLE;
        if (vkCreateShaderModule(vk, &si, nullptr, &sm) != VK_SUCCESS) { destroy(); return false; }

        VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        ci.layout = impl_->pipelineLayout;
        ci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = sm; ci.stage.pName = "main";
        VkResult r = vkCreateComputePipelines(vk, VK_NULL_HANDLE, 1, &ci, nullptr, &impl_->pipeline);
        vkDestroyShaderModule(vk, sm, nullptr);
        if (r != VK_SUCCESS) {
            SCENE_LOG_WARN("[RasterCull] compute pipeline olusturulamadi.");
            destroy(); return false;
        }

        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = impl_->descPool; ai.descriptorSetCount = 1;
        ai.pSetLayouts = &impl_->descLayout;
        if (vkAllocateDescriptorSets(vk, &ai, &impl_->descSet) != VK_SUCCESS) {
            destroy(); return false;
        }
    }

    // ---- buffer'lar (geometrik buyume, asla kucultme) -----------------------
    const bool needGrow =
        instanceCapacity > impl_->instanceCapacity ||
        outCapacity      > impl_->outCapacity      ||
        meshCount        > impl_->meshCapacity     ||
        drawSlotCount    > impl_->drawSlotCapacity ||
        impl_->dstMatrices.buffer == VK_NULL_HANDLE;

    if (needGrow) {
        const uint32_t newInst  = (std::max)(instanceCapacity,
                                             impl_->instanceCapacity + impl_->instanceCapacity / 2u);
        const uint32_t newOut   = (std::max)(outCapacity,
                                             impl_->outCapacity + impl_->outCapacity / 2u);
        const uint32_t newMesh  = (std::max)(meshCount,
                                             impl_->meshCapacity + impl_->meshCapacity / 2u);
        const uint32_t newSlots = (std::max)(drawSlotCount,
                                             impl_->drawSlotCapacity + impl_->drawSlotCapacity / 2u);

        impl_->destroyBuffers();

        auto make = [&](uint64_t size, VulkanRT::BufferUsage usage,
                        VulkanRT::MemoryLocation loc) {
            VulkanRT::BufferCreateInfo ci{};
            ci.size = size; ci.usage = usage; ci.location = loc; ci.initialData = nullptr;
            return device.createBuffer(ci);
        };
        using U = VulkanRT::BufferUsage;
        using M = VulkanRT::MemoryLocation;

        impl_->bounds      = make(uint64_t(newInst)  * 16, U::STORAGE, M::CPU_TO_GPU);
        impl_->meshParams  = make(uint64_t(newMesh)  * sizeof(MeshBinding), U::STORAGE, M::CPU_TO_GPU);
        impl_->globals     = make(sizeof(GlobalsGPU), U::STORAGE, M::CPU_TO_GPU);
        impl_->state       = make(uint64_t(newMesh)  * sizeof(MeshStateGPU), U::STORAGE, M::CPU_TO_GPU);
        impl_->dstMatrices = make(uint64_t(newOut)   * 64, U::STORAGE | U::VERTEX, M::GPU_ONLY);
        // ★ Mesh basina DORT uint: full, proxy, maxDistSqBits, reserved.
        //   raster_cull.comp counters[meshIndex * 4u + k] ile indeksliyor;
        //   burayi 2'ye dusurursen sinir disina yazar.
        impl_->counters    = make(uint64_t(newMesh)  * 16, U::STORAGE | U::TRANSFER_DST, M::GPU_ONLY);
        impl_->cmds        = make(uint64_t(newSlots) * kCommandStride,
                                  U::STORAGE | U::INDIRECT | U::TRANSFER_DST, M::GPU_ONLY);

        if (!impl_->bounds.buffer || !impl_->meshParams.buffer || !impl_->globals.buffer ||
            !impl_->state.buffer || !impl_->dstMatrices.buffer || !impl_->counters.buffer ||
            !impl_->cmds.buffer) {
            SCENE_LOG_WARN("[RasterCull] buffer ayirma basarisiz; GPU culling kapali.");
            impl_->destroyBuffers();
            return false;
        }

        impl_->boundsMapped     = device.mapBuffer(impl_->bounds);
        impl_->meshParamsMapped = device.mapBuffer(impl_->meshParams);
        impl_->globalsMapped    = device.mapBuffer(impl_->globals);
        impl_->stateMapped      = device.mapBuffer(impl_->state);
        if (!impl_->boundsMapped || !impl_->meshParamsMapped ||
            !impl_->globalsMapped || !impl_->stateMapped) {
            SCENE_LOG_WARN("[RasterCull] mapleme basarisiz; GPU culling kapali.");
            impl_->destroyBuffers();
            return false;
        }

        std::memset(impl_->boundsMapped, 0, uint64_t(newInst) * 16);
        std::memset(impl_->meshParamsMapped, 0, uint64_t(newMesh) * sizeof(MeshBinding));
        std::memset(impl_->globalsMapped, 0, sizeof(GlobalsGPU));

        // ★ LOD esigini "cok uzak" ile tohumla: ilk kare her seyi TAM cizer,
        //   sonra butceye dogru yakinsar. Tersi (kucuk tohum) ilk kareyi
        //   tamamen proxy gosterirdi ve bu bir arizaya benzerdi.
        auto* st = static_cast<MeshStateGPU*>(impl_->stateMapped);
        for (uint32_t i = 0; i < newMesh; ++i) {
            st[i] = MeshStateGPU{ 1.0e18f, 0.0f, 0.0f, 0.0f };
        }

        impl_->instanceCapacity = newInst;
        impl_->outCapacity      = newOut;
        impl_->meshCapacity     = newMesh;
        impl_->drawSlotCapacity = newSlots;
        impl_->boundSrcInstanceBuffer = VK_NULL_HANDLE; // descriptor yeniden yazilmali
    }

    return true;
}

void RasterGpuCull::setGlobals(const float planes[24],
                               const float cameraPos[3],
                               uint32_t scatterTriangleTarget,
                               float lodMinDistanceSq,
                               float lodMaxDistanceSq,
                               uint32_t meshCount,
                               float lodBandWidth) {
    if (!impl_->globalsMapped) return;
    GlobalsGPU g{};
    std::memcpy(g.planes, planes, sizeof(float) * 24);
    g.cameraPos[0] = cameraPos[0];
    g.cameraPos[1] = cameraPos[1];
    g.cameraPos[2] = cameraPos[2];
    g.cameraPos[3] = 0.0f;
    g.scatterTriangleTarget = scatterTriangleTarget;
    g.lodMinDistanceSq = lodMinDistanceSq;
    g.lodMaxDistanceSq = lodMaxDistanceSq;
    g.meshCount = meshCount;
    g.lodBandWidth = lodBandWidth;
    std::memcpy(impl_->globalsMapped, &g, sizeof(g));
}

void RasterGpuCull::setMeshBindings(const std::vector<MeshBinding>& bindings) {
    impl_->bindings = bindings;
    impl_->activeMeshCount =
        (std::min)(static_cast<uint32_t>(bindings.size()), impl_->meshCapacity);
    if (!impl_->meshParamsMapped || impl_->activeMeshCount == 0) return;
    std::memcpy(impl_->meshParamsMapped, bindings.data(),
                static_cast<size_t>(impl_->activeMeshCount) * sizeof(MeshBinding));
}

void RasterGpuCull::setBounds(uint32_t firstInstance,
                              const float* centerRadius4,
                              uint32_t count) {
    if (!impl_->boundsMapped || count == 0 || !centerRadius4) return;
    if (static_cast<uint64_t>(firstInstance) + count > impl_->instanceCapacity) return;
    std::memcpy(static_cast<uint8_t*>(impl_->boundsMapped) +
                    static_cast<size_t>(firstInstance) * 16,
                centerRadius4, static_cast<size_t>(count) * 16);
}

void RasterGpuCull::record(void* commandBuffer, void* srcInstanceBuffer, uint32_t meshCount) {
    if (!isReady() || !impl_->device) return;
    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(commandBuffer);
    VkBuffer src = static_cast<VkBuffer>(srcInstanceBuffer);
    if (cmd == VK_NULL_HANDLE || src == VK_NULL_HANDLE) return;

    const uint32_t meshes = (std::min)(meshCount, impl_->activeMeshCount);
    if (meshes == 0) return;

    VkDevice vk = impl_->device->getDevice();

    // Kaynak instance buffer'i buyudugunde yeniden olusur; descriptor o zaman
    // yeniden yazilir. Bunu atlarsan SILINMIS bir buffer'dan okursun.
    if (impl_->boundSrcInstanceBuffer != src) {
        const VkBuffer bufs[8] = {
            src, impl_->bounds.buffer, impl_->meshParams.buffer, impl_->dstMatrices.buffer,
            impl_->counters.buffer, impl_->cmds.buffer, impl_->globals.buffer, impl_->state.buffer
        };
        VkDescriptorBufferInfo bi[8]{};
        VkWriteDescriptorSet w[8]{};
        for (uint32_t i = 0; i < 8; ++i) {
            bi[i].buffer = bufs[i]; bi[i].offset = 0; bi[i].range = VK_WHOLE_SIZE;
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = impl_->descSet; w[i].dstBinding = i; w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(vk, 8, w, 0, nullptr);
        impl_->boundSrcInstanceBuffer = src;
    }

    auto barrier = [&](VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                       VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = srcAccess;
        mb.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };

    // 1) Sayaclari sifirla. Bunu atlamak "her kare bir oncekinin ustune say"
    //    demektir; instanceCount saticak ve cizim bolge disina tasacakti.
    vkCmdFillBuffer(cmd, impl_->counters.buffer, 0,
                    static_cast<VkDeviceSize>(meshes) * 16, 0);

    // ★★★ Komut buffer'ini da sifirla. HER cizilebilir mesh bir yuva alir ama
    //     HER yuva yazilmaz: LOD ayrimi kapali bir scatter grubunun proxy'si,
    //     ya da sahibi olmayan bir mesh yuvasi bos kalir. Sifirlamazsak o yuva
    //     BASLATILMAMIS GPU bellegi olur ve vkCmdDraw*Indirect onu gecerli bir
    //     komut sanip cop instanceCount ile cizer -- device lost'un sessiz
    //     kaynagi. Sifir dolgu = elementCount 0, instanceCount 0 = gecerli
    //     no-op. Maliyet birkac KB.
    vkCmdFillBuffer(cmd, impl_->cmds.buffer, 0,
                    static_cast<VkDeviceSize>(impl_->drawSlotCapacity) * kCommandStride, 0);
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipelineLayout,
                            0, 1, &impl_->descSet, 0, nullptr);

    // 2) pass 0 -- mesh basina bir dispatch (mesh sayisi onlarca mertebesinde;
    //    is instance sayisiyla olcekleniyor, dispatch SAYISIYLA degil).
    for (uint32_t i = 0; i < meshes; ++i) {
        const uint32_t n = impl_->bindings[i].instanceCount;
        if (n == 0) continue;
        const uint32_t push[2] = { 0u, i };
        vkCmdPushConstants(cmd, impl_->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), push);
        vkCmdDispatch(cmd, (n + kLocalSize - 1) / kLocalSize, 1, 1);
    }

    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // 3) pass 1 -- indirect komutlar + LOD esigi
    {
        const uint32_t push[2] = { 1u, 0u };
        vkCmdPushConstants(cmd, impl_->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), push);
        vkCmdDispatch(cmd, (meshes + kLocalSize - 1) / kLocalSize, 1, 1);
    }

    // 4) Cizim bunlari indirect komut ve vertex attribute olarak okuyacak.
    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
}

void* RasterGpuCull::compactedInstanceBuffer() const { return impl_->dstMatrices.buffer; }
void* RasterGpuCull::commandBuffer() const { return impl_->cmds.buffer; }

bool RasterGpuCull::readMeshResult(uint32_t meshIndex, MeshResult& out) const {
    if (!impl_->stateMapped || meshIndex >= impl_->activeMeshCount) return false;
    const auto* st = static_cast<const MeshStateGPU*>(impl_->stateMapped) + meshIndex;
    out.lodDistanceSq  = st->lodDistanceSq;
    out.fullInstances  = static_cast<uint32_t>(st->lastFull);
    out.proxyInstances = static_cast<uint32_t>(st->lastProxy);
    return true;
}

void RasterGpuCull::destroy() {
    if (!impl_ || !impl_->device) { if (impl_) impl_->device = nullptr; return; }
    VkDevice vk = impl_->device->getDevice();
    impl_->destroyBuffers();
    if (vk != VK_NULL_HANDLE) {
        if (impl_->pipeline)       vkDestroyPipeline(vk, impl_->pipeline, nullptr);
        if (impl_->pipelineLayout) vkDestroyPipelineLayout(vk, impl_->pipelineLayout, nullptr);
        if (impl_->descLayout)     vkDestroyDescriptorSetLayout(vk, impl_->descLayout, nullptr);
        if (impl_->descPool)       vkDestroyDescriptorPool(vk, impl_->descPool, nullptr);
    }
    impl_->pipeline = VK_NULL_HANDLE;
    impl_->pipelineLayout = VK_NULL_HANDLE;
    impl_->descLayout = VK_NULL_HANDLE;
    impl_->descPool = VK_NULL_HANDLE;
    impl_->descSet = VK_NULL_HANDLE;
    impl_->instanceCapacity = 0;
    impl_->outCapacity = 0;
    impl_->meshCapacity = 0;
    impl_->drawSlotCapacity = 0;
    impl_->activeMeshCount = 0;
    impl_->bindings.clear();
    impl_->device = nullptr;
}

} // namespace Backend
