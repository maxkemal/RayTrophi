#include "Viewport/RayFusionViewportPasses.h"
#include "Backend/VulkanStartupDiagnostics.h"
#include "Backend/VulkanBackend.h"
#include "Backend/VulkanMaterialTextureBinding.h"
#include "globals.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <vector>

namespace Backend {

namespace {

struct RayFusionFrameUniform {
    float view[16];
    float lightDir2[4];
};
static_assert(sizeof(RayFusionFrameUniform) == 80);

} // namespace

bool VulkanBackendAdapter::ensureMaterialPreviewFrameUniform() {
    auto& viewport = m_interactiveViewport;
    if (viewport.materialPreviewFrameUniform.buffer) {
        return true;
    }
    if (!m_device || !viewport.materialPreviewDescSet) {
        return false;
    }

    VulkanRT::BufferCreateInfo info{};
    info.size = sizeof(RayFusionFrameUniform);
    info.usage = VulkanRT::BufferUsage::UNIFORM | VulkanRT::BufferUsage::TRANSFER_DST;
    info.location = VulkanRT::MemoryLocation::GPU_ONLY;
    viewport.materialPreviewFrameUniform = m_device->createBuffer(info);
    const VkBuffer buffer = viewport.materialPreviewFrameUniform.buffer;
    if (!buffer) {
        return false;
    }

    const VkDescriptorBufferInfo descriptor{buffer, 0, sizeof(RayFusionFrameUniform)};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = viewport.materialPreviewDescSet;
    write.dstBinding = kRayFusionFrameUniformBinding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &descriptor;
    vkUpdateDescriptorSets(m_device->getDevice(), 1, &write, 0, nullptr);
    return true;
}

void VulkanBackendAdapter::recordMaterialPreviewFrameUniform(
    VkCommandBuffer cmd, const Matrix4x4& view) {
    const VkBuffer buffer = m_interactiveViewport.materialPreviewFrameUniform.buffer;
    if (!buffer) {
        return;
    }

    RayFusionFrameUniform frame{{}, {-0.2f, 0.4f, -0.8f, 1.2f}};
    const Matrix4x4 transposed = view.transpose();
    std::memcpy(frame.view, transposed.m, sizeof(frame.view));

    // A previous queued frame may still be reading this buffer.
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.size = sizeof(frame);
    barrier.srcAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    vkCmdUpdateBuffer(cmd, buffer, 0, sizeof(frame), &frame);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void VulkanBackendAdapter::destroyMaterialPreviewFrameUniform() {
    auto& buffer = m_interactiveViewport.materialPreviewFrameUniform;
    if (buffer.buffer) {
        m_device->destroyBuffer(buffer);
        buffer = {};
    }
}

std::string resolveRayFusionShaderDirectory() {
    for (const char* directory : {"shaders", "source/shaders", "../shaders"}) {
        if (std::filesystem::exists(std::filesystem::path(directory) / "material_preview.spv")) {
            return directory;
        }
    }
    return (std::filesystem::current_path() / "shaders").string();
}

VkResult selectRayFusionInstanceVersion(PFN_vkGetInstanceProcAddr getProc, uint32_t& version) {
    const auto enumerate = getProc ? reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        getProc(VK_NULL_HANDLE, "vkEnumerateInstanceVersion")) : nullptr;
    if (!enumerate) {
#ifdef RT_PLAYER
        if (currentVulkanStartup && getProc) {
            currentVulkanStartup->loaderApi = VK_API_VERSION_1_0;
        }
#endif
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    }
    uint32_t reportedVersion = 0;
    const VkResult result = enumerate(&reportedVersion);
    if (result != VK_SUCCESS) {
        return result;
    }
    const uint32_t loaderVersion = androidDiagnosticVulkanApi(
        reportedVersion, "debug.raytrophi.force_loader_api");
#ifdef RT_PLAYER
    if (currentVulkanStartup) {
        currentVulkanStartup->loaderApi = loaderVersion;
    }
#endif
#if defined(__ANDROID__) && defined(RT_PLAYER)
    if (loaderVersion >= VK_API_VERSION_1_1 && loaderVersion < VK_API_VERSION_1_2) {
        // Physical-device commands needed before vkCreateDevice are core 1.1 or older.
        version = VK_API_VERSION_1_2;
        return VK_SUCCESS;
    }
#endif
    if (loaderVersion < VK_API_VERSION_1_2) {
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    }
    version = (std::min)(loaderVersion, VK_API_VERSION_1_3);
    return VK_SUCCESS;
}

namespace {
std::mutex g_rejection_mutex;
std::string g_rejections;
} // namespace

void recordRayFusionDeviceRejection(const char* adapter, const char* reason) {
    std::lock_guard<std::mutex> lock(g_rejection_mutex);
    if (!g_rejections.empty()) g_rejections += "; ";
    g_rejections += std::string(adapter ? adapter : "?") + ": " + (reason ? reason : "?");
}

#if defined(__ANDROID__) && defined(RT_PLAYER)
void recordAndroidRayFusionDeviceRejection(const VkPhysicalDeviceProperties& props,
                                          const char* reason) {
    const std::string adapter = std::string(props.deviceName) + " (driver version " +
                                std::to_string(props.driverVersion) + ")";
    const uint32_t loader = currentVulkanStartup ? currentVulkanStartup->loaderApi : 0;
    const std::string detail = std::string(reason) + " — Vulkan " +
        vulkanApiString(props.apiVersion) + ", 1.1 + scalar block layout + buffer device address "
        "required (loader " +
        vulkanApiString(loader) + ", device " + vulkanApiString(props.apiVersion) + ")";
    recordRayFusionDeviceRejection(adapter.c_str(), detail.c_str());
}
#endif

void clearRayFusionDeviceRejections() {
    std::lock_guard<std::mutex> lock(g_rejection_mutex);
    g_rejections.clear();
}

std::string rayFusionDeviceRejections() {
    std::lock_guard<std::mutex> lock(g_rejection_mutex);
    return g_rejections;
}

const char* rayFusionDeviceRejection(
    VkPhysicalDevice device, const VkPhysicalDeviceProperties& properties) {
    // Vulkan 1.1 devices (Mali-G76 Omix X700, Adreno 610) are accepted when the
    // two 1.2-core features the Player shaders rely on exist as extensions:
    // scalar block layout (every RayFusion SSBO) and buffer device address
    // (SDF surface / volume / NanoVDB). Android ships SPIR-V 1.3 for this
    // (scripts/compile_player_android_vk11.py); descriptor indexing is no
    // longer needed (texture arrays). Nothing else from 1.2 core is called.
    if (properties.apiVersion < VK_API_VERSION_1_1) {
        return "device API below Vulkan 1.1";
    }
    if (properties.apiVersion < VK_API_VERSION_1_2) {
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());
        const auto has = [&](const char* name) {
            for (const auto& e : extensions)
                if (std::strcmp(e.extensionName, name) == 0) return true;
            return false;
        };
        if (!has(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME)) {
            return "Vulkan 1.1 device without VK_EXT_scalar_block_layout";
        }
        if (!has(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
            return "Vulkan 1.1 device without VK_KHR_buffer_device_address";
        }
    }

#if defined(__ANDROID__) && defined(RT_PLAYER)
    const auto& limits = properties.limits;
    const uint32_t storageBindings = (std::min)(
        limits.maxPerStageDescriptorStorageBuffers, limits.maxDescriptorSetStorageBuffers);
    if (storageBindings < 15) {
        return "insufficient SSBO bindings for sim_particle_collide (15) and paint_wet (17)";
    }
    if (storageBindings < 17) {
        return "insufficient SSBO bindings for paint_wet (17)";
    }
#endif

    // Material textures use 32 sampler2DArray pages with constant indices: no
    // descriptor indexing feature is required (Mali-G52 / Lenovo TB330FU).
    VkPhysicalDeviceFeatures coreFeatures{};
    Backend::MaterialTextureBinding::queryCoreFeatures(device, coreFeatures);
    if (!coreFeatures.independentBlend || !coreFeatures.fragmentStoresAndAtomics) {
        return "independentBlend or fragmentStoresAndAtomics unavailable";
    }

    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, queues.data());
    constexpr VkQueueFlags required = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    for (uint32_t i = 0; i < queueCount; ++i) {
        if (queues[i].queueCount > 0 && (queues[i].queueFlags & required) == required) {
            return nullptr;
        }
    }
    return "no combined graphics and compute queue";
}

void logRayFusionDeviceLimits(
    VkPhysicalDevice device, const VkPhysicalDeviceLimits& limits) {
    const uint32_t storageBindings = (std::min)(
        limits.maxPerStageDescriptorStorageBuffers, limits.maxDescriptorSetStorageBuffers);
    SCENE_LOG_INFO("[Player GPU] SSBO per_stage=" +
                   std::to_string(limits.maxPerStageDescriptorStorageBuffers) +
                   " per_set=" + std::to_string(limits.maxDescriptorSetStorageBuffers) +
                   " max_range=" + std::to_string(limits.maxStorageBufferRange) +
                   " collision_15=" + (storageBindings >= 15 ? "yes" : "no") +
                   " wet_paint_17=" + (storageBindings >= 17 ? "yes" : "no"));

    struct ImageRequirement {
        const char* name;
        VkFormat format;
        VkImageUsageFlags usage;
        VkFormatFeatureFlags features = 0;
    };
    constexpr VkImageUsageFlags sampledColor =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const ImageRequirement images[] = {
        {"LDR post", VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        {"HDR scene", VK_FORMAT_R16G16B16A16_SFLOAT, sampledColor |
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT},
        {"HDR normals", kRfGbufferNormalFormat, sampledColor},
        {"HDR previous clip", VK_FORMAT_R32G32B32A32_SFLOAT, sampledColor},
        {"Scene depth", VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT},
        {"Transmission color snapshot", VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        {"Transmission depth snapshot", VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        {"Transmission back depth", VK_FORMAT_R32_UINT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT},
    };
    for (const auto& image : images) {
        VkFormatProperties format{};
        vkGetPhysicalDeviceFormatProperties(device, image.format, &format);
        VkImageFormatProperties imageProperties{};
        const VkResult result = vkGetPhysicalDeviceImageFormatProperties(
            device, image.format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
            image.usage, 0, &imageProperties);
        const bool featuresPresent =
            (format.optimalTilingFeatures & image.features) == image.features;
        SCENE_LOG_INFO(std::string("[Player GPU] Format ") + image.name +
                       " vk_format=" + std::to_string(image.format) +
                       " usage_result=" + std::to_string(static_cast<int>(result)) +
                       " required_features=" + (featuresPresent ? "yes" : "no") +
                       " max_extent=" + std::to_string(imageProperties.maxExtent.width) +
                       "x" + std::to_string(imageProperties.maxExtent.height));
    }
}

uint32_t rayFusionMaterialTextureCapacity(
    const VkPhysicalDeviceLimits& limits,
    const VkDescriptorSetLayoutBinding* bindings,
    uint32_t bindingCount,
    uint32_t requestedCapacity) {
    uint32_t fixedSamplers = 0;
    uint32_t storageBuffers = 0;
    uint32_t storageImages = 0;
    uint32_t uniformBuffers = 0;
    for (uint32_t i = 0; i < bindingCount; ++i) {
        const auto& binding = bindings[i];
        switch (binding.descriptorType) {
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            if (binding.binding != 1u) {
                fixedSamplers += binding.descriptorCount;
            }
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            storageBuffers += binding.descriptorCount;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            storageImages += binding.descriptorCount;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            uniformBuffers += binding.descriptorCount;
            break;
        default:
            return 0;
        }
    }

    constexpr uint32_t colorAttachments = 4;
    if (limits.maxPerStageDescriptorStorageBuffers < storageBuffers ||
        limits.maxDescriptorSetStorageBuffers < storageBuffers ||
        limits.maxPerStageDescriptorStorageImages < storageImages ||
        limits.maxDescriptorSetStorageImages < storageImages ||
        limits.maxPerStageDescriptorUniformBuffers < uniformBuffers ||
        limits.maxDescriptorSetUniformBuffers < uniformBuffers ||
        limits.maxUniformBufferRange < sizeof(RayFusionFrameUniform)) {
        return 0;
    }

    const uint32_t samplerLimit = (std::min)({
        limits.maxPerStageDescriptorSamplers, limits.maxDescriptorSetSamplers,
        limits.maxPerStageDescriptorSampledImages, limits.maxDescriptorSetSampledImages});
    const uint32_t fixedResources =
        fixedSamplers + storageBuffers + storageImages + uniformBuffers + colorAttachments;
    if (samplerLimit <= fixedSamplers || limits.maxPerStageResources <= fixedResources) {
        return 0;
    }

    return (std::min)({requestedCapacity, samplerLimit - fixedSamplers,
                      limits.maxPerStageResources - fixedResources});
}

bool ensureRayFusionViewportPasses(
    VkDevice vkDevice,
    VkRenderPass& renderPass,
    VkRenderPass& renderPassLoad,
    VkRenderPass& hdrRenderPass,
    VkRenderPass& hdrRenderPassLoad,
    bool storeReflectionGbuffer) {
    if (hdrRenderPass == VK_NULL_HANDLE && !storeReflectionGbuffer) {
        SCENE_LOG_INFO("[RayFusion] HDR render pass omitting unused reflection G-buffer "
                       "attachments (normal/specular): ray-query unsupported.");
    }
    auto createViewportRenderPass = [&](VkFormat colorFormat,
                                        VkAttachmentLoadOp colorLoad,
                                        VkImageLayout colorInitial,
                                        VkImageLayout colorFinal,
                                        VkAttachmentLoadOp depthLoad,
                                        VkImageLayout depthInitial,
                                        bool withGbuffer,
                                        VkRenderPass* out) -> bool {
        if (*out != VK_NULL_HANDLE) return true;

        VkAttachmentDescription attachments[5]{};
        attachments[0].format = colorFormat;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = colorLoad;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = colorInitial;
        attachments[0].finalLayout = colorFinal;

        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = depthLoad;
        // STORE (not DONT_CARE): the selection-outline mask pass re-loads this
        // depth right after the main pass to depth-test the visible channel,
        // and the post compute samples it for the circle of confusion.
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = depthInitial;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // ★★ G-buffer'in load op'u SAHNE RENGINI TAKIP EDER, kendi basina
        //   CLEAR olamaz. Sebep RT shadow'un gecisi ikiye bolmesi: kare
        //   CLEAR varyantiyla baslar (G-buffer burada temizlenir), compute
        //   icin kapanir, LOAD varyantiyla devam eder. LOAD varyantini
        //   CLEAR yapmak `resumeRtShadowShading`'in `clearValueCount = 0`
        //   cagrisini gecersiz kilardi; ayrica temizlik zaten kare basinda
        //   bir kez olmali, gecisin her yeniden acilisinda degil.
        //   O8 (TBDR): on GPUs without ray-query, reflectionNormal (2) and
        //   reflectionSpecular (3) are never read; mark them UNUSED/DONT_CARE
        //   so tiled GPUs (Mali-G52) skip 96 bit/px of tile RAM and store BW.
        for (uint32_t g = 2; g < 5; ++g) {
            const bool omit = !storeReflectionGbuffer && (g == 2 || g == 3);
            attachments[g].format = g == 2 ? kRfGbufferNormalFormat
                                           : (g == 3 ? kRfGbufferSpecularFormat : VK_FORMAT_R32G32B32A32_SFLOAT);
            attachments[g].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[g].loadOp = omit ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : colorLoad;
            attachments[g].storeOp = omit ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
            attachments[g].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[g].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[g].initialLayout = (!omit && colorLoad == VK_ATTACHMENT_LOAD_OP_LOAD)
                ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[g].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        VkAttachmentReference colorRefs[4]{};
        colorRefs[0].attachment = 0;
        colorRefs[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorRefs[1].attachment = storeReflectionGbuffer ? 2u : VK_ATTACHMENT_UNUSED;
        colorRefs[1].layout = storeReflectionGbuffer
            ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        colorRefs[2].attachment = storeReflectionGbuffer ? 3u : VK_ATTACHMENT_UNUSED;
        colorRefs[2].layout = storeReflectionGbuffer
            ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;

        colorRefs[3] = {4, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{};
        depthRef.attachment = 1;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = withGbuffer ? 4u : 1u;
        subpass.pColorAttachments = colorRefs;
        subpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        // ★ COMPUTE_SHADER srcStage'e eklendi: LDR-LOAD varyantindan once
        //   post compute AYNI goruntuye yaziyor. Bagimlilik kurulmazsa
        //   izgara/gizmo, post'un yazmasi BITMEDEN cizilebilir ve belirti
        //   "bazen izgara kayboluyor" olur -- kare kare degisen, kimsenin
        //   tekrar uretemedigi bir hata.
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = withGbuffer ? 5u : 2u;
        renderPassInfo.pAttachments = attachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
        return vulkanStartupResult(withGbuffer ? "vkCreateRenderPass (HDR)" : "vkCreateRenderPass (LDR)",
                                   vkCreateRenderPass(vkDevice, &renderPassInfo, nullptr, out)) == VK_SUCCESS;
    };

    return
        // LDR, her seyi temizleyen varyant (Solid/Matcap modu ve ilk kare)
        createViewportRenderPass(VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_ATTACHMENT_LOAD_OP_CLEAR, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_ATTACHMENT_LOAD_OP_CLEAR, VK_IMAGE_LAYOUT_UNDEFINED,
                                 false,
                                 &renderPass) &&
        // LDR, post'un yazdigi kareyi KORUYAN varyant (Material modu)
        createViewportRenderPass(VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_ATTACHMENT_LOAD_OP_LOAD, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_ATTACHMENT_LOAD_OP_LOAD,
                                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 false,
                                 &renderPassLoad) &&
        // HDR sahne gecisi: cikisi compute tarafindan ORNEKLENIR
        // ★ finalLayout GENERAL, SHADER_READ_ONLY degil: transmission
        //   replay gecisi bu goruntuye GENERAL'den TRANSFER_SRC'ye gecip
        //   geri donuyor ve uzerine tekrar CIZIYOR. Tek bir "her ise
        //   yarayan" layout, iki tuketici arasindaki gecisleri ortadan
        //   kaldirir; ornekleme GENERAL'den de gecerlidir.
        createViewportRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_ATTACHMENT_LOAD_OP_CLEAR, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_ATTACHMENT_LOAD_OP_CLEAR, VK_IMAGE_LAYOUT_UNDEFINED,
                                 true,
                                 &hdrRenderPass) &&
        createViewportRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_ATTACHMENT_LOAD_OP_LOAD, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_ATTACHMENT_LOAD_OP_LOAD, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 true,
                                 &hdrRenderPassLoad);

}

} // namespace Backend
