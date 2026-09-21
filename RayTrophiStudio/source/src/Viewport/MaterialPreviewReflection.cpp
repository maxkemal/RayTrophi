#include "Backend/VulkanBackend.h"
#include "globals.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>
#include <Viewport/RasterImageBarrier.h>

// RayFusion — piksel basina spekuler yansima geçişi.
//
// ★★★★★ BU GECIS BIR DUZELTMEDIR, BIR KAYNAK DEGIL. Fragment shader
//   `envSpecular = env(R) * agirlik` yazdi ve agirligi G-buffer'a birakti;
//   burasi `agirlik * (izlenenRadyans - env(R))` ekliyor. Bunun iki sonucu var
//   ve ikisi de tasarim:
//     1. Isin iskalarsa fark SIFIR olur -- izlenen ve izlenmeyen bolge
//        arasindaki dikis YAPISAL OLARAK olusamaz.
//     2. Gecis hic kosmazsa goruntu bugunkunun aynisidir -- donanim yok,
//        kapali, ya da dispatch elendi: hicbiri siyah spekuler uretmez.
//
// ★★ Ve yansimanin metalik ile SINIRLI OLMAMASI da buradan gelir: agirlik bir
//   Fresnel terimi oldugu icin verniklenmis ahsap, boyali zemin, seramik ve
//   plastik ayni carpandan gecer. Kapi roughness + agirlik, `metallic > x`
//   DEGIL -- o kapi tam olarak bu yuzeyleri elerdi.

namespace Backend {
namespace {

struct alignas(16) ReflectionPush {
    float invViewProj[16]{};
    float cameraPos[4]{};
    float params[4]{};   // tmin tabani, agirlik kapisi, max mesafe, roughness kapisi
    float params2[4]{};  // gokyuzu donmesi, env olcegi, isik sayisi, bindless kapasite
    uint32_t shape[4]{}; // width, height, ornek sayisi, env mip sayisi
};
static_assert(sizeof(ReflectionPush) == 128u, "reflection push ABI");

// ★ `reflection_trace.comp` icindeki RfReflectionCounters ile BIREBIR: 8 uint.
// ★★ Bu tampon binding 1'de KENDI sayaclarimiz, binding 8'de ise
//   RfBounceCounters'in yer tutucusu olarak baglaniyor (bu gecis onlari
//   saymiyor ama include BEYAN ediyor). Yani boyut o blogun ABI'sinden
//   KUCUK OLAMAZ; sabit birakmak, blok buyudugunde sessizce tasan bir
//   descriptor demekti.
constexpr VkDeviceSize kReflectionCounterBytes = sizeof(RayFusion::BounceCounters);
static_assert(kReflectionCounterBytes >= 32, "reflection counters need 8 slots of their own");

// ★★★ Fragment shader'in literal'i: `reflectionRoughness * 8.0`. Iki yerde
//   yasiyor ve AYNI olmak zorunda -- cikarilan env terimi ile eklenen env
//   terimi ayni mip'ten okunmazsa fark sifir yerine bir OFSET olur, ve belirtisi
//   "parlak yuzeyler biraz fazla/az parliyor" olur. Yani hicbir yerde yazmayan
//   bir kalibrasyon hatasi.
constexpr uint32_t kReflectionEnvMipScale = 8u;

void reflectionBufferBarrier(VkCommandBuffer cmd, VkBuffer buffer,
                             VkAccessFlags src, VkAccessFlags dst,
                             VkPipelineStageFlags from, VkPipelineStageFlags to) {
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buffer;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, from, to, 0, 0, nullptr, 1, &b, 0, nullptr);
}

bool reflectionPipeline(VkDevice device, VkPipelineLayout layout, const std::string& path,
                        VkPipeline& pipeline, std::string& error) {
    if (pipeline) return true;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { error = "missing shader: " + path; return false; }
    const auto size = file.tellg();
    if (size <= 0 || size_t(size) % 4) { error = "invalid SPIR-V: " + path; return false; }
    std::vector<uint32_t> words(size_t(size) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), size);
    if (!file || words[0] != 0x07230203u) { error = "invalid SPIR-V: " + path; return false; }
    VkShaderModuleCreateInfo mi{};
    mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mi.codeSize = words.size() * 4;
    mi.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &mi, nullptr, &module) != VK_SUCCESS) {
        error = "reflection shader module failed";
        return false;
    }
    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.layout = layout;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    const auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    if (result != VK_SUCCESS) {
        pipeline = VK_NULL_HANDLE;
        error = "reflection pipeline creation failed";
        return false;
    }
    return true;
}

// ★★★★★ ETKIN ayarlar. Preset'i takip ediyorsa depolanan sayilar YOK SAYILIR.
//   Bu fonksiyon TEK yerde yasiyor ve hem `reflectionStatus()` hem
//   `recordReflectionPass()` onu cagirir -- ikisi ayri hesaplasa, panel
//   uygulanandan BASKA bir sey gosterirdi ve bu deponun en pahali hata sinifi
//   tam olarak budur (olcu aleti yalan soyluyor).
RayFusion::ReflectionSettings resolveReflection(
    const RayFusion::ReflectionSettings& stored,
    RasterViewportQualityPreset preset) {
    RayFusion::ReflectionSettings out = stored;
    if (!stored.followQualityPreset) return out;
    out.samples = rasterReflectionSamples(preset);
    out.roughnessGate = rasterReflectionRoughnessGate(preset);
    out.weightGate = rasterReflectionWeightGate(preset);
    return out;
}

const char* presetName(RasterViewportQualityPreset preset) {
    switch (preset) {
        case RasterViewportQualityPreset::Performance: return "performance";
        case RasterViewportQualityPreset::Balanced:    return "balanced";
        case RasterViewportQualityPreset::Quality:     return "quality";
        case RasterViewportQualityPreset::Full:        return "full";
        case RasterViewportQualityPreset::Auto:
        default: return "auto";
    }
}

}  // namespace

class ReflectionResources {
public:
    VulkanRT::BufferHandle counters{};
    void* countersMapped = nullptr;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline trace = VK_NULL_HANDLE, filter = VK_NULL_HANDLE;
    // Piksel basina 8 bayt paketli delta. Trace yazar, filtre okur ve kompozit
    // eder -- HDR'a yazan artik YALNIZCA filtre.
    VulkanRT::BufferHandle deltas{};
    uint32_t width = 0, height = 0, textureCount = 1, lightCount = 0, emissiveCount = 0;
    bool prepared = false, recorded = false;
    // ★★★★ Descriptor'i HER KAREDE yazmak, uzerine yazmadan once kare
    //   halkasini TAMAMEN bosaltmak demektir (in-flight bir set'i guncellemek
    //   gecersiz). Bu depo o bedeli bir kez odedi. Bu yuzden yazilan handle'lar
    //   hatirlanir ve degismediyse guncelleme HIC yapilmaz -- drain de yok.
    //   Imza kaynaklarin KIMLIGINI tasir, icerigini degil: icerik degisimi
    //   ayni tamponun icinde olur ve zaten gorunur.
    uint64_t descriptorSignature = 0;
    // ★★ Son OLCULEN parti. Sayaclar bir kare geriden gelir: kare komut
    //   tamponunda sifirlanir, GPU doldurur, CPU sonraki karede okur.
    uint64_t gatedPixels = 0, rays = 0, shadedHits = 0, skyMisses = 0;
    std::string reason = "not prepared";
};

RayFusion::ReflectionStatus VulkanBackendAdapter::reflectionStatus() const {
    RayFusion::ReflectionStatus out;
    const auto preset = ::render_settings.raster_viewport_quality_preset;
    // ETKIN degerler raporlanir, depolanan degil.
    out.settings = resolveReflection(m_reflectionSettings, preset);
    out.followedQualityPreset = m_reflectionSettings.followQualityPreset;
    out.qualityPreset = presetName(preset);
    out.supported = m_device && m_device->isInitialized() &&
                    m_device->getCapabilities().supportsRayQuery &&
                    m_device->getCapabilities().supportsDescriptorIndexing;
    if (!out.supported) {
        out.reason = "hardware ray query and descriptor indexing required";
    } else if (!out.settings.enabled) {
        out.reason = "disabled";
    } else if (m_viewportMode != ViewportMode::MaterialPreview ||
               ::render_settings.material_preview_lighting_preset !=
                   MaterialPreviewLightingPreset::Scene) {
        // ★ Studio/matcap yollari yazili sahne isiklarini temsil etmiyor ve
        //   `rfEnvSpecularWeight` orada hic doldurulmuyor: kapi kapali.
        out.reason = "requires material viewport and scene lighting";
    } else if (m_reflection) {
        const auto& s = *m_reflection;
        out.ready = s.prepared && s.recorded;
        out.width = s.width;
        out.height = s.height;
        out.gatedPixels = s.gatedPixels;
        out.rays = s.rays;
        out.shadedHits = s.shadedHits;
        out.skyMisses = s.skyMisses;
        out.reason = s.reason;
    }
    return out;
}

bool VulkanBackendAdapter::setReflection(const RayFusion::ReflectionSettings& settings,
                                         std::string& error) {
    if (!RayFusion::validateReflection(settings, error)) return false;
    if (settings.enabled && !reflectionStatus().supported) {
        error = "specular reflections are unsupported on this backend";
        return false;
    }
    m_reflectionSettings = settings;
    if (m_reflection) {
        m_reflection->prepared = false;
        m_reflection->recorded = false;
        m_reflection->reason = "waiting for next raster frame";
    }
    m_interactiveViewport.dirty = true;
    return true;
}

// Her material karesinde, HERHANGI bir erken donusten ONCE cagrilir: bayat bir
// "ready" bayragi, kapatilmis bir gecisi acik gostermek olurdu.
void VulkanBackendAdapter::resetReflectionFrame(uint32_t width, uint32_t height) {
    if (!m_device) return;
    if (!m_reflection) m_reflection = std::make_shared<ReflectionResources>();
    auto& s = *m_reflection;
    s.prepared = false;
    s.recorded = false;
    s.width = width;
    s.height = height;
    s.reason = m_reflectionSettings.enabled ? "raster preparation unavailable" : "disabled";
    m_reflectionRecordedLastFrame = false;

    // ★★★ Sayaclari SIFIRLAMIYORUZ, OKUYORUZ. Onceki karenin GPU yazimi bu
    //   noktada tamamlanmistir (kare halkasi bekledi). Sifirlama, dispatch'ten
    //   hemen once komut tamponunda yapiliyor -- burada sifirlamak, olcumu
    //   raporlanmadan silmek olurdu.
    if (s.countersMapped) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = s.counters.memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        // GPU_TO_CPU HOST_CACHED'dir ve HOST_COHERENT garanti DEGIL: bu cagri
        // olmadan bayat bir cache satiri okunur ve sayac "hic degismedi" gibi
        // gorunur -- yani gecisin hic kosmadigiyla ayni belirti.
        vkInvalidateMappedMemoryRanges(m_device->getDevice(), 1, &range);
        uint32_t raw[8]{};
        std::memcpy(raw, s.countersMapped, sizeof(raw));
        s.gatedPixels = raw[0];
        s.rays = raw[1];
        s.shadedHits = raw[2];
        s.skyMisses = raw[3];
    }
}

void VulkanBackendAdapter::prepareReflectionFrame() {
    if (!m_reflectionSettings.enabled || !m_reflection) return;
    auto& s = *m_reflection;
    if (!s.width || !s.height) { s.reason = "viewport has no extent"; return; }
    if (m_viewportMode != ViewportMode::MaterialPreview ||
        ::render_settings.material_preview_lighting_preset !=
            MaterialPreviewLightingPreset::Scene) {
        s.reason = "requires material viewport and scene lighting";
        return;
    }

    // ★★★★★ AYNI KAPI, IKI TARAFTAN. Fragment shader agirligi yalnizca
    //   `sceneFlags & 32u` (hasPrefilteredIbl) dalinda dolduruyor, ve o bayrak
    //   tam olarak bu cagrinin basarisiyla kuruluyor. Yani prefiltered env
    //   yoksa ne agirlik yazilir ne de gecis kosar -- cikarilan terimin
    //   formulunun gecerli oldugu tek durum budur. Bu iki kapiyi ayirmak,
    //   `envSpecular`i baska bir formulle ekleyip burada BASKA bir terimi geri
    //   cikarmak olurdu ve belirtisi sessiz bir parlaklik kaymasi olurdu.
    VkImageView envView = VK_NULL_HANDLE;
    VkSampler envSampler = VK_NULL_HANDLE;
    if (!getMaterialPreviewEnvRadiance(envView, envSampler)) {
        s.reason = "prefiltered environment radiance is unavailable";
        return;
    }
    if (!m_interactiveViewport.depthImage.view || !m_interactiveViewport.postSampler) {
        s.reason = "depth unavailable";
        return;
    }
    if (!m_interactiveViewport.reflectionNormalImage.view ||
        !m_interactiveViewport.reflectionSpecularImage.view ||
        !m_interactiveViewport.hdrColorImage.view) {
        s.reason = "reflection G-buffer is unavailable";
        return;
    }

    prepareRayFusionBounce();  // ortak flat hit/material/light servisi
    VulkanRT::BufferHandle bounce[4]{};  // hits, materials, lights, emissive triangles
    const auto bounceState = rayFusionBounceStatus();
    if (!bounceState.ready || !rayFusionBounceBuffers(bounce)) {
        s.reason = "flat bounce tables are unavailable";
        return;
    }
    s.lightCount = bounceState.lights;
    // ★★ Emissive ucgen sayisi. 0 ise shader hem NEE'yi hem emission
    //   bastirmasini kapatir -- ikisi AYNI kapiya bagli olmak zorunda, yoksa
    //   o malzemenin emission'i hem listeden (yok) hem isindan (bastirildi)
    //   duserdi.
    s.emissiveCount = bounceState.emissiveTriangles;

    const auto device = m_device->getDevice();
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_device->getPhysicalDevice(), &props);

    if (!s.setLayout) {
        s.textureCount = (std::min)(
            uint32_t(VULKAN_TEXTURE_CAPACITY),
            props.limits.maxPerStageDescriptorSampledImages > 4
                ? props.limits.maxPerStageDescriptorSampledImages - 4 : 1u);
        if (props.limits.maxPushConstantsSize < sizeof(ReflectionPush)) {
            s.reason = "reflection push constants unsupported";
            return;
        }
        // ★ 13 binding: 12 = emissive ucgen tablosu. `rfEmissives` statik olarak
        //   kullanilan bir kaynak, yani BEYAN eden gecis onu BAGLAMAK zorunda.
        VkDescriptorSetLayoutBinding bindings[14]{};
        for (uint32_t i = 0; i < 14; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        for (auto i : {2u, 3u, 7u, 9u, 10u})
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[7].descriptorCount = s.textureCount;
        bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 14;  // 13 = delta buffer (trace writes, filter reads)
        ci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &s.setLayout) != VK_SUCCESS) {
            s.reason = "reflection descriptor layout failed";
            return;
        }
    }

    // Hicbir malzeme doku slotu sessizce placeholder'a ya da skalara donusmesin.
    for (const auto& m : m_cachedGpuMaterials) {
        for (uint32_t slot : {m.albedo_tex, m.emission_tex, m.opacity_tex,
                              m.metallic_tex, m.specular_tex}) {
            if (slot >= s.textureCount) {
                s.reason = "material texture exceeds reflection descriptor capacity";
                return;
            }
        }
    }

    if (!s.pool) {
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s.textureCount + 4},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.maxSets = 1;
        ci.poolSizeCount = 4;
        ci.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(device, &ci, nullptr, &s.pool) != VK_SUCCESS) {
            s.reason = "reflection descriptor pool failed";
            return;
        }
    }
    if (!s.set) {
        VkDescriptorSetAllocateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ci.descriptorPool = s.pool;
        ci.descriptorSetCount = 1;
        ci.pSetLayouts = &s.setLayout;
        if (vkAllocateDescriptorSets(device, &ci, &s.set) != VK_SUCCESS) {
            s.reason = "reflection descriptor allocation failed";
            return;
        }
    }
    if (!s.layout) {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReflectionPush)};
        VkPipelineLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &s.setLayout;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &ci, nullptr, &s.layout) != VK_SUCCESS) {
            s.reason = "reflection pipeline layout failed";
            return;
        }
    }
    if (!reflectionPipeline(device, s.layout,
                            m_rayFusionShaderDir + "/reflection_trace.spv",
                            s.trace, s.reason) ||
        !reflectionPipeline(device, s.layout,
                            m_rayFusionShaderDir + "/reflection_filter.spv",
                            s.filter, s.reason))
        return;

    // ★★ Delta tamponu. Boyut kontrolu maxStorageBufferRange'e karsi yapilir:
    //   4K'da 66 MB, ve bu sinirin altinda kalmak bir varsayim degil OLCUM.
    {
        VkDeviceSize bytes = VkDeviceSize(s.width) * s.height * 8ull;
        if (bytes > props.limits.maxStorageBufferRange) {
            s.reason = "reflection delta buffer exceeds storage-buffer range";
            return;
        }
        if (!s.deltas.buffer || s.deltas.size < bytes) {
            drainInteractiveViewportInFlight();
            if (s.deltas.buffer) m_device->destroyBuffer(s.deltas);
            VulkanRT::BufferCreateInfo ci;
            ci.size = bytes;
            ci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            ci.usage = VulkanRT::BufferUsage::STORAGE;
            s.deltas = m_device->createBuffer(ci);
            if (!s.deltas.buffer) { s.reason = "reflection delta allocation failed"; return; }
        }
    }

    if (!s.counters.buffer) {
        VulkanRT::BufferCreateInfo ci;
        ci.size = kReflectionCounterBytes;
        // GPU_TO_CPU: sayaclar olcu aleti, yani CPU'nun okumasi GEREK. GPU_ONLY
        // birakmak "olculdu ama kimse bakamiyor" demek olurdu.
        ci.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
        ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
        s.counters = m_device->createBuffer(ci);
        if (!s.counters.buffer) { s.reason = "reflection counter allocation failed"; return; }
        s.countersMapped = m_device->mapBuffer(s.counters);
    }

    VkDescriptorImageInfo env{envSampler, envView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo depth{m_interactiveViewport.postSampler,
                                m_interactiveViewport.depthImage.view,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo gbNormal{m_interactiveViewport.postSampler,
                                   m_interactiveViewport.reflectionNormalImage.view,
                                   VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo gbSpecular{m_interactiveViewport.postSampler,
                                     m_interactiveViewport.reflectionSpecularImage.view,
                                     VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo hdr{VK_NULL_HANDLE, m_interactiveViewport.hdrColorImage.view,
                              VK_IMAGE_LAYOUT_GENERAL};
    std::vector<VkDescriptorImageInfo> textures(s.textureCount, env);
    for (const auto& entry : m_uploadedImages) {
        if (entry.first > 0 && uint64_t(entry.first) < s.textureCount &&
            entry.second.view && entry.second.sampler)
            textures[size_t(entry.first)] = {entry.second.sampler, entry.second.view,
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }

    const VkAccelerationStructureKHR tlas = m_device->getTLASHandle();
    if (tlas == VK_NULL_HANDLE) { s.reason = "scene acceleration structure is unavailable"; return; }
    VkWriteDescriptorSetAccelerationStructureKHR as{};
    as.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as.accelerationStructureCount = 1;
    as.pAccelerationStructures = &tlas;

    // Binding 8 = RfBounceCounters. Bu gecis onlari saymiyor
    // (RF_BOUNCE_COUNT bos), ama include onlari BEYAN ediyor: gecerli bir
    // tampon baglanmasi gerek. Kendi sayac tamponumuz yeterince buyuk.
    VkDescriptorBufferInfo bufferInfos[] = {
        {s.counters.buffer, 0, VK_WHOLE_SIZE},   // 1  kendi sayaclarimiz
        {bounce[0].buffer, 0, VK_WHOLE_SIZE},    // 4  hits
        {bounce[1].buffer, 0, VK_WHOLE_SIZE},    // 5  materials
        {bounce[2].buffer, 0, VK_WHOLE_SIZE},    // 6  lights
        {s.counters.buffer, 0, VK_WHOLE_SIZE},   // 8  bounce counters (kullanilmiyor)
        {bounce[3].buffer, 0, VK_WHOLE_SIZE},    // 12 emissive ucgenler
        {s.deltas.buffer, 0, VK_WHOLE_SIZE}};   // 13 yansima deltalari

    VkWriteDescriptorSet writes[14]{};
    for (uint32_t i = 0; i < 14; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = s.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[0].pNext = &as;
    writes[1].pBufferInfo = &bufferInfos[0];
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &env;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &depth;
    writes[4].pBufferInfo = &bufferInfos[1];
    writes[5].pBufferInfo = &bufferInfos[2];
    writes[6].pBufferInfo = &bufferInfos[3];
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].descriptorCount = s.textureCount;
    writes[7].pImageInfo = textures.data();
    writes[8].pBufferInfo = &bufferInfos[4];
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[9].pImageInfo = &gbNormal;
    writes[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[10].pImageInfo = &gbSpecular;
    writes[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[11].pImageInfo = &hdr;
    writes[12].pBufferInfo = &bufferInfos[5];
    writes[13].pBufferInfo = &bufferInfos[6];

    // ★ Imza: baglanan her kaynagin handle'i + kapasiteler. Doku dizisinin
    //   tamami hash'lenir -- tek bir slotun view'i degistiyse (asset yeniden
    //   yuklendi) set bayat kalirdi ve belirtisi "yansimada yanlis doku" olur.
    uint64_t signature = 1469598103934665603ull;
    auto mix = [&signature](uint64_t value) {
        signature ^= value;
        signature *= 1099511628211ull;
    };
    mix(uint64_t(tlas));
    mix(uint64_t(envView));
    mix(uint64_t(envSampler));
    mix(uint64_t(m_interactiveViewport.depthImage.view));
    mix(uint64_t(m_interactiveViewport.reflectionNormalImage.view));
    mix(uint64_t(m_interactiveViewport.reflectionSpecularImage.view));
    mix(uint64_t(m_interactiveViewport.hdrColorImage.view));
    mix(uint64_t(s.counters.buffer));
    mix(uint64_t(s.deltas.buffer));
    for (int i = 0; i < 4; ++i) mix(uint64_t(bounce[i].buffer));
    for (const auto& texture : textures) mix(uint64_t(texture.imageView));
    mix(uint64_t(s.textureCount));
    if (s.descriptorSignature != signature) {
        drainInteractiveViewportInFlight();
        vkUpdateDescriptorSets(device, 14, writes, 0, nullptr);
        s.descriptorSignature = signature;
    }

    if (bounceState.unsupportedMaterials || bounceState.unsupportedLights) {
        // Desteklenmeyen isabetler env yolunda kalir: sinir yazili, sessiz degil.
        s.reason = "unsupported hits keep the environment lookup; bounce supports "
                   "scene point/directional lights only";
    } else {
        s.reason.clear();
    }
    s.prepared = true;
}

void VulkanBackendAdapter::recordReflectionPass(VkCommandBuffer cmd,
                                                const Matrix4x4& viewProj,
                                                const Matrix4x4& view,
                                                uint32_t width, uint32_t height) {
    if (!m_reflection || !m_reflection->prepared || !cmd) return;
    auto& s = *m_reflection;
    if (s.width != width || s.height != height) {
        s.reason = "viewport extent changed mid-frame";
        return;
    }

    // Sayaclar dispatch'ten HEMEN once sifirlanir; `resetReflectionFrame` onceki
    // karenin degerlerini bundan once okumus olur.
    reflectionBufferBarrier(cmd, s.counters.buffer,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const uint32_t zero[8]{};
    vkCmdUpdateBuffer(cmd, s.counters.buffer, 0, sizeof(zero), zero);
    reflectionBufferBarrier(cmd, s.counters.buffer,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // ★★★ Derinlik, ana gecisin DEPTH_STENCIL_ATTACHMENT_OPTIMAL'inden okunur
    //   hale getirilir ve gecisin sonunda GERI verilir. Layout'u geri
    //   vermemek, bu karede sonraki cizimlerin (LDR overlay) derinligi
    //   kullanmasini bozar ve belirtisi "bazen izgara nesnelerin onune
    //   geciyor" olur.
    rasterImageBarrier(cmd, m_interactiveViewport.depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // G-buffer ve HDR hedefi ikisi de GENERAL'de bitiyor (render pass
    // finalLayout'u); gerekli olan tek sey renk yaziminin GORULMESI.
    for (VkImage image : {m_interactiveViewport.reflectionNormalImage.image,
                          m_interactiveViewport.reflectionSpecularImage.image,
                          m_interactiveViewport.hdrColorImage.image}) {
        rasterImageBarrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    ReflectionPush push{};
    const Matrix4x4 inverse = viewProj.inverse();
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) push.invViewProj[c * 4 + r] = float(inverse.m[r][c]);
    // ★ Kamera konumu view matrisinin TERSINDEN gelir. invViewProj'dan bir
    //   nokta cikarmak yakin duzlemi verir, goz noktasini DEGIL -- ve yansima
    //   bakis yonunu oradan turettigi icin hata dogrudan yone gecerdi.
    const Matrix4x4 inverseView = view.inverse();
    push.cameraPos[0] = float(inverseView.m[0][3]);
    push.cameraPos[1] = float(inverseView.m[1][3]);
    push.cameraPos[2] = float(inverseView.m[2][3]);
    // ★★★ .w bir dolgu DEGIL: emissive ucgen sayisi. Shader bunu hem NEE
    //   kapisi hem emission bastirma kapisi olarak okuyor.
    push.cameraPos[3] = float(s.emissiveCount);

    // ★ AYNI cozucuden: preset'i takip ediyorsa etkin butce buradan gelir ve
    //   `reflectionStatus()` ile birebir ayni sayilari kullanir.
    const auto effective = resolveReflection(
        m_reflectionSettings, ::render_settings.raster_viewport_quality_preset);
    push.params[0] = 0.02f;
    push.params[1] = effective.weightGate;
    push.params[2] = effective.maxDistance;
    push.params[3] = effective.roughnessGate;

    // ★★★★ Donme ve env yogunlugu ORTAK PAKETLEYICIDEN okunur, `m_cachedWorld`
    //   alanlarindan YENIDEN TURETILMEZ. Paketleyicinin kendi yorumu bunu
    //   soyluyor: iki kopya, bake edilmis ambient'in cizilen arka plandan
    //   kaymasina izin verir ve tek belirti "renk biraz tuhaf" olur. Burada o
    //   risk daha da keskin -- cikarilan env terimi ile eklenen env terimi ayni
    //   yonden/yogunluktan okunmazsa fark sifir yerine bir OFSET olur.
    float worldColor[4]{}, worldParams[4]{}, worldSun[4]{};
    float atmosphereA[4]{}, atmosphereB[4]{};
    packPreviewWorldUniforms(worldColor, worldParams, worldSun, atmosphereA, atmosphereB);
    push.params2[0] = worldParams[0];
    // Fragment shader ile birebir: `worldMode == 1u ? max(worldParams.y, 0) : 1`.
    push.params2[1] = m_cachedWorld.mode == WORLD_MODE_HDRI
                          ? (std::max)(worldParams[1], 0.0f) : 1.0f;
    push.params2[2] = float(s.lightCount);
    push.params2[3] = float(s.textureCount);

    push.shape[0] = width;
    push.shape[1] = height;
    push.shape[2] = (std::max)(effective.samples, 1u);
    push.shape[3] = kReflectionEnvMipScale;

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout, 0, 1, &s.set, 0, nullptr);
    vkCmdPushConstants(cmd, s.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.trace);
    vkCmdDispatch(cmd, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    markRasterStage(cmd, RasterStage::Reflection, true);

    // ★★★ Trace'in yazdigi delta, filtrenin okumasindan ONCE gorulmeli. Bu
    //   bariyeri atlamak "bazen yansima bir kare geriden geliyor" uretir --
    //   yani kimsenin tekrar uretemedigi, kare kare degisen bir hata.
    reflectionBufferBarrier(cmd, s.deltas.buffer,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // ★★ Filtre yaricapi roughness'la buyur ve AYNADA SIFIR kalir: bir aynayi
    //   bulanistirmak onu ayna olmaktan cikarir. Kompoziti artik bu gecis yapar.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.filter);
    vkCmdDispatch(cmd, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    markRasterStage(cmd, RasterStage::ReflectionFilter, true);

    // HDR yazimi post gecisinin OKUMASINDAN once gorulmeli.
    rasterImageBarrier(cmd, m_interactiveViewport.hdrColorImage.image, VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
    reflectionBufferBarrier(cmd, s.counters.buffer,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    rasterImageBarrier(cmd, m_interactiveViewport.depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

    s.recorded = true;
    m_reflectionRecordedLastFrame = true;
}

void VulkanBackendAdapter::destroyReflection() {
    if (!m_reflection || !m_device) { m_reflection.reset(); return; }
    auto& s = *m_reflection;
    const auto device = m_device->getDevice();
    if (s.trace) vkDestroyPipeline(device, s.trace, nullptr);
    if (s.filter) vkDestroyPipeline(device, s.filter, nullptr);
    if (s.deltas.buffer) m_device->destroyBuffer(s.deltas);
    if (s.layout) vkDestroyPipelineLayout(device, s.layout, nullptr);
    if (s.pool) vkDestroyDescriptorPool(device, s.pool, nullptr);
    if (s.setLayout) vkDestroyDescriptorSetLayout(device, s.setLayout, nullptr);
    if (s.counters.buffer) {
        if (s.countersMapped) m_device->unmapBuffer(s.counters);
        m_device->destroyBuffer(s.counters);
    }
    m_reflection.reset();
}

}  // namespace Backend
