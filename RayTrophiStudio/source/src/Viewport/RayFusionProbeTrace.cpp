// RayFusion Adım 1b-α — probe üreticisi: ray query ile görünürlük.
//
// ★★★ Bu dosya DİKİŞİN ÜRETİCİ tarafıdır ve tüketiciye hiç dokunmaz.
//   probe_field.glsl ile material_preview_frag.frag bu partide bir satır bile
//   değişmedi. Bunun sebebi disiplin değil ÖLÇÜM: tüketici sabitken görüntüde
//   çıkan fark yalnızca üreticiden gelebilir. İki taraf aynı anda değişseydi
//   "GI geldi mi" sorusunun cevabı ölçülemez olurdu.
//
// Üretilen değerin BİRİMİ tüketiciye aittir: rfSampleProbeField normal yönünde
// TEK texel okur ve doğrudan albedo ile çarpar, yani texel "normali bu olan
// yüzeyin gördüğü kosinüs ağırlıklı ortalama ışıma"dır (irradiance/PI).
// Kosinüs integralini shader yapıyor; ön-integre edilmiş irradiance haritasını
// kaynak almak iki kez integre etmek olurdu.

#include "Backend/VulkanBackend.h"
#include "RayFusion/ProbeField.h"
#include "globals.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace Backend {
namespace {

// Shader'daki RF_RAYS ve local_size_x ile AYNI olmak zorunda. Işın sayısı hem
// dağılımı hem paylaşılan belleği tanımlıyor; ikisi ayrışırsa shader okunmamış
// bellekten toplar ve sonuç "biraz karanlık" olur — kimsenin bug diye
// raporlamayacağı bir belirti.
constexpr uint32_t kProbeTraceRays = 64u;
constexpr uint32_t kProbeTraceLocalSize = 64u;

// Bir probe'un ışını nereye kadar gider. Sonsuz değil: gökyüzünü gören ışın
// zaten ıskalayacak, ve sonlu bir tmax mesafe momentlerine SONLU bir üst sınır
// verir. 1e4 gibi bir "sonsuz" moment, Chebyshev testini her yerde "engel yok"
// yapardı — yani ölçüm gibi görünen bir varsayılan.
constexpr float kProbeRayTMax = RayFusion::kProbeTraceDistance;
// Probe tam bir yüzeyin üstünde doğduğunda kendi zeminine çarpmasın diye.
constexpr float kProbeRayTMin = 0.05f;
// Mesafe lobu kosinüs kuvveti. Işıma yarım küreyi ortalamalı, mesafe ise
// normale yakın yönü temsil etmeli; aynı lobu kullanmak momentleri her yönde
// eşitler ve görünürlük testini anlamsız kılar.
constexpr float kProbeDistancePower = 8.0f;

struct alignas(16) ProbeTracePush {
    float params[4];  // x = probe sayısı, y = tmin, z = tmax, w = ışın sayısı
    float params2[4]; // x = mesafe lobu kuvveti
};
static_assert(sizeof(ProbeTracePush) == 32u, "rayfusion_probe_trace.comp push ABI");

std::vector<uint32_t> loadProbeTraceSpv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || (bytes % 4) != 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4u);
    if (!file.read(reinterpret_cast<char*>(words.data()), bytes)) return {};
    return words;
}

} // namespace

class RayFusionProbeTraceResources {
public:
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VulkanRT::BufferHandle origins;  // vec4 per probe (CPU -> GPU)
    VulkanRT::BufferHandle results;  // ProbeTexel array (GPU -> CPU)
    VulkanRT::BufferHandle counters; // BounceCounters (GPU -> CPU)
    uint32_t capacity = 0;           // kac probe icin yer var
    // Bindless malzeme dokusu dizisinin uzunlugu. Cihaz limitinden turetilir,
    // sabit degil: bazi ICD'ler descriptor indexing bildirip diziyi cok daha
    // dusuk bir sayida keser ve fazlasi layout'tan gecip pipeline olustururken
    // surucu icinde coker (material preview ayni hesabi yapiyor).
    uint32_t textureArrayLen = 1u;
    // Doku descriptor'larinin hangi kusakta yazildigi. Kusak degismedikce
    // yeniden yazmak, her senkron dispatch'e bir descriptor turu eklerdi.
    uint64_t textureGeneration = 0;
    size_t textureCount = 0;
    RayFusion::BounceCounters lastCounters{};
    bool ready = false;
    // Ölçüm yüzeyi.
    uint64_t dispatches = 0;
    uint64_t probesTraced = 0;
    double lastTraceMs = 0.0;
    std::string inactiveReason;
};

bool VulkanBackendAdapter::ensureRayFusionProbeTraceResources(uint32_t capacity) {
    // Kapasiteyi tabanla: bir parti 16, sonraki 24 probe isteyince boru hattini
    // yeniden kurmak, olculen maliyeti kalite ayarina bagli bir gurultuye
    // cevirirdi. 64 probe'luk sonuc tamponu 128 KB.
    capacity = capacity < 64u ? 64u : capacity;
    if (!m_device || !m_device->isInitialized()) return false;
    if (!m_device->hasHardwareRT() ||
        !m_device->getCapabilities().supportsRayQuery) {
        if (m_rayFusionProbeTrace)
            m_rayFusionProbeTrace->inactiveReason = m_device->hasHardwareRT()
                ? "device reports no ray query support"
                : "device reports no hardware ray tracing";
        return false;
    }
    auto state = m_rayFusionProbeTrace;
    if (!state) {
        state = std::make_shared<RayFusionProbeTraceResources>();
        m_rayFusionProbeTrace = state;
    }
    if (state->ready && state->capacity >= capacity) return true;
    if (state->ready) {
        // Kapasite büyüdü: yalnızca tamponları büyüt, boru hattını değil.
        destroyRayFusionProbeTraceResources();
        state = std::make_shared<RayFusionProbeTraceResources>();
        m_rayFusionProbeTrace = state;
    }

    const auto words = loadProbeTraceSpv(m_rayFusionShaderDir + "/rayfusion_probe_trace_beta.spv");
    if (words.empty()) {
        // Eksik shader SESSİZ kalmamalı: probe alanı gökyüzü üreticisinde
        // kalır ve görüntü doğru görünür, yani hiçbir belirti olmaz.
        state->inactiveReason = "rayfusion_probe_trace_beta.spv is missing; rebuild shaders";
        static bool warned = false;
        if (!warned) {
            SCENE_LOG_WARN("[RayFusion] rayfusion_probe_trace_beta.spv is missing; the probe "
                           "field keeps the sky-bake producer until shaders are rebuilt.");
            warned = true;
        }
        return false;
    }

    VkDevice device = m_device->getDevice();

    // * Dizi uzunlugu CIHAZ LIMITINDEN turetilir, VULKAN_TEXTURE_CAPACITY'den
    //   degil. Bazi ICD'ler descriptor indexing bildirir ama UPDATE_AFTER_BIND
    //   olmayan sampled-image dizisini cok daha dusuk bir sayida keser; fazlasi
    //   layout olusturmadan GECER ve pipeline olustururken surucu icinde coker.
    //   Material preview ayni hesabi yapiyor; ikisi ayrisirsa bu shader onun
    //   yazamadigi bir slotu okur.
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(m_device->getPhysicalDevice(), &props);
        const uint32_t limit = props.limits.maxPerStageDescriptorSampledImages;
        const uint32_t want = static_cast<uint32_t>(VULKAN_TEXTURE_CAPACITY);
        // 1 slot ortam isimasi (binding 2) icin ayrili, 3 slot pay birakildi.
        state->textureArrayLen = (limit > 4u) ? (std::min)(want, limit - 4u) : 1u;
    }

    VkDescriptorSetLayoutBinding bindings[9]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    for (uint32_t i = 4; i < 7; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    // binding 7: RT/raster hattinin okudugu AYNI bindless doku dizisi.
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // binding 8: BounceCounters -- bu dilimin KABUL ALETI.
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    for (uint32_t i = 0; i < 9; ++i) {
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[7].descriptorCount = state->textureArrayLen;

    // Seyrek dizi: yalniz gercekten yuklenmis doku ID'lerinin slotu yazilir.
    // PARTIALLY_BOUND olmadan yazilmamis slot spec'e gore TANIMSIZDIR ve bazi
    // suruculer onu kosulsuz dereference eder. Yine de asagida her slot bir yer
    // tutucu ile dolduruluyor: tanimsiz bir descriptor'i yalnizca bir bayraga
    // guvenerek birakmak, bu depoda bir kez cihaz kaybina donustu.
    const bool hasDescIdx = m_device->getCapabilities().supportsDescriptorIndexing;
    if (!hasDescIdx) {
        // Tek slot ile bir malzeme dokusu dizisi taklit etmek, HER dokuyu
        // 0 numarali slota okumak olurdu -- sessizce yanlis bir goruntu.
        // Dilim burada durur ve gerekcesini soyler.
        state->inactiveReason =
            "device reports no descriptor indexing; the bounce cannot read material textures";
        return false;
    }
    VkDescriptorBindingFlags bindingFlags[9]{};
    bindingFlags[7] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{};
    bindingFlagsCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsCI.bindingCount = 9;
    bindingFlagsCI.pBindingFlags = bindingFlags;

    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 9;
    dlci.pBindings = bindings;
    dlci.pNext = &bindingFlagsCI;
    if (vkCreateDescriptorSetLayout(device, &dlci, nullptr, &state->layout) != VK_SUCCESS) {
        state->inactiveReason = "descriptor set layout creation failed";
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(ProbeTracePush);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &state->layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &state->pipelineLayout) != VK_SUCCESS) {
        state->inactiveReason = "pipeline layout creation failed";
        return false;
    }

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = words.size() * sizeof(uint32_t);
    smci.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) {
        state->inactiveReason = "shader module creation failed";
        return false;
    }
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.layout = state->pipelineLayout;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    const VkResult pipelineResult = vkCreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &cpci, nullptr, &state->pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    if (pipelineResult != VK_SUCCESS) {
        // Ray query desteklenmiyorsa boru hattı BURADA reddedilir. Cihaz
        // yeteneği bildirmiş ama sürücü derleyemiyorsa gerekçe budur.
        state->inactiveReason = "compute pipeline creation failed (ray query unsupported?)";
        return false;
    }

    VkDescriptorPoolSize poolSizes[3] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1u},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6u},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, state->textureArrayLen + 1u}
    };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = poolSizes;
    dpci.maxSets = 1u;
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &state->pool) != VK_SUCCESS) {
        state->inactiveReason = "descriptor pool creation failed";
        return false;
    }
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = state->pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &state->layout;
    if (vkAllocateDescriptorSets(device, &dsai, &state->set) != VK_SUCCESS) {
        state->inactiveReason = "descriptor set allocation failed";
        return false;
    }

    VulkanRT::BufferCreateInfo originInfo;
    originInfo.size = static_cast<uint64_t>(capacity) * 4u * sizeof(float);
    originInfo.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
    originInfo.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
    state->origins = m_device->createBuffer(originInfo);

    VulkanRT::BufferCreateInfo resultInfo;
    resultInfo.size = static_cast<uint64_t>(capacity) * RayFusion::kProbeTexels *
                      sizeof(RayFusion::ProbeTexel);
    resultInfo.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_SRC;
    resultInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
    state->results = m_device->createBuffer(resultInfo);

    VulkanRT::BufferCreateInfo counterInfo;
    counterInfo.size = sizeof(RayFusion::BounceCounters);
    counterInfo.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_SRC |
                        VulkanRT::BufferUsage::TRANSFER_DST;
    counterInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
    state->counters = m_device->createBuffer(counterInfo);

    if (!state->origins.buffer || !state->results.buffer || !state->counters.buffer) {
        state->inactiveReason = "probe trace buffer allocation failed";
        destroyRayFusionProbeTraceResources();
        return false;
    }

    state->capacity = capacity;
    state->inactiveReason.clear();
    state->ready = true;
    return true;
}

bool VulkanBackendAdapter::traceRayFusionProbes(const float* origins, uint32_t probeCount,
                                               std::vector<RayFusion::ProbeTexel>& out) {
    out.clear();
    if (!origins || probeCount == 0u) return false;
    if (!ensureRayFusionProbeTraceResources(probeCount)) return false;
    auto state = m_rayFusionProbeTrace;
    if (!state || !state->ready) return false;

    RayFusionSceneASStatus sceneStatus{};
    if (!getRayFusionSceneASStatus(sceneStatus) || !sceneStatus.ready) {
        state->inactiveReason = "scene acceleration structure is not ready";
        return false;
    }

    // ★ Sahne yapısı ÖNCE hazır olmalı. Kapı burada değil ensure'da: izlenecek
    //   bir sahne yokken ışın atmak, "her yön gökyüzü" diyen ve doğru görünen
    //   bir sonuç üretirdi — yani arızayı makul bir sayıya çevirirdi.
    const VkAccelerationStructureKHR tlas = m_device->getTLASHandle();
    if (tlas == VK_NULL_HANDLE) {
        state->inactiveReason = "scene acceleration structure is not built yet";
        return false;
    }
    VkImageView envView = VK_NULL_HANDLE;
    VkSampler envSampler = VK_NULL_HANDLE;
    if (!getMaterialPreviewEnvRadiance(envView, envSampler)) {
        state->inactiveReason = "environment radiance map is not generated yet";
        return false;
    }

    VkDevice device = m_device->getDevice();
    // A destroyed TLAS/image can be recreated with the SAME numeric handle.
    // Refresh every synchronous trace batch: handle equality cannot prove that
    // the descriptor still references the current resource allocation.
    {
        // Uçuştaki bir kare bu seti kullanıyor olabilir; yeniden yazmadan önce
        // boşalt. Bu depoda AS ile ilgili senkronizasyon eksiği bir kez
        // doğrudan cihaz kaybına dönüştü.
        drainInteractiveViewportInFlight();

        VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
        asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        asInfo.accelerationStructureCount = 1;
        asInfo.pAccelerationStructures = &tlas;

        VkDescriptorBufferInfo resultInfo{};
        resultInfo.buffer = state->results.buffer;
        resultInfo.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo originInfo{};
        originInfo.buffer = state->origins.buffer;
        originInfo.range = VK_WHOLE_SIZE;
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = envView;
        imageInfo.sampler = envSampler;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // ★★ Dizi 4 genis: ortak servis artik emissive ucgen tablosunu da
        //   dondurur. Bu gecis onu BAGLAMAZ ve BAGLAMAMALI -- shader'inda
        //   RF_EMISSIVE_COUNT tanimli degil, yani ne binding'i var ne de
        //   emission bastirmasi. Ikisi ayni derleme zamani kapisinda.
        VulkanRT::BufferHandle bounceBuffers[4]{};
        if (!rayFusionBounceBuffers(bounceBuffers)) {
            state->inactiveReason = "bounce descriptors are not ready";
            return false;
        }
        VkDescriptorBufferInfo counterInfo{};
        counterInfo.buffer = state->counters.buffer;
        counterInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo bounceInfo[3]{};
        VkWriteDescriptorSet writes[8]{};
        for (uint32_t i = 0; i < 8; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = state->set;
            writes[i].dstBinding = i < 7u ? i : 8u;
            writes[i].descriptorCount = 1;
        }
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[7].pBufferInfo = &counterInfo;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[0].pNext = &asInfo;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &resultInfo;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo = &imageInfo;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &originInfo;
        for (uint32_t i = 0; i < 3; ++i) {
            bounceInfo[i].buffer = bounceBuffers[i].buffer;
            bounceInfo[i].range = VK_WHOLE_SIZE;
            writes[4+i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[4+i].pBufferInfo = &bounceInfo[i];
        }
        vkUpdateDescriptorSets(device, 8, writes, 0, nullptr);

        // ** Bindless doku dizisi: RT/raster ile AYNI slot numaralari.
        //   Yalniz kusak degistiginde yeniden yazilir -- her senkron dispatch'te
        //   binlerce descriptor yazmak, olculen trace_ms'i descriptor maliyeti
        //   ile kirletirdi ve o kirlilik "isin izleme pahaliymis" diye okunurdu.
        const uint64_t texGeneration = textureCacheGeneration();
        if (texGeneration != state->textureGeneration ||
            m_uploadedImages.size() != state->textureCount) {
            VkDescriptorImageInfo placeholder{};
            for (const auto& entry : m_uploadedImages) {
                if (entry.second.view && entry.second.sampler) {
                    placeholder.sampler = entry.second.sampler;
                    placeholder.imageView = entry.second.view;
                    placeholder.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    break;
                }
            }
            if (placeholder.imageView) {
                // Once HER slot gecerli bir goruntu ile doldurulur. Bir slotu
                // "nasilsa okunmaz" diye bos birakmak, malzeme tablosu ile
                // yuklenmis doku kumesinin bir kare bile ayrismasi durumunda
                // tanimsiz bir descriptor okumak demektir.
                std::vector<VkDescriptorImageInfo> infos(state->textureArrayLen, placeholder);
                for (const auto& entry : m_uploadedImages) {
                    if (entry.first <= 0 ||
                        static_cast<uint32_t>(entry.first) >= state->textureArrayLen) continue;
                    if (!entry.second.view || !entry.second.sampler) continue;
                    auto& slot = infos[static_cast<size_t>(entry.first)];
                    slot.sampler = entry.second.sampler;
                    slot.imageView = entry.second.view;
                    slot.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                }
                VkWriteDescriptorSet texWrite{};
                texWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                texWrite.dstSet = state->set;
                texWrite.dstBinding = 7;
                texWrite.dstArrayElement = 0;
                texWrite.descriptorCount = state->textureArrayLen;
                texWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                texWrite.pImageInfo = infos.data();
                vkUpdateDescriptorSets(device, 1, &texWrite, 0, nullptr);
                state->textureGeneration = texGeneration;
                state->textureCount = m_uploadedImages.size();
            }
        }
    }

    m_device->uploadBuffer(state->origins, origins,
                           static_cast<size_t>(probeCount) * 4u * sizeof(float));

    // ★★ BİLİNEN MALİYET, ölçülmek üzere bırakıldı: bu gönderim SENKRON —
    //   endSingleTimeCommands bekler, sonra readback yapılır. Alan dolduktan
    //   sonra schedule() boş döndüğü için bu bir kerelik bir bedeldir; ama bir
    //   nesne gizmo ile SÜRÜKLENİRKEN instance imzası her kare değişir, alan
    //   her kare geçersizleşir ve bu durak her kare ödenir. Asenkron okuma
    //   (bu kare gönder, gelecek kare oku) bu maliyeti kaldırır — ama önce
    //   ÖLÇÜLMELİ: `trace_ms` bunun için raporlanıyor. Tahmine dayalı bir
    //   optimizasyon, ölçmediğim bir sorunu çözmüş gibi görünürdü.
    const auto started = std::chrono::steady_clock::now();
    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        state->inactiveReason = "command buffer allocation failed";
        return false;
    }
    VkMemoryBarrier asBarrier{};
    asBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    asBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    asBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &asBarrier, 0, nullptr, 0, nullptr);
    // Sayaclar DISPATCH BASINA sifirlanir. Birikmis bir sayac "gecen sefer
    // calismisti" der ve su anki partiyi olcmez -- olcu aletinin en kotu
    // arizasi, gecmisi simdiki zaman gibi gostermesidir.
    vkCmdFillBuffer(cmd, state->counters.buffer, 0, sizeof(RayFusion::BounceCounters), 0u);
    VkMemoryBarrier counterBarrier{};
    counterBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    counterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    counterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &counterBarrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state->pipelineLayout,
                            0, 1, &state->set, 0, nullptr);
    ProbeTracePush push{};
    push.params[0] = static_cast<float>(probeCount);
    push.params[1] = kProbeRayTMin;
    push.params[2] = kProbeRayTMax;
    push.params[3] = static_cast<float>(kProbeTraceRays);
    push.params2[0] = kProbeDistancePower;
    const auto bounce = rayFusionBounceStatus();
    push.params2[1] = bounce.requested && bounce.ready ? 1.0f : 0.0f;
    const uint32_t packedLightsHair = (bounce.lights & 0xFFFFu) | ((bounce.hairMaterials & 0xFFFFu) << 16u);
    std::memcpy(&push.params2[2], &packedLightsHair, sizeof(float));
    push.params2[3] = static_cast<float>(state->textureArrayLen);
    vkCmdPushConstants(cmd, state->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    // Bir workgroup = bir probe. local_size_x, ışın sayısı ile aynı.
    static_assert(kProbeTraceLocalSize == RayFusion::kProbeTexels,
                  "one lane per texel in the accumulation phase");
    vkCmdDispatch(cmd, probeCount, 1, 1);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    m_device->endSingleTimeCommands(cmd);

    out.resize(static_cast<size_t>(probeCount) * RayFusion::kProbeTexels);
    m_device->downloadBuffer(state->results, out.data(),
                             out.size() * sizeof(RayFusion::ProbeTexel));
    m_device->downloadBuffer(state->counters, &state->lastCounters,
                             sizeof(RayFusion::BounceCounters));
    state->lastTraceMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ++state->dispatches;
    state->probesTraced += probeCount;
    state->inactiveReason.clear();
    return true;
}

bool VulkanBackendAdapter::getRayFusionProbeTraceStatus(bool& supported, double& lastMs,
                                                        uint64_t& dispatches,
                                                        uint64_t& probesTraced,
                                                        std::string& reason) const {
    const auto state = m_rayFusionProbeTrace;
    supported = false;
    lastMs = 0.0;
    dispatches = 0;
    probesTraced = 0;
    reason.clear();
    if (!state) {
        reason = "probe tracing has never been requested on this backend";
        return false;
    }
    supported = state->ready;
    lastMs = state->lastTraceMs;
    dispatches = state->dispatches;
    probesTraced = state->probesTraced;
    reason = state->inactiveReason;
    return true;
}

RayFusion::BounceCounters VulkanBackendAdapter::rayFusionBounceCounters() const {
    const auto state = m_rayFusionProbeTrace;
    return state ? state->lastCounters : RayFusion::BounceCounters{};
}

void VulkanBackendAdapter::destroyRayFusionProbeTraceResources() {
    if (!m_rayFusionProbeTrace) return;
    if (!m_device || !m_device->isInitialized()) {
        m_rayFusionProbeTrace.reset();
        return;
    }
    drainInteractiveViewportInFlight();
    auto& state = *m_rayFusionProbeTrace;
    VkDevice device = m_device->getDevice();
    if (state.origins.buffer) m_device->destroyBuffer(state.origins);
    if (state.results.buffer) m_device->destroyBuffer(state.results);
    if (state.counters.buffer) m_device->destroyBuffer(state.counters);
    if (state.pipeline) vkDestroyPipeline(device, state.pipeline, nullptr);
    if (state.pipelineLayout) vkDestroyPipelineLayout(device, state.pipelineLayout, nullptr);
    if (state.pool) vkDestroyDescriptorPool(device, state.pool, nullptr);
    if (state.layout) vkDestroyDescriptorSetLayout(device, state.layout, nullptr);
    m_rayFusionProbeTrace.reset();
}

} // namespace Backend
