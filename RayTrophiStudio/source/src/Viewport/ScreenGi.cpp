#include "Backend/VulkanBackend.h"
#include "globals.h"
#include <algorithm>
#include <fstream>
#include <vector>

namespace Backend {
namespace {
struct alignas(16) GiPush {
    float invViewProj[16]{};
    float params[4]{};
    float params2[4]{};
    uint32_t shape[4]{};
};
static_assert(sizeof(GiPush)==112, "screen GI push ABI");
constexpr uint32_t kGiHeader=16;
// ★★★ 32 -> 48: GiPixel'e `vec4 aux` (olculen gokyuzu gorunurlugu) eklendi.
// Bu sayi UC shader ile paylasilan bir ABI'dir: screen_gi_trace.comp,
// screen_gi_filter.comp ve material_preview_screen_gi.glsl.
constexpr uint32_t kGiPixel=48;
// ★★★★★ SIHIRLI SAYI DA DEGISTI ("SGI1" -> "SGI2"), ve bu zorunlu: adim
//   degisti ama tampon AYNI boyutta gecerli gorunur. Eski bir .spv yeni bir
//   exe ile (veya tersi) eslesirse, tuketici yanlis adimla okuyup KOMSU
//   PIKSELIN alanlarini bu pikselin isigi sanar -- cokme yok, hata yok,
//   yalnizca yanlis aydinlatma. Sayiyi degistirmek bunu temiz bir "GI yok"a
//   cevirir. Bu depoda bayat .spv zaten bilinen bir tuzak.
constexpr uint32_t kGiMagic=0x53474932u;
void giBarrier(VkCommandBuffer cmd,VkBuffer buffer,VkAccessFlags src,VkAccessFlags dst,
               VkPipelineStageFlags from,VkPipelineStageFlags to) {
    VkBufferMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask=src; b.dstAccessMask=dst;
    b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    b.buffer=buffer; b.size=VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,from,to,0,0,nullptr,1,&b,0,nullptr);
}
bool giPipeline(VkDevice device,VkPipelineLayout layout,const std::string& path,
                VkPipeline& pipeline,std::string& error) {
    if (pipeline) return true;
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file) {error="missing shader: "+path;return false;}
    const auto size=file.tellg();
    if (size<=0 || size_t(size)%4) {error="invalid SPIR-V: "+path;return false;}
    std::vector<uint32_t> words(size_t(size)/4);
    file.seekg(0);file.read(reinterpret_cast<char*>(words.data()),size);
    if (!file || words[0]!=0x07230203u) {error="invalid SPIR-V: "+path;return false;}
    VkShaderModuleCreateInfo mi{};mi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mi.codeSize=words.size()*4;mi.pCode=words.data();VkShaderModule module=VK_NULL_HANDLE;
    if (vkCreateShaderModule(device,&mi,nullptr,&module)!=VK_SUCCESS) {error="screen GI shader module failed";return false;}
    VkComputePipelineCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.layout=layout;ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";
    const auto result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&pipeline);
    vkDestroyShaderModule(device,module,nullptr);
    if (result!=VK_SUCCESS) {pipeline=VK_NULL_HANDLE;error="screen GI pipeline creation failed";return false;}
    return true;
}
}
// Filtre gecisinin `GiStats` blogu ile BIRE BIR ayni sira. Degistirirsen
// `screen_gi_filter.comp` icindeki blogu da degistir.
struct GiStatsBlock {
    uint32_t visSumQ, visCount, fullConf, anyConf, sampled, lumSumQ, visFilled, reserved1;
};
static_assert(sizeof(GiStatsBlock)==32,"GiStats shader block ABI");

class ScreenGiResources {
public:
    VulkanRT::BufferHandle output{},raw{},counters{},stats{};
    // Bir kare gecikmeli okuma burada saklanir; `screenGiStatus()` const oldugu
    // icin ve GPU'yu BEKLETMEDEN okundugu icin.
    mutable RayFusion::ScreenGiMeasurement measurement{};
    VkDescriptorSetLayout setLayout=VK_NULL_HANDLE;
    VkDescriptorPool pool=VK_NULL_HANDLE;
    VkDescriptorSet set=VK_NULL_HANDLE;
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkPipeline trace=VK_NULL_HANDLE,filter=VK_NULL_HANDLE;
    VkDescriptorSet consumerSet=VK_NULL_HANDLE;
    VkBuffer consumerBuffer=VK_NULL_HANDLE;
    uint32_t width=0,height=0,textureCount=1,lightCount=0,emissiveCount=0;
    bool prepared=false,recorded=false,outputReady=false;
    std::string reason="not prepared";
};

RayFusion::ScreenGiStatus VulkanBackendAdapter::screenGiStatus() const {
    RayFusion::ScreenGiStatus out;out.settings=m_screenGiSettings;
    out.supported=m_device && m_device->isInitialized() &&
        m_device->getCapabilities().supportsRayQuery && m_device->getCapabilities().supportsDescriptorIndexing;
    if (!out.supported) out.reason="hardware ray query and descriptor indexing required";
    else if (!out.settings.enabled) out.reason="disabled";
    else if (m_viewportMode!=ViewportMode::MaterialPreview ||
        ::render_settings.material_preview_lighting_preset!=MaterialPreviewLightingPreset::Scene)
        out.reason="requires material viewport and scene lighting";
    else if (!m_rtShadowAllowed) out.reason="prototype requires RT shadows and their depth prepass";
    else if (m_screenGi) {
        const auto& s=*m_screenGi;
        out.ready=s.prepared && s.recorded;
        out.width=s.width;out.height=s.height;
        out.primaryRayBudget=out.ready?uint64_t(s.width)*s.height*out.settings.samples:0;
        out.reason=s.reason;
        // ★★ BEKLEMEDEN okunur: deger en son tamamlanan karenindir. Bir kare
        //   gecikme, olcum icin kabul edilebilir; kareyi durdurmak degildir.
        if (out.ready && s.stats.buffer) {
            GiStatsBlock blk{};
            m_device->downloadBuffer(s.stats,&blk,sizeof(blk));
            if (blk.sampled>0u) {
                auto& m=s.measurement;
                m.measured=true;
                m.stats_pixels=blk.sampled;
                m.mean_sky_visibility=blk.visCount>0u
                    ? float(blk.visSumQ)/(4096.0f*float(blk.visCount)) : 0.0f;
                m.full_confidence_fraction=float(blk.fullConf)/float(blk.sampled);
                m.any_confidence_fraction=float(blk.anyConf)/float(blk.sampled);
                m.mean_gi_luminance=float(blk.lumSumQ)/(1024.0f*float(blk.sampled));
                m.visibility_coverage=float(blk.visCount)/float(blk.sampled);
                // ★★★ AYRI bir oran, `visibility_coverage`a KATILMAZ. Ikisini
                //   toplamak, doldurmanin ne kadar is yaptigini gorunmez
                //   yapardi -- ve bir duzeltmenin kendi olcu aletini bozmasi
                //   bu depoda tekrar eden bir hata sinifi.
                m.visibility_filled_fraction=float(blk.visFilled)/float(blk.sampled);
            }
        }
        out.measurement=s.measurement;
    }
    return out;
}
bool VulkanBackendAdapter::setScreenGi(const RayFusion::ScreenGiSettings& settings,std::string& error) {
    if (!RayFusion::validateScreenGi(settings,error)) return false;
    if (settings.enabled && !screenGiStatus().supported) {error="screen GI is unsupported on this backend";return false;}
    m_screenGiSettings=settings;
    if (m_screenGi) {m_screenGi->prepared=false;m_screenGi->recorded=false;m_screenGi->reason="waiting for next raster frame";}
    m_interactiveViewport.dirty=true;
    return true;
}

// Called on EVERY material frame, before any RT-shadow early return. This
// clears stale GI even when shadows, resources, scene or lighting become invalid.
void VulkanBackendAdapter::resetScreenGiFrame(VkCommandBuffer cmd,uint32_t width,uint32_t height) {
    if (!m_device || !cmd) return;
    if (!m_screenGi) m_screenGi=std::make_shared<ScreenGiResources>();
    auto& s=*m_screenGi;
    s.prepared=false;s.recorded=false;s.outputReady=false;
    s.width=width;s.height=height;
    s.reason=m_screenGiSettings.enabled?"RT shadow/depth preparation unavailable":"disabled";
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_device->getPhysicalDevice(),&props);
    const VkDeviceSize requested=kGiHeader+uint64_t(width)*height*kGiPixel;
    const bool fits=requested<=props.limits.maxStorageBufferRange && width && height;
    const VkDeviceSize bytes=m_screenGiSettings.enabled && fits?requested:kGiHeader;
    if (!s.output.buffer || s.output.size<bytes) {
        drainInteractiveViewportInFlight();
        s.consumerBuffer=VK_NULL_HANDLE;
        if (s.output.buffer) m_device->destroyBuffer(s.output);
        VulkanRT::BufferCreateInfo ci;ci.size=bytes;ci.location=VulkanRT::MemoryLocation::GPU_ONLY;
        ci.usage=VulkanRT::BufferUsage::STORAGE|VulkanRT::BufferUsage::TRANSFER_DST;
        s.output=m_device->createBuffer(ci);
    }
    s.outputReady=s.output.buffer && fits && s.output.size>=requested;
    if (!fits) s.reason="screen GI image exceeds storage-buffer range or is empty";
    if (!s.output.buffer) s.reason="screen GI output allocation failed";
    VkDescriptorBufferInfo info{};
    info.buffer=s.output.buffer?s.output.buffer:m_interactiveViewport.materialPreviewSceneGlobals.buffer;
    info.range=VK_WHOLE_SIZE;
    if (info.buffer && m_interactiveViewport.materialPreviewDescSet &&
        (s.consumerSet!=m_interactiveViewport.materialPreviewDescSet || s.consumerBuffer!=info.buffer)) {
        drainInteractiveViewportInFlight();
        VkWriteDescriptorSet write{};write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet=m_interactiveViewport.materialPreviewDescSet;write.dstBinding=24;
        write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;write.descriptorCount=1;write.pBufferInfo=&info;
        vkUpdateDescriptorSets(m_device->getDevice(),1,&write,0,nullptr);
        s.consumerSet=write.dstSet;s.consumerBuffer=info.buffer;
    }
    if (!s.output.buffer) return;
    giBarrier(cmd,s.output.buffer,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const uint32_t zero[4]{};
    vkCmdUpdateBuffer(cmd,s.output.buffer,0,kGiHeader,zero);
    giBarrier(cmd,s.output.buffer,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void VulkanBackendAdapter::prepareScreenGiFrame() {
    if (!m_screenGiSettings.enabled || !m_screenGi || !m_screenGi->outputReady) return;
    auto& s=*m_screenGi;
    VkImageView envView=VK_NULL_HANDLE;VkSampler envSampler=VK_NULL_HANDLE;
    if (!getMaterialPreviewEnvRadiance(envView,envSampler)) {s.reason="environment radiance is unavailable";return;}
    if (!m_interactiveViewport.depthImage.view || !m_interactiveViewport.postSampler) {s.reason="depth unavailable";return;}
    prepareRayFusionBounce(); // canonical flat hit/material/light service, independent of probe enable
    VulkanRT::BufferHandle bounce[4]{};  // hits, materials, lights, emissive triangles
    const auto status=rayFusionBounceStatus();
    if (!status.ready || !rayFusionBounceBuffers(bounce)) {s.reason="flat bounce tables are unavailable";return;}
    if (status.unsupportedMaterials || status.unsupportedLights) {
        // Per-pixel fallback is used for unsupported hit BSDFs; expose limitations below.
        s.reason="unsupported hits use existing ambient; bounce supports scene point/directional lights";
    }
    s.lightCount=status.lights;
    // ★★ Emissive ucgen SAYISI, `emissiveTriangles` -- CPU tablosunun boyu.
    //   0 ise shader hem NEE'yi hem emission bastirmasini kapatir.
    s.emissiveCount=status.emissiveTriangles;
    const auto device=m_device->getDevice();
    VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(m_device->getPhysicalDevice(),&props);
    if (!s.setLayout) {
        s.textureCount=(std::min)(uint32_t(VULKAN_TEXTURE_CAPACITY),
            props.limits.maxPerStageDescriptorSampledImages>2?props.limits.maxPerStageDescriptorSampledImages-2:1u);
        // No sampled material slot may silently turn into a placeholder or scalar.
        if (props.limits.maxPushConstantsSize<sizeof(GiPush)) {s.reason="screen GI push constants unsupported";return;}
        // ★ 13 binding: 12 = emissive ucgen tablosu. `rfEmissives` statik olarak
        //   kullanilan bir kaynak, yani BEYAN eden gecis onu BAGLAMAK zorunda.
        VkDescriptorSetLayoutBinding bindings[13]{};
        for (uint32_t i=0;i<13;++i) {
            bindings[i].binding=i;bindings[i].descriptorCount=1;
            bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        bindings[0].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        for (auto i:{2u,3u,7u}) bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[7].descriptorCount=s.textureCount;
        VkDescriptorSetLayoutCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount=13;ci.pBindings=bindings;
        if (vkCreateDescriptorSetLayout(device,&ci,nullptr,&s.setLayout)!=VK_SUCCESS) {s.reason="screen GI descriptor layout failed";return;}
    }
    for (const auto& m:m_cachedGpuMaterials) {
        for (uint32_t slot:{m.albedo_tex,m.emission_tex,m.opacity_tex,m.metallic_tex,m.specular_tex}) {
            if (slot>=s.textureCount) {s.reason="material texture exceeds screen GI descriptor capacity";return;}
        }
    }
    if (!s.pool) {
        VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,10},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,s.textureCount+2}};
        VkDescriptorPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.maxSets=1;ci.poolSizeCount=3;ci.pPoolSizes=sizes;
        if (vkCreateDescriptorPool(device,&ci,nullptr,&s.pool)!=VK_SUCCESS) {s.reason="screen GI descriptor pool failed";return;}
    }
    if (!s.set) {
        VkDescriptorSetAllocateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ci.descriptorPool=s.pool;ci.descriptorSetCount=1;ci.pSetLayouts=&s.setLayout;
        if (vkAllocateDescriptorSets(device,&ci,&s.set)!=VK_SUCCESS) {s.reason="screen GI descriptor allocation failed";return;}
    }
    if (!s.layout) {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(GiPush)};
        VkPipelineLayoutCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.setLayoutCount=1;ci.pSetLayouts=&s.setLayout;ci.pushConstantRangeCount=1;ci.pPushConstantRanges=&range;
        if (vkCreatePipelineLayout(device,&ci,nullptr,&s.layout)!=VK_SUCCESS) {s.reason="screen GI pipeline layout failed";return;}
    }
    if (!giPipeline(device,s.layout,m_rayFusionShaderDir+"/screen_gi_trace.spv",s.trace,s.reason) ||
        !giPipeline(device,s.layout,m_rayFusionShaderDir+"/screen_gi_filter.spv",s.filter,s.reason)) return;
    auto allocate=[&](VulkanRT::BufferHandle& b,VkDeviceSize bytes) {
        if (b.buffer && b.size>=bytes) return true;
        if (b.buffer) m_device->destroyBuffer(b);
        VulkanRT::BufferCreateInfo ci;ci.size=bytes;ci.location=VulkanRT::MemoryLocation::GPU_ONLY;
        ci.usage=VulkanRT::BufferUsage::STORAGE;b=m_device->createBuffer(ci);
        return b.buffer!=VK_NULL_HANDLE;
    };
    if (!allocate(s.raw,kGiHeader+uint64_t(s.width)*s.height*kGiPixel) || !allocate(s.counters,sizeof(RayFusion::BounceCounters))) {
        s.reason="screen GI scratch allocation failed";return;
    }
    // ★★★ Olcum tamponu HOST-VISIBLE: `downloadBuffer` onu dogrudan map eder,
    //   yani staging kopyasi, submit ve fence YOK. Bedeli bir kare gecikmedir
    //   ve o bedel bilincli: kareyi durduran bir enstruman, olctugu seyi
    //   degistirir.
    if (!s.stats.buffer) {
        VulkanRT::BufferCreateInfo sci;sci.size=sizeof(GiStatsBlock);
        sci.location=VulkanRT::MemoryLocation::GPU_TO_CPU;
        sci.usage=VulkanRT::BufferUsage::STORAGE|VulkanRT::BufferUsage::TRANSFER_DST;
        s.stats=m_device->createBuffer(sci);
    }
    if (!s.stats.buffer) {s.reason="screen GI stats allocation failed";return;}
    VkDescriptorImageInfo env{envSampler,envView,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo depth{m_interactiveViewport.postSampler,m_interactiveViewport.depthImage.view,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    std::vector<VkDescriptorImageInfo> textures(s.textureCount,env);
    for (const auto& entry:m_uploadedImages) {
        if (entry.first>0 && uint64_t(entry.first)<s.textureCount && entry.second.view && entry.second.sampler)
            textures[size_t(entry.first)]={entry.second.sampler,entry.second.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }
    const VkAccelerationStructureKHR tlas=m_device->getTLASHandle();
    VkWriteDescriptorSetAccelerationStructureKHR as{};as.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as.accelerationStructureCount=1;as.pAccelerationStructures=&tlas;
    VkDescriptorBufferInfo bufferInfos[]={{s.raw.buffer,0,VK_WHOLE_SIZE},{bounce[0].buffer,0,VK_WHOLE_SIZE},
        {bounce[1].buffer,0,VK_WHOLE_SIZE},{bounce[2].buffer,0,VK_WHOLE_SIZE},
        {s.counters.buffer,0,VK_WHOLE_SIZE},{s.output.buffer,0,VK_WHOLE_SIZE},
        {bounce[3].buffer,0,VK_WHOLE_SIZE},{}};
    VkWriteDescriptorSet writes[13]{};
    for (uint32_t i=0;i<13;++i) {
        writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=s.set;
        writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[0].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;writes[0].pNext=&as;
    writes[1].pBufferInfo=&bufferInfos[0];
    writes[2].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;writes[2].pImageInfo=&env;
    writes[3].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;writes[3].pImageInfo=&depth;
    for (uint32_t i=0;i<3;++i) writes[i+4].pBufferInfo=&bufferInfos[i+1];
    writes[7].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;writes[7].descriptorCount=s.textureCount;writes[7].pImageInfo=textures.data();
    writes[8].pBufferInfo=&bufferInfos[4];writes[9].pBufferInfo=&bufferInfos[5];
    // 10 ve 11 bu gecisin shader'inda YOK; kullanilmayan bir binding'i gecerli
    // bir tamponla doldurmak, layout'u tek sayi degistirmekten daha guvenli.
    // 10 artik GERCEK bir tampon: olcum sayaclari. 11 hala yer tutucu.
    bufferInfos[7]={s.stats.buffer,0,VK_WHOLE_SIZE};
    writes[10].pBufferInfo=&bufferInfos[7];writes[11].pBufferInfo=&bufferInfos[4];
    writes[12].pBufferInfo=&bufferInfos[6];
    // ★★★★★ 10 -> 13. Dizi 13 yazi ile DOLDURULUYORDU ama yalnizca ilk 10'u
    //   gonderiliyordu; binding 10, 11 ve 12 hic baglanmadi. 12 = `rfEmissives`,
    //   ve layout'un kendi yorumu "statik olarak kullanilan bir kaynak, yani
    //   BEYAN eden gecis onu BAGLAMAK zorunda" diyor. Baglanmamis bir
    //   descriptor'u statik olarak okumak TANIMSIZ davranistir; pratikte surucu
    //   sifir dondurur, yani emissive NEE sessizce HICBIR sey katmaz. Cokme
    //   yok, dogrulama katmani kapaliysa uyari yok, yalnizca eksik isik --
    //   ve eksik dolayli isik, ambient'in fazla baskin gorunmesinin
    //   sebeplerinden biridir.
    vkUpdateDescriptorSets(device,13,writes,0,nullptr);
    s.prepared=true;
    s.reason="same-frame diffuse GI; geometric depth normals; scene point/directional bounce only; unsupported hits fall back";
}

// Depth is already shader-readable inside recordRtShadowPass. No extra raster
// pass, camera ray, history image, CPU readback or progressive convergence.
void VulkanBackendAdapter::recordScreenGi(VkCommandBuffer cmd,const Matrix4x4& viewProj) {
    if (!m_screenGi || !m_screenGi->prepared) {
        markRasterStage(cmd,RasterStage::ScreenGiTrace,false);
        markRasterStage(cmd,RasterStage::ScreenGiFilter,false);
        return;
    }
    auto& s=*m_screenGi;
    GiPush push{};const auto inv=viewProj.inverse();
    for (int c=0;c<4;++c) for (int r=0;r<4;++r) push.invViewProj[c*4+r]=float(inv.m[r][c]);
    push.params[0]=float(s.emissiveCount);
    push.params[2]=m_screenGiSettings.maxDistance;
    // ★★★★★ Zamansal birikim karesi. Bunu gonderdigimiz an ekran GI'nin
    //   gurultusu TAA'nin ortalayabilecegi bir sey haline gelir: yonler her
    //   karede doner, yani 4 ornek x TAA hedefi kadar kare = o kadar etkin
    //   ornek, ISIN BUTCESI ARTMADAN.
    //
    // ★★★ Bu tohum bir zamanlar `m_taaFrameIndex`ti ve gerekcesi "TAA
    //   kapaliyken indis 0'da kalir, ortalanacak bir sey yokken donmus
    //   gurultu dogrudur" idi. Gerekce YANLIS DEGILDI, EKSIKTI: viewport'un
    //   bosta OLMADIGI ama TAA'nin doymus oldugu ara durumu kapsamiyordu.
    // ★★★★ YAKINSAMA sayaci degil, SERBEST AKAN kare sayaci. `m_taaFrameIndex`
    //   hedefte doyar ve tohumu tam da ortalamanin gerektigi anda dondururdu;
    //   gerekce VulkanBackend.h'de `m_rasterFrameCounter` uzerinde.
    // ★★ 4096'ya modlanir: `float`a cevriliyor (push alani vec4) ve buyuk
    //   tamsayilar float'ta tam temsil edilemez -- modlamadan tohum bir sure
    //   sonra ARTMAYI BIRAKIR, yani ayni ariza sessizce geri gelirdi.
    push.params2[0]=static_cast<float>(m_rasterFrameCounter % 4096u);
    push.params2[1]=m_cachedWorld.mode==WORLD_MODE_HDRI?(std::max)(m_cachedWorld.env_intensity,0.0f):1.0f;
    const auto status = rayFusionBounceStatus();
    uint32_t packedLightsHair = (status.lights & 0xFFFFu) | ((status.hairMaterials & 0xFFFFu) << 16u);
    std::memcpy(&push.params2[2], &packedLightsHair, sizeof(float));
    push.params2[3]=float(s.textureCount);
    push.shape[0]=s.width;push.shape[1]=s.height;
    push.shape[2]=m_screenGiSettings.samples;push.shape[3]=m_screenGiSettings.filterRadius;
    // Olcum sayaclarini SIFIRLA. Her kare bastan sayilir; birikmis bir toplam
    // "bu karede ne oldu"yu degil "uygulama acildigindan beri ne oldu"yu
    // raporlardi ve ikisi ayni sayiya benzer.
    if (s.stats.buffer) {
        const GiStatsBlock zeroStats{};
        giBarrier(cmd,s.stats.buffer,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_HOST_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdUpdateBuffer(cmd,s.stats.buffer,0,sizeof(GiStatsBlock),&zeroStats);
        giBarrier(cmd,s.stats.buffer,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    giBarrier(cmd,s.raw.buffer,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s.layout,0,1,&s.set,0,nullptr);
    vkCmdPushConstants(cmd,s.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(push),&push);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s.trace);
    vkCmdDispatch(cmd,(s.width+7)/8,(s.height+7)/8,1);
    giBarrier(cmd,s.raw.buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    markRasterStage(cmd,RasterStage::ScreenGiTrace,true);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s.filter);
    vkCmdDispatch(cmd,(s.width+7)/8,(s.height+7)/8,1);
    giBarrier(cmd,s.output.buffer,VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
    const uint32_t header[]={kGiMagic,s.width,s.height,m_screenGiSettings.samples};
    vkCmdUpdateBuffer(cmd,s.output.buffer,0,kGiHeader,header);
    giBarrier(cmd,s.output.buffer,VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    if (s.stats.buffer)
        giBarrier(cmd,s.stats.buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT);
    markRasterStage(cmd,RasterStage::ScreenGiFilter,true);
    s.recorded=true;
}
void VulkanBackendAdapter::destroyScreenGi() {
    if (!m_screenGi || !m_device) return;
    auto& s=*m_screenGi;const auto device=m_device->getDevice();
    if (s.trace) vkDestroyPipeline(device,s.trace,nullptr);
    if (s.filter) vkDestroyPipeline(device,s.filter,nullptr);
    if (s.layout) vkDestroyPipelineLayout(device,s.layout,nullptr);
    if (s.pool) vkDestroyDescriptorPool(device,s.pool,nullptr);
    if (s.setLayout) vkDestroyDescriptorSetLayout(device,s.setLayout,nullptr);
    for (auto b:{&s.raw,&s.output,&s.counters,&s.stats}) if (b->buffer) m_device->destroyBuffer(*b);
    m_screenGi.reset();
}
}
