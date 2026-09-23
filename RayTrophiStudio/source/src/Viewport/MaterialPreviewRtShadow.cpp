// Same-frame directional shadow service. Flat TLAS hit data, shared raster UV policy.
#include "Backend/VulkanBackend.h"
#include "Viewport/RasterImageBarrier.h"
#include "RayFusion/ProbeBounce.h"
#include "globals.h"
#include "Light.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace Backend {
namespace {
struct alignas(16) RtShadowPush {
    float invViewProj[16];
    float sunDirection[4];
    float params[4];
    uint32_t flags[4];
    uint32_t volumeMeta[4];
};
static_assert(sizeof(RtShadowPush) == 128u, "RT shadow push ABI");
// material_preview_rt_shadow.glsl: uvec4 rtShadowMeta + uvec4 rtShadowCoverage.
constexpr uint32_t kRtShadowHeaderBytes = 32u;
void bufferBarrier(VkCommandBuffer cmd, VkBuffer buffer, VkAccessFlags src,
                   VkAccessFlags dst, VkPipelineStageFlags from, VkPipelineStageFlags to) {
    VkBufferMemoryBarrier b{};
    b.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask=src; b.dstAccessMask=dst;
    b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    b.buffer=buffer; b.size=VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,from,to,0,0,nullptr,1,&b,0,nullptr);
}
}
class MaterialPreviewRtShadowResources {
public:
    VkDescriptorSetLayout setLayout=VK_NULL_HANDLE;
    VkDescriptorPool pool=VK_NULL_HANDLE;
    VkDescriptorSet set=VK_NULL_HANDLE;
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkPipeline pipeline=VK_NULL_HANDLE;
    VkDescriptorSet consumerSet=VK_NULL_HANDLE;
    VkBuffer consumerBuffer=VK_NULL_HANDLE;
    VulkanRT::BufferHandle mask{}, hits{}, materials{};
    uint32_t width=0, height=0, textureCount=1, instanceCount=0;
    VulkanBackendAdapter::RtShadowCoverage coverage{};
    bool prepared=false, recorded=false;
    std::string reason="not prepared";
};

bool VulkanBackendAdapter::ensureRtShadowResources(uint32_t width, uint32_t height) {
    if (!m_device || !m_device->isInitialized()) return false;
    if (!m_rtShadow) m_rtShadow=std::make_shared<MaterialPreviewRtShadowResources>();
    auto& s=*m_rtShadow;
    VkDevice vk=m_device->getDevice();
    // ★ Baslik IKI uvec4: metadata (magic,w,h,birincil isik) + coverage
    //   (sahne isigi bit maskesi, dunya gunesi bayragi). std430'da pixels[]
    //   bu yuzden 32. bayttan baslar -- shader'daki ile TEK sayi.
    const VkDeviceSize bytes=kRtShadowHeaderBytes+VkDeviceSize(width)*height*2u*sizeof(float);
    if (!s.mask.buffer || s.mask.size<bytes) {
        drainInteractiveViewportInFlight();
        if (s.mask.buffer) m_device->destroyBuffer(s.mask);
        s.consumerBuffer=VK_NULL_HANDLE;
        VulkanRT::BufferCreateInfo ci;
        ci.size=bytes; ci.location=VulkanRT::MemoryLocation::GPU_ONLY;
        ci.usage=VulkanRT::BufferUsage::STORAGE|VulkanRT::BufferUsage::TRANSFER_DST;
        s.mask=m_device->createBuffer(ci);
        if (!s.mask.buffer) {s.reason="shadow buffer allocation failed";return false;}
    }
    if (!width || !height) return true; // disabled path only needs the inert header
    s.width=width; s.height=height;
    if (s.pipeline) return true;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_device->getPhysicalDevice(),&props);
    s.textureCount=(std::min)(uint32_t(VULKAN_TEXTURE_CAPACITY),
        props.limits.maxPerStageDescriptorSampledImages>1u?
        props.limits.maxPerStageDescriptorSampledImages-1u:1u);
    VkDescriptorSetLayoutBinding bindings[7]{};
    for(uint32_t i=0;i<7;++i) {
        bindings[i].binding=i; bindings[i].descriptorCount=1;
        bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    bindings[0].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].descriptorType=bindings[5].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[5].descriptorCount=s.textureCount;
    if (!s.setLayout) {
        VkDescriptorSetLayoutCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount=7;ci.pBindings=bindings;
        if(vkCreateDescriptorSetLayout(vk,&ci,nullptr,&s.setLayout)!=VK_SUCCESS) {
            s.reason="shadow descriptor layout failed";return false;
        }
    }
    if(!s.pool) {
        VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,s.textureCount+1u}};
        VkDescriptorPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.maxSets=1;ci.poolSizeCount=3;ci.pPoolSizes=sizes;
        if(vkCreateDescriptorPool(vk,&ci,nullptr,&s.pool)!=VK_SUCCESS) {
            s.reason="shadow descriptor pool failed";return false;
        }
    }
    if(!s.set) {
        VkDescriptorSetAllocateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ci.descriptorPool=s.pool;ci.descriptorSetCount=1;ci.pSetLayouts=&s.setLayout;
        if(vkAllocateDescriptorSets(vk,&ci,&s.set)!=VK_SUCCESS) {
            s.reason="shadow descriptor allocation failed";return false;
        }
    }
    if(!s.layout) {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(RtShadowPush)};
        VkPipelineLayoutCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.setLayoutCount=1;ci.pSetLayouts=&s.setLayout;ci.pushConstantRangeCount=1;ci.pPushConstantRanges=&range;
        if(vkCreatePipelineLayout(vk,&ci,nullptr,&s.layout)!=VK_SUCCESS) {
            s.reason="shadow pipeline layout failed";return false;
        }
    }
    std::ifstream file(m_rayFusionShaderDir+"/rayfusion_rt_shadow.spv",std::ios::binary|std::ios::ate);
    if(!file) {s.reason="rayfusion_rt_shadow.spv missing; compile shaders";return false;}
    auto size=file.tellg();
    if(size<=0 || static_cast<size_t>(size)%4u) {s.reason="invalid shadow SPIR-V";return false;}
    std::vector<uint32_t> words(static_cast<size_t>(size)/4u);
    file.seekg(0);file.read(reinterpret_cast<char*>(words.data()),size);
    if(!file) {s.reason="shadow SPIR-V read failed";return false;}
    VkShaderModuleCreateInfo mi{};mi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mi.codeSize=words.size()*4u;mi.pCode=words.data();VkShaderModule module=VK_NULL_HANDLE;
    if(vkCreateShaderModule(vk,&mi,nullptr,&module)!=VK_SUCCESS) {s.reason="shadow shader module failed";return false;}
    VkComputePipelineCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";ci.layout=s.layout;
    VkResult result=vkCreateComputePipelines(vk,VK_NULL_HANDLE,1,&ci,nullptr,&s.pipeline);
    vkDestroyShaderModule(vk,module,nullptr);
    if(result!=VK_SUCCESS) {s.pipeline=VK_NULL_HANDLE;s.reason="shadow compute pipeline failed";return false;}
    return true;
}

bool VulkanBackendAdapter::rtShadowCoveredLight(RtShadowCoverage& coverage,
                                                std::string& reason) const {
    // Same constant the cascade record array and the shader's meta.w use: the
    // world sun sits one slot past the last scene light. Spelling 32 here again
    // would be a second copy of a number that already lives in globals.h.
    static constexpr uint32_t kWorldSunSlot =
        static_cast<uint32_t>(::kMaterialPreviewMaxSceneLights);
    // ~0.8 derece. Senkron kodu yonu BIREBIR kopyaladigi icin pratikte dot=1;
    // esik, elle bir kac ondalik oynatilmis bir isigi da ayni gunes saymak icin
    // var, ayri bir yonu kaza ile yutacak kadar genis degil.
    static constexpr float kAlignDot = 0.9999f;
    coverage = RtShadowCoverage{};
    coverage.primaryLight = kWorldSunSlot;
    reason.clear();
    if(!m_rtShadowAllowed){reason="disabled";return false;}
    if(!m_device || !m_device->isInitialized()){reason="no device";return false;}
    const auto& caps=m_device->getCapabilities();
    if(!caps.supportsRayQuery || !caps.supportsDescriptorIndexing) {
        reason="ray query and descriptor indexing are required";return false;
    }
    if(m_viewportMode!=ViewportMode::MaterialPreview) {
        reason="material preview viewport is not active";return false;
    }
    if(::render_settings.material_preview_lighting_preset!=MaterialPreviewLightingPreset::Scene) {
        reason="requires scene lighting";return false;
    }
    if(m_rasterInstances.empty()){reason="no raster geometry";return false;}
    if(m_rtShadowHasMaterialPrograms) {
        reason="material graphs require cascade fallback; graph opacity prepass is unavailable";
        return false;
    }
    auto normalized=[](Vec3 v,Vec3& out)->bool{
        const float length=v.length();
        if(!std::isfinite(length) || length<1e-6f)return false;
        out=v*(1.0f/length);return true;
    };
    Vec3 sunDir{};
    const bool haveWorldSun =
        m_cachedWorld.mode==WORLD_MODE_NISHITA &&
        m_cachedWorld.nishita.sun_intensity>0.0f &&
        normalized(Vec3(m_cachedWorld.nishita.sun_direction.x,
                        m_cachedWorld.nishita.sun_direction.y,
                        m_cachedWorld.nishita.sun_direction.z),sunDir);

    // The scene light index the shading shader uses. Invisible lights are
    // skipped WITHOUT advancing it, exactly as the GPU light packer and the
    // cascade builder do -- all three walk m_cachedLights the same way, and a
    // fourth walk that counted differently would shadow the wrong light.
    struct Candidate { uint32_t index; Vec3 toLight; };
    std::vector<Candidate> directionals;
    uint32_t index=0;
    for(const auto& light:m_cachedLights) {
        if(!light || !light->visible)continue;
        if(index>=kWorldSunSlot)break;
        if(light->type()==LightType::Directional && light->intensity>0.0f) {
            Vec3 toLight{};
            if(normalized(light->direction*-1.0f,toLight))
                directionals.push_back({index,toLight});
        }
        ++index;
    }
    // Birincil yon: ilk kullanilabilir directional, yoksa dunya gunesi. Sira
    // eskisiyle ayni tutuldu -- maskenin meta.w'si ve `reason` metinleri bu
    // secime bagli ve bir onceki olcumlerle karsilastirilabilir kalmali.
    if(!directionals.empty()) {
        coverage.primaryLight=directionals.front().index;
        coverage.toLight=directionals.front().toLight;
    } else if(haveWorldSun) {
        coverage.primaryLight=kWorldSunSlot;
        coverage.toLight=sunDir;
    } else {
        reason="no valid directional light or world sun";return false;
    }
    for(const auto& candidate:directionals) {
        if(Vec3::dot(candidate.toLight,coverage.toLight)>=kAlignDot)
            coverage.sceneLightMask|=(1u<<candidate.index);
    }
    coverage.worldSun = haveWorldSun && Vec3::dot(sunDir,coverage.toLight)>=kAlignDot;
    return true;
}

bool VulkanBackendAdapter::prepareRtShadowFrame(VkCommandBuffer cmd,uint32_t width,uint32_t height,bool eligible) {
    if(!m_device || !cmd) return false;
    resetScreenGiFrame(cmd,width,height);
    // Always publish an inert header, including after disable or failed preparation.
    // An allocated scene-global buffer is a safe descriptor fallback: its first
    // word is a light count (<=32), never the shadow magic 0x52545348.
    ensureRtShadowResources(0,0);
    if(!m_rtShadow) return false;
    auto& s=*m_rtShadow;s.prepared=false;s.recorded=false;
    // ★★★ Cleared FIRST, set only on the success path below. Every early return
    //   in this function is a frame the ray pass does not own the light, and the
    //   cascade builder reads this next frame to decide whether to stand down.
    //   Clearing at the top means a return added later cannot forget to do it --
    //   the fail-safe direction is "draw the cascade", never "leave a hole".
    m_rtShadowCoveredLastFrame=false;
    auto bindConsumer=[&]() {
        VkDescriptorBufferInfo info{};
        info.buffer=s.mask.buffer?s.mask.buffer:m_interactiveViewport.materialPreviewSceneGlobals.buffer;
        info.range=VK_WHOLE_SIZE;
        if(!info.buffer || !m_interactiveViewport.materialPreviewDescSet)return;
        if(s.consumerSet==m_interactiveViewport.materialPreviewDescSet && s.consumerBuffer==info.buffer)return;
        drainInteractiveViewportInFlight();
        VkWriteDescriptorSet w{};w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet=m_interactiveViewport.materialPreviewDescSet;w.dstBinding=23;
        w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.descriptorCount=1;w.pBufferInfo=&info;
        vkUpdateDescriptorSets(m_device->getDevice(),1,&w,0,nullptr);
        s.consumerSet=w.dstSet;s.consumerBuffer=info.buffer;
    };
    auto clearHeader=[&]() {
        if(!s.mask.buffer)return;
        bufferBarrier(cmd,s.mask.buffer,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const uint32_t header[8]={};
        vkCmdUpdateBuffer(cmd,s.mask.buffer,0,kRtShadowHeaderBytes,header);
        bufferBarrier(cmd,s.mask.buffer,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    };
    // The header must be cleared on the FINAL allocation; do not replace a
    // buffer after recording commands that refer to it.
    if(!m_rtShadowAllowed || !eligible) {
        s.reason=!m_rtShadowAllowed?"disabled":"requires scene lighting, HDR and depth prepass";
        bindConsumer();clearHeader();return false;
    }
    drainInteractiveViewportInFlight(); // shared compute descriptors and CPU upload tables
    const auto& caps=m_device->getCapabilities();
    if(!caps.supportsRayQuery || !caps.supportsDescriptorIndexing) {
        s.reason="ray query and descriptor indexing are required";
        bindConsumer();clearHeader();return false;
    }
    const bool resources=ensureRtShadowResources(width,height);
    bindConsumer();clearHeader();
    if(!resources || !s.mask.buffer)return false;
    if(!m_interactiveViewport.depthImage.view || !m_interactiveViewport.postSampler) {
        s.reason="viewport depth image or sampler is not ready";return false;
    }
    const VkAccelerationStructureKHR tlas=m_device->getTLASHandle();
    RayFusionSceneASStatus as{};getRayFusionSceneASStatus(as);
    if(tlas == VK_NULL_HANDLE || !as.ready || as.instances_skipped || as.meshes_skipped) {
        s.reason="scene TLAS is missing or incomplete; using cascades";return false;
    }
    if(m_rtShadowHasMaterialPrograms) {
        s.reason="material graphs require cascade fallback; graph opacity prepass is unavailable";return false;
    }
    std::vector<RayFusion::HitInstance> hits;
    if(!getRayFusionHitInstances(hits) || m_cachedGpuMaterials.empty()) {
        s.reason="flat hit/material tables are not ready";return false;
    }
    // Same call the cascade builder made when it stood down. Anything this
    // rejects, the cascade is still drawing -- the two cannot disagree.
    if(!rtShadowCoveredLight(s.coverage,s.reason))return false;
    std::vector<VulkanRT::VkGpuMaterialCore> materials;
    for(const auto& material:m_cachedGpuMaterials) {
        VulkanRT::VkGpuMaterialCore core{};VulkanRT::VkGpuMaterialExt ext{};
        VulkanRT::splitGpuMaterial(material,core,ext);materials.push_back(core);
        const auto id=core.opacity_tex;
        if(id && (id>=s.textureCount || m_uploadedImages.find(id)==m_uploadedImages.end() ||
            !m_uploadedImages.at(id).view || !m_uploadedImages.at(id).sampler)) {
            s.reason="opacity texture unavailable; using cascades";return false;
        }
    }
    auto upload=[&](VulkanRT::BufferHandle& buffer,const void* data,VkDeviceSize size) {
        if(!buffer.buffer || buffer.size<size) {
            if(buffer.buffer)m_device->destroyBuffer(buffer);
            VulkanRT::BufferCreateInfo ci;ci.size=size;ci.usage=VulkanRT::BufferUsage::STORAGE;
            ci.location=VulkanRT::MemoryLocation::CPU_TO_GPU;buffer=m_device->createBuffer(ci);
        }
        if(!buffer.buffer)return false;
        m_device->uploadBuffer(buffer,data,size);return true;
    };
    if(!upload(s.hits,hits.data(),hits.size()*sizeof(hits[0])) ||
       !upload(s.materials,materials.data(),materials.size()*sizeof(materials[0]))) {
        s.reason="shadow hit/material upload allocation failed";return false;
    }
    s.instanceCount=static_cast<uint32_t>(hits.size());
    VkDescriptorImageInfo depth{};depth.sampler=m_interactiveViewport.postSampler;
    depth.imageView=m_interactiveViewport.depthImage.view;depth.imageLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    // All slots are initialized. Unused slots use the valid depth descriptor;
    // CPU validation above guarantees no opacity texture reads those slots.
    std::vector<VkDescriptorImageInfo> textures(s.textureCount,depth);
    for(const auto& entry:m_uploadedImages) {
        if(entry.first<=0 || uint64_t(entry.first)>=s.textureCount || !entry.second.view || !entry.second.sampler)continue;
        textures[size_t(entry.first)]={entry.second.sampler,entry.second.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }
    VkDescriptorBufferInfo buffers[]={{s.mask.buffer,0,VK_WHOLE_SIZE},
        {s.hits.buffer,0,VK_WHOLE_SIZE},{s.materials.buffer,0,VK_WHOLE_SIZE}};
    const VkBuffer volumeBuffer = m_device->m_volumeBuffer.buffer
        ? m_device->m_volumeBuffer.buffer
        : s.mask.buffer;
    VkDescriptorBufferInfo volumeInfo{volumeBuffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSetAccelerationStructureKHR acceleration{};
    acceleration.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    acceleration.accelerationStructureCount=1;acceleration.pAccelerationStructures=&tlas;
    VkWriteDescriptorSet writes[7]{};
    for(uint32_t i=0;i<7;++i) {
        writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=s.set;
        writes[i].dstBinding=i;writes[i].descriptorCount=1;
        writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[0].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;writes[0].pNext=&acceleration;
    writes[1].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;writes[1].pImageInfo=&depth;
    for(uint32_t i=0;i<3;++i)writes[i+2].pBufferInfo=&buffers[i];
    writes[5].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[5].descriptorCount=s.textureCount;writes[5].pImageInfo=textures.data();
    writes[6].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[6].pBufferInfo=&volumeInfo;
    vkUpdateDescriptorSets(m_device->getDevice(),7,writes,0,nullptr);
    s.reason.clear();s.prepared=true;m_rtShadowCoveredLastFrame=true;
    prepareScreenGiFrame();
    return true;
}

void VulkanBackendAdapter::recordRtShadowPass(VkCommandBuffer cmd,const Matrix4x4& viewProj,
                                               uint32_t width,uint32_t height) {
    if(!m_rtShadow || !m_rtShadow->prepared || !cmd)return;
    auto& s=*m_rtShadow;
    rasterImageBarrier(cmd,m_interactiveViewport.depthImage.image,VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    bufferBarrier(cmd,s.mask.buffer,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const uint32_t header[]={0x52545348u,width,height,s.coverage.primaryLight,
                            s.coverage.sceneLightMask,s.coverage.worldSun?1u:0u,0u,0u};
    vkCmdUpdateBuffer(cmd,s.mask.buffer,0,sizeof(header),header);
    bufferBarrier(cmd,s.mask.buffer,VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    RtShadowPush push{};const Matrix4x4 inv=viewProj.inverse();
    for(int c=0;c<4;++c)for(int r=0;r<4;++r)push.invViewProj[c*4+r]=float(inv.m[r][c]);
    push.sunDirection[0]=s.coverage.toLight.x;push.sunDirection[1]=s.coverage.toLight.y;
    push.sunDirection[2]=s.coverage.toLight.z;
    push.params[0]=0.02f;push.params[1]=1e5f;push.params[2]=float(width);push.params[3]=float(height);
    push.flags[0]=s.textureCount;push.flags[1]=uint32_t(m_cachedGpuMaterials.size());
    push.flags[2]=s.instanceCount;push.flags[3]=s.coverage.primaryLight;
    push.volumeMeta[0] = m_device->m_volumeBuffer.buffer
        ? m_device->m_volumeCount
        : 0u;
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s.pipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s.layout,0,1,&s.set,0,nullptr);
    vkCmdPushConstants(cmd,s.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(push),&push);
    vkCmdDispatch(cmd,(width+7u)/8u,(height+7u)/8u,1);
    bufferBarrier(cmd,s.mask.buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    markRasterStage(cmd,RasterStage::RtShadow,true);
    recordScreenGi(cmd,viewProj);
    rasterImageBarrier(cmd,m_interactiveViewport.depthImage.image,VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
    s.recorded=true;
}
void VulkanBackendAdapter::resumeRtShadowShading(VkCommandBuffer cmd,const Matrix4x4& viewProj,
                                                 const VkRenderPassBeginInfo& original) {
    vkCmdEndRenderPass(cmd);
    recordRtShadowPass(cmd,viewProj,original.renderArea.extent.width,original.renderArea.extent.height);
    auto load=original;load.renderPass=m_interactiveViewport.hdrRenderPassLoad;
    load.clearValueCount=0;load.pClearValues=nullptr;
    vkCmdBeginRenderPass(cmd,&load,VK_SUBPASS_CONTENTS_INLINE);
}
void VulkanBackendAdapter::updateRtShadowMaterialPrograms(const std::vector<uint32_t>& words) {
    m_rtShadowHasMaterialPrograms=false;
    if(words.empty())return;
    const uint32_t count=words[0];
    if(size_t(count)>=words.size()) {m_rtShadowHasMaterialPrograms=true;return;}
    for(uint32_t i=0;i<count;++i)if(words[1u+i]!=UINT32_MAX) {
        m_rtShadowHasMaterialPrograms=true;return;
    }
}
bool VulkanBackendAdapter::setRtShadow(bool enabled) {
    if(m_rtShadowAllowed==enabled)return true;
    m_rtShadowAllowed=enabled;
    m_rtShadowCoveredLastFrame=false;m_rtShadowCascadesReplaced=0;
    if(m_rtShadow){m_rtShadow->prepared=false;m_rtShadow->recorded=false;}
    m_interactiveViewport.dirty=true;return true;
}
bool VulkanBackendAdapter::getRtShadowStatus(bool& supported,bool& ready,uint32_t& rays,
                                             uint32_t& cascadesReplaced,std::string& reason) const {
    cascadesReplaced=m_rtShadowCascadesReplaced;
    supported=m_device && m_device->isInitialized() && m_device->getCapabilities().supportsRayQuery &&
        m_device->getCapabilities().supportsDescriptorIndexing;
    ready=false;rays=0;
    if(!m_rtShadowAllowed){reason="disabled";return supported;}
    if(m_viewportMode!=ViewportMode::MaterialPreview){reason="material preview viewport is not active";return supported;}
    if(m_rasterInstances.empty() || ::render_settings.material_preview_lighting_preset!=MaterialPreviewLightingPreset::Scene) {
        reason="requires mesh geometry and scene lighting";return supported;
    }
    if(!m_rtShadow){reason="not prepared";return supported;}
    const auto& s=*m_rtShadow;
    ready=supported && s.recorded && s.prepared && m_device->getTLASHandle()!=VK_NULL_HANDLE;
    rays=ready?s.width*s.height:0u;reason=s.reason;
    if(!ready && reason.empty())reason="shadow dispatch has not been recorded";
    return supported;
}
void VulkanBackendAdapter::destroyRtShadowResources() {
    destroyScreenGi();
    // ★★ GPU secim gecisi de burada sokulur: kendi render pass'i, framebuffer'i
    //   ve iki goruntusu AYNI VkDevice'a bagli. Viewport kaynaklari yeniden
    //   yaratildiginda geride kalan bir framebuffer, artik var olmayan
    //   goruntulere isaret eden bayat bir tutamak olurdu.
    destroyObjectPick();
    // ★ Yansima RT golgeye BAGLI degil, ama viewport kaynaklari sokulurken onun
    //   da sokulmesi gerek: descriptor'lari ayni VkDevice'in goruntulerine
    //   (G-buffer, HDR hedefi) isaret ediyor. Burada birakmak, yeniden
    //   yaratilmis goruntulere bakan bayat bir set demek olurdu.
    destroyReflection();
    if(!m_rtShadow || !m_device)return;
    auto& s=*m_rtShadow;auto vk=m_device->getDevice();
    if(s.pipeline)vkDestroyPipeline(vk,s.pipeline,nullptr);
    if(s.layout)vkDestroyPipelineLayout(vk,s.layout,nullptr);
    if(s.pool)vkDestroyDescriptorPool(vk,s.pool,nullptr);
    if(s.setLayout)vkDestroyDescriptorSetLayout(vk,s.setLayout,nullptr);
    for(auto* buffer:{&s.mask,&s.hits,&s.materials})if(buffer->buffer)m_device->destroyBuffer(*buffer);
    m_rtShadow.reset();
}
} // namespace Backend
