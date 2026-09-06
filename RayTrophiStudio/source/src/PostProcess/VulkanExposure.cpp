#include "PostProcess/VulkanExposure.h"
#include "PostProcess/Exposure.h"
#include <cstring>
namespace rtpost {
bool VulkanExposure::initialize(VkDevice d,VkPhysicalDevice physical,const std::vector<uint32_t>& code) {
    if(pipeline) return true;
    if(code.empty()) return false;
    device=d;
    auto fail=[&](){shutdown();return false;};
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0]={0,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    bindings[1]={1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    si.bindingCount=2;si.pBindings=bindings;
    if(vkCreateDescriptorSetLayout(d,&si,nullptr,&setLayout)!=VK_SUCCESS)return fail();
    VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,2},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets=2;pi.poolSizeCount=2;pi.pPoolSizes=sizes;
    if(vkCreateDescriptorPool(d,&pi,nullptr,&pool)!=VK_SUCCESS)return fail();
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,4};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount=1;li.pSetLayouts=&setLayout;li.pushConstantRangeCount=1;li.pPushConstantRanges=&range;
    if(vkCreatePipelineLayout(d,&li,nullptr,&layout)!=VK_SUCCESS)return fail();
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize=code.size()*4;mi.pCode=code.data();VkShaderModule module;
    if(vkCreateShaderModule(d,&mi,nullptr,&module)!=VK_SUCCESS)return fail();
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.layout=layout;ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";
    VkResult result=vkCreateComputePipelines(d,VK_NULL_HANDLE,1,&ci,nullptr,&pipeline);
    vkDestroyShaderModule(d,module,nullptr);
    if(result!=VK_SUCCESS)return fail();
    VkPhysicalDeviceMemoryProperties props;vkGetPhysicalDeviceMemoryProperties(physical,&props);
    for(auto& s:slots) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size=HistogramBins*4;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        if(vkCreateBuffer(d,&bi,nullptr,&s.buffer)!=VK_SUCCESS)return fail();
        VkMemoryRequirements req;vkGetBufferMemoryRequirements(d,s.buffer,&req);
        uint32_t type=UINT32_MAX;
        const auto flags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for(uint32_t i=0;i<props.memoryTypeCount;++i)
            if((req.memoryTypeBits&(1u<<i)) && (props.memoryTypes[i].propertyFlags&flags)==flags) {type=i;break;}
        if(type==UINT32_MAX)return fail();
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=type;
        if(vkAllocateMemory(d,&ai,nullptr,&s.memory)!=VK_SUCCESS)return fail();
        if(vkBindBufferMemory(d,s.buffer,s.memory,0)!=VK_SUCCESS)return fail();
        if(vkMapMemory(d,s.memory,0,VK_WHOLE_SIZE,0,&s.mapped)!=VK_SUCCESS)return fail();
        VkDescriptorSetAllocateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        di.descriptorPool=pool;di.descriptorSetCount=1;di.pSetLayouts=&setLayout;
        if(vkAllocateDescriptorSets(d,&di,&s.set)!=VK_SUCCESS)return fail();
    }
    return true;
}
void VulkanExposure::shutdown() {
    if(!device)return;
    for(auto& s:slots) {
        if(s.mapped)vkUnmapMemory(device,s.memory);
        if(s.buffer)vkDestroyBuffer(device,s.buffer,nullptr);
        if(s.memory)vkFreeMemory(device,s.memory,nullptr);
        s={};
    }
    if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);
    if(layout)vkDestroyPipelineLayout(device,layout,nullptr);
    if(pool)vkDestroyDescriptorPool(device,pool,nullptr);
    if(setLayout)vkDestroyDescriptorSetLayout(device,setLayout,nullptr);
    device=VK_NULL_HANDLE;pipeline=VK_NULL_HANDLE;layout=VK_NULL_HANDLE;pool=VK_NULL_HANDLE;setLayout=VK_NULL_HANDLE;
}
void VulkanExposure::consume(unsigned index) {
    if(index>=2)return;auto& s=slots[index];if(!s.pending)return;
    MeterResult h;h.generation=s.generation;std::memcpy(h.bins.data(),s.mapped,HistogramBins*4);
    s.pending=false;submitMeter(h,"Vulkan GPU histogram");
}
void VulkanExposure::record(VkCommandBuffer cmd,VkImageView hdr,unsigned index) {
    if(!pipeline || !hdr || index>=2 || !meterEnabled())return;
    auto& s=slots[index];
    VkDescriptorImageInfo image{VK_NULL_HANDLE,hdr,VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo buffer{s.buffer,0,HistogramBins*4};
    VkWriteDescriptorSet w[2]{};
    for(unsigned i=0;i<2;++i) {w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[i].dstSet=s.set;w[i].dstBinding=i;w[i].descriptorCount=1;}
    w[0].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;w[0].pImageInfo=&image;
    w[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[1].pBufferInfo=&buffer;
    vkUpdateDescriptorSets(device,2,w,0,nullptr);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&s.set,0,nullptr);
    const float center=meterSettings().center_weight;
    vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&center);
    vkCmdDispatch(cmd,1,1,1);
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&b,0,nullptr,0,nullptr);
    s.generation=meterGeneration();s.pending=true;
}
}
