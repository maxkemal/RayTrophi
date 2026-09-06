#include "Backend/VulkanBackend.h"
#include "Viewport/RasterGpuCull.h"
#include "Viewport/RasterInstanceUpload.h"
#include "globals.h"

#include <algorithm>
#include <cstring>

extern RenderSettings render_settings;

namespace Backend {

class MaterialPreviewTransmissionResources {
public:
    VulkanRT::ImageHandle sceneColor;
    VulkanRT::ImageHandle sceneDepth;
    VulkanRT::ImageHandle backDepth;
    VkSampler sampler = VK_NULL_HANDLE;
    VkRenderPass loadRenderPass = VK_NULL_HANDLE;
    VkFramebuffer loadFramebuffer = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    bool snapshotValid = false;
};

namespace {

void matrixToGL(const Matrix4x4& matrix, float out[16]) {
    const Matrix4x4 transposed = matrix.transpose();
    std::memcpy(out, transposed.m, sizeof(float) * 16u);
}

struct PreviewPush {
    float viewProj[16];
    float view[16];
    float cameraPos[4];
    float lightDir0[4];
    float lightDir1[4];
    float lightDir2[4];
    uint32_t materialMeta[4];
};
static_assert(sizeof(PreviewPush) == 208u, "Material preview push ABI changed");

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                  VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                  VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace

void VulkanBackendAdapter::destroyMaterialPreviewTransmissionResources() {
    if (!m_materialPreviewTransmission || !m_device) return;
    auto& r = *m_materialPreviewTransmission;
    VkDevice device = m_device->getDevice();
    if (r.loadFramebuffer) vkDestroyFramebuffer(device, r.loadFramebuffer, nullptr);
    if (r.loadRenderPass) vkDestroyRenderPass(device, r.loadRenderPass, nullptr);
    if (r.sampler) vkDestroySampler(device, r.sampler, nullptr);
    if (r.sceneColor.image) m_device->destroyImage(r.sceneColor);
    if (r.sceneDepth.image) m_device->destroyImage(r.sceneDepth);
    if (r.backDepth.image) m_device->destroyImage(r.backDepth);
    m_materialPreviewTransmission.reset();
}

bool VulkanBackendAdapter::ensureMaterialPreviewTransmissionResources(uint32_t width,
                                                                      uint32_t height) {
    if (!m_device || !m_interactiveViewport.colorImage.image ||
        !m_interactiveViewport.depthImage.image ||
        m_interactiveViewport.materialPreviewDescSet == VK_NULL_HANDLE) return false;
    if (m_materialPreviewTransmission &&
        m_materialPreviewTransmission->width == width &&
        m_materialPreviewTransmission->height == height) return true;

    destroyMaterialPreviewTransmissionResources();
    auto r = std::make_shared<MaterialPreviewTransmissionResources>();
    r->width = width;
    r->height = height;
    r->sceneColor = m_device->createImage2D(
        width, height, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    r->sceneDepth = m_device->createImage2D(
        width, height, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT);
    r->backDepth = m_device->createImage2D(
        width, height, VK_FORMAT_R32_UINT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    if (!r->sceneColor.image || !r->sceneDepth.image || !r->backDepth.image) {
        m_materialPreviewTransmission = r;
        destroyMaterialPreviewTransmissionResources();
        return false;
    }

    VkDevice device = m_device->getDevice();
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // D32 sampling is guaranteed with nearest filtering; linear-filter support
    // is optional on several otherwise valid viewport GPUs.
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(device, &sci, nullptr, &r->sampler) != VK_SUCCESS) {
        m_materialPreviewTransmission = r;
        destroyMaterialPreviewTransmissionResources();
        return false;
    }

    VkAttachmentDescription attachments[2]{};
    attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;
    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 2;
    rpci.pAttachments = attachments;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    if (vkCreateRenderPass(device, &rpci, nullptr, &r->loadRenderPass) != VK_SUCCESS) {
        m_materialPreviewTransmission = r;
        destroyMaterialPreviewTransmissionResources();
        return false;
    }
    VkImageView views[2] = {m_interactiveViewport.colorImage.view,
                            m_interactiveViewport.depthImage.view};
    VkFramebufferCreateInfo fbci{};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = r->loadRenderPass;
    fbci.attachmentCount = 2;
    fbci.pAttachments = views;
    fbci.width = width;
    fbci.height = height;
    fbci.layers = 1;
    if (vkCreateFramebuffer(device, &fbci, nullptr, &r->loadFramebuffer) != VK_SUCCESS) {
        m_materialPreviewTransmission = r;
        destroyMaterialPreviewTransmissionResources();
        return false;
    }

    VkDescriptorImageInfo images[2]{};
    images[0] = {r->sampler, r->sceneColor.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[1] = {r->sampler, r->sceneDepth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_interactiveViewport.materialPreviewDescSet;
        writes[i].dstBinding = 17u + i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &images[i];
    }
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    VkDescriptorImageInfo backInfo{};
    backInfo.imageView = r->backDepth.view;
    backInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet backWrite{};
    backWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    backWrite.dstSet = m_interactiveViewport.materialPreviewDescSet;
    backWrite.dstBinding = 19;
    backWrite.descriptorCount = 1;
    backWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    backWrite.pImageInfo = &backInfo;
    vkUpdateDescriptorSets(device, 1, &backWrite, 0, nullptr);
    m_materialPreviewTransmission = std::move(r);
    return true;
}

void VulkanBackendAdapter::prepareMaterialPreviewTransmissionThickness(VkCommandBuffer cmd) {
    if (!m_materialPreviewTransmission || !cmd) return;
    auto& r = *m_materialPreviewTransmission;
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, r.backDepth.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    imageBarrier(cmd, r.backDepth.image, VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void VulkanBackendAdapter::recordMaterialPreviewTransmissionPass(
    VkCommandBuffer cmd, const Matrix4x4& viewProj, const Matrix4x4& view,
    uint32_t width, uint32_t height) {
    if (!m_materialPreviewTransmission || !cmd ||
        m_interactiveViewport.materialPreviewPipeline == VK_NULL_HANDLE) return;
    auto& r = *m_materialPreviewTransmission;

    imageBarrier(cmd, m_interactiveViewport.colorImage.image, VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    imageBarrier(cmd, m_interactiveViewport.depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkImageLayout oldSnapshotLayout = r.snapshotValid
        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier(cmd, r.sceneColor.image, VK_IMAGE_ASPECT_COLOR_BIT,
                 oldSnapshotLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 r.snapshotValid ? VK_ACCESS_SHADER_READ_BIT : 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                 r.snapshotValid ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    imageBarrier(cmd, r.sceneDepth.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                 oldSnapshotLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 r.snapshotValid ? VK_ACCESS_SHADER_READ_BIT : 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                 r.snapshotValid ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageCopy colorCopy{};
    colorCopy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    colorCopy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    colorCopy.extent = {width, height, 1};
    vkCmdCopyImage(cmd, m_interactiveViewport.colorImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   r.sceneColor.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &colorCopy);
    VkImageCopy depthCopy = colorCopy;
    depthCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vkCmdCopyImage(cmd, m_interactiveViewport.depthImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   r.sceneDepth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &depthCopy);
    imageBarrier(cmd, r.sceneColor.image, VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    imageBarrier(cmd, r.sceneDepth.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    imageBarrier(cmd, m_interactiveViewport.colorImage.image, VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    imageBarrier(cmd, m_interactiveViewport.depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
    r.snapshotValid = true;
    imageBarrier(cmd, r.backDepth.image, VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = r.loadRenderPass;
    begin.framebuffer = r.loadFramebuffer;
    begin.renderArea.extent = {width, height};
    vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {width, height}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Participating media integrate against the opaque snapshot first. The SDF
    // dielectric is then allowed to replace nearer pixels and refract the same
    // stable snapshot without read/write attachment feedback.
    recordMaterialPreviewVolumePass(
        cmd, viewProj, view, width, height, true);
    recordMaterialPreviewSdfSurfacePass(
        cmd, viewProj, view, width, height, true);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_interactiveViewport.materialPreviewPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_interactiveViewport.materialPreviewPipelineLayout,
                            0, 1, &m_interactiveViewport.materialPreviewDescSet, 0, nullptr);

    PreviewPush push{};
    matrixToGL(viewProj, push.viewProj);
    matrixToGL(view, push.view);
    push.cameraPos[0] = m_camera.origin.x;
    push.cameraPos[1] = m_camera.origin.y;
    push.cameraPos[2] = m_camera.origin.z;
    uint32_t quality = 2u;
    if (::render_settings.raster_viewport_quality_preset == ::RasterViewportQualityPreset::Performance) quality = 1u;
    else if (::render_settings.raster_viewport_quality_preset == ::RasterViewportQualityPreset::Quality ||
             ::render_settings.raster_viewport_quality_preset == ::RasterViewportQualityPreset::Full) quality = 3u;
    push.materialMeta[0] = m_interactiveViewport.materialPreviewBoundMaterialCount;
    push.materialMeta[1] = quality | (2u << 8u);
    push.materialMeta[2] = uint32_t(::render_settings.material_preview_lighting_preset);
    push.materialMeta[3] = m_interactiveViewport.materialPreviewTextureArrayLen;
    vkCmdPushConstants(cmd, m_interactiveViewport.materialPreviewPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(push), &push);

    for (const auto& [key, mesh] : m_rasterMeshes) {
        const bool cullDriven = m_rasterGpuCullActive && m_rasterGpuCull &&
                                mesh.cullDrawSlot != UINT32_MAX;
        VkBuffer instances = cullDriven
            ? static_cast<VkBuffer>(m_rasterGpuCull->compactedInstanceBuffer())
            : ((m_rasterUseGlobalInstBuffer && m_rasterGlobalInstBuf)
                ? static_cast<VkBuffer>(m_rasterGlobalInstBuf->vkBuffer()) : mesh.instanceBuffer.buffer);
        if (!mesh.vertexBuffer.buffer || !mesh.normalBuffer.buffer || !mesh.matIdBuffer.buffer ||
            !mesh.uvBuffer.buffer || !instances || mesh.vertexCount == 0 ||
            (!cullDriven && mesh.instanceCount == 0)) continue;
        VkBuffer buffers[5] = {mesh.vertexBuffer.buffer, mesh.normalBuffer.buffer,
                               mesh.matIdBuffer.buffer, instances, mesh.uvBuffer.buffer};
        VkDeviceSize offsets[5] = {0, 0, 0,
            cullDriven ? VkDeviceSize(mesh.cullOutBase) * 64ull : 0, 0};
        vkCmdBindVertexBuffers(cmd, 0, 5, buffers, offsets);
        if (cullDriven) {
            VkBuffer indirect = static_cast<VkBuffer>(m_rasterGpuCull->commandBuffer());
            VkDeviceSize offset = VkDeviceSize(RasterGpuCull::commandOffset(mesh.cullDrawSlot));
            if (mesh.indexBuffer.buffer && mesh.indexCount) {
                vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexedIndirect(cmd, indirect, offset, 1, RasterGpuCull::kCommandStride);
            } else vkCmdDrawIndirect(cmd, indirect, offset, 1, RasterGpuCull::kCommandStride);
        } else {
            uint32_t first = m_rasterUseGlobalInstBuffer ? mesh.firstInstance : 0u;
            if (mesh.indexBuffer.buffer && mesh.indexCount) {
                vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, mesh.indexCount, mesh.instanceCount, 0, 0, first);
            } else vkCmdDraw(cmd, mesh.vertexCount, mesh.instanceCount, 0, first);
        }
    }
    vkCmdEndRenderPass(cmd);
}

} // namespace Backend
