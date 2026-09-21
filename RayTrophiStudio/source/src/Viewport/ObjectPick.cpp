// ============================================================================
// GPU OBJE SECIMI — raster viewport
// ============================================================================
//
// ★★★★★ NEDEN AYRI BIR GECIS. Ana render pass'e ikinci bir hedef eklemek
//   oradaki HER pipeline'i degistirirdi (solid, matcap, materyal,
//   transmission, derinlik on gecisi) ve hepsi render pass UYUMLULUGUNA
//   bagli. Bu gecis yalnizca TIK aninda kosar ve mevcut hicbir seye dokunmaz.
//   Bedeli bir tiklik: tam kare vertex isi, 1 piksellik raster.
//
// ★★★★ NEDEN GPU. CPU secimi bu depoda tekrar tekrar arizalandi, cunku
//   geometrinin CPU kopyasi ile GPU'nun CIZDIGI geometri ayri seylerdi
//   (bkz. docs/dev/POSTMORTEM_CPU_PICK_P_VS_PORIG.md). GPU secimi ayni
//   tamponlari, ayni matrisleri ve ayni cizim dongusunu kullanir; o sinifa
//   yapisal olarak bagisiktir.
//
// ★★★ KIMLIK COZUMLEMESI ADA UGRAMAZ. Eski OptiX GPU secimi tam da
//   ID -> ad -> secim onbellegi yolundan cozdugu icin kapatilmisti
//   ("GPU pick name lookup occasionally returns an unsafe path into the
//   selection cache", scene_ui_selection.cpp). Burada shader'in yazdigi
//   (meshSlot, instanceSlot) ciftinden `m_rasterInstances` indeksine
//   dogrudan gidilir:
//       rasterInstance = mesh.instanceIndices[instanceSlot - mesh.firstInstance]
//   Ad, yalnizca SONUCU raporlamak icin okunur; hicbir aramada kullanilmaz.
//
// ★★ GPU CULLING BILEREK ATLANIR. Sikistirilmis instance buffer'da
//   gl_InstanceIndex sikistirilmis yuvayi verir ve geri eslenmez. Secim
//   global (sikistirilmamis) buffer'dan cizer; gorunurluk kaybi yok, cunku
//   culling bir hizlandirmadir, bir sahne tanimi degil.

#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace Backend {

namespace {

constexpr VkFormat kPickFormat = VK_FORMAT_R32G32_UINT;

struct alignas(16) PickPush {
    float viewProj[16];
    uint32_t meshId;
    uint32_t firstInst;
    uint32_t pad0, pad1;
};
static_assert(sizeof(PickPush) == 80, "object_pick.vert push ABI");

bool loadModule(VkDevice device, const std::string& path, VkShaderModule& out,
                std::string& error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { error = "missing shader: " + path; return false; }
    const auto size = file.tellg();
    if (size <= 0 || static_cast<size_t>(size) % 4u) {
        error = "invalid SPIR-V: " + path; return false;
    }
    std::vector<uint32_t> words(static_cast<size_t>(size) / 4u);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), size);
    // ★ Sihirli sayi kontrolu: bayat veya yarim yazilmis bir .spv bu depoda
    //   bilinen bir tuzak, ve onsuz belirti "pipeline kuruldu ama ciz(e)medi".
    if (!file || words.empty() || words[0] != 0x07230203u) {
        error = "invalid SPIR-V: " + path; return false;
    }
    VkShaderModuleCreateInfo mi{};
    mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mi.codeSize = words.size() * 4u;
    mi.pCode = words.data();
    if (vkCreateShaderModule(device, &mi, nullptr, &out) != VK_SUCCESS) {
        error = "shader module failed: " + path; return false;
    }
    return true;
}

} // namespace

class ObjectPickResources {
public:
    VulkanRT::ImageHandle idImage{};
    VulkanRT::ImageHandle depthImage{};
    VulkanRT::BufferHandle readback{};
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    // Cizim sirasindaki mesh yuvalari. Shader meshId yazar, burasi anahtara
    // cevirir. Her secimde yeniden kurulur: mesh kumesi kareler arasinda
    // degisebilir ve bayat bir tablo YANLIS OBJE secerdi.
    std::vector<std::string> meshKeys;
};

void VulkanBackendAdapter::destroyObjectPick() {
    if (!m_objectPick || !m_device) { m_objectPick.reset(); return; }
    const VkDevice device = m_device->getDevice();
    auto& s = *m_objectPick;
    if (s.pipeline) vkDestroyPipeline(device, s.pipeline, nullptr);
    if (s.layout) vkDestroyPipelineLayout(device, s.layout, nullptr);
    if (s.framebuffer) vkDestroyFramebuffer(device, s.framebuffer, nullptr);
    if (s.renderPass) vkDestroyRenderPass(device, s.renderPass, nullptr);
    m_device->destroyImage(s.idImage);
    m_device->destroyImage(s.depthImage);
    m_device->destroyBuffer(s.readback);
    m_objectPick.reset();
}

bool VulkanBackendAdapter::pickObjectAtNormalized(float u, float v, ObjectPickResult& out) {
    out = ObjectPickResult{};
    const uint32_t w = m_interactiveViewport.width;
    const uint32_t h = m_interactiveViewport.height;
    if (w == 0u || h == 0u) { out.reason = "viewport has no size"; return false; }
    // v yukari dogru, piksel satiri asagi dogru.
    const int px = std::min<int>(int(w) - 1, (std::max)(0, int(u * float(w))));
    const int py = std::min<int>(int(h) - 1, (std::max)(0, int((1.0f - v) * float(h))));
    return pickObjectAtPixel(px, py, out);
}

bool VulkanBackendAdapter::pickObjectAtPixel(int x, int y, ObjectPickResult& out) {
    out = ObjectPickResult{};
    if (!m_device) { out.reason = "no device"; return false; }
    if (!m_rasterPickHasViewProj) {
        // ★ Matris karenin kendisinden gelir. Hic kare cizilmediyse secim
        //   yapilamaz -- burada bir matris UYDURMAK, cizilenden baska bir
        //   sahneyi secmek olurdu.
        out.reason = "no raster frame has been drawn yet";
        return false;
    }
    const uint32_t width = m_interactiveViewport.width;
    const uint32_t height = m_interactiveViewport.height;
    if (width == 0u || height == 0u) { out.reason = "viewport has no size"; return false; }
    if (x < 0 || y < 0 || static_cast<uint32_t>(x) >= width ||
        static_cast<uint32_t>(y) >= height) {
        out.reason = "pixel outside the viewport";
        return false;
    }
    if (m_rasterMeshes.empty() || m_rasterInstances.empty()) {
        out.ok = true; out.hit = false; out.reason = "no raster geometry";
        return true;
    }

    const VkDevice device = m_device->getDevice();
    if (!m_objectPick) m_objectPick = std::make_shared<ObjectPickResources>();
    auto& s = *m_objectPick;

    // ── Kaynaklar: boyut degistiyse yeniden kur ─────────────────────────────
    if (s.width != width || s.height != height) {
        if (s.framebuffer) { vkDestroyFramebuffer(device, s.framebuffer, nullptr); s.framebuffer = VK_NULL_HANDLE; }
        m_device->destroyImage(s.idImage);
        m_device->destroyImage(s.depthImage);
        s.width = 0; s.height = 0;
    }
    if (s.idImage.image == VK_NULL_HANDLE) {
        s.idImage = m_device->createImage2D(
            width, height, kPickFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        s.depthImage = m_device->createImage2D(
            width, height, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);
        if (!s.idImage.image || !s.depthImage.image) {
            out.reason = "pick targets could not be allocated";
            return false;
        }
        s.width = width; s.height = height;
    }
    if (!s.readback.buffer) {
        VulkanRT::BufferCreateInfo bi{};
        bi.size = sizeof(uint32_t) * 2u;
        bi.usage = VulkanRT::BufferUsage::TRANSFER_DST;
        // GPU_TO_CPU: `downloadBuffer` dogrudan map eder, kareyi DURDURMAZ.
        bi.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
        s.readback = m_device->createBuffer(bi);
        if (!s.readback.buffer) { out.reason = "pick readback buffer failed"; return false; }
    }

    // ── Render pass ─────────────────────────────────────────────────────────
    if (s.renderPass == VK_NULL_HANDLE) {
        VkAttachmentDescription att[2]{};
        att[0].format = kPickFormat;
        att[0].samples = VK_SAMPLE_COUNT_1_BIT;
        att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        att[1].format = VK_FORMAT_D32_SFLOAT;
        att[1].samples = VK_SAMPLE_COUNT_1_BIT;
        att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &colorRef;
        sub.pDepthStencilAttachment = &depthRef;

        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = 2; rp.pAttachments = att;
        rp.subpassCount = 1; rp.pSubpasses = &sub;
        if (vkCreateRenderPass(device, &rp, nullptr, &s.renderPass) != VK_SUCCESS) {
            out.reason = "pick render pass failed"; return false;
        }
    }
    if (s.framebuffer == VK_NULL_HANDLE) {
        VkImageView views[2] = { s.idImage.view, s.depthImage.view };
        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = s.renderPass;
        fb.attachmentCount = 2; fb.pAttachments = views;
        fb.width = width; fb.height = height; fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &s.framebuffer) != VK_SUCCESS) {
            out.reason = "pick framebuffer failed"; return false;
        }
    }

    // ── Pipeline ────────────────────────────────────────────────────────────
    if (s.pipeline == VK_NULL_HANDLE) {
        VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        std::string error;
        if (!loadModule(device, m_rayFusionShaderDir + "/object_pick.spv", vs, error) ||
            !loadModule(device, m_rayFusionShaderDir + "/object_pick_frag.spv", fs, error)) {
            if (vs) vkDestroyShaderModule(device, vs, nullptr);
            out.reason = error;
            return false;
        }
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcr.offset = 0; pcr.size = sizeof(PickPush);
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device, &pl, nullptr, &s.layout) != VK_SUCCESS) {
            vkDestroyShaderModule(device, vs, nullptr);
            vkDestroyShaderModule(device, fs, nullptr);
            out.reason = "pick pipeline layout failed"; return false;
        }

        // Binding 0 = konum (vertex), binding 1 = model matrisi (instance).
        // Ana gecisin binding 3'u ile AYNI stride (16 float) -- ayni buffer.
        VkVertexInputBindingDescription bind[2]{};
        bind[0].binding = 0; bind[0].stride = sizeof(float) * 3;
        bind[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bind[1].binding = 1; bind[1].stride = sizeof(float) * 16;
        bind[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

        VkVertexInputAttributeDescription attr[5]{};
        attr[0].location = 0; attr[0].binding = 0; attr[0].format = VK_FORMAT_R32G32B32_SFLOAT; attr[0].offset = 0;
        for (uint32_t i = 0; i < 4u; ++i) {
            attr[1 + i].location = 1 + i;
            attr[1 + i].binding = 1;
            attr[1 + i].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            attr[1 + i].offset = sizeof(float) * 4 * i;
        }

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 2; vi.pVertexBindingDescriptions = bind;
        vi.vertexAttributeDescriptionCount = 5; vi.pVertexAttributeDescriptions = attr;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        // ★★ YUZ AYIKLAMA YOK. Ana gecis ic mekanlarda cift yuzlu ciziyor ve
        //   bu sahnelerde duvar normalleri disa bakiyor; burada ayiklamak,
        //   kullanicinin GORDUGU bir yuzeyin secilememesi demek olurdu.
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
        cba.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &cba;

        const VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs; stages[1].pName = "main";

        VkGraphicsPipelineCreateInfo gp{};
        gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp.stageCount = 2; gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dynState;
        gp.layout = s.layout;
        gp.renderPass = s.renderPass;
        gp.subpass = 0;
        const VkResult pr = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp,
                                                      nullptr, &s.pipeline);
        vkDestroyShaderModule(device, vs, nullptr);
        vkDestroyShaderModule(device, fs, nullptr);
        if (pr != VK_SUCCESS) { out.reason = "pick pipeline failed"; return false; }
    }

    // ── Kayit ───────────────────────────────────────────────────────────────
    const auto started = std::chrono::steady_clock::now();
    s.meshKeys.clear();

    const bool useGlobal = m_rasterUseGlobalInstBuffer && m_rasterGlobalInstBuf &&
                           m_rasterGlobalInstBuf->isReady();
    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) { out.reason = "no command buffer"; return false; }

    VkClearValue clears[2]{};
    clears[0].color.uint32[0] = 0u;   // 0 = hicbir sey; shader kimligi +1 yazar
    clears[0].color.uint32[1] = 0u;
    clears[1].depthStencil = {1.0f, 0u};

    VkRenderPassBeginInfo rpb{};
    rpb.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpb.renderPass = s.renderPass;
    rpb.framebuffer = s.framebuffer;
    rpb.renderArea.offset = {0, 0};
    rpb.renderArea.extent = {width, height};
    rpb.clearValueCount = 2; rpb.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rpb, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vpRect{};
    vpRect.x = 0.0f; vpRect.y = 0.0f;
    vpRect.width = static_cast<float>(width);
    vpRect.height = static_cast<float>(height);
    vpRect.minDepth = 0.0f; vpRect.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vpRect);
    // ★★★★ SCISSOR TEK PIKSEL. Viewport tam boy kalir -- izdusum matematigi
    //   cizilen kareyle ayni olmak zorunda. Kirpilan sey yalnizca RASTER
    //   ciktisidir, yani maliyet piksel tarafinda neredeyse sifira iner ama
    //   secilen piksel cizilenle AYNI yerde kalir.
    VkRect2D scissor{};
    scissor.offset = {x, y};
    scissor.extent = {1u, 1u};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipeline);

    PickPush push{};
    const Matrix4x4& vpm = m_rasterPickViewProj;
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) push.viewProj[c * 4 + r] = vpm.m[r][c];

    for (const auto& [meshKey, rmb] : m_rasterMeshes) {
        VkBuffer instBuf = useGlobal
            ? static_cast<VkBuffer>(m_rasterGlobalInstBuf->vkBuffer())
            : rmb.instanceBuffer.buffer;
        if (!rmb.vertexBuffer.buffer || !instBuf || rmb.vertexCount == 0) continue;
        if (rmb.instanceCount == 0) continue;

        const uint32_t meshId = static_cast<uint32_t>(s.meshKeys.size());
        s.meshKeys.push_back(meshKey);

        push.meshId = meshId;
        push.firstInst = useGlobal ? rmb.firstInstance : 0u;
        vkCmdPushConstants(cmd, s.layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(PickPush), &push);

        VkBuffer bufs[2] = { rmb.vertexBuffer.buffer, instBuf };
        VkDeviceSize offs[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, bufs, offs);

        const uint32_t firstInst = useGlobal ? rmb.firstInstance : 0u;
        if (rmb.indexBuffer.buffer && rmb.indexCount > 0) {
            vkCmdBindIndexBuffer(cmd, rmb.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, rmb.indexCount, rmb.instanceCount, 0, 0, firstInst);
        } else {
            vkCmdDraw(cmd, rmb.vertexCount, rmb.instanceCount, 0, firstInst);
        }
    }
    vkCmdEndRenderPass(cmd);

    VkBufferImageCopy copy{};
    copy.bufferOffset = 0;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = {x, y, 0};
    copy.imageExtent = {1u, 1u, 1u};
    vkCmdCopyImageToBuffer(cmd, s.idImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           s.readback.buffer, 1, &copy);
    m_device->endSingleTimeCommands(cmd);

    uint32_t pixel[2] = {0u, 0u};
    m_device->downloadBuffer(s.readback, pixel, sizeof(pixel));
    out.gpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    out.ok = true;

    if (pixel[0] == 0u) { out.hit = false; return true; }  // bos piksel

    const uint32_t meshSlot = pixel[0] - 1u;   // shader +1 yazdi
    const uint32_t instSlot = pixel[1];
    if (meshSlot >= s.meshKeys.size()) {
        out.reason = "pick id out of range";
        return true;
    }
    out.mesh_key = s.meshKeys[meshSlot];
    auto meshIt = m_rasterMeshes.find(out.mesh_key);
    if (meshIt == m_rasterMeshes.end()) { out.reason = "mesh disappeared"; return true; }
    const auto& mesh = meshIt->second;

    // ★★★ KIMLIK COZUMLEMESI. Ad aramasi YOK: yuva dogrudan
    //   `m_rasterInstances` indeksine cevriliyor.
    const uint32_t base = useGlobal ? mesh.firstInstance : 0u;
    if (instSlot < base) { out.reason = "instance slot below mesh base"; return true; }
    const size_t local = static_cast<size_t>(instSlot - base);
    if (local >= mesh.instanceIndices.size()) {
        out.reason = "instance slot outside the mesh range";
        return true;
    }
    const uint32_t rasterIndex = mesh.instanceIndices[local];
    if (rasterIndex >= m_rasterInstances.size()) {
        out.reason = "raster instance index out of range";
        return true;
    }
    out.hit = true;
    out.instance_index = static_cast<int>(rasterIndex);
    out.object = m_rasterInstances[rasterIndex].nodeName;
    return true;
}

} // namespace Backend
