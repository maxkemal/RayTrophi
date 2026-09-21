/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          RasterTaa.cpp
 * Description:   Realtime raster temporal anti-aliasing: jittered sampling
 *                accumulated across frames. Closes TWO gaps at once -- the
 *                raster viewport had no anti-aliasing of any kind, and its
 *                screen-space GI/reflections had no temporal history, so
 *                both edges and Monte Carlo noise were spent every frame.
 * =========================================================================
 */
#include "Backend/VulkanViewportBackend.h"
#include "globals.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace Backend {
namespace {

// ★★★ Push ABI, `raster_taa.comp` ile BIREBIR. Bir alani buradan ekleyip
//   shader'dan eklememek, sessizce kaymis bir matris demektir ve belirtisi
//   "TAA hayalet birakiyor" olur -- yani yanlis yerde aranir.
struct alignas(16) RasterTaaPush {
    // ★★★★ TEK matris, iki degil: `prevViewProj * inverse(viewProj)`. Iki mat4
    //   160 bayt eder ve `maxPushConstantsSize` bircok GPU'da TAM 128'dir --
    //   pipeline kurulmaz, belirti "bu makinede TAA yok" olur. Carpimi burada
    //   bir kez yapmak piksel basina bir matris carpimini da kaldiriyor.
    float reprojection[16];
    uint32_t width;
    uint32_t height;
    float blend;
    uint32_t historyValid;
};
static_assert(sizeof(RasterTaaPush) == 80, "raster_taa.comp push ABI");

std::vector<uint32_t> loadTaaSPV(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) return {};
    std::vector<uint32_t> buffer(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) return {};
    return buffer;
}

// ★ Aspect PARAMETRE, sabit degil. Ilk yazimda COLOR sabitlenmisti ve ayni
//   yardimci derinlik goruntusu icin de cagriliyordu: validation hatasi verir,
//   vermezse de bariyer YANLIS alt kaynagi kapsar -- yani "bazen titriyor".
void taaImageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                     VkImageLayout oldLayout, VkImageLayout newLayout,
                     VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                     VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void matrixToGL(const Matrix4x4& m, float* out) {
    // Column-major for GLSL `mat4`, matching every other push in this backend.
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            out[c * 4 + r] = m.m[r][c];
}

} // namespace

// ───────────────────────────────────────────────────────────────────────────
// Kaynaklar
// ───────────────────────────────────────────────────────────────────────────
bool VulkanViewportBackend::ensureRasterTaaResources(uint32_t width, uint32_t height) {
    if (!m_device || !m_device->isInitialized() || width == 0u || height == 0u) return false;
    VkDevice vkDevice = m_device->getDevice();

    // ★ HDR ve derinlik ornekleyicisi post ile PAYLASILIR (NEAREST, ve dogrusu
    //   o: ikisini de texel merkezinden okuyoruz). Post kurulamadiysa o
    //   ornekleyici de yoktur; TAA'yi gecersiz bir descriptor ile kurmak
    //   yerine kapaniriz.
    if (m_interactiveViewport.postSampler == VK_NULL_HANDLE) return false;

    const bool sized =
        m_interactiveViewport.taaHistoryImage[0].image != VK_NULL_HANDLE &&
        m_interactiveViewport.taaHistoryImage[0].width == width &&
        m_interactiveViewport.taaHistoryImage[0].height == height;
    if (sized && m_interactiveViewport.taaPipeline != VK_NULL_HANDLE) return true;

    // Pipeline bir kez kurulur; yalnizca goruntuler yeniden boyutlanir.
    if (m_interactiveViewport.taaPipeline == VK_NULL_HANDLE) {
        std::string shaderDir = "shaders";
        if (!std::filesystem::exists(shaderDir + "/raster_post.spv")) shaderDir = "source/shaders";
        if (!std::filesystem::exists(shaderDir + "/raster_post.spv")) shaderDir = "../shaders";
        const std::vector<uint32_t> spv = loadTaaSPV(shaderDir + "/raster_taa.spv");
        if (spv.empty()) {
            static bool warned = false;
            if (!warned) {
                SCENE_LOG_WARN("[Viewport] raster_taa.spv missing; the realtime viewport keeps "
                               "rendering WITHOUT anti-aliasing and without temporal "
                               "accumulation. Run compile_shaders.bat.");
                warned = true;
            }
            return false;
        }
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode = spv.data();
        VkShaderModule module = VK_NULL_HANDLE;
        bool ok = vkCreateShaderModule(vkDevice, &smci, nullptr, &module) == VK_SUCCESS;

        if (ok) {
            VkDescriptorSetLayoutBinding binds[4]{};
            for (uint32_t i = 0; i < 4; ++i) {
                binds[i].binding = i;
                binds[i].descriptorCount = 1;
                binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                binds[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            }
            binds[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            VkDescriptorSetLayoutCreateInfo dslci{};
            dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dslci.bindingCount = 4;
            dslci.pBindings = binds;
            ok = vkCreateDescriptorSetLayout(vkDevice, &dslci, nullptr,
                                             &m_interactiveViewport.taaDescLayout) == VK_SUCCESS;
        }
        if (ok) {
            // Iki set: ping-pong'un iki parite durumu.
            VkDescriptorPoolSize sizes[2]{};
            sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sizes[0].descriptorCount = 6;
            sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            sizes[1].descriptorCount = 2;
            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets = 2;
            dpci.poolSizeCount = 2;
            dpci.pPoolSizes = sizes;
            ok = vkCreateDescriptorPool(vkDevice, &dpci, nullptr,
                                        &m_interactiveViewport.taaDescPool) == VK_SUCCESS;
        }
        if (ok) {
            VkDescriptorSetLayout layouts[2] = {m_interactiveViewport.taaDescLayout,
                                                m_interactiveViewport.taaDescLayout};
            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool = m_interactiveViewport.taaDescPool;
            dsai.descriptorSetCount = 2;
            dsai.pSetLayouts = layouts;
            ok = vkAllocateDescriptorSets(vkDevice, &dsai,
                                          m_interactiveViewport.taaDescSet) == VK_SUCCESS;
        }
        if (ok && m_interactiveViewport.taaHistorySampler == VK_NULL_HANDLE) {
            // LINEAR: yeniden izdusum alt-piksel bir konuma dusuyor ve o konumu
            // yuvarlamak TAA'yi sessizce ise yaramaz hale getirir.
            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = sci.addressModeV = sci.addressModeW =
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod = 0.0f;
            ok = vkCreateSampler(vkDevice, &sci, nullptr,
                                 &m_interactiveViewport.taaHistorySampler) == VK_SUCCESS;
        }
        if (ok) {
            VkPushConstantRange pcr{};
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.offset = 0;
            pcr.size = sizeof(RasterTaaPush);
            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &m_interactiveViewport.taaDescLayout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcr;
            ok = vkCreatePipelineLayout(vkDevice, &plci, nullptr,
                                        &m_interactiveViewport.taaPipelineLayout) == VK_SUCCESS;
        }
        if (ok) {
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = module;
            stage.pName = "main";
            VkComputePipelineCreateInfo cpci{};
            cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage = stage;
            cpci.layout = m_interactiveViewport.taaPipelineLayout;
            ok = vkCreateComputePipelines(vkDevice, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                          &m_interactiveViewport.taaPipeline) == VK_SUCCESS;
        }
        if (module != VK_NULL_HANDLE) vkDestroyShaderModule(vkDevice, module, nullptr);
        if (!ok) {
            SCENE_LOG_WARN("[Viewport] TAA pipeline creation failed; the realtime viewport "
                           "renders without anti-aliasing.");
            destroyRasterTaaResources(false);
            return false;
        }
    }

    // ── Gecmis goruntuleri ──────────────────────────────────────────────────
    for (auto& img : m_interactiveViewport.taaHistoryImage) {
        if (img.image) { m_device->destroyImage(img); img = {}; }
    }
    for (auto& img : m_interactiveViewport.taaHistoryImage) {
        img = m_device->createImage2D(
            width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
            // SAMPLED: gecmis okunur. STORAGE: cozum yazilir. TRANSFER_SRC:
            // cozulmus kare post'un okudugu HDR hedefine kopyalanir.
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        if (!img.image) {
            SCENE_LOG_WARN("[Viewport] TAA history allocation failed; anti-aliasing is off.");
            destroyRasterTaaResources(true);
            return false;
        }
    }

    // ★ Yeniden boyutlanan gecmis GECERSIZDIR. Bunu soylemeyi unutmak, ilk
    //   karede eski boyuttan kalma icerigi yeni cozunurluge esnetmek olurdu.
    m_taaFrameIndex = 0;
    m_taaHasPrevViewProj = false;

    // Goruntuler GENERAL'e alinir ve ORADA kalir: bu gecis ayni goruntuye hem
    // ornekleyerek hem storage olarak eriyor, layout gidip gelmesi bedava degil.
    if (VkCommandBuffer init = m_device->beginSingleTimeCommands()) {
        for (auto& img : m_interactiveViewport.taaHistoryImage) {
            taaImageBarrier(init, img.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                            0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        m_device->endSingleTimeCommands(init);
    }

    // ── Descriptor'lar: parite i -> cikti history[i], girdi history[1-i] ────
    for (uint32_t i = 0; i < 2; ++i) {
        VkDescriptorImageInfo hdr{m_interactiveViewport.postSampler,
                                  m_interactiveViewport.hdrColorImage.view,
                                  VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo depth{m_interactiveViewport.postSampler,
                                    m_interactiveViewport.depthImage.view,
                                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo history{m_interactiveViewport.taaHistorySampler,
                                      m_interactiveViewport.taaHistoryImage[1u - i].view,
                                      VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo out{VK_NULL_HANDLE,
                                  m_interactiveViewport.taaHistoryImage[i].view,
                                  VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo infos[4] = {hdr, depth, history, out};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t b = 0; b < 4; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = m_interactiveViewport.taaDescSet[i];
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = (b == 3) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[b].pImageInfo = &infos[b];
        }
        vkUpdateDescriptorSets(vkDevice, 4, writes, 0, nullptr);
    }
    return true;
}

void VulkanViewportBackend::destroyRasterTaaResources(bool keepPipeline) {
    if (!m_device) return;
    VkDevice vkDevice = m_device->getDevice();
    for (auto& img : m_interactiveViewport.taaHistoryImage) {
        if (img.image) { m_device->destroyImage(img); img = {}; }
    }
    m_taaFrameIndex = 0;
    m_taaHasPrevViewProj = false;
    if (keepPipeline) return;
    if (m_interactiveViewport.taaPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.taaPipeline, nullptr);
        m_interactiveViewport.taaPipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.taaPipelineLayout) {
        vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.taaPipelineLayout, nullptr);
        m_interactiveViewport.taaPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.taaDescPool) {
        vkDestroyDescriptorPool(vkDevice, m_interactiveViewport.taaDescPool, nullptr);
        m_interactiveViewport.taaDescPool = VK_NULL_HANDLE;
        m_interactiveViewport.taaDescSet[0] = VK_NULL_HANDLE;
        m_interactiveViewport.taaDescSet[1] = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.taaDescLayout) {
        vkDestroyDescriptorSetLayout(vkDevice, m_interactiveViewport.taaDescLayout, nullptr);
        m_interactiveViewport.taaDescLayout = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.taaHistorySampler) {
        vkDestroySampler(vkDevice, m_interactiveViewport.taaHistorySampler, nullptr);
        m_interactiveViewport.taaHistorySampler = VK_NULL_HANDLE;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Kayit
// ───────────────────────────────────────────────────────────────────────────
void VulkanViewportBackend::recordRasterTaaPass(VkCommandBuffer cmd,
                                                uint32_t width, uint32_t height,
                                                const Matrix4x4& invViewProjUnjittered) {
    if (!cmd || width == 0u || height == 0u) return;
    if (m_interactiveViewport.taaPipeline == VK_NULL_HANDLE) return;
    if (!m_interactiveViewport.taaHistoryImage[0].image) return;

    const auto started = std::chrono::steady_clock::now();
    const uint32_t parity = m_taaHistoryParity;
    VulkanRT::ImageHandle& dst = m_interactiveViewport.taaHistoryImage[parity];
    VulkanRT::ImageHandle& src = m_interactiveViewport.taaHistoryImage[1u - parity];

    // Derinlik: attachment -> ornekleme. recordRasterPostPass da ayni gecisi
    // yapiyor; iki kez yapmak zararsiz (ayni layout) ama TAA once kostugu icin
    // burada yapilmali, yoksa attachment layout'unda ornekleriz.
    taaImageBarrier(cmd, m_interactiveViewport.depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    taaImageBarrier(cmd, m_interactiveViewport.hdrColorImage.image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    taaImageBarrier(cmd, src.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    taaImageBarrier(cmd, dst.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    RasterTaaPush pc{};
    // ★★★★★ ANALİTİK DOUBLE-PRECISION REPROJEKSIYON
    //   Önceki fix (double ile çarpım, float32 giriş): EKSİKTİ.
    //   Neden: Matrix4x4::inverse() perspektif matrisine uygulanınca
    //   zFar/zNear koşul sayısı (~1e8) × float32_eps (~1e-7) ≈ 10 NDC hata
    //   üretir — double ile çarpmak bu hatayı KALDIRMAZ, çünkü girdi zaten
    //   yanlış. Kapalı sahnelerde koordinatlar küçük (~10 birim) olduğu için
    //   hata gizli kalır; geniş arazide kamera 100+ birimde olunca hata ~10 px.
    //
    //   Gerçek fix: inv(VP) analitik formülle ve double hassasiyetle üretilir.
    //   Rijit gövde matrisi inv(V) için — sadece rotasyon transpozu + kamera
    //   konumu — tam sonuç verir. inv(P) için kapalı formül kullanılır
    //   (perspektif/ortho her iki durum için ayrı). Bu sayede koşul sayısı
    //   amplifikasyonu ve kataştrofik iptale beraber çözüm sunulur.
    if (m_taaPrevDValid) {
        // ── Analitik inv(currentV): rijit gövde matrisi tam tersi ─────────
        // inv(V) = [R^T | eye], R^T = transpoz rotasyon, eye = kamera konumu.
        // eye[i] = -Σ_r V[r][i] * V[r][3]  (R^T * (-t) = e, t = çeviri)
        double invV[4][4] = {};
        const double (*v)[4] = m_taaCurViewD;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c)
                invV[r][c] = v[c][r];          // R^T
            invV[r][3] = 0.0;
            for (int k = 0; k < 3; ++k)
                invV[r][3] -= v[k][r] * v[k][3]; // kamera konumu = -R^T * t
        }
        invV[3][3] = 1.0;

        // ── Analitik inv(currentP): perspektif ve ortho için kapalı form ──
        // Perspektif: P[3][2] = -1, P[3][3] = 0
        //   inv(P)[0][0] = 1/P[0][0], [1][1] = 1/P[1][1],
        //   [2][3] = -1, [3][2] = 1/B, [3][3] = A/B
        // Ortho:   P[3][2] = 0, P[3][3] = 1
        //   inv(P)[0][0]=1/P[0][0], [1][1]=1/P[1][1],
        //   [2][2]=1/P[2][2], [2][3]=-P[2][3]/P[2][2], [3][3]=1
        double invP[4][4] = {};
        const double (*p)[4] = m_taaCurProjD;
        if (p[3][3] > 0.5) {
            // Ortho (P[3][3]=1)
            invP[0][0] = 1.0 / p[0][0];
            invP[1][1] = 1.0 / p[1][1];
            invP[2][2] = 1.0 / p[2][2];
            invP[2][3] = -p[2][3] / p[2][2];
            invP[3][3] = 1.0;
        } else {
            // Perspektif (P[3][2] = -1)
            invP[0][0] = 1.0 / p[0][0];   // aspect / f
            invP[1][1] = 1.0 / p[1][1];   // -1 / f
            invP[2][3] = -1.0;             // perspektif satırı
            invP[3][2] = 1.0 / p[2][3];   // 1 / B
            invP[3][3] = p[2][2] / p[2][3]; // A / B
        }

        // inv(currentVP) = inv(currentV) * inv(currentP)
        double invVP[4][4] = {};
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                for (int k = 0; k < 4; ++k)
                    invVP[r][c] += invV[r][k] * invP[k][c];

        // prevVP = prevP * prevV (double'da, float32 yuvarlama hatası yok)
        double prevVP[4][4] = {};
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                for (int k = 0; k < 4; ++k)
                    prevVP[r][c] += m_taaPrevProjD[r][k] * m_taaPrevViewD[k][c];

        // reprojection = prevVP * inv(currentVP), column-major GLSL mat4 için
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row) {
                double val = 0.0;
                for (int k = 0; k < 4; ++k) val += prevVP[row][k] * invVP[k][col];
                pc.reprojection[col * 4 + row] = static_cast<float>(val);
            }
    } else {
        // Henüz geçmiş yok: birim matris (historyValid=0 ile gelmez,
        // güvenli başlangıç değeri olarak ayarlanır).
        pc.reprojection[0]  = 1.0f;
        pc.reprojection[5]  = 1.0f;
        pc.reprojection[10] = 1.0f;
        pc.reprojection[15] = 1.0f;
    }

    pc.width = width;
    pc.height = height;
    // ★★★ blend = 1/(n+1): ilk kare TAM agirlikli, n. kare 1/(n+1). Bu bir
    //   "feedback katsayisi" degil gercek bir CALISAN ORTALAMA -- hareketsiz
    //   kamerada N kare sonra N ornekli bir tahmin verir, exponential bir
    //   karisimin asla ulasamayacagi bir varyans dususu. Hedefe varinca sabit
    //   kalir ki sahne degisimlerine hala tepki verebilsin.
    const uint32_t n = m_taaFrameIndex < m_taaTargetSamples ? m_taaFrameIndex
                                                            : m_taaTargetSamples;
    pc.blend = 1.0f / static_cast<float>(n + 1u);
    pc.historyValid = (m_taaFrameIndex > 0u && m_taaHasPrevViewProj && m_taaPrevDValid) ? 1u : 0u;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_interactiveViewport.taaPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_interactiveViewport.taaPipelineLayout, 0, 1,
                            &m_interactiveViewport.taaDescSet[parity], 0, nullptr);
    vkCmdPushConstants(cmd, m_interactiveViewport.taaPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (width + 7u) / 8u, (height + 7u) / 8u, 1u);

    // ── Cozulmus kare -> post'un okudugu HDR hedefi ─────────────────────────
    // ★★ BILINEN MALIYET, olculmek uzere birakildi: tam cozunurlukte bir
    //   RGBA16F kopyasi. Alternatifi post'un descriptor'ini TAA ciktisina
    //   cevirmekti; o zaman TAA'nin acik/kapali olmasi post'un descriptor
    //   durumunu degistirirdi ve "kapaliyken hangi goruntu okunuyor" sorusu
    //   iki yere dagilirdi. `viewport.taa` bu gecisin suresini raporluyor.
    taaImageBarrier(cmd, dst.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    taaImageBarrier(cmd, m_interactiveViewport.hdrColorImage.image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.dstSubresource = copy.srcSubresource;
    copy.extent = {width, height, 1};
    vkCmdCopyImage(cmd, dst.image, VK_IMAGE_LAYOUT_GENERAL,
                   m_interactiveViewport.hdrColorImage.image, VK_IMAGE_LAYOUT_GENERAL,
                   1, &copy);
    taaImageBarrier(cmd, m_interactiveViewport.hdrColorImage.image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    m_taaHistoryParity = 1u - parity;
    if (m_taaFrameIndex < m_taaTargetSamples) ++m_taaFrameIndex;
    m_taaLastMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

// ───────────────────────────────────────────────────────────────────────────
// Olcum
// ───────────────────────────────────────────────────────────────────────────
// ★★★★★ `enabled` ISTEK, `accumulated_samples` OLCUM. Hat dort ayri yerde
//   sessizce kapanabilir (shader yok / pipeline kurulamadi / gecmis tahsisi
//   basarisiz / mod raster degil) ve dordu de `enabled = true` ile birlikte
//   yasar. Tek bir bayrak bunlari ayirt edemezdi; sebep ADIYLA doner.
bool VulkanBackendAdapter::getRasterTaaStatus(RasterTaaStatus& out) const {
    out = {};
    out.enabled = m_taaEnabled;
    out.target_samples = m_taaTargetSamples;
    out.accumulated_samples = m_taaFrameIndex;
    out.last_ms = m_taaLastMs;
    out.supported = m_interactiveViewport.taaPipeline != VK_NULL_HANDLE &&
                    m_interactiveViewport.taaHistoryImage[0].image != VK_NULL_HANDLE;
    out.converged = out.supported && m_taaEnabled &&
                    m_taaFrameIndex >= m_taaTargetSamples;
    if (!out.supported) {
        out.inactive_reason = m_interactiveViewport.taaPipeline == VK_NULL_HANDLE
            ? "raster_taa.spv is missing or the compute pipeline could not be created"
            : "TAA history images are not allocated";
    } else if (!m_taaEnabled) {
        out.inactive_reason = "disabled by viewport.set_taa";
    } else if (m_viewportMode != ViewportMode::MaterialPreview &&
               m_viewportMode != ViewportMode::Solid &&
               m_viewportMode != ViewportMode::Matcap) {
        out.inactive_reason = "the raster viewport is not the mode on screen";
    }
    return true;
}

} // namespace Backend
