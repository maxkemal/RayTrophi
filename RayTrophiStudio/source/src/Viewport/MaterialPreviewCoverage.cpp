#include "Backend/VulkanBackend.h"
#include "Viewport/ViewportPipelineCache.h"
#include "Backend/VulkanViewportBackend.h"
#include "PerfProfile.h"
#include "globals.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace Backend {
void VulkanBackendAdapter::createMaterialPreviewCoveredPipeline(
    const VkGraphicsPipelineCreateInfo& base, const std::string& shaderDir) {
    if (!m_device || base.stageCount != 2 || !base.pDepthStencilState) return;
    const auto path = std::filesystem::path(shaderDir) / "material_preview_covered.spv";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { SCENE_LOG_WARN("[Coverage] material_preview_covered.spv missing; using regular material shading."); return; }
    const auto length = file.tellg();
    if (length <= 0 || static_cast<std::size_t>(length) % sizeof(uint32_t)) return;
    std::vector<uint32_t> words(static_cast<std::size_t>(length) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(length))) return;
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = words.size() * sizeof(uint32_t);
    shader.pCode = words.data();
    VkDevice device = m_device->getDevice();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &shader, nullptr, &module) != VK_SUCCESS) return;
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{base.pStages[0], base.pStages[1]};
    for (auto& stage : stages) if (stage.stage == VK_SHADER_STAGE_FRAGMENT_BIT) stage.module = module;
    auto depth = *base.pDepthStencilState;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_EQUAL;
    auto info = base;
    info.pStages = stages.data();
    info.pDepthStencilState = &depth;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, Backend::currentViewportPipelineCache(device),
                                  1, &info, nullptr, &pipeline) == VK_SUCCESS) {
        Backend::markViewportPipelineCacheDirty(device);
        m_interactiveViewport.materialPreviewCoveredPipeline = pipeline;
        SCENE_LOG_INFO("[Coverage] EQUAL / early-tests material pipeline ready.");
    } else {
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        SCENE_LOG_WARN("[Coverage] Covered pipeline creation failed; using regular material shading.");
    }
    vkDestroyShaderModule(device, module, nullptr);
}

void VulkanBackendAdapter::destroyMaterialPreviewCoveredPipeline() {
    if (!m_device) return;
    auto& pipeline = m_interactiveViewport.materialPreviewCoveredPipeline;
    if (pipeline) vkDestroyPipeline(m_device->getDevice(), pipeline, nullptr);
    pipeline = VK_NULL_HANDLE;
}

// ── Scene-feature specialization (PLAYER_ANDROID_PERF_PLAN O3) ──────────────
// Mirror of material_preview_frag.frag: constant_id 0 = RT_SPEC_TERRAIN,
// 1 = RT_SPEC_GRAPH, 2 = RT_SPEC_WEATHER, 3 = RT_SPEC_GLASS. A bit that is OFF removes a branch
// the driver would otherwise allocate registers/stack for on every fragment
// (malioc, Mali-G52: 1664 B -> 704 B stack, LS 988 -> 332 cycles, all off).
// ★★★ Correctness rule: a bit may be off ONLY when no fragment can take that
//   branch. Each test below is the shader's own gate, evaluated over every
//   bound material; when the CPU copy cannot prove it (external material
//   buffer, short cache) the bit stays on.
void VulkanViewportBackend::fillMaterialPreviewSpec(uint32_t mask, MaterialPreviewSpecStorage& out) {
    const uint32_t bits[kMpSpecCount] = {
        kMpSpecTerrain, kMpSpecGraph, kMpSpecWeather, kMpSpecGlass,
        kMpSpecExtLobes, kMpSpecAltLighting, kMpSpecFallbackSky, kMpSpecRayFusionGi,
        kMpSpecOptTex, kMpSpecClearcoat
    };
    for (uint32_t i = 0; i < kMpSpecCount; ++i) {
        out.values[i] = (mask & bits[i]) ? VK_TRUE : VK_FALSE;
        out.entries[i].constantID = i;
        out.entries[i].offset = static_cast<uint32_t>(i * sizeof(VkBool32));
        out.entries[i].size = sizeof(VkBool32);
    }
    out.info.mapEntryCount = kMpSpecCount;
    out.info.pMapEntries = out.entries;
    out.info.dataSize = sizeof(out.values);
    out.info.pData = out.values;
}

uint32_t VulkanViewportBackend::materialPreviewSceneFeatureMask() const {
    uint32_t mask = 0;
    const bool external = m_externalMaterialBuffer != VK_NULL_HANDLE;
    const uint32_t count = external ? m_externalMaterialCount
                                    : (m_device ? m_device->m_materialCount : 0u);
    // Shader: (mat.flags & FLAG_TERRAIN) for the fragment's material.
    if (external || m_cachedGpuMaterials.size() < count) {
        mask |= kMpSpecTerrain;
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            if (m_cachedGpuMaterials[i].flags & (1u << 16)) { mask |= kMpSpecTerrain; break; }
        }
    }
    // Shader: matProgramOffset(materialIndex) != MATPROG_NONE. Same table the
    // covered-pipeline proof reads; an unparsable stream answers "has program".
    if (external) {
        mask |= kMpSpecGraph;
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            if (m_rasterMaterialPrograms.hasProgram(i)) { mask |= kMpSpecGraph; break; }
        }
    }
    // Shader: sceneFlags bit 64 (weather surface, cloud shadow, lightning).
    // The globals writer sets it from these two plus a descriptor-set match,
    // so this is a superset of the bit.
    if (m_cloudShadowBound && m_cloudRtRendering) mask |= kMpSpecWeather;
    // Shader: glassPath / transmissiveFragment / bubbleSurface / interiorDepth.
    // With the bit off the shader FORCES transmission = 0, bubble = false and
    // Interior Depth = 0, so the proof must show each is already exactly that
    // for every bound material:
    //   transmission scalar <= 0 (the glass gate is `> 0.0`, NOT the 0.001
    //     replay threshold), no transmission map, no graph (it may write
    //     transmission or opacity), no bubble flag, no legacy partial opacity
    //     (opacity map or scalar < 0.99 on a non-cutout material becomes
    //     transmission), and transmission_density <= 0.
    // Negated comparisons keep NaN on the "glass" side.
    if (external || m_cachedGpuMaterials.size() < count) {
        mask |= kMpSpecGlass | kMpSpecExtLobes | kMpSpecOptTex | kMpSpecClearcoat;
    } else if (count > 0u) {
        auto materialNeedsGlass = [&](uint32_t i) -> bool {
            const auto& m = m_cachedGpuMaterials[i];
            const bool cutout = (m.flags & MATERIAL_FLAGS_PREVIEW_CUTOUT) != 0u;
            return !(m.transmission <= 0.0f) || m.transmission_tex != 0u ||
                   m_rasterMaterialPrograms.hasProgram(i) ||
                   (m.flags & (1u << 19u)) != 0u ||
                   (!cutout && (m.opacity_tex != 0u || !(m.opacity >= 0.99f))) ||
                   !(m.transmission_density <= 0.0f);
        };
        auto materialNeedsExtLobes = [&](uint32_t i) -> bool {
            const auto& m = m_cachedGpuMaterials[i];
            return !(m.sheen <= 0.0f) ||
                   !(m.subsurface_amount <= 0.0f) ||
                   !(m.translucent <= 0.0f) ||
                   !(m.micro_detail_strength <= 0.0f) ||
                   !(m.tile_break_strength <= 0.0f) ||
                   m_rasterMaterialPrograms.hasProgram(i);
        };
        auto materialNeedsClearcoat = [&](uint32_t i) -> bool {
            const auto& m = m_cachedGpuMaterials[i];
            return !(m.clearcoat <= 0.001f) || m_rasterMaterialPrograms.hasProgram(i);
        };
        auto materialNeedsOptTex = [&](uint32_t i) -> bool {
            const auto& m = m_cachedGpuMaterials[i];
            return m.metallic_tex != 0u || m.specular_tex != 0u ||
                   m.emission_tex != 0u || m_rasterMaterialPrograms.hasProgram(i);
        };
        constexpr uint32_t kAllMatBits =
            kMpSpecGlass | kMpSpecExtLobes | kMpSpecOptTex | kMpSpecClearcoat;
        if (m_rasterMeshes.empty()) {
            for (uint32_t i = 0; i < count; ++i) {
                if (materialNeedsGlass(i)) mask |= kMpSpecGlass;
                if (materialNeedsExtLobes(i)) mask |= kMpSpecExtLobes;
                if (materialNeedsClearcoat(i)) mask |= kMpSpecClearcoat;
                if (materialNeedsOptTex(i)) mask |= kMpSpecOptTex;
                if ((mask & kAllMatBits) == kAllMatBits) {
                    break;
                }
            }
        } else {
            // Check only materials referenced by raster meshes so orphan/unused
            // materials in the table do not force optional branches on Mali-G52.
            for (const auto& [key, mesh] : m_rasterMeshes) {
                (void)key;
                if (mesh.vertexCount == 0u) continue;
                if (mesh.cpuMatIds.size() < mesh.vertexCount) {
                    mask |= kAllMatBits;
                    break;
                }
                for (uint32_t encoded : mesh.materialUsage.ids(mesh.cpuMatIds)) {
                    if ((encoded & 0x80000000u) != 0u) continue;
                    const uint32_t id = (std::min)(encoded & 0x7fffffffu, count - 1u);
                    if (materialNeedsGlass(id)) mask |= kMpSpecGlass;
                    if (materialNeedsExtLobes(id)) mask |= kMpSpecExtLobes;
                    if (materialNeedsClearcoat(id)) mask |= kMpSpecClearcoat;
                    if (materialNeedsOptTex(id)) mask |= kMpSpecOptTex;
                    if ((mask & kAllMatBits) == kAllMatBits) {
                        break;
                    }
                }
                if ((mask & kAllMatBits) == kAllMatBits) {
                    break;
                }
            }
        }
    }
    // Alt lighting rigs (three-point, studio, outdoor): off when in Scene lighting (3).
    if (static_cast<uint32_t>(::render_settings.material_preview_lighting_preset) != 3u) {
        mask |= kMpSpecAltLighting;
    }
    // Unfiltered sky fallback: off once prefiltered IBL is ready for the active world.
    MaterialPreviewIblStatus iblStatus{};
    if (!getMaterialPreviewIblStatus(iblStatus) || !iblStatus.ready) {
        mask |= kMpSpecFallbackSky;
    }
    // RayFusion GI / probe visibility / RT screen shadow / volume deep shadow:
    // off only when the device has no ray query AND the scene has no volumes.
    const bool hasRayQuery = m_device && m_device->getCapabilities().supportsRayQuery;
    const bool hasVolumes = m_device && m_device->m_volumeCount > 0u;
    if (hasRayQuery || hasVolumes) {
        mask |= kMpSpecRayFusionGi;
    }
    return mask;
}

struct VulkanViewportBackend::MaterialPreviewSpecRecipe {
    std::string shaderDir;
    VkPipeline pipeline = VK_NULL_HANDLE;   // the main pipeline this recipe built/captured
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    VkPipelineViewportStateCreateInfo viewport{};
    VkPipelineRasterizationStateCreateInfo rasterization{};
    VkPipelineMultisampleStateCreateInfo multisample{};
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments;
    VkPipelineDynamicStateCreateInfo dynamic{};
    std::vector<VkDynamicState> dynamicStates;
    VkGraphicsPipelineCreateInfo info{};

    void link() {
        vertexInput.pVertexBindingDescriptions = bindings.data();
        vertexInput.pVertexAttributeDescriptions = attributes.data();
        colorBlend.pAttachments = blendAttachments.data();
        dynamic.pDynamicStates = dynamicStates.data();
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &inputAssembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &rasterization;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depthStencil;
        info.pColorBlendState = &colorBlend;
        info.pDynamicState = &dynamic;
        info.pTessellationState = nullptr;
        info.pStages = nullptr;
        info.stageCount = 0;
    }
};

void VulkanViewportBackend::captureMaterialPreviewSpecRecipe(
    const VkGraphicsPipelineCreateInfo& base, const std::string& shaderDir, uint32_t mask) {
    m_mpSpecRecipe.reset();
    m_mpSpecMask = mask;
    m_mpSpecShrinkFrames = 0;
    // Deep copy of the state blocks the material pipeline is built from. Any
    // extension chain or pointer this copy does not own disables the rebuild
    // path (the pipeline then keeps the mask it was born with, which is
    // correct only until a new feature appears -- so it falls back to ALL).
    const bool plain = base.stageCount == 2 && base.pNext == nullptr &&
        base.pVertexInputState && !base.pVertexInputState->pNext &&
        base.pInputAssemblyState && !base.pInputAssemblyState->pNext &&
        base.pViewportState && !base.pViewportState->pNext &&
        !base.pViewportState->pViewports && !base.pViewportState->pScissors &&
        base.pRasterizationState && !base.pRasterizationState->pNext &&
        base.pMultisampleState && !base.pMultisampleState->pNext &&
        !base.pMultisampleState->pSampleMask &&
        base.pDepthStencilState && !base.pDepthStencilState->pNext &&
        base.pColorBlendState && !base.pColorBlendState->pNext &&
        base.pDynamicState && !base.pDynamicState->pNext &&
        !base.pTessellationState;
    if (!plain) {
        m_mpSpecFailed = true;
        SCENE_LOG_WARN("[MaterialSpec] Pipeline state not capturable; feature specialization off.");
        return;
    }
    auto recipe = std::make_shared<MaterialPreviewSpecRecipe>();
    recipe->shaderDir = shaderDir;
    recipe->pipeline = m_interactiveViewport.materialPreviewPipeline;
    recipe->vertexInput = *base.pVertexInputState;
    recipe->bindings.assign(base.pVertexInputState->pVertexBindingDescriptions,
        base.pVertexInputState->pVertexBindingDescriptions +
            base.pVertexInputState->vertexBindingDescriptionCount);
    recipe->attributes.assign(base.pVertexInputState->pVertexAttributeDescriptions,
        base.pVertexInputState->pVertexAttributeDescriptions +
            base.pVertexInputState->vertexAttributeDescriptionCount);
    recipe->inputAssembly = *base.pInputAssemblyState;
    recipe->viewport = *base.pViewportState;
    recipe->rasterization = *base.pRasterizationState;
    recipe->multisample = *base.pMultisampleState;
    recipe->depthStencil = *base.pDepthStencilState;
    recipe->colorBlend = *base.pColorBlendState;
    recipe->blendAttachments.assign(base.pColorBlendState->pAttachments,
        base.pColorBlendState->pAttachments + base.pColorBlendState->attachmentCount);
    recipe->dynamic = *base.pDynamicState;
    recipe->dynamicStates.assign(base.pDynamicState->pDynamicStates,
        base.pDynamicState->pDynamicStates + base.pDynamicState->dynamicStateCount);
    recipe->info = base;
    recipe->link();
    m_mpSpecRecipe = std::move(recipe);
    m_mpSpecFailed = false;
    SCENE_LOG_INFO("[MaterialSpec] material pipeline built with terrain=" +
                   std::to_string((mask & kMpSpecTerrain) != 0) +
                   " graph=" + std::to_string((mask & kMpSpecGraph) != 0) +
                   " weather=" + std::to_string((mask & kMpSpecWeather) != 0) +
                   " glass=" + std::to_string((mask & kMpSpecGlass) != 0) +
                   " ext=" + std::to_string((mask & kMpSpecExtLobes) != 0) +
                   " alt_light=" + std::to_string((mask & kMpSpecAltLighting) != 0) +
                   " fb_sky=" + std::to_string((mask & kMpSpecFallbackSky) != 0) +
                   " rf_gi=" + std::to_string((mask & kMpSpecRayFusionGi) != 0) +
                   " opt_tex=" + std::to_string((mask & kMpSpecOptTex) != 0) +
                   " cc=" + std::to_string((mask & kMpSpecClearcoat) != 0));
}

namespace {
bool readSpirv(const std::filesystem::path& path, std::vector<uint32_t>& words) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto length = file.tellg();
    if (length <= 0 || static_cast<std::size_t>(length) % sizeof(uint32_t)) return false;
    words.resize(static_cast<std::size_t>(length) / sizeof(uint32_t));
    file.seekg(0);
    return static_cast<bool>(
        file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(length)));
}

std::string specMaskName(uint32_t mask) {
    std::string name;
    if (mask & Backend::VulkanViewportBackend::kMpSpecTerrain) name += "terrain+";
    if (mask & Backend::VulkanViewportBackend::kMpSpecGraph) name += "graph+";
    if (mask & Backend::VulkanViewportBackend::kMpSpecWeather) name += "weather+";
    if (mask & Backend::VulkanViewportBackend::kMpSpecGlass) name += "glass+";
    if (mask & Backend::VulkanViewportBackend::kMpSpecExtLobes) name += "ext+";
    if (mask & Backend::VulkanViewportBackend::kMpSpecAltLighting) name += "alt_light+";
    if (mask & Backend::VulkanViewportBackend::kMpSpecFallbackSky) name += "fb_sky+";
    if (mask & Backend::VulkanViewportBackend::kMpSpecRayFusionGi) name += "rf_gi+";
    if (mask & Backend::VulkanViewportBackend::kMpSpecOptTex) name += "opt_tex+";
    if (mask & Backend::VulkanViewportBackend::kMpSpecClearcoat) name += "cc+";
    if (name.empty()) return "none";
    name.pop_back();
    return name;
}
} // namespace

void VulkanViewportBackend::serviceMaterialPreviewFeatureSpec() {
    auto& iv = m_interactiveViewport;
    if (!m_mpSpecRecipe || m_mpSpecFailed || !m_device) return;
    // A teardown/recreate (resize with !keepPipeline, device rebuild) replaced
    // the pipeline: the captured layout/render pass are no longer ours. The
    // next init captures a fresh recipe.
    if (iv.materialPreviewPipeline == VK_NULL_HANDLE ||
        iv.materialPreviewPipeline != m_mpSpecRecipe->pipeline) {
        m_mpSpecRecipe.reset();
        return;
    }
    const uint32_t want = materialPreviewSceneFeatureMask();
    uint32_t next = m_mpSpecMask;
    if (want & ~m_mpSpecMask) {
        // A branch the live pipeline lacks is reachable NOW: rebuild before
        // this frame records, never draw it with the old pipeline.
        next = m_mpSpecMask | want;
        m_mpSpecShrinkFrames = 0;
    } else if (want != m_mpSpecMask) {
        // Dropping a feature only saves time; wait until the scene settles so
        // a slider crossing zero does not recompile every frame. Exception:
        // initial IBL readiness (kMpSpecFallbackSky) is a one-time startup
        // transition and should take effect immediately rather than waiting 240 frames.
        const uint32_t dropped = m_mpSpecMask & ~want;
        const uint32_t shrinkDelay =
            (m_mpSpecRebuilds == 0u && dropped == kMpSpecFallbackSky) ? 0u : 240u;
        if (++m_mpSpecShrinkFrames < shrinkDelay) return;
        next = want;
        m_mpSpecShrinkFrames = 0;
    } else {
        m_mpSpecShrinkFrames = 0;
        return;
    }

    const auto started = std::chrono::steady_clock::now();
    MaterialPreviewSpecRecipe& recipe = *m_mpSpecRecipe;
    const std::filesystem::path dir(recipe.shaderDir);
    std::vector<uint32_t> vertWords, fragWords;
    VkDevice device = m_device->getDevice();
    if (!readSpirv(dir / "material_preview.spv", vertWords) ||
        !readSpirv(dir / "material_preview_frag.spv", fragWords)) {
        m_mpSpecFailed = true;
        SCENE_LOG_ERROR("[MaterialSpec] material_preview SPIR-V unreadable; cannot rebuild for " +
                        specMaskName(next) + ". Material shading may miss a feature.");
        return;
    }
    auto makeModule = [&](const std::vector<uint32_t>& words) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = words.size() * sizeof(uint32_t);
        ci.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS) return VkShaderModule(VK_NULL_HANDLE);
        return module;
    };
    VkShaderModule vert = makeModule(vertWords);
    VkShaderModule frag = makeModule(fragWords);

    auto build = [&](uint32_t mask, VkPipeline& out) -> VkResult {
        MaterialPreviewSpecStorage spec{};
        fillMaterialPreviewSpec(mask, spec);
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";
        stages[1].pSpecializationInfo = &spec.info;
        VkGraphicsPipelineCreateInfo info = recipe.info;
        info.stageCount = 2;
        info.pStages = stages;
        const VkResult result = vkCreateGraphicsPipelines(
            device, Backend::currentViewportPipelineCache(device), 1, &info, nullptr, &out);
        if (result == VK_SUCCESS) {
            Backend::markViewportPipelineCacheDirty(device);
            // The in-flight frames still read the old pipelines.
            drainInteractiveViewportInFlight();
            if (iv.materialPreviewPipeline) vkDestroyPipeline(device, iv.materialPreviewPipeline, nullptr);
            destroyMaterialPreviewCoveredPipeline();
            iv.materialPreviewPipeline = out;
            recipe.pipeline = out;
            // Copies stages[1] (and with it the specialization) into the
            // EQUAL / early-tests variant.
            createMaterialPreviewCoveredPipeline(info, recipe.shaderDir);
            Backend::saveViewportPipelineCache(device);
        }
        return result;
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = (vert && frag) ? build(next, pipeline) : VK_ERROR_INITIALIZATION_FAILED;
    if (result != VK_SUCCESS && next != kMpSpecAll && vert && frag) {
        // The full shader is what the session ran before O3; prefer it over a
        // pipeline that lacks a reachable branch.
        next = kMpSpecAll;
        result = build(next, pipeline);
    }
    if (vert) vkDestroyShaderModule(device, vert, nullptr);
    if (frag) vkDestroyShaderModule(device, frag, nullptr);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (result != VK_SUCCESS) {
        m_mpSpecFailed = true;
        SCENE_LOG_ERROR("[MaterialSpec] Rebuild failed (VkResult " +
                        std::to_string(static_cast<int>(result)) +
                        "); material shading keeps " + specMaskName(m_mpSpecMask) +
                        " and may miss " + specMaskName(want & ~m_mpSpecMask) + ".");
        return;
    }
    m_mpSpecMask = next;
    ++m_mpSpecRebuilds;
    iv.dirty = true;
    m_currentSamples = 0;
    // IPC surface until the frame-timing `applied` block carries it (that is a
    // shared-header change, held for the next full rebuild): perf.list shows
    // each rebuild's cost, and the entry with the highest `seq` among
    // "viewport.material_spec.mask.*" is the live mask.
    rtperf::recordFast("viewport.material_spec.rebuild", ms);
    rtperf::recordFast("viewport.material_spec.mask." + specMaskName(next), 0.0);
    SCENE_LOG_INFO("[MaterialSpec] material pipeline rebuilt for " + specMaskName(next) +
                   " in " + std::to_string(static_cast<int>(ms)) + " ms (rebuild #" +
                   std::to_string(m_mpSpecRebuilds) + ")");
}

RasterAppliedState VulkanViewportBackend::sampleRasterAppliedState() const {
    RasterAppliedState out = VulkanBackendAdapter::sampleRasterAppliedState();
    const bool active = m_mpSpecRecipe != nullptr && !m_mpSpecFailed;
    out.material_spec_active = active;
    const uint32_t mask = active ? m_mpSpecMask : kMpSpecAll;
    out.material_spec_terrain = (mask & kMpSpecTerrain) != 0;
    out.material_spec_graph = (mask & kMpSpecGraph) != 0;
    out.material_spec_weather = (mask & kMpSpecWeather) != 0;
    out.material_spec_glass = (mask & kMpSpecGlass) != 0;
    out.material_spec_rebuilds = m_mpSpecRebuilds;
    return out;
}

VkPipeline VulkanBackendAdapter::materialPreviewShadingPipeline(
    const RasterMeshBuffer& mesh, bool depthPrepassActive) const {
    if (depthPrepassActive && m_interactiveViewport.materialPreviewCoveredPipeline &&
        rasterMeshHasExactCoverage(mesh.materialUsage, mesh.cpuMatIds, mesh.vertexCount,
            m_cachedGpuMaterials, m_interactiveViewport.materialPreviewBoundMaterialCount,
            m_rasterMaterialPrograms, m_interactiveViewport.materialPreviewUsesExternalMaterials))
        return m_interactiveViewport.materialPreviewCoveredPipeline;
    return m_interactiveViewport.materialPreviewPipeline;
}
} // namespace Backend
