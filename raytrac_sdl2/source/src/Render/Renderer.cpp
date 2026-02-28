#include "renderer.h"
#include <SDL_image.h>
#include <filesystem>
#include <chrono>      // For wall-clock deltaTime in animation fallback
#include <execution>
#include <cstring>      // std::memcpy for camera hash
#include <imgui.h>
#include <imgui_impl_sdlrenderer2.h>
#include <scene_ui.h>
#include "OptixWrapper.h"
#include "Backend/IBackend.h"
#include "Backend/OptixBackend.h"
#include <future>
#include <thread>
#include <functional>
#include <vector_types.h>  // CUDA float4, float3 types for hair GPU upload
#include "Hair/HairBSDF.h"


// Includes moved from renderer.h
#include "Camera.h"
#include "light.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "AreaLight.h"
#include "SpotLight.h"
#include "Volumetric.h"
#include "PrincipledBSDF.h"
#include "Dielectric.h"
#include "Material.h"
#include "VDBVolume.h"
#include "VDBVolumeManager.h"
#include "VolumeShader.h"
#include "Triangle.h"
#include "Mesh.h"
#include "AABB.h"
#include "Ray.h"
#include "Hittable.h"
#include "HittableList.h"
#include "ParallelBVHNode.h"
#include "AnimatedObject.h"
#include "AnimationController.h"
#include "AnimationNodes.h"
#include "scene_data.h"

// Unified rendering system for CPU/GPU parity
#include "unified_types.h"
#include "unified_brdf.h"
#include "unified_light_sampling.h"
#include "unified_converters.h"
#include "MaterialManager.h"
#include "CameraPresets.h"
#include "TerrainManager.h"
#include "WaterSystem.h"      // For water/FFT keyframe animation
#include "InstanceManager.h"  // For wind animation in render_Animation
#include "water_shaders_cpu.h"  // CPU water shader functions
#include "HittableInstance.h"
#include "VolumetricRenderer.h"
#include "AtmosphereLUT.h"       // Required for CPU transmittance sampling
#include "Hair/HairBSDF.h"       // Hair BSDF for shading
#include <Backend/VulkanBackend.h>
bool Renderer::isCudaAvailable() {
    try {
        oidn::DeviceRef testDevice = oidn::newDevice(oidn::DeviceType::CUDA);
        testDevice.commit();
        return true; // CUDA destekleniyor
    }
    catch (const std::exception& e) {
        return false; // CUDA desteklenmiyor
    }
}
// Helper to initialize OIDN device once
void Renderer::initOIDN() {
    if (oidnInitialized) return;

    if (g_hasOptix) {
        try {
            oidnDevice = oidn::newDevice(oidn::DeviceType::CUDA);
            oidnDevice.commit();
            oidnInitialized = true;
            SCENE_LOG_INFO("[OIDN] Initialized with CUDA.");
            return;
        }
        catch (const std::exception& e) {
            SCENE_LOG_WARN(std::string("[OIDN] CUDA initialization failed, falling back to CPU: ") + e.what());
        }
    }

    try {
        oidnDevice = oidn::newDevice(oidn::DeviceType::CPU);
        oidnDevice.commit();
        oidnInitialized = true;
        SCENE_LOG_INFO("[OIDN] Initialized with CPU.");
    }
    catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("[OIDN] CPU initialization failed: ") + e.what());
    }
}

void Renderer::applyOIDNDenoising(SDL_Surface* surface, int numThreads, bool denoise, float blend) {
    if (!surface) return;
    std::lock_guard<std::mutex> lock(oidnMutex);

    if (!oidnInitialized) {
        initOIDN();
        if (!oidnInitialized) return;
    }

    int width = surface->w;
    int height = surface->h;
    size_t pixelCount = (size_t)width * height;
    size_t bufferSize = pixelCount * 3;  // Float3

    bool sizeChanged = (width != oidnCachedWidth || height != oidnCachedHeight);
    if (sizeChanged) {
        oidnColorData.resize(bufferSize);
        oidnOriginalData.resize(bufferSize);

        try {
            // Buffer'ları device'a göre oluştur
            // CUDA device'da newBuffer → GPU memory
            // CPU device'da newBuffer  → CPU memory
            oidnColorBuffer = oidnDevice.newBuffer(bufferSize * sizeof(float));
            oidnOutputBuffer = oidnDevice.newBuffer(bufferSize * sizeof(float));

            oidnFilter = oidnDevice.newFilter("RT");
            oidnFilter.setImage("color", oidnColorBuffer,
                oidn::Format::Float3, width, height);
            oidnFilter.setImage("output", oidnOutputBuffer,
                oidn::Format::Float3, width, height);

            // HDR: Vulkan'dan gelen veri linear float — hdr:true olmalı
            // srgb: false — tonemap/gamma SDL'e yazarken biz yapıyoruz
            oidnFilter.set("hdr", true);
            oidnFilter.set("srgb", false);
            oidnFilter.commit();

            oidnCachedWidth = width;
            oidnCachedHeight = height;
        }
        catch (const std::exception& e) {
            SCENE_LOG_ERROR(std::string("[OIDN] Buffer creation failed: ") + e.what());
            return;
        }
    }

    // ===========================================================================
    // SDL surface → OIDN input
    // Eğer Vulkan buffer'ı direkt expose edilebilirse bu kısım tamamen kalkar
    // Şimdilik: CPU'da float dönüşümü → buffer write
    // ===========================================================================
    const SDL_PixelFormat* fmt = surface->format;
    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    const float inv255 = 1.0f / 255.0f;
    const float blend_inv = 1.0f - blend;

    // Pixel okuma + linear'e çevir
    // sRGB → linear: OIDN HDR mode linear input bekliyor
    // Eğer SDL surface zaten gamma corrected (8bit sRGB) ise düzeltmemiz lazım
    for (size_t i = 0; i < pixelCount; ++i) {
        Uint32 pixel = pixels[i];
        float r = ((pixel & fmt->Rmask) >> fmt->Rshift) * inv255;
        float g = ((pixel & fmt->Gmask) >> fmt->Gshift) * inv255;
        float b = ((pixel & fmt->Bmask) >> fmt->Bshift) * inv255;

        // sRGB → linear (OIDN HDR mode için gerekli)
        // Yaklaşık: pow(x, 2.2) yerine hızlı versiyon
        r = r * r;
        g = g * g;
        b = b * b;

        size_t idx = i * 3;
        oidnColorData[idx] = r;
        oidnColorData[idx + 1] = g;
        oidnColorData[idx + 2] = b;
        oidnOriginalData[idx] = r;
        oidnOriginalData[idx + 1] = g;
        oidnOriginalData[idx + 2] = b;
    }

    try {
        oidnColorBuffer.write(0, bufferSize * sizeof(float), oidnColorData.data());
        oidnFilter.execute();

        const char* errMsg;
        if (oidnDevice.getError(errMsg) != oidn::Error::None) {
            SCENE_LOG_ERROR(std::string("[OIDN] ") + errMsg);
            return;
        }

        oidnOutputBuffer.read(0, bufferSize * sizeof(float), oidnColorData.data());
    }
    catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("[OIDN] Execution failed: ") + e.what());
        return;
    }

    // ===========================================================================
    // Blend + write back — linear → sRGB + tonemap
    // ===========================================================================
    for (size_t i = 0; i < pixelCount; ++i) {
        size_t idx = i * 3;

        float r = oidnColorData[idx];
        float g = oidnColorData[idx + 1];
        float b = oidnColorData[idx + 2];

        // Blend (düşük sample'larda denoised ile original karıştır)
        if (blend < 0.999f) {
            r = r * blend + oidnOriginalData[idx] * blend_inv;
            g = g * blend + oidnOriginalData[idx + 1] * blend_inv;
            b = b * blend + oidnOriginalData[idx + 2] * blend_inv;
        }

        // Linear → sRGB (sqrt yaklaşımı, pow(x, 1/2.2) yerine)
        r = sqrtf(std::max(r, 0.0f));
        g = sqrtf(std::max(g, 0.0f));
        b = sqrtf(std::max(b, 0.0f));

        Uint8 ri = static_cast<Uint8>(std::min(r, 1.0f) * 255.0f + 0.5f);
        Uint8 gi = static_cast<Uint8>(std::min(g, 1.0f) * 255.0f + 0.5f);
        Uint8 bi = static_cast<Uint8>(std::min(b, 1.0f) * 255.0f + 0.5f);

        Uint32 alpha = pixels[i] & fmt->Amask;
        pixels[i] = alpha
            | ((Uint32)ri << fmt->Rshift)
            | ((Uint32)gi << fmt->Gshift)
            | ((Uint32)bi << fmt->Bshift);
    }
}
Renderer::Renderer(int image_width, int image_height, int samples_per_pixel, int max_depth)
    : image_width(image_width), image_height(image_height), aspect_ratio(static_cast<double>(image_width) / image_height), halton_cache(new float[MAX_DIMENSIONS * MAX_SAMPLES_HALTON]), color_processor(image_width, image_height)
{
    initialize_halton_cache();

    frame_buffer.resize(image_width * image_height);
    sample_counts.resize(image_width * image_height, 0);
    max_halton_index = MAX_SAMPLES_HALTON - 1; // Halton dizisi i�in maksimum indeks

    // Adaptive sampling i�in bufferlar
    variance_buffer.resize(image_width * image_height, 0.0f);

    rendering_complete = false;

    variance_map.resize(image_width * image_height, 0.0f);


}
void Renderer::resetResolution(int w, int h) {
    image_width = w;
    image_height = h;
    aspect_ratio = static_cast<double>(image_width) / image_height;

    const size_t pixel_count = w * h;

    // Buffers resize
    frame_buffer.resize(pixel_count);
    variance_buffer.resize(pixel_count, 0.0f);  // reset variance
    sample_counts.resize(pixel_count, 0);       // reset counts
    variance_map.resize(pixel_count, 0.0f);     // optional if used in display

    // Optional: zero the actual frame buffer content
    std::fill(frame_buffer.begin(), frame_buffer.end(), Vec3(0.0f));

    // OIDN cache invalidate - buffer'lar bir sonraki denoise'da yeniden olu�turulacak
    oidnCachedWidth = 0;
    oidnCachedHeight = 0;

    // Pixel list cache invalidate - resolution changed
    cpu_pixel_list_valid = false;
}


Renderer::~Renderer()
{
    frame_buffer.clear();
    sample_counts.clear();
    variance_map.clear();
}


bool Renderer::SaveSurface(SDL_Surface* surface, const char* file_path)
{

    // Ayn� isimde dosya varsa silmeye �al�� (zorla yazma)
    if (std::filesystem::exists(file_path)) {
        std::error_code ec;
        std::filesystem::remove(file_path, ec);
        if (ec) {
            SDL_Log("Dosya silinemiyor. Ba�ka bir i�lem taraf�ndan kullan�l�yor olabilir.");
            return false;
        }
    }

    SDL_Surface* surface_to_save =
        SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);

    if (!surface_to_save) {
        SDL_Log("Surface format conversion failed: %s", SDL_GetError());
        return false;
    }


    int result = IMG_SavePNG(surface_to_save, file_path);
    SDL_FreeSurface(surface_to_save);

    if (result != 0) {
        SDL_Log("Failed to save image: %s", IMG_GetError());
        return false;
    }

    return true;
}

Vec3 Renderer::getColorFromSurface(SDL_Surface* surface, int i, int j) {
    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    Uint32 pixel = pixels[(surface->h - 1 - j) * surface->pitch / 4 + i];

    Uint8 r, g, b;
    SDL_GetRGB(pixel, surface->format, &r, &g, &b);

    // sRGB to linear d�n���m� istersen buraya koy
    return Vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}

static int point_light_pick_count = 0;
static int directional_pick_count = 0;

// ============================================================================
// NEW ANIMATION SYSTEM INTEGRATION
// ============================================================================

void Renderer::initializeAnimationSystem(SceneData& scene) {
    // Initialize per-model animators
    for (auto& ctx : scene.importedModelContexts) {
        if (ctx.hasAnimation && !ctx.animator) {
            ctx.animator = std::make_shared<AnimationController>();

            // Filter clips for this model
            std::vector<std::shared_ptr<AnimationData>> modelClips;
            for (auto& anim : scene.animationDataList) {
                if (anim && (anim->modelName == ctx.importName || (anim->modelName.empty() && scene.importedModelContexts.size() == 1))) {
                    modelClips.push_back(anim);
                }
            }

            ctx.animator->registerClips(modelClips);

            if (!modelClips.empty()) {
                // [FIX] Do NOT auto-play on import. 
                // Keep character in Bind Pose (T-Pose) until an animation node is added to AnimGraph.
                // ctx.animator->play(modelClips[0]->name, 0.0f);
                SCENE_LOG_INFO("[Renderer] Created animator for model: " + ctx.importName + " (Clips: " + std::to_string(modelClips.size()) + ")");
            }
        }
    }
}

bool Renderer::updateAnimationWithGraph(SceneData& scene, float deltaTime, bool apply_cpu_skinning) {
    bool anyChanged = false;

    // Resize internal matrix buffer to match scene total bone count
    // (kept for global access, e.g. GPU upload)
    // IMPORTANT: Use resize(), NOT assign()! When a second model is imported,
    // totalBones increases. assign() would destroy Model A's existing bone matrices,
    // causing its skinned mesh to collapse to origin on the next GPU skinning pass
    // (before Model A's animator re-computes). resize() preserves existing entries.
    size_t totalBones = scene.boneData.boneNameToIndex.size();
    if (this->finalBoneMatrices.size() < totalBones) {
        this->finalBoneMatrices.resize(totalBones, Matrix4x4::identity());
    }
    // Optimization: avoid resizing every frame, but ensure identity for non-animated models
    // Actually, AnimationController::update already fills with identity.

    // Update each model context independently
    for (auto& ctx : scene.importedModelContexts) {
        // Per-model bone matrices for isolated skinning
        std::vector<Matrix4x4> modelBoneMatrices;
        bool modelChanged = false;

        if (ctx.useAnimGraph && ctx.graph) {
            // EVALUATE NODE GRAPH (Unity/Unreal style)
            if (ctx.animator) {
                ctx.graph->evalContext.clipsPtr = &ctx.animator->getAllClips();
            }

            // CRITICAL FIX: Fetch robust global inverse transform from boneData
            // ImportedModelContext's copy might be uninitialized identity
            Matrix4x4 finalInv = scene.boneData.globalInverseTransform;
            if (!ctx.importName.empty()) {
                 auto invIt = scene.boneData.perModelInverses.find(ctx.importName);
                 if (invIt != scene.boneData.perModelInverses.end()) finalInv = invIt->second;
            }
            ctx.globalInverseTransform = finalInv; // Update cache
            ctx.graph->evalContext.globalInverseTransform = finalInv;
            
            // --- ROOT MOTION PREPARATION FOR ANIM GRAPH ---
            ctx.graph->evalContext.useRootMotion = ctx.useRootMotion;
            if (ctx.useRootMotion && ctx.animator) {
                auto clips = ctx.animator->getAllClips();
                if (!clips.empty() && clips[0].sourceData) {
                    ctx.graph->evalContext.rootMotionBone = ctx.animator->findBestRootMotionBone(clips[0].name);
                }
            }
            ctx.graph->evalContext.rootMotion = RootMotionDelta(); // reset

            AnimationGraph::PoseData pose = ctx.graph->evaluate(deltaTime, scene.boneData);

            if (pose.isValid()) {
                if (pose.wasUpdated) {
                    modelChanged = true;
                    anyChanged = true; 
                    
                    // --- APPLY ROOT MOTION FOR ANIM GRAPH ---
                    if (ctx.useRootMotion && pose.rootMotion.hasPosition && !ctx.members.empty()) {
                        std::vector<Transform*> processed;
                        for (auto& member : ctx.members) {
                            if (auto tri = std::dynamic_pointer_cast<Triangle>(member)) {
                                auto h = tri->getTransformHandle();
                                if (h && std::find(processed.begin(), processed.end(), h.get()) == processed.end()) {
                                    h->position = h->position + pose.rootMotion.positionDelta;
                                    h->updateMatrix();
                                    h->markDirty();
                                    processed.push_back(h.get());
                                }
                            }
                        }
                    }
                }
                modelBoneMatrices = pose.boneTransforms;

                // ===========================================================================
                // CRITICAL FIX: Direct bone-to-index merging for AnimGraph
                // Graph pose.boneTransforms order matches the order in boneData.boneIndexToName
                // but we must ONLY update the indices that belong to THIS model.
                // CRITICAL FIX: Always copy the matrix.
                // If it's identity, it means the bone IS at origin/bind pose.
                // Skipping identity used to cause "stuck" bones from previous poses.
                // ===========================================================================
                for (size_t localIdx = 0; localIdx < modelBoneMatrices.size(); ++localIdx) {
                    const std::string& boneName = scene.boneData.getBoneNameByIndex(localIdx);
                    
                    // Only update if this bone belongs to this model prefix
                    if (boneName.find(ctx.importName + "_") == 0) {
                        // Find the global index for this bone (it should match localIdx here 
                        // IF scene.boneData was built in the same order, but let's be safe)
                        auto it = scene.boneData.boneNameToIndex.find(boneName);
                        if (it != scene.boneData.boneNameToIndex.end()) {
                            unsigned int globalIdx = it->second;
                            if (globalIdx < this->finalBoneMatrices.size()) {
                                this->finalBoneMatrices[globalIdx] = modelBoneMatrices[localIdx];
                            }
                        }
                    }
                }
            }
        }
        else if (ctx.animator) {
            // Sync UI toggle to animator state
            if (ctx.useRootMotion) {
                std::string activeClip = ctx.animator->getCurrentClipName();
                std::string bestRoot = ctx.animator->findBestRootMotionBone(activeClip);
                ctx.animator->setRootMotionEnabled(true, bestRoot);
            }
            else {
                ctx.animator->setRootMotionEnabled(false);
            }

            bool changed = ctx.animator->update(deltaTime, scene.boneData);

            // Get this model's bone matrices (current state)
            modelBoneMatrices = ctx.animator->getFinalBoneMatrices();

            // ===========================================================================
            // CRITICAL FIX: Map Animator Local Indices to Global Indices
            // The animator's matrices are per-model. We must map them to global slots.
            // ===========================================================================
            const auto& allClips = ctx.animator->getAllClips();
            if (!allClips.empty()) {
                // We can use the bone mapping from the first clip as a reference for node names
                const auto& source = allClips[0].sourceData;
                if (source) {
                    // This is more complex because Animator doesn't easily expose local-to-global mapping
                    // But we can iterate over ALL bones in the scene and see which ones belong to this model
            // MODEL ISOLATION FIX: 
            // In the AnimationController, cachedFinalBoneMatrices is already sized for the global bone count.
            // However, it only contains valid data for bones that belong to the model it's controlling.
            // We need to copy ONLY the bones that this model 'owns' to avoid overwriting 
            // other models' poses with the fallback Identity pose from this animator's cache.
            for (const auto& [boneName, globalIdx] : scene.boneData.boneNameToIndex) {
                if (boneName.find(ctx.importName + "_") == 0) {
                    if (globalIdx < this->finalBoneMatrices.size() && globalIdx < modelBoneMatrices.size()) {
                        this->finalBoneMatrices[globalIdx] = modelBoneMatrices[globalIdx];
                    }
                }
            }
                }
            }

            if (changed) {
                modelChanged = true;


                // --- ROOT MOTION (Pivot movement) ---
                if (ctx.useRootMotion) {
                    RootMotionDelta delta = ctx.animator->consumeRootMotion();
                    if (delta.hasPosition && !ctx.members.empty()) {
                        std::vector<Transform*> processed;
                        for (auto& member : ctx.members) {
                            if (auto tri = std::dynamic_pointer_cast<Triangle>(member)) {
                                auto h = tri->getTransformHandle();
                                if (h && std::find(processed.begin(), processed.end(), h.get()) == processed.end()) {
                                    h->position = h->position + delta.positionDelta;
                                    h->updateMatrix(); h->markDirty();
                                    processed.push_back(h.get());
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if (modelChanged) {
            anyChanged = true;
            
            // ============================================================
            // PER-MODEL SKINNING: Apply ONLY to this model's own members
            // If ctx.members is empty (e.g. after project load), lazy-init
            // by matching triangle nodeName prefix to ctx.importName.
            // ============================================================
            if (apply_cpu_skinning && !modelBoneMatrices.empty()) {
                // Lazy populate members if empty (project load doesn't serialize them)
                if (ctx.members.empty() && !ctx.importName.empty()) {
                    std::string prefix = ctx.importName + "_";
                    for (auto& obj : scene.world.objects) {
                        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                        if (tri && tri->nodeName.find(prefix) == 0) {
                            ctx.members.push_back(tri);
                        }
                    }
                }
                
                for (auto& member : ctx.members) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(member);
                    if (tri && tri->hasSkinData()) {
                        tri->apply_skinning(modelBoneMatrices);
                    }
                }
            }
        }
    }

    if (!anyChanged && !this->finalBoneMatrices.empty()) {
        // Even if no clip changed, we might need initial matrices
    }

    // ============================================================
    // POST-SKINNING: Update hair system to follow deformed mesh
    // Hair must be updated AFTER skinning so it reads fresh vertex positions.
    // ============================================================
    // NOTE: hairSystem.updateAllTransforms now called at the end of updateAnimationState
    // to ensure it runs for both Graph and Legacy/Manual animations.

    // Only reset CPU accumulation when geometry actually changed
    // Otherwise sample counter keeps resetting to 0 every frame
    if (anyChanged) {
        resetCPUAccumulation();
    }

    // Clear dirty flags for all animators
    for (auto& ctx : scene.importedModelContexts) {
        if (ctx.animator) {
            ctx.animator->clearDirtyFlag();
        }
    }

    return anyChanged;  // Only report geometry change when it actually happened
}


bool Renderer::updateAnimationState(SceneData& scene, float current_time, bool apply_cpu_skinning, bool force_bind_pose) {
    // ===========================================================================
    // GEOMETRY CHANGE TRACKING (Animation Performance Optimization)
    // ===========================================================================
    // Track if actual geometry (vertex positions) changed.
    // Return false for camera-only or material-only animations to avoid unnecessary BVH rebuilds.
    bool geometry_changed = false;

    static bool was_in_bind_pose = false;
    if (force_bind_pose) {
        if (!was_in_bind_pose) {
            was_in_bind_pose = true;
            geometry_changed = true;
            
            if (!scene.boneData.boneNameToIndex.empty()) {
                this->finalBoneMatrices.assign(scene.boneData.boneNameToIndex.size(), Matrix4x4::identity());
            }

            // Ensure animation groups are built
            if (animation_groups_dirty || animation_groups.empty()) {
                animation_groups.clear();
                std::unordered_map<void*, size_t> transformToGroup;
                for (auto& obj : scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (!tri) continue;
                    void* transformKey = tri->getTransformHandle().get();
                    if (transformToGroup.find(transformKey) == transformToGroup.end()) {
                        transformToGroup[transformKey] = animation_groups.size();
                        AnimatableGroup newGroup;
                        newGroup.nodeName = tri->getNodeName();
                        newGroup.isSkinned = tri->hasSkinData();
                        newGroup.transformHandle = tri->getTransformHandle();
                        animation_groups.push_back(newGroup);
                    }
                    animation_groups[transformToGroup[transformKey]].triangles.push_back(tri);
                }
                animation_groups_dirty = false;
            }

            for (auto& group : animation_groups) {
                if (group.isSkinned) {
                    if (apply_cpu_skinning) {
                        for (auto& tri : group.triangles) {
                            tri->apply_skinning(static_cast<const std::vector<Matrix4x4>&>(this->finalBoneMatrices));
                        }
                    }
                }
            }
            
            // Rebuild BVHs
            auto embree_ptr = std::dynamic_pointer_cast<EmbreeBVH>(scene.bvh);
            if (embree_ptr) {
                embree_ptr->updateGeometryFromTrianglesFromSource(scene.world.objects);
            }
            
            if (hairSystem.getTotalStrandCount() > 0) {
                hairSystem.updateAllTransforms(scene.world.objects, this->finalBoneMatrices);
                if (hairSystem.isBVHDirty()) {
                    hairSystem.buildBVH(true);
                    uploadHairToGPU();
                }
            }
        }
        return geometry_changed;
    } else {
        if (was_in_bind_pose) {
            was_in_bind_pose = false;
            // Force animation resync
        }
    }


    // Unified Animation Check: If we have clips and bones, use the Controller

    // NOTE: resetCPUAccumulation is now called inside updateAnimationWithGraph
    // only when geometry actually changes. Calling it here unconditionally
    // was preventing sample accumulation beyond 1.

    lastAnimationUpdateTime = current_time;

    // Unified Animation Check: If we have clips and bones, use the Controller
    bool useAnimationController = !scene.animationDataList.empty() && !scene.boneData.boneNameToIndex.empty();

    if (useAnimationController) {
        static float last_sim_time = -1.0f;
        static int last_timeline_frame = -1;
        static auto last_wall_time = std::chrono::steady_clock::now();

        auto now = std::chrono::steady_clock::now();
        float wallDelta = std::chrono::duration<float>(now - last_wall_time).count();
        last_wall_time = now;
        if (wallDelta > 0.1f) wallDelta = 1.0f / 60.0f;

        float deltaTime = (last_sim_time >= 0.0f) ? (current_time - last_sim_time) : (1.0f / 60.0f);
        if (deltaTime < -0.5f || deltaTime > 0.5f) deltaTime = 0.0f;

        // SCRUBBING FIX: Absolute time seek support
        // If current_time changed drastically or timeline was scrubbed, sync animators
        bool timelineScrubbed = (scene.timeline.current_frame != last_timeline_frame);
        
        if (!timelineScrubbed && std::abs(deltaTime) < 0.0001f) {
            deltaTime = wallDelta;
        }

        if (timelineScrubbed) {
            for (auto& modelCtx : scene.importedModelContexts) {
                if (modelCtx.animator) {
                    modelCtx.animator->setTime(current_time, 0); // Seek to current simulation time
                }
            }
            // Use zero delta for seek frames to avoid double-advancing
            deltaTime = 0.0f; 
        }

        // Drive the animation
        bool changed = updateAnimationWithGraph(scene, deltaTime, apply_cpu_skinning);
        
        geometry_changed = changed || timelineScrubbed;
        
        last_sim_time = current_time;
        last_timeline_frame = scene.timeline.current_frame;
    }
    else {
        // --- LEGACY FALLBACK PATH ---
        // This part is only reached if useAnimationController is false.


    // --- 1. Ad�m: Animasyonlu Node Hiyerar�isini G�ncelle ---
    std::unordered_map<std::string, Matrix4x4> animatedGlobalNodeTransforms;

    // Ensure bone matrices buffer is large enough for all bones in the scene
    if (!scene.boneData.boneNameToIndex.empty()) {
        if (this->finalBoneMatrices.size() < scene.boneData.boneNameToIndex.size()) {
            this->finalBoneMatrices.resize(scene.boneData.boneNameToIndex.size(), Matrix4x4::identity());
        }
    }

    // Iterate over ALL imported models to update their respective hierarchies
    for (const auto& modelCtx : scene.importedModelContexts) {
        if (!modelCtx.loader || !modelCtx.loader->getScene() || !modelCtx.loader->getScene()->mRootNode) continue;

        // ===========================================================================
        // CRITICAL FIX: Skip models without animation data
        // ===========================================================================
        if (!modelCtx.hasAnimation) {
            continue; // Skip non-animated models - preserve their transforms
        }

        // Build lookups for THIS model's animations
        std::map<std::string, std::shared_ptr<AnimationData>> animationLookupMap;
        for (const auto& anim : scene.animationDataList) {
            if (!anim) continue;
            for (const auto& pair : anim->positionKeys) animationLookupMap[pair.first] = anim;
            for (const auto& pair : anim->rotationKeys) animationLookupMap[pair.first] = anim;
            for (const auto& pair : anim->scalingKeys) animationLookupMap[pair.first] = anim;
        }

        // Temporary map for THIS model's node transforms
        std::unordered_map<std::string, Matrix4x4> modelNodeTransforms;

        modelCtx.loader->calculateAnimatedNodeTransformsRecursive(
            modelCtx.loader->getScene()->mRootNode,
            Matrix4x4::identity(),
            animationLookupMap,
            current_time,
            modelNodeTransforms
        );

        // Merge into global map (for later use by non-bone animated objects)
        for (const auto& pair : modelNodeTransforms) {
            animatedGlobalNodeTransforms[pair.first] = pair.second;
        }

        // --- PRE-CALCULATE GLOBAL BONE MATRICES for THIS model ---
        for (const auto& [boneName, boneIndex] : scene.boneData.boneNameToIndex) {
            // Only process bones that belong to THIS model context
            if (boneName.find(modelCtx.importName + "_") == 0) {
                if (modelNodeTransforms.count(boneName) > 0 && scene.boneData.boneOffsetMatrices.count(boneName) > 0) {
                    Matrix4x4 animatedBoneGlobal = modelNodeTransforms[boneName];
                    Matrix4x4 offsetMatrix = scene.boneData.boneOffsetMatrices[boneName];

                    // P_world = model_globalInv * animGlobal * offset
                    finalBoneMatrices[boneIndex] = modelCtx.globalInverseTransform * animatedBoneGlobal * offsetMatrix;
                }
                else {
                    // Fallback to identity for missing keys in this model's rigged hierarchy
                    if (boneIndex < finalBoneMatrices.size()) {
                        finalBoneMatrices[boneIndex] = Matrix4x4::identity();
                    }
                }
            }
        }
    }


    // Ensure finalBoneMatrices is sized correctly for any remaining bones
    if (finalBoneMatrices.size() < scene.boneData.boneNameToIndex.size()) {
        finalBoneMatrices.resize(scene.boneData.boneNameToIndex.size(), Matrix4x4::identity());
    }

    // --- 0. Ad�m: Performans �nbelle�i Haz�rl��� ---
    if (animation_groups_dirty || animation_groups.empty()) {
        animation_groups.clear();
        std::unordered_map<void*, size_t> transformToGroup;

        for (auto& obj : scene.world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (!tri) continue;

            void* transformKey = tri->getTransformHandle().get();
            if (transformToGroup.find(transformKey) == transformToGroup.end()) {
                transformToGroup[transformKey] = animation_groups.size();
                AnimatableGroup newGroup;
                newGroup.nodeName = tri->getNodeName();
                newGroup.isSkinned = tri->hasSkinData();
                newGroup.transformHandle = tri->getTransformHandle();
                animation_groups.push_back(newGroup);
            }
            animation_groups[transformToGroup[transformKey]].triangles.push_back(tri);
        }
        animation_groups_dirty = false;
        SCENE_LOG_INFO("Animation groups rebuilt: " + std::to_string(animation_groups.size()) + " groups.");
    }

    // --- 2. Ad�m: Gruplar� Animasyon T�r�ne G�re G�ncelle ---
    for (auto& group : animation_groups) {
        if (group.triangles.empty()) continue;

        if (group.isSkinned) {
            // Skeleton animation modifies geometry
            geometry_changed = true;
            if (apply_cpu_skinning) {
                for (auto& tri : group.triangles) {
                    tri->apply_skinning(static_cast<const std::vector<Matrix4x4>&>(finalBoneMatrices));
                }
            }
        }
        else {
            // --- RIGID ANIMATION ---
            bool nodeHasAnimation = false;
            for (const auto& anim : scene.animationDataList) {
                if (!anim) continue;
                if (anim->positionKeys.count(group.nodeName) > 0 ||
                    anim->rotationKeys.count(group.nodeName) > 0 ||
                    anim->scalingKeys.count(group.nodeName) > 0) {
                    nodeHasAnimation = true;
                    break;
                }
            }

            if (nodeHasAnimation && animatedGlobalNodeTransforms.count(group.nodeName) > 0) {
                Matrix4x4 animTransform = animatedGlobalNodeTransforms[group.nodeName];
                if (group.transformHandle) {
                    group.transformHandle->setBase(animTransform);
                    group.transformHandle->setCurrent(Matrix4x4::identity());
                    if (apply_cpu_skinning) {
                        for (auto& tri : group.triangles) tri->updateTransformedVertices();
                    }
                    geometry_changed = true;
                }
            }
            else {
                if (animatedGlobalNodeTransforms.count(group.nodeName) > 0) {
                    Matrix4x4 staticTransform = animatedGlobalNodeTransforms[group.nodeName];
                    if (group.transformHandle) {
                        group.transformHandle->setBase(staticTransform);
                        group.transformHandle->setCurrent(Matrix4x4::identity());
                        if (apply_cpu_skinning) {
                            for (auto& tri : group.triangles) tri->updateTransformedVertices();
                        }
                        geometry_changed = true;
                    }
                }

                // --- MANUAL KEYFRAME SUPPORT ---
                if (!nodeHasAnimation && !scene.timeline.tracks.empty()) {
                    extern RenderSettings render_settings;
                    int current_frame = static_cast<int>(current_time * render_settings.animation_fps);
                    auto track_it = scene.timeline.tracks.find(group.nodeName);
                    if (track_it != scene.timeline.tracks.end() && !track_it->second.keyframes.empty()) {
                        Keyframe kf = track_it->second.evaluate(current_frame);
                        if (kf.has_transform) {
                            Matrix4x4 translation = Matrix4x4::translation(kf.transform.position);
                            float rx = kf.transform.rotation.x * (3.14159265f / 180.0f);
                            float ry = kf.transform.rotation.y * (3.14159265f / 180.0f);
                            float rz = kf.transform.rotation.z * (3.14159265f / 180.0f);
                            Matrix4x4 rotation = Matrix4x4::rotationZ(rz) * Matrix4x4::rotationY(ry) * Matrix4x4::rotationX(rx);
                            Matrix4x4 scale = Matrix4x4::scaling(kf.transform.scale);
                            Matrix4x4 final_transform = translation * rotation * scale;

                            if (group.transformHandle) {
                                group.transformHandle->setBase(final_transform);
                                group.transformHandle->setCurrent(Matrix4x4::identity());
                                if (apply_cpu_skinning) {
                                    for (auto& tri : group.triangles) tri->updateTransformedVertices();
                                }
                                geometry_changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // --- 3. Ad�m: I��k ve Kamera Animasyonlar� (from files AND manual keyframes) ---

    // FILE-BASED Light Animation (existing code)
    for (auto& light : scene.lights) {
        bool lightHasAnimation = false;
        for (const auto& anim : scene.animationDataList) {
            if (!anim) continue;
            if (anim->positionKeys.count(light->nodeName) > 0 ||
                anim->rotationKeys.count(light->nodeName) > 0 ||
                anim->scalingKeys.count(light->nodeName) > 0) {
                lightHasAnimation = true;
                break;
            }
        }

        if (lightHasAnimation && animatedGlobalNodeTransforms.count(light->nodeName) > 0) {
            Matrix4x4 finalTransform = animatedGlobalNodeTransforms[light->nodeName];
            light->position = finalTransform.transform_point(Vec3(0, 0, 0));
            if (light->type() == LightType::Directional || light->type() == LightType::Spot) {
                light->direction = finalTransform.transform_vector(Vec3(0, 0, -1)).normalize();
            }

            // Link Nishita Sun to first Directional Light
            // Link Nishita Sun to first Directional Light
            if (light->type() == LightType::Directional) {
                world.setSunDirection(-light->direction);
                world.setSunIntensity(light->intensity);
            }
        }

        // MANUAL KEYFRAME Light Animation (NEW!)
        if (!lightHasAnimation && !scene.timeline.tracks.empty()) {
            extern RenderSettings render_settings;
            int current_frame = static_cast<int>(current_time * render_settings.animation_fps);

            auto track_it = scene.timeline.tracks.find(light->nodeName);
            if (track_it != scene.timeline.tracks.end() && !track_it->second.keyframes.empty()) {
                Keyframe kf = track_it->second.evaluate(current_frame);

                if (kf.has_light) {
                    // Only apply properties that were explicitly keyed
                    if (kf.light.has_position) {
                        light->position = kf.light.position;
                    }
                    if (kf.light.has_color) {
                        light->color = kf.light.color;
                    }
                    if (kf.light.has_intensity) {
                        light->intensity = kf.light.intensity;
                    }

                    if (kf.light.has_direction) {
                        if (light->type() == LightType::Directional || light->type() == LightType::Spot) {
                            light->direction = kf.light.direction.normalize();

                            if (light->type() == LightType::Directional) {
                                world.setSunDirection(-light->direction);
                                world.setSunIntensity(light->intensity);
                            }
                        }
                    }
                }
            }
        }
    }

    // FILE-BASED Camera Animation (existing code)
    bool cameraHasAnimation = false;
    if (scene.camera) {
        for (const auto& anim : scene.animationDataList) {
            if (!anim) continue;
            if (anim->positionKeys.count(scene.camera->nodeName) > 0 ||
                anim->rotationKeys.count(scene.camera->nodeName) > 0 ||
                anim->scalingKeys.count(scene.camera->nodeName) > 0) {
                cameraHasAnimation = true;
                break;
            }
        }
    }

    if (scene.camera && cameraHasAnimation && animatedGlobalNodeTransforms.count(scene.camera->nodeName) > 0) {
        // Apply global inverse REMOVED. Static camera works without it.
        // We suspect globalInverse was introducing the "tilt" (sola yat�k).
        // We KEEP the manual UP vector flip because user confirmed it fixed "tepetaklak". (Upside down)

        Matrix4x4 animTransform = animatedGlobalNodeTransforms[scene.camera->nodeName];

        Vec3 pos = animTransform.transform_point(Vec3(0, 0, 0));
        // Blender cameras usually point down -Z.
        Vec3 forward = animTransform.transform_vector(Vec3(0, 0, -1)).normalize();

        // FIX: Force Global Up (0, 1, 0) to prevent unwanted Roll/Tilt (sola yat�k).
        // This mimics "Track To" constraint behavior where the camera stays level.
        // If the camera needs to roll (bank), this line should be reverted to use transformed up.
        // Note on position precision: Small errors (e.g. -0.003 instad of 0) are due to Linear vs Bezier interpolation differences.
        Vec3 up = Vec3(0, 1, 0);

        scene.camera->lookfrom = pos;
        scene.camera->lookat = pos + forward;

        // Handle Gimbal Lock: If looking straight up/down, keep previous up or logic might fail slightly.
        if (abs(Vec3::dot(forward, up)) < 0.99f) {
            scene.camera->vup = up;
        }
        scene.camera->update_camera_vectors();
    }

    // MANUAL KEYFRAME Camera Animation (NEW!)
    if (scene.camera && !cameraHasAnimation && !scene.timeline.tracks.empty()) {
        extern RenderSettings render_settings;
        int current_frame = static_cast<int>(current_time * render_settings.animation_fps);

        auto track_it = scene.timeline.tracks.find(scene.camera->nodeName);
        if (track_it != scene.timeline.tracks.end() && !track_it->second.keyframes.empty()) {
            Keyframe kf = track_it->second.evaluate(current_frame);

            if (kf.has_camera) {
                // Only apply properties that were explicitly keyed
                if (kf.camera.has_position) {
                    scene.camera->lookfrom = kf.camera.position;
                }
                if (kf.camera.has_target) {
                    scene.camera->lookat = kf.camera.target;
                }
                if (kf.camera.has_fov) {
                    scene.camera->vfov = kf.camera.fov;
                }
                if (kf.camera.has_focus) {
                    scene.camera->focus_dist = kf.camera.focus_distance;
                }
                if (kf.camera.has_aperture) {
                    scene.camera->lens_radius = kf.camera.lens_radius;
                }

                // Update camera vectors
                Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                Vec3 up = Vec3(0, 1, 0);
                if (abs(Vec3::dot(forward, up)) < 0.99f) {
                    scene.camera->vup = up;
                }
                scene.camera->update_camera_vectors();
            }
        }
    }

    // --- MATERIAL KEYFRAME EVALUATION (OPTIMIZED!) ---
    // OPTIMIZATION: Instead of iterating through ALL objects (10M+), iterate only through
    // timeline tracks that have material keyframes. This is O(tracks) instead of O(objects).
    if (!scene.timeline.tracks.empty()) {
        extern RenderSettings render_settings;
        int current_frame = static_cast<int>(current_time * render_settings.animation_fps);

        // Build a cache of node names to material IDs for fast lookup (done once)
        // OPTIMIZATION: This cache should ideally be built once when scene loads,
        // but for now we only process tracks that have keyframes - much faster than 10M objects

        for (auto& [track_name, track] : scene.timeline.tracks) {
            // Skip tracks without material keyframes
            if (track.keyframes.empty()) continue;

            // Evaluate keyframe at current frame
            Keyframe kf = track.evaluate(current_frame);
            if (!kf.has_material) continue;

            // Get material ID from keyframe
            uint16_t mat_id = kf.material.material_id;

            // If no material ID in keyframe, we need to find it from an object with this node name
            // This is a fallback - ideally material ID should be stored in the keyframe
            if (mat_id == 0) {
                // Quick lookup: find first object with this node name
                // TODO: Cache this mapping for better performance
                for (const auto& obj : scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (tri && tri->getNodeName() == track_name) {
                        mat_id = tri->getMaterialID();
                        break; // Found it, no need to continue
                    }
                }
            }

            if (mat_id == 0) continue; // No material ID found

            // Get material and apply keyframe values
            Material* mat_ptr = MaterialManager::getInstance().getMaterial(mat_id);
            if (mat_ptr && mat_ptr->gpuMaterial) {
                // Apply interpolated material properties to GpuMaterial
                kf.material.applyTo(*mat_ptr->gpuMaterial);

                // Also update CPU-side material properties if it's a PrincipledBSDF
                if (auto* pbsdf = dynamic_cast<PrincipledBSDF*>(mat_ptr)) {
                    pbsdf->albedoProperty.color = kf.material.albedo;
                    pbsdf->roughnessProperty.color = Vec3(kf.material.roughness);
                    pbsdf->metallicProperty.intensity = kf.material.metallic;
                    pbsdf->emissionProperty.color = kf.material.emission;
                    pbsdf->ior = kf.material.ior;
                    pbsdf->transmission = kf.material.transmission;
                    pbsdf->opacityProperty.alpha = kf.material.opacity;
                }
            }
        }

        // --- WORLD KEYFRAME EVALUATION (NEW!) ---
        auto world_track_it = scene.timeline.tracks.find("World");
        if (world_track_it != scene.timeline.tracks.end() && !world_track_it->second.keyframes.empty()) {
            Keyframe kf = world_track_it->second.evaluate(current_frame);
            if (kf.has_world) {
                const WorldKeyframe& wk = kf.world;

                // Apply each property independently based on its flag

                // Background Color
                if (wk.has_background_color) {
                    world.setColor(wk.background_color);
                    scene.background_color = wk.background_color;
                }
                // Background Strength
                if (wk.has_background_strength) {
                    world.setColorIntensity(wk.background_strength);
                }
                // HDRI Rotation
                if (wk.has_hdri_rotation) {
                    world.setHDRIRotation(wk.hdri_rotation);
                }

                // Nishita Sky - Get params once, update as needed, set once at end
                bool need_nishita_update = (wk.has_sun_elevation || wk.has_sun_azimuth ||
                    wk.has_sun_intensity || wk.has_sun_size ||
                    wk.has_air_density || wk.has_dust_density ||
                    wk.has_ozone_density || wk.has_altitude ||
                    wk.has_mie_anisotropy ||
                    wk.has_cloud_density || wk.has_cloud_coverage ||
                    wk.has_cloud_scale || wk.has_cloud_offset);

                if (need_nishita_update) {
                    NishitaSkyParams np = world.getNishitaParams();

                    // Sun properties
                    if (wk.has_sun_elevation) np.sun_elevation = wk.sun_elevation;
                    if (wk.has_sun_azimuth) np.sun_azimuth = wk.sun_azimuth;
                    if (wk.has_sun_intensity) np.sun_intensity = wk.sun_intensity;
                    if (wk.has_sun_size) np.sun_size = wk.sun_size;

                    // Recalculate sun direction if elevation or azimuth changed
                    if (wk.has_sun_elevation || wk.has_sun_azimuth) {
                        float elRad = np.sun_elevation * 3.14159265f / 180.0f;
                        float azRad = np.sun_azimuth * 3.14159265f / 180.0f;
                        np.sun_direction = make_float3(
                            cosf(elRad) * sinf(azRad),
                            sinf(elRad),
                            cosf(elRad) * cosf(azRad)
                        );
                    }

                    // Atmosphere properties
                    if (wk.has_air_density) np.air_density = wk.air_density;
                    if (wk.has_dust_density) np.dust_density = wk.dust_density;
                    if (wk.has_ozone_density) np.ozone_density = wk.ozone_density;
                    if (wk.has_altitude) np.altitude = wk.altitude;
                    if (wk.has_mie_anisotropy) np.mie_anisotropy = wk.mie_anisotropy;

                    // Cloud properties
                    if (wk.has_cloud_density) np.cloud_density = wk.cloud_density;
                    if (wk.has_cloud_coverage) np.cloud_coverage = wk.cloud_coverage;
                    if (wk.has_cloud_scale) np.cloud_scale = wk.cloud_scale;
                    if (wk.has_cloud_offset) {
                        np.cloud_offset_x = wk.cloud_offset_x;
                        np.cloud_offset_z = wk.cloud_offset_z;
                    }

                    // Enable clouds if coverage > 0
                    if (wk.has_cloud_coverage && np.cloud_coverage > 0.0f) {
                        np.clouds_enabled = 1;
                    }

                    world.setNishitaParams(np);
                }
            }
        }
    }

    } // End of Legacy Else
    
    // --- 4. Ad�m: BVH G�ncelle (only if geometry changed) ---
    // OPTIMIZATION: Skip CPU BVH rebuild for camera-only or material-only animations
    if (geometry_changed) {
        auto embree_ptr = std::dynamic_pointer_cast<EmbreeBVH>(scene.bvh);
        if (embree_ptr) {
            embree_ptr->updateGeometryFromTrianglesFromSource(scene.world.objects);
        }
    }
    
    // ===========================================================================
    // HAIR SYSTEM UPDATE (Rigid & Skeletal Synchronization)
    // ===========================================================================
    // This MUST run after ALL bone/rigid transformations are final.
    if (hairSystem.getTotalStrandCount() > 0) {
        // We pass the final calculated bone matrices.
        // This handles guide skinning and BVH updates.
        hairSystem.updateAllTransforms(scene.world.objects, this->finalBoneMatrices);
        
        if (hairSystem.isBVHDirty()) {
            hairSystem.buildBVH(true);
            uploadHairToGPU();
        }
    }

    return geometry_changed;
}

void Renderer::render_Animation(SDL_Surface* surface, SDL_Window* window, SDL_Texture* raytrace_texture, SDL_Renderer* renderer,
    const int total_samples_per_pixel, const int samples_per_pass, float fps, float duration, int start_frame, int end_frame, SceneData& scene,
    const std::string& output_folder, bool use_denoiser, float denoiser_blend,
    Backend::IBackend* backend, bool use_optix, UIContext* ui_ctx) {

    render_finished = false;
    rendering_complete = false;
    rendering_in_progress = true;
    rendering_stopped_cpu = false;
    rendering_stopped_gpu = false;
    
    // Reset pause state at start of new animation render
    extern std::atomic<bool> rendering_paused;
    rendering_paused = false;

    // Backend is already synced to this->m_backend in Main.cpp or passed via parameter
    if (!m_backend && backend) m_backend = backend;
      // LOCK VIEWPORT/CAMERA INPUT during animation render
    if (ui_ctx) {
        ui_ctx->render_settings.animation_render_locked = true;
    }

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    auto start_time = std::chrono::steady_clock::now();
    float frame_time = 1.0f / fps;

    extern RenderSettings render_settings;
    extern bool g_hasOptix; // Ensure we can access global flag

    // DISABLE GRID/OVERLAYS FOR ANIMATION RENDER
    bool original_render_mode = render_settings.is_final_render_mode;
    render_settings.is_final_render_mode = true;

    // Frame range is validated in Main.cpp before calling this function
    // We trust the values passed as parameters
    SCENE_LOG_INFO("render_Animation: Frame range " + std::to_string(start_frame) + " - " + std::to_string(end_frame) + 
                   " (" + std::to_string(end_frame - start_frame + 1) + " frames)");
    SCENE_LOG_INFO("render_Animation: " + std::to_string(total_samples_per_pixel) + " samples per frame, " + 
                   std::to_string(fps) + " FPS, Mode: " + (use_optix ? "OptiX" : "CPU"));

    int total_frames = end_frame - start_frame + 1;
    if (total_frames <= 0) {
        SCENE_LOG_ERROR("Invalid frame range! Aborting animation render.");
        render_finished = true;
        rendering_complete = true;
        rendering_in_progress = false;
        render_settings.is_final_render_mode = original_render_mode;
        return;
    }

    if (!output_folder.empty()) {
        std::filesystem::create_directories(output_folder);
        SCENE_LOG_INFO("Animation frames will be saved to: " + output_folder);
    }

    SCENE_LOG_INFO("Starting animation render: " + std::to_string(total_frames) + " frames (Frame " +
        std::to_string(start_frame) + " to " + std::to_string(end_frame) + ") at " + std::to_string(fps) + " FPS");

    // Sync frame range to UI context for accurate progress display
    if (ui_ctx) {
        ui_ctx->render_settings.animation_start_frame = start_frame;
        ui_ctx->render_settings.animation_end_frame = end_frame;
        ui_ctx->render_settings.animation_total_frames = total_frames;
    }

    // Check if OptiX is valid
    bool run_optix = use_optix && m_backend && g_hasOptix;

    // IMPORTANT: Sync local optix pointer for updateWind updates to work
    // This is no longer needed as m_backend is used directly
    // if (run_optix) {
    //     this->m_backend = backend;
    // }

    if (use_optix && !run_optix) {
        SCENE_LOG_WARN("OptiX requested but not available/valid. Falling back to CPU.");
    }

    for (int frame = start_frame; frame <= end_frame; ++frame) {

        // Update BOTH global render_settings AND ui_ctx for UI synchronization
        render_settings.animation_current_frame = frame;
        if (ui_ctx) {
            ui_ctx->render_settings.animation_current_frame = frame;
        }

        if (rendering_stopped_cpu || rendering_stopped_gpu) {
            SCENE_LOG_WARN("Animation rendering stopped by user at frame " + std::to_string(frame));
            break;
        }

        // Clear CPU buffers anyway (for safety/consistency)
        std::fill(frame_buffer.begin(), frame_buffer.end(), Vec3(0.0f));
        std::fill(sample_counts.begin(), sample_counts.end(), 0);
        // REMOVED: SDL_FillRect(surface, NULL, 0) to prevent main window blackout

        // Use absolute time based on frame number (not relative to start_frame)
        float current_time = frame * frame_time;

        SCENE_LOG_INFO("Rendering frame " + std::to_string(frame) + "/" + std::to_string(end_frame) +
            " at time " + std::to_string(current_time) + "s (Mode: " + (run_optix ? "OptiX" : "CPU") + ")");

        // --- SYNC TIMELINE FRAME FOR MATERIAL KEYFRAME EVALUATION ---
        // CRITICAL: updateAnimationState's material keyframe code (lines 797-842) reads
        // render_settings.animation_fps to calculate current_frame. We must sync the 
        // timeline frame counter so material evaluation works correctly!
        extern RenderSettings render_settings;
        render_settings.animation_current_frame = frame;
        render_settings.animation_playback_frame = frame;
        scene.timeline.current_frame = frame;

        // Returns true if geometry changed
        // Disable CPU skinning if running on OptiX to save performance and prevent crashes
        // We only need CPU skinning for CPU rendering or if we need to update CPU BVH
        bool geometry_changed = this->updateAnimationState(scene, current_time, !run_optix);

        // --- WIND ANIMATION ---
        // Apply wind simulation for this frame
        // FIX: Use same pattern as Play Mode (Main.cpp line 691-697) for consistent behavior
        if (run_optix && m_backend) {
            // Calculate wind transforms on CPU (same as Play Mode)
            InstanceManager::getInstance().updateWind(current_time, scene);

            // Efficiently update instance transforms on GPU (no full rebuild)
            // This is the critical step that was missing - ensures GPU TLAS has updated matrices
            m_backend->updateInstanceTransforms(scene.world.objects);

            // Wind updates don't require geometry_changed = true
            // Only instance transforms changed, not vertex data
        }

        // --- VDB VOLUME ANIMATION (FIX) ---
        // Update VDB sequences for current frame (loads new grid from disk if needed)
        scene.updateVDBVolumesFromTimeline(frame);

        // Sync VDBs to GPU if running OptiX
        // This ensures the new grid data is uploaded to GPU memory
        if (run_optix && ui_ctx) {
            SceneUI::syncVDBVolumesToGPU(*ui_ctx);
            // Note: syncVDBVolumesToGPU handles geometry flag updates internally if needed
        }

        // --- TERRAIN ANIMATION ---
        // Apply terrain keyframes for this frame (morphing animation)
        for (auto& [track_name, track] : scene.timeline.tracks) {
            // Check if this track has terrain keyframes
            bool has_terrain_kf = false;
            for (auto& kf : track.keyframes) {
                if (kf.has_terrain) {
                    has_terrain_kf = true;
                    break;
                }
            }

            if (has_terrain_kf) {
                // Find terrain by name
                auto& terrains = TerrainManager::getInstance().getTerrains();
                for (auto& terrain : terrains) {
                    if (terrain.name == track_name) {
                        TerrainManager::getInstance().updateFromTrack(&terrain, track, frame);
                        geometry_changed = true;  // Terrain morph = geometry change
                        break;
                    }
                }
            }
        }

        // --- WATER/FFT OCEAN ANIMATION ---
        // Apply water keyframes for this frame (FFT parameter animation)
        for (auto& [track_name, track] : scene.timeline.tracks) {
            // Check if this is a Water track (name starts with "Water_")
            if (track_name.rfind("Water_", 0) != 0) continue;
            
            // Check if this track has water keyframes
            bool has_water_kf = false;
            for (auto& kf : track.keyframes) {
                if (kf.has_water) {
                    has_water_kf = true;
                    break;
                }
            }
            
            if (has_water_kf) {
                // Extract water surface ID from track name (format: "Water_X")
                auto& waters = WaterManager::getInstance().getWaterSurfaces();
                for (auto& water : waters) {
                    std::string expected_name = "Water_" + std::to_string(water.id);
                    if (track_name == expected_name) {
                        WaterManager::getInstance().updateFromTrack(&water, track, frame);
                        // FFT changes don't need geometry rebuild - they're shader-based
                        // But geometric waves do need BVH update
                        if (water.params.use_geometric_waves && water.animate_mesh) {
                            geometry_changed = true;
                        }
                        break;
                    }
                }
            }
        }

        // --- WATER ANIMATION UPDATE ---
        // Update FFT ocean simulation and geometric wave mesh animation
        // Frame delta: 1.0/fps gives the time for one frame
        float frame_delta = 1.0f / static_cast<float>(render_settings.animation_fps);
        if (WaterManager::getInstance().update(frame_delta)) {
            // If update returns true, water mesh changed (geometric waves)
            geometry_changed = true;
        }

        // --- RENDER BUFFER SETUP ---
        // Use a dedicated off-screen surface for BOTH CPU and OptiX to prevent 
        // access violations when render resolution != window size.
        SDL_Surface* target_surface = surface; // Default fallback
        SDL_Surface* render_surface = nullptr;

        int rw = render_settings.final_render_width;
        int rh = render_settings.final_render_height;

        // Always create off-screen surface for animation
        render_surface = SDL_CreateRGBSurfaceWithFormat(0, rw, rh, 32, SDL_PIXELFORMAT_RGBA32);
        if (render_surface) {
            target_surface = render_surface;
        }

        if (run_optix) {
            // --- OPTIX RENDER PATH ---

            // 1. Update Geometry if needed
            // IMPORTANT: Terrain erosion can change triangle count, not just positions.
            // Full rebuildOptiXGeometry is required to handle topology changes.
            if (geometry_changed) {
                // Determine if we can do a fast update or need a full rebuild
                // For now, let the backend or renderer-level rebuild handle it
                this->rebuildBackendGeometry(scene);
            }
              // 1.5. Update GPU Materials (CRITICAL for material keyframe animations!)
            // This uploads the materials updated by updateAnimationState to OptiX
            this->updateBackendMaterials(scene);
            // 2. Set Scene Params
            this->syncCameraToBackend(*scene.camera);
            if (m_backend) {
                m_backend->setLights(scene.lights);
                WorldData wd = this->world.getGPUData();
                m_backend->setWorldData(&wd);
                // If Vulkan backend, upload AtmosphereLUT host arrays so shaders can sample LUTs
                if (m_backend) {
                    auto* vulkanBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(m_backend);
                    if (vulkanBackend) {
                        auto* al = this->world.getLUT();
                        if (al && al->is_initialized()) vulkanBackend->uploadAtmosphereLUT(al);
                    }
                }
                m_backend->resetAccumulation();
            }
              // 4. Render Loop (Accumulate Samples until target reached)
            // 4. Render Loop (Accumulate Samples until target reached)
            std::vector<uchar4> temp_framebuffer(target_surface->w * target_surface->h);

            // Render until max samples reached
            while (m_backend && !m_backend->isAccumulationComplete() && !rendering_stopped_gpu) {
                // PAUSE WAIT - Block here while paused, check stop flag periodically
                while (rendering_paused.load() && !rendering_stopped_gpu.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (rendering_stopped_gpu.load()) break;
                
                // Launch progressive render
                // Pass 'nullptr' for window to disable title updates (headless like)
                 if (m_backend) {
                     void* framebuffer_ptr = (void*)&temp_framebuffer;
                     m_backend->renderProgressive(target_surface, nullptr, renderer, 
                                                 target_surface->w, target_surface->h, 
                                                 framebuffer_ptr, raytrace_texture);
                 }
            }
            
            // IMMEDIATE EXIT CHECK after GPU render loop
            if (rendering_stopped_gpu.load()) {
                SCENE_LOG_WARN("Animation render stopped during GPU frame " + std::to_string(frame));
                if (render_surface) { SDL_FreeSurface(render_surface); }
                break;
            }
        }
        else {
            // --- CPU RENDER PATH ---

            // Reset CPU accumulation for new frame
            resetCPUAccumulation();

            // Render until max samples reached
            while (!isCPUAccumulationComplete() && !rendering_stopped_cpu) {
                // PAUSE WAIT - Block here while paused, check stop flag periodically
                while (rendering_paused.load() && !rendering_stopped_cpu.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (rendering_stopped_cpu.load()) break;
                
                // For animation CPU render: pass 'total_samples_per_pixel' as the target
                // This ensures we reach the "final render samples" count, not viewport "max samples"
                render_progressive_pass(target_surface, window, scene, 1, total_samples_per_pixel);
            }
            
            // IMMEDIATE EXIT CHECK after CPU render loop
            if (rendering_stopped_cpu.load()) {
                SCENE_LOG_WARN("Animation render stopped during CPU frame " + std::to_string(frame));
                if (render_surface) { SDL_FreeSurface(render_surface); }
                break;
            }
        }

        // --- COMMON: Denoiser & Save ---

        if (use_denoiser) {
            SCENE_LOG_INFO("Applying denoiser to frame " + std::to_string(frame));
            // Note: For OptiX, launch_random... might have already denoised if internal settings were set,
            // but Renderer::applyOIDNDenoising works on SDL Surface, so it's safe to call again or instead.
            // If OptiX Wrapper already denoised, we might be double denoising?
            // OptixWrapper::launch... doesn't verify denoiser usage inside the loop shown in step 7/38.
            // But Main.cpp calls ray_renderer.applyOIDNDenoising for single frame.
            // So we call it here too.
            // Note: Renderer::applyOIDNDenoising works on SDL Surface
            applyOIDNDenoising(target_surface, 0, true, denoiser_blend);
        }

        if (!output_folder.empty()) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/frame_%04d.png", output_folder.c_str(), frame);
            // Ensure texture is updated for display before saving
           // REMOVED: Unsafe SDL calls from worker thread. Main thread handles display.

           // REMOVED: Unsafe SDL calls from worker thread. Main thread handles display.

            if (SaveSurface(target_surface, filename)) {
                SCENE_LOG_INFO("Frame saved: " + std::string(filename));

                // --- UPDATE ANIMATION PREVIEW ---
                // Access global UI context to update preview buffer
                if (ui_ctx) {
                    std::lock_guard<std::mutex> lock(ui_ctx->animation_preview_mutex);
                    int w = target_surface->w;
                    int h = target_surface->h;
                    size_t pixel_count = w * h;

                    if (ui_ctx->animation_preview_buffer.size() != pixel_count) {
                        ui_ctx->animation_preview_buffer.resize(pixel_count);
                        ui_ctx->animation_preview_width = w;
                        ui_ctx->animation_preview_height = h;
                    }

                    // Copy pixels (ensure thread safety)
                    std::memcpy(ui_ctx->animation_preview_buffer.data(), target_surface->pixels, pixel_count * sizeof(uint32_t));
                    ui_ctx->animation_preview_ready = true;
                }
            }
            else {
                SCENE_LOG_ERROR("Failed to save frame: " + std::string(filename));
            }
        }

        // Cleanup temp surface
        if (render_surface) {
            SDL_FreeSurface(render_surface);
            render_surface = nullptr;
        }
        
        // FINAL STOP CHECK - exit loop immediately if stop was requested during save/denoising
        if (rendering_stopped_cpu.load() || rendering_stopped_gpu.load()) {
            SCENE_LOG_WARN("Animation render stopped after frame " + std::to_string(frame) + " processing");
            break;
        }
    }



    rendering_complete = true;
    rendering_in_progress = false;
    render_finished = true;
    
    // UNLOCK viewport/camera input and disable animation mode
    if (ui_ctx) {
        ui_ctx->is_animation_mode = false;
        ui_ctx->render_settings.animation_render_locked = false;
    }

    // RESTORE RENDER MODE
    render_settings.is_final_render_mode = original_render_mode;

    auto end_time = std::chrono::steady_clock::now();

    if (rendering_stopped_cpu || rendering_stopped_gpu) {
        SCENE_LOG_WARN("Animation rendering was stopped by user.");
    }
    else {
        SCENE_LOG_INFO("Animation rendering completed successfully!");
    }
}

void Renderer::rebuildBVH(SceneData& scene, bool use_embree) {
    if (!scene.initialized) {
        SCENE_LOG_WARN("Scene not initialized, BVH rebuild skipped.");
        return;
    }

    // Create a temporary list of ALL hittable objects for the BVH
    // This includes World Objects (Triangles), VDB Volumes, and Gas Volumes
    std::vector<std::shared_ptr<Hittable>> all_hittables;
    // Reserve estimation: objects + gas + vdb
    all_hittables.reserve(scene.world.objects.size() + scene.gas_volumes.size() + scene.vdb_volumes.size());

    // 1. Add standard objects (Triangles/Meshes)
    for (const auto& obj : scene.world.objects) {
        if (obj && obj->visible) {
            all_hittables.push_back(obj);
        }
    }

    // 2. Add Gas Volumes (they are Hittables)
    for (const auto& gas : scene.gas_volumes) {
        if (gas && gas->visible) all_hittables.push_back(gas);
    }

    // 3. Add VDB Volumes
    for (const auto& vdb : scene.vdb_volumes) {
        if (vdb && vdb->visible) all_hittables.push_back(vdb);
    }

    // Handle empty scene
    if (all_hittables.empty()) {
        scene.bvh = nullptr;  // Clear BVH for empty scene
        SCENE_LOG_INFO("Scene is empty, BVH cleared.");
        return;
    }

    if (use_embree) {
        auto embree_bvh = std::make_shared<EmbreeBVH>();
        embree_bvh->build(all_hittables);
        scene.bvh = embree_bvh;
    }
    else {
        scene.bvh = std::make_shared<ParallelBVHNode>(all_hittables, 0, all_hittables.size(), 0.0, 1.0, 0);
        SCENE_LOG_INFO("[RayTrophi: RT_BVH]  structure built successfully.");
    }

    // IMPORTANT: Always rebuild hair BVH when main BVH is rebuilt
    // This ensures hair-to-mesh and mesh-to-hair shadows are accurate
    if (hairSystem.getTotalStrandCount() > 0) {
        hairSystem.buildBVH(!hideInterpolatedHair);
    }
}

void Renderer::updateBVH(SceneData& scene, bool use_embree) {
    if (scene.world.objects.empty()) {
        scene.bvh = nullptr;
        return;
    }
    rebuildBVH(scene, use_embree);
}

void Renderer::create_scene(SceneData& scene, Backend::IBackend* backend, const std::string& model_path,
    std::function<void(int progress, const std::string& stage)> progress_callback,
    bool append, const std::string& import_prefix) {

    // Helper lambda for progress updates
    auto update_progress = [&](int progress, const std::string& stage) {
        if (progress_callback) {
            progress_callback(progress, stage);
        }
        };

    // Only clear scene if not appending
    if (!append) {
        update_progress(0, "Cleaning previous scene...");
        SCENE_LOG_INFO("========================================");
        SCENE_LOG_INFO("SCENE CLEANUP: Starting comprehensive cleanup...");
        SCENE_LOG_INFO("========================================");

        // ---- 1. Sahne verilerini s�f�rla ----
        scene.world.clear();
        scene.lights.clear();
        scene.animationDataList.clear();
        scene.boneData.clear();

        // ---- 1b. Clear per-model animator caches (prevents bone corruption) ----
        for (auto& ctx : scene.importedModelContexts) {
            if (ctx.animator) {
                ctx.animator->clear();
            }
            if (ctx.graph) {
                ctx.graph.reset();
            }
            ctx.members.clear();
        }
        scene.importedModelContexts.clear();

        // ---- 1c. Clear renderer's cached bone matrices ----
        this->finalBoneMatrices.clear();

        scene.camera = nullptr;
        scene.bvh = nullptr;
        scene.initialized = false;
        SCENE_LOG_INFO("[SCENE CLEANUP] Scene data structures cleaned.");

        update_progress(5, "Clearing materials...");

        // ---- 2. MaterialManager'� temizle ----
        size_t material_count_before = MaterialManager::getInstance().getMaterialCount();
        MaterialManager::getInstance().clear();
        SCENE_LOG_INFO("[MATERIAL CLEANUP] MaterialManager cleared: " + std::to_string(material_count_before) + " materials removed.");

        // ---- 3. CPU Texture Cache'leri temizle ----
        assimpLoader.clearTextureCache();

        // ---- 4. GPU OptiX Texture'lar�n� temizle ----
        if (g_hasOptix && backend) {
            try {
                Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(backend);
                OptixWrapper* optix_gpu = optixBackend ? optixBackend->getOptixWrapper() : nullptr;
                if (optix_gpu) optix_gpu->destroyTextureObjects();
                SCENE_LOG_INFO("[GPU CLEANUP] OptiX texture objects destroyed.");
            }
            catch (std::exception& e) {
                SCENE_LOG_WARN("[GPU CLEANUP] Exception during texture cleanup: " + std::string(e.what()));
            }
        }

        SCENE_LOG_INFO("========================================");
        SCENE_LOG_INFO("SCENE CLEANUP: Completed successfully!");
        SCENE_LOG_INFO("========================================");
    }
    else {
        update_progress(0, "Appending to scene...");
        SCENE_LOG_INFO("Appending model to existing scene (no cleanup)");
    }

    update_progress(10, "Loading model file...");
    SCENE_LOG_INFO("Starting scene creation from: " + model_path);

    std::filesystem::path path(model_path);
    baseDirectory = path.parent_path().string() + "/";
    SCENE_LOG_INFO("Base directory set to: " + baseDirectory);

    // ---- 1. Geometri ve animasyon y�kle ----
    update_progress(15, "Loading geometry & animations...");
    SCENE_LOG_INFO("Loading model geometry and animations...");

    // Create a dedicated loader for this import to keep the aiScene alive
    auto newLoader = std::make_shared<AssimpLoader>();
    auto [loaded_triangles, loaded_animations, loaded_bone_data] = newLoader->loadModelToTriangles(model_path, nullptr, import_prefix);

    // Store the context
    SceneData::ImportedModelContext modelCtx;
    modelCtx.loader = newLoader;
    modelCtx.importName = newLoader->currentImportName;
    modelCtx.hasAnimation = (newLoader->getScene() && newLoader->getScene()->mNumAnimations > 0);
    modelCtx.globalInverseTransform = loaded_bone_data.globalInverseTransform;
    scene.importedModelContexts.push_back(modelCtx);

    update_progress(40, "Processing triangles...");

    if (loaded_triangles.empty()) {
        SCENE_LOG_ERROR("No triangle data, scene loading failed: " + model_path);
        SCENE_LOG_ERROR("Please provide a valid model file.");
    }
    else {
        // --- MERGE ANIMATION DATA & BONES ---
        // Verify bone/animation usage
        bool hasBones = !loaded_bone_data.boneNameToIndex.empty();

        // Calculate Offset for Bone Indices (Append Mode)
        unsigned int boneIndexOffset = 0;
        if (append) {
            boneIndexOffset = static_cast<unsigned int>(scene.boneData.boneNameToIndex.size());
        }
        else {
            // New Scene - already cleared in Step 0, but ensure boneData is fresh
            scene.boneData.boneNameToIndex.clear();
            scene.boneData.boneOffsetMatrices.clear();
            scene.boneData.boneNameToNode.clear();
        }

        // 1. Update Triangle Bone Indices with Offset
        if (hasBones && boneIndexOffset > 0) {
            for (auto& tri : loaded_triangles) {
                if (tri->hasSkinData()) {
                    // Access bone weights via accessor which returns reference to internal data
                    auto& vertexWeightsList = tri->getVertexBoneWeights();
                    for (auto& vertexWeights : vertexWeightsList) {
                        for (auto& bw : vertexWeights) {
                            bw.first += boneIndexOffset; // .first is the bone index
                        }
                    }
                }
            }
        }

        // 2. Merge Bone Data
        if (hasBones) {
            for (const auto& [name, id] : loaded_bone_data.boneNameToIndex) {
                scene.boneData.boneNameToIndex[name] = id + boneIndexOffset;
            }
            // Merge Offset Matrices and Node Pointers
            scene.boneData.boneOffsetMatrices.insert(loaded_bone_data.boneOffsetMatrices.begin(), loaded_bone_data.boneOffsetMatrices.end());
            scene.boneData.boneNameToNode.insert(loaded_bone_data.boneNameToNode.begin(), loaded_bone_data.boneNameToNode.end());
            scene.boneData.perModelInverses.insert(loaded_bone_data.perModelInverses.begin(), loaded_bone_data.perModelInverses.end());
            scene.boneData.boneParents.insert(loaded_bone_data.boneParents.begin(), loaded_bone_data.boneParents.end());
            scene.boneData.boneDefaultTransforms.insert(loaded_bone_data.boneDefaultTransforms.begin(), loaded_bone_data.boneDefaultTransforms.end());

            // Only set global inverse if it's the first model or handle separately? 
            // It seems unused for skinning (offset matrix handles it), so valid to leave or overwrite.
            if (!append) scene.boneData.globalInverseTransform = loaded_bone_data.globalInverseTransform;

            // CRITICAL: Rebuild reverse lookup after merge for O(1) index->name queries
            scene.boneData.rebuildReverseLookup();
        }

        // 3. Merge Animations
        if (append) {
            scene.animationDataList.insert(scene.animationDataList.end(), loaded_animations.begin(), loaded_animations.end());
        }
        else {
            scene.animationDataList = loaded_animations;
        }

        SCENE_LOG_INFO("Successfully loaded triangles: " + std::to_string(loaded_triangles.size()));
        SCENE_LOG_INFO("Loaded animations: " + std::to_string(loaded_animations.size()));
        SCENE_LOG_INFO("Total Bones (Merged): " + std::to_string(scene.boneData.boneNameToIndex.size()));
    }

    update_progress(45, "Adding triangles to scene...");
    SCENE_LOG_INFO("Adding triangles to scene world...");

    // Add triangles to scene - animation is handled via TransformHandle and skinning
    // NOTE: AnimatedObject wrappers removed (were unused, wasted memory)
    // Add triangles to scene and model context members
    for (const auto& tri : loaded_triangles) {
        scene.world.add(tri);
        modelCtx.members.push_back(tri);
    }
    SCENE_LOG_INFO("Added " + std::to_string(loaded_triangles.size()) + " triangles to scene member list.");

    // Initialize animation system for the new model
    if (modelCtx.hasAnimation) {
        initializeAnimationSystem(scene);
        // Settle bone matrices for frame 0, applying per-model inverses
        updateAnimationWithGraph(scene, 0.0f, true);
    }

    // ---- 2. Kamera ve ���k verisi ----
    update_progress(55, "Loading camera & lights...");
    SCENE_LOG_INFO("Loading camera and lighting data...");

    // Get new cameras and lights from loaded model using NEW loader
    auto new_lights = newLoader->getLights();
    auto new_cameras = newLoader->getCameras();  // Get ALL cameras

    // Handle cameras: Add all to the list
    if (append) {
        // Append mode: Add new cameras but keep active camera
        for (auto& cam : new_cameras) {
            if (cam) {
                cam->update_camera_vectors();
                scene.cameras.push_back(cam);
                SCENE_LOG_INFO("Append mode: Added camera (total: " + std::to_string(scene.cameras.size()) + ")");
            }
        }
        // If no camera was set before, set the first one as active
        if (!scene.camera && !scene.cameras.empty()) {
            scene.setActiveCamera(0);
        }
    }
    else {
        // New scene: Replace camera list
        scene.cameras.clear();
        for (auto& cam : new_cameras) {
            if (cam) {
                cam->save_initial_state();
                cam->update_camera_vectors();
                scene.cameras.push_back(cam);
            }
        }

        // If no cameras from model, create default
        if (scene.cameras.empty()) {
            auto new_camera = newLoader->getDefaultCamera();
            if (new_camera) {
                new_camera->save_initial_state();
                new_camera->update_camera_vectors();
                scene.cameras.push_back(new_camera);
            }
        }

        // Set first camera as active
        if (!scene.cameras.empty()) {
            scene.setActiveCamera(0);
            SCENE_LOG_INFO("Loaded " + std::to_string(scene.cameras.size()) + " camera(s). Active: Camera #0");
        }
        else {
            SCENE_LOG_WARN("No camera found in model.");
        }
    }

    // Handle lights: In append mode, merge with existing lights
    if (append) {
        // Append new lights to existing ones
        for (auto& light : new_lights) {
            scene.lights.push_back(light);
        }
        SCENE_LOG_INFO("Append mode: Added " + std::to_string(new_lights.size()) + " new lights (total: " + std::to_string(scene.lights.size()) + ")");
    }
    else {
        // Replace lights
        scene.lights = new_lights;
        SCENE_LOG_INFO("Loaded lights: " + std::to_string(scene.lights.size()));
    }

    // CRITICAL: Sync World Sun with first Directional Light from import/new project
    if (!append) { // Only force sync on new scene load, not append
        for (const auto& light : scene.lights) {
            if (light->type() == LightType::Directional) {
                // FORCE Default Sun Intensity to 10.0 (Override file default which is often too low, e.g. 2.4/pi)
                // Also ensures World Sun and Directional Light are coupled at start.
                light->intensity = 10.0f;

                world.setSunDirection(-light->direction);
                world.setSunIntensity(light->intensity);
                SCENE_LOG_INFO("World Sun synced with imported Directional Light (Forced Intensity: 10.0).");
                break; // Only sync to the first one
            }
        }
    }
    // ...
    // Note: OptiX conversion below (in original code) referenced 'assimpLoader' which was the member.
    // We must update that block too, but replacing only up to line 1335 handles the loading/merging logic.
    // The OptiX block is below 1335. I should include it in replacement range or do another replace.
    // The instruction requested updating create_scene. I'll replace the block covering loading to lighting.

    // BUT wait, I need to check if 'assimpLoader.convertTrianglesToOptixData' is called later.
    // Yes, line 1364. That uses member assimpLoader. Since it's a stateless helper (except texture cache maybe?), it MIGHT be okay?
    // BUT convertTrianglesToOptixData uses `MaterialManager` and triangle data. It seems stateless.
    // However, it's safer to use `newLoader`.

    // I will replace up to line 1400.

    // Force initial animation synchronization (poses correctly + updates CPU vertices)
    // CRITICAL: Must happen BEFORE BVH build so BVH sees correctly posed vertices!
    if (!scene.animationDataList.empty() && !scene.boneData.boneNameToIndex.empty()) {
        SCENE_LOG_INFO("[SceneCreation] Forcing initial pose sync for " + std::to_string(scene.boneData.getBoneCount()) + " bones.");
        updateAnimationWithGraph(scene, 0.0f, true); // true = apply_cpu_skinning
    }

    //  Selectable BVH (Embree or in-house BVH)
    update_progress(60, "Building BVH structure...");
    SCENE_LOG_INFO("Building BVH structure...");
    if (use_embree) {
        auto embree_bvh = std::make_shared<EmbreeBVH>();
        embree_bvh->build(scene.world.objects);
        scene.bvh = embree_bvh;
        SCENE_LOG_INFO("[Embree] BVH structure built successfully.");
    }
    else {
        scene.bvh = std::make_shared<ParallelBVHNode>(scene.world.objects, 0, scene.world.size(), 0.0f, 1.0f);
        SCENE_LOG_INFO("[RayTrophi: RT_BVH]  structure built successfully.");
    }

    update_progress(75, "Setting up GPU rendering...");

    // ---- 3. GPU OptiX setup ----
    // CRITICAL: In append mode, SKIP GPU build here!
    // The caller (ProjectManager::importModel) will call rebuildOptiXGeometry() 
    // which properly builds from ALL triangles in the scene.
    // If we build here with only loaded_triangles, we destroy the existing 
    // materials' texture handles, causing the second object to use first object's textures!
    if (g_hasOptix && backend && !append)
    {
        try
        {
            update_progress(78, "Creating OptiX geometry...");
            SCENE_LOG_INFO("OptiX GPU detected. Creating OptiX geometry data...");
            // Use newLoader here
            OptixGeometryData optix_data = newLoader->convertTrianglesToOptixData(loaded_triangles);
            SCENE_LOG_INFO("Converting " + std::to_string(loaded_triangles.size()) + " triangles to OptiX format.");

            Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(backend);
            OptixWrapper* optix_gpu = optixBackend ? optixBackend->getOptixWrapper() : nullptr;

            update_progress(82, "Validating materials...");
            if (optix_gpu) optix_gpu->validateMaterialIndices(optix_data);
            SCENE_LOG_INFO("Material indices validated.");

            update_progress(85, "Building OptiX acceleration...");
            if (optix_gpu) optix_gpu->buildFromData(optix_data);
            SCENE_LOG_INFO("OptiX BVH and acceleration structures built.");

            update_progress(90, "Configuring OptiX camera...");
            if (scene.camera) {
                SCENE_LOG_INFO("Setting up OptiX camera parameters...");
                Backend::CameraParams cp;
                cp.origin = scene.camera->lookfrom;
                cp.lookAt = scene.camera->lookat;
                cp.up = scene.camera->vup;
                cp.fov = scene.camera->vfov;
                cp.aperture = scene.camera->aperture;
                cp.focusDistance = scene.camera->focus_dist;
                cp.aspectRatio = scene.camera->aspect;
                backend->setCamera(cp);
                SCENE_LOG_INFO("OptiX camera configured successfully.");
            }

            update_progress(93, "Setting up OptiX lights...");
            if (!scene.lights.empty()) {
                SCENE_LOG_INFO("Configuring " + std::to_string(scene.lights.size()) + " lights for OptiX...");
                backend->setLights(scene.lights);
                SCENE_LOG_INFO("OptiX light parameters set successfully.");
            }

            // Consistently sync all volumes using unified logic
            VolumetricRenderer::syncVolumetricData(scene, m_backend);
            update_progress(96, "Finalizing world environment...");
            // backend->setBackgroundColor(scene.background_color);
            // Do not force COLOR mode - preserve existing mode set by createDefaultScene or UI
            // this->world.setMode(WORLD_MODE_COLOR); 

            // Only update solid color provided by scene if we are in Color mode, 
            // OR if we want to ensure scene.background_color is synced.
            // But for default scene, we want Nishita.
            if (this->world.getMode() == WORLD_MODE_COLOR) {
                this->world.setColor(scene.background_color);
            }
            if (optix_gpu) optix_gpu->setWorld(this->world.getGPUData());

            SCENE_LOG_INFO("World environment set for OptiX rendering.");
        }
        catch (std::exception& e)
        {
            SCENE_LOG_ERROR(std::string("OptiX exception occurred: ") + e.what());
            SCENE_LOG_WARN("Falling back to CPU-only rendering.");
            g_hasOptix = false;
        }
    }
    else
    {
        if (!g_hasOptix) {
            SCENE_LOG_INFO("OptiX not available. Using CPU-only path.");
        }
        else {
            SCENE_LOG_INFO("OptiX disabled or not initialized. Using CPU-only path.");
        }
    }

    // ---- 4. Son bilgiler ----
    update_progress(100, "Complete!");
    SCENE_LOG_INFO("Scene creation completed successfully.");
    SCENE_LOG_INFO("Scene info - Triangles: " + std::to_string(loaded_triangles.size()) +
        ", Lights: " + std::to_string(scene.lights.size()) +
        ", Animations: " + std::to_string(scene.animationDataList.size()));

    scene.initialized = true;
    SCENE_LOG_INFO("Scene initialization flag set to true.");
}


std::uniform_int_distribution<> dis_width(0, image_width - 1);
std::uniform_int_distribution<> dis_height(0, image_height - 1);



void Renderer::apply_normal_map(HitRecord& rec) {
    if (!rec.material) {
        return;
    }

    if (rec.material->has_normal_map()) {
        Vec3 tangent = rec.tangent;
        Vec3 bitangent = rec.bitangent;

        if (!rec.has_tangent) {
            create_coordinate_system(rec.normal, tangent, bitangent);
            tangent = (tangent - rec.normal * Vec3::dot(rec.normal, tangent));
            bitangent = Vec3::cross(rec.normal, tangent);
            if (Vec3::dot(Vec3::cross(tangent, bitangent), rec.normal) < 0.0f) {
                bitangent = -bitangent;
            }
        }

        Vec3 normal_from_map = rec.material->get_normal_from_map(rec.u, rec.v);
        normal_from_map = normal_from_map * 2.0 - Vec3(1.0, 1.0, 1.0);

        float normal_strength = rec.material->get_normal_strength();
        normal_from_map.x *= normal_strength;
        normal_from_map.y *= normal_strength;

        Mat3x3 TBN(tangent, bitangent, rec.normal);
        rec.interpolated_normal = (TBN * normal_from_map);
        //rec.interpolated_normal = (rec.interpolated_normal + 0.5*rec.normal).normalize();
    }
    else {
        rec.interpolated_normal = rec.normal;
    }
}

void Renderer::create_coordinate_system(const Vec3& N, Vec3& T, Vec3& B) {
    Vec3 N_norm = N.normalize();

    // E�er normal z eksenine paralelse, �ok k���k d�z y�zeyler i�in �zel durum
    if (N_norm.z < -0.999999f) {
        T = Vec3(0, -1, 0);  // Ters y�nlendirilmi� bir tangent
        B = Vec3(-1, 0, 0);
    }
    else {
        // Normalden tangent ve bitangent hesaplamas�
        float a = 1.0f / (1.0f + N_norm.z);
        float b = -N_norm.x * N_norm.y * a;

        // Daha hassas bir hesaplama, d�z y�zeylerdeki ters d�nme sorunu engellenebilir
        T = Vec3(1.0f - N_norm.x * N_norm.x * a, b, -N_norm.x);
        B = Vec3(b, 1.0f - N_norm.y * N_norm.y * a, -N_norm.y);

        // D�z y�zeylerde y�nleri do�ru tutmak i�in k���k d�zeltme
        if (std::abs(N_norm.z) > 0.9999f) {
            T = Vec3(1.0f, 0.0f, 0.0f);  // x y�n�yle tangent d�zeltmesi
            B = Vec3(0.0f, 1.0f, 0.0f);  // y y�n�yle bitangent d�zeltmesi
        }
    }
}

void Renderer::initialize_halton_cache() {
    halton_cache = std::make_unique<float[]>(MAX_DIMENSIONS * MAX_SAMPLES_HALTON);

    for (int d = 0; d < MAX_DIMENSIONS; ++d) {
        int base = (d == 0) ? 2 : 3;
        for (size_t i = 0; i < MAX_SAMPLES_HALTON; ++i) {
            // Tek boyutlu array'de 2D array gibi indeksleme
            halton_cache[d * MAX_SAMPLES_HALTON + i] = halton(i, base);
        }
    }
}

float Renderer::get_halton_value(size_t index, int dimension) {
    if (dimension < 0 || dimension >= MAX_DIMENSIONS ||
        index >= MAX_SAMPLES_HALTON) {
        return halton(index, dimension == 0 ? 2 : 3);
    }

    return halton_cache[dimension * MAX_SAMPLES_HALTON + index];
}

float Renderer::halton(int index, int base) {
    float r = 0;
    float f = 1;
    int i = index;

    while (i > 0) {
        f = f / base;
        r = r + f * (i % base);
        i = i / base;
    }

    return r;
}

Vec2 Renderer::stratified_halton(int x, int y, int sample_index, int samples_per_pixel) {
    // Daha iyi da��l�m i�in perm�tasyon ekliyoruz
    const uint32_t pixel_hash = (x * 73856093) ^ (y * 19349663); // Basit bir hash fonksiyonu
    const uint32_t sample_hash = sample_index * 83492791;

    // Halton dizisinde farkl� offsetler kullan�yoruz
    const int base_index = (pixel_hash + sample_hash) % MAX_SAMPLES_HALTON;

    // Farkl� asal say� tabanlar� kullanarak daha iyi da��l�m
    const float u = halton_cache[base_index];                     // Taban 2
    const float v = halton_cache[(base_index + MAX_SAMPLES_HALTON / 2) % MAX_SAMPLES_HALTON]; // Taban 3

    // Stratifikasyon eklemek i�in jitter
    const float jitter_u = (rand() / (float)RAND_MAX) * 0.8f / samples_per_pixel;
    const float jitter_v = (rand() / (float)RAND_MAX) * 0.8f / samples_per_pixel;

    return Vec2(
        (x + u + jitter_u) / image_width,
        (y + v + jitter_v) / image_height
    );
}



float Renderer::luminance(const Vec3& color) {
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

// --- Ak�ll� ���k se�imi ---
int Renderer::pick_smart_light(const std::vector<std::shared_ptr<Light>>& lights, const Vec3& hit_position) {
    int light_count = (int)lights.size();
    if (light_count == 0) return -1;

    // --- 1. Directional light varsa %33 ihtimalle se� ---
    for (int i = 0; i < light_count; i++) {
        if (!lights[i]->visible) continue; // Skip invisible lights
        if (lights[i]->type() == LightType::Directional) {
            if (Vec3::random_float() < 0.33) {
                directional_pick_count++;
                return i;
            }
        }
    }

    // --- 2. T�m ���k t�rlerinden a��rl�kl� se�im (GPU ile uyumlu) ---
    std::vector<float> weights(light_count, 0.0f);
    float total_weight = 0.0f;

    for (int i = 0; i < light_count; i++) {
        if (!lights[i]->visible) {
            weights[i] = 0.0f;
            continue; // Skip invisible lights
        }
        Vec3 delta = lights[i]->position - hit_position;
        float distance = std::max(1.0f, delta.length());
        float falloff = 1.0f / (distance * distance);
        float intensity = luminance(lights[i]->color * lights[i]->intensity);

        if (lights[i]->type() == LightType::Point) {
            weights[i] = falloff * intensity;
        }
        else if (lights[i]->type() == LightType::Area) {
            // GPU ile uyumlu: area etkisi
            auto areaLight = std::dynamic_pointer_cast<AreaLight>(lights[i]);
            if (areaLight) {
                float area = areaLight->getWidth() * areaLight->getHeight();
                weights[i] = falloff * intensity * std::min(area, 10.0f);
            }
        }
        else if (lights[i]->type() == LightType::Spot) {
            weights[i] = falloff * intensity * 0.8f;
        }
        else {
            weights[i] = 0.0f;
        }

        total_weight += weights[i];
    }

    // --- E�er a��rl�k yoksa fallback rastgele se�im ---
    if (total_weight < 1e-6f) {
        return std::clamp(int(Vec3::random_float() * light_count), 0, light_count - 1);
    }

    // --- Weighted se�im ---
    float r = Vec3::random_float() * total_weight;
    float accum = 0.0f;
    for (int i = 0; i < light_count; i++) {
        accum += weights[i];
        if (r <= accum) {
            return i;
        }
    }

    // --- G�venlik fallback ---
    return std::clamp(int(Vec3::random_float() * light_count), 0, light_count - 1);
}

Vec3 Renderer::calculate_direct_lighting_single_light(
    const Hittable* bvh,
    const std::shared_ptr<Light>& light,
    const HitRecord& rec,
    const Vec3& normal,
    const Ray& r_in
) {
    Vec3 direct_light(0.0f);

    Vec3 hit_point = rec.point;
    Vec2 uv = Vec2(rec.u, rec.v);

    // Malzeme �zellikleri
    // Malzeme �zellikleri (Blending Support)
    Vec3 albedo;
    float metallic;
    float roughness;
    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.03f;

    if (rec.use_custom_data) {
        albedo = rec.custom_albedo;
        metallic = rec.custom_metallic;
        roughness = rec.custom_roughness;
        clearcoat = rec.custom_clearcoat;
        clearcoatRoughness = rec.custom_clearcoat_roughness;
    } else {
        albedo = rec.material->getPropertyValue(rec.material->albedoProperty, uv);
        metallic = rec.material->getPropertyValue(rec.material->metallicProperty, uv).z;
        roughness = rec.material->getPropertyValue(rec.material->roughnessProperty, uv).y;
        
        // Try to get clearcoat from material
        if (auto pMat = std::dynamic_pointer_cast<PrincipledBSDF>(rec.material)) {
            clearcoat = pMat->clearcoat;
            clearcoatRoughness = pMat->clearcoatRoughness;
        }
    }
    Vec3 F0 = Vec3::lerp(Vec3(0.04f), albedo, metallic);

    Vec3 V = -r_in.direction.normalize();
    Vec3 N = normal;

    Vec3 light_sample, to_light, Li;
    float light_distance = 1.0f;
    Vec3 L;

    float pdf_light = 1.0f;
    float pdf_light_select = 1.0f;
    float attenuation = 1.0f;

    // --- Light sampling ---
    if (auto directional = std::dynamic_pointer_cast<DirectionalLight>(light)) {
        L = -directional->random_point();
        light_sample = hit_point + L * 1e8f;
        to_light = L;
        attenuation = 1.0f; // Directional light falloff yok
        light_distance = std::numeric_limits<float>::infinity();
        Li = directional->getIntensity(hit_point, light_sample);
    }
    else if (auto point = std::dynamic_pointer_cast<PointLight>(light)) {
        light_sample = point->random_point();
        to_light = light_sample - hit_point;
        light_distance = to_light.length();
        L = to_light / light_distance;

        // D�ZELTME: PointLight s�n�f� getIntensity i�inde zaten falloff (1/d^2) uyguluyor.
        // Burada tekrar uygularsak ���k �ok zay�fl�yor (1/d^4).
        attenuation = 1.0f;

        // Point Light Specific Boost: Global �arpan kald�r�ld�, sadece Point Light 10 kat g��lendirildi.
        Li = point->getIntensity(hit_point, light_sample) * attenuation;

        float area = 4.0f * M_PI * point->getRadius() * point->getRadius();
        pdf_light = (1.0f / area) * pdf_light_select;
    }
    else if (auto areaLight = std::dynamic_pointer_cast<AreaLight>(light)) {
        // GPU ile uyumlu AreaLight sampling
        light_sample = areaLight->random_point();
        to_light = light_sample - hit_point;
        light_distance = to_light.length();
        L = to_light / light_distance;

        // Light normal (cross of u and v vectors)
        Vec3 light_normal = Vec3::cross(areaLight->getU(), areaLight->getV()).normalize();
        float cos_light = std::fmax(Vec3::dot(-L, light_normal), 0.0f);
        attenuation = cos_light / (light_distance * light_distance);

        Li = areaLight->getIntensity(hit_point, light_sample) * attenuation;

        float area = areaLight->getWidth() * areaLight->getHeight();
        pdf_light = (1.0f / std::fmax(area, 1e-4f)) * pdf_light_select;
    }
    else if (auto spotLight = std::dynamic_pointer_cast<SpotLight>(light)) {
        // GPU ile uyumlu SpotLight sampling
        light_sample = spotLight->position;
        to_light = light_sample - hit_point;
        light_distance = to_light.length();
        L = to_light / light_distance;

        // Spot cone falloff
        float cos_theta = Vec3::dot(-L, spotLight->direction.normalize());
        float angleDeg = spotLight->getAngleDegrees();
        float angleRad = angleDeg * (M_PI / 180.0f);
        float inner_cos = cosf(angleRad * 0.8f);
        float outer_cos = cosf(angleRad);

        float falloff = 0.0f;
        if (cos_theta > inner_cos) falloff = 1.0f;
        else if (cos_theta > outer_cos) {
            float t = (cos_theta - outer_cos) / (inner_cos - outer_cos + 1e-6f);
            falloff = t * t;
        }

        if (falloff < 1e-4f) return direct_light;

        attenuation = falloff / (light_distance * light_distance);
        Li = spotLight->getIntensity(hit_point, light_sample) * attenuation;

        float solid_angle = 2.0f * M_PI * (1.0f - outer_cos);
        pdf_light = (1.0f / std::fmax(solid_angle, 1e-4f)) * pdf_light_select;
    }
    else {
        return direct_light;
    }

    // --- Shadow ---
    // GPU ile uyumlu shadow bias (eski: 0.0001f -> self-shadowing yapabilir)
    // Volumetric Shadow Logic (Transparent Shadows)
    Ray shadow_ray_current(hit_point + N * 0.001f, L);
    float remaining_dist = light_distance;
    Vec3 shadow_transmittance(1.0f); // Changed from float to Vec3 for colored shadows
    int shadow_layers = 0;
    
    // Check hair shadows first (hair casts strong shadows on meshes)
    if (hairSystem.getTotalStrandCount() > 0 && !hairSystem.isBVHDirty()) {
        Hair::HairHitInfo hairShadowHit;
        Vec3 hairShadowOrigin = shadow_ray_current.origin;
        Vec3 hairShadowDir = shadow_ray_current.direction;
        float hairShadowDist = std::min(remaining_dist, 100.0f); // Limit to 100 units
        
        // Trace through multiple hair strands for accumulated shadow
        int hairShadowSamples = 0;
        while (hairShadowSamples < 8 && hairShadowDist > 0.01f) {
            if (hairSystem.intersect(hairShadowOrigin, hairShadowDir, 0.002f, hairShadowDist, hairShadowHit)) {
                // Hair casts stronger shadows - higher opacity per strand
                float hairOpacity = 0.5f + 0.2f * (1.0f - hairShadowHit.v); // 50-70% per strand
                shadow_transmittance = shadow_transmittance * (1.0f - hairOpacity);
                
                // Continue tracing through hair
                hairShadowOrigin = hairShadowHit.position + hairShadowDir * 0.003f;
                hairShadowDist -= (hairShadowHit.t + 0.003f);
                hairShadowSamples++;
                
                if (shadow_transmittance.max_component() < 0.01f) break;
            } else {
                break;
            }
        }
    }


    while (remaining_dist > 0.001f && shadow_layers < 4) {
        HitRecord shadow_rec;
        if (bvh->hit(shadow_ray_current, 0.001f, remaining_dist, shadow_rec)) {

            // Check if blocker is a Volume (VDB or Unified Gas)
            const VDBVolume* vdb = shadow_rec.vdb_volume;
            int live_vol_id = -1;
            std::shared_ptr<VolumeShader> vol_shader = nullptr;
            Matrix4x4 inv_transform = Matrix4x4::identity();
            float den_scale = 1.0f;

            if (vdb) {
                live_vol_id = vdb->getVDBVolumeID();
                vol_shader = vdb->volume_shader;
                inv_transform = vdb->getInverseTransform();
                den_scale = vdb->density_scale;
            }
            else if (shadow_rec.gas_volume && shadow_rec.gas_volume->render_path == GasVolume::VolumeRenderPath::VDBUnified) {
                live_vol_id = shadow_rec.gas_volume->live_vdb_id;
                vol_shader = shadow_rec.gas_volume->getShader();
                if (shadow_rec.gas_volume->getTransformHandle()) {
                    Matrix4x4 m = shadow_rec.gas_volume->getTransformHandle()->getFinal();
                    Vec3 gsize = shadow_rec.gas_volume->getSettings().grid_size;
                    if (gsize.x > 0 && gsize.y > 0 && gsize.z > 0) {
                        m = m * Matrix4x4::scaling(Vec3(1.0f / gsize.x, 1.0f / gsize.y, 1.0f / gsize.z));
                    }
                    inv_transform = m.inverse();
                }
                den_scale = 1.0f;
            }

            if (live_vol_id >= 0) {
                // Get intersection interval
                float t_enter, t_exit;
                bool hit_box = false;

                if (vdb) {
                    hit_box = vdb->intersectTransformedAABB(shadow_ray_current, 0.001f, remaining_dist, t_enter, t_exit);
                }
                else {
                    // Manual box check for unified gas
                    AABB box; shadow_rec.gas_volume->bounding_box(0, 0, box);
                    hit_box = box.hit_interval(shadow_ray_current, 0.001f, remaining_dist, t_enter, t_exit);
                }

                if (hit_box) {
                    if (t_enter < 0.001f) t_enter = 0.001f;
                    if (t_exit > remaining_dist) t_exit = remaining_dist;

                    float shadow_step = vol_shader ? vol_shader->quality.step_size * 4.0f : 0.4f;
                    if (shadow_step < 0.05f) shadow_step = 0.05f;

                    float density_scale = (vol_shader ? vol_shader->density.multiplier : 1.0f) * den_scale;
                    float shadow_strength = vol_shader ? vol_shader->quality.shadow_strength : 1.0f;

                    float t = t_enter;
                    auto& mgr = VDBVolumeManager::getInstance();
                    t += ((float)rand() / RAND_MAX) * shadow_step;

                    while (t < t_exit) {
                        Vec3 p = shadow_ray_current.at(t);
                        Vec3 local_p = inv_transform.transform_point(p);
                        float density = mgr.sampleDensityCPU(live_vol_id, local_p.x, local_p.y, local_p.z);

                        if (density < 0.01f) density = 0.0f;
                        if (density > 0.0f) {
                            float s_sigma_s = density * density_scale * (vol_shader ? vol_shader->scattering.coefficient : 1.0f);
                            float s_sigma_a = density * (vol_shader ? vol_shader->absorption.coefficient : 0.1f);
                            float sigma_t = (s_sigma_s + s_sigma_a) * shadow_strength;
                            shadow_transmittance = shadow_transmittance * std::exp(-sigma_t * shadow_step);
                        }
                        if (shadow_transmittance.max_component() < 0.01f) break;
                        t += shadow_step;
                    }
                }

                if (shadow_transmittance.max_component() < 0.01f) return direct_light;

                float advance = t_exit + 0.001f;
                shadow_ray_current = Ray(shadow_ray_current.at(advance), L);
                remaining_dist -= advance;
                shadow_layers++;
            }
            else {
                // Check for Generic Mesh Volume Traversal
                bool is_volumetric = false;
                if (shadow_rec.material && shadow_rec.material->type() == MaterialType::Volumetric) {
                    is_volumetric = true;
                    auto vol = std::static_pointer_cast<Volumetric>(shadow_rec.material);

                    // Find exit point (assume convex/closed mesh for now)
                    Ray exit_ray(shadow_ray_current.at(shadow_rec.t + 0.001f), shadow_ray_current.direction);
                    HitRecord exit_rec;
                    float exit_dist = remaining_dist - shadow_rec.t;

                    // Trace to find the back side of this volume
                    // Note: Ideally we check if we hit the SAME object, but checking material is a good proxy
                    float t_vol_exit = 0.0f;
                    bool found_exit = false;

                    if (bvh->hit(exit_ray, 0.001f, exit_dist, exit_rec)) {
                        // If we hit something, assume it's the exit if it's the same material
                        // Or just treat the segment [enter, next_hit] as the volume
                        t_vol_exit = exit_rec.t;
                        found_exit = true;
                    }
                    else {
                        // Didn't hit anything within light distance -> Volume covers the rest of the path?
                        // Or we exited without hitting a backface (geometry error?). 
                        // For shadows, assume we exit at light distance.
                        t_vol_exit = exit_dist;
                    }

                    // Ray Marching Params
                    float step_size = vol->getStepSize();
                    float density_mult = vol->getDensity();
                    float t = 0.0f; // Relative to exit_ray origin (which is entry + epsilon)

                    // Jitter
                    float jitter = ((float)rand() / RAND_MAX) * step_size;
                    t += jitter;

                    // March
                    while (t < t_vol_exit) {
                        Vec3 p = exit_ray.at(t);
                        float d = vol->calculate_density(p);

                        if (d > 0.001f) {
                            float sigma_t = d * 1.0f; // Assuming extinction = density for shadows
                            shadow_transmittance = shadow_transmittance * std::exp(-sigma_t * step_size);
                        }

                        if (shadow_transmittance.max_component() < 0.01f) break;
                        t += step_size;
                    }

                    // Advance Shadow Ray Logic
                    if (found_exit) {
                        if (exit_rec.materialID == shadow_rec.materialID) {
                            // Hit Volume Backface: Advance PAST it
                            float total_advance = shadow_rec.t + t_vol_exit + 0.001f;
                            shadow_ray_current = Ray(shadow_ray_current.at(total_advance), shadow_ray_current.direction);
                            remaining_dist -= total_advance;
                        }
                        else {
                            // Hit Obstruction (different material): Advance TO it (just before)
                            // We want to hit this obstruction in the next loop iteration.
                            float total_advance = shadow_rec.t + t_vol_exit - 0.001f;
                            if (total_advance < 0.0f) total_advance = 0.0f; // Safety
                            shadow_ray_current = Ray(shadow_ray_current.at(total_advance), shadow_ray_current.direction);
                            remaining_dist -= total_advance;
                        }
                    }
                    else {
                        // Reached Light or end of trace without hitting anything
                        break;
                    }
                }

                if (!is_volumetric) {
                    // Check for Transparent Surface (Glass, Water, Alpha Cutout)
                    bool is_transparent = false;
                    Vec3 transmission_filter(1.0f);

                    if (shadow_rec.material) {
                        auto pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(shadow_rec.material);
                        if (pbsdf) {
                            Vec2 uv(shadow_rec.u, shadow_rec.v);
                            float tr = pbsdf->getTransmission(uv);
                            float op = pbsdf->get_opacity(uv);

                            if (tr > 0.001f || op < 0.999f) {
                                is_transparent = true;
                                Vec3 base_color = pbsdf->getPropertyValue(pbsdf->albedoProperty, uv);

                                // Simple Transmission approximation for shadows:
                                // Light passes through colored glass
                                // Mix based on opacity and transmission
                                Vec3 tf = (base_color);

                                // 1. Alpha Cutout (Leafs, Fences)
                                if (op < 0.999f) {
                                    // Corrected: Vec3::lerp is static.
                                    // Lerp between 1.0 (air) and MaterialColor (if transmitted) - Simplified approach
                                    transmission_filter = Vec3::lerp(Vec3(1.0f), transmission_filter, op);
                                }

                                // 2. Glass Transmission
                                if (tr > 0.001f) {
                                    // Corrected: No .pow member, do component-wise manually
                                    Vec3 bc = (base_color);
                                    Vec3 bc_pow(std::pow(bc.x, 0.5f), std::pow(bc.y, 0.5f), std::pow(bc.z, 0.5f));
                                    transmission_filter = transmission_filter * bc_pow;
                                }
                            }
                        }
                    }

                    if (is_transparent) {
                        shadow_transmittance *= transmission_filter;
                        if (shadow_transmittance.luminance() < 0.01f) return direct_light; // Absorbed

                        // Advance ray past the transparent surface
                        float advance = shadow_rec.t + 0.001f;
                        shadow_ray_current = Ray(shadow_ray_current.at(advance), L);
                        remaining_dist -= advance;
                        shadow_layers++;
                        continue; // Continue tracing
                    }
                    else {
                        // Opaque blocker found
                        return direct_light;
                    }
                }
            }
        }
        else {
            // No blocker found
            break;
        }
    }

    // Apply calculated transmittance to light intensity
    // shadow_transmittance is now Vec3
    if (shadow_transmittance.max_component() < 0.99f) {
        Li = Li * shadow_transmittance;
    }

    float NdotL = std::fmax(Vec3::dot(N, L), 0.0001f);


    // --- BRDF Hesab� (Specular + Diffuse) ---
    Vec3 H = (L + V).normalize();
    float NdotV = std::fmax(Vec3::dot(N, V), 0.0001f);
    float NdotH = std::fmax(Vec3::dot(N, H), 0.0001f);
    float VdotH = std::fmax(Vec3::dot(V, H), 0.0001f);

    float alpha = max(roughness * roughness, 0.01f);
    PrincipledBSDF psdf;
    // Specular bile�eni
    float D = psdf.DistributionGGX(N, H, roughness);
    float G = psdf.GeometrySmith(N, V, L, roughness);
    Vec3 F = psdf.fresnelSchlickRoughness(VdotH, F0, roughness);

    Vec3 specular = psdf.evalSpecular(N, V, L, F0, roughness);

    // Diffuse bile�eni - GPU ile uyumlu
    Vec3 F_avg = F0 + (Vec3(1.0f) - F0) / 21.0f;
    // GPU form�l�: k_d = (1 - F_avg) * (1 - metallic)
    Vec3 k_d = (Vec3(1.0f) - F_avg) * (1.0f - metallic);
    Vec3 diffuse = k_d * albedo / M_PI;

    // Toplam BRDF
    Vec3 brdf = diffuse + specular;

    // Clearcoat Contribution
    if (clearcoat > 0.001f) {
        psdf.clearcoatRoughness = clearcoatRoughness;
        // Check signature: computeClearcoat(V, L, N)
        // V is view vector (-ray.dir), L is light vector, N is normal
        Vec3 cc = psdf.computeClearcoat(V, L, N); 
        brdf = brdf + cc * clearcoat;
    }

    // --- MIS (Multiple Importance Sampling) ---
    // PDF BRDF hesapla
    Vec3 incoming = -L; // Light direction (incoming to surface)
    Vec3 outgoing = V;  // View direction
    float pdf_brdf_val = psdf.pdf(rec, incoming, outgoing);
    float pdf_brdf_val_mis = std::clamp(pdf_brdf_val, 0.01f, 5000.0f);

    float mis_weight = power_heuristic(pdf_light, pdf_brdf_val_mis);

    // I��k katk�s�
    // GPU form�l�: (f * Li * NdotL) * mis_weight
    Vec3 direct = brdf * Li * NdotL * mis_weight;

    return direct;
}


Vec3 Renderer::ray_color(const Ray& r, const Hittable* bvh,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color, int depth, int sample_index, const SceneData& scene) {

    // =========================================================================
    // UNIFIED RAY COLOR - Matches GPU ray_color.cuh exactly
    // =========================================================================

    Vec3f color(0.0f);
    Vec3f throughput(1.0f);
    Vec3f current_medium_absorb(0.0f); // Default: Air (no absorption)
    Ray current_ray = r;
    int transparent_hits = 0;
    float first_hit_t = -1.0f;
    float vol_trans_accum = 1.0f;
    float first_vol_t = -1.0f;

    int light_count = static_cast<int>(lights.size());

    // Pre-convert lights to unified format for this ray
    // Note: In production, this should be done once per frame, not per ray
    thread_local std::vector<UnifiedLight> unified_lights;
    if (unified_lights.size() != lights.size()) {
        unified_lights.clear();
        unified_lights.reserve(lights.size());
        for (const auto& light : lights) {
            unified_lights.push_back(toUnifiedLight(light));
        }
    }

    for (int bounce = 0; bounce < render_settings.max_bounces; ++bounce) {
        HitRecord rec;
        HitRecord solid_rec;
        bool hit_solid = false;
        bool hit_hair = false;
        Hair::HairHitInfo hairHit;

        bool hit_any = false;
        if (bvh) {
            hit_any = bvh->hit(current_ray, 0.001f, std::numeric_limits<float>::infinity(), rec, false);
        }
        
        // Check hair intersection (if hair system has strands)
        if (hairSystem.getTotalStrandCount() > 0) {
            float maxT = hit_any ? rec.t : std::numeric_limits<float>::infinity();
            hit_hair = hairSystem.intersect(
                current_ray.origin, current_ray.direction,
                0.001f, maxT, hairHit
            );

            // Hair hit - process it (simplified condition)
            if (hit_hair) {
                // [MODIFIED] Random variation support using Strand ID
                Hair::HairMaterialParams hairMat = hairHit.material; // Copy material for per-strand mod

                // ===================================================================
                // ROOT UV TEXTURE SAMPLING (Inherit color from scalp mesh)
                // ===================================================================
                if (hairMat.colorMode == Hair::HairMaterialParams::ColorMode::ROOT_UV_MAP) {

                    // --- Custom Independent Texture Support ---
                    // If the user has assigned a specific texture to the hair material itself,
                    // we use that INSTEAD of the mesh texture.

                    bool usedCustomTexture = false;

                    if (hairMat.customAlbedoTexture) {
                        // Sample the custom texture using root UVs
                        // We use the texture's get_color method directly
                        hairMat.color = hairMat.customAlbedoTexture->get_color(hairHit.rootUV.u, hairHit.rootUV.v);
                        hairMat.colorMode = Hair::HairMaterialParams::ColorMode::DIRECT_COLORING;
                        usedCustomTexture = true;
                    }

                    // Apply Roughness Map if exists
                    if (hairMat.customRoughnessTexture) {
                        // Roughness maps are usually grayscale, we take the Red channel or intensity
                        float rMap = hairMat.customRoughnessTexture->get_color(hairHit.rootUV.u, hairHit.rootUV.v).x;
                        hairMat.roughness *= rMap;
                        hairMat.radialRoughness *= rMap;
                    }

                    // Only proceed to Mesh Inheritance if we didn't use a custom albedo
                    if (!usedCustomTexture) {
                        auto& matMgr = MaterialManager::getInstance();
                        const auto& all_mats = matMgr.getAllMaterials();
                        bool textureFound = false;

                        // 1. FAST PATH: Use cached Material ID
                        int matID = hairHit.meshMaterialID;
                        if (matID >= 0 && matID != 0xFFFF && (size_t)matID < all_mats.size()) {
                            const auto& mat = all_mats[matID];
                            if (mat) {
                                if (mat->albedoProperty.texture) {
                                    hairMat.color = mat->albedoProperty.evaluate(hairHit.rootUV);
                                    textureFound = true;
                                }
                                else {
                                    // Material exists but has no texture -> take base color
                                    hairMat.color = mat->albedoProperty.color;
                                    textureFound = true; // technically found, just no texture
                                }
                            }
                        }

                        // 2. SLOW PATH (Fallback): Search geometry by name
                        // This covers legacy grooms or cases where ID sync failed
                        if (!textureFound) {
                            if (const Hair::HairGroom* groom = hairSystem.getGroom(hairHit.groomName)) {
                                std::string scalpName = groom->boundMeshName;
                                for (const auto& h : scene.world.objects) {
                                    if (auto tri = std::dynamic_pointer_cast<Triangle>(h)) {
                                        if (tri->getNodeName() == scalpName) {
                                            int fallbackID = tri->getMaterialID();
                                            if (fallbackID >= 0 && (size_t)fallbackID < all_mats.size()) {
                                                const auto& mat = all_mats[fallbackID];
                                                if (mat && mat->albedoProperty.texture) {
                                                    hairMat.color = mat->albedoProperty.evaluate(hairHit.rootUV);
                                                }
                                                else if (mat) {
                                                    hairMat.color = mat->albedoProperty.color;
                                                }
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        // Treat as direct coloring for BSDF evaluation
                        hairMat.colorMode = Hair::HairMaterialParams::ColorMode::DIRECT_COLORING;
                    }
                    // End if (!usedCustomTexture)
                } // End if (hairMat.colorMode == Hair::HairMaterialParams::ColorMode::ROOT_UV_MAP) 

        // Note: Tint is now applied inside HairBSDF::evaluate() as post-process
        // Do NOT modify hairMat.color here to avoid double-tinting

                if (hairMat.randomHue > 0.0f || hairMat.randomValue > 0.0f) {
                    uint32_t id = hairHit.strandID;
                    // Fast integer hash for stable randomness per strand
                    uint32_t h = id * 747796405u + 2891336453u;
                    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
                    float r1 = ((h >> 22u) ^ h) / 4294967296.0f; // [0, 1]

                    uint32_t h2 = id * 123456789u + 987654321u;
                    float r2 = (h2 & 0x00FFFFFF) / 16777216.0f; // [0, 1]

                    if (hairMat.colorMode == Hair::HairMaterialParams::ColorMode::MELANIN) {
                        // For Melanin mode, vary the physical parameters
                        hairMat.melanin = std::clamp(hairMat.melanin + (r1 - 0.5f) * hairMat.randomValue, 0.0f, 1.0f);
                        hairMat.melaninRedness = std::clamp(hairMat.melaninRedness + (r2 - 0.5f) * hairMat.randomHue, 0.0f, 1.0f);
                        
                        // Update the base color for the ambient/fallback term
                        Vec3 sigma = Hair::HairBSDF::melaninToAbsorption(hairMat.melanin, hairMat.melaninRedness);
                        hairMat.color = Vec3(
                            std::exp(-sigma.x * 0.5f),
                            std::exp(-sigma.y * 0.5f),
                            std::exp(-sigma.z * 0.5f)
                        );
                    } else {
                        // Random Brightness (Value) for Direct/Root UV modes
                        if (hairMat.randomValue > 0.0f) {
                            float vScale = 1.0f + (r1 - 0.5f) * hairMat.randomValue * 2.0f;
                            hairMat.color = hairMat.color * vScale;
                        }

                        // Random Hue (Shift) for Direct/Root UV modes
                        if (hairMat.randomHue > 0.0f) {
                            // Rodrigues rotation around Grey Axis (1,1,1)
                            float angle = (r2 - 0.5f) * hairMat.randomHue * 2.0f * 3.14159f;
                            float c = std::cos(angle);
                            float s = std::sin(angle);

                            Vec3 k(0.57735f); // 1/sqrt(3) normalized
                            Vec3& p = hairMat.color;
                            Vec3 crossP = Vec3::cross(k, p);
                            float dotP = Vec3::dot(k, p);

                            hairMat.color = p * c + crossP * s + k * dotP * (1.0f - c);
                        }
                    }

                    // Simple clamp
                    if (hairMat.color.x < 0) hairMat.color.x = 0;
                    if (hairMat.color.y < 0) hairMat.color.y = 0;
                    if (hairMat.color.z < 0) hairMat.color.z = 0;
                }

                Vec3 wo = -current_ray.direction;
                Vec3 baseHairColor = hairMat.color;


                // ===================================================================
                // FULL MARSCHNER HAIR SHADING WITH STRONG SHADOWS
                // ===================================================================

                Vec3 T = hairHit.tangent;
                Vec3 N = hairHit.normal; // Camera-facing normal for shadow offsets

                // Re-verify tangent if needed (Optional, usually hairHit.tangent is stable)
                if (T.length() < 0.1f) T = Vec3(0, 1, 0);

                // Get main light direction (sun or first light)
                Vec3 mainLightDir = Vec3(0.5f, 0.8f, 0.3f).normalize();
                Vec3 mainLightColor = Vec3(1.0f, 0.95f, 0.9f); // Warm sunlight
                float mainLightDist = 1e6f;

                if (!lights.empty()) {
                    auto& firstLight = lights[0];
                    if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(firstLight)) {
                        // DirectionalLight::getDirection() already returns the vector towards the light source (-direction)
                        // This is exactly what we need for L in shading and shadow tracing
                        mainLightDir = dl->getDirection(hairHit.position).normalize();
                        mainLightColor = dl->getIntensity(hairHit.position, Vec3(0));
                        mainLightDist = 1e6f;
                    }
                    else if (auto pl = std::dynamic_pointer_cast<PointLight>(firstLight)) {
                        Vec3 toLight = pl->getPosition() - hairHit.position;
                        mainLightDist = toLight.length();
                        mainLightDir = toLight / (mainLightDist + 1e-6f);
                        mainLightColor = pl->getIntensity(hairHit.position, pl->getPosition());
                    }
                }

                mainLightDir = mainLightDir.normalize();

                // ========================
                // DEEP SHADOW CALCULATION
                // ========================
                float totalShadow = 1.0f;
                {
                    // 1. Mesh Shadow (Solid occlusion)
                    Ray meshShadowRay(hairHit.position + N * 0.0002f, mainLightDir);
                    HitRecord mRec;
                    if (bvh && bvh->hit(meshShadowRay, 0.0002f, mainLightDist, mRec, true)) {
                        totalShadow = 0.0f;
                    }

                    // 2. Hair Transmission Shadow (Light filtering through strands)
                    if (totalShadow > 0.0f && !hairSystem.isBVHDirty()) {
                        Vec3 shadowOrigin = hairHit.position + N * 0.001f; // Larger offset to avoid self-hit
                        Hair::HairHitInfo sHit;
                        int hits = 0;
                        float shadowTraceDist = std::min(mainLightDist, 100.0f);

                        // Trace light through multiple strands for realistic deep shadows
                        int maxHits = 8;
                        while (hits < maxHits && shadowTraceDist > 0.01f) {
                            if (hairSystem.intersect(shadowOrigin, mainLightDir, 0.001f, shadowTraceDist, sHit)) {
                                totalShadow *= 0.4f; // Stronger shadow per strand for visibility
                                shadowOrigin = sHit.position + mainLightDir * 0.002f;
                                shadowTraceDist -= (sHit.t + 0.002f);
                                hits++;
                                if (totalShadow < 0.01f) { totalShadow = 0.0f; break; }
                            }
                            else {
                                break;
                            }
                        }
                    }
                }



                // ========================
                // PURE MARSCHNER SHADING
                // ========================
                // Pass longitudinal (v) and azimuthal (u) correctly
                // Final Color = BSDF * LightColor * Shadow + Ambient
                Vec3 bsdf = Hair::HairBSDF::evaluate(wo, mainLightDir, T, hairMat, hairHit.v, hairHit.u);
                
                // Physically plausible ambient: Small portion of sunlight as 'sky' contribution
                Vec3 hair_color = (bsdf * totalShadow * mainLightColor) + (baseHairColor * mainLightColor * 0.02f);




                // ========================
                // ADDITIONAL LIGHTS
                // ========================
                for (size_t li = 0; li < lights.size(); li++) {
                    const auto& light = lights[li];
                    Vec3 lightDir, lightPos;
                    float lightDist = 0.0f;
                    Vec3 Li(0.0f);

                    if (auto pl = std::dynamic_pointer_cast<PointLight>(light)) {
                        lightPos = pl->getPosition();
                        Vec3 toLight = lightPos - hairHit.position;
                        lightDist = toLight.length();
                        lightDir = toLight / lightDist;
                        Li = pl->getIntensity(hairHit.position, lightPos);
                    }
                    else if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(light)) {
                        if (li == 0) continue; // Skip main light (already processed)
                        lightDir = -dl->getDirection(hairHit.position);
                        lightDist = 1e6f;
                        Li = dl->getIntensity(hairHit.position, Vec3(0));
                    }
                    else if (auto al = std::dynamic_pointer_cast<AreaLight>(light)) {
                        lightPos = al->random_point();
                        Vec3 toLight = lightPos - hairHit.position;
                        lightDist = toLight.length();
                        lightDir = toLight / lightDist;
                        Li = al->getIntensity(hairHit.position, lightPos);
                    }
                    else {
                        continue;
                    }

                    if (lightDist < 0.001f) continue;

                    // Shadow check (mesh + hair)
                    Ray shadowRay(hairHit.position + N * 0.003f, lightDir);
                    HitRecord shadowRec;
                    bool inShadow = bvh && bvh->hit(shadowRay, 0.001f, lightDist - 0.01f, shadowRec, true);

                    if (!inShadow) {
                        // Hair self-shadow (only if BVH is ready)
                        float lightShadow = 1.0f;
                        if (!hairSystem.isBVHDirty()) {
                            Hair::HairHitInfo lsh;
                            Vec3 org = shadowRay.origin;
                            float d = std::min(lightDist, 100.0f);
                            int h = 0;
                            // Standard 8-step deep shadow for additional lights
                            while (h < 8 && d > 0.01f && lightShadow > 0.01f) {
                                if (hairSystem.intersect(org, lightDir, 0.002f, d, lsh)) {
                                    lightShadow *= 0.4f;
                                    org = lsh.position + lightDir * 0.003f;
                                    d -= lsh.t + 0.003f;
                                    h++;
                                }
                                else break;
                            }
                        }



                        // Evaluate BSDF for this light (hairHit.v is along, hairHit.u is h)
                        Vec3 lBsdf = Hair::HairBSDF::evaluate(wo, lightDir, T, hairMat, hairHit.v, hairHit.u);

                        // For hair, we don't use standard NdotL with camera-facing N
                        // The BSDF already handles the cylindrical scattering geometry.
                        Vec3 lightContrib = lBsdf * Li * lightShadow;
                        hair_color = hair_color + lightContrib;
                    }
                }

                // Final output
                color += throughput * toVec3f(hair_color);
                return toVec3(color);

            }

            if (bounce == 0 && hit_any) {
                first_hit_t = rec.t;
            }
        }
        // --- God Rays (Bounce 0 only) ---
        if (bounce == 0 && world.data.mode == WORLD_MODE_NISHITA && world.data.nishita.godrays_enabled && world.data.nishita.godrays_intensity > 0.0f) {
            float hit_dist = hit_any ? rec.t : world.data.nishita.fog_distance * 0.5f;
            Vec3 god_rays = VolumetricRenderer::calculateGodRays(scene, world.data, current_ray, hit_dist, bvh, world.getLUT());

            // Check for NaNs/Infs to prevent black screen
            if (std::isfinite(god_rays.x) && std::isfinite(god_rays.y) && std::isfinite(god_rays.z)) {
                color += throughput * toVec3f(god_rays);
            }
        }

        if (!hit_any) {
            // No hit at all? Skip second pass and go to background
        }
        else if (rec.gas_volume || rec.vdb_volume || (rec.material && rec.material->type() == MaterialType::Volumetric)) {
            // Hit a volume? 2. Second Pass: Find what's behind/inside (ignore volumes)
            if (bvh) {
                hit_solid = bvh->hit(current_ray, 0.001f, std::numeric_limits<float>::infinity(), solid_rec, true);
            }
        }
        else {
            // Hit a solid surface directly. Use it for both.
            solid_rec = rec;
            hit_solid = true;
        }

        if (bounce == 0) {
            first_hit_t = hit_solid ? solid_rec.t : -1.0f;
        }

        // === VOLUMETRIC ABSORPTION (Beer's Law) ===
        // Apply absorption based on the distance traveled in the current medium (matches GPU)
        if ((rec.gas_volume || rec.vdb_volume || hit_solid) && (current_medium_absorb.x > 0.0f || current_medium_absorb.y > 0.0f || current_medium_absorb.z > 0.0f)) {
            float dist = hit_solid ? solid_rec.t : rec.t;
            Vec3f transmission(
                expf(-current_medium_absorb.x * dist),
                expf(-current_medium_absorb.y * dist),
                expf(-current_medium_absorb.z * dist)
            );
            throughput *= transmission;
        }

        if (!hit_any && !hit_solid) {
            // --- Infinite Grid Logic (Floor Plane Y=0) ---
            Vec3f final_bg_color = render_settings.show_background ?
                toVec3f(world.evaluate(current_ray.direction, current_ray.origin)) :
                Vec3f(0.0f);

            // Only draw grid if looking down (dir.y < 0) AND NOT in final render mode AND grid enabled
            if (render_settings.grid_enabled && !render_settings.is_final_render_mode && current_ray.direction.y < -0.0001f) {
                float t = -current_ray.origin.y / current_ray.direction.y;
                if (t > 0.0f) {
                    Vec3 p = current_ray.origin + current_ray.direction * t;

                    // --- Improved Infinite Grid Shader ---

                    // 1. Distance Fading (Horizon Fog)
                    // Increased fade start significantly to reduce "foggy" look in viewport
                    float fade_start = 100.0f;
                    float fade_end = render_settings.grid_fade_distance;
                    if (fade_end < fade_start) fade_end = fade_start + 100.0f;

                    float dist = t;
                    float alpha_fade = 1.0f - std::clamp((dist - fade_start) / (fade_end - fade_start), 0.0f, 1.0f);

                    if (alpha_fade > 0.0f) {
                        // 2. Grid Structure (Primary & Secondary Lines)
                        float scale_primary = 10.0f;  // Major lines every 10 units
                        float scale_secondary = 1.0f; // Minor lines every 1 unit

                        // Line width scales with distance to reduce aliasing
                        float line_width_base = 0.02f;
                        float line_width = line_width_base * (1.0f + dist * 0.02f);

                        // Modulo coordinates
                        float x_mod_p = abs(fmod(p.x, scale_primary));
                        float z_mod_p = abs(fmod(p.z, scale_primary));
                        float x_mod_s = abs(fmod(p.x, scale_secondary));
                        float z_mod_s = abs(fmod(p.z, scale_secondary));

                        // Check line hints
                        // Handle wrap-around near scale boundary
                        auto is_line = [&](float val, float scale, float width) {
                            return val < width || val >(scale - width);
                            };

                        bool x_line_p = is_line(x_mod_p, scale_primary, line_width);
                        bool z_line_p = is_line(z_mod_p, scale_primary, line_width);
                        bool x_line_s = is_line(x_mod_s, scale_secondary, line_width);
                        bool z_line_s = is_line(z_mod_s, scale_secondary, line_width);

                        // Axis Lines (Thicker)
                        bool x_axis = abs(p.z) < line_width * 2.5f;
                        bool z_axis = abs(p.x) < line_width * 2.5f;

                        // Determine Color & Alpha
                        Vec3f grid_col(0.0f);
                        float grid_alpha = 0.0f;

                        if (x_axis) {
                            grid_col = Vec3f(0.8f, 0.2f, 0.2f); grid_alpha = 0.9f; // Red X
                        }
                        else if (z_axis) {
                            grid_col = Vec3f(0.2f, 0.8f, 0.2f); grid_alpha = 0.9f; // Green Z
                        }
                        else if (x_line_p || z_line_p) {
                            grid_col = Vec3f(0.40f); grid_alpha = 0.5f; // Major Lines (Darker Grey)
                        }
                        else if (x_line_s || z_line_s) {
                            grid_col = Vec3f(0.25f); grid_alpha = 0.2f; // Minor Lines (Subtle)
                        }

                        // Compose
                        if (grid_alpha > 0.0f) {
                            float final_alpha = grid_alpha * alpha_fade;
                            // Alpha blending
                            final_bg_color = final_bg_color * (1.0f - final_alpha) + grid_col * final_alpha;
                        }
                    }
                }
            }

            // --- Background contribution (matching GPU exactly) ---
            float bg_factor = background_factor(bounce);
            Vec3f bg_contribution = final_bg_color * bg_factor;
            color += throughput * bg_contribution;
            break;
        }

        // Limit volumes by solid depth
        bool hit_solid_inside = false;
        float t_solid = std::numeric_limits<float>::infinity();
        if (hit_solid) t_solid = solid_rec.t;

        // ---------------------------------------------------------------------
        // GAS VOLUME RENDERING (Legacy Path)
        // ---------------------------------------------------------------------
        if (rec.gas_volume && rec.gas_volume->render_path == GasVolume::VolumeRenderPath::Legacy) {
            Vec3 min_b, max_b;
            rec.gas_volume->getWorldBounds(min_b, max_b);

            float t_enter = rec.t;
            float t_exit = t_enter;

            float t1 = std::numeric_limits<float>::infinity();
            for (int i = 0; i < 3; ++i) {
                float invD = 1.0f / current_ray.direction[i];
                float t1_slab = (max_b[i] - current_ray.origin[i]) * invD;
                if (invD < 0.0f) {
                    float t0_slab = (min_b[i] - current_ray.origin[i]) * invD;
                    t1_slab = t0_slab;
                }
                t1 = std::min(t1, t1_slab);
            }
            t_exit = t1;

            if (hit_solid && t_solid < t_exit) {
                t_exit = t_solid;
                hit_solid_inside = true;
            }

            float current_transmittance = 1.0f;
            if (t_exit > t_enter) {
                auto shader = rec.gas_volume->getShader();

                // Step size from shader or default (FASTER rendering with larger steps)
                float step_size = shader ? shader->quality.step_size : 0.15f;
                float anisotropy_g = shader ? shader->scattering.anisotropy : 0.0f;
                float anisotropy_back = shader ? shader->scattering.anisotropy_back : -0.3f;
                float lobe_mix = shader ? shader->scattering.lobe_mix : 0.7f;
                float multi_scatter = shader ? shader->scattering.multi_scatter : 0.3f;

                int max_steps = shader ? shader->quality.max_steps : 256;

                float t = t_enter;
                Vec3f accumulated_emission(0.0f);
                t += ((float)rand() / RAND_MAX) * step_size;

                float density_scale = 1.0f;
                float absorption_coeff = 0.1f;
                float blackbody_intensity = 10.0f;
                float temperature_scale_shader = 1.0f;
                bool use_blackbody = false;

                if (shader) {
                    density_scale = shader->density.multiplier;
                    absorption_coeff = shader->absorption.coefficient;
                    blackbody_intensity = shader->emission.blackbody_intensity;
                    temperature_scale_shader = shader->emission.temperature_scale;
                    use_blackbody = (shader->emission.mode == VolumeEmissionMode::Blackbody);
                }

                float ambient_temp = rec.gas_volume->getSettings().ambient_temperature;
                int steps = 0;

                while (t < t_exit && steps < max_steps) {
                    Vec3 pos = current_ray.at(t);
                    float density = rec.gas_volume->sampleDensity(pos);

                    if (density > 0.001f) {
                        float sigma_t = (density * density_scale) * (1.0f + absorption_coeff);
                        float step_trans = exp(-sigma_t * step_size);

                        // Emission from temperature (fire/flame)
                        if (use_blackbody) {
                            float temperature = rec.gas_volume->sampleTemperature(pos);
                            float flame = rec.gas_volume->sampleFlameIntensity(pos);

                            // Use temperature directly in Kelvin for blackbody
                            // Apply temperature_scale as multiplier
                            float temp_k = (temperature - ambient_temp) * temperature_scale_shader;
                            temp_k = std::max(100.0f, std::min(temp_k + 500.0f, 10000.0f)); // +500 baseline for visible color
                            float tk = temp_k / 100.0f;
                            float r, g, b;

                            if (tk <= 66.0f) {
                                r = 255.0f;
                                g = std::max(0.0f, 99.4708f * logf(tk) - 161.12f);
                                b = (tk <= 19.0f) ? 0.0f : std::max(0.0f, 138.52f * logf(tk - 10.0f) - 305.04f);
                            }
                            else {
                                r = 329.7f * powf(tk - 60.0f, -0.133f);
                                g = 288.12f * powf(tk - 60.0f, -0.0755f);
                                b = 255.0f;
                            }

                            Vec3f bb_color(r / 255.0f, g / 255.0f, b / 255.0f);
                            bb_color.x = std::max(0.0f, std::min(bb_color.x, 1.0f));
                            bb_color.y = std::max(0.0f, std::min(bb_color.y, 1.0f));
                            bb_color.z = std::max(0.0f, std::min(bb_color.z, 1.0f));

                            float flame_boost = 1.0f + flame * 5.0f;
                            float brightness = density * density_scale * blackbody_intensity * flame_boost;
                            Vec3f emission = bb_color * brightness;

                            accumulated_emission += emission * (1.0f - step_trans) * current_transmittance;
                        }

                        current_transmittance *= step_trans;
                    }
                    if (current_transmittance < 0.01f) break;
                    t += step_size;
                    steps++;
                }

                // Apply emission to color
                color = color + Vec3f(accumulated_emission.x, accumulated_emission.y, accumulated_emission.z);

                if (bounce == 0) {
                    if (first_vol_t < 0.0f && (1.0f - current_transmittance) > 0.01f) first_vol_t = t_enter;
                    vol_trans_accum *= current_transmittance;
                }

                throughput *= current_transmittance;
                if (current_transmittance < 0.01f) break;
            }

            if (hit_solid_inside) {
                rec = solid_rec;
            }
            else {
                // Move ray to exit point to continue tracing scene
                current_ray = Ray(current_ray.at(t_exit + 0.001f), current_ray.direction);

                // BOUNCE REFUND for Transparent Gas Volumes (Match VDB/GPU)
                if (current_transmittance > 0.01f) {
                    bounce--;
                }

                continue;
            }
        }


        // --- VDB Volume Rendering (High Quality Unified Path) ---
        const VDBVolume* vdb = rec.vdb_volume;
        int live_vol_id = -1;
        std::shared_ptr<VolumeShader> vol_shader = nullptr;
        Matrix4x4 inv_transform = Matrix4x4::identity();
        float den_scale = 1.0f;

        if (vdb) {
            live_vol_id = vdb->getVDBVolumeID();
            vol_shader = vdb->volume_shader;
            inv_transform = vdb->getInverseTransform();
            den_scale = vdb->density_scale;
        }
        else if (rec.gas_volume && rec.gas_volume->render_path == GasVolume::VolumeRenderPath::VDBUnified) {
            live_vol_id = rec.gas_volume->live_vdb_id;
            vol_shader = rec.gas_volume->getShader();
            if (rec.gas_volume->getTransformHandle()) {
                Matrix4x4 m = rec.gas_volume->getTransformHandle()->getFinal();
                Vec3 gsize = rec.gas_volume->getSettings().grid_size;
                if (gsize.x > 0 && gsize.y > 0 && gsize.z > 0) {
                    m = m * Matrix4x4::scaling(Vec3(1.0f / gsize.x, 1.0f / gsize.y, 1.0f / gsize.z));
                }
                inv_transform = m.inverse();
            }
            den_scale = 1.0f;
        }
        float actual_step_size = 0.0f;
        if (live_vol_id >= 0) {
            // Get entry and exit points
            float t_enter, t_exit;
            bool hit_box = false;
           
            if (vdb) {
                hit_box = vdb->intersectTransformedAABB(current_ray, 0.001f, std::numeric_limits<float>::infinity(), t_enter, t_exit);
            }
            else {
                AABB box; rec.gas_volume->bounding_box(0, 0, box);
                hit_box = box.hit_interval(current_ray, 0.001f, std::numeric_limits<float>::infinity(), t_enter, t_exit);
            }

            if (hit_box) {
                if (hit_solid && t_solid < t_exit) {
                    t_exit = t_solid;
                    hit_solid_inside = true;
                }
                if (t_enter < 0.001f) t_enter = 0.001f;

                float step_size = vol_shader ? vol_shader->quality.step_size : 0.1f;
                if (step_size < 0.001f) step_size = 0.001f;

                float density_scale = (vol_shader ? vol_shader->density.multiplier : 1.0f) * den_scale;

                // --- INITIALIZE VOLUME PARAMETERS (Moved up for scoping) ---
                int max_steps = vol_shader ? vol_shader->quality.max_steps : 256;
                Vec3 aabb_size = vdb ? (vdb->getWorldBounds().max - vdb->getWorldBounds().min) : (rec.gas_volume->getSettings().grid_size);
                float volume_size = aabb_size.length();
                actual_step_size = std::max(step_size, volume_size / (float)max_steps);

                auto& mgr = VDBVolumeManager::getInstance();

                // --- VOLUMETRIC RENDERING (CPU) ---
                // Physical Integration using Beer-Lambert Law

                // Access generic Volume Shader properties
                auto shader = vol_shader;

                step_size = shader ? shader->quality.step_size : 0.1f;
                // Avoid infinite loops with bad step size
                if (step_size < 0.001f) step_size = 0.001f;

                density_scale = (shader ? shader->density.multiplier : 1.0f) * den_scale;

                // Shader parameters - Convert to Vec3f for rendering consistency
                Vec3 albedo_raw = shader ? shader->scattering.color : Vec3(1.0f);
                Vec3f volume_albedo = toVec3f(albedo_raw);
                float scattering_intensity = shader ? shader->scattering.coefficient : 1.0f;

                // ===============================================================
                // Absorption parameters (sigma_a)
                // ===============================================================
                Vec3 absorption_color_raw = shader ? shader->absorption.color : Vec3(0.0f);
                Vec3f absorption_color = toVec3f(absorption_color_raw);
                float absorption_coeff = shader ? shader->absorption.coefficient : 0.0f;

                // ===============================================================
                // Emission parameters
                // ===============================================================
                VolumeEmissionMode emission_mode = shader ? shader->emission.mode : VolumeEmissionMode::None;
                Vec3 emission_color_raw = shader ? shader->emission.color : Vec3(1.0f, 0.5f, 0.1f);
                Vec3f emission_color = toVec3f(emission_color_raw);
                float emission_intensity = shader ? shader->emission.intensity : 0.0f;

                // ============================������������������������������������������
                // Density and Remapping Properties
                // ============================������������������������������������������
                float remap_low = shader ? shader->density.remap_low : 0.0f;
                float remap_high = shader ? shader->density.remap_high : 1.0f;
                float remap_range = std::max(1e-5f, remap_high - remap_low);
                float shadow_strength = shader ? shader->quality.shadow_strength : 1.0f;              
                float anisotropy_g = shader ? shader->scattering.anisotropy : 0.0f;
                float anisotropy_back = shader ? shader->scattering.anisotropy_back : -0.3f;
                float lobe_mix = shader ? shader->scattering.lobe_mix : 0.7f;
                float multi_scatter = shader ? shader->scattering.multi_scatter : 0.3f;

                auto blackbody_to_rgb = [](float kelvin) -> Vec3 {
                    kelvin = std::max(1000.0f, std::min(kelvin, 40000.0f));
                    float t = kelvin / 100.0f;
                    float r, g, b;
                    if (t <= 66.0f) {
                        r = 255.0f;
                        g = 99.4708025861f * std::log(std::max(t, 1e-6f)) - 161.1195681661f;
                        if (t <= 19.0f) b = 0.0f;
                        else b = 138.5177312231f * std::log(std::max(t - 10.0f, 1e-6f)) - 305.0447927307f;
                    } else {
                        r = 329.698727446f * std::pow(t - 60.0f, -0.1332047592f);
                        g = 288.1221695283f * std::pow(t - 60.0f, -0.0755148492f);
                        b = 255.0f;
                    }
                    return Vec3(std::clamp(r / 255.0f, 0.0f, 1.0f), 
                                std::clamp(g / 255.0f, 0.0f, 1.0f), 
                                std::clamp(b / 255.0f, 0.0f, 1.0f));
                };

                // Initialize path state
                float current_transparency = 1.0f;
                Vec3f accumulated_vol_color(0.0f);

                // Jitter to reduce banding
                float jitter = ((float)rand() / RAND_MAX) * actual_step_size;
                float t = t_enter + jitter;

                int steps = 0;

                while (t < t_exit && steps < max_steps && current_transparency > 0.01f) {
                    float threshold = ((float)rand() / RAND_MAX) * 0.01f;

                    Vec3 p = current_ray.at(t);
                    Vec3 local_p = inv_transform.transform_point(p);

                    float density = mgr.sampleDensityCPU(live_vol_id, local_p.x, local_p.y, local_p.z);

                    float edge_falloff = shader ? shader->density.edge_falloff : 0.0f;
                    if (edge_falloff > 0.0f && density > 0.0f) {
                        Vec3 local_min(0), local_max(1);
                        if (vdb) {
                            local_min = vdb->getLocalBoundsMin();
                            local_max = vdb->getLocalBoundsMax();
                        } else {
                            local_max = rec.gas_volume->getSettings().grid_size;
                        }

                        float dx = std::min(local_p.x - local_min.x, local_max.x - local_p.x);
                        float dy = std::min(local_p.y - local_min.y, local_max.y - local_p.y);
                        float dz = std::min(local_p.z - local_min.z, local_max.z - local_p.z);
                        float edge_dist = std::min({ dx, dy, dz });

                        if (edge_dist < edge_falloff) {
                            float edge_factor = edge_dist / edge_falloff;
                            density *= edge_factor * edge_factor;
                        }
                    }

                    float d_remapped = std::max(0.0f, (density - remap_low) / remap_range);
                    float d = d_remapped * density_scale; 

                    if (d > threshold) {
                        if (bounce == 0 && first_vol_t < 0.0f && d > 0.05f) {
                            first_vol_t = t;
                        }

                        float sigma_s = d * scattering_intensity;
                        float sigma_a = d * absorption_coeff;
                        float sigma_t = sigma_s + sigma_a;

                        float albedo_avg = volume_albedo.luminance();
                        float T_single = exp(-sigma_t * actual_step_size);
                        float T_multi_p = exp(-sigma_t * actual_step_size * 0.25f);
                        float step_transmittance = T_single * (1.0f - multi_scatter * albedo_avg) + 
                                                   T_multi_p * (multi_scatter * albedo_avg);

                        Vec3f total_radiance(0.0f);

                        // --- LIGHT SAMPLING ---
                        for (const auto& light : lights) {
                            if (!light) continue;

                            Vec3 light_dir;
                            float light_dist;

                            if (light->type() == LightType::Directional) {
                                light_dir = light->getDirection(p);
                                light_dist = 1e9f;
                            } else {
                                Vec3 to_light = light->position - p;
                                light_dist = to_light.length();
                                light_dir = to_light / std::max(light_dist, 0.0001f);
                            }

                            Vec3f light_intensity = toVec3f(light->getIntensity(p, light->position));
                            
                            // --- PARITY: Atmospheric Extinction for Sun (Matches GPU) ---
                            // If this is the main sun light in Nishita mode, apply transmittance
                            if (light->type() == LightType::Directional && world.data.mode == WORLD_MODE_NISHITA && 
                                world.getLUT() && world.getLUT()->is_initialized()) {
                                
                                float Rg = world.data.nishita.planet_radius;
                                if (Rg < 1000.0f) Rg = 6360000.0f;
                                
                                // Coordinates: Planet center is (0, -Rg, 0)
                                Vec3 p_planet = p + Vec3(0, Rg, 0); 
                                float altitude = p_planet.length() - Rg;
                                Vec3 up = p_planet.normalize(); 
                                
                                float cosTheta = Vec3::dot(up, light_dir);
                                float3 t_sun = world.getLUT()->sampleTransmittance(cosTheta, altitude, world.data.nishita.atmosphere_height);
                                light_intensity = light_intensity * toVec3f((t_sun.x, t_sun.y, t_sun.z));
                            }
                            if (light_intensity.luminance() < 1e-5f) continue;

                            Ray shadow_ray_vol(p + light_dir * 0.001f, light_dir);
                            float shadow_transmittance = 1.0f;
                            HitRecord shadow_rec;
                            
                            if (bvh->hit(shadow_ray_vol, 0.001f, light_dist, shadow_rec)) {
                                if (!shadow_rec.vdb_volume) {
                                    shadow_transmittance = 0.0f;
                                } else {
                                    float density_accum = 0.0f;
                                    float tv_enter, tv_exit;
                                    if (vdb->intersectTransformedAABB(shadow_ray_vol, 0.0f, light_dist, tv_enter, tv_exit)) {
                                        int shadow_steps = shader ? shader->quality.shadow_steps : 8;
                                        float shadow_march_step = volume_size / std::max((float)shadow_steps, 1.0f);
                                        if (shadow_march_step < 0.01f) shadow_march_step = 0.01f;
                                        if (tv_exit > light_dist) tv_exit = light_dist;
                                        float t_shadow = ((float)rand() / RAND_MAX) * shadow_march_step;

                                        while (t_shadow < tv_exit) {
                                            Vec3 slocal_p = inv_transform.transform_point(shadow_ray_vol.at(t_shadow));
                                            float s_density = mgr.sampleDensityCPU(live_vol_id, slocal_p.x, slocal_p.y, slocal_p.z);
                                            float s_rem = std::max(0.0f, (s_density - remap_low) / remap_range);
                                            if (s_rem > 1e-4f) {
                                                density_accum += (s_rem * density_scale * (scattering_intensity + absorption_coeff)) * shadow_march_step;
                                            }
                                            if (density_accum > 10.0f) break; 
                                            t_shadow += shadow_march_step;
                                        }

                                        float beers = exp(-density_accum);
                                        float phys_trans = beers;
                                        
                                        // Match GPU Fix: Only use multi-scatter softening if scattering is actually present
                                        if (scattering_intensity > 1e-6f && multi_scatter > 1e-6f) {
                                            float beers_soft = exp(-density_accum * 0.25f);
                                            phys_trans = beers * (1.0f - multi_scatter * albedo_avg) + beers_soft * (multi_scatter * albedo_avg);
                                        }
                                        shadow_transmittance = 1.0f - shadow_strength * (1.0f - phys_trans);
                                    }
                                }
                            }

                            if (shadow_transmittance > 0.01f) {
                                float cos_theta = Vec3::dot(current_ray.direction.normalize(), light_dir);
                                auto hg = [](float ct, float g) {
                                    float g2 = g * g;
                                    float denom = 1.0f + g2 - 2.0f * g * ct;
                                    return (1.0f - g2) / (4.0f * 3.14159f * std::pow(std::max(denom, 0.0001f), 1.5f));
                                };
                                float phase = hg(cos_theta, anisotropy_g) * lobe_mix + hg(cos_theta, anisotropy_back) * (1.0f - lobe_mix);
                                float powder = 1.0f - std::exp(-d * 2.0f);
                                float forward_bias = 0.5f + 0.5f * std::max(0.0f, cos_theta);
                                phase *= (1.0f + powder * forward_bias * 0.5f);

                                total_radiance += light_intensity * shadow_transmittance * phase;
                            }
                        }

                        // Sky Lighting (Physical Parity: Sample atmospheric color)
                        total_radiance += toVec3f(world.evaluate(Vec3(0, 1, 0))) * 0.15f * world.data.nishita.sun_intensity;

                        // --- EMISSION ---
                        Vec3f step_emission(0.0f);
                        if (emission_mode == VolumeEmissionMode::Constant) {
                            step_emission = emission_color * emission_intensity * d;
                        } else if (emission_mode == VolumeEmissionMode::Blackbody || emission_mode == VolumeEmissionMode::ChannelDriven) {
                            float temp_val = mgr.hasTemperatureGrid(live_vol_id) ? mgr.sampleTemperatureCPU(live_vol_id, local_p.x, local_p.y, local_p.z) : density;
                            
                            float kelvin = 0.0f;
                            float t_ramp = 0.0f;
                            float t_scale = shader ? shader->emission.temperature_scale : 1.0f;
                            float t_max = shader ? shader->emission.temperature_max : 1500.0f;

                            if (temp_val > 20.0f) {
                                kelvin = temp_val * t_scale;
                                t_ramp = temp_val / std::max(1.0f, t_max);
                            } else {
                                // Fallback for density-driven or normalized channels (GPU Parity)
                                kelvin = (temp_val * 3000.0f + 1000.0f) * t_scale;
                                // For density-driven fire, we want it to map more aggressively to the hot end
                                t_ramp = std::max(0.0f, std::min(1.0f, temp_val * 2.0f)); 
                            }

                            if (shader && shader->emission.color_ramp.enabled) {
                                step_emission = toVec3f(shader->emission.color_ramp.sample(t_ramp)) * d * shader->emission.blackbody_intensity;
                            } else {
                                step_emission = toVec3f(blackbody_to_rgb(kelvin)) * d * (shader ? shader->emission.blackbody_intensity : 10.0f);
                            }
                        }

                        // --- VOLUMETRIC INTEGRATION: Multi-Scattering Stable (Parity with GPU) ---
                        float sigma_t_safe = std::max(sigma_t, 1e-6f);
                        Vec3f albedo = (volume_albedo);

                        // Multi-scattering energy gain (Simulates diffuse internal bounces)
                        Vec3f ms_boost = Vec3f(1.0f) + albedo * multi_scatter * 2.0f;
                        Vec3f source = (albedo * total_radiance * sigma_s * ms_boost + step_emission);
                        
                        // Stable Analytical Integration over step
                        Vec3f step_color = source * ((1.0f - step_transmittance) );
                        accumulated_vol_color += step_color * current_transparency;

                        current_transparency *= step_transmittance;
                    }

                    if (current_transparency < 0.01f) break;
                    t += actual_step_size;
                    steps++;
                }

                // Apply volumetric result to path tracer state
                color += throughput * accumulated_vol_color;
                if (bounce == 0) {
                    if (first_vol_t < 0.0f && (1.0f - current_transparency) > 0.01f) first_vol_t = t_enter;
                    vol_trans_accum *= current_transparency;
                }
                throughput *= current_transparency;

                if (hit_solid_inside) {
                    // Handoff to solid surface
                    rec = solid_rec;
                    // Fall through to surface shading
                }
                else {
                    // Move ray to exit point to continue tracing background
                    current_ray = Ray(current_ray.at(t_exit + 0.001f), current_ray.direction);

                    // BOUNCE REFUND for Transparent VDBs (Always refund if not fully opaque)
                    if (current_transparency > 0.01f) {
                        bounce--;
                    }

                    if (current_transparency < 0.01f) break;
                    continue; // Continue to next bounce
                }
            }
            else {
                // AABB intersection failed but VDB was hit - move ray past VDB bounding box
                // This prevents falling through to normal material processing
                AABB world_bounds = vdb->getWorldBounds();
                // Find exit point of world AABB and move ray past it
                float t_far = std::numeric_limits<float>::infinity();
                Vec3 inv_dir = Vec3(1.0f / current_ray.direction.x,
                    1.0f / current_ray.direction.y,
                    1.0f / current_ray.direction.z);
                for (int i = 0; i < 3; i++) {
                    float t1 = (world_bounds.min[i] - current_ray.origin[i]) * inv_dir[i];
                    float t2 = (world_bounds.max[i] - current_ray.origin[i]) * inv_dir[i];
                    t_far = std::min(t_far, std::max(t1, t2));
                }
                current_ray = Ray(current_ray.at(t_far + 0.0001f), current_ray.direction);
                continue; // Continue tracing behind VDB
            }
        }

        // --- Mesh Volume Rendering ---
        if (rec.material && rec.material->type() == MaterialType::Volumetric) {
            auto vol = std::static_pointer_cast<Volumetric>(rec.material);
            float step_size = vol->getStepSize();
            float density_mult = vol->getDensity();

            // Find exit point
            Ray exit_ray(current_ray.at(rec.t + 0.001f), current_ray.direction);
            HitRecord exit_rec;
            float t_vol_enter = rec.t;
            float t_vol_exit = t_vol_enter + 10.0f; // Default if no exit found
            float march_dist = 10.0f;
            bool found_exit = false;

            // Trace to find backface
            if (bvh->hit(exit_ray, 0.001f, std::numeric_limits<float>::infinity(), exit_rec)) {
                // Next hit (could be backface or another object)
                march_dist = exit_rec.t;
                t_vol_exit = t_vol_enter + march_dist;
                found_exit = true;
            }
            else {
                march_dist = 10.0f;
            }

            // Global limit from Dual Pass
            if (hit_solid && t_solid < t_vol_exit) {
                float dist_to_solid = t_solid - t_vol_enter;
                if (dist_to_solid < march_dist) {
                    march_dist = dist_to_solid;
                    t_vol_exit = t_solid;
                    hit_solid_inside = true;
                    found_exit = false; // We didn't reach the backface, we hit an obstruction
                }
            }

            float t = 0.0f; // Relative to entry
            float current_transparency = 1.0f;
            Vec3f accumulated_vol_color(0.0f);

            // Jitter
            float jitter = ((float)rand() / RAND_MAX) * step_size;
            t += jitter;

            int steps = 0;
            int max_steps = vol->getMaxSteps();

            Vec3f volume_albedo = toVec3f(vol->getAlbedo());
            float scattering_intensity = vol->getScattering();
            float absorption_coeff = vol->getAbsorption();
            Vec3f emission_color = toVec3f(vol->getEmissionColor());

            while (t < march_dist && steps < max_steps) {
                Vec3 p = exit_ray.at(t);
                float d = vol->calculate_density(p);

                // Edge falloff for procedural noise to avoid hard cuts if needed
                // (Optional: Implement if vol has properties for it)

                if (d > 0.001f) {
                    float sigma_s_scalar = d * density_mult * scattering_intensity;
                    float sigma_a = d * density_mult * absorption_coeff;
                    float sigma_t = sigma_s_scalar + sigma_a;

                    float step_transmittance = exp(-sigma_t * step_size);

                    // In-Scattering
                    Vec3f total_incoming_light(0.0f);

                    if (current_transparency > 0.01f) {
                        for (const auto& light : lights) {
                            if (!light) continue;

                            Vec3 light_dir;
                            float light_dist;

                            if (light->type() == LightType::Directional) {
                                light_dir = light->getDirection(p);
                                light_dist = 1e9f;
                            }
                            else {
                                Vec3 to_light = light->position - p;
                                light_dist = to_light.length();
                                light_dir = to_light / std::max(light_dist, 0.0001f);
                            }

                            Vec3f light_intensity = toVec3f(light->getIntensity(p, light->position));
                            if (light_intensity.luminance() < 1e-5f) continue;

                            // Shadow Ray
                            Ray shadow_ray_vol(p + light_dir * 0.001f, light_dir);
                            float shadow_transmittance = 1.0f;

                            // Shadow Trace
                            // We reuse the logic: Trace, if hit VDB/Vol -> March, else Opaque
                            float dist_trace = light_dist;
                            int shadow_layers = 0;

                            while (dist_trace > 0.001f && shadow_layers < 2) {
                                HitRecord sidx_rec;
                                if (bvh->hit(shadow_ray_vol, 0.001f, dist_trace, sidx_rec)) {

                                    bool is_transp_shadow = false;

                                    if (sidx_rec.vdb_volume) {
                                        // Assume transparent by default for safety
                                        is_transp_shadow = true;

                                        const VDBVolume* vdb_s = sidx_rec.vdb_volume;
                                        float t_enter_s, t_exit_s;
                                        if (vdb_s->intersectTransformedAABB(shadow_ray_vol, 0.001f, dist_trace, t_enter_s, t_exit_s)) {
                                            // Quick Shadow March
                                            float s_step = 0.5f;
                                            if (vdb_s->volume_shader) s_step = vdb_s->volume_shader->quality.step_size * 2.0f;
                                            float t_s = t_enter_s + ((float)rand() / RAND_MAX) * s_step;
                                            auto& mgr = VDBVolumeManager::getInstance();
                                            int vid = vdb_s->getVDBVolumeID();
                                            Matrix4x4 inv = vdb_s->getInverseTransform();
                                            float ds = (vdb_s->volume_shader ? vdb_s->volume_shader->density.multiplier : 1.0f) * vdb_s->density_scale;

                                            while (t_s < t_exit_s) {
                                                Vec3 sp = shadow_ray_vol.at(t_s);
                                                Vec3 local_sp = inv.transform_point(sp);
                                                float dens = mgr.sampleDensityCPU(vid, local_sp.x, local_sp.y, local_sp.z);
                                                if (dens > 0.01f) shadow_transmittance *= exp(-dens * ds * s_step);
                                                if (shadow_transmittance < 0.01f) break;
                                                t_s += s_step;
                                            }

                                            // Advance past exit
                                            float adv = t_exit_s + 0.001f;
                                            shadow_ray_vol = Ray(shadow_ray_vol.at(adv), shadow_ray_vol.direction);
                                            dist_trace -= adv;
                                        }
                                        else {
                                            // BVH Hit but Intersect Failed (Edge/Precision)
                                            // Advance past BVH hit to allow continuation
                                            float adv = sidx_rec.t + 0.001f;
                                            shadow_ray_vol = Ray(shadow_ray_vol.at(adv), shadow_ray_vol.direction);
                                            dist_trace -= adv;
                                        }
                                    }
                                    else if (sidx_rec.material && sidx_rec.material->type() == MaterialType::Volumetric) {
                                        auto vol_s = std::static_pointer_cast<Volumetric>(sidx_rec.material);
                                        is_transp_shadow = true;

                                        // Find Exit Point or Obstruction
                                        Ray s_exit_ray(shadow_ray_vol.at(sidx_rec.t + 0.001f), shadow_ray_vol.direction);
                                        HitRecord s_exit_rec;
                                        float s_march_dist = dist_trace;

                                        if (bvh->hit(s_exit_ray, 0.001f, dist_trace, s_exit_rec)) {
                                            s_march_dist = s_exit_rec.t;
                                        }

                                        // Homogenous Volume Integration (Beer's Law)
                                        float dens = vol_s->getDensity();
                                        if (dens > 0.0f) {
                                            shadow_transmittance *= exp(-dens * s_march_dist);
                                        }

                                        // Advance Ray
                                        float adv = sidx_rec.t + s_march_dist + 0.001f;
                                        shadow_ray_vol = Ray(shadow_ray_vol.at(adv), shadow_ray_vol.direction);
                                        dist_trace -= adv;
                                    }

                                    if (!is_transp_shadow) {
                                        shadow_transmittance = 0.0f; // Opaque
                                    }
                                }
                                else {
                                    break; // No hit
                                }

                                if (shadow_transmittance < 0.01f) break;
                                shadow_layers++;
                            }
                        }

                        current_transparency *= step_transmittance;
                    }

                    if (current_transparency < 0.01f) break;
                    t += actual_step_size;
                    steps++;
                }

                color += throughput * accumulated_vol_color;
                if (bounce == 0) {
                    // For mesh volumes, we use the entry point if we hit density
                    // (Mesh volumes are usually dense throughout, but we could add a check here too)
                    if (first_vol_t < 0.0f && (1.0f - current_transparency) > 0.01f) {
                        first_vol_t = t_vol_enter;
                    }
                    vol_trans_accum *= current_transparency;
                }
                throughput *= current_transparency;

                // ---------------------------------------------------------
                // EXIT LOGIC: Handle Backface vs Obstruction
                // ---------------------------------------------------------

                // Check if we hit the Volume Backface or a different object (Obstruction)
                bool hit_backface = found_exit && (exit_rec.materialID == rec.materialID);

                if (hit_backface || !found_exit) {
                    // CASE A: Standard Volume Exit (Backface) OR Infinite 
                    // We marched through the volume and exited at the other side.
                    // Advance ray PAST the volume backface to continue tracing scene.
                    current_ray = Ray(exit_ray.at(march_dist + 0.001f), current_ray.direction);

                    // BOUNCE REFUND for Transparent Volumes (Match GPU / Gas / VDB)
                    if (current_transparency > 0.01f) {
                        bounce--;
                    }

                    if (current_transparency < 0.01f) break;
                    continue; // Continue loop with new ray

                }
                else {
                    // CASE B: Hit an Obstruction inside/behind Volume (e.g. Wall)
                    // We hit something that is NOT the volume itself.
                    // We must SHADE this object, not skip it.
                    // We have already accumulated volume opacity up to this point.

                    // Update the main HitRecord to the obstruction
                    rec = exit_rec;

                    // Fall through to standard surface shading code below!
                    // (Do NOT continue loop, do NOT update current_ray yet)

                    // Applying volume throughput to the current path
                    // color += ... was already done above.
                    // throughput *= ... was done above.

                    // Proceed to shade 'rec' (The Wall)
                }
            }
        }
        // --- Normal map application ---
        apply_normal_map(rec);

        // --- Ensure correct normal orientation (faceforward) ---
        Vec3f wo = toVec3f(-current_ray.direction.normalize());
        Vec3f N = toVec3f(rec.interpolated_normal);
        Vec3f geom_N = toVec3f(rec.normal);

        // Faceforward: flip normal if we hit backface
        if (dot(wo, geom_N) < 0.0f) {
            N = -N;
            geom_N = -geom_N;
        }

        Vec3f hit_pos = toVec3f(rec.point);

        // --- Extract material parameters (with texture sampling) ---
        Vec3f albedo(0.8f);
        float roughness = 0.5f;
        float metallic = 0.0f;
        float opacity = 1.0f;
        float transmission = 0.0f;
        Vec3f emission(0.0f);

        if (rec.material) {
            auto pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(rec.material);
            if (pbsdf) {
                Vec2 uv(rec.u, rec.v);

                // Opacity
                opacity = pbsdf->get_opacity(uv);

                // === STOCHASTIC TRANSPARENCY BOUNCE REFUND ===
                // Prevents "Ghost Silhouette" when max bounces is reached on transparent geometry.
                if (opacity < 0.999f) {
                    if (Vec3::random_float() > opacity) {
                        transparent_hits++;

                        // Pass-through: move ray and refund bounce count if under limit
                        current_ray = Ray(rec.point + current_ray.direction * 0.001f, current_ray.direction);

                        if (transparent_hits <= render_settings.transparent_max_bounces) {
                            bounce--;
                        }
                        continue;
                    }
                }

                // --- Albedo, Roughness, Metallic, Normal (Terrain vs Standard) ---
                bool terrain_handled = false;
                if (rec.terrain_id != -1) {
                    TerrainObject* terrain = TerrainManager::getInstance().getTerrain(rec.terrain_id);
                    if (terrain && terrain->splatMap && !terrain->layers.empty()) {
                        Vec3 splat_rgb = terrain->splatMap->get_color(uv.u, uv.v);
                        float splat_a = terrain->splatMap->get_alpha(uv.u, uv.v);
                        float weights[4] = { (float)splat_rgb.x, (float)splat_rgb.y, (float)splat_rgb.z, splat_a };
                        
                        Vec3f b_albedo(0.0f);
                        float b_rough = 0.0f;
                        float b_metal = 0.0f;
                        Vec3 blended_n(0.0f);
                        bool has_n = false;

                        for (int i = 0; i < 4 && i < (int)terrain->layers.size(); ++i) {
                            if (weights[i] < 0.001f) continue;
                            auto layer = std::dynamic_pointer_cast<PrincipledBSDF>(terrain->layers[i]);
                            if (layer) {
                                float scale = (i < (int)terrain->layer_uv_scales.size()) ? terrain->layer_uv_scales[i] : 1.0f;
                                Vec2 luv(uv.u * scale, uv.v * scale);
                                
                                b_albedo += toVec3f(layer->getPropertyValue(layer->albedoProperty, luv)) * weights[i];
                                b_rough += static_cast<float>(layer->getPropertyValue(layer->roughnessProperty, luv).y) * weights[i];
                                b_metal += static_cast<float>(layer->getPropertyValue(layer->metallicProperty, luv).z) * weights[i];
                                
                                if (layer->has_normal_map()) {
                                    blended_n += (layer->get_normal_from_map(luv.u, luv.v) * 2.0f - Vec3(1.0f)) * weights[i];
                                    has_n = true;
                                }
                            }
                        }
                        albedo = b_albedo.clamp(0.01f, 1.0f);
                        roughness = std::clamp(b_rough, 0.01f, 1.0f);
                        metallic = std::clamp(b_metal, 0.0f, 1.0f);
                        
                        if (has_n) {
                            Vec3 T, B;
                            Renderer::create_coordinate_system(rec.normal, T, B);
                            Mat3x3 TBN(T, B, rec.normal);
                            rec.interpolated_normal = (TBN * blended_n.normalize()).normalize();
                            N = toVec3f(rec.interpolated_normal);
                        }
                        
                        // Pass blended data to HitRecord for scatter/pdf
                        rec.use_custom_data = true;
                        rec.custom_albedo = toVec3(albedo);
                        rec.custom_roughness = roughness;
                        rec.custom_metallic = metallic;
                        rec.custom_transmission = transmission;
                        
                        terrain_handled = true;
                    }
                }

                if (!terrain_handled) {
                    // Albedo
                    Vec3 alb = pbsdf->getPropertyValue(pbsdf->albedoProperty, uv);
                    albedo = toVec3f(alb).clamp(0.01f, 1.0f);

                    // Roughness (Y channel)
                    Vec3 rough = pbsdf->getPropertyValue(pbsdf->roughnessProperty, uv);
                    roughness = static_cast<float>(rough.y);

                    // Metallic (Z channel)
                    Vec3 metal = pbsdf->getPropertyValue(pbsdf->metallicProperty, uv);
                    metallic = static_cast<float>(metal.z);
                }

                // Transmission
                transmission = pbsdf->getTransmission(uv);

                // === WATER WAVE SHADER (CPU) ===
                // Detect water materials using sheen > 0 (IS_WATER flag)
                bool is_water = (pbsdf->sheen > 0.0001f && transmission > 0.1f);

                if (is_water) {
                    extern RenderSettings render_settings;
                    float fps = static_cast<float>(render_settings.animation_fps > 0 ? render_settings.animation_fps : 24);
                    float water_time = static_cast<float>(render_settings.animation_current_frame) / fps;

                    // Pack parameters (Mirror GPU raygen.cu packing)
                    WaterParamsCPU params;
                    if (rec.material->gpuMaterial) {
                        auto& g_mat = *rec.material->gpuMaterial;
                        params.wave_speed = g_mat.anisotropic;
                        params.wave_strength = g_mat.sheen;
                        params.wave_frequency = g_mat.sheen_tint;

                        params.shallow_color = Vec3(g_mat.emission.x, g_mat.emission.y, g_mat.emission.z);
                        params.deep_color = Vec3(g_mat.albedo.x, g_mat.albedo.y, g_mat.albedo.z);
                        params.absorption_color = Vec3(g_mat.subsurface_color.x, g_mat.subsurface_color.y, g_mat.subsurface_color.z);

                        params.depth_max = g_mat.subsurface * 100.0f;
                        params.absorption_density = g_mat.subsurface_scale;
                        params.clarity = std::fmax(0.1f, 1.0f - params.absorption_density);

                        params.foam_level = g_mat.translucent;
                        params.shore_foam_distance = g_mat.subsurface_radius.x;
                        params.shore_foam_intensity = g_mat.clearcoat;

                        params.caustic_intensity_scale = g_mat.clearcoat_roughness;
                        params.caustic_scale = g_mat.subsurface_radius.y;
                        params.caustic_speed = g_mat.subsurface_anisotropy;

                        params.sss_intensity = g_mat.subsurface_radius.z;
                        params.sss_color = params.absorption_color;

                        params.use_fft_ocean = (g_mat.fft_height_tex != 0);
                        params.fft_ocean_size = g_mat.fft_ocean_size;
                        params.fft_choppiness = g_mat.fft_choppiness;

                        params.micro_detail_strength = g_mat.micro_detail_strength;
                        params.micro_detail_scale = g_mat.micro_detail_scale;
                        params.micro_anim_speed = g_mat.micro_anim_speed;
                        params.micro_morph_speed = g_mat.micro_morph_speed;
                        params.foam_noise_scale = g_mat.foam_noise_scale;
                        params.foam_threshold = g_mat.foam_threshold;
                        
                        // Wind animation for micro details
                        params.wind_direction = g_mat.fft_wind_direction;
                        params.wind_speed = g_mat.fft_wind_speed;
                        params.time = water_time;  // Use pre-calculated water time (seconds)
                    } else {
                        // Use PrincipledBSDF fields directly (UI alignment)
                        params.wave_speed = pbsdf->anisotropic;
                        params.wave_strength = pbsdf->sheen;
                        params.wave_frequency = pbsdf->sheen_tint;
                        
                        params.shallow_color = pbsdf->emissionProperty.color;
                        params.deep_color = pbsdf->albedoProperty.color;
                        params.absorption_color = pbsdf->subsurfaceColor;
                        
                        params.depth_max = pbsdf->subsurface * 100.0f;
                        params.absorption_density = pbsdf->subsurfaceScale;
                        params.clarity = std::fmax(0.1f, 1.0f - params.absorption_density);

                        params.foam_level = 0.01f; // High/Starting quality default
                        params.shore_foam_distance = pbsdf->subsurfaceRadius.x;
                        params.shore_foam_intensity = pbsdf->clearcoat;
                        
                        params.caustic_intensity_scale = pbsdf->clearcoatRoughness;
                        params.caustic_scale = pbsdf->subsurfaceRadius.y;
                        params.caustic_speed = pbsdf->subsurfaceAnisotropy;
                        
                        params.sss_intensity = pbsdf->subsurfaceRadius.z;
                        params.sss_color = params.absorption_color;

                        params.use_fft_ocean = false; // FFT always requires GPU
                        params.micro_detail_strength = 0.0f;
                        params.micro_detail_scale = 1.0f;
                        params.micro_anim_speed = 0.1f;
                        params.micro_morph_speed = 1.0f;
                        params.foam_noise_scale = 1.0f;
                        params.foam_threshold = 0.5f;
                        
                        // Wind animation defaults
                        params.wind_direction = 0.0f;
                        params.wind_speed = 10.0f;
                        params.time = water_time;
                    }

                    // Evaluate water displacement and appearance
                    Vec3 base_normal(0.0f, 1.0f, 0.0f);
                    WaterResultCPU wave = evaluateWaterCPU(
                        rec.point, base_normal, water_time, params
                    );

                    // Apply wave normal
                    rec.interpolated_normal = wave.normal;
                    N = toVec3f(wave.normal);

                    // --- Apply Water Appearance (Color Parity) ---
                    // Overwrite base albedo with calculated water color (Shallow/Deep mix)
                    albedo = toVec3f(wave.water_color);

                    // Foam blending
                    float total_foam = wave.foam * params.foam_level;
                    if (total_foam > 0.01f) {
                        Vec3f foam_color(0.92f, 0.96f, 1.0f); // Slightly blue-tinted foam
                        albedo = albedo * (1.0f - total_foam) + foam_color * total_foam;
                        roughness = roughness * (1.0f - total_foam) + 0.8f * total_foam; // Foam is rough
                    }
                }

                // NOTE: Emission is now retrieved polymorphically for all materials after scatter
            }
        }

        // --- Add Emission (Volumetric & Surface) ---
        if (rec.material && rec.material->type() == MaterialType::Volumetric) {
            auto vol = std::static_pointer_cast<Volumetric>(rec.material);

            // Calculate AABB for Object Space Noise
            Vec3 aabb_min(0), aabb_max(0);
            if (rec.triangle) {
                AABB box;
                if (rec.triangle->bounding_box(0, 0, box)) {
                    aabb_min = box.min;
                    aabb_max = box.max;
                    // Add padding to match GPU logic
                    float padding = 0.001f;
                    aabb_min = aabb_min - Vec3(padding);
                    aabb_max = aabb_max + Vec3(padding);
                }
            }

            Vec3 vol_emit = vol->getVolumetricEmission(toVec3(hit_pos), current_ray.direction, aabb_min, aabb_max);
            emission += toVec3f(vol_emit);
        }

        // NOTE: Emission is added to total_contribution AFTER scatter (matching GPU)

        // --- Throughput clamp (matching GPU) ---
        float max_throughput = throughput.max_component();
        if (max_throughput > UnifiedConstants::MAX_CONTRIBUTION) {
            throughput *= (UnifiedConstants::MAX_CONTRIBUTION / max_throughput);
        }

        // --- Russian Roulette (matching GPU exactly) ---
        // GPU: if (bounce > 2) { p = clamp(p, 0.05f, 0.95f); ... }
        if (bounce > UnifiedConstants::RR_START_BOUNCE) {
            float p = russian_roulette_probability(throughput);
            if (Vec3::random_float() > p) {
                break;
            }
            throughput /= p;

            // Post-RR clamp (matching GPU)
            max_throughput = throughput.max_component();
            if (max_throughput > UnifiedConstants::MAX_CONTRIBUTION) {
                throughput *= (UnifiedConstants::MAX_CONTRIBUTION / max_throughput);
            }
        }

        // --- Scatter ray (GPU Parity: Happens before contribution accumulation) ---
        Vec3 attenuation(1.0f);
        Ray scattered;
        bool is_specular = false;
        bool can_scatter = rec.material && rec.material->scatter(current_ray, rec, attenuation, scattered, is_specular);

        if (!can_scatter) break;

        Vec3f atten_f = toVec3f(attenuation);
        throughput *= atten_f;

        // --- Volumetric Medium Tracking (for Beer's Law) ---
        // Matches GPU logic: Check if we are entering or exiting a transmissive material
        if (is_specular && transmission > 0.01f) {
            Vec3 N = rec.normal;
            Vec3 D = current_ray.direction.normalize();
            float NdotD = Vec3::dot(D, N);
            bool entering = NdotD < 0.0f;

            if (entering) {
                // Entering medium: Set absorption coefficient
                // sigma_a = (1 - color) * density
                auto pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(rec.material);
                if (pbsdf) {
                    Vec3 subColor = pbsdf->subsurfaceColor; 
                    float subScale = pbsdf->subsurfaceScale;
                    
                    Vec3 absorb_base = Vec3(1.0f) - subColor;
                    absorb_base = Vec3(fmaxf(absorb_base.x, 0.0f), fmaxf(absorb_base.y, 0.0f), fmaxf(absorb_base.z, 0.0f));
                    
                    current_medium_absorb = toVec3f(absorb_base * subScale);
                }
            } else {
                // Exiting medium: Reset to air
                current_medium_absorb = Vec3f(0.0f);
            }
        }
        
        // --- Emissive Contribution ---
        Vec3 emitted = rec.material ? rec.material->getEmission(rec.uv, rec.point) : Vec3(0.0f);
        emission = toVec3f(emitted/2);

        // --- Direct lighting ---
        Vec3f direct_light(0.0f);
        if (!is_specular && light_count > 0 && transmission < 0.99f) {
            // --- Smart Light Selection (Importance Sampling) ---
            int light_index = -1;
            float pdf_select = 1.0f;
            float r = Vec3::random_float();

            // --- SMART LIGHT SELECTION (Same as GPU) ---
            // Use the unified function to pick light based on distance/intensity importance
            // Pass &pdf_select to get the actual probability used for selection
            light_index = pick_smart_light_unified(unified_lights.data(), light_count, toVec3f(rec.point), r, &pdf_select);

            if (light_index >= 0) {
                const UnifiedLight& light = unified_lights[light_index];
                Vec3f wi; float distance; float light_attenuation;
                if (sample_light_direction(light, hit_pos, Vec3::random_float(), Vec3::random_float(), &wi, &distance, &light_attenuation)) {
                    if (dot(N, wi) > 0.001f) {
                        Vec3 shadow_origin = rec.point + rec.interpolated_normal * UnifiedConstants::SHADOW_BIAS;
                        Ray shadow_ray(shadow_origin, toVec3(wi));
                        bool meshOccluded = bvh->occluded(shadow_ray, UnifiedConstants::SHADOW_BIAS, distance);
                        
                        // --- VOLUMETRIC SHADOWING (VDB & Gas) ---
                        float volShadowTransmittance = 1.0f;
                        if (!meshOccluded && (!scene.vdb_volumes.empty() || !scene.gas_volumes.empty())) {
                            for (const auto& v_ptr : scene.vdb_volumes) {
                                if (!v_ptr || !v_ptr->visible) continue;
                                float t0_v, t1_v;
                                if (v_ptr->intersectTransformedAABB(shadow_ray, 0.001f, distance, t0_v, t1_v)) {
                                    // Use a coarser step for shadow marching to preserve performance
                                    float s_step = (v_ptr->volume_shader ? v_ptr->volume_shader->quality.step_size : 0.25f) * 2.5f;
                                    float t_v = t0_v + 0.001f;
                                    auto& mgr = VDBVolumeManager::getInstance();
                                    int vid = v_ptr->getVDBVolumeID();
                                    Matrix4x4 inv = v_ptr->getInverseTransform();
                                    float ds = (v_ptr->volume_shader ? v_ptr->volume_shader->density.multiplier : 1.0f) * v_ptr->density_scale;
                                    float sigma_t = (v_ptr->volume_shader ? (v_ptr->volume_shader->scattering.coefficient + v_ptr->volume_shader->absorption.coefficient) : 1.1f);
                                    
                                    while (t_v < t1_v) {
                                        Vec3 sp = shadow_ray.at(t_v);
                                        Vec3 local_sp = inv.transform_point(sp);
                                        float dens = mgr.sampleDensityCPU(vid, local_sp.x, local_sp.y, local_sp.z);
                                        if (dens > 0.01f) volShadowTransmittance *= expf(-dens * ds * sigma_t * s_step);
                                        if (volShadowTransmittance < 0.01f) {
                                            volShadowTransmittance = 0.0f;
                                            break;
                                        }
                                        t_v += s_step;
                                    }
                                }
                                if (volShadowTransmittance < 0.01f) break;
                            }
                        }

                        // Hair shadow check (hair can cast shadows on meshes)
                        float hairShadowTransmittance = 1.0f;
                        if (!meshOccluded && volShadowTransmittance > 0.01f && hairSystem.getTotalStrandCount() > 0 && !hairSystem.isBVHDirty()) {
                            Hair::HairHitInfo hsh;
                            Vec3 hairShadowOrigin = shadow_origin;
                            Vec3 hairShadowDir = toVec3(wi);
                            float hairShadowDist = std::min(distance, 100.0f);
                            int hHits = 0;
                            while (hHits < 6 && hairShadowDist > 0.01f && hairShadowTransmittance > 0.05f) {
                                if (hairSystem.intersect(hairShadowOrigin, hairShadowDir, 0.002f, hairShadowDist, hsh)) {
                                    hairShadowTransmittance *= 0.4f; // Each strand blocks 60%
                                    hairShadowOrigin = hsh.position + hairShadowDir * 0.003f;
                                    hairShadowDist -= (hsh.t + 0.003f);
                                    hHits++;
                                } else break;
                            }
                        }
                        
                        if (!meshOccluded && hairShadowTransmittance > 0.01f && volShadowTransmittance > 0.01f) {

                            // Light Radiance (Intensity * Color * Attenuation)
                            Vec3f Li = light.color * light.intensity * light_attenuation * hairShadowTransmittance * volShadowTransmittance;

                            // Evaluate BRDF and PDF for MIS
                            Vec3f f = evaluate_brdf_unified(N, wo, wi, albedo, roughness, metallic, transmission, Vec3::random_float());
                            float pdf_brdf = pdf_brdf_unified(N, wo, wi, roughness);

                            // MIS Weight (Currently 1.0 for analytic lights to match GPU, can be enabled for Area lights)
                            float mis_weight = 1.0f;
                            // GPU Parity: GPU calculates MIS using only geometry PDF, ignoring selection PDF
                            if (light.type == (int)UnifiedLightType::Area) {
                                float pdf_geo = compute_light_pdf(light, distance, 1.0f); // Pass 1.0 for selection PDF
                                mis_weight = power_heuristic(pdf_geo, std::clamp(pdf_brdf, 0.01f, 5000.0f));
                            }

                            // Contribution: (f * Li * cos * mis_weight)
                            // GPU Parity: Do NOT divide by pdf_select. GPU performs biased accumulation based on selection frequency.
                            direct_light = (f * Li * std::max(0.0f, dot(N, wi)) * mis_weight);

                            direct_light = clamp_contribution(direct_light, UnifiedConstants::MAX_CONTRIBUTION);
                        }

                    }
                }
            }
        }
        
        // --- Accumulate contribution (GPU Match: Total = direct + emission) ---
        Vec3f total_contribution = direct_light + emission;

        // Final contribution clamp
        float total_lum = total_contribution.luminance();
        if (total_lum > UnifiedConstants::MAX_CONTRIBUTION * 2.0f) {
            total_contribution *= (UnifiedConstants::MAX_CONTRIBUTION * 2.0f / total_lum);
        }

        color += throughput * total_contribution;

        current_ray = scattered;
    }
    

    // --- Post-Process: Aerial Perspective (Matches GPU) ---
    if (world.data.mode == WORLD_MODE_NISHITA) {
        float ap_dist = first_hit_t;

        // --- WEIGHTED FOG DISTANCE (Ghost Box Protection) ---
        // Match GPU: lerp between background distance and volume distance based on opacity
        if (first_vol_t > 0.0f) {
            float background_t = (first_hit_t > 0.0f) ? first_hit_t : 10000.0f;
            float weight = 1.0f - vol_trans_accum;
            ap_dist = background_t * (1.0f - weight) + first_vol_t * weight;
        }
        else if (ap_dist <= 0.0f) {
            ap_dist = 10000.0f;
        }

        Vec3 final_c = VolumetricRenderer::applyAerialPerspective(scene, world.data, r.origin, r.direction, ap_dist, toVec3(color), world.getLUT());
        color = toVec3f(final_c);
    }

    else if (world.data.nishita.fog_enabled && world.data.nishita.fog_density > 0.0f) {
        // Fallback for simple height fog on other modes
        // ... (Optional: could add simple fog here too if needed, but Nishita Parity is the main goal)
    }

    // --- Final NaN/Inf check and clamp (matching GPU) ---
    // GPU: color.x = isfinite(color.x) ? fminf(fmaxf(color.x, 0.0f), 100.0f) : 0.0f;
    if (!color.is_valid()) {
        color = Vec3f(0.0f);
    }
    color = color.clamp(0.0f, 100.0f);

    return toVec3(color);
}

// ============================================================================
// ACCUMULATIVE RENDERING (CPU)
// ============================================================================

uint64_t Renderer::computeCPUCameraHash(const Camera& cam) const {
    // FNV-1a hash of camera parameters
    uint64_t hash = 14695981039346656037ULL;
    auto hashFloat = [&hash](float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
        };

    hashFloat(cam.lookfrom.x);
    hashFloat(cam.lookfrom.y);
    hashFloat(cam.lookfrom.z);
    hashFloat(cam.lookat.x);
    hashFloat(cam.lookat.y);
    hashFloat(cam.lookat.z);
    hashFloat(cam.vup.x);
    hashFloat(cam.vup.y);
    hashFloat(cam.vup.z);
    hashFloat(cam.vfov);

    return hash;
}

void Renderer::resetCPUAccumulation() {
    cpu_accumulated_samples = 0;
    cpu_accumulation_valid = false;
    cpu_last_camera_hash = 0;  // Reset camera hash for animation frames
    cpu_pixel_list_valid = false;  // Force pixel list rebuild + shuffle on next pass
    if (!cpu_accumulation_buffer.empty()) {
        std::fill(cpu_accumulation_buffer.begin(), cpu_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    // Reset variance buffer for adaptive sampling
    if (!cpu_variance_buffer.empty()) {
        std::fill(cpu_variance_buffer.begin(), cpu_variance_buffer.end(), 0.0f);
    }
}

// ============================================================================
// Hair System GPU Integration
// ============================================================================



void Renderer::uploadHairToGPU() {
    if (!m_backend || !g_hasCUDA) return;

    Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(m_backend);
    OptixWrapper* optix_gpu = optixBackend ? optixBackend->getOptixWrapper() : nullptr;

    if (hairSystem.getTotalStrandCount() == 0) {
        if (optix_gpu) optix_gpu->clearHairGeometry();
        return;
    }
    
    // Clear previous hair states in OptiX
    if (optix_gpu) optix_gpu->clearHairGeometry();
    
    auto groomNames = hairSystem.getGroomNames();
    bool first = true;
    
    for (const auto& name : groomNames) {
        std::vector<float> hairVertices4;
        std::vector<unsigned int> hairIndices;
        std::vector<uint32_t> hairStrandIDs;
        std::vector<float> hairTangents3;
        std::vector<float> hairRootUVs2;
        std::vector<float> hairStrandVs;
        size_t vertexCount = 0;
        size_t segmentCount = 0;
        Hair::HairMaterialParams matParams;
        int hairMatID = 0;
        int meshMatID = -1;
        
        // Call with all 13 arguments explicitly
        bool isSpline = hairSystem.getOptiXCurveDataByGroom(
            name,                   // 1
            hairVertices4,          // 2
            hairIndices,            // 3
            hairStrandIDs,          // 4
            hairTangents3,          // 5
            hairRootUVs2,           // 6
            hairStrandVs,           // 7
            vertexCount,            // 8
            segmentCount,           // 9
            matParams,              // 10
            hairMatID,              // 11
            meshMatID,              // 12
            !hideInterpolatedHair   // 13
        );
        
        if (segmentCount > 0) {
            // Convert to float4/float3 arrays
            std::vector<float4> vertices4(vertexCount);
            for (size_t i = 0; i < vertexCount; ++i) {
                vertices4[i] = make_float4(hairVertices4[i * 4 + 0], hairVertices4[i * 4 + 1], hairVertices4[i * 4 + 2], hairVertices4[i * 4 + 3]);
            }
            
            std::vector<float3> tangents3(segmentCount);
            for (size_t i = 0; i < segmentCount; ++i) {
                tangents3[i] = make_float3(hairTangents3[i * 3 + 0], hairTangents3[i * 3 + 1], hairTangents3[i * 3 + 2]);
            }
            
            std::vector<float2> rootUVs2(segmentCount);
            for (size_t i = 0; i < segmentCount; ++i) {
                rootUVs2[i] = make_float2(hairRootUVs2[i * 2 + 0], hairRootUVs2[i * 2 + 1]);
            }
            
            // Generate GPU Hair Material
            GpuHairMaterial hairMat = Hair::HairBSDF::convertToGpu(matParams);

            // Add this groom to OptiX
            if (optix_gpu) optix_gpu->buildHairGeometry(
                vertices4.data(),
                hairIndices.data(),
                hairStrandIDs.data(),
                tangents3.data(),
                rootUVs2.data(),
                hairStrandVs.data(),
                vertexCount,
                segmentCount,
                hairMat,
                name,
                hairMatID,
                meshMatID,
                isSpline,
                false
            );
        }
    }
    
    SCENE_LOG_INFO("[Hair GPU] Integrated " + std::to_string(groomNames.size()) + " hair grooms into TLAS");
}

void Renderer::updateHairGeometryOnGPU(bool forceRebuild) {
    if (!m_backend || !g_hasCUDA) return;
    
    // Without HairGPUManager, we always perform a full upload if anything changed.
    // In the future, we can implement a CPU-based vertex refit here.
    uploadHairToGPU();
}

void Renderer::setHairMaterial(const Hair::HairMaterialParams& mat) {
    this->hairMaterial = mat;

    if (m_backend && g_hasCUDA) {
        Backend::IBackend::HairMaterialData hmd;
        hmd.color = mat.color;
        hmd.absorption = mat.absorptionCoefficient;
        hmd.melanin = mat.melanin;
        hmd.melaninRedness = mat.melaninRedness;
        hmd.roughness = mat.roughness;
        hmd.radialRoughness = mat.radialRoughness;
        hmd.ior = mat.ior;
        hmd.coat = mat.coat;
        hmd.cuticleAngle = mat.cuticleAngle * 3.14159f / 180.0f;
        hmd.randomHue = mat.randomHue;
        hmd.randomValue = mat.randomValue;
        hmd.colorMode = static_cast<int>(mat.colorMode);

        // 1. Albedo Texture
        if (mat.customAlbedoTexture && mat.customAlbedoTexture->is_loaded()) {
            if (!mat.customAlbedoTexture->isUploaded()) mat.customAlbedoTexture->upload_to_gpu();
            if (mat.customAlbedoTexture->isUploaded()) {
                hmd.albedoTexture = (int64_t)mat.customAlbedoTexture->getTextureObject();
            }
        }

        // 2. Roughness Texture
        if (mat.customRoughnessTexture && mat.customRoughnessTexture->is_loaded()) {
            if (!mat.customRoughnessTexture->isUploaded()) mat.customRoughnessTexture->upload_to_gpu();
            if (mat.customRoughnessTexture->isUploaded()) {
                hmd.roughnessTexture = (int64_t)mat.customRoughnessTexture->getTextureObject();
            }
        }

        // 3. Scalp Mesh Texture (Automatic detection for ROOT_UV_MAP mode)
        if (mat.colorMode == Hair::HairMaterialParams::ColorMode::ROOT_UV_MAP && hmd.albedoTexture == -1) {
            auto groomNames = hairSystem.getGroomNames();
            if (!groomNames.empty()) {
                auto* groom = hairSystem.getGroom(groomNames[0]);
                if (groom) {
                    auto& matMgr = MaterialManager::getInstance();
                    const auto& all_mats = matMgr.getAllMaterials();
                    int scalpMatID = groom->params.defaultMaterialID;

                    if (scalpMatID >= 0 && (size_t)scalpMatID < all_mats.size()) {
                        const auto& scalpMat = all_mats[scalpMatID];
                        if (scalpMat) {
                            if (scalpMat->albedoProperty.texture && scalpMat->albedoProperty.texture->is_loaded()) {
                                if (!scalpMat->albedoProperty.texture->isUploaded()) scalpMat->albedoProperty.texture->upload_to_gpu();
                                if (scalpMat->albedoProperty.texture->isUploaded()) {
                                    hmd.scalpAlbedoTexture = (int64_t)scalpMat->albedoProperty.texture->getTextureObject();
                                }
                            }
                            hmd.scalpBaseColor = scalpMat->albedoProperty.color;
                        }
                    }
                }
            }
        }

        std::vector<Backend::IBackend::HairMaterialData> materials = { hmd };
        m_backend->uploadHairMaterials(materials);
        
        // For OptiX, we also need to trigger per-groom updates if they are managed separately
        // Although the combined upload should handle the global params.
        Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(m_backend);
        if (optixBackend && optixBackend->getOptixWrapper()) {
            optixBackend->getOptixWrapper()->updateHairMaterialsOnly(hairSystem);
        }
    }
    resetCPUAccumulation();
}

bool Renderer::isCPUAccumulationComplete() const {
    extern RenderSettings render_settings;
    int target_max_samples = render_settings.max_samples > 0 ? render_settings.max_samples : 100;

    // Check override for final render / animation
    if (render_settings.is_final_render_mode) {
        target_max_samples = render_settings.final_render_samples > 0 ? render_settings.final_render_samples : 128;
    }

    return cpu_accumulated_samples >= target_max_samples;
}

void Renderer::render_progressive_pass(SDL_Surface* surface, SDL_Window* window, SceneData& scene, int samples_this_pass, int override_target_samples) {
    // [SAFETY CHECK] Prevent rendering if stopped or loading a new project
    // This prevents access violations when camera/scene data is being destroyed
    extern std::atomic<bool> rendering_stopped_cpu;
    extern std::atomic<bool> g_scene_loading_in_progress;

    if (rendering_stopped_cpu.load()) {
        // SCENE_LOG_WARN("CPU Render aborted: rendering_stopped_cpu is true");
        return;
    }
    if (g_scene_loading_in_progress.load()) {
        SCENE_LOG_WARN("CPU Render aborted: scene loading in progress");
        return;
    }
    if (!scene.camera) {
        SCENE_LOG_WARN("CPU Render aborted: no camera");
        return;
    }

    extern RenderSettings render_settings;


    // Ensure accumulation buffer is allocated
    const size_t pixel_count = image_width * image_height;
    if (cpu_accumulation_buffer.size() != pixel_count) {
        cpu_accumulation_buffer.resize(pixel_count, Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    cpu_accumulation_valid = true;

    // Ensure variance buffer is allocated for adaptive sampling
    if (cpu_variance_buffer.size() != pixel_count) {
        cpu_variance_buffer.resize(pixel_count, 0.0f);
    }

    // Camera change detection
    if (scene.camera) {
        uint64_t current_hash = computeCPUCameraHash(*scene.camera);
        bool is_first = (cpu_last_camera_hash == 0);

        if (current_hash != cpu_last_camera_hash) {
            // Camera changed - reset accumulation and variance
            std::fill(cpu_accumulation_buffer.begin(), cpu_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
            std::fill(cpu_variance_buffer.begin(), cpu_variance_buffer.end(), 0.0f);
            cpu_accumulated_samples = 0;
            cpu_pixel_list_valid = false;  // Force pixel list rebuild + shuffle

            if (!is_first) {
                // SCENE_LOG_INFO("CPU: Camera changed - resetting accumulation");
            }

            cpu_last_camera_hash = current_hash;
        }
    }

    // Auto-rebuild hair BVH for live updates
    if (hairSystem.isBVHDirty()) {
        hairSystem.buildBVH(!hideInterpolatedHair);
    }

    // Check if already complete
    // Priority: 1. Override (Animation), 2. Final Render Mode (F12), 3. Viewport Settings
    int target_max_samples = 100;
    if (override_target_samples > 0) {
        target_max_samples = override_target_samples;
    }
    else if (render_settings.is_final_render_mode) {
        target_max_samples = render_settings.final_render_samples > 0 ? render_settings.final_render_samples : 128;
    }
    else {
        target_max_samples = render_settings.max_samples > 0 ? render_settings.max_samples : 100;
    }
    if (cpu_accumulated_samples >= target_max_samples) {
        // SCENE_LOG_INFO("CPU Render returned early: cpu_accumulated_samples(" + std::to_string(cpu_accumulated_samples) + ") >= target_max_samples(" + std::to_string(target_max_samples) + ")");
        return; // Already done
    }

    // Multi-threaded rendering with accumulation
    unsigned int num_threads = std::thread::hardware_concurrency();

    // UI/OS Responsiveness Optimization:
    // Leave 2 threads free if we have plenty (e.g., > 4 cores)
    // Leave 1 thread free if we have few (e.g., <= 4 cores)
    if (num_threads > 4) num_threads -= 2;
    else if (num_threads > 1) num_threads -= 1;

    // Safety cap
    if (num_threads > 32) num_threads = 32;

    std::vector<std::thread> threads;

    // ===========================================================================
    // OPTIMIZATION: Cache pixel list - only rebuild + shuffle when necessary
    // This avoids O(n) allocation + O(n) shuffle on EVERY pass
    // ===========================================================================
    if (!cpu_pixel_list_valid || cpu_cached_pixel_list.size() != pixel_count) {
        // Rebuild pixel list (resolution changed or first time)
        cpu_cached_pixel_list.clear();
        cpu_cached_pixel_list.reserve(pixel_count);
        for (int j = 0; j < image_height; ++j) {
            for (int i = 0; i < image_width; ++i) {
                cpu_cached_pixel_list.emplace_back(i, j);
            }
        }

        // Shuffle ONCE for visual distribution (random device seeded)
        std::shuffle(cpu_cached_pixel_list.begin(), cpu_cached_pixel_list.end(),
            std::mt19937(std::random_device{}()));

        cpu_pixel_list_valid = true;
    }

    // Use cached pixel list (no per-pass allocation or shuffle!)
    const auto& pixel_list = cpu_cached_pixel_list;

    // Sparse progressive: First few samples render fewer pixels for faster preview
    // Sample 1: 1/16 pixels, Sample 2: 1/8, Sample 3: 1/4, Sample 5+: all
    int sparse_divisor = 1;
    if (cpu_accumulated_samples == 0) sparse_divisor = 16;  // First pass: 1/16 pixels
    else if (cpu_accumulated_samples == 1) sparse_divisor = 8;
    else if (cpu_accumulated_samples == 2) sparse_divisor = 4;
    else if (cpu_accumulated_samples == 3) sparse_divisor = 2;

    size_t pixels_to_render = pixel_list.size() / sparse_divisor;
    if (pixels_to_render < 1000) pixels_to_render = pixel_list.size();  // Safety minimum

    std::atomic<int> next_pixel_index{ 0 };
    std::atomic<bool> should_stop{ false };

    // Worker function for progressive accumulation
    auto progressive_worker = [&](int thread_id) {
        std::mt19937 rng(std::random_device{}() + thread_id + cpu_accumulated_samples * 1337);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        while (!should_stop.load(std::memory_order_relaxed)) {
            // Check global stop flag or FORCE STOP from UI
            if (rendering_stopped_cpu.load(std::memory_order_relaxed) || force_stop_rendering.load(std::memory_order_relaxed)) {
                should_stop.store(true, std::memory_order_relaxed);
                break;
            }

            int idx = next_pixel_index.fetch_add(1, std::memory_order_relaxed);
            if (idx == 0) {
                 // Print first pixel trace to confirm thread starts
                 // SCENE_LOG_INFO("Worker thread started processing pixels.");
            }
            if (idx >= static_cast<int>(pixels_to_render)) break;

            int i = pixel_list[idx].first;
            int j = pixel_list[idx].second;
            int pixel_index = j * image_width + i;

            // ===================================================================
            // ADAPTIVE SAMPLING - Early exit for converged pixels
            // Same logic as GPU: skip if variance is below threshold AND we have enough samples
            // ===================================================================
            // ADAPTIVE SAMPLING - Coefficient of Variation (CV = ?/�) based convergence
            // Industry standard: relative error is consistent across all brightness levels
            // ===================================================================
            if (render_settings.use_adaptive_sampling) {
                Vec4& accum_check = cpu_accumulation_buffer[pixel_index];
                float prev_samples_check = accum_check.w;
                float current_variance = cpu_variance_buffer[pixel_index];

                // Compute mean luminance for CV calculation
                float mean_lum_check = 0.2126f * accum_check.x + 0.7152f * accum_check.y + 0.0722f * accum_check.z;

                // Coefficient of Variation: CV = ? / � (relative standard deviation)
                // This gives consistent convergence across dark and bright regions
                float cv = (mean_lum_check > 0.00001f) ? std::sqrt(current_variance) / mean_lum_check : 1.0f;

                // OIDN-aware threshold: if denoiser is enabled, we can be more aggressive
                // OIDN handles low-frequency noise well, so we can stop earlier
                float effective_threshold = render_settings.variance_threshold;
                if (render_settings.use_denoiser) {
                    effective_threshold *= 2.0f;  // OIDN will clean up remaining noise
                }

                // Check convergence: enough samples AND low relative variance (CV)
                if (prev_samples_check >= render_settings.min_samples &&
                    current_variance > 0.0f &&  // Variance must be computed
                    cv < effective_threshold) {
                    // Pixel has converged - skip ray tracing but still write existing color to surface
                    // This ensures the display stays updated even when pixels are skipped
                    Vec3 cached_color(accum_check.x, accum_check.y, accum_check.z);

                    // Tone mapping for display (same as below)
                    auto toSRGB_fast = [](float c) {
                        return (c <= 0.0031308f) ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.2f) - 0.055f;
                        };

                    float exp_factor = scene.camera ? (scene.camera->auto_exposure ?
                        std::pow(2.0f, scene.camera->ev_compensation) : 1.0f) : 1.0f;
                    Vec3 exposed = cached_color * exp_factor;
                    exposed.x = exposed.x / (exposed.x + 1.0f);
                    exposed.y = exposed.y / (exposed.y + 1.0f);
                    exposed.z = exposed.z / (exposed.z + 1.0f);

                    int r = static_cast<int>(255 * std::clamp(toSRGB_fast(std::max(0.0f, exposed.x)), 0.0f, 1.0f));
                    int g = static_cast<int>(255 * std::clamp(toSRGB_fast(std::max(0.0f, exposed.y)), 0.0f, 1.0f));
                    int b = static_cast<int>(255 * std::clamp(toSRGB_fast(std::max(0.0f, exposed.z)), 0.0f, 1.0f));

                    Uint32* pixels_ptr = static_cast<Uint32*>(surface->pixels);
                    int screen_idx = (surface->h - 1 - j) * (surface->pitch / 4) + i;
                    pixels_ptr[screen_idx] = SDL_MapRGB(surface->format, r, g, b);

                    continue;  // Skip to next pixel - FAST PATH!
                }
            }

            Vec3 color_sum(0.0f);

            // Render samples for this pass in 8-wide packets
            for (int s = 0; s < samples_this_pass; ++s) {
                float u = (float(i) + dist(rng)) / (image_width - 1);
                float v = (float(j) + dist(rng)) / (image_height - 1);

                Ray r = scene.camera->get_ray(u, v);

                // Scalar path tracing call
                color_sum = color_sum + ray_color(r, scene.bvh.get(), scene.lights, scene.background_color, max_depth, 0, scene);
            }

            Vec3 new_color = color_sum / float(samples_this_pass);

            // Accumulate with previous samples
            Vec4& accum = cpu_accumulation_buffer[pixel_index];
            float prev_samples = accum.w;
            Vec3 blended_color = new_color;
            float new_total_samples = float(samples_this_pass);

            if (prev_samples > 0.0f) {
                // Progressive blend
                new_total_samples = prev_samples + samples_this_pass;
                Vec3 prev_color(accum.x, accum.y, accum.z);
                blended_color = (prev_color * prev_samples + new_color * samples_this_pass) / new_total_samples;

                accum.x = blended_color.x;
                accum.y = blended_color.y;
                accum.z = blended_color.z;
                accum.w = new_total_samples;
            }
            else {
                // First sample
                accum.x = new_color.x;
                accum.y = new_color.y;
                accum.z = new_color.z;
                accum.w = float(samples_this_pass);
            }

            // ===================================================================
            // ADAPTIVE SAMPLING - Variance calculation for next pass decision
            // ===================================================================
            // ADAPTIVE SAMPLING - Welford's Online Variance Algorithm
            // More numerically stable than naive variance calculation
            // Stores variance (?�), CV is computed at check time as ?/�
            // ===================================================================
            if (render_settings.use_adaptive_sampling) {
                // Compute luminance (Rec.709 weights)
                auto compute_luminance = [](const Vec3& c) {
                    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
                    };

                // new_color = this pass's raw result, blended_color = accumulated mean
                float new_lum = compute_luminance(new_color);
                float mean_lum = compute_luminance(blended_color);

                // Welford's online algorithm for running variance
                // More stable than naive E[X] - E[X] approach
                float diff = new_lum - mean_lum;
                float prev_variance = cpu_variance_buffer[pixel_index];

                // Incremental variance update: Var_n = Var_{n-1} + (x - mean) / n - Var_{n-1} / n
                // Simplified: exponential moving average with decreasing alpha
                float alpha = 1.0f / std::max(new_total_samples, 2.0f);
                float updated_variance = prev_variance * (1.0f - alpha) + (diff * diff) * alpha;

                // Clamp to prevent numerical issues (very bright fireflies)
                cpu_variance_buffer[pixel_index] = std::clamp(updated_variance, 0.0f, 100.0f);
            }

            // Use blended color for display
            new_color = blended_color;

            // Write to surface with tone mapping
            auto toSRGB = [](float c) {
                if (c <= 0.0031308f)
                    return 12.92f * c;
                else
                    return 1.055f * std::pow(c, 1.0f / 2.2f) - 0.055f;
                };

            // Calculate Exposure Factor (MUST match GPU path - OptixWrapper for consistency)
            float exposure_factor = 1.0f;
            if (scene.camera) {
                if (scene.camera->auto_exposure) {
                    // Auto Exposure: use EV compensation only
                    exposure_factor = std::pow(2.0f, scene.camera->ev_compensation);
                }
                else if (scene.camera->use_physical_exposure) {
                    // Physical Manual Mode: calculate from ISO/Shutter/F-stop (matches GPU)
                    float iso_mult = (scene.camera->iso_preset_index >= 0 && scene.camera->iso_preset_index < (int)CameraPresets::ISO_PRESET_COUNT) ?
                                     CameraPresets::ISO_PRESETS[scene.camera->iso_preset_index].exposure_multiplier : 1.0f;
                    float shutter_time = (scene.camera->shutter_preset_index >= 0 && scene.camera->shutter_preset_index < (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT) ?
                                         CameraPresets::SHUTTER_SPEED_PRESETS[scene.camera->shutter_preset_index].speed_seconds : 0.004f;

                    // Use F-Stop Number
                    float f_number = 16.0f;
                    if (scene.camera->fstop_preset_index > 0 && scene.camera->fstop_preset_index < (int)CameraPresets::FSTOP_PRESET_COUNT) {
                        f_number = CameraPresets::FSTOP_PRESETS[scene.camera->fstop_preset_index].f_number;
                    }
                    else {
                        // Custom Mode: Estimate f-number from aperture (diameter/radius)
                        if (scene.camera->aperture > 0.001f)
                            f_number = 0.8f / scene.camera->aperture;
                        else
                            f_number = 16.0f;
                    }
                    float aperture_sq = f_number * f_number + 1e-6f;
                    float ev_comp = std::pow(2.0f, scene.camera->ev_compensation);
                    float current_val = (iso_mult * shutter_time) / (aperture_sq + 0.001f);
                    float baseline_val = 0.00003125f; // Sunny 16 baseline
                    exposure_factor = (current_val / baseline_val) * ev_comp;
                }
                else {
                    // Manual Mode (non-physical): use EV compensation only
                    exposure_factor = std::pow(2.0f, scene.camera->ev_compensation);
                }

            }


            // Apply Reinhard Tone Mapping (CPU Parity with GPU)
            // GPU uses: color / (color + 1.0f) in make_color
            // Note: Exposure is applied BEFORE tone mapping in some pipelines, 
            // but GPU make_color applies it AFTER "new_color" is computed.
            // Wait, GPU raygen applies exposure to new_color, THEN calls make_color.
            // So Tone Mapping happens on EXPOSED color.

            Vec3 exposed_color = new_color * exposure_factor;

            // Reinhard Operator
            exposed_color.x = exposed_color.x / (exposed_color.x + 1.0f);
            exposed_color.y = exposed_color.y / (exposed_color.y + 1.0f);
            exposed_color.z = exposed_color.z / (exposed_color.z + 1.0f);

            int r = static_cast<int>(255 * std::clamp(toSRGB(std::max(0.0f, exposed_color.x)), 0.0f, 1.0f));
            int g = static_cast<int>(255 * std::clamp(toSRGB(std::max(0.0f, exposed_color.y)), 0.0f, 1.0f));
            int b = static_cast<int>(255 * std::clamp(toSRGB(std::max(0.0f, exposed_color.z)), 0.0f, 1.0f));

            Uint32* pixels = static_cast<Uint32*>(surface->pixels);
            int screen_index = (surface->h - 1 - j) * (surface->pitch / 4) + i;
            pixels[screen_index] = SDL_MapRGB(surface->format, r, g, b);
        }
        };

    // Launch threads
    auto start_time = std::chrono::high_resolution_clock::now();

    for (unsigned int t = 0; t < num_threads; ++t) {
        threads.emplace_back(progressive_worker, t);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float pass_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Check if stopped
    if (rendering_stopped_cpu.load()) {
        SCENE_LOG_WARN("CPU Render stopped by user");
        return;
    }

    cpu_accumulated_samples += samples_this_pass;
        // SCENE_LOG_INFO("CPU Render Pass Completed. Total samples: " + std::to_string(cpu_accumulated_samples));

    // Update window title with progress
    if (window) {
        extern std::string active_model_path;
        std::string projectName = active_model_path;
        if (projectName.empty() || projectName == "Untitled") {
            projectName = "Untitled";
        } else {
            // Extract filename from path
            size_t lastSlash = projectName.find_last_of("\\/");
            if (lastSlash != std::string::npos) {
                projectName = projectName.substr(lastSlash + 1);
            }
        }

        float progress = 100.0f * cpu_accumulated_samples / target_max_samples;
        std::string title = "RayTrophi Studio [" + projectName + "] - CPU - Sample " + std::to_string(cpu_accumulated_samples) +
            "/" + std::to_string(target_max_samples) +
            " (" + std::to_string(int(progress)) + "%) - " +
            std::to_string(int(pass_ms)) + "ms/sample";
        SDL_SetWindowTitle(window, title.c_str());
    }
}

// Add this implementation at the end of Renderer.cpp before the closing brace


// Rebuild OptiX geometry after scene modifications (deletion/addition)
void Renderer::rebuildBackendGeometry(SceneData& scene) {
    // Rebuild geometry TLAS
    rebuildBackendGeometryWithList(scene.world.objects);
    // Sync all volumes (VDB/Gas)
    VolumetricRenderer::syncVolumetricData(scene, m_backend);
}


void Renderer::rebuildBackendGeometryWithList(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!m_backend) {
        SCENE_LOG_WARN("[Backend] Cannot rebuild - no backend pointer");
        return;
    }
    if (render_settings.use_vulkan) {
        m_backend->updateGeometry(objects);
        return;
    }
    if (!g_hasCUDA) return;
      Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(m_backend);
    OptixWrapper* optix_gpu_ptr = optixBackend ? optixBackend->getOptixWrapper() : nullptr;

    // Handle empty list
    size_t hairCount = hairSystem.getTotalStrandCount();
    SCENE_LOG_INFO("[OptiX Rebuild] Start Rebuild. Objects: " + std::to_string(objects.size()) + 
                   ", Hair Strands: " + std::to_string(hairCount));

    if (objects.empty() && hairCount == 0) {
        SCENE_LOG_INFO("[OptiX] Scene empty (No objects, no hair), clearing GPU scene");
        if (optix_gpu_ptr) optix_gpu_ptr->clearScene();
        m_backend->resetAccumulation();
        return;
    }


    // Global flag to block concurrent updates
    extern bool g_optix_rebuild_in_progress;
    g_optix_rebuild_in_progress = true;

    try {
        // Parallel Extraction of Triangles from Hittable objects
        size_t num_objects = objects.size();
        std::vector<std::shared_ptr<Triangle>> triangles;
        triangles.reserve(num_objects);

        if (num_objects > 1000) {
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;

            size_t chunk_size = (num_objects + num_threads - 1) / num_threads;
            std::vector<std::future<std::vector<std::shared_ptr<Triangle>>>> futures;

            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, num_objects);
                if (start >= end) continue;

                futures.push_back(std::async(std::launch::async,
                    [&objects, start, end]() {
                        std::vector<std::shared_ptr<Triangle>> local_tris;
                        local_tris.reserve((end - start));
                        for (size_t i = start; i < end; ++i) {
                            if (auto tri = std::dynamic_pointer_cast<Triangle>(objects[i])) {
                                if (tri->visible) {
                                    local_tris.push_back(tri);
                                }
                            }
                        }
                        return local_tris;
                    }
                ));
            }

            for (auto& f : futures) {
                auto part = f.get();
                triangles.insert(triangles.end(), part.begin(), part.end());
            }
        }
        else {
            std::function<void(const std::shared_ptr<Hittable>&)> collect;
            collect = [&](const std::shared_ptr<Hittable>& obj) {
                if (!obj) return;
                if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                    if (tri->visible) triangles.push_back(tri);
                } else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
                    for (auto& child : list->objects) collect(child);
                } else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                    // Collect from source if it exists
                    if (inst->source_triangles) {
                        for (auto& tri : *inst->source_triangles) triangles.push_back(tri);
                    }
                } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
                    if (bvh->left) collect(bvh->left);
                    if (bvh->right) collect(bvh->right);
                }
            };
            for (const auto& obj : objects) collect(obj);
        }

        OptixGeometryData optix_data;
        if (!triangles.empty()) {
            optix_data = assimpLoader.convertTrianglesToOptixData(triangles);
        }

        // ===================================================================
        // HAIR GEOMETRY DATA (OptiX Curve Primitives)
        // ===================================================================
        // TODO: Fix OptiX 8 compatibility (Error 7001). Disabled for stability.
        // Hair Geometry Generation (OptiX Curves)
        if (hairSystem.getTotalStrandCount() > 0) {
            auto groomNames = hairSystem.getGroomNames();
            for (const auto& groomName : groomNames) {
                std::vector<float> hairVertices4;
                std::vector<unsigned int> hairIndices;
                std::vector<uint32_t> hairStrandIDs;
                std::vector<float> hairTangents3;
                std::vector<float> hairRootUVs2;
                std::vector<float> hairStrandVs;
                size_t vertexCount = 0;
                size_t segmentCount = 0;
                
                Hair::HairMaterialParams matParams;
                int matID = 0;
                int meshMatID = -1;
                
                // Call with all 13 arguments explicitly
                bool isSpline = hairSystem.getOptiXCurveDataByGroom(
                    groomName,          // 1
                    hairVertices4,      // 2
                    hairIndices,        // 3
                    hairStrandIDs,      // 4
                    hairTangents3,      // 5
                    hairRootUVs2,       // 6
                    hairStrandVs,       // 7
                    vertexCount,        // 8
                    segmentCount,       // 9
                    matParams,          // 10
                    matID,              // 11
                    meshMatID,          // 12
                    true                // 13 (includeInterpolated)
                );
                
                if (segmentCount > 0) {
                    CurveGeometry curve_geom;
                    curve_geom.name = groomName; // Use actual groom name
                    curve_geom.vertex_count = vertexCount;
                    curve_geom.segment_count = segmentCount;
                    curve_geom.use_bspline = isSpline; 
                    
                    // Determine material ID from this groom
                    curve_geom.material_id = matID;
                    curve_geom.mesh_material_id = meshMatID;
                    curve_geom.hair_material = Hair::HairBSDF::convertToGpu(matParams); // Fix: assign material properly on load
                    curve_geom.strand_v = hairStrandVs;
                    curve_geom.strand_ids = hairStrandIDs;
                    
                    // Copy Root UVs
                    curve_geom.root_uvs.resize(segmentCount);
                    for (size_t i = 0; i < segmentCount; ++i) {
                        curve_geom.root_uvs[i] = make_float2(hairRootUVs2[i*2], hairRootUVs2[i*2+1]);
                    }

                    // Copy to CurveGeometry vectors
                    curve_geom.vertices.resize(vertexCount);
                    for (size_t i = 0; i < vertexCount; ++i) {
                        curve_geom.vertices[i] = make_float4(hairVertices4[i*4], hairVertices4[i*4+1], hairVertices4[i*4+2], hairVertices4[i*4+3]);
                    }
                    curve_geom.indices = hairIndices;
                    curve_geom.tangents.resize(segmentCount);
                    for (size_t i = 0; i < segmentCount; ++i) {
                        curve_geom.tangents[i] = make_float3(hairTangents3[i*3], hairTangents3[i*3+1], hairTangents3[i*3+2]);
                    }
                    
                    // Copy root UVs
                    curve_geom.root_uvs.resize(segmentCount);
                    for (size_t i = 0; i < segmentCount; ++i) {
                        curve_geom.root_uvs[i] = make_float2(hairRootUVs2[i*2], hairRootUVs2[i*2+1]);
                    }

                    optix_data.curves.push_back(curve_geom);
                   // SCENE_LOG_INFO("[Hair GPU] Uploaded groom '" + groomName + "' (" + std::to_string(segmentCount) + " segments) with MaterialID=" + std::to_string(matID));
                }
            }
        }

        if (optix_gpu_ptr) {
            optix_gpu_ptr->validateMaterialIndices(optix_data);
            optix_gpu_ptr->buildFromDataTLAS(optix_data, objects);
        }
        
        m_backend->resetAccumulation();


        if (triangles.size() > 1000) {
           // SCENE_LOG_INFO("[OptiX] Geometry rebuilt (Snapshot) - " + std::to_string(triangles.size()) + " triangles");
        }
    }
    catch (std::exception& e) {
        SCENE_LOG_ERROR("[OptiX] Rebuild failed: " + std::string(e.what()));
        optix_gpu_ptr->clearScene();
    }

    g_optix_rebuild_in_progress = false;
}


//
// Performance: ~1-5ms vs ~200-500ms for rebuildOptiXGeometry
// ============================================================================
void Renderer::updateBackendMaterials(SceneData& scene) {
    if (!m_backend) return;

    // Prefer OptiX/CUDA path if available
    Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(m_backend);
    OptixWrapper* optix_gpu_ptr = (optixBackend && g_hasCUDA) ? optixBackend->getOptixWrapper() : nullptr;

    if (optix_gpu_ptr) {
        try {
            // Existing OptiX-specific material sync (unchanged)
            auto& mgr = MaterialManager::getInstance();
            const auto& all_materials = mgr.getAllMaterials();
            if (all_materials.empty()) return;

            std::vector<GpuMaterial> gpu_materials;
            std::vector<OptixGeometryData::VolumetricInfo> volumetric_info;
            gpu_materials.reserve(all_materials.size());
            volumetric_info.reserve(all_materials.size());

            for (size_t i = 0; i < all_materials.size(); ++i) {
                const auto& mat = all_materials[i];
                if (!mat) continue;

                GpuMaterial gpu_mat = {};
                OptixGeometryData::VolumetricInfo vol_info = {};

                if (mat->type() == MaterialType::Volumetric) {
                    Volumetric* vol_mat = static_cast<Volumetric*>(mat.get());
                    if (vol_mat) {
                        vol_info.is_volumetric = 1;
                        Vec3 albedo = vol_mat->getAlbedo();
                        Vec3 emission = vol_mat->getEmissionColor();
                        vol_info.density = static_cast<float>(vol_mat->getDensity());
                        vol_info.absorption = static_cast<float>(vol_mat->getAbsorption());
                        vol_info.scattering = static_cast<float>(vol_mat->getScattering());
                        vol_info.albedo = make_float3(albedo.x, albedo.y, albedo.z);
                        vol_info.emission = make_float3(emission.x, emission.y, emission.z);
                        vol_info.g = static_cast<float>(vol_mat->getG());
                        vol_info.step_size = vol_mat->getStepSize();
                        vol_info.max_steps = vol_mat->getMaxSteps();
                        vol_info.noise_scale = vol_mat->getNoiseScale();

                        vol_info.multi_scatter = vol_mat->getMultiScatter();
                        vol_info.g_back = vol_mat->getGBack();
                        vol_info.lobe_mix = vol_mat->getLobeMix();
                        vol_info.light_steps = vol_mat->getLightSteps();
                        vol_info.shadow_strength = vol_mat->getShadowStrength();

                        vol_info.aabb_min = make_float3(0, 0, 0);
                        vol_info.aabb_max = make_float3(1, 1, 1);

                        gpu_mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
                        gpu_mat.roughness = 1.0f;
                        gpu_mat.metallic = 0.0f;
                        gpu_mat.emission = make_float3(0.0f, 0.0f, 0.0f);
                        gpu_mat.ior = 1.0f;
                        gpu_mat.transmission = 0.0f;
                        gpu_mat.opacity = 1.0f;

                        if (vol_mat->hasVDBVolume()) {
                            void* grid_ptr = VDBVolumeManager::getInstance().getGPUGrid(vol_mat->getVDBVolumeID());
                            vol_info.nanovdb_grid = grid_ptr;
                            vol_info.has_nanovdb = (grid_ptr != nullptr) ? 1 : 0;
                        }
                    }
                } else {
                    auto getCudaTex = [](const std::shared_ptr<Texture>& tex) -> cudaTextureObject_t {
                        extern bool g_hasOptix;
                        if (tex && tex->is_loaded()) {
                            if (!tex->is_gpu_uploaded && g_hasOptix) tex->upload_to_gpu();
                            return tex->get_cuda_texture();
                        }
                        return 0;
                    };

                    if (mat->type() == MaterialType::PrincipledBSDF) {
                        PrincipledBSDF* pbsdf = static_cast<PrincipledBSDF*>(mat.get());
                        if (mat->gpuMaterial) gpu_mat = *(mat->gpuMaterial);
                        bool is_water = mat->materialName.find("Water") != std::string::npos || mat->materialName.find("River") != std::string::npos;
                        if (!is_water) {
                            Vec3 alb = pbsdf->albedoProperty.color;
                            gpu_mat.albedo = make_float3((float)alb.x, (float)alb.y, (float)alb.z);
                            gpu_mat.roughness = (float)pbsdf->roughnessProperty.color.x;
                            gpu_mat.metallic = (float)pbsdf->metallicProperty.intensity;
                            Vec3 em = pbsdf->emissionProperty.color;
                            float emStr = pbsdf->emissionProperty.intensity;
                            gpu_mat.emission = make_float3((float)em.x * emStr, (float)em.y * emStr, (float)em.z * emStr);
                            gpu_mat.ior = pbsdf->ior;
                            gpu_mat.transmission = pbsdf->transmission;
                            gpu_mat.opacity = pbsdf->opacityProperty.alpha;
                            gpu_mat.subsurface = pbsdf->subsurface;
                            Vec3 sssColor = pbsdf->subsurfaceColor;
                            gpu_mat.subsurface_color = make_float3((float)sssColor.x, (float)sssColor.y, (float)sssColor.z);
                            Vec3 sssRadius = pbsdf->subsurfaceRadius;
                            gpu_mat.subsurface_radius = make_float3((float)sssRadius.x, (float)sssRadius.y, (float)sssRadius.z);
                            gpu_mat.subsurface_scale = pbsdf->subsurfaceScale;
                            gpu_mat.subsurface_anisotropy = pbsdf->subsurfaceAnisotropy;
                            gpu_mat.subsurface_ior = pbsdf->subsurfaceIOR;
                            gpu_mat.clearcoat = pbsdf->clearcoat;
                            gpu_mat.clearcoat_roughness = pbsdf->clearcoatRoughness;
                            gpu_mat.translucent = pbsdf->translucent;
                            gpu_mat.anisotropic = pbsdf->anisotropic;
                            gpu_mat.sheen = pbsdf->sheen;
                            gpu_mat.sheen_tint = pbsdf->sheen_tint;
                        }

                        gpu_mat.albedo_tex      = getCudaTex(pbsdf->albedoProperty.texture);
                        gpu_mat.normal_tex      = getCudaTex(pbsdf->normalProperty.texture);
                        gpu_mat.roughness_tex   = getCudaTex(pbsdf->roughnessProperty.texture);
                        gpu_mat.metallic_tex    = getCudaTex(pbsdf->metallicProperty.texture);
                        gpu_mat.emission_tex    = getCudaTex(pbsdf->emissionProperty.texture);
                        gpu_mat.opacity_tex     = getCudaTex(pbsdf->opacityProperty.texture);
                        gpu_mat.transmission_tex= getCudaTex(pbsdf->transmissionProperty.texture);
                        gpu_mat.height_tex      = getCudaTex(pbsdf->heightProperty.texture);
                    } else if (mat->gpuMaterial) {
                        gpu_mat = *mat->gpuMaterial;
                        vol_info.is_volumetric = 0;
                    } else {
                        gpu_mat.albedo = make_float3(0.8f, 0.8f, 0.8f);
                        gpu_mat.roughness = 0.5f;
                        gpu_mat.metallic = 0.0f;
                        gpu_mat.emission = make_float3(0.0f, 0.0f, 0.0f);
                        gpu_mat.ior = 1.5f;
                        gpu_mat.transmission = 0.0f;
                        gpu_mat.opacity = 1.0f;
                        vol_info.is_volumetric = 0;
                    }
                }

                gpu_materials.push_back(gpu_mat);
                volumetric_info.push_back(vol_info);
            }

            if (!gpu_materials.empty()) {
                optix_gpu_ptr->updateMaterialBuffer(gpu_materials);
                if (!volumetric_info.empty()) optix_gpu_ptr->updateSBTVolumetricData(volumetric_info);
                optix_gpu_ptr->syncSBTMaterialData(gpu_materials, true);
            }

            optix_gpu_ptr->updateHairMaterialsOnly(hairSystem);
            setHairMaterial(hairMaterial);
            VolumetricRenderer::syncVolumetricData(scene, m_backend);
            optix_gpu_ptr->resetAccumulation();
        }
        catch (std::exception& e) {
            SCENE_LOG_ERROR("[OptiX] updateOptiXMaterialsOnly failed: " + std::string(e.what()));
        }

        return;
    }

    // Generic backend path (Vulkan, CPU backends etc.)
    try {
        auto& mgr = MaterialManager::getInstance();
        const auto& all_materials = mgr.getAllMaterials();
        if (all_materials.empty()) return;

        std::vector<Backend::IBackend::MaterialData> backendMaterials;
        backendMaterials.reserve(all_materials.size());

        for (const auto& mat : all_materials) {
            Backend::IBackend::MaterialData data = {};
            if (!mat) { backendMaterials.push_back(data); continue; }

            data.albedo = mat->albedo;
            data.ior = mat->ior;
            data.opacity = 1.0f;

            if (mat->type() == MaterialType::PrincipledBSDF) {
                PrincipledBSDF* pbsdf = static_cast<PrincipledBSDF*>(mat.get());
                data.albedo = pbsdf->albedoProperty.color;
                data.opacity = pbsdf->opacityProperty.alpha;
                data.roughness = (float)pbsdf->roughnessProperty.color.x;
                data.metallic = (float)pbsdf->metallicProperty.intensity;
                data.clearcoat = pbsdf->clearcoat;
                data.transmission = pbsdf->transmission;
                data.emission = pbsdf->emissionProperty.color;
                data.emissionStrength = pbsdf->emissionProperty.intensity;
                data.ior = pbsdf->ior;

                data.subsurface = pbsdf->subsurface;
                data.subsurfaceColor = pbsdf->subsurfaceColor;
                data.subsurfaceRadius = pbsdf->subsurfaceRadius;
                data.subsurfaceScale = pbsdf->subsurfaceScale;
                data.subsurfaceAnisotropy = pbsdf->subsurfaceAnisotropy;
                data.subsurfaceIOR = pbsdf->subsurfaceIOR;

                data.clearcoatRoughness = pbsdf->clearcoatRoughness;
                data.translucent = pbsdf->translucent;
                data.anisotropic = pbsdf->anisotropic;
                data.sheen = pbsdf->sheen;
                data.sheenTint = pbsdf->sheen_tint;

                auto getH = [](const std::shared_ptr<Texture>& tex) -> int64_t { return tex ? reinterpret_cast<int64_t>(tex.get()) : 0; };
                data.albedoTexture = getH(pbsdf->albedoProperty.texture);
                data.normalTexture = getH(pbsdf->normalProperty.texture);
                data.roughnessTexture = getH(pbsdf->roughnessProperty.texture);
                data.metallicTexture = getH(pbsdf->metallicProperty.texture);
                data.emissionTexture = getH(pbsdf->emissionProperty.texture);
                data.transmissionTexture = getH(pbsdf->transmissionProperty.texture);
                data.opacityTexture = getH(pbsdf->opacityProperty.texture);
                data.heightTexture = getH(pbsdf->heightProperty.texture);
            }

            backendMaterials.push_back(data);
        }

        m_backend->uploadMaterials(backendMaterials);
        VolumetricRenderer::syncVolumetricData(scene, m_backend);
        m_backend->resetAccumulation();
    }
    catch (std::exception& e) {
        SCENE_LOG_ERROR(std::string("[Renderer] updateBackendMaterials failed: ") + e.what());
    }
}

// ============================================================================
// WIND ANIMATION SYSTEM
// ============================================================================



void Renderer::updateWind(SceneData& scene, float time) {
    auto& im = InstanceManager::getInstance();
    bool any_update = false;

    // Iterate all instance groups
    for (auto& groupRef : im.getGroups()) {
        InstanceGroup* group = &groupRef;

        // 1. Initial State Capture (Lazy Init)
        // If we loaded a scene from disk, initial_instances might be empty.
        // We capture the current state as the "Rest Pose".
        if (group->initial_instances.empty() && !group->instances.empty()) {
            group->initial_instances = group->instances;
        }

        // Safety sync if size mismatch (e.g. instances deleted externally without sync)
        if (group->initial_instances.size() != group->instances.size()) {
            group->initial_instances = group->instances; // Reset rest pose to current to capture new layout
        }

        if (!group->wind_settings.enabled) continue;

        // 2. Wind Parameters
        float speed = group->wind_settings.speed;
        float strength = group->wind_settings.strength;
        float turbulence = group->wind_settings.turbulence;
        float wave = group->wind_settings.wave_size > 0.1f ? group->wind_settings.wave_size : 50.0f;
        Vec3 dir = group->wind_settings.direction.normalize();

        // ===========================================================================
        // ENHANCED WIND PARAMETERS
        // ===========================================================================

        // Constant lean: How much the tree bends TOWARDS wind direction (in degrees)
        // This creates the "permanently bent by strong wind" look
        float lean_amount = strength * 0.6f;  // 60% of strength goes to constant lean

        // Oscillation: The back-and-forth sway
        float sway_amount = strength * 0.4f;  // 40% of strength goes to dynamic sway

        // Maximum bend angle (prevents unnatural 90-degree bends)
        const float max_bend_angle = 25.0f;  // degrees

        // 3. Animation Loop
        bool has_active_links = (group->active_hittables.size() == group->instances.size());

        for (size_t i = 0; i < group->instances.size(); ++i) {
            const auto& init = group->initial_instances[i];
            auto& curr = group->instances[i];

            // ===========================================================================
            // MULTI-FREQUENCY OSCILLATION
            // Creates natural, organic movement by combining multiple wave frequencies
            // ===========================================================================

            // Phase based on world position (creates wave propagation effect)
            float pos_phase = (init.position.x * dir.x + init.position.z * dir.z) / wave;
            float t_phase = time * speed;

            // Primary wave: Slow, large movement
            float wave_primary = sinf(pos_phase + t_phase) * 1.0f;

            // Secondary wave: Faster, smaller movement (flutter)
            float wave_secondary = sinf(pos_phase * 2.3f + t_phase * 1.7f) * 0.35f;

            // Tertiary wave: High frequency micro-movement (turbulence)
            float wave_tertiary = sinf(pos_phase * 4.1f + t_phase * 2.9f * turbulence) * 0.15f;

            // Combined oscillation (-1 to +1 range, weighted)
            float oscillation = wave_primary + wave_secondary + wave_tertiary;
            oscillation = oscillation / 1.5f;  // Normalize back to ~[-1, 1]

            // ===========================================================================
            // DIRECTIONAL BENDING
            // Tree leans TOWARDS wind direction + oscillates around that lean
            // ===========================================================================

            // Constant lean towards wind direction
            // R�zgar +X y�n�nde esiyorsa, a�a� +X y�n�ne do�ru e�ilir
            float lean_x = dir.z * lean_amount;   // Lean around X-axis based on wind Z component
            float lean_z = -dir.x * lean_amount;  // Lean around Z-axis based on wind X component

            // Dynamic oscillation around the lean point
            float osc_x = dir.z * oscillation * sway_amount;
            float osc_z = -dir.x * oscillation * sway_amount;

            // Combined rotation = initial + constant lean + oscillation
            float final_rot_x = init.rotation.x + lean_x + osc_x;
            float final_rot_z = init.rotation.z + lean_z + osc_z;

            // ===========================================================================
            // ANGLE CLAMPING (Prevent unnatural over-bending)
            // ===========================================================================
            float total_bend = sqrtf((final_rot_x - init.rotation.x) * (final_rot_x - init.rotation.x) +
                (final_rot_z - init.rotation.z) * (final_rot_z - init.rotation.z));

            if (total_bend > max_bend_angle) {
                float scale = max_bend_angle / total_bend;
                final_rot_x = init.rotation.x + (final_rot_x - init.rotation.x) * scale;
                final_rot_z = init.rotation.z + (final_rot_z - init.rotation.z) * scale;
            }

            // Apply final rotation
            curr.rotation.x = final_rot_x;
            curr.rotation.z = final_rot_z;

            // Update Active HittableInstance if linked
            if (has_active_links) {
                if (auto hittable = group->active_hittables[i].lock()) {
                    if (auto hi = std::dynamic_pointer_cast<HittableInstance>(hittable)) {
                        Matrix4x4 new_mat = curr.toMatrix();
                        hi->setTransform(new_mat);

                        // OPTIMIZATION: Direct GPU Update
                        if (this->m_backend && g_hasCUDA && !hi->optix_instance_ids.empty()) {
                            float t[12];
                            const Matrix4x4& m = new_mat;
                            t[0] = m.m[0][0]; t[1] = m.m[0][1]; t[2] = m.m[0][2]; t[3] = m.m[0][3];
                            t[4] = m.m[1][0]; t[5] = m.m[1][1]; t[6] = m.m[1][2]; t[7] = m.m[1][3];
                            t[8] = m.m[2][0]; t[9] = m.m[2][1]; t[10] = m.m[2][2]; t[11] = m.m[2][3];

                            for (int id : hi->optix_instance_ids) {
                                this->m_backend->updateInstanceTransform(id, t);
                            }
                        }
                    }
                }
            }
        }

        group->gpu_dirty = true;
        any_update = true;

        // ===========================================================================
        // GPU SHADER WIND PARAMETERS
        // Upload wind direction, strength, speed, time for shader-based displacement
        // ===========================================================================
            if (this->m_backend && g_hasCUDA) {
            // Normalize strength to 0-1 range for shader (divide by max angle)
            float normalized_strength = strength / 25.0f;  // max_bend_angle = 25 degrees
            this->m_backend->setWindParams(
                group->wind_settings.direction,
                normalized_strength,
                speed,
                time
            );
        }
        // Commit changes to TLAS if any updates occurred
        if (any_update && this->m_backend && g_hasCUDA) {
            this->m_backend->rebuildAccelerationStructure(); // Fast refit/update
        }
    }
}

// ===============================================================================
// GAS VOLUME OPTIX SYNC
// ===============================================================================
// �����������������������������������������������������������������������������
// Volumetric Sync - Unified logic for Gas and VDB
// �����������������������������������������������������������������������������
void Renderer::updateBackendGasVolumes(SceneData& scene) {
    VolumetricRenderer::syncVolumetricData(scene, m_backend);
}

void Renderer::updateMeshMaterialBinding(const std::string& node_name, int old_mat_id, int new_mat_id) {
    if (m_backend) {
        m_backend->updateInstanceMaterialBinding(node_name, old_mat_id, new_mat_id);
    }
    resetCPUAccumulation(); // Ensure CPU path also resets
}

void Renderer::syncCameraToBackend(const Camera& cam) {
    if (!m_backend) return;
    Backend::CameraParams cp;
    cp.origin = cam.lookfrom;
    cp.lookAt = cam.lookat;
    cp.up = cam.vup;
    cp.fov = cam.vfov;
    cp.aperture = cam.aperture;
    cp.focusDistance = cam.focus_dist;
    cp.aspectRatio = cam.aspect_ratio;
    cp.exposureFactor = cam.getPhysicalExposureMultiplier();
    cp.ev_compensation = cam.ev_compensation;
    cp.isoPresetIndex = cam.iso_preset_index;
    cp.shutterPresetIndex = cam.shutter_preset_index;
    cp.fstopPresetIndex = cam.fstop_preset_index;
    cp.autoAE = cam.auto_exposure;
    cp.usePhysicalExposure = cam.use_physical_exposure;
    cp.motionBlurEnabled = cam.enable_motion_blur;
    cp.vignettingEnabled = cam.enable_vignetting;
    cp.chromaticAberrationEnabled = cam.enable_chromatic_aberration;
    
    // Pro Features
    cp.distortion = cam.distortion;
    cp.lens_quality = cam.lens_quality;
    cp.vignetting_amount = cam.vignetting_amount;
    cp.vignetting_falloff = cam.vignetting_falloff;
    cp.chromatic_aberration = cam.chromatic_aberration;
    cp.chromatic_aberration_r = cam.chromatic_aberration_r;
    cp.chromatic_aberration_b = cam.chromatic_aberration_b;
    cp.camera_mode = (int)cam.camera_mode;
    cp.blade_count = cam.blade_count;
    
    // Shake / Handheld
    cp.shake_enabled = cam.enable_camera_shake;
    cp.shake_intensity = cam.shake_intensity;
    cp.shake_frequency = cam.shake_frequency;
    cp.handheld_sway_amplitude = cam.handheld_sway_amplitude;
    cp.handheld_sway_frequency = cam.handheld_sway_frequency;
    cp.breathing_amplitude = cam.breathing_amplitude;
    cp.breathing_frequency = cam.breathing_frequency;
    cp.enable_focus_drift = cam.enable_focus_drift;
    cp.focus_drift_amount = cam.focus_drift_amount;
    cp.operator_skill = (int)cam.operator_skill;
    cp.ibis_enabled = cam.ibis_enabled;
    cp.ibis_effectiveness = cam.ibis_effectiveness;
    cp.rig_mode = (int)cam.rig_mode;

    m_backend->setCamera(cp);
}
