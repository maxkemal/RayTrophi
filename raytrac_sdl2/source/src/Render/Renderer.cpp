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
#include <future>
#include <thread>

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
#include "InstanceManager.h"  // For wind animation in render_Animation
#include "water_shaders_cpu.h"  // CPU water shader functions
#include "HittableInstance.h"
#include "VolumetricRenderer.h"
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

    try {
        if (g_hasOptix) {
            oidnDevice = oidn::newDevice(oidn::DeviceType::CUDA);
        }
        else {
            oidnDevice = oidn::newDevice(oidn::DeviceType::CPU);
        }
        oidnDevice.commit();
        oidnInitialized = true;

    }
    catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("Failed to initialize OIDN device: ") + e.what());
    }
}

void Renderer::applyOIDNDenoising(SDL_Surface* surface, int numThreads, bool denoise, float blend) {
    if (!surface) return;

    std::lock_guard<std::mutex> lock(oidnMutex);

    // Initialize device if not ready
    if (!oidnInitialized) {
        initOIDN();
        if (!oidnInitialized) return; // Failed to init
    }

    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    int width = surface->w;
    int height = surface->h;
    size_t pixelCount = static_cast<size_t>(width) * height;
    size_t bufferSize = pixelCount * 3;

    // ═══════════════════════════════════════════════════════════════════════════
    // PERFORMANCE CONSTANTS
    // ═══════════════════════════════════════════════════════════════════════════
    constexpr float inv255 = 1.0f / 255.0f;  // Precomputed for multiplication
    const float blend_inv = 1.0f - blend;    // Precomputed inverse blend
    
    // Detect pixel format for direct access (bypass SDL_GetRGB)
    // Most common formats: ARGB8888, RGBA8888, BGRA8888
    const SDL_PixelFormat* fmt = surface->format;
    const bool is_direct_format = (fmt->BytesPerPixel == 4);
    const Uint32 r_mask = fmt->Rmask;
    const Uint32 g_mask = fmt->Gmask;
    const Uint32 b_mask = fmt->Bmask;
    const Uint8 r_shift = fmt->Rshift;
    const Uint8 g_shift = fmt->Gshift;
    const Uint8 b_shift = fmt->Bshift;

    // ===== BUFFER CACHE OPTIMIZATION =====
    // Boyut değiştiyse buffer'ları yeniden oluştur
    bool sizeChanged = (width != oidnCachedWidth || height != oidnCachedHeight);

    if (sizeChanged) {
        // CPU buffer'ları resize et
        oidnColorData.resize(bufferSize);
        oidnOriginalData.resize(bufferSize);  // Original için cache

        // OIDN buffer'larını yeniden oluştur
        try {
            oidnColorBuffer = oidnDevice.newBuffer(bufferSize * sizeof(float));
            oidnOutputBuffer = oidnDevice.newBuffer(bufferSize * sizeof(float));

            // Filter'ı yeniden oluştur ve commit et (boyut değiştiği için)
            oidnFilter = oidnDevice.newFilter("RT");
            oidnFilter.setImage("color", oidnColorBuffer, oidn::Format::Float3, width, height);
            oidnFilter.setImage("output", oidnOutputBuffer, oidn::Format::Float3, width, height);
            oidnFilter.set("hdr", false);
            oidnFilter.set("srgb", true);
            oidnFilter.commit();

            oidnCachedWidth = width;
            oidnCachedHeight = height;

            SCENE_LOG_INFO("OIDN buffers recreated for new resolution: " +
                std::to_string(width) + "x" + std::to_string(height));
        }
        catch (const std::exception& e) {
            SCENE_LOG_ERROR(std::string("OIDN buffer creation failed: ") + e.what());
            return;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIMIZED PIXEL READ - Direct access, no SDL_GetRGB calls
    // Also caches original values for blend (eliminates second read loop)
    // ═══════════════════════════════════════════════════════════════════════════
    if (is_direct_format) {
        // Fast path: Direct bit manipulation (no function calls)
        for (size_t i = 0; i < pixelCount; ++i) {
            Uint32 pixel = pixels[i];
            float r = ((pixel & r_mask) >> r_shift) * inv255;
            float g = ((pixel & g_mask) >> g_shift) * inv255;
            float b = ((pixel & b_mask) >> b_shift) * inv255;
            
            size_t idx = i * 3;
            oidnColorData[idx]     = r;
            oidnColorData[idx + 1] = g;
            oidnColorData[idx + 2] = b;
            
            // Cache original for blend (avoid second read)
            oidnOriginalData[idx]     = r;
            oidnOriginalData[idx + 1] = g;
            oidnOriginalData[idx + 2] = b;
        }
    } else {
        // Fallback: Use SDL_GetRGB for non-standard formats
        for (size_t i = 0; i < pixelCount; ++i) {
            Uint8 r, g, b;
            SDL_GetRGB(pixels[i], surface->format, &r, &g, &b);
            
            size_t idx = i * 3;
            float rf = r * inv255;
            float gf = g * inv255;
            float bf = b * inv255;
            
            oidnColorData[idx]     = rf;
            oidnColorData[idx + 1] = gf;
            oidnColorData[idx + 2] = bf;
            
            oidnOriginalData[idx]     = rf;
            oidnOriginalData[idx + 1] = gf;
            oidnOriginalData[idx + 2] = bf;
        }
    }

    try {
        // Cache'lenmiş buffer'a yaz
        oidnColorBuffer.write(0, bufferSize * sizeof(float), oidnColorData.data());

        // Denoise çalıştır (filter zaten commit edilmiş durumda)
        oidnFilter.execute();

        const char* errorMessage;
        if (oidnDevice.getError(errorMessage) != oidn::Error::None) {
            SCENE_LOG_ERROR(std::string("OIDN error: ") + errorMessage);
        }

        // Sonucu oku
        oidnOutputBuffer.read(0, bufferSize * sizeof(float), oidnColorData.data());
    }
    catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("OIDN execution failed: ") + e.what());
        return;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIMIZED BLEND & WRITE - Single pass, no second pixel read
    // Uses cached original values, direct pixel write
    // ═══════════════════════════════════════════════════════════════════════════
    if (is_direct_format) {
        // Fast path: Direct bit manipulation
        for (size_t i = 0; i < pixelCount; ++i) {
            size_t idx = i * 3;
            
            // Blend: denoised * blend + original * (1 - blend)
            float r_final = oidnColorData[idx]     * blend + oidnOriginalData[idx]     * blend_inv;
            float g_final = oidnColorData[idx + 1] * blend + oidnOriginalData[idx + 1] * blend_inv;
            float b_final = oidnColorData[idx + 2] * blend + oidnOriginalData[idx + 2] * blend_inv;
            
            // Clamp and convert to 8-bit
            Uint8 r = static_cast<Uint8>(std::clamp(r_final, 0.0f, 1.0f) * 255.0f + 0.5f);
            Uint8 g = static_cast<Uint8>(std::clamp(g_final, 0.0f, 1.0f) * 255.0f + 0.5f);
            Uint8 b = static_cast<Uint8>(std::clamp(b_final, 0.0f, 1.0f) * 255.0f + 0.5f);
            
            // Direct pixel write (preserve alpha if present)
            Uint32 alpha = pixels[i] & fmt->Amask;
            pixels[i] = alpha | (r << r_shift) | (g << g_shift) | (b << b_shift);
        }
    } else {
        // Fallback: Use SDL_MapRGB
        for (size_t i = 0; i < pixelCount; ++i) {
            size_t idx = i * 3;
            
            float r_final = oidnColorData[idx]     * blend + oidnOriginalData[idx]     * blend_inv;
            float g_final = oidnColorData[idx + 1] * blend + oidnOriginalData[idx + 1] * blend_inv;
            float b_final = oidnColorData[idx + 2] * blend + oidnOriginalData[idx + 2] * blend_inv;
            
            Uint8 r = static_cast<Uint8>(std::clamp(r_final, 0.0f, 1.0f) * 255.0f + 0.5f);
            Uint8 g = static_cast<Uint8>(std::clamp(g_final, 0.0f, 1.0f) * 255.0f + 0.5f);
            Uint8 b = static_cast<Uint8>(std::clamp(b_final, 0.0f, 1.0f) * 255.0f + 0.5f);
            
            pixels[i] = SDL_MapRGB(surface->format, r, g, b);
        }
    }
}

Renderer::Renderer(int image_width, int image_height, int samples_per_pixel, int max_depth)
    : image_width(image_width), image_height(image_height), aspect_ratio(static_cast<double>(image_width) / image_height), halton_cache(new float[MAX_DIMENSIONS * MAX_SAMPLES_HALTON]), color_processor(image_width, image_height)
{
    initialize_halton_cache();

    frame_buffer.resize(image_width * image_height);
    sample_counts.resize(image_width * image_height, 0);
    max_halton_index = MAX_SAMPLES_HALTON - 1; // Halton dizisi için maksimum indeks

    // Adaptive sampling için bufferlar
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

    // OIDN cache invalidate - buffer'lar bir sonraki denoise'da yeniden oluşturulacak
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

    // Aynı isimde dosya varsa silmeye çalış (zorla yazma)
    if (std::filesystem::exists(file_path)) {
        std::error_code ec;
        std::filesystem::remove(file_path, ec);
        if (ec) {
            SDL_Log("Dosya silinemiyor. Başka bir işlem tarafından kullanılıyor olabilir.");
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

    // sRGB to linear dönüşümü istersen buraya koy
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
                ctx.animator->play(modelClips[0]->name, 0.0f);
                SCENE_LOG_INFO("[Renderer] Created animator for model: " + ctx.importName + " (Clips: " + std::to_string(modelClips.size()) + ")");
            }
        }
    }
}

bool Renderer::updateAnimationWithGraph(SceneData& scene, float deltaTime, bool apply_cpu_skinning) {
    bool anyChanged = false;
    
    // Resize internal matrix buffer to match scene total bone count
    size_t totalBones = scene.boneData.boneNameToIndex.size();
    if (this->finalBoneMatrices.size() != totalBones) {
        this->finalBoneMatrices.assign(totalBones, Matrix4x4::identity());
    }

    // Update each model context independently
    for (auto& ctx : scene.importedModelContexts) {
        if (ctx.useAnimGraph && ctx.graph) {
            // EVALUATE NODE GRAPH (Unity/Unreal style)
            // Passes deltaTime and clips to the graph for full logic execution
            if (ctx.animator) {
                ctx.graph->evalContext.clipsPtr = &ctx.animator->getAllClips();
            }
            AnimationGraph::PoseData pose = ctx.graph->evaluate(deltaTime, scene.boneData);
            
            if (pose.isValid()) {
                anyChanged = true;
                for (size_t i = 0; i < pose.boneTransforms.size() && i < this->finalBoneMatrices.size(); ++i) {
                    if (!(pose.boneTransforms[i] == Matrix4x4::identity())) {
                        this->finalBoneMatrices[i] = pose.boneTransforms[i];
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
             } else {
                 ctx.animator->setRootMotionEnabled(false);
             }
             
             bool changed = ctx.animator->update(deltaTime, scene.boneData);
             if (changed) {
                 anyChanged = true;
                 
                 // Merge model matrices into global buffer
                 const auto& modelMatrices = ctx.animator->getFinalBoneMatrices();
                 for (size_t i = 0; i < modelMatrices.size() && i < this->finalBoneMatrices.size(); ++i) {
                     // MERGE LOGIC: copy only non-identity values or use bone prefix check
                     // Here we use identity check as a simple heuristic because bone indices are unique per model
                     if (!(modelMatrices[i] == Matrix4x4::identity())) {
                         this->finalBoneMatrices[i] = modelMatrices[i];
                     }
                 }
                 
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
    }

    if (!anyChanged && !this->finalBoneMatrices.empty()) {
        // Even if no clip changed, we might need initial matrices
        // Check if we need to return early or re-pose
    }

    // Apply skinning to triangles if CPU skinning is requested
    if (apply_cpu_skinning) {
        for (auto& obj : scene.world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->hasSkinData()) {
                // No longer forcing identity here! 
                // The TransformHandle stores the model's root placement.
                tri->apply_skinning(static_cast<const std::vector<Matrix4x4>&>(this->finalBoneMatrices));
            }
        }
    }

    // Reset CPU accumulation since geometry changed
    resetCPUAccumulation();

    // Clear dirty flags for all animators
    for (auto& ctx : scene.importedModelContexts) {
        if (ctx.animator) {
            ctx.animator->clearDirtyFlag();
        }
    }

    return true;  // Geometry changed
}


bool Renderer::updateAnimationState(SceneData& scene, float current_time, bool apply_cpu_skinning) {
    // ═══════════════════════════════════════════════════════════════════════════
    // GEOMETRY CHANGE TRACKING (Animation Performance Optimization)
    // ═══════════════════════════════════════════════════════════════════════════
    // Track if actual geometry (vertex positions) changed.
    // Return false for camera-only or material-only animations to avoid unnecessary BVH rebuilds.
    bool geometry_changed = false;

    // Optimization: Skip update if time hasn't changed significantly
    if (std::abs(current_time - lastAnimationUpdateTime) < 0.0001f) {
        return false;
    }

    // Time changed - Force Accumulation Reset for CPU Render
    // This prevents ghosting during animation playback
    resetCPUAccumulation();

    lastAnimationUpdateTime = current_time;

    // Unified Animation Check: If we have clips and bones, use the Controller
    bool useAnimationController = !scene.animationDataList.empty() && !scene.boneData.boneNameToIndex.empty();

    if (useAnimationController) {
        static float last_sim_time = -1.0f;
        float deltaTime = (last_sim_time >= 0.0f) ? (current_time - last_sim_time) : (1.0f / 60.0f);
        if (deltaTime < 0.0f || deltaTime > 0.5f) deltaTime = 0.0f; 
        last_sim_time = current_time;

        // Drive the animation

        // Use the new AnimationController system
        // ALWAYS use this if animations exist, legacy path is too unreliable
        geometry_changed = updateAnimationWithGraph(scene, deltaTime, apply_cpu_skinning);
        
        // Return result
        return geometry_changed;
    }

    // --- 1. Adım: Animasyonlu Node Hiyerarşisini Güncelle ---
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

        // ═══════════════════════════════════════════════════════════════════════════
        // CRITICAL FIX: Skip models without animation data
        // ═══════════════════════════════════════════════════════════════════════════
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
                } else {
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

    // --- 0. Adım: Performans Önbelleği Hazırlığı ---
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

    // --- 2. Adım: Grupları Animasyon Türüne Göre Güncelle ---
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

    // --- 3. Adım: Işık ve Kamera Animasyonları (from files AND manual keyframes) ---

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
        // We suspect globalInverse was introducing the "tilt" (sola yatık).
        // We KEEP the manual UP vector flip because user confirmed it fixed "tepetaklak". (Upside down)

        Matrix4x4 animTransform = animatedGlobalNodeTransforms[scene.camera->nodeName];

        Vec3 pos = animTransform.transform_point(Vec3(0, 0, 0));
        // Blender cameras usually point down -Z.
        Vec3 forward = animTransform.transform_vector(Vec3(0, 0, -1)).normalize();

        // FIX: Force Global Up (0, 1, 0) to prevent unwanted Roll/Tilt (sola yatık).
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

    // --- 4. Adım: BVH Güncelle (only if geometry changed) ---
    // OPTIMIZATION: Skip CPU BVH rebuild for camera-only or material-only animations
    if (geometry_changed) {
        auto embree_ptr = std::dynamic_pointer_cast<EmbreeBVH>(scene.bvh);
        if (embree_ptr) {
            embree_ptr->updateGeometryFromTrianglesFromSource(scene.world.objects);
        }
    }

    return geometry_changed;
}

void Renderer::render_Animation(SDL_Surface* surface, SDL_Window* window, SDL_Texture* raytrace_texture, SDL_Renderer* renderer,
    const int total_samples_per_pixel, const int samples_per_pass, float fps, float duration, int start_frame, int end_frame, SceneData& scene,
    const std::string& output_folder, bool use_denoiser, float denoiser_blend,
    OptixWrapper* optix_gpu, bool use_optix, UIContext* ui_ctx) {

    render_finished = false;
    rendering_complete = false;
    rendering_in_progress = true;
    rendering_stopped_cpu = false;
    rendering_stopped_gpu = false;

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    auto start_time = std::chrono::steady_clock::now();
    float frame_time = 1.0f / fps;

    extern RenderSettings render_settings;
    extern bool g_hasOptix; // Ensure we can access global flag

    // DISABLE GRID/OVERLAYS FOR ANIMATION RENDER
    bool original_render_mode = render_settings.is_final_render_mode;
    render_settings.is_final_render_mode = true;

    // Frame range parameters passed from UI
    // int start_frame = render_settings.animation_start_frame; // REMOVED: Using parameter
    // int end_frame = render_settings.animation_end_frame; // REMOVED: Using parameter


    if (start_frame == 0 && end_frame == 0 && !scene.animationDataList.empty()) {
        auto& anim = scene.animationDataList[0];
        if (anim) {
            start_frame = anim->startFrame;
            end_frame = anim->endFrame;
        }
        SCENE_LOG_INFO("Settings range invalid, using animation frame range from file: " + std::to_string(start_frame) + " - " + std::to_string(end_frame));
    }
    else {
        SCENE_LOG_INFO("Using animation frame range from User Settings: " + std::to_string(start_frame) + " - " + std::to_string(end_frame));
    }
    SCENE_LOG_INFO("DEBUG: Final Animation Range -> Start: " + std::to_string(start_frame) + " End: " + std::to_string(end_frame));

    int total_frames = end_frame - start_frame + 1;

    if (!output_folder.empty()) {
        std::filesystem::create_directories(output_folder);
        SCENE_LOG_INFO("Animation frames will be saved to: " + output_folder);
    }

    SCENE_LOG_INFO("Starting animation render: " + std::to_string(total_frames) + " frames (Frame " +
        std::to_string(start_frame) + " to " + std::to_string(end_frame) + ") at " + std::to_string(fps) + " FPS");

    // Check if OptiX is valid
    bool run_optix = use_optix && optix_gpu && g_hasOptix;

    // IMPORTANT: Sync local optix pointer for updateWind updates to work
    // render_Animation receives optix_gpu as argument, but updateWind uses member this->optix_gpu_ptr
    if (run_optix) {
        this->optix_gpu_ptr = optix_gpu;
    }

    if (use_optix && !run_optix) {
        SCENE_LOG_WARN("OptiX requested but not available/valid. Falling back to CPU.");
    }

    for (int frame = start_frame; frame <= end_frame; ++frame) {

        render_settings.animation_current_frame = frame;

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
        if (run_optix && optix_gpu) {
            // Calculate wind transforms on CPU (same as Play Mode)
            InstanceManager::getInstance().updateWind(current_time, scene);

            // Efficiently update instance transforms on GPU (no full rebuild)
            // This is the critical step that was missing - ensures GPU TLAS has updated matrices
            optix_gpu->updateTLASMatricesOnly(scene.world.objects);

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
                if (optix_gpu->isUsingTLAS()) {
                    // FAST PATH: GPU Skinning & Transform Update (Refit)
                    // Pass computed bone matrices to the GPU kernel
                    optix_gpu->updateTLASGeometry(scene.world.objects, this->finalBoneMatrices);
                }
                else {
                    // Fallback: Full Rebuild (Slow)
                    this->rebuildOptiXGeometry(scene, optix_gpu);
                }
            }

            // 1.5. Update GPU Materials (CRITICAL for material keyframe animations!)
            // This uploads the materials updated by updateAnimationState to OptiX
            this->updateOptiXMaterialsOnly(scene, optix_gpu);

            // 2. Set Scene Params
            if (scene.camera) optix_gpu->setCameraParams(*scene.camera);
            optix_gpu->setLightParams(scene.lights);
            // optix_gpu->setBackgroundColor(scene.background_color);
            optix_gpu->setWorld(this->world.getGPUData());

            // 3. Reset Cycles-style Accumulation for new frame
            optix_gpu->resetAccumulation();

            // 4. Render Loop (Accumulate Samples until target reached)
            // 4. Render Loop (Accumulate Samples until target reached)
            std::vector<uchar4> temp_framebuffer(target_surface->w * target_surface->h);

            // Render until max samples reached
            while (!optix_gpu->isAccumulationComplete() && !rendering_stopped_gpu) {
                // Launch progressive render
                // Pass 'nullptr' for window to disable title updates (headless like)
                optix_gpu->launch_random_pixel_mode_progressive(target_surface, nullptr, renderer, target_surface->w, target_surface->h, temp_framebuffer, raytrace_texture);
            }
        }
        else {
            // --- CPU RENDER PATH ---

            // Reset CPU accumulation for new frame
            resetCPUAccumulation();

            // Render until max samples reached
            while (!isCPUAccumulationComplete() && !rendering_stopped_cpu) {
                // For animation CPU render: pass 'total_samples_per_pixel' as the target
                // This ensures we reach the "final render samples" count, not viewport "max samples"
                render_progressive_pass(target_surface, window, scene, 1, total_samples_per_pixel);
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
        // REMOVED: Orphaned error block that was causing premature stop
        // SCENE_LOG_ERROR("Failed to save frame " + std::to_string(frame));
        // rendering_stopped_cpu = true;
        // break;
    }



    rendering_complete = true;
    rendering_in_progress = false;
    render_finished = true;

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
}

void Renderer::updateBVH(SceneData& scene, bool use_embree) {
    if (scene.world.objects.empty()) {
        scene.bvh = nullptr;
        return;
    }
    rebuildBVH(scene, use_embree);
}

void Renderer::create_scene(SceneData& scene, OptixWrapper* optix_gpu_ptr, const std::string& model_path,
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

        // ---- 1. Sahne verilerini sıfırla ----
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

        // ---- 2. MaterialManager'ı temizle ----
        size_t material_count_before = MaterialManager::getInstance().getMaterialCount();
        MaterialManager::getInstance().clear();
        SCENE_LOG_INFO("[MATERIAL CLEANUP] MaterialManager cleared: " + std::to_string(material_count_before) + " materials removed.");

        // ---- 3. CPU Texture Cache'leri temizle ----
        assimpLoader.clearTextureCache();

        // ---- 4. GPU OptiX Texture'larını temizle ----
        if (g_hasOptix && optix_gpu_ptr) {
            try {
                optix_gpu_ptr->destroyTextureObjects();
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

    // ---- 1. Geometri ve animasyon yükle ----
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

    // ---- 2. Kamera ve ışık verisi ----
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
    if (g_hasOptix && optix_gpu_ptr && !append)
    {
        try
        {
            update_progress(78, "Creating OptiX geometry...");
            SCENE_LOG_INFO("OptiX GPU detected. Creating OptiX geometry data...");
            // Use newLoader here
            OptixGeometryData optix_data = newLoader->convertTrianglesToOptixData(loaded_triangles);
            SCENE_LOG_INFO("Converting " + std::to_string(loaded_triangles.size()) + " triangles to OptiX format.");

            update_progress(82, "Validating materials...");
            optix_gpu_ptr->validateMaterialIndices(optix_data);
            SCENE_LOG_INFO("Material indices validated.");

            update_progress(85, "Building OptiX acceleration...");
            optix_gpu_ptr->buildFromData(optix_data);
            SCENE_LOG_INFO("OptiX BVH and acceleration structures built.");

            update_progress(90, "Configuring OptiX camera...");
            if (scene.camera) {
                SCENE_LOG_INFO("Setting up OptiX camera parameters...");
                optix_gpu_ptr->setCameraParams(*scene.camera);
                SCENE_LOG_INFO("OptiX camera configured successfully.");
            }

            update_progress(93, "Setting up OptiX lights...");
            if (!scene.lights.empty()) {
                SCENE_LOG_INFO("Configuring " + std::to_string(scene.lights.size()) + " lights for OptiX...");
                optix_gpu_ptr->setLightParams(scene.lights);
                SCENE_LOG_INFO("OptiX light parameters set successfully.");
            }

            // Consistently sync all volumes using unified logic
            VolumetricRenderer::syncVolumetricData(scene, optix_gpu_ptr);
            update_progress(96, "Finalizing world environment...");
            // optix_gpu_ptr->setBackgroundColor(scene.background_color);
            // Do not force COLOR mode - preserve existing mode set by createDefaultScene or UI
            // this->world.setMode(WORLD_MODE_COLOR); 

            // Only update solid color provided by scene if we are in Color mode, 
            // OR if we want to ensure scene.background_color is synced.
            // But for default scene, we want Nishita.
            if (this->world.getMode() == WORLD_MODE_COLOR) {
                this->world.setColor(scene.background_color);
            }
            optix_gpu_ptr->setWorld(this->world.getGPUData());

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

    // Eğer normal z eksenine paralelse, çok küçük düz yüzeyler için özel durum
    if (N_norm.z < -0.999999f) {
        T = Vec3(0, -1, 0);  // Ters yönlendirilmiş bir tangent
        B = Vec3(-1, 0, 0);
    }
    else {
        // Normalden tangent ve bitangent hesaplaması
        float a = 1.0f / (1.0f + N_norm.z);
        float b = -N_norm.x * N_norm.y * a;

        // Daha hassas bir hesaplama, düz yüzeylerdeki ters dönme sorunu engellenebilir
        T = Vec3(1.0f - N_norm.x * N_norm.x * a, b, -N_norm.x);
        B = Vec3(b, 1.0f - N_norm.y * N_norm.y * a, -N_norm.y);

        // Düz yüzeylerde yönleri doğru tutmak için küçük düzeltme
        if (std::abs(N_norm.z) > 0.9999f) {
            T = Vec3(1.0f, 0.0f, 0.0f);  // x yönüyle tangent düzeltmesi
            B = Vec3(0.0f, 1.0f, 0.0f);  // y yönüyle bitangent düzeltmesi
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
    // Daha iyi dağılım için permütasyon ekliyoruz
    const uint32_t pixel_hash = (x * 73856093) ^ (y * 19349663); // Basit bir hash fonksiyonu
    const uint32_t sample_hash = sample_index * 83492791;

    // Halton dizisinde farklı offsetler kullanıyoruz
    const int base_index = (pixel_hash + sample_hash) % MAX_SAMPLES_HALTON;

    // Farklı asal sayı tabanları kullanarak daha iyi dağılım
    const float u = halton_cache[base_index];                     // Taban 2
    const float v = halton_cache[(base_index + MAX_SAMPLES_HALTON / 2) % MAX_SAMPLES_HALTON]; // Taban 3

    // Stratifikasyon eklemek için jitter
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

// --- Akıllı ışık seçimi ---
int Renderer::pick_smart_light(const std::vector<std::shared_ptr<Light>>& lights, const Vec3& hit_position) {
    int light_count = (int)lights.size();
    if (light_count == 0) return -1;

    // --- 1. Directional light varsa %33 ihtimalle seç ---
    for (int i = 0; i < light_count; i++) {
        if (!lights[i]->visible) continue; // Skip invisible lights
        if (lights[i]->type() == LightType::Directional) {
            if (Vec3::random_float() < 0.33) {
                directional_pick_count++;
                return i;
            }
        }
    }

    // --- 2. Tüm ışık türlerinden ağırlıklı seçim (GPU ile uyumlu) ---
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

    // --- Eğer ağırlık yoksa fallback rastgele seçim ---
    if (total_weight < 1e-6f) {
        return std::clamp(int(Vec3::random_float() * light_count), 0, light_count - 1);
    }

    // --- Weighted seçim ---
    float r = Vec3::random_float() * total_weight;
    float accum = 0.0f;
    for (int i = 0; i < light_count; i++) {
        accum += weights[i];
        if (r <= accum) {
            return i;
        }
    }

    // --- Güvenlik fallback ---
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

    // Malzeme özellikleri
    Vec3 albedo = rec.material->getPropertyValue(rec.material->albedoProperty, uv);
    float metallic = rec.material->getPropertyValue(rec.material->metallicProperty, uv).z;
    float roughness = rec.material->getPropertyValue(rec.material->roughnessProperty, uv).y;
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

        // DÜZELTME: PointLight sınıfı getIntensity içinde zaten falloff (1/d^2) uyguluyor.
        // Burada tekrar uygularsak ışık çok zayıflıyor (1/d^4).
        attenuation = 1.0f;

        // Point Light Specific Boost: Global çarpan kaldırıldı, sadece Point Light 10 kat güçlendirildi.
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
            } else if (shadow_rec.gas_volume && shadow_rec.gas_volume->render_path == GasVolume::VolumeRenderPath::VDBUnified) {
                live_vol_id = shadow_rec.gas_volume->live_vdb_id;
                vol_shader = shadow_rec.gas_volume->getShader();
                if (shadow_rec.gas_volume->getTransformHandle()) {
                    Matrix4x4 m = shadow_rec.gas_volume->getTransformHandle()->getFinal();
                    Vec3 gsize = shadow_rec.gas_volume->getSettings().grid_size;
                    if (gsize.x > 0 && gsize.y > 0 && gsize.z > 0) {
                        m = m * Matrix4x4::scaling(Vec3(1.0f/gsize.x, 1.0f/gsize.y, 1.0f/gsize.z));
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
                } else {
                    // Manual box check for unified gas
                    AABB box; shadow_rec.gas_volume->bounding_box(0,0,box);
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


    // --- BRDF Hesabı (Specular + Diffuse) ---
    Vec3 H = (L + V).normalize();
    float NdotV = std::fmax(Vec3::dot(N, V), 0.0001f);
    float NdotH = std::fmax(Vec3::dot(N, H), 0.0001f);
    float VdotH = std::fmax(Vec3::dot(V, H), 0.0001f);

    float alpha = max(roughness * roughness, 0.01f);
    PrincipledBSDF psdf;
    // Specular bileşeni
    float D = psdf.DistributionGGX(N, H, roughness);
    float G = psdf.GeometrySmith(N, V, L, roughness);
    Vec3 F = psdf.fresnelSchlickRoughness(VdotH, F0, roughness);

    Vec3 specular = psdf.evalSpecular(N, V, L, F0, roughness);

    // Diffuse bileşeni - GPU ile uyumlu
    Vec3 F_avg = F0 + (Vec3(1.0f) - F0) / 21.0f;
    // GPU formülü: k_d = (1 - F_avg) * (1 - metallic)
    Vec3 k_d = (Vec3(1.0f) - F_avg) * (1.0f - metallic);
    Vec3 diffuse = k_d * albedo / M_PI;

    // Toplam BRDF
    Vec3 brdf = diffuse + specular;

    // --- MIS (Multiple Importance Sampling) ---
    // PDF BRDF hesapla
    Vec3 incoming = -L; // Light direction (incoming to surface)
    Vec3 outgoing = V;  // View direction
    float pdf_brdf_val = psdf.pdf(rec, incoming, outgoing);
    float pdf_brdf_val_mis = std::clamp(pdf_brdf_val, 0.01f, 5000.0f);

    float mis_weight = power_heuristic(pdf_light, pdf_brdf_val_mis);

    // Işık katkısı
    // GPU formülü: (f * Li * NdotL) * mis_weight
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

        bool hit_any = false;
        if (bvh) {
            hit_any = bvh->hit(current_ray, 0.01f, std::numeric_limits<float>::infinity(), rec, false);
        }

        if (bounce == 0 && hit_any) {
            first_hit_t = rec.t;
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

        if (!hit_any && !hit_solid) {
            // --- Infinite Grid Logic (Floor Plane Y=0) ---
            Vec3f final_bg_color = render_settings.show_background ? 
                                   toVec3f(world.evaluate(current_ray.direction)) : 
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

            if (t_exit > t_enter) {
                auto shader = rec.gas_volume->getShader();
                
                // Step size from shader or default (FASTER rendering with larger steps)
                float step_size = shader ? shader->quality.step_size : 0.15f;
                int max_steps = shader ? shader->quality.max_steps : 64;
                
                float t = t_enter;
                float current_transmittance = 1.0f;
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
                            } else {
                                r = 329.7f * powf(tk - 60.0f, -0.133f);
                                g = 288.12f * powf(tk - 60.0f, -0.0755f);
                                b = 255.0f;
                            }
                            
                            Vec3f bb_color(r/255.0f, g/255.0f, b/255.0f);
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
                throughput *= current_transmittance;
                if (current_transmittance < 0.01f) break;
            }

            if (hit_solid_inside) {
                rec = solid_rec;
            }
            else {
                current_ray = Ray(current_ray.at(t_exit + 0.001f), current_ray.direction);
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
        } else if (rec.gas_volume && rec.gas_volume->render_path == GasVolume::VolumeRenderPath::VDBUnified) {
            live_vol_id = rec.gas_volume->live_vdb_id;
            vol_shader = rec.gas_volume->getShader();
            if (rec.gas_volume->getTransformHandle()) {
                Matrix4x4 m = rec.gas_volume->getTransformHandle()->getFinal();
                Vec3 gsize = rec.gas_volume->getSettings().grid_size;
                if (gsize.x > 0 && gsize.y > 0 && gsize.z > 0) {
                    m = m * Matrix4x4::scaling(Vec3(1.0f/gsize.x, 1.0f/gsize.y, 1.0f/gsize.z));
                }
                inv_transform = m.inverse();
            }
            den_scale = 1.0f;
        }

        if (live_vol_id >= 0) {
            // Get entry and exit points
            float t_enter, t_exit;
            bool hit_box = false;
            
            if (vdb) {
                hit_box = vdb->intersectTransformedAABB(current_ray, 0.001f, std::numeric_limits<float>::infinity(), t_enter, t_exit);
            } else {
                AABB box; rec.gas_volume->bounding_box(0,0,box);
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

                // ═══════════════════════════════════════════════════════════════
                // Absorption parameters (sigma_a)
                // ═══════════════════════════════════════════════════════════════
                Vec3 absorption_color_raw = shader ? shader->absorption.color : Vec3(0.0f);
                Vec3f absorption_color = toVec3f(absorption_color_raw);
                float absorption_coeff = shader ? shader->absorption.coefficient : 0.0f;

                // ═══════════════════════════════════════════════════════════════
                // Emission parameters
                // ═══════════════════════════════════════════════════════════════
                VolumeEmissionMode emission_mode = shader ? shader->emission.mode : VolumeEmissionMode::None;
                Vec3 emission_color_raw = shader ? shader->emission.color : Vec3(1.0f, 0.5f, 0.1f);
                Vec3f emission_color = toVec3f(emission_color_raw);
                float emission_intensity = shader ? shader->emission.intensity : 0.0f;

                // ════════════════════════════──────────────────────────────────────────
                // Density and Remapping Properties
                // ════════════════════════════──────────────────────────────────────────
                float remap_low = shader ? shader->density.remap_low : 0.0f;
                float remap_high = shader ? shader->density.remap_high : 1.0f;
                float remap_range = std::max(1e-5f, remap_high - remap_low);
                float shadow_strength = shader ? shader->quality.shadow_strength : 1.0f;

                // ─────────────────────────────────────────────────────────────────────────
                // Anisotropy (Henyey-Greenstein phase function G parameter)
                // ─────────────────────────────────────────────────────────────────────────
                float anisotropy_g = shader ? shader->scattering.anisotropy : 0.0f;

                float current_transparency = 1.0f;
                Vec3f accumulated_vol_color(0.0f);


                // Jitter to reduce banding
                float jitter = ((float)rand() / RAND_MAX) * step_size;
                float t = t_enter + jitter;

                int steps = 0;
                int max_steps = shader ? shader->quality.max_steps : 256;

                while (t < t_exit && steps < max_steps) {
                    Vec3 p = current_ray.at(t);
                    Vec3 local_p = inv_transform.transform_point(p);

                    // Sample Density (Standard Coordinates)
                    float density = mgr.sampleDensityCPU(live_vol_id, local_p.x, local_p.y, local_p.z);

                    // CUTOFF REMOVED: User requested - was zeroing low densities

                    // Edge falloff - smooth fade near bounding box boundaries
                    float edge_falloff = shader ? shader->density.edge_falloff : 0.0f;
                    if (edge_falloff > 0.0f && density > 0.0f) {
                        // Calculate distance from edges (in local space)
                        Vec3 local_min(0), local_max(1);
                        if (vdb) {
                            local_min = vdb->getLocalBoundsMin();
                            local_max = vdb->getLocalBoundsMax();
                        } else {
                            // Gas volumes use physical grid dimension for edges
                            local_max = rec.gas_volume->getSettings().grid_size;
                        }

                        float dx = std::min(local_p.x - local_min.x, local_max.x - local_p.x);
                        float dy = std::min(local_p.y - local_min.y, local_max.y - local_p.y);
                        float dz = std::min(local_p.z - local_min.z, local_max.z - local_p.z);
                        float edge_dist = std::min({ dx, dy, dz });

                        // Smooth falloff near edges
                        if (edge_dist < edge_falloff) {
                            float edge_factor = edge_dist / edge_falloff;
                            density *= edge_factor * edge_factor; // Quadratic falloff
                        }
                    }

                    // --- GPU-MATCHING DENSITY REMAPPING ---
                    float d_remapped = std::max(0.0f, (density - remap_low) / remap_range);

                    if (d_remapped > 0.001f) {
                        float d = d_remapped * density_scale; // Combined Multiplier (Shader * Volume)
                        
                        // NEW: Update depth for fogging calculation only when we hit substance
                        // Threshold 0.05 is chosen to prevent very thin noise from jumping fog forward
                        if (bounce == 0 && first_vol_t < 0.0f && d > 0.05f) {
                            first_vol_t = t;
                        }

                        // Physical coefficients: sigma_t = sigma_s + sigma_a
                        float sigma_s = d * scattering_intensity;
                        float sigma_a = d * absorption_coeff;
                        float sigma_t = sigma_s + sigma_a;

                        // Beer-Lambert Transmittance for this step
                        float step_transmittance = exp(-sigma_t * step_size);

                        // --- LIGHT SAMPLING (In-Scattering) ---
                        // Iterate scene lights to calculate incoming light at point 'p'
                        Vec3f total_incoming_light(0.0f);

                        // Only sample if transparency is significant
                        if (current_transparency > 0.01f) {
                            for (const auto& light : lights) {
                                if (!light) continue;

                                Vec3 light_dir;
                                float light_dist;

                                // Calculate Light Direction & Distance
                                if (light->type() == LightType::Directional) {
                                    // Directional Light: Direction is constant (stored in light->direction)
                                    // getDirection(p) returns normalized direction towards light (inverse of light flow)
                                    light_dir = light->getDirection(p);
                                    light_dist = 1e9f; // Infinite distance
                                }
                                else {
                                    // Point/Spot/Area: Calculate from position
                                    Vec3 to_light = light->position - p;
                                    light_dist = to_light.length();
                                    light_dir = to_light / std::max(light_dist, 0.0001f);
                                }

                                // Get Intensity at point p
                                Vec3f light_intensity = toVec3f(light->getIntensity(p, light->position));

                                // OPTIMIZATION: Early exit for negligible light
                                if (light_intensity.luminance() < 1e-5f) continue;

                                // Shadow Ray (p -> Light)
                                Ray shadow_ray_vol(p + light_dir * 0.001f, light_dir);
                                float shadow_transmittance = 1.0f;

                                // 1. Check Opaque Occlusion (Fast)
                                HitRecord shadow_rec;
                                bool hit_something = bvh->hit(shadow_ray_vol, 0.001f, light_dist, shadow_rec);

                                if (hit_something) {
                                    if (!shadow_rec.vdb_volume) {
                                        shadow_transmittance = 0.0f;
                                    }
                                    else {
                                        // Hit Volume (Self-Shadow)
                                        // Ray march towards light
                                        float t_vol_enter = 0.0f;
                                        float t_vol_exit = 0.0f;

                                        // Find exit point
                                        if (vdb->intersectTransformedAABB(shadow_ray_vol, 0.0f, light_dist, t_vol_enter, t_vol_exit)) {

                                            // OPTIMIZATION: Moderate step for shadows (High Quality)
                                            float shadow_march_step = step_size * 1.5f;
                                            if (shadow_march_step < 0.1f) shadow_march_step = 0.1f; // Minimum clamp

                                            if (t_vol_exit > light_dist) t_vol_exit = light_dist;

                                            // Jitter shadow start to hide banding
                                            float t_shadow = ((float)rand() / RAND_MAX) * shadow_march_step;

                                             while (t_shadow < t_vol_exit) {
                                                Vec3 sp = shadow_ray_vol.at(t_shadow);
                                                Vec3 slocal_p = inv_transform.transform_point(sp);
                                                float s_density = mgr.sampleDensityCPU(live_vol_id, slocal_p.x, slocal_p.y, slocal_p.z);

                                                // GPU-matching Shadow Remapping
                                                float s_remapped = std::max(0.0f, (s_density - remap_low) / remap_range);

                                                if (s_remapped > 1e-4f) {
                                                    float sd = s_remapped * density_scale;
                                                    float s_sigma_t = sd * (scattering_intensity + absorption_coeff) * shadow_strength;
                                                    shadow_transmittance *= exp(-s_sigma_t * shadow_march_step);
                                                }
                                                if (shadow_transmittance < 0.01f) break;
                                                t_shadow += shadow_march_step;
                                            }
                                        }
                                    }
                                }

                                // Henyey-Greenstein Phase Function
                                // g > 0: forward scattering, g < 0: back scattering, g = 0: isotropic
                                if (shadow_transmittance > 0.0f) {
                                    float phase = 1.0f; // Default isotropic
                                    if (std::abs(anisotropy_g) > 0.001f) {
                                        // cos(theta) between view direction and light direction
                                        float cos_theta = Vec3::dot(-current_ray.direction.normalize(), light_dir);
                                        float g2 = anisotropy_g * anisotropy_g;
                                        float denom = 1.0f + g2 - 2.0f * anisotropy_g * cos_theta;
                                        phase = (1.0f - g2) / (4.0f * 3.14159265f * powf(std::max(denom, 0.0001f), 1.5f));
                                        // Normalize to reasonable range (HG can be very peaked)
                                        phase = std::min(phase, 10.0f);
                                    }
                                    total_incoming_light += light_intensity * shadow_transmittance * phase;
                                }
                            }
                        }

                        // Combine: In-Scattering + Absorption
                        // Light scattered towards camera from this step
                        // L_scat = (Li * Phase) * (1 - exp(-sigma * dt))
                        // Or approximation: Li * Density * Step

                        // In-Scattering contribution
                        // Fixed: Use Ratio (sigma_s / sigma_t) for source term integration
                        // Formula: L_added = L_in * (sigma_s / sigma_t) * (1 - exp(-sigma_t * dt))
                        float sigma_t_safe = std::max(sigma_t, 1e-6f);
                        Vec3f scattering_ratio = volume_albedo * (sigma_s / sigma_t_safe);
                        Vec3f step_scattering_term = total_incoming_light * scattering_ratio;

                        // ═══════════════════════════════════════════════════════════════
                        // EMISSION contribution
                        // ═══════════════════════════════════════════════════════════════
                        Vec3f step_emission(0.0f);
                        if (emission_mode == VolumeEmissionMode::Constant && emission_intensity > 0.0f) {
                            float scaled_density = density * (shader ? shader->density.multiplier : 1.0f);
                            step_emission = emission_color * emission_intensity * scaled_density;
                        }
                        else if (emission_mode == VolumeEmissionMode::Blackbody && density > 0.0f) {
                            // GPU-matching blackbody emission
                            float temperature_scale = shader ? shader->emission.temperature_scale : 1.0f;
                            float blackbody_intensity = shader ? shader->emission.blackbody_intensity : 10.0f;

                            // Sample temperature - prefer GasVolume live simulation data
                            float temperature = 0.0f;
                            float flame_intensity = 0.0f;
                            
                            if (rec.gas_volume) {
                                // Live Gas Simulation: sample temperature and flame directly
                                temperature = rec.gas_volume->sampleTemperature(p);
                                flame_intensity = rec.gas_volume->sampleFlameIntensity(p);
                                
                                // Keep temperature in Kelvin, subtract ambient only
                                // Will be scaled later by temperature_scale
                                float ambient_temp = rec.gas_volume->getSettings().ambient_temperature;
                                temperature = std::max(0.0f, temperature - ambient_temp);
                            }
                            else if (mgr.hasTemperatureGrid(live_vol_id)) {
                                // VDB file with temperature grid
                                temperature = mgr.sampleTemperatureCPU(live_vol_id, local_p.x, local_p.y, local_p.z);
                            }
                            else {
                                // Fallback: use density as temperature proxy
                                temperature = density;
                            }

                            Vec3f blackbody_color(0.0f);

                            // Get temperature range from shader
                            float temp_min = shader ? shader->emission.temperature_min : 0.0f;
                            float temp_max = shader ? shader->emission.temperature_max : 1500.0f;
                            float temp_range = std::max(1.0f, temp_max - temp_min);

                            // Check if ColorRamp is enabled
                            bool use_color_ramp = shader && shader->emission.color_ramp.enabled;

                            if (use_color_ramp) {
                                // Normalize temperature to 0-1 using shader's temp range
                                float ramp_t = (temperature - temp_min) / temp_range;
                                ramp_t = std::max(0.0f, std::min(ramp_t, 1.0f));
                                Vec3 ramp_color = shader->emission.color_ramp.sample(ramp_t);
                                blackbody_color = toVec3f(ramp_color);
                            }
                            else {
                                // Physical blackbody calculation
                                // Map temperature through shader's range to Kelvin
                                // temp_min → 500K (dark red), temp_max → 2000K+ (white)
                                float normalized = (temperature - temp_min) / temp_range;
                                normalized = std::max(0.0f, std::min(normalized, 1.0f));
                                float temp_k = 500.0f + normalized * 1500.0f * temperature_scale; // 500-2000K range
                                temp_k = std::max(100.0f, std::min(temp_k, 10000.0f));
                                float t = temp_k / 100.0f;
                                float r, g, b;

                                if (t <= 66.0f) {
                                    r = 255.0f;
                                    g = 99.4708025861f * logf(t) - 161.1195681661f;
                                    if (t <= 19.0f) {
                                        b = 0.0f;
                                    }
                                    else {
                                        b = 138.5177312231f * logf(t - 10.0f) - 305.0447927307f;
                                    }
                                }
                                else {
                                    r = 329.698727446f * powf(t - 60.0f, -0.1332047592f);
                                    g = 288.1221695283f * powf(t - 60.0f, -0.0755148492f);
                                    b = 255.0f;
                                }

                                blackbody_color.x = std::max(0.0f, std::min(r / 255.0f, 1.0f));
                                blackbody_color.y = std::max(0.0f, std::min(g / 255.0f, 1.0f));
                                blackbody_color.z = std::max(0.0f, std::min(b / 255.0f, 1.0f));
                            }

                            // GPU-matching: linear brightness formula
                            // density is already scaled by sampleDensityCPU, apply shader multiplier
                            float scaled_density = density * (shader ? shader->density.multiplier : 1.0f);
                            
                            // Flame intensity boost: if combustion is happening, emission is stronger
                            float flame_boost = 1.0f + flame_intensity * 5.0f;  // flame_intensity from live sim
                            float brightness = scaled_density * blackbody_intensity * flame_boost;
                            step_emission = blackbody_color * brightness;
                        }



                        // Accumulate scattering + emission
                        // GPU-matching: Use (1 - step_transmittance) instead of step_size for proper Beer-Lambert integration
                        float integration_weight = 1.0f - step_transmittance;
                        accumulated_vol_color += (step_scattering_term + step_emission) * integration_weight * current_transparency;

                        // Update Ray Transmittance
                        current_transparency *= step_transmittance;
                    }

                    if (current_transparency < 0.01f) break; // Early exit (opaque)

                    t += step_size;
                    steps++;
                }

                // Apply volumetric result to path tracer state
                color += throughput * accumulated_vol_color;
                if (bounce == 0) {
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

                    // BOUNCE REFUND for Transparent VDBs
                    if (current_transparency > 0.95f) {
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

                            if (shadow_transmittance > 0.01f) {
                                // Phase Function (Isotropic for now, or get G)
                                float phase = 1.0f / (4.0f * M_PI); // Uniform phase
                                float g_val = vol->getG();
                                if (std::abs(g_val) > 0.001f) {
                                    float cos_theta = Vec3::dot(-current_ray.direction.normalize(), light_dir);
                                    float g2 = g_val * g_val;
                                    float denom = 1.0f + g2 - 2.0f * g_val * cos_theta;
                                    phase = (1.0f - g2) / (4.0f * M_PI * powf(std::max(denom, 0.0001f), 1.5f));
                                }

                                total_incoming_light += light_intensity * shadow_transmittance * phase;
                            }
                        }
                    }

                    // Accumulate using Correct Integration Ratio
                    float sigma_t_safe = std::max(sigma_t, 1e-6f);
                    Vec3f scat_ratio = volume_albedo * (sigma_s_scalar / sigma_t_safe);
                    accumulated_vol_color += total_incoming_light * scat_ratio * (1.0f - step_transmittance) * current_transparency;

                    // Emission
                    // accumulated_vol_color += emission_color * density * step_size * current_transparency; // Simplified

                    current_transparency *= step_transmittance;
                }

                if (current_transparency < 0.01f) break;
                t += step_size;
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

                // BOUNCE REFUND for Transparent Volumes
                if (current_transparency > 0.95f) {
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

                // Albedo
                Vec3 alb = pbsdf->getPropertyValue(pbsdf->albedoProperty, uv);
                albedo = toVec3f(alb).clamp(0.01f, 1.0f);

                // Roughness (Y channel)
                Vec3 rough = pbsdf->getPropertyValue(pbsdf->roughnessProperty, uv);
                roughness = static_cast<float>(rough.y);

                // Metallic (Z channel)
                Vec3 metal = pbsdf->getPropertyValue(pbsdf->metallicProperty, uv);
                metallic = static_cast<float>(metal.z);

                // Transmission
                transmission = pbsdf->getTransmission(uv);

                // === WATER WAVE SHADER (CPU) ===
                // Detect water materials using sheen > 0 OR high transmission + low roughness
                // GPU uses mat.sheen > 0.0001f as IS_WATER flag, CPU checks anisotropic (wave_speed)
                bool is_water = (pbsdf->anisotropic > 0.0001f && transmission > 0.9f);

                if (is_water) {
                    // Get frozen water time for CPU (matches GPU behavior)
                    extern RenderSettings render_settings;
                    float water_time = 0.0f;

                    // Animation mode: frame-based time
                    if (render_settings.start_animation_render || render_settings.animation_is_playing) {
                        float fps = static_cast<float>(render_settings.animation_fps > 0 ? render_settings.animation_fps : 24);
                        water_time = static_cast<float>(render_settings.animation_current_frame) / fps;
                    }
                    else {
                        // Viewport mode: use SDL ticks (will be frozen per accumulation pass)
                        water_time = SDL_GetTicks() / 1000.0f;
                    }

                    // Wave parameters from material (packed like GPU)
                    float wave_speed = pbsdf->anisotropic;   // GPU: anisotropic
                    float wave_strength = std::fmax(0.001f, pbsdf->clearcoat);  // GPU: sheen (but clearcoat for CPU fallback)
                    float wave_frequency = pbsdf->clearcoatRoughness > 0.001f ? pbsdf->clearcoatRoughness : 1.0f;

                    // Evaluate Gerstner waves (CPU version)
                    Vec3 world_pos = rec.point;
                    Vec3 base_normal(0.0f, 1.0f, 0.0f);  // Water is horizontal

                    WaterResultCPU wave = evaluateGerstnerWaveCPU(
                        world_pos, base_normal, water_time,
                        wave_speed, wave_strength, wave_frequency
                    );

                    // Apply wave normal to interpolated normal
                    rec.interpolated_normal = wave.normal;
                    N = toVec3f(wave.normal);

                    // Foam blending: lighten albedo where foam > 0
                    float foam = wave.foam * pbsdf->translucent;  // translucent = foam_level
                    if (foam > 0.0f) {
                        Vec3f foam_color(0.95f, 0.97f, 1.0f);
                        albedo = albedo * (1.0f - foam) + foam_color * foam;
                        roughness = roughness * (1.0f - foam) + 0.3f * foam;  // Foam is rougher
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

        // --- Emissive Contribution ---
        Vec3 emitted = rec.material ? rec.material->getEmission(rec.uv, rec.point) : Vec3(0.0f);
        emission = toVec3f(emitted);

        // --- Direct lighting ---
        Vec3f direct_light(0.0f);
        if (light_count > 0 && transmission < 0.99f) {
            // --- Smart Light Selection (Importance Sampling) ---
            int bg_light_index = -1;
            // Find directional light (Sun)
            for (int k = 0; k < light_count; ++k) {
                for (int k = 0; k < light_count; ++k) {
                    if (unified_lights[k].getType() == UnifiedLightType::Directional) { bg_light_index = k; break; }
                }
            }

            int light_index = -1;
            float pdf_select = 1.0f;
            float r = Vec3::random_float();

            if (bg_light_index >= 0 && light_count > 1) {
                // Strategy: 50% Sun, 50% Others
                if (r < 0.5f) {
                    light_index = bg_light_index;
                    pdf_select = 0.5f;
                }
                else {
                    // Pick one of the others uniformly
                    float r2 = (r - 0.5f) * 2.0f;
                    int other_idx = std::min((int)(r2 * (light_count - 1)), light_count - 2);
                    if (other_idx >= bg_light_index) other_idx++;
                    light_index = other_idx;
                    pdf_select = 0.5f * (1.0f / (float)(light_count - 1));
                }
            }
            else {
                // Fallback: Uniform
                light_index = std::clamp((int)(r * light_count), 0, light_count - 1);
                pdf_select = 1.0f / (float)std::max(1, light_count);
            }

            if (light_index >= 0) {
                const UnifiedLight& light = unified_lights[light_index];
                Vec3f wi; float distance; float light_attenuation;
                if (sample_light_direction(light, hit_pos, Vec3::random_float(), Vec3::random_float(), &wi, &distance, &light_attenuation)) {
                    if (dot(N, wi) > 0.001f) {
                        Vec3 shadow_origin = rec.point + rec.interpolated_normal * UnifiedConstants::SHADOW_BIAS;
                        Ray shadow_ray(shadow_origin, toVec3(wi));
                        if (!bvh->occluded(shadow_ray, UnifiedConstants::SHADOW_BIAS, distance)) {

                            // Calculate PDF early to use for Irradiance Normalization
                            float pdf_geo = compute_light_pdf(light, distance, 1.0f);
                            float combined_pdf = pdf_geo * pdf_select;

                            Vec3f Li = light.color * light.intensity * light_attenuation;

                            // [IRRADIANCE NORMALIZATION]
                            // For Directional Lights, Input Intensity = Irradiance.
                            // But Li represents Radiance. 
                            // We know E = L * SolidAngle. So L = E / SolidAngle.
                            // SolidAngle is exactly 1.0 / pdf_geo (for Directional).
                            // So we boost Li by pdf_geo.
                            // [IRRADIANCE NORMALIZATION]
                            // [IRRADIANCE NORMALIZATION]
                            if (light.getType() == UnifiedLightType::Directional) {
                                Li *= pdf_geo;
                            }

                            Vec3f f = evaluate_brdf_unified(N, wo, wi, albedo, roughness, metallic, transmission, Vec3::random_float());
                            float pdf_brdf = pdf_brdf_unified(N, wo, wi, roughness);

                            // MIS Weight
                            float mis_weight = 1.0f;

                            // Apply MIS only for Area/Environment lights
                            // Fix: Use integer comparison for enum or cast
                            if (light.type != (int)UnifiedLightType::Directional &&
                                light.type != (int)UnifiedLightType::Point &&
                                light.type != (int)UnifiedLightType::Spot) {
                                mis_weight = power_heuristic(combined_pdf, std::clamp(pdf_brdf, 0.01f, 5000.0f));
                            }

                            // Estimator: (f * Li * cos * weight) / pdf
                            direct_light = f * Li * std::max(0.0f, dot(N, wi)) * mis_weight / std::max(combined_pdf, 1e-6f);

                            direct_light = clamp_contribution(direct_light, UnifiedConstants::MAX_CONTRIBUTION);
                        }
                    }
                }
            }
        }

        // NOTE: Transmission handling is now inside evaluate_brdf_unified (GPU-matching)

        // --- Accumulate contribution (GPU: total = direct + brdf_mis + emission) ---
        Vec3f total_contribution = direct_light + emission;  // Emission added here (GPU parity)

        // Final contribution clamp
        float total_lum = total_contribution.luminance();
        if (total_lum > UnifiedConstants::MAX_CONTRIBUTION * 2.0f) {
            total_contribution *= (UnifiedConstants::MAX_CONTRIBUTION * 2.0f / total_lum);
        }

        color += throughput * total_contribution;

        // --- Scatter ray ---
        Vec3 attenuation(1.0f);
        Ray scattered;
        bool can_scatter = rec.material && rec.material->scatter(current_ray, rec, attenuation, scattered);

        Vec3f atten_f = toVec3f(attenuation);

        if (!can_scatter) break;

        throughput *= atten_f; // APPLY ATTENUATION FOR NEXT BOUNCE
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

    if (rendering_stopped_cpu.load()) return;
    if (g_scene_loading_in_progress.load()) return;
    if (!scene.camera) return;

    extern RenderSettings render_settings;


    // Ensure accumulation buffer is allocated
    const size_t pixel_count = image_width * image_height;
    if (cpu_accumulation_buffer.size() != pixel_count) {
        cpu_accumulation_buffer.resize(pixel_count, Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
        cpu_accumulation_valid = true;
    }
    
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

    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIMIZATION: Cache pixel list - only rebuild + shuffle when necessary
    // This avoids O(n) allocation + O(n) shuffle on EVERY pass
    // ═══════════════════════════════════════════════════════════════════════════
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
            if (idx >= static_cast<int>(pixels_to_render)) break;

            int i = pixel_list[idx].first;
            int j = pixel_list[idx].second;
            int pixel_index = j * image_width + i;

            // ═══════════════════════════════════════════════════════════════════
            // ADAPTIVE SAMPLING - Early exit for converged pixels
            // Same logic as GPU: skip if variance is below threshold AND we have enough samples
            // ═══════════════════════════════════════════════════════════════════
            // ADAPTIVE SAMPLING - Coefficient of Variation (CV = σ/μ) based convergence
            // Industry standard: relative error is consistent across all brightness levels
            // ═══════════════════════════════════════════════════════════════════
            if (render_settings.use_adaptive_sampling) {
                Vec4& accum_check = cpu_accumulation_buffer[pixel_index];
                float prev_samples_check = accum_check.w;
                float current_variance = cpu_variance_buffer[pixel_index];
                
                // Compute mean luminance for CV calculation
                float mean_lum_check = 0.2126f * accum_check.x + 0.7152f * accum_check.y + 0.0722f * accum_check.z;
                
                // Coefficient of Variation: CV = σ / μ (relative standard deviation)
                // This gives consistent convergence across dark and bright regions
                float cv = (mean_lum_check > 0.001f) ? std::sqrt(current_variance) / mean_lum_check : 1.0f;
                
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
            
            // ═══════════════════════════════════════════════════════════════════
            // ADAPTIVE SAMPLING - Variance calculation for next pass decision
            // ═══════════════════════════════════════════════════════════════════
            // ADAPTIVE SAMPLING - Welford's Online Variance Algorithm
            // More numerically stable than naive variance calculation
            // Stores variance (σ²), CV is computed at check time as σ/μ
            // ═══════════════════════════════════════════════════════════════════
            if (render_settings.use_adaptive_sampling) {
                // Compute luminance (Rec.709 weights)
                auto compute_luminance = [](const Vec3& c) {
                    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
                };
                
                // new_color = this pass's raw result, blended_color = accumulated mean
                float new_lum = compute_luminance(new_color);
                float mean_lum = compute_luminance(blended_color);
                
                // Welford's online algorithm for running variance
                // More stable than naive E[X²] - E[X]² approach
                float diff = new_lum - mean_lum;
                float prev_variance = cpu_variance_buffer[pixel_index];
                
                // Incremental variance update: Var_n = Var_{n-1} + (x - μ)² / n - Var_{n-1} / n
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

            // Calculate Exposure Factor
            float exposure_factor = 1.0f;
            if (scene.camera) {
                if (scene.camera->auto_exposure) {
                    exposure_factor = std::pow(2.0f, scene.camera->ev_compensation);
                }
                else {
                    // Manual Exposure calculation
                    float iso_mult = CameraPresets::ISO_PRESETS[scene.camera->iso_preset_index].exposure_multiplier;
                    float shutter_time = CameraPresets::SHUTTER_SPEED_PRESETS[scene.camera->shutter_preset_index].speed_seconds;

                    // Use F-Stop Number
                    float f_number = 16.0f;
                    if (scene.camera->fstop_preset_index > 0) {
                        f_number = CameraPresets::FSTOP_PRESETS[scene.camera->fstop_preset_index].f_number;
                    }
                    else {
                        // Custom Mode: Estimate f-number from aperture (diameter/radius)
                        // Using aperture=0.05 -> f/16 as reference (K=0.8)
                        if (scene.camera->aperture > 0.001f)
                            f_number = 0.8f / scene.camera->aperture;
                        else
                            f_number = 16.0f;
                    }
                    float aperture_sq = f_number * f_number;

                    float ev_comp = std::pow(2.0f, scene.camera->ev_compensation);

                    float current_val = (iso_mult * shutter_time) / (aperture_sq + 0.001f);
                    float baseline_val = 0.00003125f; // Sunny 16 baseline

                    exposure_factor = (current_val / baseline_val) * ev_comp;
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

    // Update window title with progress
    if (window) {
        float progress = 100.0f * cpu_accumulated_samples / target_max_samples;
        std::string title = "RayTrophi CPU - Sample " + std::to_string(cpu_accumulated_samples) +
            "/" + std::to_string(target_max_samples) +
            " (" + std::to_string(int(progress)) + "%) - " +
            std::to_string(int(pass_ms)) + "ms/sample";
        SDL_SetWindowTitle(window, title.c_str());
    }
}

// Add this implementation at the end of Renderer.cpp before the closing brace


// Rebuild OptiX geometry after scene modifications (deletion/addition)
void Renderer::rebuildOptiXGeometry(SceneData& scene, OptixWrapper* optix_gpu_ptr) {
    // Rebuild geometry TLAS
    rebuildOptiXGeometryWithList(scene.world.objects, optix_gpu_ptr);
    // Sync all volumes (VDB/Gas)
    VolumetricRenderer::syncVolumetricData(scene, optix_gpu_ptr);
}

void Renderer::rebuildOptiXGeometryWithList(const std::vector<std::shared_ptr<Hittable>>& objects, OptixWrapper* optix_gpu_ptr) {
    if (!optix_gpu_ptr) {
        SCENE_LOG_WARN("[OptiX] Cannot rebuild - no OptiX pointer");
        return;
    }

    // Handle empty list
    if (objects.empty()) {
        SCENE_LOG_INFO("[OptiX] Object list is empty, clearing GPU scene");
        optix_gpu_ptr->clearScene();
        optix_gpu_ptr->resetAccumulation();
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
            for (const auto& obj : objects) {
                if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                    if (tri->visible) {
                        triangles.push_back(tri);
                    }
                }
            }
        }

        OptixGeometryData optix_data;
        if (!triangles.empty()) {
            optix_data = assimpLoader.convertTrianglesToOptixData(triangles);
        }

        optix_gpu_ptr->validateMaterialIndices(optix_data);
        optix_gpu_ptr->buildFromDataTLAS(optix_data, objects);
        optix_gpu_ptr->resetAccumulation();

        if (triangles.size() > 1000) {
            SCENE_LOG_INFO("[OptiX] Geometry rebuilt (Snapshot) - " + std::to_string(triangles.size()) + " triangles");
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
void Renderer::updateOptiXMaterialsOnly(SceneData& scene, OptixWrapper* optix_gpu_ptr) {
    if (!optix_gpu_ptr) {
        return;
    }

    try {
        // Get all materials directly from MaterialManager - NO triangle iteration!
        auto& mgr = MaterialManager::getInstance();
        const auto& all_materials = mgr.getAllMaterials();

        if (all_materials.empty()) {
            return;
        }

        // Build GpuMaterial array directly from MaterialManager
        std::vector<GpuMaterial> gpu_materials;
        std::vector<OptixGeometryData::VolumetricInfo> volumetric_info;

        gpu_materials.reserve(all_materials.size());
        volumetric_info.reserve(all_materials.size());

        for (size_t i = 0; i < all_materials.size(); ++i) {
            const auto& mat = all_materials[i];
            if (!mat) continue;

            GpuMaterial gpu_mat = {};
            OptixGeometryData::VolumetricInfo vol_info = {};

            // Check if it's a PrincipledBSDF or Volumetric material
            if (mat->type() == MaterialType::Volumetric) {
                // Volumetric material
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

                    // Multi-Scattering Parameters
                    vol_info.multi_scatter = vol_mat->getMultiScatter();
                    vol_info.g_back = vol_mat->getGBack();
                    vol_info.lobe_mix = vol_mat->getLobeMix();
                    vol_info.light_steps = vol_mat->getLightSteps();
                    vol_info.shadow_strength = vol_mat->getShadowStrength();

                    // Default AABB (will be correct from initial build)
                    vol_info.aabb_min = make_float3(0, 0, 0);
                    vol_info.aabb_max = make_float3(1, 1, 1);

                    // Default GpuMaterial for volumetric
                    gpu_mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
                    gpu_mat.roughness = 1.0f;
                    gpu_mat.metallic = 0.0f;
                    gpu_mat.emission = make_float3(0.0f, 0.0f, 0.0f);
                    gpu_mat.ior = 1.0f;
                    gpu_mat.transmission = 0.0f;
                    gpu_mat.opacity = 1.0f;

                    // Fetch NanoVDB Grid Pointer if available
                    if (vol_mat->hasVDBVolume()) {
                        void* grid_ptr = VDBVolumeManager::getInstance().getGPUGrid(vol_mat->getVDBVolumeID());
                        vol_info.nanovdb_grid = grid_ptr;
                        vol_info.has_nanovdb = (grid_ptr != nullptr) ? 1 : 0;
                    }
                }
            }
            else if (mat->gpuMaterial) {
                // PrincipledBSDF with gpuMaterial
                gpu_mat = *mat->gpuMaterial;
                vol_info.is_volumetric = 0;
            }
            else {
                // Fallback default material
                gpu_mat.albedo = make_float3(0.8f, 0.8f, 0.8f);
                gpu_mat.roughness = 0.5f;
                gpu_mat.metallic = 0.0f;
                gpu_mat.emission = make_float3(0.0f, 0.0f, 0.0f);
                gpu_mat.ior = 1.5f;
                gpu_mat.transmission = 0.0f;
                gpu_mat.opacity = 1.0f;
                vol_info.is_volumetric = 0;
            }

            gpu_materials.push_back(gpu_mat);
            volumetric_info.push_back(vol_info);
        }

        // Update GPU buffers - FAST! Just memory copies
        if (!gpu_materials.empty()) {
            optix_gpu_ptr->updateMaterialBuffer(gpu_materials);
        }

        // Update volumetric SBT data
        if (!volumetric_info.empty()) {
            optix_gpu_ptr->updateSBTVolumetricData(volumetric_info);
        }

        // Update independent volumes as well (GpuVDBVolume)
        VolumetricRenderer::syncVolumetricData(scene, optix_gpu_ptr);
    }
    catch (std::exception& e) {
        SCENE_LOG_ERROR("[OptiX] updateOptiXMaterialsOnly failed: " + std::string(e.what()));
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

        // ═══════════════════════════════════════════════════════════════════════════
        // ENHANCED WIND PARAMETERS
        // ═══════════════════════════════════════════════════════════════════════════

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

            // ═══════════════════════════════════════════════════════════════════════════
            // MULTI-FREQUENCY OSCILLATION
            // Creates natural, organic movement by combining multiple wave frequencies
            // ═══════════════════════════════════════════════════════════════════════════

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

            // ═══════════════════════════════════════════════════════════════════════════
            // DIRECTIONAL BENDING
            // Tree leans TOWARDS wind direction + oscillates around that lean
            // ═══════════════════════════════════════════════════════════════════════════

            // Constant lean towards wind direction
            // Rüzgar +X yönünde esiyorsa, ağaç +X yönüne doğru eğilir
            float lean_x = dir.z * lean_amount;   // Lean around X-axis based on wind Z component
            float lean_z = -dir.x * lean_amount;  // Lean around Z-axis based on wind X component

            // Dynamic oscillation around the lean point
            float osc_x = dir.z * oscillation * sway_amount;
            float osc_z = -dir.x * oscillation * sway_amount;

            // Combined rotation = initial + constant lean + oscillation
            float final_rot_x = init.rotation.x + lean_x + osc_x;
            float final_rot_z = init.rotation.z + lean_z + osc_z;

            // ═══════════════════════════════════════════════════════════════════════════
            // ANGLE CLAMPING (Prevent unnatural over-bending)
            // ═══════════════════════════════════════════════════════════════════════════
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
                        if (this->optix_gpu_ptr && !hi->optix_instance_ids.empty()) {
                            float t[12];
                            const Matrix4x4& m = new_mat;
                            t[0] = m.m[0][0]; t[1] = m.m[0][1]; t[2] = m.m[0][2]; t[3] = m.m[0][3];
                            t[4] = m.m[1][0]; t[5] = m.m[1][1]; t[6] = m.m[1][2]; t[7] = m.m[1][3];
                            t[8] = m.m[2][0]; t[9] = m.m[2][1]; t[10] = m.m[2][2]; t[11] = m.m[2][3];

                            for (int id : hi->optix_instance_ids) {
                                this->optix_gpu_ptr->updateInstanceTransform(id, t);
                            }
                        }
                    }
                }
            }
        }

        group->gpu_dirty = true;
        any_update = true;

        // ═══════════════════════════════════════════════════════════════════════════
        // GPU SHADER WIND PARAMETERS
        // Upload wind direction, strength, speed, time for shader-based displacement
        // ═══════════════════════════════════════════════════════════════════════════
        if (this->optix_gpu_ptr) {
            // Normalize strength to 0-1 range for shader (divide by max angle)
            float normalized_strength = strength / 25.0f;  // max_bend_angle = 25 degrees
            this->optix_gpu_ptr->setWindParams(
                group->wind_settings.direction,
                normalized_strength,
                speed,
                time
            );
        }
        // Commit changes to TLAS if any updates occurred
        if (any_update && this->optix_gpu_ptr) {
            this->optix_gpu_ptr->rebuildTLAS(); // Fast refit/update
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GAS VOLUME OPTIX SYNC
// ═══════════════════════════════════════════════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────
// Volumetric Sync - Unified logic for Gas and VDB
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::updateOptiXGasVolumes(SceneData& scene, OptixWrapper* optix_gpu_ptr) {
    VolumetricRenderer::syncVolumetricData(scene, optix_gpu_ptr);
}
