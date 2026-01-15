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
#include "scene_data.h"
#include "VDBVolumeManager.h"

// Unified rendering system for CPU/GPU parity
#include "unified_types.h"
#include "unified_brdf.h"
#include "unified_light_sampling.h"
#include "unified_converters.h"
#include "MaterialManager.h"
#include "CameraPresets.h"
#include "TerrainManager.h"
#include "water_shaders_cpu.h"  // CPU water shader functions

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

    // ===== BUFFER CACHE OPTIMIZATION =====
    // Boyut değiştiyse buffer'ları yeniden oluştur
    bool sizeChanged = (width != oidnCachedWidth || height != oidnCachedHeight);
    
    if (sizeChanged) {
        // CPU buffer'ı resize et
        oidnColorData.resize(bufferSize);
        
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

    // ===== PIXEL DATA TRANSFER =====
    // Surface'den color buffer'a aktar (bu her zaman gerekli)
    for (size_t i = 0; i < pixelCount; ++i) {
        Uint8 r, g, b;
        SDL_GetRGB(pixels[i], surface->format, &r, &g, &b);
        oidnColorData[i * 3] = r / 255.0f;
        oidnColorData[i * 3 + 1] = g / 255.0f;
        oidnColorData[i * 3 + 2] = b / 255.0f;
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

    // ===== RESULT BLENDING =====
    // Sonucu karıştır ve geri yaz
    for (size_t i = 0; i < pixelCount; ++i) {
        Uint8 r_orig, g_orig, b_orig;
        SDL_GetRGB(pixels[i], surface->format, &r_orig, &g_orig, &b_orig);

        float r_denoised = std::clamp(oidnColorData[i * 3], 0.0f, 1.0f);
        float g_denoised = std::clamp(oidnColorData[i * 3 + 1], 0.0f, 1.0f);
        float b_denoised = std::clamp(oidnColorData[i * 3 + 2], 0.0f, 1.0f);

        Uint8 r = static_cast<Uint8>((r_denoised * blend + r_orig / 255.0f * (1 - blend)) * 255);
        Uint8 g = static_cast<Uint8>((g_denoised * blend + g_orig / 255.0f * (1 - blend)) * 255);
        Uint8 b = static_cast<Uint8>((b_denoised * blend + b_orig / 255.0f * (1 - blend)) * 255);

        pixels[i] = SDL_MapRGB(surface->format, r, g, b);
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

void Renderer::render_image(SDL_Surface* surface, SDL_Window* window, SDL_Texture* raytrace_texture, SDL_Renderer* renderer,
    const int total_samples_per_pixel, const int samples_per_pass, SceneData& scene) {
     render_finished = false;
    rendering_complete = false;
	rendering_in_progress = true;
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    rendering_stopped_cpu = false;
   // std::thread display_thread(&Renderer::update_display, this, window, raytrace_texture, surface, renderer);

    const int num_passes = (total_samples_per_pixel + samples_per_pass - 1) / samples_per_pass;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int pass = 0; pass < num_passes; ++pass) {
        
        // Shuffle full-resolution pixel list
        std::vector<std::pair<int, int>> shuffled_pixel_list;
        for (int j = 0; j < image_height; ++j) {
            for (int i = 0; i < image_width; ++i) {
                shuffled_pixel_list.emplace_back(i, j);
            }
        }
        std::shuffle(shuffled_pixel_list.begin(), shuffled_pixel_list.end(), std::mt19937(std::random_device{}()));

        std::atomic<int> next_pixel_index = 0;

        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back(&Renderer::render_chunk, this,
                surface,
                std::cref(shuffled_pixel_list),
                std::ref(next_pixel_index),
                std::cref(scene.world),
                std::cref(scene.lights),
                scene.background_color,
                scene.bvh.get(),
                scene.camera,
                samples_per_pass,
                pass * samples_per_pass
            );
            
        }
       
        for (auto& thread : threads) {
            thread.join();
        }
        //threads.clear();

        // ----- İlerleme hesaplama -----
        float progress = static_cast<float>(pass + 1) / num_passes;
        auto current_time = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() / 1000.0f;

        float pixels_done = static_cast<float>((pass + 1) * image_width * image_height * samples_per_pass);
        float total_pixels = static_cast<float>(image_width * image_height * total_samples_per_pixel);
        float pixels_per_sec = pixels_done / std::max(0.001f, elapsed);
        float remaining_time = (total_pixels - pixels_done) / std::max(1.0f, pixels_per_sec);
        float fps = pixels_per_sec / (image_width * image_height);
        // SDL başlık
        char title[128];
        std::snprintf(title, sizeof(title),
            "Progress: %.1f%% | %.1fK px/s | ETA: %ds | FPS: %.1f",
            progress * 100, pixels_per_sec / 1000.0f, static_cast<int>(remaining_time), fps);
       // SDL_SetWindowTitle(window, title);
        SCENE_LOG_INFO(std::string(title));
      
    }
    render_finished = true;
	rendering_in_progress = false;
    //display_thread.join();
   
}

// ============================================================================
// NEW ANIMATION SYSTEM INTEGRATION
// ============================================================================

void Renderer::initializeAnimationSystem(SceneData& scene) {
    auto& animCtrl = AnimationController::getInstance();
    
    // Register all animation clips from the scene
    if (!scene.animationDataList.empty()) {
        animCtrl.registerClips(scene.animationDataList);
        
        SCENE_LOG_INFO("[AnimSystem] Initialized with " + 
            std::to_string(scene.animationDataList.size()) + " animation clips.");
        
        // Auto-play first clip if available
        const auto& clips = animCtrl.getAllClips();
        if (!clips.empty()) {
            animCtrl.play(clips[0].name, 0.0f);  // Instant start
            SCENE_LOG_INFO("[AnimSystem] Auto-playing: " + clips[0].name);
        }
    }
}

bool Renderer::updateAnimationWithGraph(SceneData& scene, float deltaTime, bool apply_cpu_skinning) {
    auto& animCtrl = AnimationController::getInstance();
    
    // Update animation controller
    bool changed = animCtrl.update(deltaTime, scene.boneData);
    
    if (!changed) {
        return false;
    }
    
    // Get computed bone matrices from animation controller
    const auto& matrices = animCtrl.getFinalBoneMatrices();
    
    // Resize our matrices if needed
    if (this->finalBoneMatrices.size() != matrices.size()) {
        this->finalBoneMatrices.resize(matrices.size());
    }
    
    // Copy matrices
    for (size_t i = 0; i < matrices.size(); ++i) {
        this->finalBoneMatrices[i] = matrices[i];
    }
    
    // Apply skinning to triangles if CPU skinning is requested
    if (apply_cpu_skinning) {
        for (auto& obj : scene.world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->hasSkinData()) {
                // Disable transform handle for skinned meshes
                auto transformHandle = tri->getTransformHandle();
                if (transformHandle) {
                    transformHandle->setBase(Matrix4x4::identity());
                    transformHandle->setCurrent(Matrix4x4::identity());
                }
                
                tri->apply_skinning(static_cast<const std::vector<Matrix4x4>&>(this->finalBoneMatrices));
            }
        }
    }
    
    // Reset CPU accumulation since geometry changed
    resetCPUAccumulation();
    
    // Clear dirty flag in controller
    animCtrl.clearDirtyFlag();
    
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
    
    // ═══════════════════════════════════════════════════════════════════════════
    // FALLBACK: Use AnimationController when importedModelContexts is empty
    // ═══════════════════════════════════════════════════════════════════════════
    // This happens after project load - loaders are not serialized but animation
    // data is. Use the AnimationController-based system in this case.
    // ═══════════════════════════════════════════════════════════════════════════
    bool useAnimationController = scene.importedModelContexts.empty() && 
                                   !scene.animationDataList.empty() && 
                                   !scene.boneData.boneNameToIndex.empty();
    
    if (useAnimationController) {
        // Use REAL wall-clock deltaTime instead of timeline time
        // This ensures animation plays continuously regardless of timeline state
        static auto lastRealTime = std::chrono::high_resolution_clock::now();
        auto nowTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(nowTime - lastRealTime).count();
        lastRealTime = nowTime;
        
        // Clamp deltaTime to reasonable bounds
        if (deltaTime < 0.0f || deltaTime > 0.5f) deltaTime = 1.0f / 60.0f;
        
        // Use the new AnimationController system
        geometry_changed = updateAnimationWithGraph(scene, deltaTime, apply_cpu_skinning);
    }

    // --- 1. Adım: Animasyonlu Node Hiyerarşisini Güncelle ---
    std::unordered_map<std::string, Matrix4x4> animatedGlobalNodeTransforms;

    // Iterate over ALL imported models to update their respective hierarchies
    for (const auto& modelCtx : scene.importedModelContexts) {
        if (!modelCtx.loader || !modelCtx.loader->getScene() || !modelCtx.loader->getScene()->mRootNode) continue;
        
        // ═══════════════════════════════════════════════════════════════════════════
        // CRITICAL FIX: Skip models without animation data
        // ═══════════════════════════════════════════════════════════════════════════
        // Calling calculateAnimatedNodeTransformsRecursive on non-animated models:
        // 1. Recalculates node hierarchy with FBX root transform (Z-up to Y-up)
        // 2. Overwrites existing user transforms (translations applied via gizmo)
        // 3. Causes static rigged objects to "jump" to bind pose at world origin
        // ═══════════════════════════════════════════════════════════════════════════
        if (!modelCtx.hasAnimation) {
            continue; // Skip non-animated models - preserve their transforms
        }
        
        // Restore context (names, etc.) - implicitly handled by using the specific loader instance
        Matrix4x4 identityParentTransform = Matrix4x4::identity();
        
        // Build lookups for THIS model's animations
        std::map<std::string, const AnimationData*> animationLookupMap;
        for (const auto& anim : scene.animationDataList) {
             for (const auto& pair : anim.positionKeys) animationLookupMap[pair.first] = &anim;
             for (const auto& pair : anim.rotationKeys) animationLookupMap[pair.first] = &anim;
             for (const auto& pair : anim.scalingKeys) animationLookupMap[pair.first] = &anim;
        }

        modelCtx.loader->calculateAnimatedNodeTransformsRecursive(
            modelCtx.loader->getScene()->mRootNode,
            identityParentTransform,
            animationLookupMap,
            current_time,
            animatedGlobalNodeTransforms // Accumulate into global map
        );
    }

    // --- 1.5. PRE-CALCULATE GLOBAL BONE MATRICES (Optimization) ---
    // Instead of recalculating per-mesh (which causes massive slowdown), do it once globally.
    if (!scene.boneData.boneNameToIndex.empty()) {
        // Ensure vector is large enough for the largest index
        // Since indices are 0-based, size needs to be max_index + 1. 
        // Using map size is usually correct unless there are gaps/overlaps. 
        // We'll trust map size but verify inside loop.
        if (this->finalBoneMatrices.size() < scene.boneData.boneNameToIndex.size()) {
            this->finalBoneMatrices.resize(scene.boneData.boneNameToIndex.size());
        }

        for (const auto& [boneName, boneIndex] : scene.boneData.boneNameToIndex) {
            // Robustness: Handle out-of-bounds indices due to potential merges
            if (boneIndex >= finalBoneMatrices.size()) {
                finalBoneMatrices.resize(boneIndex + 1);
            }

            if (animatedGlobalNodeTransforms.count(boneName) > 0 && scene.boneData.boneOffsetMatrices.count(boneName) > 0) {
                 Matrix4x4 animatedBoneGlobal = animatedGlobalNodeTransforms[boneName];
                 Matrix4x4 offsetMatrix = scene.boneData.boneOffsetMatrices[boneName];
                 finalBoneMatrices[boneIndex] = animatedBoneGlobal * offsetMatrix;
            } else {
                 // Fallback to identity to prevent explosions, though this usually implies broken rig or missing update
                 finalBoneMatrices[boneIndex] = Matrix4x4::identity();
            }
        }
    }

    // --- 2. Adım: Üçgenleri Animasyon Türüne Göre Güncelle ---
    for (auto& obj : scene.world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (!tri) continue;

        std::string nodeName = tri->getNodeName();
        bool isSkinnedMesh = tri->hasSkinData();

        if (isSkinnedMesh) {
            // ═══════════════════════════════════════════════════════════════════════════
            // Check if this mesh belongs to an animated model
            // ═══════════════════════════════════════════════════════════════════════════
            // animatedGlobalNodeTransforms only contains nodes from models with animation.
            // If NONE of this mesh's bones are in the map, the mesh is from a non-animated
            // model and should be skipped to preserve its transform.
            // ═══════════════════════════════════════════════════════════════════════════
            bool meshBelongsToAnimatedModel = false;
            const auto& boneWeights = tri->getSkinBoneWeights(0);
            for (const auto& [boneIdx, weight] : boneWeights) {
                if (weight < 0.001f) continue;
                
                // OPTIMIZATION: O(1) reverse lookup instead of O(n) map traversal
                const std::string& boneName = scene.boneData.getBoneNameByIndex(static_cast<unsigned int>(boneIdx));
                if (!boneName.empty() && animatedGlobalNodeTransforms.count(boneName) > 0) {
                    meshBelongsToAnimatedModel = true;
                    break;
                }
            }
            
            // Skip meshes from non-animated models
            if (!meshBelongsToAnimatedModel) {
                continue; // Preserve transform for non-animated rigged objects
            }
            
            // Disable transform handle for animated skinned meshes
            auto transformHandle = tri->getTransformHandle();
            if (transformHandle) {
                transformHandle->setBase(Matrix4x4::identity());
                transformHandle->setCurrent(Matrix4x4::identity());
            }

            // Geometry changes (either on GPU or CPU), so flag it.
            // This ensures GPU update is triggered even if CPU skinning is skipped.
            geometry_changed = true;

            // Only apply CPU skinning if requested (skip for GPU rendering to save perf)
            if (apply_cpu_skinning) {
                // Now just use the pre-calculated finalBoneMatrices
                tri->apply_skinning(static_cast<const std::vector<Matrix4x4>&>(finalBoneMatrices));
            }
        }
        else {
            bool nodeHasAnimation = false;
            for (const auto& anim : scene.animationDataList) {
                if (anim.positionKeys.count(nodeName) > 0 ||
                    anim.rotationKeys.count(nodeName) > 0 ||
                    anim.scalingKeys.count(nodeName) > 0) {
                    nodeHasAnimation = true;
                    break;
                }
            }
            
            if (nodeHasAnimation && animatedGlobalNodeTransforms.count(nodeName) > 0) {
                Matrix4x4 animTransform = animatedGlobalNodeTransforms[nodeName];
                auto transformHandle = tri->getTransformHandle();
                if (transformHandle) {
                    transformHandle->setBase(animTransform);
                    transformHandle->setCurrent(Matrix4x4::identity());
                    tri->updateTransformedVertices();
                    geometry_changed = true;  // File-based animation modifies vertex positions
                }
            }
            else {
                if (animatedGlobalNodeTransforms.count(nodeName) > 0) {
                    Matrix4x4 staticTransform = animatedGlobalNodeTransforms[nodeName];
                    auto transformHandle = tri->getTransformHandle();
                    if (transformHandle) {
                        transformHandle->setBase(staticTransform);
                        transformHandle->setCurrent(Matrix4x4::identity());
                        tri->updateTransformedVertices();
                        geometry_changed = true;  // Static transform from hierarchy modifies vertex positions
                    }
                }
                
                // --- MANUAL KEYFRAME SUPPORT (NEW!) ---
                // Apply Timeline keyframes for objects NOT animated by file data
                if (!nodeHasAnimation && !scene.timeline.tracks.empty()) {
                    // Convert current_time to frame number (assuming standard FPS)
                    extern RenderSettings render_settings;
                    int current_frame = static_cast<int>(current_time * render_settings.animation_fps);
                    
                    // Check if this object has keyframes
                    auto track_it = scene.timeline.tracks.find(nodeName);
                    if (track_it != scene.timeline.tracks.end() && !track_it->second.keyframes.empty()) {
                        // Evaluate keyframe at current frame
                        Keyframe kf = track_it->second.evaluate(current_frame);
                        
                        if (kf.has_transform) {
                            // Build transform matrix from keyframe
                            Matrix4x4 translation = Matrix4x4::translation(kf.transform.position);
                            
                            // Convert Euler angles to rotation matrix (deg to rad)
                            float rx = kf.transform.rotation.x * (3.14159265f / 180.0f);
                            float ry = kf.transform.rotation.y * (3.14159265f / 180.0f);
                            float rz = kf.transform.rotation.z * (3.14159265f / 180.0f);
                            
                            Matrix4x4 rotation = Matrix4x4::rotationZ(rz) * 
                                               Matrix4x4::rotationY(ry) * 
                                               Matrix4x4::rotationX(rx);
                            
                            Matrix4x4 scale = Matrix4x4::scaling(kf.transform.scale);
                            
                            // Combine: T * R * S
                            Matrix4x4 final_transform = translation * rotation * scale;
                            
                            // Apply to triangle
                            auto transformHandle = tri->getTransformHandle();
                            if (transformHandle) {
                                transformHandle->setBase(final_transform);
                                transformHandle->setCurrent(Matrix4x4::identity());
                                tri->updateTransformedVertices();
                                geometry_changed = true;  // Keyframe transform modifies vertex positions
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
            if (anim.positionKeys.count(light->nodeName) > 0 ||
                anim.rotationKeys.count(light->nodeName) > 0 ||
                anim.scalingKeys.count(light->nodeName) > 0) {
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
            if (light->type() == LightType::Directional) {
                world.setSunDirection(-light->direction);
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
            if (anim.positionKeys.count(scene.camera->nodeName) > 0 ||
                anim.rotationKeys.count(scene.camera->nodeName) > 0 ||
                anim.scalingKeys.count(scene.camera->nodeName) > 0) {
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
        start_frame = anim.startFrame;
        end_frame = anim.endFrame;
        SCENE_LOG_INFO("Settings range invalid, using animation frame range from file: " + std::to_string(start_frame) + " - " + std::to_string(end_frame));
    } else {
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
        
        // --- UPDATE ANIMATION ---
        // Returns true if geometry changed
        // Returns true if geometry changed
        // --- UPDATE ANIMATION ---
        // Disable CPU skinning if running on OptiX to save performance and prevent crashes
        // UPDATE (Fix GPU T-Pose): Re-enable CPU skinning so we can upload deformed vertices to GPU
        // Foliage (Fast) is not skinned, so it won't be affected.
        bool geometry_changed = this->updateAnimationState(scene, current_time, true); // Always calc CPU skinning for now
        
        // --- WIND ANIMATION ---
        // Apply wind simulation for this frame
        if (g_hasOptix) {  // Wind is currently GPU/OptiX instance based
            this->updateWind(scene, current_time);
            // Wind updates are self-contained in updateWind (Direct GPU update + TLAS Refit)
            // No need to set geometry_changed = true, which would trigger expensive BLAS checks.
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
                 } else {
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
                render_progressive_pass(target_surface,window, scene, 1, total_samples_per_pixel);
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
    } else {
        SCENE_LOG_INFO("Animation rendering completed successfully!");
    }
}

void Renderer::rebuildBVH(SceneData& scene, bool use_embree) {
    if (!scene.initialized) {
        SCENE_LOG_WARN("Scene not initialized, BVH rebuild skipped.");
        return;
    }
    
    // Handle empty scene (e.g., all objects deleted)
    if (scene.world.objects.empty()) {
        scene.bvh = nullptr;  // Clear BVH for empty scene
        SCENE_LOG_INFO("Scene is empty, BVH cleared.");
        return;
    }

    // VDB Volume Support: Now handled natively by EmbreeBVH via User Geometry.
    // No need to fallback to ParallelBVH.

    if (use_embree) {
        auto embree_bvh = std::make_shared<EmbreeBVH>();
        embree_bvh->build(scene.world.objects);
        scene.bvh = embree_bvh;
    }
    else {
        scene.bvh = std::make_shared<ParallelBVHNode>(scene.world.objects, 0, scene.world.size(), 0.0, 1.0, 0);
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
    bool append) {

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
    } else {
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
    auto [loaded_triangles, loaded_animations, loaded_bone_data] = newLoader->loadModelToTriangles(model_path);

    // Store the context
    SceneData::ImportedModelContext modelCtx;
    modelCtx.loader = newLoader;
    modelCtx.importName = newLoader->currentImportName;
    modelCtx.hasAnimation = (newLoader->getScene() && newLoader->getScene()->mNumAnimations > 0);
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
        } else {
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
            
            // Only set global inverse if it's the first model or handle separately? 
            // It seems unused for skinning (offset matrix handles it), so valid to leave or overwrite.
            if (!append) scene.boneData.globalInverseTransform = loaded_bone_data.globalInverseTransform;
            
            // CRITICAL: Rebuild reverse lookup after merge for O(1) index->name queries
            scene.boneData.rebuildReverseLookup();
        }

        // 3. Merge Animations
        if (append) {
            scene.animationDataList.insert(scene.animationDataList.end(), loaded_animations.begin(), loaded_animations.end());
        } else {
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
    for (const auto& tri : loaded_triangles) {
        scene.world.add(tri);
    }
    SCENE_LOG_INFO("Added " + std::to_string(loaded_triangles.size()) + " triangles to scene.");

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
    } else {
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
        } else {
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
    } else {
        // Replace lights
        scene.lights = new_lights;
        SCENE_LOG_INFO("Loaded lights: " + std::to_string(scene.lights.size()));
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

void Renderer::render_worker(
    SDL_Surface* surface,
    const std::vector<std::pair<int, int>>& shuffled_pixel_list,
    std::atomic<int>& next_pixel_index,
    const HittableList& world,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color,
    const  Hittable* bvh, const std::shared_ptr<Camera>& camera,
    const int samples_per_pass,
    const int current_sample) {

    render_chunk(
        surface,
        shuffled_pixel_list,
        next_pixel_index,
        world,
        lights,
        background_color,
        bvh,
        camera,
        samples_per_pass,
        current_sample
    );
}


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


void Renderer::render_chunk_adaptive(SDL_Surface* surface,
    const std::vector<std::pair<int, int>>& shuffled_pixel_list,
    std::atomic<int>& next_pixel_index,
    const HittableList& world,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color,
    const Hittable* bvh,
    const std::shared_ptr<Camera>& camera,
    const int total_samples_per_pixel)
{
   
    const int min_samples = render_settings.min_samples;
    const int max_samples = render_settings.max_samples;
    const float base_variance_threshold = render_settings.variance_threshold;
    const int total_pixels = shuffled_pixel_list.size();

    while (true) {

        if (rendering_stopped_cpu.load(std::memory_order_relaxed)) {
            return;
        }

        const int index = next_pixel_index.fetch_add(1, std::memory_order_relaxed);
        if (index >= total_pixels) break;

        const auto& [i, j] = shuffled_pixel_list[index];
        const int pixel_index = j * image_width + i;

        Vec3 accumulated_color(0.0f);
        Vec3 mean(0.0f);
        Vec3 variance(0.0f);

        int dynamic_min_samples = min_samples;

        // Komşu varyans kontrolü
        bool has_high_variance_neighbor = false;
        float neighbor_variance_sum = 0.0f;
        int neighbor_count = 0;

        if (i >= 2 && i < image_width - 2 && j >= 2 && j < image_height - 2) {
            for (int dj = -2; dj <= 2; ++dj) {
                for (int di = -2; di <= 2; ++di) {
                    if (di == 0 && dj == 0) continue;
                    int ni = i + di;
                    int nj = j + dj;
                    float neighbor_var = variance_buffer[nj * image_width + ni];
                    neighbor_variance_sum += neighbor_var;
                    neighbor_count++;
                    if (neighbor_var > base_variance_threshold * 1.5f) {
                        has_high_variance_neighbor = true;
                    }
                }
            }

            if (has_high_variance_neighbor) {
                dynamic_min_samples = std::min(min_samples * 2, max_samples);
            }
        }

        int sample_count = 0;
        bool converged = false;

        for (int s = 0; s < max_samples && !converged; ++s) {
            Vec2 uv = stratified_halton(i, j, s, max_samples);
            Ray r = camera->get_ray(uv.u, uv.v);
            Vec3 sample_color = ray_color(r, bvh, lights, background_color,
                render_settings.max_bounces, s);

            const Vec3 delta = sample_color - mean;
            mean += delta / float(s + 1);
            variance += delta * (sample_color - mean);
            accumulated_color += sample_color;
            sample_count++;

            // Yakınsama kontrolü her 4 örnekte bir
            if (s >= dynamic_min_samples && (s & 0x3) == 0) {
                Vec3 var = variance / std::max(float(sample_count - 1), 1e-5f);
                float luminance_mean = mean.luminance();
                float luminance_var = var.luminance();
                float adaptive_threshold = base_variance_threshold;

                if (luminance_mean < 0.1f)
                    adaptive_threshold *= 2.0f;
                else if (luminance_mean > 0.9f)
                    adaptive_threshold *= 0.5f;

                if (neighbor_count > 0) {
                    float avg_neighbor_var = neighbor_variance_sum / neighbor_count;
                    adaptive_threshold *= std::max(0.5f, 1.0f - avg_neighbor_var * 0.5f);
                }

                float progress = float(s - dynamic_min_samples) / std::max(1.0f, float(max_samples - dynamic_min_samples));
                adaptive_threshold *= (1.0f - progress * 0.5f);

                if (luminance_var < adaptive_threshold) {
                    converged = true;
                }
            }
        }

        const Vec3 final_color = accumulated_color / float(sample_count);
        frame_buffer[pixel_index] = accumulated_color;
        sample_counts[pixel_index] = sample_count;
        variance_buffer[pixel_index] = variance.luminance() / std::max(float(sample_count - 1), 1e-5f);

        // TEK YER: ColorProcessor her şeyi yapsın → sRGB 0-1 döner
        Vec3 ldr = color_processor.processColor(final_color, i, j);
        // Gamma 2.2 (Standart sRGB - GPU ile aynı: powf(color, 1.0f / 2.2f))
        float cpu_gamma = 1.0f / 2.2f;
        uint8_t r = uint8_t(powf(ldr.x, cpu_gamma) * 255.0f + 0.5f);
        uint8_t g = uint8_t(powf(ldr.y, cpu_gamma) * 255.0f + 0.5f);
        uint8_t b = uint8_t(powf(ldr.z, cpu_gamma) * 255.0f + 0.5f);
        // SDL surface'a yaz (y ekseni ters olduğu için height-1-j)
        Uint32* pixel = (Uint32*)surface->pixels + (surface->h - 1 - j) * (surface->pitch / 4) + i;
        *pixel = SDL_MapRGB(surface->format, r, g, b);
        
        frame_buffer[pixel_index] = ldr;
        sample_counts[pixel_index] = samples_per_pixel;
    }
}

void Renderer::render_chunk_fixed_sampling(SDL_Surface* surface,
    const std::vector<std::pair<int, int>>& shuffled_pixel_list,
    std::atomic<int>& next_pixel_index,
    const HittableList& world,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color,
    const Hittable* bvh,
    const std::shared_ptr<Camera>& camera,
    const int total_samples_per_pixel)
{
    const int total_pixels = shuffled_pixel_list.size();
    const int samples_per_pixel = total_samples_per_pixel;

    while (true) {
        if (rendering_stopped_cpu.load(std::memory_order_relaxed)) {
            return;
        }

        const int index = next_pixel_index.fetch_add(1, std::memory_order_relaxed);
        if (index >= total_pixels) break;

        const auto& [i, j] = shuffled_pixel_list[index];
        const int pixel_index = j * image_width + i;

        Vec3 accumulated(0.0f);

        // Unrolled sample loop (çok güzel, dokunma)
        for (int s = 0; s < samples_per_pixel; s += 4) {
            Vec3 batch_color(0.0f);
            const int remaining = std::min(4, samples_per_pixel - s);

            for (int b = 0; b < remaining; ++b) {
                const Vec2 uv = stratified_halton(i, j, s + b, samples_per_pixel);
                const Ray r = camera->get_ray(uv.u, uv.v);
                batch_color += ray_color(r, bvh, lights, background_color,
                    render_settings.max_bounces, s + b);
            }
            accumulated += batch_color;
        }

        // Ortalama al
        const Vec3 avg_color = accumulated / float(samples_per_pixel);

        // TEK YER: ColorProcessor her şeyi yapsın → sRGB 0-1 döner
        Vec3 ldr = color_processor.processColor(avg_color, i, j);
        // Gamma 2.2 (Standart sRGB - GPU ile aynı: powf(color, 1.0f / 2.2f))
        float cpu_gamma = 1.0f / 2.2f;
        uint8_t r = uint8_t(powf(ldr.x, cpu_gamma) * 255.0f + 0.5f);
        uint8_t g = uint8_t(powf(ldr.y, cpu_gamma) * 255.0f + 0.5f);
        uint8_t b = uint8_t(powf(ldr.z, cpu_gamma) * 255.0f + 0.5f);

        // SDL surface'a yaz (y ekseni ters olduğu için height-1-j)
        Uint32* pixel = (Uint32*)surface->pixels + (surface->h - 1 - j) * (surface->pitch / 4) + i;
        *pixel = SDL_MapRGB(surface->format, r, g, b);

        frame_buffer[pixel_index] = ldr;
        sample_counts[pixel_index] = samples_per_pixel;
    }
}

void Renderer::render_chunk(SDL_Surface* surface,
    const std::vector<std::pair<int, int>>& shuffled_pixel_list,
    std::atomic<int>& next_pixel_index,
    const HittableList& world,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color,
    const Hittable* bvh,
    const std::shared_ptr<Camera>& camera,
    const int total_samples_per_pixel,
    const int current_sample)
{
    // color_processor.preprocess(frame_buffer);

    render_chunk_adaptive(surface, shuffled_pixel_list, next_pixel_index,
        world, lights, background_color, bvh, camera, total_samples_per_pixel);
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
        Li = point->getIntensity(hit_point, light_sample) * attenuation ;

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
    float shadow_transmittance = 1.0f;
    int shadow_layers = 0;
    
    while (remaining_dist > 0.001f && shadow_layers < 4) {
        HitRecord shadow_rec;
        if (bvh->hit(shadow_ray_current, 0.001f, remaining_dist, shadow_rec)) {
            
            // Check if blocker is a Volume
            if (shadow_rec.vdb_volume) {
                const VDBVolume* vdb = shadow_rec.vdb_volume;
                auto shader = vdb->volume_shader;
                
                // Get intersection interval
                float t_enter, t_exit;
                if (vdb->intersectTransformedAABB(shadow_ray_current, 0.001f, remaining_dist, t_enter, t_exit)) {
                    if (t_enter < 0.001f) t_enter = 0.001f;
                    if (t_exit > remaining_dist) t_exit = remaining_dist;
                    
                    // Shadow Ray Marching
                    // Lower quality for shadows is usually acceptable
                    float shadow_step = shader ? shader->quality.step_size * 2.0f : 0.2f;
                    if (shadow_step < 0.01f) shadow_step = 0.01f;
                    
                    float density_scale = (shader ? shader->density.multiplier : 1.0f) * vdb->density_scale;
                    float shadow_strength = shader ? shader->quality.shadow_strength : 1.0f;
                    
                    float t = t_enter;
                    Matrix4x4 inv_transform = vdb->getInverseTransform();
                    auto& mgr = VDBVolumeManager::getInstance();
                    int vol_id = vdb->getVDBVolumeID();
                    
                    // Jitter shadow start
                    t += ((float)rand() / RAND_MAX) * shadow_step;
                    
                    while (t < t_exit) {
                        Vec3 p = shadow_ray_current.at(t);
                        Vec3 local_p = inv_transform.transform_point(p);
                        float density = mgr.sampleDensityCPU(vol_id, local_p.x, local_p.y, local_p.z);
                        
                        // Apply same threshold as primary ray to prevent box shadows
                        if (density < 0.01f) density = 0.0f;
                        
                        if (density > 0.0f) {
                            float sigma_t = density * density_scale * shadow_strength;
                            shadow_transmittance *= exp(-sigma_t * shadow_step);
                        }
                        
                        if (shadow_transmittance < 0.01f) break;
                        t += shadow_step;
                    }
                }
                
                if (shadow_transmittance < 0.01f) {
                    return direct_light; // Fully blocked by volume
                }
                
                // Continue through volume
                // Move ray to exit point + epsilon
                float advance = t_exit + 0.001f;
                shadow_ray_current = Ray(shadow_ray_current.at(advance), L);
                remaining_dist -= advance;
                shadow_layers++;
            }
            else {
                // Opaque blocker found
                return direct_light;
            }
        }
        else {
            // No blocker found
            break; 
        }
    }
    
    // Apply calculated transmittance to light intensity
    if (shadow_transmittance < 0.99f) {
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
    const Vec3& background_color, int depth, int sample_index) {
    
    // =========================================================================
    // UNIFIED RAY COLOR - Matches GPU ray_color.cuh exactly
    // =========================================================================
    
    Vec3f color(0.0f);
    Vec3f throughput(1.0f);
    Ray current_ray = r;
    
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

        if (!bvh->hit(current_ray, 0.001f, std::numeric_limits<float>::infinity(), rec)) {
            // --- Infinite Grid Logic (Floor Plane Y=0) ---
            Vec3f final_bg_color = toVec3f(world.evaluate(current_ray.direction));
            
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
                             return val < width || val > (scale - width);
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
                         } else if (z_axis) { 
                             grid_col = Vec3f(0.2f, 0.8f, 0.2f); grid_alpha = 0.9f; // Green Z
                         } else if (x_line_p || z_line_p) { 
                             grid_col = Vec3f(0.40f); grid_alpha = 0.5f; // Major Lines (Darker Grey)
                         } else if (x_line_s || z_line_s) { 
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


        // --- VDB Volume Rendering ---
        if (rec.vdb_volume) {
            const VDBVolume* vdb = rec.vdb_volume;
             
            // Get entry and exit points
            float t_enter, t_exit;
            if (vdb->intersectTransformedAABB(current_ray, 0.001f, std::numeric_limits<float>::infinity(), t_enter, t_exit)) {
                 
                // Clamp t_enter
                if (t_enter < 0.001f) t_enter = 0.001f;
                
                // Ray marching setup
                // Use fixed step for now (TODO: Get from VolumeShader)
                float step_size = 0.1f; 
                
                float t = t_enter;
                float transparency = 1.0f;
                Vec3f accumulated_color(0.0f);
                
                auto& mgr = VDBVolumeManager::getInstance();
                int vol_id = vdb->getVDBVolumeID();
                Matrix4x4 inv_transform = vdb->getInverseTransform();

                // --- VOLUMETRIC RENDERING (CPU) ---
                // Physical Integration using Beer-Lambert Law

                // Access generic Volume Shader properties
                auto shader = vdb->volume_shader;
                
                 step_size = shader ? shader->quality.step_size : 0.1f;
                // Avoid infinite loops with bad step size
                if (step_size < 0.001f) step_size = 0.001f;

                float density_scale = (shader ? shader->density.multiplier : 1.0f) * vdb->density_scale;
                
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
                
                // ═══════════════════════════════════════════════════════════════
                // Anisotropy (Henyey-Greenstein phase function G parameter)
                // ═══════════════════════════════════════════════════════════════
                float anisotropy_g = shader ? shader->scattering.anisotropy : 0.0f;

                float current_transparency = 1.0f;
                Vec3f accumulated_vol_color(0.0f);

                
                // Jitter to reduce banding
                float jitter = ((float)rand() / RAND_MAX) * step_size;
                t += jitter;

                int steps = 0;
                int max_steps = shader ? shader->quality.max_steps : 256;

                while (t < t_exit && steps < max_steps) {
                    Vec3 p = current_ray.at(t);
                    Vec3 local_p = inv_transform.transform_point(p);
                    
                    // Sample Density (Standard Coordinates)
                    float density = mgr.sampleDensityCPU(vol_id, local_p.x, local_p.y, local_p.z);
                    
                    // CUTOFF REMOVED: User requested - was zeroing low densities
                    
                    // Edge falloff - smooth fade near bounding box boundaries
                    float edge_falloff = shader ? shader->density.edge_falloff : 0.0f;
                    if (edge_falloff > 0.0f && density > 0.0f) {
                        Vec3 local_min = vdb->getLocalBoundsMin();
                        Vec3 local_max = vdb->getLocalBoundsMax();
                        
                        // Calculate distance from edges (in local space)
                        float dx = std::min(local_p.x - local_min.x, local_max.x - local_p.x);
                        float dy = std::min(local_p.y - local_min.y, local_max.y - local_p.y);
                        float dz = std::min(local_p.z - local_min.z, local_max.z - local_p.z);
                        float edge_dist = std::min({dx, dy, dz});
                        
                        // Smooth falloff near edges
                        if (edge_dist < edge_falloff) {
                            float edge_factor = edge_dist / edge_falloff;
                            density *= edge_factor * edge_factor; // Quadratic falloff
                        }
                    }
                    
                    if (density > 0.001f) {  // GPU-matching threshold to filter float noise
                        // Physical coefficients: sigma_t = sigma_s + sigma_a
                        float sigma_s = density * density_scale * scattering_intensity;
                        float sigma_a = density * absorption_coeff;
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
                                 } else {
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
                                                 float s_density = mgr.sampleDensityCPU(vol_id, slocal_p.x, slocal_p.y, slocal_p.z);
                                                 
                                                 if (s_density > 1e-4f) {
                                                     shadow_transmittance *= exp(-s_density * density_scale * shadow_march_step);
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
                        Vec3f step_scattering_term = total_incoming_light * volume_albedo * sigma_s;
                        
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
                            
                            // Sample real temperature grid if available (GPU-matching)
                            float temperature = 0.0f;
                            if (mgr.hasTemperatureGrid(vol_id)) {
                                temperature = mgr.sampleTemperatureCPU(vol_id, local_p.x, local_p.y, local_p.z);
                            } else {
                                // Fallback: use density as temperature proxy
                                temperature = density;
                            }
                            
                            Vec3f blackbody_color(0.0f);
                            
                            // Check if ColorRamp is enabled
                            bool use_color_ramp = shader && shader->emission.color_ramp.enabled;
                            
                            if (use_color_ramp) {
                                // GPU-matching: Use temperature for ramp (not density)
                                float ramp_t = temperature * temperature_scale;
                                ramp_t = std::max(0.0f, std::min(ramp_t, 1.0f));
                                Vec3 ramp_color = shader->emission.color_ramp.sample(ramp_t);
                                blackbody_color = toVec3f(ramp_color);
                            }
                            else {
                                // Physical blackbody calculation
                                float temp_k = temperature * temperature_scale * 1500.0f;
                                temp_k = std::max(100.0f, std::min(temp_k, 10000.0f));
                                float t = temp_k / 100.0f;
                                float r, g, b;
                                
                                if (t <= 66.0f) {
                                    r = 255.0f;
                                    g = 99.4708025861f * logf(t) - 161.1195681661f;
                                    if (t <= 19.0f) {
                                        b = 0.0f;
                                    } else {
                                        b = 138.5177312231f * logf(t - 10.0f) - 305.0447927307f;
                                    }
                                } else {
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
                            float brightness = scaled_density * blackbody_intensity;
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
                // Both accumulated_vol_color and throughput are Vec3f now
                color += throughput * accumulated_vol_color;
                throughput *= current_transparency;
                
                // Move ray to exit point (handled below)

                
                // Move ray to exit point to continue tracing background
                current_ray = Ray(current_ray.at(t_exit + 0.0001f), current_ray.direction);
                
                // Check if we should stop
                if (current_transparency < 0.01f) break;  // Fixed: was using wrong variable 'transparency'
                
                continue; // Continue to next bounce (tracing what's behind volume)
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
                
                // Albedo
                Vec3 alb = pbsdf->getPropertyValue(pbsdf->albedoProperty, uv);
                albedo = toVec3f(alb).clamp(0.01f, 1.0f);
                
                // Roughness (Y channel)
                Vec3 rough = pbsdf->getPropertyValue(pbsdf->roughnessProperty, uv);
                roughness = static_cast<float>(rough.y);
                
                // Metallic (Z channel)
                Vec3 metal = pbsdf->getPropertyValue(pbsdf->metallicProperty, uv);
                metallic = static_cast<float>(metal.z);
                
                // Opacity
                opacity = pbsdf->get_opacity(uv);
                
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
                    } else {
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

        // --- Scatter ray ---
        Vec3 attenuation;
        Ray scattered;
        if (!rec.material || !rec.material->scatter(current_ray, rec, attenuation, scattered)) {
            break;
        }

        Vec3f atten_f = toVec3f(attenuation);
        throughput *= atten_f;
        
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

        // --- Emissive Contribution (OLD WORKING PATTERN) ---
        // Use polymorphic getEmission() for ALL material types, not just PrincipledBSDF
        Vec3 emitted = rec.material->getEmission(rec.uv, rec.point);
        emission = toVec3f(emitted);

        // --- Direct lighting with unified functions ---
        Vec3f direct_light(0.0f);
        
        if (light_count > 0 && transmission < 0.99f) {
            // Pick light using smart selection (matches GPU pick_smart_light)
            int light_index = pick_smart_light_unified(
                unified_lights.data(),
                light_count,
                hit_pos,
                Vec3::random_float()
            );
            
            if (light_index >= 0) {
                const UnifiedLight& light = unified_lights[light_index];
                
                // Sample light direction
                Vec3f wi;
                float distance;
                float light_attenuation;
                float rand_u = Vec3::random_float();
                float rand_v = Vec3::random_float();
                
                bool valid = sample_light_direction(
                    light, hit_pos, rand_u, rand_v,
                    &wi, &distance, &light_attenuation
                );
                
                if (valid) {
                    float NdotL = dot(N, wi);
                    
                    if (NdotL > 0.001f) {
                        // Shadow test
                        Vec3 shadow_origin = rec.point + rec.interpolated_normal * UnifiedConstants::SHADOW_BIAS;
                        Ray shadow_ray(shadow_origin, toVec3(wi));
                        
                        if (!bvh->occluded(shadow_ray, UnifiedConstants::SHADOW_BIAS, distance)) {
                            // Evaluate BRDF using unified function (with transmission handling like GPU)
                            float brdf_rand = Vec3::random_float();
                            Vec3f f = evaluate_brdf_unified(N, wo, wi, albedo, roughness, metallic, transmission, brdf_rand);
                            
                            // Light PDF
                            float pdf_light = compute_light_pdf(light, distance, 1.0f / light_count);
                            
                            // BRDF PDF for MIS
                            float pdf_brdf = pdf_brdf_unified(N, wo, wi, roughness);
                            float pdf_brdf_clamped = std::clamp(pdf_brdf, 0.01f, 5000.0f);
                            
                            // MIS weight
                            float mis_weight = power_heuristic(pdf_light, pdf_brdf_clamped);
                            
                            // Light radiance
                            Vec3f Li = light.color * light.intensity * light_attenuation;
                            
                            // Final contribution (matching GPU exactly)
                            direct_light = f * Li * NdotL * mis_weight * static_cast<float>(light_count);
                            
                            // Firefly clamp (matching GPU)
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
        
        color += throughput * total_contribution * opacity;

        current_ray = scattered;
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
// CYCLES-STYLE ACCUMULATIVE RENDERING (CPU)
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
    if (!cpu_accumulation_buffer.empty()) {
        std::fill(cpu_accumulation_buffer.begin(), cpu_accumulation_buffer.end(), Vec4{0.0f, 0.0f, 0.0f, 0.0f});
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
    extern RenderSettings render_settings;


    // Ensure accumulation buffer is allocated
    const size_t pixel_count = image_width * image_height;
    if (cpu_accumulation_buffer.size() != pixel_count) {
        cpu_accumulation_buffer.resize(pixel_count, Vec4{0.0f, 0.0f, 0.0f, 0.0f});
        cpu_accumulation_valid = true;
    }
    
    // Camera change detection
    if (scene.camera) {
        uint64_t current_hash = computeCPUCameraHash(*scene.camera);
        bool is_first = (cpu_last_camera_hash == 0);
        
        if (current_hash != cpu_last_camera_hash) {
            // Camera changed - reset accumulation
            std::fill(cpu_accumulation_buffer.begin(), cpu_accumulation_buffer.end(), Vec4{0.0f, 0.0f, 0.0f, 0.0f});
            cpu_accumulated_samples = 0;
            
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
    } else if (render_settings.is_final_render_mode) {
        target_max_samples = render_settings.final_render_samples > 0 ? render_settings.final_render_samples : 128;
    } else {
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
    
    // Build pixel list
    std::vector<std::pair<int, int>> pixel_list;
    pixel_list.reserve(pixel_count);
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            pixel_list.emplace_back(i, j);
        }
    }
    
    // Shuffle for better cache and visual distribution
    std::shuffle(pixel_list.begin(), pixel_list.end(), std::mt19937(std::random_device{}()));
    
    // Sparse progressive: First few samples render fewer pixels for faster preview
    // Sample 1: 1/16 pixels, Sample 2: 1/8, Sample 3: 1/4, Sample 4: 1/2, Sample 5+: all
    int sparse_divisor = 1;
    if (cpu_accumulated_samples == 0) sparse_divisor = 16;  // First pass: 1/16 pixels
    else if (cpu_accumulated_samples == 1) sparse_divisor = 8;
    else if (cpu_accumulated_samples == 2) sparse_divisor = 4;
    else if (cpu_accumulated_samples == 3) sparse_divisor = 2;
    
    size_t pixels_to_render = pixel_list.size() / sparse_divisor;
    if (pixels_to_render < 1000) pixels_to_render = pixel_list.size();  // Safety minimum
    
    std::atomic<int> next_pixel_index{0};
    std::atomic<bool> should_stop{false};
    
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
            
            Vec3 color_sum(0.0f);
            
            // Render samples for this pass
            for (int s = 0; s < samples_this_pass; ++s) {
                float u = (i + dist(rng)) / float(image_width);
                float v = (j + dist(rng)) / float(image_height);
                
                Ray ray = scene.camera->get_ray(u, v);
                
                // CRITICAL: Ensure we use the latest BVH
                // If scene.bvh was replaced, local pointer might be stale if captured?
                // But we access scene.bvh.get() directly.
                // However, check if scene.bvh is valid.
                if (!scene.bvh) {
                    // Empty scene: Draw Sky + Grid (matches ray_color logic)
                    Vec3 bg_color = world.evaluate(ray.direction); // Access Renderer::world
                    
                    if (!render_settings.is_final_render_mode && ray.direction.y < -0.0001f) {
                        float t = -ray.origin.y / ray.direction.y;
                        if (t > 0.0f) {
                             Vec3 p = ray.origin + ray.direction * t;
                             
                             float scale_major = 1.0f; 
                             float line_width = 0.02f;
                             float x_mod = std::abs(std::fmod(p.x, scale_major));
                             float z_mod = std::abs(std::fmod(p.z, scale_major));
                             bool x_line = x_mod < line_width || x_mod > (scale_major - line_width);
                             bool z_line = z_mod < line_width || z_mod > (scale_major - line_width);
                             bool x_axis = std::abs(p.z) < line_width * 2.0f;
                             bool z_axis = std::abs(p.x) < line_width * 2.0f;
                             
                             Vec3 grid_color(0.4f); // Brighter grey lines to be visible against background
                             bool hit_grid = false;
                             
                             if (x_axis) { grid_color = Vec3(0.8f, 0.2f, 0.2f); hit_grid = true; } 
                             else if (z_axis) { grid_color = Vec3(0.2f, 0.8f, 0.2f); hit_grid = true; } 
                             else if (x_line || z_line) { grid_color = Vec3(0.4f); hit_grid = true; } // Explicit check
                             
                             float dist = t;
                             float alpha = std::clamp((dist - 10.0f) / 30.0f, 0.0f, 1.0f); // 10-40 fade
                             
                             if (hit_grid) {
                                  bg_color = grid_color * (1.0f - alpha) + bg_color * alpha;
                             }
                        }
                    }
                    
                    color_sum += bg_color;
                    continue;
                }

                Vec3 sample_color = ray_color(ray, scene.bvh.get(), scene.lights, 
                                              scene.background_color, render_settings.max_bounces);
                color_sum += sample_color;
            }
            
            Vec3 new_color = color_sum / float(samples_this_pass);
            
            // Accumulate with previous samples
            Vec4& accum = cpu_accumulation_buffer[pixel_index];
            float prev_samples = accum.w;
            
            if (prev_samples > 0.0f) {
                // Progressive blend
                float new_total = prev_samples + samples_this_pass;
                Vec3 prev_color(accum.x, accum.y, accum.z);
                Vec3 blended = (prev_color * prev_samples + new_color * samples_this_pass) / new_total;
                
                accum.x = blended.x;
                accum.y = blended.y;
                accum.z = blended.z;
                accum.w = new_total;
                new_color = blended;
            } else {
                // First sample
                accum.x = new_color.x;
                accum.y = new_color.y;
                accum.z = new_color.z;
                accum.w = float(samples_this_pass);
            }
            
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
                } else {
                    // Manual Exposure calculation
                    float iso_mult = CameraPresets::ISO_PRESETS[scene.camera->iso_preset_index].exposure_multiplier;
                    float shutter_time = CameraPresets::SHUTTER_SPEED_PRESETS[scene.camera->shutter_preset_index].speed_seconds;
                    
                    // Use F-Stop Number
                    float f_number = 16.0f;
                    if (scene.camera->fstop_preset_index > 0) {
                        f_number = CameraPresets::FSTOP_PRESETS[scene.camera->fstop_preset_index].f_number;
                    } else {
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

// ============================================================================
// rebuildOptiXGeometry - Full OptiX Scene Rebuild
// ============================================================================
// This function performs a complete rebuild of OptiX geometry and material
// bindings. It regenerates ALL buffers including the critical material index
// buffer that maps triangles to SBT records.
//
// WHEN TO USE:
//   ✅ Object deletion (triangle count changes)
//   ✅ Object addition (new materials/triangles)
//   ✅ Material reassignment (SBT mapping changes)
//
// PERFORMANCE: ~200-500ms for medium scenes
// This is acceptable for infrequent operations like deletion.
//
// See OPTIX_MATERIAL_FIX.md for detailed explanation of why this is necessary.
// ============================================================================
void Renderer::rebuildOptiXGeometry(SceneData& scene, OptixWrapper* optix_gpu_ptr) {
    if (!optix_gpu_ptr) {
        SCENE_LOG_WARN("[OptiX] Cannot rebuild - no OptiX pointer");
        return;
    }
    
    // Handle empty scene
    if (scene.world.objects.empty()) {
        SCENE_LOG_INFO("[OptiX] Scene is empty, clearing GPU scene");
        optix_gpu_ptr->clearScene(); 
        optix_gpu_ptr->resetAccumulation();
        return;
    }
    
    // Global flag to block concurrent updates (Animation vs Rebuild)
    extern bool g_optix_rebuild_in_progress;
    g_optix_rebuild_in_progress = true;

    try {
        // Convert current Hittable objects to Triangle list
        if (scene.world.objects.empty()) {
            SCENE_LOG_WARN("[OptiX] No objects in scene");
            optix_gpu_ptr->clearScene();
            optix_gpu_ptr->resetAccumulation();
            g_optix_rebuild_in_progress = false;
            return;
        }

        // Parallel Extraction of Triangles from Hittable objects
        // (Replaces slow sequential dynamic_pointer_cast loop)
        size_t num_objects = scene.world.objects.size();
        std::vector<std::shared_ptr<Triangle>> triangles;
        triangles.reserve(num_objects);

        if (num_objects > 1000) {
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;

            size_t chunk_size = num_objects / num_threads;
            std::vector<std::future<std::vector<std::shared_ptr<Triangle>>>> futures;

            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t start = t * chunk_size;
                size_t end = (t == num_threads - 1) ? num_objects : (start + chunk_size);

                futures.push_back(std::async(std::launch::async, 
                    [&scene, start, end]() {
                        std::vector<std::shared_ptr<Triangle>> local_tris;
                        local_tris.reserve((end - start)); // Heuristic reservation
                        for (size_t i = start; i < end; ++i) {
                            if (auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[i])) {
                                // OPTIMIZATION: Skip invisible objects (e.g. source meshes for instancing)
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
        } else {
            // Sequential fallback for small object counts
            for (const auto& obj : scene.world.objects) {
                if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                    if (tri->visible) {
                         triangles.push_back(tri);
                    }
                }
            }
        }
        
        // ALLOW instance-only scenes (triangles.empty() is OK if we have instances)
        // If both are empty, clearScene was handled above.
        
        // Convert to OptiX format using internal assimpLoader
        OptixGeometryData optix_data;
        if (!triangles.empty()) {
             optix_data = assimpLoader.convertTrianglesToOptixData(triangles);
        }
        
        // CRITICAL: Validate material indices before build to prevent crash
        optix_gpu_ptr->validateMaterialIndices(optix_data);
        
        // ═══════════════════════════════════════════════════════════════════════════
        // BUILD MODE SELECTION: Single GAS vs TLAS/BLAS
        // ═══════════════════════════════════════════════════════════════════════════
        // TLAS mode is currently disabled by default to ensure stability.
        // To enable: Set use_tlas_mode = true in OptixWrapper or call buildFromDataTLAS
        // Benefits: Faster transform updates, proper instancing support
        // Note: Texture handling is identical in both modes.
        
        // FORCE TLAS MODE for Instancing Support (Foliage)
        optix_gpu_ptr->buildFromDataTLAS(optix_data, scene.world.objects);
        
        // NOTE: Removed cudaDeviceSynchronize() here!
        // It was causing GPU to freeze even when this runs in async thread.
        // The next optixLaunch() will naturally sync with any pending CUDA work.
        
        optix_gpu_ptr->resetAccumulation();
        
        // Only log on larger rebuilds to avoid spam
        if (triangles.size() > 1000) {
            SCENE_LOG_INFO("[OptiX] Geometry rebuilt - " + std::to_string(triangles.size()) + " triangles");
        }
    }
    catch (std::exception& e) {
        SCENE_LOG_ERROR("[OptiX] Rebuild failed: " + std::string(e.what()));
        // Try to recover by clearing the scene
        try {
            optix_gpu_ptr->clearScene();
            optix_gpu_ptr->resetAccumulation();
        } catch (...) {
            SCENE_LOG_ERROR("[OptiX] Recovery failed - GPU state may be corrupted");
        }
    }
}


// ============================================================================
// updateOptiXMaterialsOnly - Fast Material Update (No GAS Rebuild)
// ============================================================================
// This function updates ONLY material data on the GPU without rebuilding the
// geometry acceleration structure. This is ~100x faster than rebuildOptiXGeometry.
//
// CRITICAL OPTIMIZATION: Uses MaterialManager directly instead of iterating
// all triangles via convertTrianglesToOptixData. This reduces the update time
// from O(triangle_count) to O(material_count).
//
// WHEN TO USE:
//   ✅ Material property changes (color, roughness, metallic, emission)
//   ✅ Volumetric parameter changes (density, scattering, etc.)
//
// WHEN NOT TO USE:
//   ❌ Object deletion/addition (geometry changed → use rebuildOptiXGeometry)
//   ❌ Material slot reassignment (triangle's material ID changed → use rebuildOptiXGeometry)
//   ❌ Texture changes (need full rebuild for texture handles)
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
            } else if (mat->gpuMaterial) {
                // PrincipledBSDF with gpuMaterial
                gpu_mat = *mat->gpuMaterial;
                vol_info.is_volumetric = 0;
            } else {
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
    }
    catch (std::exception& e) {
        SCENE_LOG_ERROR("[OptiX] updateOptiXMaterialsOnly failed: " + std::string(e.what()));
    }
}

// ============================================================================
// WIND ANIMATION SYSTEM
// ============================================================================

#include "InstanceManager.h"
#include "HittableInstance.h"

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
        
        // 3. Animation Loop (Parallelized)
        // Updating 100k instances is heavy, let's try to be cache friendly.
        // #pragma omp parallel for (if you have OpenMP enabled)
        bool has_active_links = (group->active_hittables.size() == group->instances.size());
        
        for (size_t i = 0; i < group->instances.size(); ++i) {
            const auto& init = group->initial_instances[i];
            auto& curr = group->instances[i];

            // Calculate Noise/Wave
            // Simple sine approximation for wind waves
            // Phase uses X+Z mostly
            float pos_phase = (init.position.x + init.position.z) / wave;
            float t_phase = time * speed;
            
            // Multi-octave "sway"
            // Base wave + faster turbulence
            float sway_val = sinf(pos_phase + t_phase) + 
                             0.5f * sinf(pos_phase * 2.5f + t_phase * 1.3f * turbulence);
            
            // Apply Strength
            float rot_angle = sway_val * strength; // in degrees
            
            // Directional Sway logic
            float rot_z_delta = -dir.x * rot_angle; 
            float rot_x_delta = dir.z * rot_angle;  
            
            // Drift Prevention: Always add delta to INITIAL rotation, not current.
            curr.rotation.x = init.rotation.x + rot_x_delta;
            curr.rotation.z = init.rotation.z + rot_z_delta;
            
            // Update Active HittableInstance if linked
            if (has_active_links) {
                if (auto hittable = group->active_hittables[i].lock()) {
                    // We need to cast to HittableInstance to access setTransform
                    // Since we know we created them as HittableInstance, static_cast is risky but fast.
                    // dynamic_cast is safer.
                    if (auto hi = std::dynamic_pointer_cast<HittableInstance>(hittable)) {
                         Matrix4x4 new_mat = curr.toMatrix();
                         hi->setTransform(new_mat); // Updates inv_transform too
                         
                         // OPTIMIZATION: Direct GPU Update (Avoids iterating all scene objects)
                         if (this->optix_gpu_ptr && !hi->optix_instance_ids.empty()) {
                             float t[12];
                             const Matrix4x4& m = new_mat; 
                             t[0] = m.m[0][0]; t[1] = m.m[0][1]; t[2] = m.m[0][2]; t[3] = m.m[0][3];
                             t[4] = m.m[1][0]; t[5] = m.m[1][1]; t[6] = m.m[1][2]; t[7] = m.m[1][3];
                             t[8] = m.m[2][0]; t[9] = m.m[2][1]; t[10] = m.m[2][2]; t[11] = m.m[2][3];
                             
                             for(int id : hi->optix_instance_ids) {
                                  this->optix_gpu_ptr->updateInstanceTransform(id, t);
                             }
                         }
                    }
                }
            }
        }
        
        group->gpu_dirty = true;
        any_update = true;
    }

    // Commit changes to TLAS if any updates occurred
    if (any_update && this->optix_gpu_ptr) {
        this->optix_gpu_ptr->rebuildTLAS(); // Fast refit/update
    }
}
