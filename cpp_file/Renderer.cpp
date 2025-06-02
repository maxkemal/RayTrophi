#include "renderer.h"
#include <SDL_image.h>
#include "SpotLight.h"
#include <filesystem>
#include <execution>
#include <EmbreeBVH.h>

// Global atmosphere değişkeni
AtmosphereProperties g_atmosphere = {
    0.01f,   // sigma_s → hafif sis
    0.001f,  // sigma_a → hafif absorption
    0.0f,    // g → izotropik scatter
    1.0f,    // base density
    300.0f   // sıcaklık (örnek: 300 Kelvin - 27°C)
};

void updatePixel(SDL_Surface* surface, int i, int j, const Vec3SIMD& color) {
    Uint32* pixel = static_cast<Uint32*>(surface->pixels) + (surface->h - 1 - j) * surface->pitch / 4 + i;

    // Linear to sRGB dönüşüm (basit approx veya doğru dönüşüm kullanabilirsin)
    auto toSRGB = [](float c) {
        if (c <= 0.0031308f)
            return 12.92f * c;
        else
            return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        };

    int r = static_cast<int>(255 * std::clamp(toSRGB(color.x()), 0.0f, 1.0f));
    int g = static_cast<int>(255 * std::clamp(toSRGB(color.y()), 0.0f, 1.0f));
    int b = static_cast<int>(255 * std::clamp(toSRGB(color.z()), 0.0f, 1.0f));

    *pixel = SDL_MapRGB(surface->format, r, g, b);
  
}

std::vector<Vec3> normal_buffer(image_width* image_height);
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
void Renderer::applyOIDNDenoising(SDL_Surface* surface, int numThreads = 0, bool denoise = true, float blend = 0.8f) {
    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    int width = surface->w;
    int height = surface->h;

    // Renk verisini normalize ederek buffer'a aktar
    std::vector<float> colorBuffer(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        Uint8 r, g, b;
        SDL_GetRGB(pixels[i], surface->format, &r, &g, &b);
        colorBuffer[i * 3] = static_cast<float>(r) / 255.0f;
        colorBuffer[i * 3 + 1] = static_cast<float>(g) / 255.0f;
        colorBuffer[i * 3 + 2] = static_cast<float>(b) / 255.0f;
    }

    std::vector<float> normalData(width * height * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t pixel_index = y * width + x;
            size_t color_index = pixel_index * 3;
            // Eğer normal_buffer eksik veya küçükse sınır kontrolü yap
            if (pixel_index < normal_buffer.size()) {
                const Vec3& normal = normal_buffer[pixel_index];
                normalData[color_index] = normal.x;
                normalData[color_index + 1] = normal.y;
                normalData[color_index + 2] = normal.z;
            }

        }
    }


    // CUDA veya CPU cihazını seç
    oidn::DeviceRef device;
    if (isCudaAvailable()) {
        device = oidn::newDevice(oidn::DeviceType::CUDA);
    }
    else {
        device = oidn::newDevice(oidn::DeviceType::CPU);
    }
    device.set("numThreads", numThreads);
    device.commit();

    // OIDN buffer'larını oluştur
    oidn::BufferRef colorOIDNBuffer = device.newBuffer(colorBuffer.size() * sizeof(float));
    // oidn::BufferRef normalOIDNBuffer = device.newBuffer(normalData.size() * sizeof(float)); // Normal buffer
    oidn::BufferRef outputOIDNBuffer = device.newBuffer(colorBuffer.size() * sizeof(float));

    std::memcpy(colorOIDNBuffer.getData(), colorBuffer.data(), colorBuffer.size() * sizeof(float));
    // std::memcpy(normalOIDNBuffer.getData(), normalData.data(), normalData.size() * sizeof(float));

     // Filtreyi yapılandır ve çalıştır
    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", colorOIDNBuffer, oidn::Format::Float3, width, height);
    // filter.setImage("normal", normalOIDNBuffer, oidn::Format::Float3, width, height); // Normal verisini burada ekliyoruz
    filter.setImage("output", outputOIDNBuffer, oidn::Format::Float3, width, height);

    filter.set("hdr", false); // Normal map verisi için HDR (lineer veri)
    filter.set("srgb", true); // Gamma düzeltmesi uygulama
    filter.set("denoise", denoise);
    filter.commit();

    auto start = std::chrono::high_resolution_clock::now();
    filter.execute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Hataları kontrol et
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
        std::cerr << "OIDN error: " << errorMessage << std::endl;

    // Denoised veriyi al ve karıştır
    std::memcpy(colorBuffer.data(), outputOIDNBuffer.getData(), colorBuffer.size() * sizeof(float));
    for (int i = 0; i < width * height; ++i) {
        Uint8 r_orig, g_orig, b_orig;
        SDL_GetRGB(pixels[i], surface->format, &r_orig, &g_orig, &b_orig);

        Uint8 r = static_cast<Uint8>((colorBuffer[i * 3] * blend + r_orig / 255.0f * (1 - blend)) * 255);
        Uint8 g = static_cast<Uint8>((colorBuffer[i * 3 + 1] * blend + g_orig / 255.0f * (1 - blend)) * 255);
        Uint8 b = static_cast<Uint8>((colorBuffer[i * 3 + 2] * blend + b_orig / 255.0f * (1 - blend)) * 255);

        pixels[i] = SDL_MapRGB(surface->format, r, g, b);
    }
}



Renderer::Renderer(int image_width, int image_height, int samples_per_pixel, int max_depth)
    : image_width(image_width), image_height(image_height), aspect_ratio(static_cast<double>(image_width) / image_height), halton_cache(new float[MAX_DIMENSIONS * MAX_SAMPLES_HALTON]), color_processor(image_width, image_height)
{
    initialize_halton_cache();
    initialize_sobol_cache();
    frame_buffer.resize(image_width * image_height);
	sample_counts.resize(image_width * image_height, 0);
	variance_map.resize(image_width * image_height, 0.0f);
}

Renderer::~Renderer()
{
    frame_buffer.clear();
    sample_counts.clear();
    variance_map.clear();
}
void Renderer::set_window(SDL_Window* win) {
    window = win;
}
void Renderer::draw_progress_bar(SDL_Surface* surface, float progress) {
    const int bar_width = surface->w - 40;  // Kenarlarda 20 piksel boşluk bırakıyoruz
    const int bar_height = 20;
    const int bar_y = surface->h - 40;  // Alt kenardan 40 piksel yukarıda

    char percent_text[10];
    snprintf(percent_text, sizeof(percent_text), "%.1f%%", progress * 100);

}
bool Renderer::SaveSurface(SDL_Surface* surface, const char* file_path) {
    SDL_Surface* surface_to_save = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);
    /* int imgFlags = IMG_INIT_PNG;
     if (!(IMG_Init(imgFlags) & imgFlags)) {
         SDL_Log("SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError());
         SDL_Quit();
         return 1;
     }*/

    if (surface_to_save == NULL) {
        SDL_Log("Couldn't convert surface: %s", SDL_GetError());
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
void Renderer::removeFireflies(SDL_Surface* surface) {
    const float threshold = 8.0f; // Bu değeri sahnenize göre ayarlayabilirsiniz
    const int radius = 1; // Kontrol edilecek komşu piksel sayısı
    int width = surface->w;
    int height = surface->h;
    std::vector<Uint32> newPixels(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Uint32* pixels = static_cast<Uint32*>(surface->pixels);
            Uint32 currentPixel = pixels[y * width + x];
            Uint8 r, g, b;
            SDL_GetRGB(currentPixel, surface->format, &r, &g, &b);
            float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            float sum = 0;
            int count = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        Uint32 neighborPixel = pixels[ny * width + nx];
                        Uint8 nr, ng, nb;
                        SDL_GetRGB(neighborPixel, surface->format, &nr, &ng, &nb);
                        float neighborLuminance = 0.2126f * nr + 0.7152f * ng + 0.0722f * nb;

                        sum += neighborLuminance;
                        count++;
                    }
                }
            }

            float avgLuminance = sum / count;

            if (luminance > threshold * avgLuminance) {
                float scale = avgLuminance * threshold / luminance;
                r = static_cast<Uint8>(r * scale);
                g = static_cast<Uint8>(g * scale);
                b = static_cast<Uint8>(b * scale);
                newPixels[y * width + x] = SDL_MapRGB(surface->format, r, g, b);
            }
            else {
                newPixels[y * width + x] = currentPixel;
            }
        }
    }
    SDL_LockSurface(surface);
    std::memcpy(surface->pixels, newPixels.data(), width * height * sizeof(Uint32));
    SDL_UnlockSurface(surface);
}
Vec3 Renderer::getColorFromSurface(SDL_Surface* surface, int i, int j) {
    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    Uint32 pixel = pixels[(surface->h - 1 - j) * surface->pitch / 4 + i];

    Uint8 r, g, b;
    SDL_GetRGB(pixel, surface->format, &r, &g, &b);

    // sRGB to linear dönüşümü istersen buraya koy
    return Vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}

void Renderer::update_variance_map_from_surface(SDL_Surface* surface) {
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            Vec3 color = getColorFromSurface(surface, i, j);
            float luminance = color.luminance(); // veya başka bir noise ölçütü

            // Normalize veya ölçekle
            float scaled = std::clamp(luminance * 1.0f, 0.0f, 1.0f);
            variance_map[j * image_width + i] = Vec3(scaled);
        }
    }
}

void Renderer::update_variance_map_hybrid(SDL_Surface* surface) {
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            int index = j * image_width + i;

            Vec3 denoised = getColorFromSurface(surface, i, j);
            Vec3 raw = frame_buffer[index] / std::max(sample_counts[index], 1); // Ham ortalama

            Vec3 diff = (denoised - raw).abs();
            float hybrid_error = diff.luminance(); // veya max(diff.x, diff.y, diff.z);

            variance_map[index] = Vec3(std::clamp(hybrid_error, 0.0f, 1.0f));
        }
    }
}
static int point_light_pick_count = 0;
static int directional_pick_count = 0;
void Renderer::render_image(SDL_Surface* surface, SDL_Window* window,
    const int total_samples_per_pixel, const int samples_per_pass, SceneData& scene) {
    rendering_complete = false;
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
 
    std::thread display_thread(&Renderer::update_display, this, window, surface);

    const int num_passes = (total_samples_per_pixel + samples_per_pass - 1) / samples_per_pass;

    for (int pass = 0; pass < num_passes; ++pass) {
       // std::cout << "Starting pass " << pass + 1 << " of " << num_passes << std::endl;

        //  Shuffle full-resolution pixel list
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

        threads.clear();

        float progress = static_cast<float>(pass + 1) / num_passes;
        
        char title[100];
        snprintf(title, sizeof(title), "Rendering... %.1f%% Complete", progress * 100);
        SDL_SetWindowTitle(window, title);       
       
    }

    rendering_complete = true;
    display_thread.join();
    applyOIDNDenoising(surface, 0, true, 0.9);
    SDL_UpdateWindowSurface(window);   

}

void Renderer::render_Animation(SDL_Surface* surface, SDL_Window* window,
    const int total_samples_per_pixel, const int samples_per_pass,
    float fps, float duration,SceneData& scene) {

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    auto start_time = std::chrono::steady_clock::now();
    std::thread display_thread(&Renderer::update_display, this, window, surface);
    float frame_time = 1.0f / fps;
    int total_frames = static_cast<int>(duration * fps);
    std::filesystem::create_directory("render");
   
    for (int frame = 0; frame < total_frames; ++frame) {
      


        std::fill(frame_buffer.begin(), frame_buffer.end(), Vec3(0.0f));
        std::fill(sample_counts.begin(), sample_counts.end(), 0);
        SDL_FillRect(surface, NULL, SDL_MapRGB(surface->format, 0, 0, 0));
        float current_time = frame * frame_time;
              
		//  Zamanı güncelle
        std::cout << "\nFrame " << frame << " of " << total_frames
            << " at time " << current_time << "s" << std::endl;

        //  Animasyon matrislerini uygula
        if (!scene.animationDataList.empty()) {
            for (const auto& triangle : scene.world.objects) {
                auto trianglePtr = std::dynamic_pointer_cast<Triangle>(triangle);
                if (!trianglePtr) continue;

                std::string nodeName = trianglePtr->getNodeName();
                const aiNode* node = assimpLoader.getNodeByName(nodeName);
                if (!node) continue;

                for (const auto& animation : scene.animationDataList) {
                    if (animation.positionKeys.count(nodeName) == 0 &&
                        animation.rotationKeys.count(nodeName) == 0 &&
                        animation.scalingKeys.count(nodeName) == 0)
                        continue;

                    Matrix4x4 animTransform = animation.calculateAnimationTransform(animation, current_time, nodeName);
                    Matrix4x4 baseTransform = convert(AssimpLoader::getGlobalTransform(node));
                    Matrix4x4 finalTransform = baseTransform * animTransform;

                    trianglePtr->updateAnimationTransform(finalTransform);
                }
            }
            if (use_embree) {
                auto embree_ptr = std::dynamic_pointer_cast<EmbreeBVH>(scene.bvh);
                embree_ptr->clearAndRebuild(scene.world.objects);  //  garantili temiz başlangıç
            }

            else {
                //scene.bvh->updateTree(scene.world.objects, current_time, current_time + frame_time);
            }

        }
        if (!scene.boneData.boneNameToIndex.empty()) {
            std::vector<Matrix4x4> finalBoneMatrices(scene.boneData.boneNameToIndex.size());

            for (const auto& [boneName, boneIndex] : scene.boneData.boneNameToIndex) {
                aiNode* node = scene.boneData.boneNameToNode[boneName];
                if (!node) continue;

                Matrix4x4 globalNodeTransform = convert(AssimpLoader::getGlobalTransform(node));

                Matrix4x4 animTransform = Matrix4x4::identity();
                for (const auto& anim : scene.animationDataList) {
                    if (anim.positionKeys.count(boneName) > 0 ||
                        anim.rotationKeys.count(boneName) > 0 ||
                        anim.scalingKeys.count(boneName) > 0) {
                        animTransform = anim.calculateAnimationTransform(anim, current_time, boneName);
                        break;
                    }
                }

                Matrix4x4 offsetMatrix = scene.boneData.boneOffsetMatrices[boneName];
                const Matrix4x4& globalInverseTransform = scene.boneData.globalInverseTransform;               
                finalBoneMatrices[boneIndex] = globalInverseTransform * globalNodeTransform * animTransform * offsetMatrix;

            }

            for (auto& obj : scene.world.objects) {
                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                if (!tri || tri->vertexBoneWeights.empty()) continue;
                if (tri->vertexBoneWeights.size() == 3 &&
                    (!tri->vertexBoneWeights[0].empty() ||
                        !tri->vertexBoneWeights[1].empty() ||
                        !tri->vertexBoneWeights[2].empty())) {
                    std::cout << "[DEBUG] ✅ Skin uygulanıyor: " << tri->getNodeName() << "\n";
                    tri->apply_skinning(finalBoneMatrices);
                }
                else {
                    std::cout << "[DEBUG] ❌ Skin atlandı: " << tri->getNodeName() << "\n";
                }


            }

            if (use_embree) {
                auto embree_ptr = std::dynamic_pointer_cast<EmbreeBVH>(scene.bvh);
                embree_ptr->clearAndRebuild(scene.world.objects);
            }
        }

        for (auto& light : scene.lights) {
            const aiNode* node = assimpLoader.getNodeByName(light->nodeName);
            if (!node) continue;

            for (const auto& animation : scene.animationDataList) {
                if (animation.positionKeys.count(light->nodeName) > 0 ||
                    animation.rotationKeys.count(light->nodeName) > 0) {

                    Matrix4x4 baseTransform = convert(AssimpLoader::getGlobalTransform(node));
                    Matrix4x4 animTransform = animation.calculateAnimationTransform(animation, current_time, light->nodeName);
                    Matrix4x4 finalTransform = baseTransform * animTransform;

                    // Pozisyon
                    Vec3 pos = finalTransform.transform_point(Vec3(0, 0, 0));
                    light->position = pos;

                    // Yön (Directional ve Spot için)
                    if (light->type() == LightType::Directional || light->type() == LightType::Spot) {
                        Vec3 forward = finalTransform.transform_vector(Vec3(0, 0, -1)).normalize();
                        light->direction = forward;
                    }
                }
            }
        }

        if (scene.camera) {
            for (const auto& animation : scene.animationDataList) {
                if (animation.positionKeys.count(scene.camera->nodeName) > 0 ||
                    animation.rotationKeys.count(scene.camera->nodeName) > 0) {

                    // YALNIZCA animasyon transformunu hesapla
                    Matrix4x4 animTransform = animation.calculateAnimationTransform(animation, current_time, scene.camera->nodeName);

                    // Pozisyon ve yön
                    Vec3 pos = animTransform.transform_point(Vec3(0, 0, 0));
                    Vec3 forward = animTransform.transform_vector(Vec3(0, 0, -1)).normalize();
                    Vec3 look = pos + forward;
                    scene.camera->lookfrom = pos;
                    scene.camera->lookat = look;
                    if (scene.camera->vup.length_squared() < 1e-8) {
                        scene.camera->vup = Vec3(0, 1, 0);
                    }
                  
                }
            }
        }

        //  Render per frame
        const int num_passes = (total_samples_per_pixel + samples_per_pass - 1) / samples_per_pass;
        for (int pass = 0; pass < num_passes; ++pass) {
            next_row.store(0);
            threads.clear();
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
            threads.clear();
            char title[100];
            snprintf(title, sizeof(title), "Rendering Frame %d/%d - %.1f%% Complete",
                frame + 1, total_frames, (static_cast<float>(pass + 1) / num_passes) * 100);
            SDL_SetWindowTitle(window, title);
            applyOIDNDenoising(surface, 0, true, 0.9f);
            SDL_UpdateWindowSurface(window);
        }
      
        //  Kaydet
        char filename[100];
        snprintf(filename, sizeof(filename), "render/output_frame_%03d.png", frame + 1);
        if (SaveSurface(surface, filename)) {
            std::cout << "Frame " << (frame + 1) << "/" << total_frames
                << " saved successfully as " << filename << std::endl;
        }
        else {
            std::cerr << "Failed to save frame " << (frame + 1) << std::endl;
            return;
        }
    }
    
    rendering_complete = true;
    display_thread.join();
  
    SDL_SetWindowTitle(window, "Rendering Completed - All Frames Saved");
}
//void Renderer::create_scene(SceneData& scene, OptixWrapper* optix_gpu_ptr) {
//    std::string default_path = "e:/data/home/bedroom.gltf";
//    create_scene(scene, optix_gpu_ptr, default_path);
//}

void Renderer::create_scene(SceneData& scene, OptixWrapper* optix_gpu_ptr, const std::string& model_path) {
  
    // Önce sahneyi sıfırla
    scene.world.clear();
    scene.lights.clear();
    scene.animatedObjects.clear();
    scene.animationDataList.clear();
    scene.camera = nullptr;              // << önemli!
    scene.bvh = nullptr;                // yeni BVH oluşturulacaksa önce sıfırlanmalı
    scene.initialized = false;         // yeniden kurulacağı için
    assimpLoader.clearTextureCache();
    std::filesystem::path path(model_path);
    baseDirectory = path.parent_path().string() + "/";

    // ---- 1. Geometri ve animasyon yükle ----
    auto [triangles, loadedAnimations] = assimpLoader.loadModelToTriangles(model_path);
    scene.animationDataList = loadedAnimations;
 if (triangles.empty()) {
        std::cerr << "[HATA] Üçgen verisi yok, sahne yüklenemedi: " << model_path << std::endl;
 }
 else {
	 std::cout << "Dosyadan " << triangles.size() << " üçgen yüklendi.\n";
 }
    for (const auto& tri : triangles) {
        scene.world.add(tri);

        auto hittable = std::dynamic_pointer_cast<Hittable>(tri);
        if (hittable) {
            auto animatedObj = std::make_shared<AnimatedObject>(std::vector<std::shared_ptr<Hittable>>{hittable});
            scene.animatedObjects.push_back(animatedObj);
        }
    }
	
    // ---- 2. Kamera ve ışık verisi ----
    scene.lights = assimpLoader.getLights();
    scene.camera = assimpLoader.getDefaultCamera();
    auto embree_bvh = std::make_shared<EmbreeBVH>();
    embree_bvh->build(scene.world.objects);
    scene.bvh = embree_bvh;
    // ---- 3. GPU OptiX setup ----      
        std::cout << "OptiX geometri oluşturuluyor (convertTrianglesToOptixData)...\n";
        OptixGeometryData optix_data = assimpLoader.convertTrianglesToOptixData(triangles);
        optix_gpu_ptr->validateMaterialIndices(optix_data);
        optix_gpu_ptr->buildFromData(optix_data);

        if (scene.camera) {
            std::cout << "OptiX Kamerası ayarlanıyor...\n";
            optix_gpu_ptr->setCameraParams(*scene.camera);
        }

        if (!scene.lights.empty()) {
            std::cout << "OptiX çoklu ışıklar ayarlanıyor...\n";
            optix_gpu_ptr->setLightParams(scene.lights);
        }

        optix_gpu_ptr->setBackgroundColor(scene.background_color);
    // ---- 5. Bilgi ver ----
    std::cout << "Sahne oluşturuldu.\n";
   
        scene.initialized = true;
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

void Renderer::update_display(SDL_Window* window, SDL_Surface* surface) {

    while (!rendering_complete) {

        SDL_UpdateWindowSurface(window);
        float progress = static_cast<float>(completed_pixels) / (image_width * image_height) * 100.0f;

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                rendering_complete = true;
                return;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100ms aralıklarla güncelle
    }

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

// Sobol dizisini hesaplayan bir fonksiyon. Dimension vektörlerini tablodan alıyoruz.
float Renderer::sobol(int index, int dimension) {
    static const unsigned int direction_vectors[2][32] = {
        {0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000, 0x01000000,
         0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000, 0x00020000, 0x00010000,
         0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800, 0x00000400, 0x00000200, 0x00000100,
         0x00000080, 0x00000040, 0x00000020, 0x00000010, 0x00000008, 0x00000004, 0x00000002, 0x00000001},

        {0x80000000, 0xC0000000, 0xA0000000, 0xF0000000, 0x88000000, 0xCC000000, 0xAA000000, 0xFF000000,
         0x80800000, 0xC0C00000, 0xA0A00000, 0xF0F00000, 0x88880000, 0xCCCC0000, 0xAAAA0000, 0xFFFF0000,
         0x80008000, 0xC000C000, 0xA000A000, 0xF000F000, 0x88008800, 0xCC00CC00, 0xAA00AA00, 0xFF00FF00,
         0x80808080, 0xC0C0C0C0, 0xA0A0A0A0, 0xF0F0F0F0, 0x88888888, 0xCCCCCCCC, 0xAAAAAAAA, 0xFFFFFFFF}
    };

    // Gray code kullanarak optimizasyon
    unsigned int gray = index ^ (index >> 1);

    // Lookup table kullanarak hızlı bit sayımı
    static const unsigned char BitsSetTable256[256] = {
        #define B2(n) n,     n+1,     n+1,     n+2
        #define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
        #define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
        B6(0), B6(1), B6(1), B6(2)
    };

    unsigned int result = 0;
    const unsigned int* v = direction_vectors[dimension];

    // 8 bitlik parçalar halinde işlem
    while (gray) {
        result ^= v[BitsSetTable256[gray & 0xff]];
        gray >>= 8;
    }

    return static_cast<float>(result) * (1.0f / static_cast<float>(0xFFFFFFFF));
}
Renderer::SobolCache Renderer::cache;  // Static üyenin tanımlanması

float Renderer::get_sobol_value(int index, int dimension) {
    if (index < 0 || dimension < 0 || dimension >= DIMENSION_COUNT) {
        return 0.0f;
    }
    constexpr size_t MAX_CACHE_SIZE = 1024; // Örneğin, 1024 örnek sakla
    size_t cache_size = std::min(CACHE_SIZE, MAX_CACHE_SIZE);
    // Cache büyüklüğünü kontrol et
    if (index >= cache_size) {
        return sobol(index, dimension);
    }

    // Cache'i güncelle
    size_t current_computed = cache.last_computed_index.load();
    if (index >= current_computed) {
        size_t expected = current_computed;
        if (cache.last_computed_index.compare_exchange_weak(expected, index + 1)) {
            for (int d = 0; d < DIMENSION_COUNT; ++d) {
                for (size_t i = current_computed; i <= index; ++i) {
                    cache.values[d][i] = sobol(i, d);
                }
            }
        }
    }


    return cache.values[dimension][index];
}


void Renderer::initialize_sobol_cache() {
    sobol_cache.resize(MAX_DIMENSIONS); // Dış boyutları ayarlama
    for (int d = 0; d < MAX_DIMENSIONS; ++d) {
        sobol_cache[d].resize(MAX_SAMPLES_SOBOL); // İç boyutları ayarlama
        for (int i = 0; i < MAX_SAMPLES_SOBOL; ++i) {
            sobol_cache[d][i] = sobol(i, d); // Sobol dizisini hesaplama
        }
    }
}

void Renderer::precompute_sobol(int max_index) {
    for (int i = 0; i < 2; ++i) {  // İlk iki boyutu kullanıyoruz
        sobol_cache[i].resize(max_index);
        for (int j = 0; j < max_index; ++j) {
            sobol_cache[i][j] = sobol(j, i);
        }
    }
}

Vec2 Renderer::stratified_sobol(int x, int y, int sample_index, int samples_per_pixel) {
    int index = (y * image_width + x) * samples_per_pixel + sample_index;
    index = index % MAX_SAMPLES_SOBOL;
    float u = sobol_cache[0][index];
    float v = sobol_cache[1][index];

    // Sobol dizisinin örneklerini doğrudan kullanıyoruz
    u /= samples_per_pixel;
    v /= samples_per_pixel;

    return Vec2(
        (x + u) / image_width,
        (y + v) / image_height
    );
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

    int index = (y * image_width + x) * samples_per_pixel + sample_index;
    if (MAX_SAMPLES_HALTON > 0) {
        index = index % MAX_SAMPLES_HALTON;
    }
    else {
        index = 0;  // Eğer 0 olursa, varsayılan bir değer kullan.
    }


    // Tek boyutlu cache'den 2D array gibi erişim
    float u = halton_cache[index];                    // Birinci boyut
    float v = halton_cache[index + MAX_SAMPLES_HALTON]; // İkinci boyut

    return Vec2(
        (x + u) / image_width,
        (y + v) / image_height
    );
}
inline float luminance(const Vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

//float Renderer::compute_ambient_occlusion(HitRecord& rec, const ParallelBVHNode* bvh) {
//    if (rec.ao_computed) return rec.ao;
//
//    const int baseSamples = 32;
//    const int maxAdditionalSamples = 32;
//    const float aoRadius = 8.0f;
//    const float bias = 0.001f;
//
//    // Adaptive sampling based on surface complexity
//    Vec3 normal = rec.normal;
//    float complexity = 1.0f - std::abs(normal.dot(Vec3(0, 1, 0)));
//    int numSamples = baseSamples + static_cast<int>(complexity * maxAdditionalSamples);
//
//    float occlusion = 0.0f;
//    Vec3 samplePoint = rec.point + normal * bias;
//
//    // Use thread-local RNG for better performance
//    static thread_local std::mt19937 gen(std::random_device{}());
//    static thread_local std::uniform_real_distribution<float> dis(0.0f, 1.0f);
//
//    for (int i = 0; i < numSamples; ++i) {
//        // Stratified sampling
//        float u = (i + dis(gen)) / numSamples;
//        float v = dis(gen);
//
//        Vec3 randomDir = Vec3::random_in_hemisphere(normal);
//        Ray aoRay(samplePoint, randomDir);
//
//        float hitDistance=rec.t;
//        if (bvh->hit(aoRay, bias, aoRadius, rec)) {
//            float distanceFactor = 1.0f - (hitDistance / aoRadius);
//            occlusion += distanceFactor * distanceFactor; // Quadratic falloff
//        }
//    }
//
//    // Apply contrast enhancement
//    float rawAO = occlusion / static_cast<float>(numSamples);
//    float contrastedAO = std::pow(rawAO, 2.5f);
//
//    rec.ao = 1.0f - contrastedAO;
//    rec.ao_computed = true;
//
//    return rec.ao;
//}
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
    const int lowpass_samples = render_settings.min_samples;
    const int max_adaptive_samples = render_settings.max_samples;
    const float adaptive_threshold = render_settings.variance_threshold; // örnek: 0.05 (5%)

    int total_pixels = shuffled_pixel_list.size();

    if (!render_settings.use_adaptive_sampling) {
        render_chunk_fixed_sampling(surface, shuffled_pixel_list, next_pixel_index,
            world, lights, background_color, bvh, camera, total_samples_per_pixel);
        return;
    }

    while (true) {
        int index = next_pixel_index.fetch_add(1);
        if (index >= total_pixels) break;

        const auto& [i, j] = shuffled_pixel_list[index];
        int pixel_index = j * image_width + i;

        Vec3 mean(0.0f);
        Vec3 M2(0.0f);
        Vec3 accumulated(0.0f);

        // === İlk geçiş: lowpass örnekleme ===
        for (int s = 0; s < lowpass_samples; ++s) {
            Vec2 uv = stratified_halton(i, j, s, lowpass_samples);
            Ray r = camera->get_ray(uv.u, uv.v);
            Vec3 c = ray_color(r, bvh, lights, background_color, render_settings.max_bounces, s);
            accumulated += c;

            Vec3 delta = c - mean;
            mean += delta / float(s + 1);
            M2 += delta * (c - mean);
        }

        Vec3 variance = M2 / std::max(float(lowpass_samples - 1), 1e-5f);

        // === Hata Oranı Hesabı ===
        float lum_mean = std::max(mean.luminance(), 1e-4f); // Bölme hatasına karşı
        float lum_stddev = std::sqrt(variance.luminance());
        float error_ratio = lum_stddev / lum_mean;

        // === Log tabanlı varyansa dayalı ağırlık hesaplama ===
        float luminance_weight = std::log1p(variance.luminance() * 10.0f); // daha hızlı artış
        luminance_weight = std::clamp(luminance_weight, 0.0f, 1.0f);

        int adaptive_samples = static_cast<int>(luminance_weight * max_adaptive_samples);

        // Hata oranı yüksekse zorla daha fazla sample al
        if (error_ratio > adaptive_threshold) {
            adaptive_samples = max_adaptive_samples;
        }

        // Mutlak sınırlar içinde tut
        adaptive_samples = std::clamp(adaptive_samples, 1, total_samples_per_pixel - lowpass_samples);

        for (int s = 0; s < adaptive_samples; ++s) {
            Vec2 uv = stratified_halton(i, j, s + lowpass_samples, adaptive_samples);
            Ray r = camera->get_ray(uv.u, uv.v);
            Vec3 c = ray_color(r, bvh, lights, background_color, render_settings.max_bounces, s + lowpass_samples);
            accumulated += c;
        }

        int total_samples = lowpass_samples + adaptive_samples;

        frame_buffer[pixel_index] = accumulated;
        sample_counts[pixel_index] = total_samples;

        Vec3 avg_color = accumulated / float(total_samples);
        Vec3 final_color = color_processor.processColor(avg_color, i, j);
        updatePixel(surface, i, j, final_color);
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
    int total_pixels = shuffled_pixel_list.size();

    while (true) {
        int index = next_pixel_index.fetch_add(1);
        if (index >= total_pixels) break;

        const auto& [i, j] = shuffled_pixel_list[index];
        int pixel_index = j * image_width + i;

        Vec3 accumulated(0.0f);

        for (int s = 0; s < total_samples_per_pixel; ++s) {
            Vec2 uv = stratified_halton(i, j, s, total_samples_per_pixel);
            Ray r = camera->get_ray(uv.u, uv.v);
            Vec3 c = ray_color(r, bvh, lights, background_color, render_settings.max_bounces, s);
            accumulated += c;
        }

        frame_buffer[pixel_index] = accumulated;
        sample_counts[pixel_index] = total_samples_per_pixel;

        Vec3 avg_color = accumulated / float(total_samples_per_pixel);
        Vec3 final_color = color_processor.processColor(avg_color, i, j);
        updatePixel(surface, i, j, final_color);
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
    const int current_sample)  // Artık current_sample kullanılmıyor
{
   // color_processor.preprocess(frame_buffer);

    render_chunk_adaptive(surface, shuffled_pixel_list, next_pixel_index,
        world, lights, background_color, bvh, camera, total_samples_per_pixel);
}
// Işın ve küre kesişim testi
bool intersects_sphere(const Ray& ray, const Vec3& center, float radius) {
    Vec3 oc = ray.origin - center;
    float a = ray.direction.length_squared();
    float b = 2.0f * Vec3::dot(oc, ray.direction);
    float c = oc.length_squared() - radius * radius;
    float discriminant = b * b - 4 * a * c;
    return discriminant > 0;
}
inline float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-4f);
}
// --- Akıllı ışık seçimi ---
int Renderer::pick_smart_light(const std::vector<std::shared_ptr<Light>>& lights, const Vec3& hit_position) {

    int light_count = (int)lights.size();
    if (light_count == 0) return -1;

    // --- 1. Öncelik: Eğer Directional light varsa %50 ihtimalle seç ---
    for (int i = 0; i < light_count; i++) {
        if (lights[i]->type() == LightType::Directional) {

            if (random_double() < 0.33) {
                directional_pick_count++;
                return i;
            }
        }
    }

    // --- 2. Point lightlardan ağırlıklı seçim ---
    std::vector<float> weights(light_count, 0.0f);
    float total_weight = 0.0f;

    for (int i = 0; i < light_count; i++) {
        if (lights[i]->type() == LightType::Point) {
            Vec3 delta = lights[i]->position - hit_position;
            float distance = std::max(1.0f, delta.length());

            // NOT: intensity_magnitude zaten mesafeye göre zayıflatılmış, tekrar bölmeye gerek yok.
           
        }
    }

    // Eğer hiç ağırlık yoksa rastgele seç (fallback)
    if (total_weight < 1e-6f) {
        int random_light = std::clamp(int(random_double() * light_count), 0, light_count - 1);

        // Hangi tür seçildiğini kontrol edelim:
        if (lights[random_light]->type() == LightType::Point)
            point_light_pick_count++;
        else if (lights[random_light]->type() == LightType::Directional)
            directional_pick_count++;

        return random_light;
    }

    // Ağırlıklı seçim
    float r = random_double() * total_weight;
    float accum = 0.0f;
    for (int i = 0; i < light_count; i++) {
        accum += weights[i];
        if (r <= accum) {
            // Hangi tür olduğunu kontrol et ve sayacı artır:
            if (lights[i]->type() == LightType::Point)
                point_light_pick_count++;
            else if (lights[i]->type() == LightType::Directional)
                directional_pick_count++;

            return i;
        }
    }

    // Güvenlik için fallback (normalde buraya düşmemesi lazım)
    int fallback = std::clamp(int(random_double() * light_count), 0, light_count - 1);
    if (lights[fallback]->type() == LightType::Point)
        point_light_pick_count++;
    else if (lights[fallback]->type() == LightType::Directional)
        directional_pick_count++;
    return fallback;
}
Vec3 Renderer::calculate_direct_lighting_single_light(
    const Hittable* bvh,
    const std::shared_ptr<Light>& light,
    const HitRecord& rec,
    const Vec3& normal,
    float ao_factor,
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
        light_sample = hit_point + L * 10000.0f;
        to_light = L;
        light_distance = std::numeric_limits<float>::infinity();
        Li = directional->getIntensity(hit_point, light_sample);
    }
    else if (auto point = std::dynamic_pointer_cast<PointLight>(light)) {
        light_sample = point->random_point();
        to_light = light_sample - hit_point;
        light_distance = to_light.length();
        L = to_light / light_distance;

        Li = point->getIntensity(hit_point, light_sample);

        float area = 4.0f * M_PI * point->getRadius() * point->getRadius();
        pdf_light = (1.0f / area) * pdf_light_select;
    }
    else {
        return direct_light;
    }

    // --- Shadow ---
    Ray shadow_ray(hit_point + N * 0.0001f, L);
    if (bvh->occluded(shadow_ray, 0.0001f, light_distance))
        return direct_light;

    float NdotL = std::fmax(Vec3::dot(N, L), 0.0f);
    if (NdotL < 1e-4f) return direct_light;

    // --- BRDF Hesabı (Specular + Diffuse) ---
    Vec3 H = (L + V).normalize();
    float NdotV = std::fmax(Vec3::dot(N, V), 0.00001f);
    float NdotH = std::fmax(Vec3::dot(N, H), 0.00001f);
    float VdotH = std::fmax(Vec3::dot(V, H), 0.00001f);

    float alpha = roughness * roughness;
    PrincipledBSDF psdf;
    // Specular bileşeni
    float D = psdf.DistributionGGX(N, H, roughness);
    float G = psdf.GeometrySmith(N, V, L, roughness);
    Vec3 F = psdf.fresnelSchlickRoughness(VdotH, F0, roughness);

    Vec3 specular = psdf.evalSpecular(N,V,L,F0,roughness);

    // Diffuse bileşeni
    Vec3 kd = (1.0f - metallic) * (Vec3(1.0f) - F);
    Vec3 diffuse = kd * albedo/M_PI;

    // Toplam BRDF
    Vec3 brdf = diffuse + specular;

    // Işık katkısı
    Vec3 direct = brdf * Li * NdotL;

    return direct;
}


Vec3 Renderer::ray_color(const Ray& r, const Hittable* bvh,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color, int depth, int sample_index) {
    Vec3 final_color(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray current_ray = r;    

	// --- Ray tracing döngüsü ---
    for (int bounce = 0; bounce < render_settings.max_bounces; ++bounce) {
        HitRecord rec;

        if (!bvh->hit(current_ray, 0.001f, std::numeric_limits<float>::infinity(), rec)) {
            float falloff = (bounce == 0) ? 1.0f : std::pow(0.3f, bounce); // Enerji kaybı
            final_color += throughput * background_color*falloff;
            break;
        }

        int light_index = -1;
        if (!lights.empty())
            light_index = pick_smart_light(lights, rec.point);

        apply_normal_map(rec);

        Vec3 attenuation;
        Ray scattered;
        if (!rec.material->scatter(current_ray, rec, attenuation, scattered))
            break;

        throughput *= attenuation;

        // --- Russian Roulette ---
       
            float max_channel = std::max(throughput.x, std::max(throughput.y, throughput.z));
            float continuation_prob = std::clamp(max_channel, 0.0f, 0.98f);
            if (random_double() > continuation_prob)
                break;
            throughput /= continuation_prob;       

        // --- Emissive katkı ---
        Vec3 emitted = rec.material->getEmission(rec.u, rec.v, rec.point)*2;
        float transmission = rec.material->getTransmission(rec.uv);
        float opacity = rec.material->get_opacity(rec.uv);

        Vec3 direct_light(0.0f);

        // --- Direct light sadece ışık varsa hesapla ---
        if (light_index >= 0) {
            direct_light = calculate_direct_lighting_single_light(
                bvh, lights[light_index], rec, rec.interpolated_normal, 0.0f, current_ray);
            direct_light *= (1.0f - transmission);
        }

        final_color += throughput * (emitted + 5*direct_light * opacity);

        current_ray = scattered;
    }
    return final_color;
}

float Renderer::radical_inverse(unsigned int bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f; // 1 / (2^32)
}
float Renderer::compute_ambient_occlusion(HitRecord& rec, const ParallelBVHNode* bvh) {
    

    const int baseSamples = 2;
    const int maxAdditionalSamples = 2;
    const float aoRadius = 8.0f;
    const float bias = 0.001f;

    // Adaptive sampling based on surface complexity
    Vec3 normal = rec.normal;
    float complexity = 1.0f - std::abs(normal.dot(Vec3(0, 1, 0)));
    int numSamples = baseSamples + static_cast<int>(complexity * maxAdditionalSamples);

    float occlusion = 0.0f;
    Vec3 samplePoint = rec.point + normal * bias;

    // Use thread-local RNG for better performance
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < numSamples; ++i) {
        // Stratified sampling
        float u = (i + dis(gen)) / numSamples;
        float v = dis(gen);

        Vec3 randomDir = Vec3::random_in_hemisphere(normal);
        Ray aoRay(samplePoint, randomDir);


        float hitDistance = rec.t;
        if (bvh->hit(aoRay, bias, aoRadius, rec)) {
            float distanceFactor = 1.0f - (hitDistance / aoRadius);
            occlusion += distanceFactor * distanceFactor; // Quadratic falloff
        }
    }

    // Apply contrast enhancement
    float rawAO = occlusion / static_cast<float>(numSamples);
    float contrastedAO = std::pow(rawAO, 2.5f);


    return 0;
}
Vec3 Renderer::fresnel_schlick(float cosTheta, Vec3 F0) {
    return F0 + (Vec3(1.0f) - F0) * pow(1.0f - cosTheta, 5.0f);
}
// Farklı ışık türleri için alternatif değerler
namespace LightAttenuation {
    // Küçük ışık kaynakları (masa lambası, mum, vb.)
    struct Small {
        static constexpr float constant = 1.0f;
        static constexpr float linear = 0.7f;
        static constexpr float quadratic = 1.8f;
    };

    // Orta büyüklükte ışık kaynakları (tavan lambası, sokak lambası)
    struct Medium {
        static constexpr float constant = 1.0f;
        static constexpr float linear = 0.35f;
        static constexpr float quadratic = 0.44f;
    };

    // Büyük ışık kaynakları (projektör, büyük spot ışıklar)
    struct Large {
        static constexpr float constant = 1.0f;
        static constexpr float linear = 0.22f;
        static constexpr float quadratic = 0.20f;
    };
}
inline float lerp(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}


inline Vec3 fresnelSchlickRoughness(float cosTheta, const Vec3& F0, float roughness) {
    // Roughness arttıkça Fresnel eğrisi yumuşatılır (özellikle kenar parlamaları bastırılır)
    float factor = powf(1.0f - cosTheta, 5.0f);
    Vec3 oneMinusRough = Vec3(1.0f - roughness); // roughness → F'nin tavanını etkiler
    Vec3 fresnel = F0 + (Vec3::max(oneMinusRough, F0) - F0) * factor;
    return fresnel;
}
