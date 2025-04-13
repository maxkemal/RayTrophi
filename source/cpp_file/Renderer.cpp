#include "renderer.h"
#include <SDL_image.h>
#include "SpotLight.h"
#include <filesystem>
#include <execution>
#include <EmbreeBVH.h>



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
bool SaveSurface(SDL_Surface* surface, const char* file_path) {
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

void Renderer::render_image(SDL_Surface* surface, SDL_Window* window,
    const int total_samples_per_pixel, const int samples_per_pass) {

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    auto start_time = std::chrono::steady_clock::now();

    SceneData scene = create_scene(use_embree);  // ✅ Yeni sahne yapısı

    auto create_scene_end_time = std::chrono::steady_clock::now();
    auto create_scene_duration = std::chrono::duration<double, std::milli>(create_scene_end_time - start_time);
    std::cout << "Create Scene Duration: " << create_scene_duration.count() / 1000 << " seconds" << std::endl;

    std::thread display_thread(&Renderer::update_display, this, window, surface);

    const int num_passes = (total_samples_per_pixel + samples_per_pass - 1) / samples_per_pass;

    for (int pass = 0; pass < num_passes; ++pass) {
        std::cout << "Starting pass " << pass + 1 << " of " << num_passes << std::endl;

        // 🔀 Shuffle full-resolution pixel list
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
        std::cout << "\rRendering progress: " << std::fixed << std::setprecision(2) << progress << "%" << std::flush;

        char title[100];
        snprintf(title, sizeof(title), "Rendering... %.1f%% Complete", progress * 100);
        SDL_SetWindowTitle(window, title);

        std::cout << "Pass " << pass + 1 << " completed. Progress: " << (progress * 100) << "%" << std::endl;
        float blend = 0.3f + 0.5f * (float(pass) / (num_passes - 1));
        applyOIDNDenoising(surface, 0, true, blend);
        update_variance_map_hybrid(surface);

    }

    rendering_complete = true;
    display_thread.join();
    applyOIDNDenoising(surface, 0, true, 1);
    SDL_UpdateWindowSurface(window);

    std::cout << "\nRender completed." << std::endl;

    auto render_end_time = std::chrono::steady_clock::now();
    auto render_duration = std::chrono::duration<double, std::milli>(render_end_time - create_scene_end_time);
    auto total_duration = std::chrono::duration<double, std::milli>(render_end_time - start_time);

    std::cout << "Render Duration: " << render_duration.count() / 1000 << " seconds" << std::endl;
    std::cout << "Total Duration: " << total_duration.count() / 1000 << " seconds" << std::endl;

    if (SaveSurface(surface, "image/output.png")) {
        std::cout << "Image saved successfully!" << std::endl;
    }
    else {
        std::cerr << "Failed to save image." << std::endl;
    }
}

Matrix4x4 convert(const aiMatrix4x4& aiMat) {
    Matrix4x4 mat;
    mat.m[0][0] = aiMat.a1; mat.m[0][1] = aiMat.a2; mat.m[0][2] = aiMat.a3; mat.m[0][3] = aiMat.a4;
    mat.m[1][0] = aiMat.b1; mat.m[1][1] = aiMat.b2; mat.m[1][2] = aiMat.b3; mat.m[1][3] = aiMat.b4;
    mat.m[2][0] = aiMat.c1; mat.m[2][1] = aiMat.c2; mat.m[2][2] = aiMat.c3; mat.m[2][3] = aiMat.c4;
    mat.m[3][0] = aiMat.d1; mat.m[3][1] = aiMat.d2; mat.m[3][2] = aiMat.d3; mat.m[3][3] = aiMat.d4;
    return mat;
}

void Renderer::render_Animation(SDL_Surface* surface, SDL_Window* window,
    const int total_samples_per_pixel, const int samples_per_pass,
    float fps, float duration) {

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    auto start_time = std::chrono::steady_clock::now();

    float frame_time = 1.0f / fps;
    int total_frames = static_cast<int>(duration * fps);
    std::filesystem::create_directory("render");

    for (int frame = 0; frame < total_frames; ++frame) {
        SDL_FillRect(surface, NULL, SDL_MapRGB(surface->format, 0, 0, 0));
        float current_time = frame * frame_time;

        // 🚀 Yeni sahne kur
        SceneData scene = create_scene(use_embree);

        std::cout << "\nFrame " << frame << " of " << total_frames
            << " at time " << current_time << "s" << std::endl;

        // 🎞️ Animasyon matrislerini uygula
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
                embree_ptr->updateGeometryFromTrianglesFromSource(scene.world.objects);
            }

            else {
                //scene.bvh->updateTree(scene.world.objects, current_time, current_time + frame_time);
            }

        }

        // 🎨 Render per frame
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
                threads.emplace_back(&Renderer::render_worker, this,
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

            char title[100];
            snprintf(title, sizeof(title), "Rendering Frame %d/%d - %.1f%% Complete",
                frame + 1, total_frames, (static_cast<float>(pass + 1) / num_passes) * 100);
            SDL_SetWindowTitle(window, title);

            applyOIDNDenoising(surface, 0, true, 0.85f);
            SDL_UpdateWindowSurface(window);
        }

        // 💾 Kaydet
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
    SDL_SetWindowTitle(window, "Rendering Completed - All Frames Saved");
}

SceneData Renderer::create_scene(bool use_embree) {
    SceneData scene;

    scene.world.clear();
    scene.lights.clear();
    scene.background_color = Vec3(0.10, 0.15, 0.2) * 5.0;
    scene.animatedObjects.clear();
    scene.animationDataList.clear();

    std::string model_path = "e:/data/home/interior1.gltf";

    std::filesystem::path path(model_path);
    baseDirectory = path.parent_path().string() + "/";
    std::cout << "Dosyanın dizin yolu: " << baseDirectory << std::endl;

    auto [triangles, loadedAnimations] = assimpLoader.loadModelToTriangles(model_path);
    scene.animationDataList = loadedAnimations;

    for (const auto& tri : triangles) {
        scene.world.add(tri);
        auto hittable = std::dynamic_pointer_cast<Hittable>(tri);
        if (hittable) {
            auto animatedObj = std::make_shared<AnimatedObject>(std::vector<std::shared_ptr<Hittable>>{hittable});
            scene.animatedObjects.push_back(animatedObj);
        }
    }

    scene.lights = assimpLoader.getLights();
    scene.camera = assimpLoader.getDefaultCamera();
    std::cout << "Total objects in the scene: " << scene.world.size() << std::endl;

    // ⚡️ Seçimli BVH (Embree veya kendi BVH'n)
    if (use_embree) {
        auto embree_bvh = std::make_shared<EmbreeBVH>();
        embree_bvh->build(scene.world.objects);  // Triangle'ları veriyoruz
        scene.bvh = embree_bvh;
        std::cout << "[Embree] BVH yapısı oluşturuldu." << std::endl;
    }
    else {
        scene.bvh = std::make_shared<ParallelBVHNode>(scene.world.objects, 0, scene.world.size(), 0.0, 1.0);
        std::cout << "[Custom BVH] Kendi BVH yapısı oluşturuldu." << std::endl;
    }

    return scene;
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
void Renderer::render_chunk_adaptive_pass(SDL_Surface* surface,
    const std::vector<std::pair<int, int>>& shuffled_pixel_list,
    std::atomic<int>& next_pixel_index,
    const HittableList& world,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color,
    const  Hittable* bvh,
    const std::shared_ptr<Camera>& camera,
    const int max_samples,
    const int current_sample)
{
    const float threshold = 0.001f;
    int total_pixels = shuffled_pixel_list.size();
    int start_sample = current_sample * max_samples;

    while (true) {
        int index = next_pixel_index.fetch_add(1);
        if (index >= total_pixels) break;

        const auto& [i, j] = shuffled_pixel_list[index];
        int pixel_index = j * image_width + i;

        Vec3 variance = variance_map[pixel_index];
        float luminance = std::clamp(variance.luminance(), 0.0f, 1.0f); // normalizasyon       

        if (luminance < threshold) continue; // Gürültü düşükse örnekleme yapma
        int extra_samples = std::clamp(int(luminance * 1000.0f), 1, max_samples);
        Vec3 accumulated(0.0f);
        for (int s = 0; s < extra_samples; ++s) {
            Vec2 uv = stratified_halton(i, j, s + start_sample, start_sample + extra_samples);
            Ray r = camera->get_ray(uv.u, uv.v);
            Vec3 c = ray_color(r, bvh, lights, background_color, MAX_DEPTH, s + start_sample);
            accumulated += c;
        }

        frame_buffer[pixel_index] += accumulated;
        sample_counts[pixel_index] += extra_samples;

        Vec3 avg_color = frame_buffer[pixel_index] / sample_counts[pixel_index];
        Vec3 final_color = color_processor.processColor(avg_color, i, j);
        updatePixel(surface, i, j, final_color);
    }
}

void Renderer::render_chunk_lowpass(SDL_Surface* surface,
    const std::vector<std::pair<int, int>>& shuffled_pixel_list,
    std::atomic<int>& next_pixel_index,
    const HittableList& world,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color,
    const   Hittable* bvh,
    const std::shared_ptr<Camera>& camera,
    const int min_samples)
{
    const float epsilon = 1e-5f;
   
    variance_map.resize(image_width * image_height, Vec3(0.0f));

    while (true) {
        int index = next_pixel_index.fetch_add(1);
        if (index >= shuffled_pixel_list.size()) break;
       

        const auto& [i, j] = shuffled_pixel_list[index];
        int pixel_index = j * image_width + i;

        Vec3 mean(0.0f), M2(0.0f), accumulated(0.0f);
        for (int s = 0; s < min_samples; ++s) {
            Vec2 uv = stratified_halton(i, j, s, min_samples);
            Ray r = camera->get_ray(uv.u, uv.v);
            Vec3 c = ray_color(r, bvh, lights, background_color, MAX_DEPTH, s);

            accumulated += c;

            Vec3 delta = c - mean;
            mean += delta / float(s + 1);
            M2 += delta * (c - mean);
        }

        variance_map[pixel_index] = M2 / std::max(float(min_samples - 1), epsilon);
        frame_buffer[pixel_index] += accumulated;
        sample_counts[pixel_index] += min_samples;

        Vec3 avg_color = frame_buffer[pixel_index] / sample_counts[pixel_index];
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
    const   Hittable* bvh,
    const std::shared_ptr<Camera>& camera,
    const int samples_per_pass,
    const int current_sample)
{
    if (samples_per_pass == 0) return;
    // render_chunk içinde
    int lowpass_samples =25;
   

    color_processor.preprocess(frame_buffer);

    if (current_sample == 0) {
        render_chunk_lowpass(surface, shuffled_pixel_list, next_pixel_index,
            world, lights, background_color, bvh, camera, lowpass_samples);
    }
    else {
        render_chunk_adaptive_pass(surface, shuffled_pixel_list, next_pixel_index,
            world, lights, background_color, bvh, camera, samples_per_pass, current_sample);
    }
}

Vec3 Renderer::ray_color(const Ray& r, const   Hittable* bvh,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color, int depth, int sample_index) {
    Vec3 final_color(0, 0, 0);
    Vec3 throughput(1, 1, 1);
    Ray current_ray = r;
    float total_distance = 0.0f;
    float opacity;
    // Örnekleme indeksini thread-safe şekilde takip et
    static std::atomic<uint64_t> global_sample_index(0);
    int local_sample_index = global_sample_index.fetch_add(1, std::memory_order_relaxed);

    // Ray pathing için ana döngü
    for (int bounce = 0; bounce < MAX_DEPTH; ++bounce) {
        // Minimum mesafe, kendisiyle kesişmeyi önler
        float tmin = 0.0001f;
        HitRecord rec;
        // Hiçbir şeye çarpmadan ilerleyen ışın için arkaplan rengi hesapla
        if (!bvh->hit(current_ray, tmin, std::numeric_limits<double>::infinity(), rec)) {
            // Işın hiçbir şeye çarpmadı, arkaplan rengi hesapla
            float segment_distance = FLT_MAX; // Arkaplan için max mesafe
            total_distance += segment_distance;

            if (atmosferic_effect_enabled) {
                // Atmosferik efektleri uygula
                throughput = atmosphericEffects.attenuateSegment(throughput, segment_distance, 100);
                Vec3 segment_contribution = atmosphericEffects.calculateSegmentContribution(segment_distance, 100);
                final_color += throughput * atmosphericEffects.applyAtmosphericEffects(background_color, 100);
            }
            else {
                // Temel arkaplan rengi
                final_color += throughput* background_color;
            }
            break; // Işın pathing tamamlandı
        }
        Ray scattered ;
        if (rec.material->type() == MaterialType::Dielectric) {                    
				tmin = 0.02;
            bvh->hit(current_ray, tmin, std::numeric_limits<double>::infinity(), rec);
        }
        // Normal map uygula 
        apply_normal_map(rec);
        Vec3 attenuation;
       
       
        // Materyal tipine göre farklı işlemler
        MaterialType mat_type = rec.material->type();
        // Materyal tipine göre maksimum derinliği aşıp aşmadığımızı kontrol et
        int max_depth_for_material = rec.material->get_max_depth();
        if (bounce >= max_depth_for_material) {
            break;
        }
      
        if (!rec.material->scatter(current_ray, rec, attenuation, scattered)) {
            // Materyal ışını absorbe etti, başka sıçrama yok
           
            break;
        }
             
        // Materyal ışını yansıttı, katkıyı hesapla
        throughput *= attenuation;
		// Eğer ışın absorbe edilmediyse, katkıyı ekle
        if (rec.material->type() != MaterialType::Dielectric && rec.material->type() != MaterialType::Volumetric){
            // Diğer tüm materyal tipleri (Metal, Lambertian, PrincipledBSDF, vs.)
            // Russian Roulette optimizasyonu - düşük katkılı yolları sonlandır
            float continuation_probability = 1.0f;
            float max_channel = std::max(throughput.x, std::max(throughput.y, throughput.z));
            continuation_probability = std::min(max_channel,0.95f);

            if (random_double() > continuation_probability) {
                break; // yolu sonlandır
            }

            throughput /= continuation_probability; // normalize et
            // Emisyon katkısını ekle - bu noktada throughput zaten zayıflatılmış durumda
            Vec3 emitted = rec.material->getEmission(rec.u, rec.v, rec.point);
            opacity = rec.material->get_opacity(rec.uv);
			final_color += emitted * throughput * opacity;

            // Doğrudan ışık katkısını hesapla

            Vec3 direct_light = calculate_direct_lighting(bvh, lights, rec, rec.interpolated_normal, 0);

            // Doğrudan aydınlatma katkısını ekle
            final_color += throughput * direct_light * opacity;
        }
      
        Vec3 offset_origin = rec.point + rec.normal * 0.001f; // Küçük bir bias ekle
        scattered.origin = offset_origin; // Yeni ışını güncelle
        // Yeni ışını bir sonraki sıçrama için hazırla
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
    if (rec.ao_computed) return rec.ao;

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

    rec.ao = 1.0f - contrastedAO;
    rec.ao_computed = true;

    return rec.ao;
}
Vec3 fresnel_schlick(float cosTheta, Vec3 F0) {
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

Vec3 Renderer::calculate_light_contribution(
    const std::shared_ptr<Light>& light,
    const Vec3& point,
    const Vec3& shading_normal,    
    bool is_global, const HitRecord& rec
) {
    Vec3 to_light;
    Vec3 intensity = light->intensity;
    float distance = 0.0f;
    bool is_directional = false;
    Vec3 point_vec3;
    Vec3 light_pos;
    Vec3 random_light_pos;
    float shininess = 128.0f; // Speküler sertlik (metaller için 128+ olabilir)

    // Işık türüne göre temel değerleri belirle
    if (auto directional_light = std::dynamic_pointer_cast<DirectionalLight>(light)) {
        // Directional light için mesafe sonsuz kabul edilir
        is_directional = true;
        to_light = -directional_light->random_point();
        distance = std::numeric_limits<float>::infinity();
    }
    else if (auto point_light = std::dynamic_pointer_cast<PointLight>(light)) {
        light_pos = point_light->random_point();
        to_light = light_pos - point;
        distance = to_light.length();
        is_directional = false;
        // Raw intensity - zayıflama daha sonra hesaplanacak
        intensity = point_light->getIntensity();
    }
    else if (auto area_light = std::dynamic_pointer_cast<AreaLight>(light)) {
        random_light_pos = area_light->random_point();
        to_light = random_light_pos - point;
        distance = to_light.length();
        is_directional = false;
        // Raw intensity - zayıflama daha sonra hesaplanacak
        intensity = area_light->getIntensity();
    }
    else if (auto spot_light = std::dynamic_pointer_cast<SpotLight>(light)) {
        light_pos = spot_light->position;
        to_light = light_pos - point;
        distance = to_light.length();
        is_directional = false;
        // Raw intensity - zayıflama daha sonra hesaplanacak
        intensity = spot_light->getIntensity(point);
    }
    else {
        return Vec3(0, 0, 0);
    }
    to_light = to_light.normalize();
    // Diffuse hesaplama   
      // Mesafe zayıflaması (directional ışık hariç)
   
    float attenuation = 1.0f;       
    float cos_theta = std::max(0.0, Vec3::dot(shading_normal, to_light));
    //cos_theta = std::pow(cos_theta, 0.5f); // 1'den küçük üs kullanarak zayıflamayı azaltır
  
  
    if (!is_directional) {
        float luminance = 0.2126f * intensity.x + 0.7152f * intensity.y + 0.0722f * intensity.z;

        constexpr float MAX_LUMINANCE = 64000.0f;

        float t = std::clamp(std::log2(luminance + 1.0f) / std::log2(MAX_LUMINANCE + 1.0f), 0.0f, 1.0f);

        float linear = lerp(0.7f, 0.22f, t);
        float quadratic = lerp(1.8f, 0.20f, t);

        attenuation = 1.0f / (1.0f + linear * distance + quadratic * distance * distance);

    }

    // Final diffuse
    // Vec3 emited = rec.material->emitted(rec.u, rec.v, rec.point);
    // Diffuse hesaplama (ALBEDO EKLENDİ)
    Vec3 diffuse = intensity * cos_theta*  attenuation;
    // diffuse += emited;
     // **Speküler Bileşeni**
    /* Vec3 half_vector = (to_light + view_direction).normalize();
     float spec_angle = std::max(0.0, Vec3::dot(shading_normal, half_vector));
     float specular_intensity = pow(spec_angle, shininess);
     Vec3 specular = intensity * specular_intensity* attenuation;*/
     // **Toplam Işık Katkısı**
    return  diffuse;

}

Vec3 Renderer::calculate_direct_lighting(
    const   Hittable* bvh,
    const std::vector<std::shared_ptr<Light>>& lights,
    const HitRecord& rec,
    const Vec3& normal, float ao_factor
) {
    const Vec3& background_color = Vec3(0, 0, 0);
    Vec3 direct_light(0, 0, 0);
    const Vec3& hit_point = rec.point;
    HitRecord shadow_rec;  // Create a single HitRecord object and reuse it
  
    // Sadece rastgele seçilmiş bir ışık kaynağını örnekle
    //static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, lights.size() - 1);
    const auto& light = lights[dist(gen)];
    float shininess = rec.material->get_shininess();
    float metallic = rec.material->get_metallic();
    Vec3 to_light;
    float light_distance = std::numeric_limits<float>::infinity();
    Vec3 light_contribution;
    Ray shadow_ray(hit_point, to_light);
   
    for (const auto& light : lights) {
        // Işık hesaplamaları için erken çıkış
        float max_intensity = light->getIntensity(rec.point).max_component();
        if (max_intensity < 0.001f) continue; // Çok zayıf ışıkları atla
        if (const auto* directional_light = dynamic_cast<const DirectionalLight*>(light.get())) {
            //to_light = -directional_light->direction.normalize();           

            to_light = -directional_light->random_point().normalize();
            light_contribution = calculate_light_contribution(light, hit_point, normal,  shininess, rec);
        }
        else if (const auto* point_light = dynamic_cast<const PointLight*>(light.get())) {
            to_light = point_light->random_point() - hit_point;
            light_distance = to_light.length();
            to_light = to_light.normalize();
            light_contribution = calculate_light_contribution(light, hit_point, normal,  shininess, rec);
        }
        else if (const auto* spot_light = dynamic_cast<const SpotLight*>(light.get())) {
            to_light = (spot_light->position) - hit_point;
            light_distance = to_light.length();
            to_light = to_light.normalize();

            // Spot ışığın yönü ile noktanın konumu arasındaki açıyı kontrol et
            float cos_theta = Vec3::dot(to_light, spot_light->direction);
            if (cos_theta > std::cos(spot_light->angle_degrees)) {
                // Spot ışık konisinin içindeyse katkıyı hesapla
                float falloff = std::pow(cos_theta, 2.0f); // Cosine falloff; exponent ayarlanabilir
                light_contribution = calculate_light_contribution(light, hit_point, normal,  shininess, rec) * falloff;
            }
            else {
                // Nokta koni dışında, katkı sıfır
                light_contribution = (0.0f, 0.0f, 0.0f);
            }
        }
        else if (const auto* area_light = dynamic_cast<const AreaLight*>(light.get())) {

            to_light = area_light->random_point() - hit_point;
            light_distance = to_light.length();
            to_light = to_light.normalize();
            light_contribution += calculate_light_contribution(light, hit_point, normal,  shininess, rec);

        }
        else {
            continue;
        }
        if (bvh->occluded(Ray(hit_point, to_light), 0.001f, light_distance)) {
            continue;
        }


        direct_light += light_contribution;
    }
    // Tonemapping benzeri bir sınırlama uygula
   // direct_light = direct_light / (direct_light + Vec3(1.0f, 1.0f, 1.0f));
    return direct_light;
}
