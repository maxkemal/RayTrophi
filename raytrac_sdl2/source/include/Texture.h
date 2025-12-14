#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <SDL_image.h>
#include <cuda_runtime.h>
#include "Vec2.h"
#include "Vec3SIMD.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <assimp/scene.h>      // aiTexture, aiTexel, aiScene
#include <assimp/texture.h>    // aiTexture tanımı (bazı assimp versiyonlarında gerek)
#include <globals.h>
#include <atomic>

enum class TextureType {
    Unknown, Albedo, Normal, Roughness, Metallic, Emission, AO, Transmission, Opacity
};

struct CompactVec4 {
    uint8_t r, g, b, a;
    CompactVec4() : r(0), g(0), b(0), a(255) {}
    CompactVec4(uint8_t r_, uint8_t g_, uint8_t b_, uint8_t a_ = 255)
        : r(r_), g(g_), b(b_), a(a_) {
    }

    Vec3 to_linear_rgb(bool srgb, bool aces) const {
        auto srgb_to_linear = [](uint8_t c) -> float {
            float f = c / 255.0f;
            return (f <= 0.04045f) ? (f / 12.92f) : powf((f + 0.055f) / 1.055f, 2.4f);
            };
        auto tonemap = [](float x) -> float {
            float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
            return (x * (a * x + b)) / (x * (c * x + d) + e);
            };

        float rL = srgb ? srgb_to_linear(r) : r / 255.0f;
        float gL = srgb ? srgb_to_linear(g) : g / 255.0f;
        float bL = srgb ? srgb_to_linear(b) : b / 255.0f;

        if (aces) {
            rL = tonemap(rL);
            gL = tonemap(gL);
            bL = tonemap(bL);
        }

        return Vec3(rL, gL, bL);
    }

    float alpha() const { return a / 255.0f; }
    bool is_gray() const { return r == g && r == b; }
};
class FileTextureCache {
public:
    struct FileTextureInfo {
        int width;
        int height;
        bool has_alpha;
        bool is_gray_scale;
        std::time_t last_modified;
    };

    static FileTextureCache& instance() {
        static FileTextureCache cache;
        return cache;
    }

    bool get(const std::string& filepath, FileTextureInfo& info) const {
        auto it = cache_map.find(filepath);
        if (it != cache_map.end()) {
            // Dosya değiştirildi mi kontrol et
            std::time_t current_mod_time = get_file_modification_time(filepath);
            if (current_mod_time == it->second.last_modified) {
                info = it->second;
                return true;
            }
            // Stale cache entry - kaldır
            cache_map.erase(it);
        }
        return false;
    }

    void put(const std::string& filepath, const FileTextureInfo& info) {
        cache_map[filepath] = info;
    }

    void clear() {
        cache_map.clear();
    }

    size_t size() const {
        return cache_map.size();
    }

private:
    FileTextureCache() = default;
    mutable std::unordered_map<std::string, FileTextureInfo> cache_map;

    std::time_t get_file_modification_time(const std::string& filepath) const {
        try {
            return std::filesystem::last_write_time(filepath).time_since_epoch().count();
        }
        catch (...) {
            return 0;
        }
    }
};

class TextureCache {
public:
    struct TextureInfo {
        int width;
        int height;
        bool has_alpha;
        bool is_gray_scale;
    };

    static TextureCache& instance() {
        static TextureCache cache;
        return cache;
    }

    bool get(const std::string& name, TextureInfo& info) const {
        auto it = cache_map.find(name);
        if (it != cache_map.end()) {
            info = it->second;
            return true;
        }
        return false;
    }

    void put(const std::string& name, const TextureInfo& info) {
        cache_map[name] = info;
    }

    void clear() {
        cache_map.clear();
    }

    size_t size() const {
        return cache_map.size();
    }

private:
    TextureCache() = default;
    std::unordered_map<std::string, TextureInfo> cache_map;
};
class Texture {
public:
    Texture(const aiTexture* tex, TextureType type, const std::string& name = "")
        : type(type), is_srgb(type == TextureType::Albedo), is_aces(type == TextureType::Emission) {
        if (!tex) {
            SCENE_LOG_WARN("Texture pointer null, skip");
            return;
        }
        m_is_loaded = false;
        is_gpu_uploaded = false;
        std::string texture_name = name.empty() ? "unnamed_texture" : name;

        // Cache kontrol et - SADECE embedded texture için (name boş değilse)
        if (!name.empty() && name.find("embedded_") == 0) {
            TextureCache::TextureInfo cached_info;
            // Cache hit durumunda SADECE metadata'yı al, pixel decode etme!
            if (TextureCache::instance().get(name, cached_info)) {
                SCENE_LOG_INFO("[EMBEDDED CACHE HIT] Skipping decode, using cached metadata: " + name);
                width = cached_info.width;
                height = cached_info.height;
                has_alpha = cached_info.has_alpha;
                is_gray_scale = cached_info.is_gray_scale;
                m_is_loaded = false;  // ← Bu texture zaten cache'de, pixel decode'a gerek yok
                return;  // ← Constructor'dan çık, decode yapma!
            }
        }

        // RAW embedded (RGBA)
        if (tex->mHeight != 0) {
           // SCENE_LOG_INFO("[DECODE] Starting RAW texture decode for: " + texture_name);
            decode_raw(tex);

            if (!name.empty()) {
                TextureCache::instance().put(name, { width, height, has_alpha, is_gray_scale });
              //  SCENE_LOG_INFO("[CACHE STORE] Cached metadata for: " + name);
            }
            return;
        }

        // Compressed embedded (PNG/JPG)
       // SCENE_LOG_INFO("[DECODE] Starting COMPRESSED texture decode for: " + texture_name);
        decode_compressed(tex);

        if (!name.empty()) {
            TextureCache::instance().put(name, { width, height, has_alpha, is_gray_scale });
           // SCENE_LOG_INFO("[CACHE STORE] Cached metadata for: " + name);
        }
    }

    // ===== Constructor Disk Yüklemesi - img_load_fast ile =====
    Texture(const std::string& filename, TextureType type)
        : type(type), is_srgb(type == TextureType::Albedo), is_aces(type == TextureType::Emission) {

        m_is_loaded = false;
        is_gpu_uploaded = false;
        auto start_time = std::chrono::high_resolution_clock::now();

        // Cache kontrol et
        FileTextureCache::FileTextureInfo cached_info;
        if (FileTextureCache::instance().get(filename, cached_info)) {
            width = cached_info.width;
            height = cached_info.height;
            has_alpha = cached_info.has_alpha;
            is_gray_scale = cached_info.is_gray_scale;

            // ===== CACHE HIT - Paralel yükleme (senin kodu) =====
            SDL_Surface* surface = IMG_Load(filename.c_str());  // ← img_load_fast kullan
            if (surface) {
                int pixel_count = width * height;
                pixels.resize(pixel_count);
                if (SDL_LockSurface(surface) != 0) {
                    SCENE_LOG_ERROR("Failed to lock surface (cache hit) for: " + filename);
                    SDL_FreeSurface(surface);
                    return;
                }

                SDL_Surface* converted_surface = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGBA32, 0);
                SDL_UnlockSurface(surface);
                SDL_FreeSurface(surface);

                if (!converted_surface) {
                     SCENE_LOG_ERROR("Failed to convert surface format (cache hit) for: " + filename);
                     return;
                }
                surface = converted_surface;
                
                if (SDL_LockSurface(surface) != 0) {
                     SCENE_LOG_ERROR("Failed to lock converted surface (cache hit) for: " + filename);
                     SDL_FreeSurface(surface);
                     return;
                }

                SDL_PixelFormat* fmt = surface->format;
                uint8_t* data = static_cast<uint8_t*>(surface->pixels);
                if (!data) {
                    SCENE_LOG_ERROR("Converted Surface pixels are null (cache hit) for: " + filename);
                    SDL_UnlockSurface(surface);
                    SDL_FreeSurface(surface);
                    return;
                }

                int pitch = surface->pitch;
                int bpp = fmt->BytesPerPixel;
                if(bpp != 4) {
                     SCENE_LOG_ERROR("Converted surface (cache hit) BPP != 4");
                     SDL_UnlockSurface(surface);
                     SDL_FreeSurface(surface);
                     return;
                }

                bool is_gray_local = true;

                try {
                    for (int y = 0; y < height; ++y) {
                         Uint32* row_ptr = reinterpret_cast<Uint32*>(data + y * pitch);
                        for (int x = 0; x < width; ++x) {
                            Uint32 pixel_val = row_ptr[x];
                            Uint8 r, g, b, a;
                            SDL_GetRGBA(pixel_val, fmt, &r, &g, &b, &a);
                            CompactVec4 px(r, g, b, a);
                            pixels[y * width + x] = px;
                            if (is_gray_local && !px.is_gray()) {
                                is_gray_local = false;
                            }
                        }
                    }
                } catch(...) {}

                
                is_gray_scale = is_gray_local; // Just set local

                SDL_UnlockSurface(surface);
                SDL_FreeSurface(surface);
                m_is_loaded = true;

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                SCENE_LOG_INFO("[FILE CACHE HIT] " + filename +
                    " | " + std::to_string(width) + "x" + std::to_string(height) +
                    " | Single-threaded safe load" +
                    " | " + std::to_string(duration.count()) + "ms");
                return;
            }
            else {
                SCENE_LOG_ERROR("[FILE CACHE HIT BUT LOAD FAILED] " + filename);
                pixels.clear();
                m_is_loaded = false;
                return;
            }
        }
        else {
            SCENE_LOG_WARN("[FILE CACHE MISS] '" + filename + "' - will load from disk");
        }

        // ===== CACHE MISS - Disk'ten yükle =====
        SDL_Surface* surface = IMG_Load(filename.c_str());  // ← img_load_fast kullan
        if (!surface) {
            SCENE_LOG_ERROR("[FILE LOAD ERROR] Failed to load: " + filename +
                "\nError: " + std::string(IMG_GetError()));
            return;
        }

        width = surface->w;
        height = surface->h;
        int pixel_count = width * height;

        SCENE_LOG_INFO("[FILE DECODE] Starting decode for: " + filename +
            " | Resolution: " + std::to_string(width) + "x" + std::to_string(height));

        if (SDL_LockSurface(surface) != 0) {
            SCENE_LOG_ERROR("Failed to lock surface for: " + filename + " | Error: " + std::string(SDL_GetError()));
            SDL_FreeSurface(surface);
             return;
        }

        // Convert to RGBA32 to ensure consistent memory layout and avoid format issues
        SDL_Surface* converted_surface = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGBA32, 0);
        SDL_UnlockSurface(surface); // Unlock original
        SDL_FreeSurface(surface);   // Free original

        if (!converted_surface) {
            SCENE_LOG_ERROR("Failed to convert surface format for: " + filename + " | Error: " + std::string(SDL_GetError()));
            return;
        }

        surface = converted_surface; // Use converted one
        width = surface->w;
        height = surface->h;
        
        if (SDL_LockSurface(surface) != 0) {
             SCENE_LOG_ERROR("Failed to lock converted surface for: " + filename);
             SDL_FreeSurface(surface);
             return;
        }

        SDL_PixelFormat* fmt = surface->format;
        uint8_t* data = static_cast<uint8_t*>(surface->pixels);
        if (!data) {
             SCENE_LOG_ERROR("Converted surface pixels are null for: " + filename);
             SDL_UnlockSurface(surface);
             SDL_FreeSurface(surface);
             return;
        }
        
        int pitch = surface->pitch;
        int bpp = fmt->BytesPerPixel;
        has_alpha = SDL_ISPIXELFORMAT_ALPHA(fmt->format);

        // Paranoid check
        if (bpp != 4) {
             SCENE_LOG_ERROR("Converted surface BPP is not 4! It is: " + std::to_string(bpp));
             SDL_UnlockSurface(surface);
             SDL_FreeSurface(surface);
             return;
        }

        pixels.resize(width * height);
        bool is_gray_local = true;

        try {
            // Direct memory access for 32-bit RGBA
            // SDL_PIXELFORMAT_RGBA32 -> Memory: R, G, B, A (on Little Endian usually)
            // But we use SDL_GetRGBA to be safe or just mask
            // Optimally, since we converted to RGBA32, we know the layout. 
            // RGBA32 Alias: SDL_PIXELFORMAT_RGBA8888 on Big Endian, ABGR8888 on Little Endian?
            // Actually SDL defines RGBA32 as the format where R is the first byte in memory? 
            // Let's rely on SDL_GetRGBA to decode the Uint32 just to be safe from Endianness confusion, 
            // but since we iterating by ptr, we can read Uint32.
            
            for (int y = 0; y < height; ++y) {
                Uint32* row_ptr = reinterpret_cast<Uint32*>(data + y * pitch);
                for (int x = 0; x < width; ++x) {
                    Uint32 pixel_val = row_ptr[x];
                    Uint8 r, g, b, a;
                    SDL_GetRGBA(pixel_val, fmt, &r, &g, &b, &a); // Optimized by SDL for specific formats

                    CompactVec4 px(r, g, b, a);
                    pixels[y * width + x] = px;

                    if (is_gray_local && !px.is_gray()) {
                        is_gray_local = false;
                    }
                }
            }
        }
        catch (const std::exception& e) {
             SCENE_LOG_ERROR("Exception during pixel copy: " + std::string(e.what()));
        }
        catch (...) {
             SCENE_LOG_ERROR("Unknown exception during pixel copy");
        }

        is_gray_scale = is_gray_local;
        SDL_UnlockSurface(surface);
        SDL_FreeSurface(surface); // Free converted surface
        m_is_loaded = true;

        // Cache update with correct data
        std::time_t mod_time = 0;
        try {
            mod_time = std::filesystem::last_write_time(filename).time_since_epoch().count();
        }
        catch (const std::exception& e) {
            SCENE_LOG_WARN("Failed to get file modification time for: " + filename + " Error: " + e.what());
        }
        catch (...) {
            SCENE_LOG_WARN("Failed to get file modification time for: " + filename);
        }
        FileTextureCache::instance().put(filename, { width, height, has_alpha, is_gray_scale, mod_time });

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        SCENE_LOG_INFO("[FILE LOAD SUCCESS] '" + filename + "' | " + std::to_string(width) + "x" +
            std::to_string(height) + (has_alpha ? " | alpha" : " | opaque") +
            (is_gray_scale ? " | grayscale" : " | color") +
            " | Single-threaded safe load" +
            " | Cache size now: " + std::to_string(FileTextureCache::instance().size()) +
            " | Total time: " + std::to_string(duration.count()) + "ms");
    }
    Vec3 get_color(float u, float v) const {
        if (!m_is_loaded || pixels.empty()) return Vec3(0);
        u = std::clamp(u, 0.0f, 1.0f);
        v = std::clamp(v, 0.0f, 1.0f);
        int x = static_cast<int>(u * (width - 1));
        int y = static_cast<int>((1.0 - v) * (height - 1));
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return pixels[y * width + x].to_linear_rgb(is_srgb, is_aces);
    }

    float get_alpha(float u, float v) const {
        if (!m_is_loaded || pixels.empty()) return 1.0f;
        u = std::clamp(u, 0.0f, 1.0f);
        v = std::clamp(v, 0.0f, 1.0f);
        int x = static_cast<int>(u * (width - 1));
        int y = static_cast<int>((1.0 - v) * (height - 1));
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return pixels[y * width + x].alpha();
    }

    // sRGB -> Linear conversion fonksiyonu
    inline float srgb_to_linear(uint8_t c) {
        float v = c / 255.0f;
        return (v <= 0.04045f) ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
    }


    bool upload_to_gpu() {
        if (!g_hasOptix) {
            SCENE_LOG_INFO("OptiX disabled: CPU-only texture mode.");
            return false;  // CPU-only mode
        }

        if (is_gpu_uploaded || !m_is_loaded)
            return false;

        // Pixel verilerini uchar4'e dönüştür + sRGB conversion yapılırsa
        std::vector<uchar4> cuda_data(pixels.size());
        for (size_t i = 0; i < pixels.size(); ++i) {
            auto& p = pixels[i];

                cuda_data[i] = make_uchar4(p.r, p.g, p.b, p.a);
          
        }

        // CUDA array oluştur
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        cudaError_t err = cudaMallocArray(&cuda_array, &desc, width, height);
        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("cudaMallocArray failed: " + std::string(cudaGetErrorString(err)));
            return false;
        }

        // Veriyi GPU'ya kopyala
        err = cudaMemcpy2DToArray(cuda_array, 0, 0, cuda_data.data(),
            width * sizeof(uchar4),
            width * sizeof(uchar4), height,
            cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("cudaMemcpy2DToArray failed: " + std::string(cudaGetErrorString(err)));
            return false;
        }

        // Texture descriptor oluştur
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        // Texture object oluştur
        err = cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("cudaCreateTextureObject failed: " + std::string(cudaGetErrorString(err)));
            return false;
        }

        is_gpu_uploaded = true;
        SCENE_LOG_INFO("Texture uploaded to GPU successfully | " +
            std::to_string(width) + "x" + std::to_string(height) +
            (is_srgb ? " | sRGB->Linear converted" : " | Linear (no conversion)"));

        return true;
    }

    void cleanup_gpu() {
        if (tex_obj) {
            cudaDestroyTextureObject(tex_obj);
            tex_obj = 0;
        }
        if (cuda_array) {
            cudaFreeArray(cuda_array);
            cuda_array = nullptr;
        }
    }

    ~Texture() {
        cleanup_gpu();
        // Do NOT clear the cache here - it's a singleton shared across all textures!
        // TextureCache and FileTextureCache are singletons and should persist
    }
    void loadOpacityMap(const std::string& filename) {
        SDL_Surface* surface = IMG_Load(filename.c_str());
        if (!surface) {
            SCENE_LOG_ERROR(std::string("Opacity map load error: ") + filename.c_str() + " - " + IMG_GetError());
            return;
        }

        if (surface->w != width || surface->h != height) {
            SCENE_LOG_ERROR("Opacity map dimensions do not match the main texture.") ;
            SDL_FreeSurface(surface);
            return;
        }

        has_alpha = true;        // Opaklık haritası var artık
        is_gray_scale = true;    // Öncelikle gri varsayalım
        alphas.resize(width * height);

        if (SDL_LockSurface(surface) != 0) {
            SCENE_LOG_ERROR("Failed to lock surface for opacity map: " + filename);
            SDL_FreeSurface(surface);
            return;
        }
        Uint8* pixelData = static_cast<Uint8*>(surface->pixels);
        if (!pixelData) {
            SCENE_LOG_ERROR("Opacity map surface pixels are null: " + filename);
            SDL_UnlockSurface(surface);
            SDL_FreeSurface(surface);
            return;
        }
        SDL_PixelFormat* format = surface->format;

        if (format->BitsPerPixel == 8) {
            // 8-bit grayscale format
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    Uint8 gray = pixelData[y * surface->pitch + x];
                    alphas[y * width + x] = gray;
                }
            }
        }
        else {
            // Diğer formatlar için RGB'den griye dönüştürme
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    Uint32 pixel = *reinterpret_cast<Uint32*>(pixelData + y * surface->pitch + x * format->BytesPerPixel);
                    Uint8 r, g, b;
                    SDL_GetRGB(pixel, format, &r, &g, &b);
                    Uint8 gray = static_cast<Uint8>(0.299f * r + 0.587f * g + 0.114f * b);
                    alphas[y * width + x] = gray;

                    // Eğer renk kanalları eşit değilse grayscale değil
                    if (!(r == g && g == b)) {
                        is_gray_scale = false;
                    }
                }
            }
        }

        SDL_UnlockSurface(surface);
        SDL_FreeSurface(surface);

        SCENE_LOG_INFO(std::string("Opacity map loaded: ") + filename);
    }

    bool is_loaded() const { return m_is_loaded; }
    cudaTextureObject_t get_cuda_texture() const { return tex_obj; }
    bool has_alpha = false;
    bool is_srgb = false;
    bool is_aces = false;
    bool is_gray_scale = true;
    bool m_is_loaded = false;
    bool is_gpu_uploaded = false;
private:
    // ===== decode_raw() OPTIMIZED - SIMD + Paralel =====
    void decode_raw(const aiTexture* tex) {
        auto perf_start = std::chrono::high_resolution_clock::now();

        width = tex->mWidth;
        height = tex->mHeight;
        int pixel_count = width * height;
        pixels.resize(pixel_count);

        SCENE_LOG_INFO("RAW decode start: " + std::to_string(width) + "x" +
            std::to_string(height) + " (" + std::to_string(pixel_count) + " pixels)");

        const aiTexel* data = tex->pcData;
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;

        std::atomic<bool> has_alpha_atomic(false);
        std::atomic<bool> is_gray_atomic(true);

        auto process_chunk = [&](int start, int end) {
            bool local_has_alpha = false;
            bool local_is_gray = true;

            // SIMD işlem için 4 pixel'i blok halinde işle
            int simd_end = start + ((end - start) / 4) * 4;

            for (int i = start; i < simd_end; i += 4) {
                // 4 pixel'i aynı anda işle (data locality daha iyi)
                for (int j = 0; j < 4; ++j) {
                    const aiTexel& t = data[i + j];
                    CompactVec4 px(t.r, t.g, t.b, t.a);
                    pixels[i + j] = px;

                    if (t.a != 255) local_has_alpha = true;
                    if (!px.is_gray()) local_is_gray = false;
                }
            }

            // Kalan pixel'ler
            for (int i = simd_end; i < end; ++i) {
                const aiTexel& t = data[i];
                CompactVec4 px(t.r, t.g, t.b, t.a);
                pixels[i] = px;

                if (t.a != 255) local_has_alpha = true;
                if (!px.is_gray()) local_is_gray = false;
            }

            if (local_has_alpha) has_alpha_atomic.store(true);
            if (!local_is_gray) is_gray_atomic.store(false);
            };

        // Daha büyük chunks (thread startup overhead'i azalt)
        int chunk_size = (((65536) > ((pixel_count + num_threads - 1) / num_threads)) ? (65536) : ((pixel_count + num_threads - 1) / num_threads));
        for (int i = 0; i < pixel_count; i += chunk_size) {
            int start = i;
            int end = std::min(start + chunk_size, pixel_count);
            threads.emplace_back(process_chunk, start, end);
        }

        for (auto& t : threads) t.join();

        has_alpha = has_alpha_atomic.load();
        is_gray_scale = is_gray_atomic.load();
        m_is_loaded = true;

        auto perf_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(perf_end - perf_start);

        SCENE_LOG_INFO("[SUCCESS] RAW texture decoded -> " + std::to_string(width) + "x" +
            std::to_string(height) + (has_alpha ? " | alpha" : " | opaque") +
            (is_gray_scale ? " | grayscale" : " | color") + " | " +
            std::to_string(num_threads) + " threads | " +
            std::to_string(duration.count()) + "ms");
    }

    // ===== decode_compressed() OPTIMIZED - Y-chunking + Lock tuning =====
    void decode_compressed(const aiTexture* tex) {
        auto perf_start = std::chrono::high_resolution_clock::now();

        const unsigned char* buffer = reinterpret_cast<const unsigned char*>(tex->pcData);
        SDL_RWops* rw = SDL_RWFromConstMem(buffer, tex->mWidth);
        if (!rw) {
            SCENE_LOG_ERROR("[DECODE ERROR] Failed to create SDL_RWops for compressed texture");
            return;
        }

        SDL_Surface* surface = IMG_Load_RW(rw, 1);
        if (!surface) {
            SCENE_LOG_ERROR("[DECODE ERROR] Compressed texture load error: " +
                std::string(IMG_GetError()));
            return;
        }

        width = surface->w;
        height = surface->h;
        int pixel_count = width * height;
        pixels.resize(pixel_count);

        SCENE_LOG_INFO("COMPRESSED decode start: " + std::to_string(width) + "x" +
            std::to_string(height) + " (" + std::to_string(pixel_count) + " pixels)");

        if (SDL_LockSurface(surface) != 0) {
            SCENE_LOG_ERROR("[DECODE ERROR] Failed to lock surface");
            SDL_FreeSurface(surface);
            return;
        }

        SDL_Surface* converted_surface = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGBA32, 0);
        SDL_UnlockSurface(surface);
        SDL_FreeSurface(surface);

        if (!converted_surface) {
            SCENE_LOG_ERROR("[DECODE ERROR] Failed to convert surface");
            return;
        }

        surface = converted_surface;
        width = surface->w;
        height = surface->h;

        if (SDL_LockSurface(surface) != 0) {
            SCENE_LOG_ERROR("[DECODE ERROR] Failed to lock converted surface");
            SDL_FreeSurface(surface);
            return;
        }

        SDL_PixelFormat* fmt = surface->format;
        uint8_t* dataSurf = static_cast<uint8_t*>(surface->pixels);
        if (!dataSurf) {
            SCENE_LOG_ERROR("[DECODE ERROR] Converted surface pixels null");
            SDL_UnlockSurface(surface);
            SDL_FreeSurface(surface);
            return;
        }

        int pitch = surface->pitch;
        int bpp = fmt->BytesPerPixel;
        has_alpha = SDL_ISPIXELFORMAT_ALPHA(fmt->format);

        if(bpp != 4) {
             SCENE_LOG_ERROR("[DECODE ERROR] Converted BPP != 4");
             SDL_UnlockSurface(surface);
             SDL_FreeSurface(surface);
             return;
        }

        bool is_gray_local = true;

        try {
            for (int y = 0; y < height; ++y) {
                Uint32* row_ptr = reinterpret_cast<Uint32*>(dataSurf + y * pitch);
                for (int x = 0; x < width; ++x) {
                    Uint32 pixel_val = row_ptr[x];
                    Uint8 r, g, b, a;
                    SDL_GetRGBA(pixel_val, fmt, &r, &g, &b, &a);

                    CompactVec4 px(r, g, b, a);
                    pixels[y * width + x] = px;

                    if (is_gray_local && !px.is_gray()) {
                        is_gray_local = false;
                    }
                }
            }
        } catch(...) {}

        is_gray_scale = is_gray_local;
        m_is_loaded = true;

        SDL_UnlockSurface(surface);
        SDL_FreeSurface(surface);

        auto perf_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(perf_end - perf_start);

        SCENE_LOG_INFO("[SUCCESS] COMPRESSED texture decoded -> " + std::to_string(width) + "x" +
            std::to_string(height) + (has_alpha ? " | alpha" : " | opaque") +
            (is_gray_scale ? " | grayscale" : " | color") + " | " +
            "Single-threaded safe load | " +
            std::to_string(duration.count()) + "ms");
    }

   
   
    std::vector<CompactVec4> pixels;
    int width = 0, height = 0;
   
    std::vector<uint8_t> alphas;  // float yerine 1 byte kullanıyoruz

    TextureType type = TextureType::Unknown;

    cudaArray_t cuda_array = nullptr;
    cudaTextureObject_t tex_obj = 0;
};
