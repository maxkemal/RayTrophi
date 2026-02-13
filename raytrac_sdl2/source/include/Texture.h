/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Texture.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <SDL_image.h>
#include "stb_image.h"
#include "tinyexr.h"
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
#include <cstring>  // std::memcpy for fast pixel copy

enum class TextureType {
    Unknown, Albedo, Normal, Roughness, Metallic, Emission, AO, Transmission, Opacity
};

// ===== sRGB → Linear Look-up Table (LUT) for FAST conversion =====
// 256 elemanlık statik tablo - bir kez hesaplanır, sonra O(1) lookup
// pow() çağrısı yerine tablo erişimi ~10x daha hızlı
class SRGBToLinearLUT {
public:
    static const SRGBToLinearLUT& instance() {
        static SRGBToLinearLUT lut;
        return lut;
    }

    // sRGB byte (0-255) → Linear byte (0-255)
    uint8_t operator[](uint8_t srgb) const {
        return table[srgb];
    }

private:
    uint8_t table[256];

    SRGBToLinearLUT() {
        for (int i = 0; i < 256; ++i) {
            float f = i / 255.0f;
            float linear = (f <= 0.04045f) ? (f / 12.92f) : powf((f + 0.055f) / 1.055f, 2.4f);
            table[i] = static_cast<uint8_t>(fminf(linear * 255.0f, 255.0f));
        }
    }
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

// ===== Hızlı pixel kopyalama helper =====
// SDL_PIXELFORMAT_RGBA32: Little-endian'da bellek düzeni ABGR olabilir!
// Bu yüzden doğrudan memcpy yerine Uint32 → RGBA dönüşümü yapıyoruz
// Hala SDL_GetRGBA kadar yavaş değil çünkü format lookup'ı atlanıyor
inline void fast_copy_rgba32_pixels(
    const uint8_t* src_data,
    int src_pitch,
    std::vector<CompactVec4>& dest_pixels,
    int width,
    int height,
    bool& out_has_alpha,
    bool& out_is_grayscale
) {
    dest_pixels.resize(width * height);
    bool has_alpha_local = false;
    bool is_gray_local = true;

    // SDL_PIXELFORMAT_RGBA32 tanımı:
    // "RGBA" order when read as bytes: R at lowest address
    // Bu demek oluyor ki: memory[0]=R, memory[1]=G, memory[2]=B, memory[3]=A
    // Yani CompactVec4 ile AYNI sırada! Ama endianness kontrol etmeliyiz.

    // SDL'nin RGBA32 tanımını kullan - bu her platformda doğru
    // RGBA32 = SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_RGBA, ...)
    // Little endian'da byte sırası: R G B A (düşük adresten yükseğe)

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = src_data + y * src_pitch;
        CompactVec4* dest_row = dest_pixels.data() + y * width;

        for (int x = 0; x < width; ++x) {
            // SDL_PIXELFORMAT_RGBA32 için byte erişimi:
            // Her pixel 4 byte: [R][G][B][A] sırasıyla (SDL tanımına göre)
            const uint8_t* pixel = row + x * 4;

            // RGBA32 byte order SDL tarafından garanti ediliyor:
            // index 0 = R, index 1 = G, index 2 = B, index 3 = A
            dest_row[x].r = pixel[0];
            dest_row[x].g = pixel[1];
            dest_row[x].b = pixel[2];
            dest_row[x].a = pixel[3];

            if (pixel[3] != 255) has_alpha_local = true;
            if (is_gray_local && (pixel[0] != pixel[1] || pixel[0] != pixel[2])) {
                is_gray_local = false;
            }
        }
    }

    out_has_alpha = has_alpha_local;
    out_is_grayscale = is_gray_local;
}

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
        is_hdr = false;
        this->name = name; // Store name
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
        is_hdr = false;
        this->name = filename; // Store filename


        // Detect HDR/EXR formats
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // Handle EXR files with TinyEXR
        if (ext == "exr") {
            float* rgba = nullptr;
            int w, h;
            const char* err = nullptr;

            int ret = LoadEXR(&rgba, &w, &h, filename.c_str(), &err);
            if (ret == TINYEXR_SUCCESS && rgba) {
                width = w;
                height = h;
                is_hdr = true;
                float_pixels.resize(width * height);

                // Copy RGBA float data to float4 vector
                for (int i = 0; i < width * height; ++i) {
                    float_pixels[i] = make_float4(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], rgba[i * 4 + 3]);
                }

                free(rgba);
                m_is_loaded = true;

                SCENE_LOG_INFO("[EXR LOAD] Loaded EXR texture: " + filename + " | " +
                    std::to_string(width) + "x" + std::to_string(height));
                return;
            }
            else {
                std::string errMsg = err ? err : "Unknown error";
                SCENE_LOG_ERROR("Failed to load EXR texture: " + filename + " | Error: " + errMsg);
                if (err) FreeEXRErrorMessage(err);
                // Fall through to try other loaders
            }
        }

        // Handle HDR files with stb_image
        if (ext == "hdr") {
            int w, h, c;
            float* data = stbi_loadf(filename.c_str(), &w, &h, &c, 4); // Force 4 channels
            if (data) {
                width = w;
                height = h;
                is_hdr = true;
                float_pixels.resize(width * height);

                // Copy to float4 vector
                for (int i = 0; i < width * height; ++i) {
                    float_pixels[i] = make_float4(data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]);
                }

                stbi_image_free(data);
                m_is_loaded = true;

                SCENE_LOG_INFO("[HDR LOAD] Loaded HDR texture: " + filename + " | " +
                    std::to_string(width) + "x" + std::to_string(height));
                return;
            }
            else {
                SCENE_LOG_ERROR("Failed to load HDR texture: " + filename + " | STB Error: " + stbi_failure_reason());
                // Fall through to try SDL_image
            }
        }

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
                if (bpp != 4) {
                    SCENE_LOG_ERROR("Converted surface (cache hit) BPP != 4");
                    SDL_UnlockSurface(surface);
                    SDL_FreeSurface(surface);
                    return;
                }

                // Hızlı pixel kopyalama kullan (SDL_GetRGBA'dan ~5x hızlı)
                bool alpha_detected = false;
                bool gray_detected = true;
                fast_copy_rgba32_pixels(data, pitch, pixels, width, height, alpha_detected, gray_detected);
                has_alpha = alpha_detected;
                is_gray_scale = gray_detected;

                SDL_UnlockSurface(surface);
                SDL_FreeSurface(surface);
                m_is_loaded = true;

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                // SCENE_LOG_INFO("[FILE CACHE HIT] " + filename +
                //     " | " + std::to_string(width) + "x" + std::to_string(height) +
                //     " | Fast memcpy load" +
                //     " | " + std::to_string(duration.count()) + "ms");
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

        // Hızlı pixel kopyalama kullan (SDL_GetRGBA'dan ~5x hızlı)
        bool alpha_detected = false;
        bool gray_detected = true;
        fast_copy_rgba32_pixels(data, pitch, pixels, width, height, alpha_detected, gray_detected);
        has_alpha = alpha_detected;
        is_gray_scale = gray_detected;

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

        /* SCENE_LOG_INFO("[FILE LOAD SUCCESS] '" + filename + "' | " + std::to_string(width) + "x" +
            std::to_string(height) + (has_alpha ? " | alpha" : " | opaque") +
            (is_gray_scale ? " | grayscale" : " | color") +
            " | Fast memcpy load" +
            " | Cache size now: " + std::to_string(FileTextureCache::instance().size()) +
            " | Total time: " + std::to_string(duration.count()) + "ms"); */
    }

    // ===== Constructor Procedural/Memory Only (No Disk Load) =====
    Texture(const std::string& name, int w, int h, TextureType type)
        : name(name), width(w), height(h), type(type),
        is_srgb(type == TextureType::Albedo), is_aces(type == TextureType::Emission) {

        pixels.resize(width * height);
        m_is_loaded = true;
        is_gpu_uploaded = false;
        is_hdr = false;
        has_alpha = true;
        is_gray_scale = false;
    }

    // ===== Constructor Memory Buffer - Embedded texture'lar için dosya yazmadan yükleme =====
    // Bu constructor proje dosyasından embedded texture binary'si okunduğunda kullanılır
    // Disk I/O yapmadan doğrudan bellekten yükler - daha hızlı ve temp dosya gerektirmez
    Texture(const std::vector<char>& buffer, TextureType type, const std::string& textureName = "")
        : type(type), is_srgb(type == TextureType::Albedo), is_aces(type == TextureType::Emission) {
        
        m_is_loaded = false;
        is_gpu_uploaded = false;
        is_hdr = false;
        name = textureName;
        
        if (buffer.empty()) {
            SCENE_LOG_WARN("[MEMORY LOAD] Empty buffer for texture: " + textureName);
            return;
        }
        
        auto perf_start = std::chrono::high_resolution_clock::now();
        
        // SDL_image kullanarak bellekten yükle
        SDL_RWops* rw = SDL_RWFromConstMem(buffer.data(), static_cast<int>(buffer.size()));
        if (!rw) {
            SCENE_LOG_ERROR("[MEMORY LOAD] Failed SDL_RWFromConstMem for: " + textureName);
            return;
        }
        
        SDL_Surface* surface = IMG_Load_RW(rw, 1); // 1 = auto-free RWops
        if (!surface) {
            SCENE_LOG_ERROR("[MEMORY LOAD] IMG_Load_RW failed for: " + textureName + " | " + std::string(IMG_GetError()));
            return;
        }
        
        width = surface->w;
        height = surface->h;
        pixels.resize(width * height);
        
        // RGBA32'ye dönüştür
        SDL_Surface* converted = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGBA32, 0);
        SDL_FreeSurface(surface);
        
        if (!converted) {
            SCENE_LOG_ERROR("[MEMORY LOAD] Convert failed for: " + textureName);
            return;
        }
        
        surface = converted;
        width = surface->w;
        height = surface->h;
        
        if (SDL_LockSurface(surface) != 0) {
            SDL_FreeSurface(surface);
            return;
        }
        
        if (surface->format->BytesPerPixel != 4) {
            SDL_UnlockSurface(surface);
            SDL_FreeSurface(surface);
            SCENE_LOG_ERROR("[MEMORY LOAD] BPP != 4 for: " + textureName);
            return;
        }
        
        // Hızlı pixel kopyalama
        bool alpha_detected = false;
        bool gray_detected = true;
        fast_copy_rgba32_pixels(
            static_cast<uint8_t*>(surface->pixels),
            surface->pitch, pixels, width, height, alpha_detected, gray_detected
        );
        has_alpha = alpha_detected;
        is_gray_scale = gray_detected;
        m_is_loaded = true;
        
        SDL_UnlockSurface(surface);
        SDL_FreeSurface(surface);
        
        auto perf_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(perf_end - perf_start);
        
        // SCENE_LOG_INFO("[MEMORY LOAD] Texture loaded from buffer: " + textureName + 
        //                " | " + std::to_string(width) + "x" + std::to_string(height) +
        //                " | " + std::to_string(duration.count()) + "ms");
    }

    // ===== Constructor Raw Data (Standard Vectors) =====
    Texture(int w, int h, int channels, const std::vector<unsigned char>& data, TextureType type, const std::string& textureName = "")
        : type(type), is_srgb(type == TextureType::Albedo), is_aces(type == TextureType::Emission) {
        
        width = w;
        height = h;
        name = textureName;
        m_is_loaded = true;
        is_gpu_uploaded = false;
        
        pixels.resize(width * height);
        
        if (channels == 4) {
             #pragma omp parallel for
             for(int i=0; i<width*height; ++i) {
                 pixels[i] = CompactVec4(data[i*4+0], data[i*4+1], data[i*4+2], data[i*4+3]);
                 if(data[i*4+3] < 255) has_alpha = true;
             }
        } else if (channels == 3) {
             #pragma omp parallel for
             for(int i=0; i<width*height; ++i) {
                 pixels[i] = CompactVec4(data[i*3+0], data[i*3+1], data[i*3+2], 255);
             }
        } else if (channels == 1) {
             #pragma omp parallel for
             for(int i=0; i<width*height; ++i) {
                 uint8_t v = data[i];
                 pixels[i] = CompactVec4(v, v, v, 255);
             }
             is_gray_scale = true;
        }
    }
    
    Vec3 get_color(float u, float v) const {
        if (!m_is_loaded) return Vec3(0);
        // Wrap coordinates for tiling support (Repeat mode)
        u = u - floorf(u);
        v = v - floorf(v);
        int x = static_cast<int>(u * (width - 1));
        int y = static_cast<int>((1.0f - v) * (height - 1));
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);

        if (is_hdr && !float_pixels.empty()) {
            float4 p = float_pixels[y * width + x];
            return Vec3(p.x, p.y, p.z);
        }

        if (pixels.empty()) return Vec3(0);
        return pixels[y * width + x].to_linear_rgb(is_srgb, is_aces);
    }

    float get_alpha(float u, float v) const {
        if (!m_is_loaded || pixels.empty()) return 1.0f;
        // Wrap coordinates
        u = u - floorf(u);
        v = v - floorf(v);
        int x = static_cast<int>(u * (width - 1));
        int y = static_cast<int>((1.0f - v) * (height - 1));
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return pixels[y * width + x].alpha();
    }

    // sRGB -> Linear conversion fonksiyonu
    inline float srgb_to_linear(uint8_t c) {
        float v = c / 255.0f;
        return (v <= 0.04045f) ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
    }

    // Stamp brush sampling (Luminance)
    float sampleIntensity(float u, float v) const {
        if (!m_is_loaded) return 0.0f;
        Vec3 color = get_color(u, v);
        return 0.299f * color.x + 0.587f * color.y + 0.114f * color.z; 
    }


    bool upload_to_gpu() {
        if (!g_hasOptix) {
            SCENE_LOG_INFO("OptiX disabled: CPU-only texture mode.");
            return false;  // CPU-only mode
        }

        if (is_gpu_uploaded || !m_is_loaded)
            return false;

        // Texture türüne göre sRGB dönüşümü gerekip gerekmediğini belirle
        // SADECE Albedo texture'ları sRGB formatında saklanır → Linear'a dönüştürülmeli
        // Emission: Intensity/HDR değerleri içerir, sRGB dönüşümü karartır → dönüşüm yok
        // Normal, Roughness, Metallic, AO, Transmission, Opacity → Linear data, dönüşüm yok
        bool needs_srgb_conversion = (type == TextureType::Albedo);
        cudaError_t err;

        if (is_hdr) {
            // Float4 Texture Upload
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
            err = cudaMallocArray(&cuda_array, &desc, width, height);
            if (err != cudaSuccess) {
                SCENE_LOG_ERROR("cudaMallocArray (float) failed for " + std::to_string(width) + "x" + std::to_string(height) + ": " + std::string(cudaGetErrorString(err)));
                return false;
            }
            err = cudaMemcpy2DToArray(cuda_array, 0, 0, float_pixels.data(),
                width * sizeof(float4),
                width * sizeof(float4), height,
                cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                SCENE_LOG_ERROR("cudaMemcpy2DToArray (float) failed: " + std::string(cudaGetErrorString(err)));
                return false;
            }
        }
        else {
            // Pixel verilerini uchar4'e dönüştür
            std::vector<uchar4> cuda_data(pixels.size());

            if (needs_srgb_conversion) {
                // sRGB → Linear dönüşümü LUT ile HIZLI yap
                const auto& lut = SRGBToLinearLUT::instance();
                for (size_t i = 0; i < pixels.size(); ++i) {
                    auto& p = pixels[i];
                    cuda_data[i] = make_uchar4(
                        lut[p.r],
                        lut[p.g],
                        lut[p.b],
                        p.a  // Alpha kanalı linear kalır
                    );
                }
            }
            else {
                // Linear data texture'ları - dönüşüm yok
                for (size_t i = 0; i < pixels.size(); ++i) {
                    auto& p = pixels[i];
                    cuda_data[i] = make_uchar4(p.r, p.g, p.b, p.a);
                }
            }

            // CUDA array oluştur
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
            err = cudaMallocArray(&cuda_array, &desc, width, height);
            if (err != cudaSuccess) {
                SCENE_LOG_ERROR("cudaMallocArray failed for " + std::to_string(width) + "x" + std::to_string(height) + ": " + std::string(cudaGetErrorString(err)));
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
        }

        // Texture descriptor oluştur
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = is_hdr ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;
        // NOT: cudaTextureDesc'te sRGB flag yok - dönüşümü CPU'da yaptık

        // Texture object oluştur
        err = cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("cudaCreateTextureObject failed: " + std::string(cudaGetErrorString(err)));
            return false;
        }

        is_gpu_uploaded = true;

        // Texture türünü string'e çevir
        const char* type_str = "Unknown";
        switch (type) {
        case TextureType::Albedo: type_str = "Albedo"; break;
        case TextureType::Normal: type_str = "Normal"; break;
        case TextureType::Roughness: type_str = "Roughness"; break;
        case TextureType::Metallic: type_str = "Metallic"; break;
        case TextureType::Emission: type_str = "Emission"; break;
        case TextureType::AO: type_str = "AO"; break;
        case TextureType::Transmission: type_str = "Transmission"; break;
        case TextureType::Opacity: type_str = "Opacity"; break;
        default: break;
        }

        // SCENE_LOG_INFO("Texture uploaded to GPU | " +
        //     std::to_string(width) + "x" + std::to_string(height) +
        //     " | Type: " + std::string(type_str) +
        //     (needs_srgb_conversion ? " | sRGB->Linear converted" : " | Linear (no conversion)"));

        return true;
    }

    // Efficiently update GPU data from current CPU pixels
    void updateGPU() {
        if (!is_gpu_uploaded || !g_hasOptix || !cuda_array) return;

        cudaError_t err = cudaSuccess;
        if (is_hdr) {
             err = cudaMemcpy2DToArray(cuda_array, 0, 0, float_pixels.data(),
                width * sizeof(float4),
                width * sizeof(float4), height,
                cudaMemcpyHostToDevice);
        } else {
             std::vector<uchar4> cuda_data(pixels.size());
             // No sRGB conversion for update usually, assuming we write directly linear values or consistent with initial upload
             // But for safety, replicate logic or assume SplatMap (RGBA8 unorm) which is Linear-ish data for Mask
             bool needs_srgb_conversion = (type == TextureType::Albedo);
             
             if (needs_srgb_conversion) {
                 const auto& lut = SRGBToLinearLUT::instance();
                 for (size_t i = 0; i < pixels.size(); ++i) {
                     cuda_data[i] = make_uchar4(lut[pixels[i].r], lut[pixels[i].g], lut[pixels[i].b], pixels[i].a);
                 }
             } else {
                 for (size_t i = 0; i < pixels.size(); ++i) {
                     cuda_data[i] = make_uchar4(pixels[i].r, pixels[i].g, pixels[i].b, pixels[i].a);
                 }
             }
             
             err = cudaMemcpy2DToArray(cuda_array, 0, 0, cuda_data.data(),
                width * sizeof(uchar4),
                width * sizeof(uchar4), height,
                cudaMemcpyHostToDevice);
        }

        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("Texture update failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    // Upload a specific region to GPU (useful for partial splatmap updates)
    bool upload_region_to_gpu(int x, int y, int w, int h) {
        if (!is_gpu_uploaded || !g_hasOptix || !cuda_array) return false;

        // Clamp to texture bounds
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, width - x);
        h = std::min(h, height - y);

        if (w <= 0 || h <= 0) return false;

        cudaError_t err = cudaSuccess;
        if (is_hdr) {
            std::vector<float4> region_data(w * h);
            for (int j = 0; j < h; ++j) {
                std::memcpy(&region_data[j * w], &float_pixels[(y + j) * width + x], w * sizeof(float4));
            }
            err = cudaMemcpy2DToArray(cuda_array, x * sizeof(float4), y, region_data.data(),
                w * sizeof(float4),
                w * sizeof(float4), h,
                cudaMemcpyHostToDevice);
        } else {
            bool needs_srgb_conversion = (type == TextureType::Albedo);
            std::vector<uchar4> region_data(w * h);
            
            if (needs_srgb_conversion) {
                const auto& lut = SRGBToLinearLUT::instance();
                for (int j = 0; j < h; ++j) {
                    for (int i = 0; i < w; ++i) {
                        auto& p = pixels[(y + j) * width + (x + i)];
                        region_data[j * w + i] = make_uchar4(lut[p.r], lut[p.g], lut[p.b], p.a);
                    }
                }
            } else {
                for (int j = 0; j < h; ++j) {
                    std::memcpy(&region_data[j * w], &pixels[(y + j) * width + x], w * sizeof(uchar4));
                }
            }
            
            err = cudaMemcpy2DToArray(cuda_array, x * sizeof(uchar4), y, region_data.data(),
                w * sizeof(uchar4),
                w * sizeof(uchar4), h,
                cudaMemcpyHostToDevice);
        }

        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("Texture region update failed: " + std::string(cudaGetErrorString(err)));
            return false;
        }
        return true;
    }

    cudaTextureObject_t getTextureObject() const { return tex_obj; }
    bool isUploaded() const { return is_gpu_uploaded; }

    void cleanup_gpu() {
        if (tex_obj) {
            cudaDestroyTextureObject(tex_obj);
            tex_obj = 0;
        }
        if (cuda_array) {
            cudaFreeArray(cuda_array);
            cuda_array = nullptr;
        }
        is_gpu_uploaded = false;
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
            SCENE_LOG_ERROR("Opacity map dimensions do not match the main texture.");
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
    int width = 0, height = 0;
    std::vector<CompactVec4> pixels;
    std::vector<float4> float_pixels; // For HDR
    bool is_hdr = false;
    std::string name; // Texture name/path
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

        if (bpp != 4) {
            SCENE_LOG_ERROR("[DECODE ERROR] Converted BPP != 4");
            SDL_UnlockSurface(surface);
            SDL_FreeSurface(surface);
            return;
        }

        // Hızlı pixel kopyalama kullan (SDL_GetRGBA'dan ~5x hızlı)
        bool alpha_detected = false;
        bool gray_detected = true;
        fast_copy_rgba32_pixels(dataSurf, pitch, pixels, width, height, alpha_detected, gray_detected);
        has_alpha = alpha_detected;
        is_gray_scale = gray_detected;
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


    std::vector<uint8_t> alphas;  // float yerine 1 byte kullanıyoruz

    TextureType type = TextureType::Unknown;

    cudaArray_t cuda_array = nullptr;
    cudaTextureObject_t tex_obj = 0;
};

