#pragma once
#include <vector>
#include <string>
#include <SDL_image.h>
#include <cuda_runtime.h>
#include "Vec2.h"
#include "Vec3SIMD.h"
#include <iostream>
#include <cmath>
#include <algorithm>

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

class Texture {
public:
    Texture(const std::string& filename, TextureType type)
        : type(type), is_srgb(type == TextureType::Albedo), is_aces(type == TextureType::Emission) {

        SDL_Surface* surface = IMG_Load(filename.c_str());
        if (!surface) {
            std::cerr << "Failed to load: " << filename << "\nError: " << IMG_GetError() << std::endl;
            return;
        }

        width = surface->w;
        height = surface->h;
        SDL_LockSurface(surface);

        pixels.resize(width * height);
        SDL_PixelFormat* fmt = surface->format;
        uint8_t* data = static_cast<uint8_t*>(surface->pixels);
        int pitch = surface->pitch;
        has_alpha = SDL_ISPIXELFORMAT_ALPHA(fmt->format);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Uint32 pixel;
                memcpy(&pixel, data + y * pitch + x * fmt->BytesPerPixel, fmt->BytesPerPixel);
                Uint8 r, g, b, a;
                SDL_GetRGBA(pixel, fmt, &r, &g, &b, &a);
                CompactVec4 px(r, g, b, a);
                pixels[y * width + x] = px;
                if (!px.is_gray()) is_gray_scale = false;
            }
        }

        SDL_UnlockSurface(surface);
        SDL_FreeSurface(surface);
        m_is_loaded = true;
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

    bool upload_to_gpu() {
        if (uploaded || !m_is_loaded) return false;
        std::vector<uchar4> cuda_data(pixels.size());
        for (size_t i = 0; i < pixels.size(); ++i) {
            auto& p = pixels[i];
            cuda_data[i] = make_uchar4(p.r, p.g, p.b, p.a);
        }

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        cudaError_t err = cudaMallocArray(&cuda_array, &desc, width, height);
        if (err != cudaSuccess) return false;

        err = cudaMemcpy2DToArray(cuda_array, 0, 0, cuda_data.data(), width * sizeof(uchar4),
            width * sizeof(uchar4), height, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return false;

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        err = cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess) return false;

        uploaded = true;
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
        IMG_Quit();
    }
    void loadOpacityMap(const std::string& filename) {
        SDL_Surface* surface = IMG_Load(filename.c_str());
        if (!surface) {
            std::cerr << "Error loading opacity map: " << filename << ", SDL Error: " << IMG_GetError() << std::endl;
            return;
        }

        if (surface->w != width || surface->h != height) {
            std::cerr << "Opacity map dimensions do not match the main texture." << std::endl;
            SDL_FreeSurface(surface);
            return;
        }

        has_alpha = true;        // Opaklýk haritasý var artýk
        is_gray_scale = true;    // Öncelikle gri varsayalým
        alphas.resize(width * height);

        SDL_LockSurface(surface);
        Uint8* pixelData = static_cast<Uint8*>(surface->pixels);
        SDL_PixelFormat* format = surface->format;

        if (format->BitsPerPixel == 8) {
            // 8-bit grayscale format
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    Uint8 gray = pixelData[y * surface->pitch + x];
                    alphas[y * width + x] = gray / 255.0f;
                }
            }
        }
        else {
            // Diðer formatlar için RGB'den griye dönüþtürme
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    Uint32 pixel = *reinterpret_cast<Uint32*>(pixelData + y * surface->pitch + x * format->BytesPerPixel);
                    Uint8 r, g, b;
                    SDL_GetRGB(pixel, format, &r, &g, &b);
                    Uint8 gray = static_cast<Uint8>(0.299 * r + 0.587 * g + 0.114 * b);
                    alphas[y * width + x] = gray / 255.0f;

                    // Eðer renk kanallarý eþit deðilse grayscale deðil
                    if (!(r == g && g == b)) {
                        is_gray_scale = false;
                    }
                }
            }
        }

        SDL_UnlockSurface(surface);
        SDL_FreeSurface(surface);

        std::cout << "Opacity map loaded successfully." << std::endl;
    }

    bool is_loaded() const { return m_is_loaded; }
    cudaTextureObject_t get_cuda_texture() const { return tex_obj; }
    bool has_alpha = false;
    bool is_srgb = false;
    bool is_aces = false;
    bool is_gray_scale = true;
    bool m_is_loaded = false;
    bool uploaded = false;
private:
    std::vector<CompactVec4> pixels;
    int width = 0, height = 0;
   
    std::vector<uint8_t> alphas;  // float yerine 1 byte kullanýyoruz

    TextureType type = TextureType::Unknown;

    cudaArray_t cuda_array = nullptr;
    cudaTextureObject_t tex_obj = 0;
};
