#pragma once
#include <vector>
#include <string>
#include <SDL_image.h>
#include "Vec2.h"
#include "Vec3SIMD.h"
#include <array>
#include <cuda_runtime.h>


enum class TextureType {
    Unknown,
    Albedo,
    Normal,
    Roughness,
    Metallic,
    Emission,
    AO,
    Transmission,
    Opacity
};
class Texture {
private:
    class SRGBToLinearLUT {
    private:
        static constexpr int TABLE_SIZE = 256;
        std::array<float, TABLE_SIZE> table;

    public:
        SRGBToLinearLUT();
        float convert(uint8_t value) const;
    };

    static const SRGBToLinearLUT srgbToLinearLUT;

    std::vector<Vec3> pixels;
    std::vector<float> alphas;
    int width;
    int height;
    bool m_is_loaded = false;
    bool is_gray_scale = true;
    TextureType type;
    

    void processPixel(Uint8 r, Uint8 g, Uint8 b, Uint8 a, int x, int y);
    cudaArray_t cuda_array = nullptr;
    cudaTextureObject_t tex_obj = 0;  // GPU'da kullanýlacak obje

public:
    Texture(const std::string& filename, TextureType type);
    Vec3 getColor(const Vec2& uv) const { return get_color(uv.u, uv.v); }
    void loadOpacityMap(const std::string& filename);
    float get_alpha(double u, double v) const;
    Vec3 get_color(double u, double v) const;
    ~Texture();
    bool is_loaded() const { return m_is_loaded; }
    bool has_alpha = false;
    bool is_normal_map = false;
    bool is_srgb = false;
    cudaTextureObject_t get_cuda_texture() const { return tex_obj; }
    bool uploaded = false;
    bool upload_to_gpu();  // Yeni: GPU'ya aktarým fonksiyonu
    void cleanup_gpu();    // Yeni: GPU bellek temizliði
};