#pragma once
#include <vector>
#include <string>
#include <SDL_image.h>
#include "Vec2.h"
#include "Vec3SIMD.h"
#include <array>
enum class TextureType {
    Unknown,
    Albedo,
    Normal,
    Roughness,
    Metallic,
    Emission,
    AO,
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
};