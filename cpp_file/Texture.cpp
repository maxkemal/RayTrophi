// Texture.cpp

#include "Texture.h"
#include <iostream>

float aces(float x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}
Texture::SRGBToLinearLUT::SRGBToLinearLUT() {
    for (int i = 0; i < TABLE_SIZE; ++i) {
        float channel = i / 255.0f;
        table[i] = std::min(
            (channel <= 0.04045f) ?
            channel / 12.92f :
            std::pow((channel + 0.055f) / 1.055f, 2.4f),
            1.0f
        );
    }
}

float Texture::SRGBToLinearLUT::convert(uint8_t value) const {
    return table[value];
}

const Texture::SRGBToLinearLUT Texture::srgbToLinearLUT;

void Texture::processPixel(Uint8 r, Uint8 g, Uint8 b, Uint8 a, int x, int y) {
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;

    Vec3 color(rf, gf, bf);

    // sRGB → Linear dönüşümü sadece Albedo ve Emission için
    if (type == TextureType::Albedo || type == TextureType::Emission) {
        color.x = srgbToLinearLUT.convert(r);
        color.y = srgbToLinearLUT.convert(g);
        color.z = srgbToLinearLUT.convert(b);
    }

    // ACES tonemapping sadece Emission'a uygulanır
    if (type == TextureType::Emission) {
        color.x = aces(color.x);
        color.y = aces(color.y);
        color.z = aces(color.z);
    }

        pixels[y * width + x] = Vec3(color.x, color.y, color.z);
    

    if (has_alpha) {
        alphas[y * width + x] = a / 255.0f;
    }

    if (r != g || r != b) {
        is_gray_scale = false;
    }
}

Texture::Texture(const std::string& filename, TextureType type) : type(type), is_srgb(type == TextureType::Albedo) {
    SDL_Surface* surface = IMG_Load(filename.c_str());
    if (!surface) {
        std::cerr << "Error loading image: " << filename << ", SDL Error: " << IMG_GetError() << std::endl;
        return;
    }

    width = surface->w;
    height = surface->h;
    pixels.resize(width * height);
    has_alpha = SDL_ISPIXELFORMAT_ALPHA(surface->format->format);

    if (has_alpha) {
        alphas.resize(width * height);
    }

    SDL_LockSurface(surface);
    Uint8* pixelData = static_cast<Uint8*>(surface->pixels);
    SDL_PixelFormat* format = surface->format;

    switch (format->BitsPerPixel) {
    case 8:
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Uint8* p = pixelData + (y * surface->pitch) + x;
                SDL_Color color;
                SDL_GetRGB(*p, format, &color.r, &color.g, &color.b);
                processPixel(color.r, color.g, color.b, 255, x, y);
            }
        }
        break;
    case 24:
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Uint8* p = pixelData + (y * surface->pitch) + (x * 3);
                processPixel(p[0], p[1], p[2], 255, x, y);
            }
        }
        break;
    case 32:
        for (int y = 0; y < height; ++y) {
            Uint32* row = reinterpret_cast<Uint32*>(pixelData + (y * surface->pitch));
            for (int x = 0; x < width; ++x) {
                Uint8 r, g, b, a;
                SDL_GetRGBA(row[x], format, &r, &g, &b, &a);
                processPixel(r, g, b, a, x, y);
            }
        }
        break;
    default:
        std::cerr << "Unsupported pixel format: " << static_cast<int>(format->BitsPerPixel) << " bits per pixel." << std::endl;
        SDL_UnlockSurface(surface);
        SDL_FreeSurface(surface);
        return;
    }

    SDL_UnlockSurface(surface);
    SDL_FreeSurface(surface);
    m_is_loaded = true;
}

float Texture::get_alpha(double u, double v) const {
    if (width <= 0 || height <= 0 || alphas.empty()) {
        return 1.0f; // Geçerli bir opaklık haritası yoksa varsayılan opaklık
    }

    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    int x = static_cast<int>(u * (width - 1));
    int y = static_cast<int>((1 - v) * (height - 1));

    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);

    // Eğer gri tonlamalıysa, doğrudan alfa kanalını kullan
    if (is_gray_scale) {
        return alphas[y * width + x];
    }

    return alphas[y * width + x]; // Normal opaklık haritası
}


void Texture::loadOpacityMap(const std::string& filename) {
    SDL_Surface* surface = IMG_Load(filename.c_str());
    if (!surface) {
        std::cerr << "Error loading opacity map: " << filename << ", SDL Error: " << IMG_GetError() << std::endl;
        return;
    }

    // Ensure the opacity map has the same dimensions as the main texture
    if (surface->w != width || surface->h != height) {
        std::cerr << "Opacity map dimensions do not match the main texture." << std::endl;
        SDL_FreeSurface(surface);
        return;
    }

    SDL_LockSurface(surface);
    Uint8* pixelData = static_cast<Uint8*>(surface->pixels);
    SDL_PixelFormat* format = surface->format;

    if (format->BitsPerPixel == 8) {
        // 8-bit grayscale format
        alphas.resize(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Uint8 gray = pixelData[y * surface->pitch + x];
                alphas[y * width + x] = gray / 255.0f;
            }
        }
    }
    else {
        // Other formats
        Uint8 r, g, b;
        alphas.resize(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Uint32 pixel = *reinterpret_cast<Uint32*>(pixelData + y * surface->pitch + x * format->BytesPerPixel);
                SDL_GetRGB(pixel, format, &r, &g, &b);
                Uint8 gray = static_cast<Uint8>(0.299 * r + 0.587 * g + 0.114 * b);
                alphas[y * width + x] = gray / 255.0f;
            }
        }
    }

    SDL_UnlockSurface(surface);
    SDL_FreeSurface(surface);

    std::cout << "Opacity map loaded successfully." << std::endl;
}
Vec3 Texture::get_color(double u, double v) const {
    // Sınır kontrolü
    if (width <= 0 || height <= 0 || pixels.empty()) {
        return Vec3(0, 0, 0);  // Geçersiz texture durumunda siyah döndür
    }

    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    int x = static_cast<int>(u * (width - 1));
    int y = static_cast<int>((1 - v) * (height - 1));  // Flip y-coordinate if necessary

    // Sınırları aşmadığından emin ol
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);

    // Bilineer interpolasyon için komşu pikselleri hesapla
    int x0 = x;
    int x1 = std::min(x + 1, width - 1);
    int y0 = y;
    int y1 = std::min(y + 1, height - 1);

    double tx = u * (width - 1) - x;
    double ty = (1 - v) * (height - 1) - y;

    // Güvenli indeks erişimi için helper fonksiyon
    auto safe_pixel = [this](int y, int x) -> Vec3 {
        size_t index = static_cast<size_t>(y) * width + x;
        if (index < pixels.size()) {
            return pixels[index];
        }
        return Vec3(0, 0, 0);  // Geçersiz indeks durumunda siyah döndür
        };

    Vec3 c00 = safe_pixel(y0, x0);
    Vec3 c10 = safe_pixel(y0, x1);
    Vec3 c01 = safe_pixel(y1, x0);
    Vec3 c11 = safe_pixel(y1, x1);

    // Bilineer interpolasyon
    Vec3 c0 = c00 * (1 - tx) + c10 * tx;
    Vec3 c1 = c01 * (1 - tx) + c11 * tx;
    return c0 * (1 - ty) + c1 * ty;
}
bool Texture::upload_to_gpu() {
    if (uploaded) return true;
    if (!m_is_loaded || pixels.empty()) return false;

    cudaError_t err;
    cudaChannelFormatDesc channelDesc = {};
    size_t pitch;
    void* data_ptr = nullptr;
    size_t pixel_size = 0;
    std::vector<uint8_t> gray_data(width * height);
    std::vector<uchar4> uchar_data(width * height);
    if (is_gray_scale) {
        // R8 formatında tanımla
        channelDesc = cudaCreateChannelDesc<unsigned char>();

        for (int i = 0; i < width * height; ++i) {
            float v = pixels[i].x;  // R=G=B zaten
            gray_data[i] = static_cast<uint8_t>(std::clamp(v * 255.0f, 0.0f, 255.0f));
        }

        pixel_size = sizeof(uint8_t);
        data_ptr = gray_data.data();
    }

    else {
        // RGBA8: renkli texture
       
        for (int i = 0; i < width * height; ++i) {
            float r = pixels[i].x;
            float g = pixels[i].y;
            float b = pixels[i].z;
            float a = has_alpha ? alphas[i] : 1.0f;

            uchar_data[i] = make_uchar4(
                static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(a * 255.0f, 0.0f, 255.0f))
            );
        }

        channelDesc = cudaCreateChannelDesc<uchar4>();
        pixel_size = sizeof(uchar4);
        data_ptr = uchar_data.data();
    }

    // GPU belleğine ayır
    err = cudaMallocArray(&cuda_array, &channelDesc, width, height);
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocArray failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMemcpy2DToArray(
        cuda_array, 0, 0,
        data_ptr,
        width * pixel_size,
        width * pixel_size,
        height,
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy2DToArray failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    //  Texture object oluştur
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuda_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat; //  uchar → float dönüşüm
    texDesc.normalizedCoords = 1;

    err = cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "cudaCreateTextureObject failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    uploaded = true;
    return true;
}
void Texture::cleanup_gpu() {
    if (tex_obj) {
        cudaDestroyTextureObject(tex_obj);
        tex_obj = 0;
    }
    if (cuda_array) {
        cudaFreeArray(cuda_array);
        cuda_array = nullptr;
    }
    
}


Texture::~Texture() {
    // SDL_image'in kullandığı kaynakları temizle
    cleanup_gpu();
    IMG_Quit();
}