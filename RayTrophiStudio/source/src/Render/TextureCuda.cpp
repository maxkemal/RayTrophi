// CUDA texture residency for Texture (OptiX path).
//
// OFFLINE BUILD ONLY. These four members used to be inline in Texture.h, which
// made <cuda_runtime.h> a compile dependency of every translation unit that
// sees a texture. Texture.h keeps the backend-neutral guards (Vulkan dirty
// marking, g_hasOptix / is_gpu_uploaded early-outs) and calls in here only for
// the actual CUDA work. A build without CUDA (RayTrophi Player) must not
// compile this file; it links no-op definitions of the same four members.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md

#include "Texture.h"

#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <vector>

bool Texture::cudaUploadFull() {
    // Only albedo is stored sRGB and needs converting to linear. Emission holds
    // intensity/HDR values (converting would darken it); normal, roughness,
    // metallic, AO, transmission and opacity are linear data.
    const bool needs_srgb_conversion = (type == TextureType::Albedo);
    cudaError_t err;

    if (is_hdr) {
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
        std::vector<uchar4> cuda_data(pixels.size());

        if (needs_srgb_conversion) {
            const auto& lut = SRGBToLinearLUT::instance();
            for (size_t i = 0; i < pixels.size(); ++i) {
                auto& p = pixels[i];
                // Alpha stays linear.
                cuda_data[i] = make_uchar4(lut[p.r], lut[p.g], lut[p.b], p.a);
            }
        }
        else {
            for (size_t i = 0; i < pixels.size(); ++i) {
                auto& p = pixels[i];
                cuda_data[i] = make_uchar4(p.r, p.g, p.b, p.a);
            }
        }

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        err = cudaMallocArray(&cuda_array, &desc, width, height);
        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("cudaMallocArray failed for " + std::to_string(width) + "x" + std::to_string(height) + ": " + std::string(cudaGetErrorString(err)));
            return false;
        }

        err = cudaMemcpy2DToArray(cuda_array, 0, 0, cuda_data.data(),
            width * sizeof(uchar4),
            width * sizeof(uchar4), height,
            cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("cudaMemcpy2DToArray failed: " + std::string(cudaGetErrorString(err)));
            return false;
        }
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuda_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = is_hdr ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    // cudaTextureDesc has no sRGB flag: the conversion was done on the CPU above.

    err = cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        SCENE_LOG_ERROR("cudaCreateTextureObject failed: " + std::string(cudaGetErrorString(err)));
        return false;
    }

    is_gpu_uploaded = true;
    return true;
}

void Texture::cudaUpdateFull() {
    cudaError_t err = cudaSuccess;
    if (is_hdr) {
        err = cudaMemcpy2DToArray(cuda_array, 0, 0, float_pixels.data(),
            width * sizeof(float4),
            width * sizeof(float4), height,
            cudaMemcpyHostToDevice);
    }
    else {
        std::vector<uchar4> cuda_data(pixels.size());
        // Same conversion rule as the initial upload.
        const bool needs_srgb_conversion = (type == TextureType::Albedo);

        if (needs_srgb_conversion) {
            const auto& lut = SRGBToLinearLUT::instance();
            for (size_t i = 0; i < pixels.size(); ++i) {
                cuda_data[i] = make_uchar4(lut[pixels[i].r], lut[pixels[i].g], lut[pixels[i].b], pixels[i].a);
            }
        }
        else {
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

// x/y/w/h arrive already clamped to the texture bounds by upload_region_to_gpu.
bool Texture::cudaUpdateRegion(int x, int y, int w, int h) {
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
    }
    else {
        const bool needs_srgb_conversion = (type == TextureType::Albedo);
        std::vector<uchar4> region_data(w * h);

        if (needs_srgb_conversion) {
            const auto& lut = SRGBToLinearLUT::instance();
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    auto& p = pixels[(y + j) * width + (x + i)];
                    region_data[j * w + i] = make_uchar4(lut[p.r], lut[p.g], lut[p.b], p.a);
                }
            }
        }
        else {
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

void Texture::cudaRelease(bool synchronize) {
    if (tex_obj) {
        // Notify SceneTextureManager FIRST so any cached optixTextureId entries
        // pointing at this handle are cleared before the CUDA object is destroyed.
        // Otherwise concurrent material resolves can fetch a stale handle that
        // CUDA may have just recycled to a different texture, which shows up as
        // wrong textures wrapping wrong meshes after a delete/replace.
        notifyOptixTextureDestroyed(static_cast<int64_t>(tex_obj));

        // Texture slot deletion can race with an in-flight OptiX launch.
        // Wait for outstanding CUDA work before freeing the texture object.
        if (synchronize && (g_hasOptix || rendering_in_progress.load(std::memory_order_acquire))) {
            cudaDeviceSynchronize();
        }

        cudaDestroyTextureObject(tex_obj);
        tex_obj = 0;
    }
    if (cuda_array) {
        cudaFreeArray(cuda_array);
        cuda_array = nullptr;
    }
}
