#include "GasVolume.h"
#include "globals.h"
#include <cuda_runtime.h>

void GasVolume::uploadCudaTextures(void* stream_ptr) {
    const bool use_cuda_textures = g_hasCUDA && simulator.isUsingCUDA();
    cudaStream_t stream = use_cuda_textures ? static_cast<cudaStream_t>(stream_ptr) : nullptr;

    const auto& grid = simulator.getGrid();
    const int width = grid.nx;
    const int height = grid.ny;
    const int depth = grid.nz;
    if (use_cuda_textures) {
        // Detect resolution change and free old textures
        if (width != gpu_res_x || height != gpu_res_y || depth != gpu_res_z) {
            freeGPUResources();
            gpu_res_x = width;
            gpu_res_y = height;
            gpu_res_z = depth;
        }

        auto uploadTexture = [&](void* d_sim_ptr, const float* h_ptr, void** d_array,
                                 unsigned long long& tex_obj) {
            if (!*d_array) {
                cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
                cudaExtent extent = make_cudaExtent(width, height, depth);
                cudaError_t err = cudaMalloc3DArray((cudaArray**)d_array, &channelDesc, extent);
                if (err != cudaSuccess) {
                    SCENE_LOG_ERROR("GasVolume::uploadToGPU Error: Failed to allocate 3D array - " +
                                    std::string(cudaGetErrorString(err)));
                    return;
                }
            }

            cudaMemcpy3DParms copyParams = {0};
            if (d_sim_ptr) {
                copyParams.srcPtr =
                    make_cudaPitchedPtr(d_sim_ptr, width * sizeof(float), width, height);
                copyParams.kind = cudaMemcpyDeviceToDevice;
            } else {
                copyParams.srcPtr =
                    make_cudaPitchedPtr((void*)h_ptr, width * sizeof(float), width, height);
                copyParams.kind = cudaMemcpyHostToDevice;
            }

            copyParams.dstArray = (cudaArray*)*d_array;
            copyParams.extent = make_cudaExtent(width, height, depth);

            cudaError_t copyErr =
                stream ? cudaMemcpy3DAsync(&copyParams, stream) : cudaMemcpy3D(&copyParams);
            if (copyErr != cudaSuccess) {
                SCENE_LOG_ERROR("GasVolume::uploadToGPU Error: Failed to copy data - " +
                                std::string(cudaGetErrorString(copyErr)));
                return;
            }

            if (tex_obj == 0) {
                cudaResourceDesc resDesc;
                memset(&resDesc, 0, sizeof(resDesc));
                resDesc.resType = cudaResourceTypeArray;
                resDesc.res.array.array = (cudaArray*)*d_array;

                cudaTextureDesc texDesc;
                memset(&texDesc, 0, sizeof(texDesc));
                texDesc.addressMode[0] = cudaAddressModeClamp;
                texDesc.addressMode[1] = cudaAddressModeClamp;
                texDesc.addressMode[2] = cudaAddressModeClamp;
                texDesc.filterMode = cudaFilterModeLinear;
                texDesc.readMode = cudaReadModeElementType;
                texDesc.normalizedCoords = 1;

                cudaError_t texErr = cudaCreateTextureObject((cudaTextureObject_t*)&tex_obj,
                                                             &resDesc, &texDesc, nullptr);
                if (texErr != cudaSuccess) {
                    SCENE_LOG_ERROR(
                        "GasVolume::uploadToGPU Error: Failed to create texture object - " +
                        std::string(cudaGetErrorString(texErr)));
                }
            }
        };

        uploadTexture(simulator.getGPUDensityPtr(), grid.density.data(), &density_array,
                      density_texture);
        if (shader && (shader->emission.mode == VolumeEmissionMode::Blackbody ||
                       shader->emission.mode == VolumeEmissionMode::ChannelDriven ||
                       shader->emission.color_ramp.enabled)) {
            uploadTexture(simulator.getGPUTemperaturePtr(), grid.temperature.data(),
                          &temperature_array, temperature_texture);
        }

        if (stream) {
            cudaStreamSynchronize(stream);
        }
    } else if (density_texture != 0 || density_array != nullptr || temperature_texture != 0 ||
               temperature_array != nullptr) {
        freeGPUResources();
    }
}

void GasVolume::freeGPUResources() {
    // Only do CUDA operations if we actually have resources to free
    bool has_resources =
        (density_texture != 0 || density_array != nullptr || temperature_texture != 0 ||
         temperature_array != nullptr || velocity_texture != 0 || velocity_array != nullptr);

    if (!has_resources) {
        // No GPU resources, just reset tracking
        gpu_res_x = gpu_res_y = gpu_res_z = 0;
        return;
    }
    // If CUDA isn't available, reset tracking and return
    if (!g_hasCUDA) {
        density_texture = temperature_texture = velocity_texture = 0;
        density_array = temperature_array = velocity_array = nullptr;
        gpu_res_x = gpu_res_y = gpu_res_z = 0;
        return;
    }

    // Clear any sticky errors first
    cudaGetLastError();

    // Check if CUDA is usable before sync
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        // CUDA not available, just reset tracking
        density_texture = temperature_texture = velocity_texture = 0;
        density_array = temperature_array = velocity_array = nullptr;
        gpu_res_x = gpu_res_y = gpu_res_z = 0;
        return;
    }

    // Sync with timeout protection - use non-blocking query first
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        SCENE_LOG_INFO("[GasVolume] cudaDeviceSynchronize failed in freeGPUResources: " +
                       std::string(cudaGetErrorString(err)));
        cudaGetLastError(); // Clear error
    }

    auto freeResources = [&](unsigned long long& tex_obj, void** d_array) {
        if (tex_obj != 0) {
            cudaDestroyTextureObject((cudaTextureObject_t)tex_obj);
            tex_obj = 0;
        }
        if (*d_array) {
            cudaFreeArray((cudaArray*)*d_array);
            *d_array = nullptr;
        }
    };

    freeResources(density_texture, &density_array);
    freeResources(temperature_texture, &temperature_array);
    freeResources(velocity_texture, &velocity_array);

    // Reset resolution tracking
    gpu_res_x = gpu_res_y = gpu_res_z = 0;
}
