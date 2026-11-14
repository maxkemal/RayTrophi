#pragma warning(disable : 4146) // NanoVDB unsigned hash arithmetic, as in the common TU
#include "VDBVolumeManager.h"
#include "globals.h"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/GridHandle.h>
#include <cuda_runtime.h>

bool VDBVolumeManager::uploadToGPU(int volume_id, bool silent, void* stream_ptr) {
    if (!g_hasCUDA)
        return false;

    // Check if the OptiX backend is actually active.
    // If not active, bypass CUDA allocations to avoid double allocation/redundant copying.
    if (!render_settings.use_optix) {
        // Release any existing CUDA VRAM allocations for this volume to prevent leaks
        freeGPU(volume_id);

        // Mark uploaded status as true because host grid buffer is fully processed & ready for
        // Vulkan RT
        int idx = findVolumeIndex(volume_id);
        if (idx >= 0) {
            volumes[idx].gpu_uploaded = true;
        }
        return true;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    cudaGetLastError(); // Clear sticky error flag without forcing a global stall

    int idx = findVolumeIndex(volume_id);
    if (idx < 0) {
        last_error = "Volume not found: " + std::to_string(volume_id);
        return false;
    }

    VDBVolumeData& vol = volumes[idx];

    // SEAMLESS SYNC: Even if gpu_uploaded is true, we proceed if we have a handle
    // to ensure live simulation frames are copied.

    if (!vol.internal_nano_handle) {
        last_error = "No NanoVDB handle available";
        return false;
    }

    auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_handle);
    size_t required_size = handle->bufferSize();

    if (required_size == 0) {
        last_error = "NanoVDB buffer size is 0";
        return false;
    }

    // Ensure correct CUDA context
    cudaError_t setDeviceErr = cudaSetDevice(0);
    if (setDeviceErr != cudaSuccess) {
        last_error = std::string("cudaSetDevice failed: ") + cudaGetErrorString(setDeviceErr);
        if (!silent)
            SCENE_LOG_ERROR(last_error);
        return false;
    }

    // SMART REALLOC: Allocate GPU memory only if needed or if existing buffer is too small
    cudaError_t err = cudaSuccess;
    if (vol.d_nano_grid == nullptr || vol.gpu_buffer_size < required_size) {
        // CRITICAL: synchronize before freeing memory. Live grid resets can
        // replace the NanoVDB buffer while OptiX still has the old pointer in
        // its launch params / volume table from the previous frame.
        if (stream) {
            cudaStreamSynchronize(stream);
        } else {
            cudaDeviceSynchronize();
        }

        if (vol.d_nano_grid) {
            cudaFree(vol.d_nano_grid);
            vol.d_nano_grid = nullptr;
        }

        err = cudaMalloc(&vol.d_nano_grid, required_size);
        if (err != cudaSuccess) {
            vol.d_nano_grid = nullptr;
            vol.gpu_buffer_size = 0;
            last_error = std::string("CUDA malloc failed: ") + cudaGetErrorString(err);
            if (!silent)
                SCENE_LOG_ERROR(last_error);
            return false;
        }
        vol.gpu_buffer_size = required_size;
    }

    if (!vol.d_nano_grid) {
        last_error = "Destination GPU pointer is NULL after malloc";
        if (!silent)
            SCENE_LOG_ERROR(last_error);
        return false;
    }

    // Copy to GPU (use synchronous copy for safety)
    void* host_ptr = handle->data();
    if (!host_ptr) {
        last_error = "NanoVDB handle.data() is NULL";
        if (!silent)
            SCENE_LOG_ERROR(last_error);
        return false;
    }

    // Sanity checks and logging to help diagnose illegal memory accesses.
    if (required_size == 0 || required_size > (size_t)1024 * 1024 * 1024) { // >1GB suspicious
        last_error = "Suspicious NanoVDB buffer size: " + std::to_string(required_size);
        if (!silent)
            SCENE_LOG_WARN(last_error);
    }

    // NOTE: Use async upload when a stream is provided to avoid stalling the frame.
    err = stream ? cudaMemcpyAsync(vol.d_nano_grid, host_ptr, required_size, cudaMemcpyHostToDevice,
                                   stream)
                 : cudaMemcpy(vol.d_nano_grid, host_ptr, required_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        last_error = std::string("CUDA memcpy failed: ") + cudaGetErrorString(err) +
                     " (Size: " + std::to_string(required_size) +
                     ", Dst: " + std::to_string((unsigned long long)vol.d_nano_grid) +
                     ", Src: " + std::to_string((unsigned long long)host_ptr) + ")";
        if (!silent)
            SCENE_LOG_ERROR(last_error);
        return false;
    }

    vol.gpu_buffer_size = handle->bufferSize();

    // Upload Temperature Grid if available
    if (vol.internal_nano_temperature_handle) {
        auto* temp_handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(
            vol.internal_nano_temperature_handle);
        size_t temp_required_size = temp_handle->bufferSize();

        if (temp_required_size > 0) {
            if (vol.d_nano_temperature == nullptr ||
                vol.gpu_temp_buffer_size < temp_required_size) {
                if (stream) {
                    cudaStreamSynchronize(stream);
                } else {
                    cudaDeviceSynchronize();
                }
                if (vol.d_nano_temperature)
                    cudaFree(vol.d_nano_temperature);
                cudaError_t tempAllocErr = cudaMalloc(&vol.d_nano_temperature, temp_required_size);
                if (tempAllocErr != cudaSuccess) {
                    vol.d_nano_temperature = nullptr;
                    vol.gpu_temp_buffer_size = 0;
                    last_error =
                        std::string("CUDA temp malloc failed: ") + cudaGetErrorString(tempAllocErr);
                    if (!silent)
                        SCENE_LOG_ERROR(last_error);
                    return false;
                }
                vol.gpu_temp_buffer_size = temp_required_size;
            }

            if (vol.d_nano_temperature != nullptr) {
                void* t_host_ptr = temp_handle->data();
                if (t_host_ptr) {
                    if (temp_required_size == 0 ||
                        temp_required_size > (size_t)1024 * 1024 * 1024) {
                        if (!silent)
                            SCENE_LOG_WARN("Suspicious NanoVDB temperature buffer size: " +
                                           std::to_string(temp_required_size));
                    }
                    cudaError_t tempCopyErr =
                        stream ? cudaMemcpyAsync(vol.d_nano_temperature, t_host_ptr,
                                                 temp_required_size, cudaMemcpyHostToDevice, stream)
                               : cudaMemcpy(vol.d_nano_temperature, t_host_ptr, temp_required_size,
                                            cudaMemcpyHostToDevice);
                    if (tempCopyErr != cudaSuccess) {
                        last_error = std::string("CUDA memcpy temp failed: ") +
                                     cudaGetErrorString(tempCopyErr);
                        if (!silent)
                            SCENE_LOG_ERROR(last_error);
                        return false;
                    }
                }
            }
        }
    }

    vol.gpu_uploaded = true;

    std::string size_str;
    if (vol.gpu_buffer_size < 1024 * 1024) {
        size_str = std::to_string(vol.gpu_buffer_size / 1024) + " KB";
    } else {
        size_str = std::to_string(vol.gpu_buffer_size / (1024 * 1024)) + " MB";
    }

    if (!silent) {
        SCENE_LOG_INFO("VDB uploaded to GPU: " + vol.name + " (" + size_str + ")");
    }

    return true;
}

void VDBVolumeManager::freeGPU(int volume_id) {
    if (!g_hasCUDA)
        return;
    int idx = findVolumeIndex(volume_id);
    if (idx < 0)
        return;

    VDBVolumeData& vol = volumes[idx];

    // Check if there's anything to free first
    if (!vol.d_nano_grid && !vol.d_nano_temperature) {
        return;
    }

    // Clear any sticky CUDA errors first
    cudaGetLastError();

    // Check if CUDA is available before sync
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0) {
        // CRITICAL: Sync device before freeing to ensure no async operations use these pointers
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            SCENE_LOG_INFO("[VDBVolumeManager] cudaDeviceSynchronize failed in freeGPU: " +
                           std::string(cudaGetErrorString(err)));
            cudaGetLastError(); // Clear error
        }
    }

    if (vol.d_nano_grid) {
        cudaFree(vol.d_nano_grid);
        vol.d_nano_grid = nullptr;
        vol.gpu_buffer_size = 0;  // Reset size so next upload reallocates
        vol.gpu_uploaded = false; // Mark false only if main grid freed
        // GPU memory freed
    }
    if (vol.d_nano_temperature) {
        cudaFree(vol.d_nano_temperature);
        vol.d_nano_temperature = nullptr;
        vol.gpu_temp_buffer_size = 0; // Reset size
    }
}

void VDBVolumeManager::trimCudaBuffers(VDBVolumeData& vol, size_t new_gpu_size,
                                       size_t new_gpu_temp_size) {
    if (g_hasCUDA && vol.d_nano_grid && vol.gpu_buffer_size < new_gpu_size) {
        cudaFree(vol.d_nano_grid);
        vol.d_nano_grid = nullptr;
    }
    if (g_hasCUDA && vol.d_nano_temperature && vol.gpu_temp_buffer_size < new_gpu_temp_size) {
        cudaFree(vol.d_nano_temperature);
        vol.d_nano_temperature = nullptr;
    }
}
