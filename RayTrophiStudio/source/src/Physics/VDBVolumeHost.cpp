#include "VDBVolumeManager.h"

bool VDBVolumeManager::uploadToGPU(int volume_id, bool, void*) {
    const int idx = findVolumeIndex(volume_id);
    if (idx < 0) {
        last_error = "Volume not found: " + std::to_string(volume_id);
        return false;
    }
    // Vulkan reads the common host NanoVDB representation directly.
    volumes[idx].gpu_uploaded = true;
    return true;
}

void VDBVolumeManager::freeGPU(int volume_id) {
    const int idx = findVolumeIndex(volume_id);
    if (idx < 0) {
        return;
    }
    auto& vol = volumes[idx];
    vol.d_nano_grid = nullptr;
    vol.d_nano_temperature = nullptr;
    vol.gpu_uploaded = false;
    vol.gpu_buffer_size = 0;
    vol.gpu_temp_buffer_size = 0;
}

void VDBVolumeManager::trimCudaBuffers(VDBVolumeData&, size_t, size_t) {
}
