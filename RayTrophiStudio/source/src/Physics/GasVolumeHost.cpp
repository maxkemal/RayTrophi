#include "GasVolume.h"

void GasVolume::uploadCudaTextures(void*) {
}

void GasVolume::freeGPUResources() {
    // No CUDA allocations are created by the Player implementation.
    density_texture = temperature_texture = velocity_texture = 0;
    density_array = temperature_array = velocity_array = nullptr;
    gpu_res_x = gpu_res_y = gpu_res_z = 0;
}
