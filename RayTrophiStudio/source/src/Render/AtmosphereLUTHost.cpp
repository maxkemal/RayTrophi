// Player host implementation. CPU caches feed CPU sampling and Vulkan uploads;
// CUDA texture handles stay zero. Link this module instead of AtmosphereLUTCuda.cpp.
#include "AtmosphereLUT.h"

void AtmosphereLUT::cleanup() {
    initialized = false;
}

void AtmosphereLUT::initialize() {
    cleanup();
    initialized = true;
}

void AtmosphereLUT::uploadCudaTransmittance() {
}

void AtmosphereLUT::uploadCudaMultiScatter() {
}

void AtmosphereLUT::uploadCudaSkyView() {
}
