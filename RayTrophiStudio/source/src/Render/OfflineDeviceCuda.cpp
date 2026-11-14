#include "OfflineDeviceHooks.h"
#include <cuda_runtime.h>

namespace OfflineDeviceHooks {
void synchronize() {
    cudaDeviceSynchronize();
}

void releaseBuffer(void*& buffer) {
    if (buffer) {
        cudaFree(buffer);
        buffer = nullptr;
    }
}
} // namespace OfflineDeviceHooks
