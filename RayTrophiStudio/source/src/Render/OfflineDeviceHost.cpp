#include "OfflineDeviceHooks.h"
#include <cassert>

namespace OfflineDeviceHooks {
void synchronize() {
}

void releaseBuffer(void*& buffer) {
    // Player never allocates an offline device buffer.
    assert(buffer == nullptr);
    buffer = nullptr;
}
} // namespace OfflineDeviceHooks
