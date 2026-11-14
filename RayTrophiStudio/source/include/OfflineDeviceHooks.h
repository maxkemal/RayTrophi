#pragma once

// Exactly one Studio/Player implementation is selected by the target.
namespace OfflineDeviceHooks {
void synchronize();
void releaseBuffer(void*& buffer);
} // namespace OfflineDeviceHooks
