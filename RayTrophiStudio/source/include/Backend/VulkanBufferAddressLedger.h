#pragma once

// Maps a GPU virtual address back to the buffer that owned it.
//
// VK_EXT_device_fault reports WHERE the GPU faulted (a VA), not WHAT lived
// there. This ledger records every VulkanDevice::createBuffer allocation with
// its device address, size and the creating call stack, and keeps a ring of
// recently destroyed ones, so a fault address can be named: "read of a buffer
// created by X, destroyed N allocations ago" is a use-after-free with a
// culprit; "inside a live buffer" is an out-of-bounds index.

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>

namespace VulkanRT {
namespace BufferLedger {

void noteCreated(VkDevice device, VkBuffer buffer, uint64_t address, uint64_t size);
void noteDestroyed(VkDevice device, VkBuffer buffer);
void forgetDevice(VkDevice device);

// Multi-line, human-readable: every live or recently destroyed buffer whose
// range contains `address` (or ends within `slack` bytes before it).
std::string describe(VkDevice device, uint64_t address, uint64_t slack);

} // namespace BufferLedger
} // namespace VulkanRT
