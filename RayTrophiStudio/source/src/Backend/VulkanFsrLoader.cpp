#include "Backend/VulkanFsrLoader.h"

#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace VulkanRT {
bool enableFsrImageFeatures(const VkPhysicalDeviceFeatures& supported,
    VkPhysicalDeviceFeatures& enabled, bool legacy) {
    if (legacy) {
        return false;
    }
    enabled.shaderInt16 = supported.shaderInt16;
    enabled.shaderStorageImageReadWithoutFormat = supported.shaderStorageImageReadWithoutFormat;
    enabled.shaderStorageImageWriteWithoutFormat = supported.shaderStorageImageWriteWithoutFormat;
    enabled.shaderStorageImageExtendedFormats = supported.shaderStorageImageExtendedFormats;
    return enabled.shaderStorageImageReadWithoutFormat &&
        enabled.shaderStorageImageWriteWithoutFormat && enabled.shaderStorageImageExtendedFormats;
}

void FsrDeviceFeatures::append(VkPhysicalDevice physical, void**& nextLink, bool legacy) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    if (legacy || properties.apiVersion < VK_API_VERSION_1_2) {
        return;
    }
    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    supported.pNext = &arithmetic;
    arithmetic.pNext = &storage;
    storage.pNext = &subgroup;
    vkGetPhysicalDeviceFeatures2(physical, &supported);
    *nextLink = &arithmetic;
    nextLink = &subgroup.pNext;
}

VulkanFsrLoader::~VulkanFsrLoader() {
#ifdef _WIN32
    if (m_module) {
        FreeLibrary(static_cast<HMODULE>(m_module));
    }
#endif
}

bool VulkanFsrLoader::open(std::string& reason) {
    if (loaded()) {
        return true;
    }
#ifdef _WIN32
    wchar_t executable[32768]{};
    const DWORD length = GetModuleFileNameW(nullptr, executable, 32768);
    if (length == 0 || length >= 32768) {
        reason = "cannot resolve application directory for the FSR runtime";
        return false;
    }
    const auto path = std::filesystem::path(executable).parent_path() / L"amd_fidelityfx_vk.dll";
    HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) {
        reason = "amd_fidelityfx_vk.dll load failed (Windows error " +
            std::to_string(GetLastError()) +
            "); run scripts/setup_fidelityfx.ps1, then toggle upscale off/on";
        return false;
    }
    create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
    destroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
    configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(module, "ffxConfigure"));
    query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
    dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(module, "ffxDispatch"));
    if (!create || !destroy || !configure || !query || !dispatch) {
        FreeLibrary(module);
        reason = "FSR runtime is missing one of the five required FFX API exports";
        return false;
    }
    m_module = module;
    return true;
#else
    reason = "this FSR integration requires AMD's Windows Vulkan runtime DLL";
    return false;
#endif
}
} // namespace VulkanRT
