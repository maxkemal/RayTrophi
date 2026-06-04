#ifndef DLL_LOAD_POLICY_H
#define DLL_LOAD_POLICY_H

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace Platform {
namespace Dll {

enum class DllCategory {
    Auto,
    Driver,
    Runtime
};

inline bool isKnownDriverDll(const char* dllName) {
    if (!dllName || !dllName[0]) return false;

    return lstrcmpiA(dllName, "vulkan-1.dll") == 0 ||
           lstrcmpiA(dllName, "nvcuda.dll") == 0;
}

inline HMODULE loadModuleWithPolicy(const char* dllName,
                                    DllCategory category = DllCategory::Auto,
                                    bool allowLegacySearchForRuntime = true) {
    if (!dllName || !dllName[0]) return nullptr;

    bool isDriver = (category == DllCategory::Driver) ||
                    (category == DllCategory::Auto && isKnownDriverDll(dllName));

    HMODULE module = GetModuleHandleA(dllName);
    if (module) return module;

    module = LoadLibraryExA(dllName, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (module) return module;

    module = LoadLibraryExA(dllName, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (module) return module;

    if (!isDriver && allowLegacySearchForRuntime) {
        module = LoadLibraryA(dllName);
        if (module) return module;
    }

    return nullptr;
}

} // namespace Dll
} // namespace Platform

#endif // _WIN32

#endif // DLL_LOAD_POLICY_H
