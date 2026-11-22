#pragma once

#ifdef RT_PLAYER
#include <cstdlib>
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

namespace VulkanRT {

inline bool forceNoAtomicFloatForDiagnostics() {
#ifdef __ANDROID__
    char property[PROP_VALUE_MAX]{};
    if (__system_property_get("debug.raytrophi.force_no_atomic_float", property) == 1 &&
        property[0] == '1') {
        return true;
    }
#endif
    const char* value = std::getenv("RAYTROPHI_FORCE_NO_ATOMIC_FLOAT");
    return value && value[0] == '1' && value[1] == '\0';
}

} // namespace VulkanRT
#endif
