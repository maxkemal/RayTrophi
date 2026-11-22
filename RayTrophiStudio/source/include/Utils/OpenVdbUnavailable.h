#pragma once

#include <cstdio>
#include <mutex>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace OpenVdbUnavailable {

inline constexpr char message[] = "OpenVDB not available on Android";

inline const char* report() {
    static std::once_flag logged;
    std::call_once(logged, [] {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_WARN, "RayTrophi", "%s", message);
#else
        std::fprintf(stderr, "%s\n", message);
#endif
    });
    return message;
}

} // namespace OpenVdbUnavailable
