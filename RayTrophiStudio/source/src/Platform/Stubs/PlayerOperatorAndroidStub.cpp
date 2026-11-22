// Android stub for the Player operator/kiosk layer (replaces Core/PlayerOperator.cpp,
// Core/PlayerCamera.cpp, Api/RtApiPlayerOperator.cpp and UI/PlayerOperator*UI.cpp).
// The operator console, library UI and camera-perception bridge are Windows-only and out of
// scope for the touch build (docs/dev/PLAYER_ANDROID_PORT.md). Every command returns an explicit
// error; draw/tick/shutdown are no-ops so PlayerMain's unconditional calls link and do nothing.
#include "Api/RtApiPlayerOperator.h"
#include "Core/PlayerOperator.h"
#include "UI/PlayerOperatorUI.h"

#include <cstdio>
#include <mutex>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace {
const char* kReason = "player operator layer is not available on Android";

void logOnce() {
    static std::once_flag flag;
    std::call_once(flag, [] {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_WARN, "RayTrophi", "[PlayerOperator] %s", kReason);
#else
        std::fprintf(stderr, "[PlayerOperator] %s\n", kReason);
#endif
    });
}
} // namespace

namespace rtapi {
Result playerOperator(const std::string&, const nlohmann::json&, nlohmann::json&) {
    logOnce();
    return Result::fail(kReason);
}
} // namespace rtapi

namespace AppRuntime {
rtapi::Result playerOperatorCommand(UIContext&, const std::string&, const nlohmann::json&,
                                    nlohmann::json&) {
    logOnce();
    return rtapi::Result::fail(kReason);
}
SDL_Texture* playerOperatorPreview(SDL_Renderer*) { return nullptr; }
void setPlayerOperatorLaunchOverrides(const nlohmann::json&) {}
void tickPlayerOperator() {}
void shutdownPlayerOperator() {}

void showPlayerOperator(const std::string&) { logOnce(); }
bool playerOperatorOpen() { return false; }
void closePlayerOperator() {}
void drawPlayerOperator(SDL_Renderer*) {}
void loadPlayerOperatorFonts() {}
void releasePlayerOperatorUI() {}
} // namespace AppRuntime
