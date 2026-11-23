// Android stub for the camera/perception bridge (replaces Core/PlayerCamera.cpp and
// UI/PlayerOperatorCameraUI.cpp). Everything else in the operator layer (library, game settings,
// hidden objects, display, audio, IPC) is the shared code and runs on Android. The camera needs
// a Windows Media Foundation device and a Python worker child process, neither exists here:
// every command returns an explicit error and the status reports an empty device catalog.
#include "Core/PlayerCamera.h"
#include "../../UI/PlayerOperatorWidgets.h"

namespace AppRuntime {
struct PlayerCamera::Impl {};

PlayerCamera::PlayerCamera() = default;
PlayerCamera::~PlayerCamera() = default;

rtapi::Result PlayerCamera::command(const std::string &, const nlohmann::json &) {
    return rtapi::Result::fail("camera input is not available on Android");
}
nlohmann::json PlayerCamera::status() {
    return {{"active", false},
            {"state", "unavailable"},
            {"catalog", {{"devices", nlohmann::json::array()}}}};
}
void PlayerCamera::tick() {}
SDL_Texture *PlayerCamera::preview(SDL_Renderer *) {
    return nullptr;
}
} // namespace AppRuntime

namespace PlayerConsole {
// The console hides the Camera page on Android; this keeps the symbol linked.
void camera(SDL_Renderer *) {}
} // namespace PlayerConsole
