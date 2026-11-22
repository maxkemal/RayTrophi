// Android stub for the remote TLS IPC transport (replaces Api/RtIpcTransportTls.cpp).
// The TLS transport needs Winsock + OpenSSL server code and is out of scope on Android:
// the QA channel there is the local loopback transport (RtIpcTransportLocal.cpp, batch A3).
// Nothing here fakes success: when remote IPC is requested (RAYTROPHI_REMOTE_IPC=1),
// startTls() fails with an explicit error and logs once; otherwise it starts nothing.
#include "Api/RtIpcTransport.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace rtipc_transport {
namespace {
void logOnce() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        const char* msg = "[RtIpc] TLS transport not available on Android (use the local loopback transport)";
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_WARN, "RayTrophi", "%s", msg);
#else
        std::fprintf(stderr, "%s\n", msg);
#endif
    });
}
} // namespace

bool startTls(std::atomic<bool>&, MessageHandler, std::string& error) {
    // Same contract as the real transport: remote IPC not requested -> nothing to start,
    // and the caller keeps the local transport alive. Returning false here would tear
    // down the whole IPC server (RtIpc.cpp stops the local transport on TLS failure),
    // i.e. the Android QA channel would never come up.
    const char* enabled = std::getenv("RAYTROPHI_REMOTE_IPC");
    if (!enabled || std::string(enabled) != "1") return true;
    logOnce();
    error = "TLS remote IPC is not available on Android";
    return false;
}

void stopTls() noexcept {}

bool isTlsRunning() noexcept { return false; }

TlsStatus tlsStatus() { return TlsStatus{}; }

void interruptDisconnectedSessions() noexcept {}

} // namespace rtipc_transport
