#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rtipc_transport {

struct RequestContext {
    bool remote = false;
    std::string connection_id;
    std::string peer_address;
    uint16_t peer_port = 0;
};

struct TlsStatus {
    bool running = false;
    std::string bind_address;
    uint16_t port = 0;
    std::string certificate_path;
    std::string certificate_sha256;
    std::string certificate_not_after;
    std::vector<std::string> subject_alt_names;
};

using MessageHandler = std::function<std::string(const std::string&,
                                                 const RequestContext&)>;

constexpr std::size_t kMaxMessageBytes = 16u * 1024u * 1024u;

bool startLocal(std::atomic<bool>& stop_requested, MessageHandler handler,
                std::string& error);
void stopLocal() noexcept;
void interruptLocalDisconnectedSession() noexcept;

bool startTls(std::atomic<bool>& stop_requested, MessageHandler handler,
              std::string& error);
void stopTls() noexcept;
bool isTlsRunning() noexcept;
TlsStatus tlsStatus();
void interruptDisconnectedSessions() noexcept;

} // namespace rtipc_transport
