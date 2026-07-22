#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace rtipc_session {

struct SessionInfo {
    std::string connection_id;
    std::string transport;
    std::string peer_address;
    uint16_t peer_port = 0;
    std::string tls_version;
    std::string tls_cipher;
    std::string token_id;
    int64_t connected_at = 0;
    int64_t last_activity_at = 0;
    uint64_t request_count = 0;
    uint64_t error_count = 0;
    uint64_t bytes_received = 0;
    uint64_t bytes_sent = 0;
    bool active = false;
    bool disconnect_requested = false;
};

std::string registerSession(const std::string& transport,
                            const std::string& peer_address, uint16_t peer_port,
                            const std::string& tls_version = {},
                            const std::string& tls_cipher = {});
void unregisterSession(const std::string& connection_id) noexcept;
void bindToken(const std::string& connection_id, const std::string& token_id);
void recordRequest(const std::string& connection_id, uint64_t bytes_received,
                   uint64_t bytes_sent, bool failed);
bool shouldDisconnect(const std::string& connection_id);

std::vector<SessionInfo> listSessions(bool include_closed = false);
bool getSession(const std::string& connection_id, SessionInfo& output);
bool disconnect(const std::string& connection_id);
size_t disconnectToken(const std::string& token_id);
size_t disconnectAll();
void clearClosed();
void shutdown() noexcept;

} // namespace rtipc_session
