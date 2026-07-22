#include "RtIpcSession.h"

#include <openssl/rand.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_map>

namespace rtipc_session {
namespace {

std::mutex g_mutex;
std::unordered_map<std::string, SessionInfo> g_sessions;
constexpr size_t kClosedRetention = 128;

int64_t unixNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string randomId() {
    unsigned char bytes[16]{};
    if (RAND_bytes(bytes, sizeof(bytes)) != 1) return {};
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(32, '0');
    for (size_t i = 0; i < sizeof(bytes); ++i) {
        result[i * 2] = digits[bytes[i] >> 4u];
        result[i * 2 + 1] = digits[bytes[i] & 0x0fu];
    }
    return result;
}

void trimClosedLocked() {
    std::vector<std::pair<int64_t, std::string>> closed;
    for (const auto& [id, session] : g_sessions)
        if (!session.active) closed.emplace_back(session.last_activity_at, id);
    if (closed.size() <= kClosedRetention) return;
    std::sort(closed.begin(), closed.end());
    for (size_t i = 0; i < closed.size() - kClosedRetention; ++i)
        g_sessions.erase(closed[i].second);
}

} // namespace

std::string registerSession(const std::string& transport,
                            const std::string& peer_address, uint16_t peer_port,
                            const std::string& tls_version,
                            const std::string& tls_cipher) {
    SessionInfo session;
    session.connection_id = randomId();
    if (session.connection_id.empty()) return {};
    session.transport = transport; session.peer_address = peer_address;
    session.peer_port = peer_port; session.tls_version = tls_version;
    session.tls_cipher = tls_cipher; session.connected_at = unixNow();
    session.last_activity_at = session.connected_at; session.active = true;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sessions[session.connection_id] = session;
    trimClosedLocked();
    return session.connection_id;
}

void unregisterSession(const std::string& connection_id) noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_sessions.find(connection_id);
    if (found == g_sessions.end()) return;
    found->second.active = false;
    found->second.last_activity_at = unixNow();
    trimClosedLocked();
}

void bindToken(const std::string& connection_id, const std::string& token_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_sessions.find(connection_id);
    if (found != g_sessions.end()) found->second.token_id = token_id;
}

void recordRequest(const std::string& connection_id, uint64_t received,
                   uint64_t sent, bool failed) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_sessions.find(connection_id);
    if (found == g_sessions.end()) return;
    auto& session = found->second;
    ++session.request_count;
    if (failed) ++session.error_count;
    session.bytes_received += received; session.bytes_sent += sent;
    session.last_activity_at = unixNow();
}

bool shouldDisconnect(const std::string& connection_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_sessions.find(connection_id);
    return found == g_sessions.end() || found->second.disconnect_requested;
}

std::vector<SessionInfo> listSessions(bool include_closed) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::vector<SessionInfo> result;
    for (const auto& [id, session] : g_sessions)
        if (include_closed || session.active) result.push_back(session);
    std::sort(result.begin(), result.end(), [](const SessionInfo& a, const SessionInfo& b) {
        return a.connected_at > b.connected_at;
    });
    return result;
}

bool getSession(const std::string& connection_id, SessionInfo& output) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_sessions.find(connection_id);
    if (found == g_sessions.end()) return false;
    output = found->second; return true;
}

bool disconnect(const std::string& connection_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_sessions.find(connection_id);
    if (found == g_sessions.end() || !found->second.active) return false;
    found->second.disconnect_requested = true; return true;
}

size_t disconnectToken(const std::string& token_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    size_t count = 0;
    for (auto& [id, session] : g_sessions) {
        if (session.active && session.token_id == token_id) {
            session.disconnect_requested = true; ++count;
        }
    }
    return count;
}

size_t disconnectAll() {
    std::lock_guard<std::mutex> lock(g_mutex);
    size_t count = 0;
    for (auto& [id, session] : g_sessions) {
        if (session.active) { session.disconnect_requested = true; ++count; }
    }
    return count;
}

void clearClosed() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto it = g_sessions.begin(); it != g_sessions.end();)
        it = it->second.active ? std::next(it) : g_sessions.erase(it);
}

void shutdown() noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sessions.clear();
}

} // namespace rtipc_session
