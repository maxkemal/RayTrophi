/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpc.h
 * Date:          July 2026
 * License:       MIT
 * =========================================================================
 *
 * RtIpc — Windows Named Pipe IPC server (Faz 4c).
 *
 * A local background transport listens on \\.\pipe\RayTrophiStudio; an optional
 * TLS transport exposes the same policy-controlled protocol to a private network.
 * Each message is a simplified JSON-RPC request: {"id":N, "method":"...", "params":{...}}.
 * The dispatcher enqueues the corresponding rtapi call on the main thread via
 * rtapi::enqueue(), waits for the result through a promise/future, and writes
 * the JSON response back to the pipe.
 *
 * Lifecycle:
 *   rtipc::start()  — called once from Main.cpp after rtapi::bind().
 *   rtipc::stop()   — called from Main.cpp before rtpython::shutdown().
 *
 * Named Pipe accepts one same-user client; TLS accepts up to eight clients.
 * Security, sessions, audit, transports and the management panel are independent
 * modules behind this public control-plane API.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace rtipc {

/// Start the IPC server thread. Returns false if the pipe could not be created.
bool start(std::string& error);

/// Stop the IPC server thread and release the pipe. Safe to call even if not started.
void stop() noexcept;

/// Returns true if the server thread is running.
bool isRunning();

/// True while the optional TLS remote listener is active.
bool isRemoteRunning();

struct TokenInfo {
    std::string id;
    std::string display_name;
    uint32_t capabilities = 0;
    int64_t created_at = 0;
    int64_t expires_at = 0;
    int64_t last_used_at = 0;
    bool revoked = false;
    std::vector<std::string> allowed_cidrs;
};

bool createToken(const std::string& display_name, uint32_t capabilities,
                 const std::vector<std::string>& allowed_cidrs,
                 int64_t expires_at, TokenInfo& out_info,
                 std::string& out_raw_token, std::string& error);
bool revokeToken(const std::string& token_id, std::string& error);
bool updateToken(const std::string& token_id, uint32_t capabilities,
                 const std::vector<std::string>& allowed_cidrs,
                 int64_t expires_at, std::string& error);
bool rotateToken(const std::string& token_id, TokenInfo& out_info,
                 std::string& out_raw_token, std::string& error);
std::vector<TokenInfo> listTokens();

struct SessionInfo {
    std::string connection_id, transport, peer_address, tls_version, tls_cipher, token_id;
    uint16_t peer_port = 0;
    int64_t connected_at = 0, last_activity_at = 0;
    uint64_t request_count = 0, error_count = 0, bytes_received = 0, bytes_sent = 0;
    bool active = false, disconnect_requested = false;
};

std::vector<SessionInfo> listSessions(bool include_closed = false);
bool disconnectSession(const std::string& connection_id);
size_t disconnectAllSessions();

struct AuditEvent {
    uint64_t sequence = 0;
    int64_t timestamp = 0;
    std::string connection_id, token_id, peer_address, method, outcome;
    uint64_t duration_us = 0, bytes_received = 0, bytes_sent = 0;
    bool allowed = false;
};

std::vector<AuditEvent> recentAuditEvents(size_t maximum = 256);
void clearAuditEvents();
void disableRemoteAccess() noexcept;

struct RemoteStatus {
    bool running = false;
    std::string bind_address, certificate_path, certificate_sha256, certificate_not_after;
    uint16_t port = 0;
    std::vector<std::string> subject_alt_names;
};
RemoteStatus remoteStatus();

} // namespace rtipc
