#include "RtIpcTransport.h"
#include "RtIpcSession.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#endif

namespace rtipc_transport {
namespace {

constexpr int kMaxClients = 8;
std::atomic<bool> g_stop{true};
std::atomic<bool>* g_application_stop = nullptr;
std::thread g_accept_thread;
SOCKET g_listener = INVALID_SOCKET;
SSL_CTX* g_ssl_context = nullptr;
std::mutex g_clients_mutex;
std::vector<SOCKET> g_client_sockets;
std::unordered_map<std::string, SOCKET> g_session_sockets;
std::condition_variable g_clients_stopped;
std::atomic<int> g_client_count{0};
bool g_winsock_started = false;
MessageHandler g_handler;
TlsStatus g_status;

std::string certificateFingerprint(X509* certificate) {
    unsigned char digest[EVP_MAX_MD_SIZE]{}; unsigned int size = 0;
    if (!certificate || X509_digest(certificate, EVP_sha256(), digest, &size) != 1) return {};
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string result;
    for (unsigned int i = 0; i < size; ++i) {
        if (i) result.push_back(':');
        result.push_back(digits[digest[i] >> 4u]); result.push_back(digits[digest[i] & 0x0fu]);
    }
    return result;
}

std::string asn1TimeText(const ASN1_TIME* time) {
    BIO* bio = BIO_new(BIO_s_mem());
    if (!bio) return {};
    if (time) ASN1_TIME_print(bio, time);
    char* data = nullptr; const long size = BIO_get_mem_data(bio, &data);
    std::string result = size > 0 ? std::string(data, static_cast<size_t>(size)) : std::string{};
    BIO_free(bio); return result;
}

std::vector<std::string> certificateSans(X509* certificate) {
    std::vector<std::string> result;
    if (!certificate) return result;
    auto* names = static_cast<GENERAL_NAMES*>(X509_get_ext_d2i(
        certificate, NID_subject_alt_name, nullptr, nullptr));
    if (!names) return result;
    for (int i = 0; i < sk_GENERAL_NAME_num(names); ++i) {
        const GENERAL_NAME* name = sk_GENERAL_NAME_value(names, i);
        if (name->type == GEN_DNS) {
            const auto* value = ASN1_STRING_get0_data(name->d.dNSName);
            result.emplace_back("DNS:" + std::string(reinterpret_cast<const char*>(value),
                static_cast<size_t>(ASN1_STRING_length(name->d.dNSName))));
        } else if (name->type == GEN_IPADD && ASN1_STRING_length(name->d.iPAddress) == 4) {
            const auto* value = ASN1_STRING_get0_data(name->d.iPAddress);
            result.emplace_back("IP:" + std::to_string(value[0]) + "." + std::to_string(value[1]) +
                                "." + std::to_string(value[2]) + "." + std::to_string(value[3]));
        }
    }
    GENERAL_NAMES_free(names); return result;
}

bool readExact(SSL* ssl, void* destination, size_t bytes) {
    auto* output = static_cast<unsigned char*>(destination);
    while (bytes > 0) {
        const int count = SSL_read(ssl, output,
            static_cast<int>((std::min)(bytes, size_t{1u << 20})));
        if (count <= 0) return false;
        output += count; bytes -= static_cast<size_t>(count);
    }
    return true;
}

bool writeExact(SSL* ssl, const void* source, size_t bytes) {
    const auto* input = static_cast<const unsigned char*>(source);
    while (bytes > 0) {
        const int count = SSL_write(ssl, input,
            static_cast<int>((std::min)(bytes, size_t{1u << 20})));
        if (count <= 0) return false;
        input += count; bytes -= static_cast<size_t>(count);
    }
    return true;
}

void clientLoop(SOCKET socket) {
    DWORD timeout_ms = 30000;
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    SSL* ssl = SSL_new(g_ssl_context);
    RequestContext request_context;
    request_context.remote = true;
    sockaddr_in peer{};
    int peer_size = sizeof(peer);
    if (getpeername(socket, reinterpret_cast<sockaddr*>(&peer), &peer_size) == 0) {
        char address[INET_ADDRSTRLEN]{};
        if (inet_ntop(AF_INET, &peer.sin_addr, address, sizeof(address)))
            request_context.peer_address = address;
        request_context.peer_port = ntohs(peer.sin_port);
    }
    if (ssl) SSL_set_fd(ssl, static_cast<int>(socket));
    if (ssl && SSL_accept(ssl) == 1) {
        request_context.connection_id = rtipc_session::registerSession(
            "tls", request_context.peer_address, request_context.peer_port,
            SSL_get_version(ssl), SSL_get_cipher_name(ssl));
        {
            std::lock_guard<std::mutex> lock(g_clients_mutex);
            g_session_sockets[request_context.connection_id] = socket;
        }
        auto window_start = std::chrono::steady_clock::now();
        int request_count = 0;
        while (!g_stop.load(std::memory_order_acquire) &&
               !g_application_stop->load(std::memory_order_acquire)) {
            if (rtipc_session::shouldDisconnect(request_context.connection_id)) break;
            uint32_t network_size = 0;
            if (!readExact(ssl, &network_size, sizeof(network_size))) break;
            const size_t size = ntohl(network_size);
            if (size == 0 || size > kMaxMessageBytes) break;
            const auto now = std::chrono::steady_clock::now();
            if (now - window_start >= std::chrono::seconds(1)) {
                window_start = now; request_count = 0;
            }
            if (++request_count > 120) break;
            std::string request(size, '\0');
            if (!readExact(ssl, request.data(), request.size())) break;
            const std::string response = g_handler(request, request_context);
            rtipc_session::recordRequest(request_context.connection_id, request.size(), response.size(),
                                         response.find("\"error\"") != std::string::npos);
            const uint32_t response_size = htonl(static_cast<uint32_t>(response.size()));
            if (!writeExact(ssl, &response_size, sizeof(response_size)) ||
                !writeExact(ssl, response.data(), response.size())) break;
        }
    }
    if (!request_context.connection_id.empty())
        rtipc_session::unregisterSession(request_context.connection_id);
    if (ssl) { SSL_shutdown(ssl); SSL_free(ssl); }
    shutdown(socket, SD_BOTH); closesocket(socket);
    {
        std::lock_guard<std::mutex> lock(g_clients_mutex);
        g_session_sockets.erase(request_context.connection_id);
        const auto it = std::find(g_client_sockets.begin(), g_client_sockets.end(), socket);
        if (it != g_client_sockets.end()) g_client_sockets.erase(it);
    }
    g_client_count.fetch_sub(1, std::memory_order_acq_rel);
    g_clients_stopped.notify_all();
}

void acceptLoop() {
    while (!g_stop.load(std::memory_order_acquire)) {
        const SOCKET client = accept(g_listener, nullptr, nullptr);
        if (client == INVALID_SOCKET) {
            if (g_stop.load(std::memory_order_acquire)) break;
            continue;
        }
        if (g_client_count.fetch_add(1, std::memory_order_acq_rel) >= kMaxClients) {
            g_client_count.fetch_sub(1, std::memory_order_acq_rel);
            closesocket(client); continue;
        }
        std::lock_guard<std::mutex> lock(g_clients_mutex);
        g_client_sockets.push_back(client);
        try { std::thread(clientLoop, client).detach(); }
        catch (...) {
            g_client_sockets.pop_back();
            g_client_count.fetch_sub(1, std::memory_order_acq_rel);
            closesocket(client);
        }
    }
}

} // namespace

void stopTls() noexcept {
    g_stop.store(true, std::memory_order_release);
    if (g_listener != INVALID_SOCKET) {
        shutdown(g_listener, SD_BOTH); closesocket(g_listener); g_listener = INVALID_SOCKET;
    }
    if (g_accept_thread.joinable()) g_accept_thread.join();
    {
        std::lock_guard<std::mutex> lock(g_clients_mutex);
        for (SOCKET socket : g_client_sockets) shutdown(socket, SD_BOTH);
    }
    bool clients_stopped = false;
    {
        std::unique_lock<std::mutex> lock(g_clients_mutex);
        g_clients_stopped.wait_for(lock, std::chrono::seconds(5), [] {
            return g_client_count.load(std::memory_order_acquire) == 0;
        });
        clients_stopped = g_client_count.load(std::memory_order_acquire) == 0;
        if (clients_stopped) { g_client_sockets.clear(); g_session_sockets.clear(); }
    }
    if (clients_stopped && g_ssl_context) { SSL_CTX_free(g_ssl_context); g_ssl_context = nullptr; }
    if (clients_stopped && g_winsock_started) { WSACleanup(); g_winsock_started = false; }
    if (clients_stopped) { g_handler = {}; g_application_stop = nullptr; }
    if (clients_stopped) g_status.running = false;
}

bool startTls(std::atomic<bool>& stop_requested, MessageHandler handler,
              std::string& error) {
    const char* enabled = std::getenv("RAYTROPHI_REMOTE_IPC");
    if (!enabled || std::string(enabled) != "1") return true;
    const char* certificate = std::getenv("RAYTROPHI_REMOTE_IPC_CERT");
    const char* private_key = std::getenv("RAYTROPHI_REMOTE_IPC_KEY");
    const char* bind_address = std::getenv("RAYTROPHI_REMOTE_IPC_BIND");
    const char* port_text = std::getenv("RAYTROPHI_REMOTE_IPC_PORT");
    if (!certificate || !private_key) {
        error = "remote IPC requires CERT and KEY"; return false;
    }
    char* port_end = nullptr;
    const long port = port_text ? std::strtol(port_text, &port_end, 10) : 7443;
    if (port < 1 || port > 65535 || (port_text && (!port_end || *port_end != '\0'))) {
        error = "remote IPC port must be between 1 and 65535"; return false;
    }
    WSADATA winsock{};
    if (WSAStartup(MAKEWORD(2, 2), &winsock) != 0) {
        error = "WSAStartup failed"; return false;
    }
    g_winsock_started = true;
    g_ssl_context = SSL_CTX_new(TLS_server_method());
    if (!g_ssl_context) { error = "TLS context creation failed"; stopTls(); return false; }
    SSL_CTX_set_min_proto_version(g_ssl_context, TLS1_2_VERSION);
    SSL_CTX_set_options(g_ssl_context, SSL_OP_NO_COMPRESSION | SSL_OP_CIPHER_SERVER_PREFERENCE);
    SSL_CTX_set_cipher_list(g_ssl_context, "ECDHE+AESGCM:ECDHE+CHACHA20");
    SSL_CTX_set_ciphersuites(g_ssl_context,
        "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256");
    if (SSL_CTX_use_certificate_chain_file(g_ssl_context, certificate) != 1 ||
        SSL_CTX_use_PrivateKey_file(g_ssl_context, private_key, SSL_FILETYPE_PEM) != 1 ||
        SSL_CTX_check_private_key(g_ssl_context) != 1) {
        error = "remote IPC TLS certificate/private key could not be loaded";
        stopTls(); return false;
    }
    X509* loaded_certificate = SSL_CTX_get0_certificate(g_ssl_context);
    g_status.certificate_path = certificate;
    g_status.certificate_sha256 = certificateFingerprint(loaded_certificate);
    g_status.certificate_not_after = asn1TimeText(X509_get0_notAfter(loaded_certificate));
    g_status.subject_alt_names = certificateSans(loaded_certificate);
    g_listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (g_listener == INVALID_SOCKET) {
        error = "remote IPC socket creation failed"; stopTls(); return false;
    }
    BOOL exclusive = TRUE;
    setsockopt(g_listener, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
               reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
    sockaddr_in address{};
    address.sin_family = AF_INET; address.sin_port = htons(static_cast<u_short>(port));
    const char* address_text = bind_address ? bind_address : "127.0.0.1";
    if (inet_pton(AF_INET, address_text, &address.sin_addr) != 1 ||
        bind(g_listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR ||
        listen(g_listener, kMaxClients) == SOCKET_ERROR) {
        error = "remote IPC bind/listen failed for " + std::string(address_text) +
                ":" + std::to_string(port);
        stopTls(); return false;
    }
    g_application_stop = &stop_requested;
    g_status.bind_address = address_text; g_status.port = static_cast<uint16_t>(port);
    g_status.running = true;
    g_handler = std::move(handler);
    g_stop.store(false, std::memory_order_release);
    try { g_accept_thread = std::thread(acceptLoop); }
    catch (const std::exception& e) {
        error = std::string("remote IPC thread failed: ") + e.what();
        stopTls(); return false;
    }
    return true;
}

bool isTlsRunning() noexcept {
    return g_listener != INVALID_SOCKET && !g_stop.load(std::memory_order_acquire);
}

TlsStatus tlsStatus() {
    std::lock_guard<std::mutex> lock(g_clients_mutex);
    TlsStatus result = g_status;
    result.running = isTlsRunning();
    return result;
}

void interruptDisconnectedSessions() noexcept {
    std::vector<std::string> requested;
    for (const auto& session : rtipc_session::listSessions(false))
        if (session.disconnect_requested) requested.push_back(session.connection_id);
    std::lock_guard<std::mutex> lock(g_clients_mutex);
    for (const auto& connection_id : requested) {
        const auto found = g_session_sockets.find(connection_id);
        if (found != g_session_sockets.end()) shutdown(found->second, SD_BOTH);
    }
}

} // namespace rtipc_transport
