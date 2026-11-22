/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcTransportLocal.cpp
 * License:       MIT
 * =========================================================================
 * Local named-pipe transport.
 *
 * ★★★ SEVERAL pipe instances, one worker thread each.
 *
 * This used to be a single instance with a single thread, which meant exactly
 * one client at a time. That is not a capacity limit, it is a workflow limit:
 * with the Python agent runtime attached, `scripts/ipc/RtIpc.psm1` and every
 * script under `scripts/test/` could no longer connect - the QA channel this
 * project runs on was locked out by the very agent it was built to drive. It
 * also made the manager/controller/worker hierarchy in
 * docs/dev/AGENT_DISCOVERY_LAYER_PLAN.md impossible to even try.
 *
 * Concurrency is not new here: the TLS transport has always served one thread
 * per client through the same handler, and dispatchMethod() marshals real work
 * onto the main thread through enqueueQuery/enqueueResult.
 * =========================================================================
 */

#include "RtPlayerIpcPolicy.h"
#include "RtIpcTransport.h"
#include "RtIpcSession.h"

#include "json.hpp"

#include <algorithm>
#include <array>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <sddl.h>
#else
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

using json = nlohmann::json;

#ifdef _WIN32

namespace rtipc_transport {
namespace {

constexpr const wchar_t* kPipeName = rtipc_product::pipeName;
constexpr DWORD kBufferSize = 65536;
// One for the agent runtime, one for a PowerShell/pytest session driving the
// app, and headroom for a second agent and an ad-hoc probe.
constexpr DWORD kInstanceCount = 4;

struct PipeInstance {
    HANDLE pipe = INVALID_HANDLE_VALUE;
    std::thread thread;
    std::string session;   // connection id while a client is attached
};

std::array<PipeInstance, kInstanceCount> g_instances;
std::atomic<bool>* g_stop = nullptr;
MessageHandler g_handler;
std::mutex g_session_mutex;

bool writeMessage(HANDLE pipe, const std::string& message) {
    if (message.size() > kMaxMessageBytes) return false;
    DWORD written = 0;
    return WriteFile(pipe, message.data(), static_cast<DWORD>(message.size()),
                     &written, nullptr) && written == message.size();
}

bool makeSecurity(SECURITY_ATTRIBUTES& attributes,
                  PSECURITY_DESCRIPTOR& descriptor, std::string& error) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
        error = "OpenProcessToken failed (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    DWORD bytes = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &bytes);
    std::vector<unsigned char> storage(bytes);
    if (bytes == 0 || !GetTokenInformation(token, TokenUser, storage.data(), bytes, &bytes)) {
        error = "GetTokenInformation failed (error " + std::to_string(GetLastError()) + ")";
        CloseHandle(token); return false;
    }
    CloseHandle(token);
    LPWSTR sid_text = nullptr;
    if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(storage.data())->User.Sid,
                                &sid_text)) {
        error = "ConvertSidToStringSid failed (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    const std::wstring sddl = L"D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;" +
                              std::wstring(sid_text) + L")";
    LocalFree(sid_text);
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.c_str(), SDDL_REVISION_1, &descriptor, nullptr)) {
        error = "pipe security descriptor creation failed (error " +
                std::to_string(GetLastError()) + ")";
        return false;
    }
    attributes = {};
    attributes.nLength = sizeof(attributes);
    attributes.lpSecurityDescriptor = descriptor;
    return true;
}

// Every instance carries the same ACL; only the first one claims the name.
HANDLE createInstance(bool first, std::string& error) {
    SECURITY_ATTRIBUTES security{};
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!makeSecurity(security, descriptor, error)) return INVALID_HANDLE_VALUE;
    DWORD open_mode = PIPE_ACCESS_DUPLEX;
    if (first) open_mode |= FILE_FLAG_FIRST_PIPE_INSTANCE;
    HANDLE pipe = CreateNamedPipeW(
        kPipeName, open_mode,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
        kInstanceCount, kBufferSize, kBufferSize, 0, &security);
    LocalFree(descriptor);
    if (pipe == INVALID_HANDLE_VALUE)
        error = "cannot create named pipe (error " + std::to_string(GetLastError()) + ")";
    return pipe;
}

void serveConnection(PipeInstance& instance, const RequestContext& context) {
    std::vector<char> buffer(kBufferSize);
    while (!g_stop->load(std::memory_order_acquire) &&
           !rtipc_session::shouldDisconnect(context.connection_id)) {
        std::string message;
        DWORD bytes = 0;
        BOOL ok = ReadFile(instance.pipe, buffer.data(), static_cast<DWORD>(buffer.size()),
                           &bytes, nullptr);
        DWORD read_error = ok ? ERROR_SUCCESS : GetLastError();
        if (!ok && read_error != ERROR_MORE_DATA) break;
        message.append(buffer.data(), bytes);
        bool oversized = false;
        while (!ok && read_error == ERROR_MORE_DATA) {
            ok = ReadFile(instance.pipe, buffer.data(), static_cast<DWORD>(buffer.size()),
                          &bytes, nullptr);
            read_error = ok ? ERROR_SUCCESS : GetLastError();
            if (!oversized) {
                message.append(buffer.data(), bytes);
                oversized = message.size() > kMaxMessageBytes;
            }
            if (!ok && read_error != ERROR_MORE_DATA) break;
        }
        if (oversized) {
            if (!writeMessage(instance.pipe,
                              json{{"id", 0}, {"error", "message exceeds size limit"}}.dump()))
                break;
            continue;
        }
        if (message.empty()) break;
        const std::string response = g_handler(message, context);
        rtipc_session::recordRequest(context.connection_id, message.size(), response.size(),
                                     response.find("\"error\"") != std::string::npos);
        if (!writeMessage(instance.pipe, response)) break;
    }
}

void serverLoop(size_t index) {
    PipeInstance& instance = g_instances[index];
    while (!g_stop->load(std::memory_order_acquire)) {
        const BOOL connected = ConnectNamedPipe(instance.pipe, nullptr);
        if (!connected && GetLastError() != ERROR_PIPE_CONNECTED) {
            if (g_stop->load(std::memory_order_acquire)) break;
            continue;
        }
        RequestContext context;
        context.connection_id = rtipc_session::registerSession("named_pipe", "local", 0);
        {
            std::lock_guard<std::mutex> lock(g_session_mutex);
            instance.session = context.connection_id;
        }

        serveConnection(instance, context);

        rtipc_session::unregisterSession(context.connection_id);
        {
            std::lock_guard<std::mutex> lock(g_session_mutex);
            instance.session.clear();
        }
        DisconnectNamedPipe(instance.pipe);
    }
}

} // namespace

bool startLocal(std::atomic<bool>& stop_requested, MessageHandler handler,
                std::string& error) {
    g_stop = &stop_requested;
    g_handler = std::move(handler);

    for (size_t index = 0; index < g_instances.size(); ++index) {
        g_instances[index].pipe = createInstance(index == 0, error);
        if (g_instances[index].pipe == INVALID_HANDLE_VALUE) {
            // Roll back whatever came up, so a partial start cannot leave
            // orphan instances answering on the name.
            for (size_t done = 0; done < index; ++done) {
                CloseHandle(g_instances[done].pipe);
                g_instances[done].pipe = INVALID_HANDLE_VALUE;
            }
            g_handler = {};
            g_stop = nullptr;
            return false;
        }
    }

    for (size_t index = 0; index < g_instances.size(); ++index) {
        try {
            g_instances[index].thread = std::thread(serverLoop, index);
        } catch (const std::exception& e) {
            error = std::string("cannot start IPC server thread: ") + e.what();
            stop_requested.store(true, std::memory_order_release);
            stopLocal();
            return false;
        }
    }
    return true;
}

void stopLocal() noexcept {
    // Break threads out of a blocking ConnectNamedPipe or ReadFile. The dummy
    // client is the reliable half: CancelSynchronousIo only reaches a call that
    // is in flight right now.
    for (auto& instance : g_instances) {
        if (instance.thread.joinable())
            CancelSynchronousIo(instance.thread.native_handle());
    }
    for (size_t index = 0; index < g_instances.size(); ++index) {
        HANDLE dummy = CreateFileW(kPipeName, GENERIC_READ | GENERIC_WRITE,
                                   0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (dummy != INVALID_HANDLE_VALUE) CloseHandle(dummy);
    }
    for (auto& instance : g_instances) {
        if (instance.thread.joinable()) instance.thread.join();
        if (instance.pipe != INVALID_HANDLE_VALUE) {
            CloseHandle(instance.pipe);
            instance.pipe = INVALID_HANDLE_VALUE;
        }
        instance.session.clear();
    }
    g_handler = {};
    g_stop = nullptr;
}

void interruptLocalDisconnectedSession() noexcept {
    std::vector<std::pair<size_t, std::string>> attached;
    {
        std::lock_guard<std::mutex> lock(g_session_mutex);
        for (size_t index = 0; index < g_instances.size(); ++index)
            if (!g_instances[index].session.empty())
                attached.emplace_back(index, g_instances[index].session);
    }
    for (const auto& entry : attached) {
        if (!rtipc_session::shouldDisconnect(entry.second)) continue;
        PipeInstance& instance = g_instances[entry.first];
        if (instance.thread.joinable())
            CancelSynchronousIo(instance.thread.native_handle());
    }
}

} // namespace rtipc_transport

#else // ---------------------------------------------------------------- POSIX / Android
/*
 * Abstract unix stream socket, the Android counterpart of the named pipe.
 *
 * Security boundary = peer credentials, the analogue of the pipe ACL: SO_PEERCRED
 * admits only this app's uid, root, and (Android) the adb shell uid. Any other app
 * on the device can see an abstract socket name, so the check is not optional.
 * A loopback TCP port would have no such check — that is why this is not TCP.
 *
 * Framing: 4-byte big-endian length + UTF-8 JSON, both directions (same as the TLS
 * transport). From the PC: `adb forward tcp:47391 localabstract:RayTrophiPlayer`
 * then `Connect-RtIpc -Port 47391` (scripts/ipc/RtIpc.psm1).
 */
namespace rtipc_transport {
namespace {

constexpr size_t kInstanceCount = 4;   // same client budget as the pipe
#ifdef __ANDROID__
constexpr uid_t kAdbShellUid = 2000;   // AID_SHELL
#endif

struct ClientSlot {
    int fd = -1;
    std::thread thread;
    std::atomic<bool> busy{false};
    std::string session;
};

std::array<ClientSlot, kInstanceCount> g_slots;
std::thread g_accept_thread;
int g_listen_fd = -1;
std::atomic<bool>* g_stop = nullptr;
MessageHandler g_handler;
std::mutex g_session_mutex;

bool stopping() { return !g_stop || g_stop->load(std::memory_order_acquire); }

bool readExact(int fd, void* data, size_t size) {
    auto* bytes = static_cast<unsigned char*>(data);
    while (size > 0) {
        const ssize_t got = recv(fd, bytes, size, 0);
        if (got == 0) return false;
        if (got < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        bytes += got;
        size -= static_cast<size_t>(got);
    }
    return true;
}

bool writeExact(int fd, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    while (size > 0) {
        const ssize_t sent = send(fd, bytes, size, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        bytes += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

bool writeFrame(int fd, const std::string& message) {
    if (message.size() > kMaxMessageBytes) return false;
    const uint32_t size = static_cast<uint32_t>(message.size());
    const unsigned char header[4] = {static_cast<unsigned char>(size >> 24),
                                     static_cast<unsigned char>(size >> 16),
                                     static_cast<unsigned char>(size >> 8),
                                     static_cast<unsigned char>(size)};
    return writeExact(fd, header, sizeof(header)) &&
           writeExact(fd, message.data(), message.size());
}

bool peerAllowed(int fd) {
    struct ucred credentials{};
    socklen_t length = sizeof(credentials);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &length) != 0 ||
        length != sizeof(credentials))
        return false;   // unknown peer = rejected (fail closed)
    if (credentials.uid == getuid() || credentials.uid == 0) return true;
#ifdef __ANDROID__
    if (credentials.uid == kAdbShellUid) return true;
#endif
    return false;
}

void serveClient(size_t index) {
    ClientSlot& slot = g_slots[index];
    RequestContext context;
    context.connection_id = rtipc_session::registerSession("local_socket", "local", 0);
    {
        std::lock_guard<std::mutex> lock(g_session_mutex);
        slot.session = context.connection_id;
    }
    while (!stopping() && !rtipc_session::shouldDisconnect(context.connection_id)) {
        unsigned char header[4];
        if (!readExact(slot.fd, header, sizeof(header))) break;
        const size_t size = (size_t(header[0]) << 24) | (size_t(header[1]) << 16) |
                            (size_t(header[2]) << 8) | size_t(header[3]);
        if (size == 0) break;
        if (size > kMaxMessageBytes) {
            // The stream cannot be resynchronised without reading the body; answer and drop.
            writeFrame(slot.fd, json{{"id", 0}, {"error", "message exceeds size limit"}}.dump());
            break;
        }
        std::string message(size, '\0');
        if (!readExact(slot.fd, message.data(), message.size())) break;
        const std::string response = g_handler(message, context);
        rtipc_session::recordRequest(context.connection_id, message.size(), response.size(),
                                     response.find("\"error\"") != std::string::npos);
        if (!writeFrame(slot.fd, response)) break;
    }
    rtipc_session::unregisterSession(context.connection_id);
    {
        std::lock_guard<std::mutex> lock(g_session_mutex);
        slot.session.clear();
        if (slot.fd >= 0) { close(slot.fd); slot.fd = -1; }
    }
    slot.busy.store(false, std::memory_order_release);
}

void acceptLoop() {
    while (!stopping()) {
        pollfd waiter{g_listen_fd, POLLIN, 0};
        const int ready = poll(&waiter, 1, 200);
        if (ready <= 0) continue;   // timeout or EINTR: re-check stop
        const int client = accept4(g_listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
        if (client < 0) continue;
        if (!peerAllowed(client)) { close(client); continue; }
        size_t free_index = kInstanceCount;
        for (size_t index = 0; index < kInstanceCount; ++index) {
            if (!g_slots[index].busy.load(std::memory_order_acquire)) { free_index = index; break; }
        }
        if (free_index == kInstanceCount) {
            writeFrame(client, json{{"id", 0}, {"error", "IPC client limit reached"}}.dump());
            close(client);
            continue;
        }
        ClientSlot& slot = g_slots[free_index];
        if (slot.thread.joinable()) slot.thread.join();   // previous client already finished
        {
            std::lock_guard<std::mutex> lock(g_session_mutex);
            slot.fd = client;
        }
        slot.busy.store(true, std::memory_order_release);
        try {
            slot.thread = std::thread(serveClient, free_index);
        } catch (const std::exception&) {
            slot.busy.store(false, std::memory_order_release);
            std::lock_guard<std::mutex> lock(g_session_mutex);
            close(slot.fd); slot.fd = -1;
        }
    }
}

} // namespace

bool startLocal(std::atomic<bool>& stop_requested, MessageHandler handler,
                std::string& error) {
    const char* name = rtipc_product::localSocketName;
    const size_t name_length = std::strlen(name);
    sockaddr_un address{};
    if (name_length + 1 > sizeof(address.sun_path)) {
        error = "local socket name too long";
        return false;
    }
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        error = std::string("cannot create local socket: ") + std::strerror(errno);
        return false;
    }
    address.sun_family = AF_UNIX;
    address.sun_path[0] = '\0';   // abstract namespace: no file, vanishes with the process
    std::memcpy(address.sun_path + 1, name, name_length);
    const socklen_t address_length =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + name_length);
    if (bind(fd, reinterpret_cast<const sockaddr*>(&address), address_length) != 0 ||
        listen(fd, static_cast<int>(kInstanceCount)) != 0) {
        error = std::string("cannot bind local socket @") + name + ": " + std::strerror(errno);
        close(fd);
        return false;
    }
    g_listen_fd = fd;
    g_stop = &stop_requested;
    g_handler = std::move(handler);
    try {
        g_accept_thread = std::thread(acceptLoop);
    } catch (const std::exception& e) {
        error = std::string("cannot start IPC server thread: ") + e.what();
        close(g_listen_fd); g_listen_fd = -1;
        g_handler = {};
        g_stop = nullptr;
        return false;
    }
    return true;
}

void stopLocal() noexcept {
    // The caller has set stop_requested; poll() wakes within 200 ms. Client threads
    // blocked in recv() are released by shutting their sockets down.
    if (g_accept_thread.joinable()) g_accept_thread.join();
    if (g_listen_fd >= 0) { close(g_listen_fd); g_listen_fd = -1; }
    {
        std::lock_guard<std::mutex> lock(g_session_mutex);
        for (auto& slot : g_slots)
            if (slot.fd >= 0) shutdown(slot.fd, SHUT_RDWR);
    }
    for (auto& slot : g_slots)
        if (slot.thread.joinable()) slot.thread.join();
    g_handler = {};
    g_stop = nullptr;
}

void interruptLocalDisconnectedSession() noexcept {
    std::lock_guard<std::mutex> lock(g_session_mutex);
    for (auto& slot : g_slots) {
        if (slot.fd >= 0 && !slot.session.empty() &&
            rtipc_session::shouldDisconnect(slot.session))
            shutdown(slot.fd, SHUT_RDWR);
    }
}

} // namespace rtipc_transport
#endif
