#include "RtIpcTransport.h"
#include "RtIpcSession.h"

#include "json.hpp"

#include <thread>
#include <mutex>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <sddl.h>
#endif

using json = nlohmann::json;

namespace rtipc_transport {
namespace {

constexpr const wchar_t* kPipeName = L"\\\\.\\pipe\\RayTrophiStudio";
constexpr DWORD kBufferSize = 65536;
std::thread g_thread;
HANDLE g_pipe = INVALID_HANDLE_VALUE;
std::atomic<bool>* g_stop = nullptr;
MessageHandler g_handler;
std::mutex g_session_mutex;
std::string g_current_session;

bool writeMessage(const std::string& message) {
    if (message.size() > kMaxMessageBytes) return false;
    DWORD written = 0;
    return WriteFile(g_pipe, message.data(), static_cast<DWORD>(message.size()),
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

void serverLoop() {
    std::vector<char> buffer(kBufferSize);
    while (!g_stop->load(std::memory_order_acquire)) {
        const BOOL connected = ConnectNamedPipe(g_pipe, nullptr);
        if (!connected && GetLastError() != ERROR_PIPE_CONNECTED) {
            if (g_stop->load(std::memory_order_acquire)) break;
            continue;
        }
        RequestContext context;
        context.connection_id = rtipc_session::registerSession("named_pipe", "local", 0);
        {
            std::lock_guard<std::mutex> lock(g_session_mutex);
            g_current_session = context.connection_id;
        }
        while (!g_stop->load(std::memory_order_acquire) &&
               !rtipc_session::shouldDisconnect(context.connection_id)) {
            std::string message;
            DWORD bytes = 0;
            BOOL ok = ReadFile(g_pipe, buffer.data(), static_cast<DWORD>(buffer.size()),
                               &bytes, nullptr);
            DWORD read_error = ok ? ERROR_SUCCESS : GetLastError();
            if (!ok && read_error != ERROR_MORE_DATA) break;
            message.append(buffer.data(), bytes);
            bool oversized = false;
            while (!ok && read_error == ERROR_MORE_DATA) {
                ok = ReadFile(g_pipe, buffer.data(), static_cast<DWORD>(buffer.size()),
                              &bytes, nullptr);
                read_error = ok ? ERROR_SUCCESS : GetLastError();
                if (!oversized) {
                    message.append(buffer.data(), bytes);
                    oversized = message.size() > kMaxMessageBytes;
                }
                if (!ok && read_error != ERROR_MORE_DATA) break;
            }
            if (oversized) {
                if (!writeMessage(json{{"id", 0}, {"error", "message exceeds size limit"}}.dump())) break;
                continue;
            }
            if (message.empty()) break;
            const std::string response = g_handler(message, context);
            rtipc_session::recordRequest(context.connection_id, message.size(), response.size(),
                                         response.find("\"error\"") != std::string::npos);
            if (!writeMessage(response)) break;
        }
        rtipc_session::unregisterSession(context.connection_id);
        {
            std::lock_guard<std::mutex> lock(g_session_mutex);
            g_current_session.clear();
        }
        DisconnectNamedPipe(g_pipe);
    }
    if (g_pipe != INVALID_HANDLE_VALUE) {
        CloseHandle(g_pipe); g_pipe = INVALID_HANDLE_VALUE;
    }
}

} // namespace

bool startLocal(std::atomic<bool>& stop_requested, MessageHandler handler,
                std::string& error) {
    SECURITY_ATTRIBUTES security{};
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!makeSecurity(security, descriptor, error)) return false;
    g_pipe = CreateNamedPipeW(kPipeName, PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
        1, kBufferSize, kBufferSize, 0, &security);
    LocalFree(descriptor);
    if (g_pipe == INVALID_HANDLE_VALUE) {
        error = "cannot create named pipe (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    g_stop = &stop_requested;
    g_handler = std::move(handler);
    try { g_thread = std::thread(serverLoop); }
    catch (const std::exception& e) {
        CloseHandle(g_pipe); g_pipe = INVALID_HANDLE_VALUE;
        error = std::string("cannot start IPC server thread: ") + e.what(); return false;
    }
    return true;
}

void stopLocal() noexcept {
    if (g_thread.joinable()) CancelSynchronousIo(g_thread.native_handle());
    HANDLE dummy = CreateFileW(kPipeName, GENERIC_READ | GENERIC_WRITE,
                               0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (dummy != INVALID_HANDLE_VALUE) CloseHandle(dummy);
    if (g_thread.joinable()) g_thread.join();
    g_handler = {};
    g_stop = nullptr;
}

void interruptLocalDisconnectedSession() noexcept {
    std::string current;
    {
        std::lock_guard<std::mutex> lock(g_session_mutex);
        current = g_current_session;
    }
    if (!current.empty() && rtipc_session::shouldDisconnect(current) && g_thread.joinable())
        CancelSynchronousIo(g_thread.native_handle());
}

} // namespace rtipc_transport
