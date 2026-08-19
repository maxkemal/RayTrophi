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
#endif

using json = nlohmann::json;

namespace rtipc_transport {
namespace {

constexpr const wchar_t* kPipeName = L"\\\\.\\pipe\\RayTrophiStudio";
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
