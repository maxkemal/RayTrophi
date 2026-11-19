#include "Core/PlayerCamera.h"
#include "Core/RuntimeProjectHost.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <set>
#include <tuple>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")
#endif

namespace AppRuntime {
namespace {
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
json readJson(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return json();
    }
    auto value = json::parse(stream, nullptr, false);
    return value.is_discarded() ? json() : value;
}
#ifdef _WIN32
using Microsoft::WRL::ComPtr;
std::string utf8(const wchar_t *value) {
    const int length = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (length <= 1) {
        return {};
    }
    std::string result(static_cast<size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), length, nullptr, nullptr);
    result.pop_back();
    return result;
}
std::string attribute(IMFAttributes *attributes, REFGUID key) {
    wchar_t *value = nullptr;
    UINT32 size = 0;
    if (FAILED(attributes->GetAllocatedString(key, &value, &size))) {
        return {};
    }
    const auto result = utf8(value);
    CoTaskMemFree(value);
    return result;
}
json inventory(const std::string &selected) {
    json result = {{"devices", json::array()}, {"error", ""}};
    const HRESULT initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(initialized)) {
        result["error"] = "Camera enumeration could not initialize COM";
        return result;
    }
    if (FAILED(MFStartup(MF_VERSION))) {
        CoUninitialize();
        result["error"] = "Windows Media Foundation is unavailable";
        return result;
    }
    {
        ComPtr<IMFAttributes> attributes;
        IMFActivate **devices = nullptr;
        UINT32 count = 0;
        HRESULT hr = MFCreateAttributes(&attributes, 1);
        if (SUCCEEDED(hr)) {
            hr = attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                     MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        }
        if (SUCCEEDED(hr)) {
            hr = MFEnumDeviceSources(attributes.Get(), &devices, &count);
        }
        if (FAILED(hr)) {
            result["error"] = "Windows camera inventory is unavailable";
        }
        for (UINT32 index = 0; index < count; ++index) {
            const auto id =
                attribute(devices[index], MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);
            json row = {{"id", id},
                        {"name", attribute(devices[index], MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME)},
                        {"index", index},
                        {"modes", json::array()}};
            // Only activate the explicitly selected device to inspect its native modes.
            if (!selected.empty() && selected == id) {
                ComPtr<IMFMediaSource> source;
                ComPtr<IMFSourceReader> reader;
                hr = devices[index]->ActivateObject(IID_PPV_ARGS(&source));
                if (SUCCEEDED(hr)) {
                    hr = MFCreateSourceReaderFromMediaSource(source.Get(), nullptr, &reader);
                }
                std::set<std::tuple<UINT32, UINT32, UINT32, UINT32>> unique;
                if (SUCCEEDED(hr)) {
                    for (DWORD type_index = 0; type_index < 2048; ++type_index) {
                        ComPtr<IMFMediaType> type;
                        if (FAILED(reader->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                              type_index, &type))) {
                            break;
                        }
                        UINT32 width = 0, height = 0, numerator = 0, denominator = 0;
                        if (SUCCEEDED(MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &width,
                                                         &height)) &&
                            SUCCEEDED(MFGetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, &numerator,
                                                          &denominator)) &&
                            denominator && width >= 160 && width <= 8192 && height >= 120 &&
                            height <= 8192 && static_cast<double>(numerator) / denominator >= 1.0 &&
                            static_cast<double>(numerator) / denominator <= 240.0 &&
                            unique.emplace(width, height, numerator, denominator).second) {
                            row["modes"].push_back(
                                {{"width", width},
                                 {"height", height},
                                 {"fps", static_cast<double>(numerator) / denominator}});
                        }
                    }
                } else {
                    row["error"] = "Cannot inspect modes; camera may be busy or access denied";
                }
                reader.Reset();
                if (source) {
                    source->Shutdown();
                }
                devices[index]->ShutdownObject();
            }
            result["devices"].push_back(std::move(row));
            devices[index]->Release();
        }
        CoTaskMemFree(devices);
    }
    MFShutdown();
    CoUninitialize();
    return result;
}
std::wstring quote(const std::filesystem::path &path) {
    // Windows CRT command-line quoting, including trailing backslashes.
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (const auto c : path.wstring()) {
        if (c == L'\\') {
            ++slashes;
        } else {
            result.append(slashes * (c == L'"' ? 2 : 1), L'\\');
            slashes = 0;
            if (c == L'"') {
                result += L'\\';
            }
            result += c;
        }
    }
    result.append(slashes * 2, L'\\');
    return result + L'"';
}
#endif
} // namespace

struct PlayerCamera::Impl {
    json catalog = {{"devices", json::array()}, {"error", ""}};
    json state = {{"state", "stopped"}};
    std::future<json> scanning;
    Clock::time_point last_poll{};
    Clock::time_point stop_time{};
    std::filesystem::path directory;
    SDL_Texture *texture = nullptr;
    uint32_t last_sequence = 0;
#ifdef _WIN32
    HANDLE process = nullptr;
    HANDLE job = nullptr;
    HANDLE stop_event = nullptr;
    HANDLE mapping = nullptr;
    HANDLE mutex = nullptr;
    void *memory = nullptr;
#endif
    void release() {
        if (texture) {
            SDL_DestroyTexture(texture);
            texture = nullptr;
        }
#ifdef _WIN32
        if (memory) {
            UnmapViewOfFile(memory);
        }
        for (auto handle : {process, job, stop_event, mapping, mutex}) {
            if (handle) {
                CloseHandle(handle);
            }
        }
        process = job = stop_event = mapping = mutex = nullptr;
        memory = nullptr;
#endif
        last_sequence = 0;
        if (!directory.empty()) {
            std::error_code ignored;
            // Remove only the known metadata files created by this session.
            for (const auto *name : {"request.json", "status.json", "status.json.tmp"}) {
                std::filesystem::remove(directory / name, ignored);
            }
            std::filesystem::remove(directory, ignored);
            directory.clear();
        }
    }
};
PlayerCamera::PlayerCamera() : impl_(std::make_unique<Impl>()) {}
PlayerCamera::~PlayerCamera() {
#ifdef _WIN32
    if (impl_->process) {
        SetEvent(impl_->stop_event);
        // Closing the owned job also handles a driver blocked inside read().
        if (impl_->job) {
            TerminateJobObject(impl_->job, 1);
        }
    }
#endif
    impl_->release();
}
void PlayerCamera::tick() {
    auto &s = *impl_;
    if (s.scanning.valid() &&
        s.scanning.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        s.catalog = s.scanning.get();
    }
#ifdef _WIN32
    if (!s.process) {
        return;
    }
    if (Clock::now() - s.last_poll > std::chrono::milliseconds(200)) {
        const auto status = readJson(s.directory / "status.json");
        if (status.is_object()) {
            s.state = status;
        }
        s.last_poll = Clock::now();
    }
    if (s.stop_time.time_since_epoch().count() &&
        Clock::now() - s.stop_time > std::chrono::seconds(4)) {
        TerminateJobObject(s.job, 1);
    }
    if (WaitForSingleObject(s.process, 0) == WAIT_OBJECT_0) {
        DWORD exit_code = 0;
        GetExitCodeProcess(s.process, &exit_code);
        const auto status = readJson(s.directory / "status.json");
        if (status.is_object()) {
            s.state = status;
        }
        if (exit_code && s.state.value("state", "") != "error") {
            s.state = {{"state", "error"}, {"error", "Camera worker stopped unexpectedly"}};
        } else if (s.state.value("state", "") != "error") {
            s.state["state"] = "stopped";
        }
        s.release();
    }
#endif
}
json PlayerCamera::status() {
    auto result = impl_->state;
    result["catalog"] = impl_->catalog;
    result["scanning"] = impl_->scanning.valid();
#ifdef _WIN32
    result["active"] = impl_->process != nullptr;
#else
    result["active"] = false;
#endif
    return result;
}
rtapi::Result PlayerCamera::command(const std::string &operation, const json &config) {
    auto &s = *impl_;
    // Resource reclamation belongs to the host's between-frame tick, never
    // to a button callback after ImGui has queued the preview texture.
#ifdef _WIN32
    if (operation == "refresh") {
        if (s.process || s.scanning.valid()) {
            return rtapi::Result::fail("Stop the camera or wait for device scanning first");
        }
        const auto id = config.value("device_id", std::string());
        s.scanning = std::async(std::launch::async, [id] { return inventory(id); });
        return rtapi::Result::success();
    }
    if (operation == "stop") {
        if (s.process) {
            SetEvent(s.stop_event);
            s.stop_time = Clock::now();
            s.state["state"] = "stopping";
        }
        return rtapi::Result::success();
    }
    if (operation != "start") {
        return rtapi::Result::fail("Unknown camera operation");
    }
    if (s.process || s.scanning.valid()) {
        return rtapi::Result::fail("Stop the camera or wait for device scanning first");
    }
    const auto python = projectPathFromUtf8(config.at("python").get<std::string>());
    const auto worker = projectPathFromUtf8(config.at("worker").get<std::string>());
    if (!std::filesystem::is_regular_file(python) || !std::filesystem::is_regular_file(worker)) {
        return rtapi::Result::fail("Packaged camera runtime is missing");
    }
    const auto token =
        std::to_string(GetCurrentProcessId()) + "_" + std::to_string(GetTickCount64());
    const std::string prefix = "Local\\RayTrophiOperator_" + token;
    s.directory = std::filesystem::temp_directory_path() / ("RayTrophiOperator_" + token);
    std::filesystem::create_directory(s.directory);
    auto request = config;
    request["mapping"] = prefix + "_preview";
    request["mutex"] = prefix + "_mutex";
    request["stop_event"] = prefix + "_stop";
    request["status_path"] = projectPathToUtf8(s.directory / "status.json");
    s.stop_event =
        CreateEventA(nullptr, TRUE, FALSE, request["stop_event"].get<std::string>().c_str());
    s.mutex = CreateMutexA(nullptr, FALSE, request["mutex"].get<std::string>().c_str());
    constexpr DWORD bytes = 16 + 640 * 360 * 4;
    s.mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, bytes,
                                   request["mapping"].get<std::string>().c_str());
    s.memory = s.mapping ? MapViewOfFile(s.mapping, FILE_MAP_ALL_ACCESS, 0, 0, bytes) : nullptr;
    if (!s.stop_event || !s.mutex || !s.memory) {
        s.release();
        return rtapi::Result::fail("Cannot allocate memory-only camera preview");
    }
    std::memset(s.memory, 0, bytes);
    const auto request_path = s.directory / "request.json";
    {
        std::ofstream stream(request_path, std::ios::binary);
        stream << request.dump(2);
        if (!stream) {
            s.release();
            return rtapi::Result::fail("Cannot write camera session metadata");
        }
    }
    s.job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limit{};
    limit.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!s.job ||
        !SetInformationJobObject(s.job, JobObjectExtendedLimitInformation, &limit, sizeof(limit))) {
        s.release();
        return rtapi::Result::fail("Cannot create camera process lifecycle");
    }
    std::wstring command =
        quote(python) + L" -s -u " + quote(worker) + L" --request " + quote(request_path);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(python.c_str(), command.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &startup,
                        &process)) {
        s.release();
        return rtapi::Result::fail("Camera worker could not start");
    }
    s.process = process.hProcess;
    if (!AssignProcessToJobObject(s.job, s.process)) {
        TerminateProcess(s.process, 1);
        CloseHandle(process.hThread);
        s.release();
        return rtapi::Result::fail("Camera worker lifecycle could not be attached");
    }
    if (ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {
        TerminateJobObject(s.job, 1);
        CloseHandle(process.hThread);
        s.release();
        return rtapi::Result::fail("Camera worker could not resume");
    }
    CloseHandle(process.hThread);
    s.stop_time = {};
    s.state = {{"state", "starting"}};
    return rtapi::Result::success();
#else
    return rtapi::Result::fail("Camera operator controls require Windows Media Foundation");
#endif
}
SDL_Texture *PlayerCamera::preview(SDL_Renderer *renderer) {
#ifdef _WIN32
    auto &s = *impl_;
    if (!s.memory || !s.mutex) {
        return s.texture;
    }
    const auto acquired = WaitForSingleObject(s.mutex, 0);
    if (acquired != WAIT_OBJECT_0 && acquired != WAIT_ABANDONED) {
        return s.texture;
    }
    const auto *header = static_cast<const uint32_t *>(s.memory);
    if (header[0] == 640 && header[1] == 360 && header[2] != s.last_sequence) {
        if (!s.texture) {
            s.texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
                                          SDL_TEXTUREACCESS_STREAMING, 640, 360);
        }
        if (s.texture && SDL_UpdateTexture(s.texture, nullptr, header + 4, 640 * 4) == 0) {
            s.last_sequence = header[2];
        }
    }
    ReleaseMutex(s.mutex);
    return s.texture;
#else
    return nullptr;
#endif
}
} // namespace AppRuntime
