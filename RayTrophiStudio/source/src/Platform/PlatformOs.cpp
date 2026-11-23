#include "Platform/Platform.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

#include <algorithm>
#else
#include <cerrno>
#include <dlfcn.h>
#include <sys/random.h>
#include <fstream>
#include <limits>
#include <unistd.h>
#ifdef __ANDROID__
#include <SDL_system.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <iterator>
#include <set>
#include <sstream>
#include <vector>
#elif defined(__linux__)
#include <system_error>
#endif
#endif

namespace Platform {

std::filesystem::path executablePath() {
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD capacity = static_cast<DWORD>(buffer.size());
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), capacity);
        if (length == 0) {
            return {};
        }
        if (length < capacity) {
            buffer.resize(length);
            return std::filesystem::path(buffer);
        }
        if (capacity >= 32768) {
            return {};
        }
        buffer.resize((std::min)(buffer.size() * 2, std::size_t{32768}));
    }
#else
    return {};
#endif
}

std::filesystem::path executableDir() {
#if defined(_WIN32)
    return executablePath().parent_path();
#elif defined(__ANDROID__)
    return appDataRoot();
#elif defined(__linux__)
    std::error_code error;
    return std::filesystem::read_symlink("/proc/self/exe", error).parent_path();
#else
    return {};
#endif
}

std::filesystem::path appDataRoot() {
#ifdef __ANDROID__
    const char* path = SDL_AndroidGetInternalStoragePath();
    return path ? std::filesystem::path(path) : std::filesystem::path{};
#else
    return executableDir();
#endif
}

std::uint64_t processResidentBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, counters.cb)) {
        return 0;
    }
    return counters.WorkingSetSize;
#else
    std::ifstream statm("/proc/self/statm");
    std::uint64_t totalPages = 0;
    std::uint64_t residentPages = 0;
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0 || !(statm >> totalPages >> residentPages)) {
        return 0;
    }
    const auto bytesPerPage = static_cast<std::uint64_t>(pageSize);
    if (residentPages > std::numeric_limits<std::uint64_t>::max() / bytesPerPage) {
        return 0;
    }
    return residentPages * bytesPerPage;
#endif
}

std::uint64_t systemTotalMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? static_cast<std::uint64_t>(status.ullTotalPhys) : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    return (pages > 0 && page_size > 0)
               ? static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size) : 0;
#endif
}

std::uint64_t systemAvailableMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? static_cast<std::uint64_t>(status.ullAvailPhys) : 0;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    return (pages > 0 && page_size > 0)
               ? static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size) : 0;
#endif
}

std::uint32_t processId() {
#ifdef _WIN32
    return GetCurrentProcessId();
#else
    return static_cast<std::uint32_t>(getpid());
#endif
}

void* loadLibrary(const char* name) {
    if (!name || !*name) {
        return nullptr;
    }
#ifdef _WIN32
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name, -1, nullptr, 0);
    if (size <= 0) {
        return nullptr;
    }
    std::wstring wideName(static_cast<std::size_t>(size), L'\0');
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name, -1, wideName.data(), size)) {
        return nullptr;
    }
    return LoadLibraryW(wideName.c_str());
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* findSymbol(void* library, const char* symbol) {
    if (!library || !symbol || !*symbol) {
        return nullptr;
    }
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), symbol));
#else
    return dlsym(library, symbol);
#endif
}

void unloadLibrary(void* library) {
    if (!library) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
}

bool secureRandomBytes(void* out, std::size_t size) {
    if (size == 0) {
        return true;
    }
    if (!out) {
        return false;
    }
#ifdef _WIN32
    auto* bytes = static_cast<unsigned char*>(out);
    while (size > 0) {
        const ULONG chunk = static_cast<ULONG>((std::min)(size, std::size_t{0x7fffffff}));
        if (!BCRYPT_SUCCESS(BCryptGenRandom(nullptr, bytes, chunk,
                                            BCRYPT_USE_SYSTEM_PREFERRED_RNG))) {
            return false;
        }
        bytes += chunk;
        size -= chunk;
    }
    return true;
#else
    auto* bytes = static_cast<unsigned char*>(out);
    while (size > 0) {
        const ssize_t got = getrandom(bytes, size, 0);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        bytes += got;
        size -= static_cast<std::size_t>(got);
    }
    return true;
#endif
}

#ifdef __ANDROID__
namespace {

// APK assets are read through AAssetManager, never SDL_RWFromFile: for a relative path SDL2
// first fopen()s <internal storage>/<path>, which is exactly where extracted copies live. After
// an app update it would hand back the stale extracted file and copy it onto itself.
AAssetManager* apkAssetManager() {
    static AAssetManager* manager = nullptr;
    static jobject manager_ref = nullptr;   // keeps the Java AssetManager alive for the process
    if (manager) return manager;
    auto* env = static_cast<JNIEnv*>(SDL_AndroidGetJNIEnv());
    auto activity = static_cast<jobject>(SDL_AndroidGetActivity());
    if (!env || !activity) return nullptr;
    jclass activity_class = env->GetObjectClass(activity);
    jmethodID get_assets =
        env->GetMethodID(activity_class, "getAssets", "()Landroid/content/res/AssetManager;");
    jobject assets = get_assets ? env->CallObjectMethod(activity, get_assets) : nullptr;
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (assets) {
        manager_ref = env->NewGlobalRef(assets);
        manager = AAssetManager_fromJava(env, manager_ref);
        env->DeleteLocalRef(assets);
    }
    env->DeleteLocalRef(activity_class);
    env->DeleteLocalRef(activity);
    return manager;
}

AAsset* openApkAsset(const std::string& path) {
    AAssetManager* manager = apkAssetManager();
    return manager ? AAssetManager_open(manager, path.c_str(), AASSET_MODE_STREAMING) : nullptr;
}

bool readApkAsset(const std::string& path, std::string& out) {
    AAsset* asset = openApkAsset(path);
    if (!asset) return false;
    out.resize(static_cast<std::size_t>(AAsset_getLength64(asset)));
    std::size_t got = 0;
    while (got < out.size()) {
        const int n = AAsset_read(asset, out.data() + got, out.size() - got);
        if (n <= 0) break;
        got += static_cast<std::size_t>(n);
    }
    AAsset_close(asset);
    return got == out.size();
}

bool copyApkAsset(const std::string& relative, const std::filesystem::path& target) {
    AAsset* asset = openApkAsset(relative);
    if (!asset) return false;
    std::error_code error;
    std::filesystem::create_directories(target.parent_path(), error);
    const std::filesystem::path partial = target.string() + ".partial";
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    bool ok = static_cast<bool>(out);
    std::vector<char> buffer(1 << 20);
    while (ok) {
        const int got = AAsset_read(asset, buffer.data(), buffer.size());
        if (got < 0) { ok = false; break; }
        if (got == 0) break;
        out.write(buffer.data(), static_cast<std::streamsize>(got));
        ok = static_cast<bool>(out);
    }
    AAsset_close(asset);
    out.close();
    if (!ok || !out) { std::filesystem::remove(partial, error); return false; }
    std::filesystem::rename(partial, target, error);   // replaces an older copy atomically
    return !error;
}

// Manifest: "stamp <hex>\n" then relative paths. Rejects absolute/parent paths.
bool parseManifest(const std::string& text, std::string& stamp, std::vector<std::string>& paths) {
    std::istringstream lines(text);
    std::string line;
    if (!std::getline(lines, line) || line.rfind("stamp ", 0) != 0) return false;
    stamp = line.substr(6);
    while (!stamp.empty() && (stamp.back() == '\r' || stamp.back() == ' ')) stamp.pop_back();
    while (std::getline(lines, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) continue;
        if (line.front() == '/' || line.find("..") != std::string::npos) return false;
        paths.push_back(line);
    }
    return !stamp.empty();
}

} // namespace
#endif

bool extractBundledAssets(std::string& report) {
    report.clear();
#ifndef __ANDROID__
    return true;   // packaged files already sit next to the executable
#else
    std::string manifest_text;
    if (!readApkAsset("rt_asset_manifest.txt", manifest_text)) {
        report = "no bundled asset manifest in the APK (build without staged assets)";
        return false;
    }
    std::string stamp;
    std::vector<std::string> paths;
    if (!parseManifest(manifest_text, stamp, paths)) {
        report = "bundled asset manifest is malformed";
        return false;
    }
    const std::filesystem::path root = appDataRoot();
    if (root.empty()) { report = "no app data directory"; return false; }
    const std::filesystem::path stamp_path = root / ".rt_asset_stamp";
    const std::filesystem::path old_manifest_path = root / ".rt_asset_manifest";
    {
        std::ifstream in(stamp_path, std::ios::binary);
        std::string stored;
        if (in && std::getline(in, stored) && stored == stamp) {
            report = "bundled assets up to date (" + std::to_string(paths.size()) + " files)";
            return true;
        }
    }
    // Invalidate first: a crash mid-copy must not leave a valid stamp over mixed files.
    std::error_code error;
    std::filesystem::remove(stamp_path, error);

    std::size_t removed = 0;
    {
        std::ifstream in(old_manifest_path, std::ios::binary);
        if (in) {
            const std::string old_text((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
            std::string old_stamp;
            std::vector<std::string> old_paths;
            if (parseManifest(old_text, old_stamp, old_paths)) {
                const std::set<std::string> keep(paths.begin(), paths.end());
                for (const auto& old_path : old_paths) {
                    if (keep.count(old_path)) continue;
                    if (std::filesystem::remove(root / old_path, error)) ++removed;
                }
            }
        }
    }
    std::size_t copied = 0;
    for (const auto& relative : paths) {
        if (!copyApkAsset(relative, root / relative)) {
            report = "failed to extract bundled asset '" + relative + "' (" +
                     std::to_string(copied) + "/" + std::to_string(paths.size()) + " done)";
            return false;
        }
        ++copied;
    }
    {
        std::ofstream manifest_out(old_manifest_path, std::ios::binary | std::ios::trunc);
        manifest_out << manifest_text;
        std::ofstream stamp_out(stamp_path, std::ios::binary | std::ios::trunc);
        stamp_out << stamp << "\n";
        if (!manifest_out || !stamp_out) {
            report = "bundled assets copied but the stamp could not be written";
            return false;
        }
    }
    report = "extracted " + std::to_string(copied) + " bundled files" +
             (removed ? ", removed " + std::to_string(removed) + " stale" : "") +
             " (stamp " + stamp.substr(0, 12) + ")";
    return true;
#endif
}

bool supportsChildProcesses() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

} // namespace Platform
